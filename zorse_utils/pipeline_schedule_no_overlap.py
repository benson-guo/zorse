# -*- coding: utf-8 -*-
"""
Variant of pipeline that waits until all microbatches are complete before
sending gradients/activations. This avoids issues that arise when sending
data between GPUs concurrently, but is inefficient since it prevents compute
overlap from happening.
"""

from typing import List, Optional
import torch
from torch.distributed.fsdp._runtime_utils import _post_backward_final_callback

from zorse_utils.pipeline import PipelineStage
from zorse_utils.pipeline_config import PipelineConfig
from utils.comm import get_global_rank
from zorse_utils.pipeline_schedule_v2 import StageState, PipelineScheduleGpipe


class PipelineScheduleGpipeSynchronized(PipelineScheduleGpipe):
    def __init__(
        self, pipeline_stages: List[PipelineStage], pipeline_config: PipelineConfig
    ):
        super().__init__(pipeline_stages, pipeline_config)

    def _forward_stage(self, state: StageState, global_rank: int) -> None:
        stage_id = state.stage.stage_id
        # Store outputs for all microbatches
        forward_outputs: List[Optional[torch.Tensor]] = [None] * state.num_microbatches

        for li, layer in enumerate(state.layers):
            is_first_layer = li == 0
            is_last_layer = li == len(state.layers) - 1
            is_last_layer_of_model = is_last_layer and state.stage.is_last_stage()

            self.pipeline_logger.debug(stage_id, "x", f"+ Starting layer forwards {li}")

            # Process all microbatches for this layer
            for mb_idx in range(state.num_microbatches):
                self.pipeline_logger.debug(
                    stage_id, mb_idx, " -- Processing microbatch"
                )
                if is_first_layer:
                    if not state.stage.is_first_stage():
                        recv_work = state.stage.communicator.recv_forward(
                            state.gpu_activation_buffer.shape, mb_idx=mb_idx
                        )
                        recv_work.wait()
                        self.pipeline_logger.log_forward_recv(
                            state.stage.stage_id,
                            mb_idx,
                            state.stage.communicator.comm_plan.fwd_recv_from_ranks[
                                mb_idx
                            ],
                        )
                        current_input = recv_work.result()
                        assert len(current_input) == 1
                        assert isinstance(current_input[0], torch.Tensor)
                        state.input_chunks[mb_idx] = current_input[0]
                    layer_input = state.input_chunks[mb_idx]
                else:
                    layer_input = state.cpu_buffers[
                        (li - 1) * state.num_microbatches + mb_idx
                    ]

                output = self._forward_layer(
                    layer,
                    layer_input,
                    is_first_mb=(mb_idx == 0),
                    is_last_mb=(mb_idx == state.num_microbatches - 1),
                    skip_reshard=is_last_layer_of_model,
                )

                if not is_last_layer:
                    with torch.cuda.stream(self.mem_copy_stream):
                        buffer_idx = li * state.num_microbatches + mb_idx
                        state.cpu_buffers[buffer_idx].copy_(
                            output.data, non_blocking=True
                        )
                else:
                    # Store the final output for this microbatch
                    forward_outputs[mb_idx] = output

                self.pipeline_logger.log_forward_compute(
                    state.stage.stage_id,
                    mb_idx,
                    li + state.stage.stage_config.layer_partition[0],
                )

        # Wait for all computations to complete
        torch.cuda.synchronize()

        # After processing all microbatches, send them all at once
        if not state.stage.is_last_stage():
            state.stage.communicator.wait_for_stream()
            for mb_idx in range(state.num_microbatches):
                send_work = state.stage.communicator.send_forward(
                    forward_outputs[mb_idx], mb_idx=mb_idx
                )
                if send_work is not None:
                    send_work.wait()
                self.pipeline_logger.log_forward_send(
                    state.stage.stage_id,
                    mb_idx,
                    state.stage.communicator.comm_plan.fwd_send_to_ranks[mb_idx],
                )

    def _backward_stage(self, state: StageState, global_rank: int) -> None:
        stage_id = state.stage.stage_id
        for li in range(len(state.layers) - 1, -1, -1):
            layer = state.layers[li]
            is_first_layer = li == 0
            is_last_layer = li == len(state.layers) - 1
            is_last_layer_of_model = is_last_layer and state.stage.is_last_stage()

            self.pipeline_logger.debug(
                stage_id, "x", f"+ Starting layer backwards {li}"
            )

            # Process all microbatches for this layer
            stored_grads = []
            for mb_idx in range(state.num_microbatches - 1, -1, -1):
                self.pipeline_logger.debug(
                    stage_id, mb_idx, " -- Processing microbatch"
                )
                grad = None
                if is_last_layer:
                    if not state.stage.is_last_stage():
                        recv_work = state.stage.communicator.recv_backward(
                            state.gpu_grad_buffer.shape, mb_idx=mb_idx
                        )
                        assert recv_work is not None
                        recv_work.wait()
                        grad = recv_work.result()[0]
                        stored_grads.append((mb_idx, grad))
                        self.pipeline_logger.log_backward_recv(
                            stage_id,
                            mb_idx,
                            state.stage.communicator.comm_plan.bwd_recv_from_ranks[
                                mb_idx
                            ],
                        )
                else:
                    with torch.cuda.stream(self.mem_copy_stream), torch.no_grad():
                        state.gpu_grad_buffer.copy_(
                            state.cpu_grad_buffers[mb_idx], non_blocking=True
                        )
                        grad = state.gpu_grad_buffer

                if is_first_layer:
                    layer_input = state.input_chunks[mb_idx]
                else:
                    with torch.cuda.stream(self.mem_copy_stream), torch.no_grad():
                        buffer_idx = (li - 1) * state.num_microbatches + mb_idx
                        state.gpu_activation_buffer.copy_(
                            state.cpu_buffers[buffer_idx], non_blocking=True
                        )
                        layer_input = state.gpu_activation_buffer
                        layer_input.requires_grad = True

                self._backward_layer(
                    layer,
                    layer_input,
                    grad,
                    is_first_mb=(mb_idx == state.num_microbatches - 1),
                    is_last_mb=(mb_idx == 0),
                    skip_reshard=is_last_layer_of_model,
                    is_last_layer_of_model=is_last_layer_of_model,
                    target=state.target_chunks[mb_idx],
                    sync_after_recompute=self.pipeline_config.sync_after_recompute,
                )

                if not is_first_layer:
                    with torch.cuda.stream(self.mem_copy_stream), torch.no_grad():
                        state.cpu_grad_buffers[mb_idx].copy_(
                            layer_input.grad.detach().data, non_blocking=True
                        )

                self.pipeline_logger.log_backward_compute(
                    state.stage.stage_id,
                    mb_idx,
                    li + state.stage.stage_config.layer_partition[0],
                )

            # Wait for all computations to complete
            torch.cuda.synchronize()

            # Send all gradients at once if this is the first layer
            if is_first_layer and not state.stage.is_first_stage():
                state.stage.communicator.wait_for_stream()
                for mb_idx in range(state.num_microbatches - 1, -1, -1):
                    send_work = state.stage.communicator.send_backward(
                        state.input_chunks[mb_idx].grad, mb_idx=mb_idx
                    )
                    if send_work is not None:
                        send_work.wait()
                    self.pipeline_logger.log_backward_send(
                        state.stage.stage_id,
                        mb_idx,
                        state.stage.communicator.comm_plan.bwd_send_to_ranks[mb_idx],
                    )

    def step(
        self,
        step_idx: int,
        input_chunks: Optional[List[torch.Tensor]] = None,
        target_chunks: Optional[List[torch.Tensor]] = None,
    ) -> Optional[torch.Tensor]:
        self.pipeline_logger.set_step(step_idx)
        global_rank = get_global_rank()

        if step_idx > 0:
            for state in self.stage_states:
                for layer in state.layers:
                    layer._handle._needs_pre_forward_unshard = True

        for stage_idx, state in enumerate(self.stage_states):
            if input_chunks is not None and stage_idx == 0:
                state.input_chunks = input_chunks
            if target_chunks is not None:
                state.target_chunks = target_chunks
            self._forward_stage(state, global_rank=global_rank)

        for state in reversed(self.stage_states):
            self._backward_stage(state, global_rank=global_rank)

        for state in reversed(self.stage_states):
            # Post-backward callback
            for layer in state.layers:
                _post_backward_final_callback(layer, None)


def build_pipeline_schedule(
    pipeline_stages: List[PipelineStage], pipeline_config: PipelineConfig
) -> PipelineScheduleGpipeSynchronized:
    return PipelineScheduleGpipeSynchronized(pipeline_stages, pipeline_config)
