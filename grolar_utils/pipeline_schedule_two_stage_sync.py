# -*- coding: utf-8 -*-
"""
Pipeline for case when we are interleaving stages across two GPU groups.
e.g. Stage 1 (Group 1) -> Stage 2 (Group 2) -> Stage 3 (Group 1) -> ...
The usual pipeline schedule class hangs because Group 1 is sending
activations to Group 2 between Stage 1 -> Stage 2, concurrently when
Group 2 is sending activations to Group 1 between Stage 2 -> Stage 3.
This is known to cause issues in NCCL: https://github.com/pytorch/pytorch/issues/67158

pipeline_schedule_no_overlap resolves this issue by sending all activations
after all microbatches complete. However, this does not allow for any computation
overlap across GPU groups. Here we do a hybrid approach where every even stage
sends their activations as they are computed but every odd stage waits until
all microbatches complete to send activations. This still allows for good computation
overlap while avoiding NCCL hanging. Similar logic supports sending gradients
during the backwards pass.

TODO: offload computed activations/gradients that will be deferred to save memory
"""

# -*- coding: utf-8 -*-
from typing import List, Optional
import torch
from torch.distributed.fsdp._runtime_utils import _post_backward_final_callback

from grolar_utils.pipeline import PipelineStage
from grolar_utils.pipeline_config import PipelineConfig
from grolar_utils.pipeline_schedule_v2 import StageState, PipelineScheduleGpipe
from utils.comm import get_global_rank


class PipelineScheduleGpipeTwoStageSync(PipelineScheduleGpipe):
    def __init__(
        self, pipeline_stages: List[PipelineStage], pipeline_config: PipelineConfig
    ):
        super().__init__(pipeline_stages, pipeline_config)

    def _forward_stage(self, state: StageState, global_rank: int) -> None:
        stage_id = state.stage.stage_id
        comm_plan = state.stage.communicator.comm_plan
        # Store outputs for all microbatches
        if not state.stage.is_last_stage() and not state.stage.even_stage():
            # allocate cpu buffers for storing forward outputs
            buffer_shape = state.gpu_activation_buffer.shape
            cpu_output_buffers = [
                torch.empty(buffer_shape, pin_memory=True)
                for _ in range(state.num_microbatches)
            ]

        if not state.stage.is_first_stage():
            # allocate cpu buffers for storing inputs
            buffer_shape = state.gpu_activation_buffer.shape
            for mb_idx in range(state.num_microbatches):
                state.input_chunks[mb_idx] = torch.empty(buffer_shape, pin_memory=True)

            # for even stages we receive all activations together
            if state.stage.even_stage():
                for mb_idx in range(state.num_microbatches):
                    recv_work = state.stage.communicator.recv_forward(
                        state.gpu_activation_buffer.shape, mb_idx=mb_idx
                    )
                    recv_work.wait()
                    self.pipeline_logger.log_forward_recv(
                        stage_id,
                        mb_idx,
                        comm_plan.fwd_recv_from_ranks[mb_idx],
                    )
                    current_input = recv_work.result()
                    assert len(current_input) == 1 and isinstance(
                        current_input[0], torch.Tensor
                    )
                    self.store_to_cpu_buffer(
                        current_input[0], state.input_chunks[mb_idx]
                    )

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
                    # for odd stages we receive activations one microbatch at a time
                    if not state.stage.is_first_stage():
                        if not state.stage.even_stage():
                            recv_work = state.stage.communicator.recv_forward(
                                state.gpu_activation_buffer.shape, mb_idx=mb_idx
                            )
                            recv_work.wait()
                            self.pipeline_logger.log_forward_recv(
                                stage_id,
                                mb_idx,
                                comm_plan.fwd_recv_from_ranks[mb_idx],
                            )
                            current_input = recv_work.result()
                            assert len(current_input) == 1 and isinstance(
                                current_input[0], torch.Tensor
                            )
                            layer_input = current_input[0]
                        else:
                            layer_input = self.load_from_cpu_buffer(
                                state.gpu_activation_buffer, state.input_chunks[mb_idx]
                            )
                            layer_input.requires_grad = True
                    else:
                        layer_input = state.input_chunks[mb_idx]
                else:
                    buffer_idx = (li - 1) * state.num_microbatches + mb_idx
                    layer_input = self.load_from_cpu_buffer(
                        state.gpu_activation_buffer, state.cpu_buffers[buffer_idx]
                    )

                output = self._forward_layer(
                    layer,
                    layer_input,
                    is_first_mb=(mb_idx == 0),
                    is_last_mb=(mb_idx == state.num_microbatches - 1),
                    skip_reshard=is_last_layer_of_model,
                )

                if not is_last_layer:
                    buffer_idx = li * state.num_microbatches + mb_idx
                    self.store_to_cpu_buffer(output, state.cpu_buffers[buffer_idx])
                else:
                    if not state.stage.is_last_stage():
                        if not state.stage.even_stage():
                            # Store the final output for this microbatch
                            self.store_to_cpu_buffer(output, cpu_output_buffers[mb_idx])
                        else:
                            # for even stages we send activations one microbatch at a time
                            send_work = state.stage.communicator.send_forward(
                                output, mb_idx=mb_idx
                            )
                            self.pipeline_logger.log_forward_send(
                                stage_id,
                                mb_idx,
                                comm_plan.fwd_send_to_ranks[mb_idx],
                            )

                self.pipeline_logger.log_forward_compute(
                    stage_id,
                    mb_idx,
                    li + state.stage.stage_config.layer_partition[0],
                )

        # Wait for all computations to complete
        torch.cuda.synchronize()

        # for odd stages we send all activations together
        if not state.stage.is_last_stage() and not state.stage.even_stage():
            state.stage.communicator.wait_for_stream()
            for mb_idx in range(state.num_microbatches):
                stage_output = self.load_from_cpu_buffer(
                    state.gpu_activation_buffer, cpu_output_buffers[mb_idx]
                )
                send_work = state.stage.communicator.send_forward(
                    stage_output, mb_idx=mb_idx
                )
                if send_work is not None:
                    send_work.wait()
                self.pipeline_logger.log_forward_send(
                    stage_id,
                    mb_idx,
                    comm_plan.fwd_send_to_ranks[mb_idx],
                )

    def _backward_stage(self, state: StageState, global_rank: int) -> None:
        stage_id = state.stage.stage_id
        comm_plan = state.stage.communicator.comm_plan
        # for odd stages we receive all grads together
        if not state.stage.is_last_stage() and not state.stage.even_stage():
            for mb_idx in range(state.num_microbatches - 1, -1, -1):
                recv_work = state.stage.communicator.recv_backward(
                    state.gpu_grad_buffer.shape, mb_idx=mb_idx
                )
                assert recv_work is not None
                recv_work.wait()
                grad = recv_work.result()[0]
                self.store_to_cpu_buffer(grad, state.cpu_grad_buffers[mb_idx])
                self.pipeline_logger.log_backward_recv(
                    stage_id,
                    mb_idx,
                    comm_plan.bwd_recv_from_ranks[mb_idx],
                )

        for li in range(len(state.layers) - 1, -1, -1):
            layer = state.layers[li]
            is_first_layer = li == 0
            is_last_layer = li == len(state.layers) - 1
            is_last_layer_of_model = is_last_layer and state.stage.is_last_stage()

            self.pipeline_logger.debug(
                stage_id, "x", f"+ Starting layer backwards {li}"
            )

            # Process all microbatches for this layer
            for mb_idx in range(state.num_microbatches - 1, -1, -1):
                self.pipeline_logger.debug(
                    stage_id, mb_idx, " -- Processing microbatch"
                )
                grad = None
                if is_last_layer:
                    if not state.stage.is_last_stage():
                        if state.stage.even_stage():
                            # for even stages we receive grads one microbatch at a time
                            recv_work = state.stage.communicator.recv_backward(
                                state.gpu_grad_buffer.shape, mb_idx=mb_idx
                            )
                            assert recv_work is not None
                            recv_work.wait()
                            grad = recv_work.result()[0]
                            self.pipeline_logger.log_backward_recv(
                                stage_id,
                                mb_idx,
                                comm_plan.bwd_recv_from_ranks[mb_idx],
                            )
                        else:
                            grad = self.load_from_cpu_buffer(
                                state.gpu_grad_buffer, state.cpu_grad_buffers[mb_idx]
                            )
                else:
                    grad = self.load_from_cpu_buffer(
                        state.gpu_grad_buffer, state.cpu_grad_buffers[mb_idx]
                    )

                if is_first_layer:
                    if state.stage.is_first_stage():
                        layer_input = state.input_chunks[mb_idx]
                    else:
                        layer_input = self.load_from_cpu_buffer(
                            state.gpu_activation_buffer, state.input_chunks[mb_idx]
                        )
                        layer_input.requires_grad = True
                else:
                    buffer_idx = (li - 1) * state.num_microbatches + mb_idx
                    layer_input = self.load_from_cpu_buffer(
                        state.gpu_activation_buffer, state.cpu_buffers[buffer_idx]
                    )
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

                self.pipeline_logger.log_backward_compute(
                    stage_id,
                    mb_idx,
                    li + state.stage.stage_config.layer_partition[0],
                )

                if not is_first_layer:
                    self.store_to_cpu_buffer(
                        layer_input.grad.detach(), state.cpu_grad_buffers[mb_idx]
                    )
                elif not state.stage.is_first_stage():
                    if not state.stage.even_stage():
                        # for odd stages we sends grads one microbatch at a time
                        send_work = state.stage.communicator.send_backward(
                            layer_input.grad, mb_idx=mb_idx
                        )
                        self.pipeline_logger.log_backward_send(
                            stage_id,
                            mb_idx,
                            comm_plan.bwd_send_to_ranks[mb_idx],
                        )
                    else:
                        self.store_to_cpu_buffer(
                            layer_input.grad.detach(), state.cpu_grad_buffers[mb_idx]
                        )

        # Wait for all computations to complete
        torch.cuda.synchronize()

        # for even stages we send all grads together
        if not state.stage.is_first_stage() and state.stage.even_stage():
            state.stage.communicator.wait_for_stream()
            for mb_idx in range(state.num_microbatches - 1, -1, -1):
                self.load_from_cpu_buffer(
                    state.gpu_grad_buffer, state.cpu_grad_buffers[mb_idx]
                )
                send_work = state.stage.communicator.send_backward(
                    state.gpu_grad_buffer, mb_idx=mb_idx
                )
                if send_work is not None:
                    send_work.wait()
                self.pipeline_logger.log_backward_send(
                    stage_id,
                    mb_idx,
                    comm_plan.bwd_send_to_ranks[mb_idx],
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
) -> PipelineScheduleGpipeTwoStageSync:
    return PipelineScheduleGpipeTwoStageSync(pipeline_stages, pipeline_config)
