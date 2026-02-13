# -*- coding: utf-8 -*-
from typing import List, Optional
import torch
import torch.distributed as dist
from torch.distributed.fsdp._runtime_utils import _post_backward_final_callback
from dataclasses import dataclass
from torch import nn

from grolar_utils.pipeline import PipelineStage
from grolar_utils.pipeline_config import PipelineConfig
from grolar_utils.pipeline_logger import get_pipeline_logger
from models.hub import get_all_layers
from utils.comm import get_global_rank, get_local_rank


@dataclass
class StageState:
    stage: PipelineStage
    cpu_buffers: List[torch.Tensor]
    cpu_grad_buffers: List[torch.Tensor]
    gpu_activation_buffer: torch.Tensor
    gpu_grad_buffer: torch.Tensor
    input_chunks: List[Optional[torch.Tensor]]
    target_chunks: List[Optional[torch.Tensor]]
    layers: List[nn.Module]
    num_microbatches: int
    prev_stage: Optional[PipelineStage] = None
    next_stage: Optional[PipelineStage] = None


class PipelineScheduleGpipe:
    def __init__(
        self,
        pipeline_stages: List[PipelineStage],
        pipeline_config: PipelineConfig,
        gloo_p2p: bool = False,
    ):
        self.stages = pipeline_stages
        self.pipeline_config = pipeline_config
        # mem copy stream for copying results to cpu buffer
        self.mem_copy_stream = torch.cuda.Stream()
        # use default stream for compute
        self.compute_stream = torch.cuda.current_stream()
        self.stage_states = self._initialize_stage_states()
        self.pipeline_logger = get_pipeline_logger()
        self.gloo_p2p = gloo_p2p

    def _initialize_stage_states(self) -> List[StageState]:
        states = []
        device = torch.device("cuda", get_local_rank())

        for stage in self.stages:
            buffer_shape = torch.Size(
                [
                    self.pipeline_config.microbatch_size,
                    self.pipeline_config.sequence_length,
                    self.stages[0].model._hidden_size,
                ]
            )

            layers = get_all_layers(stage.model)
            num_microbatches = stage.num_microbatches_per_rank[
                stage.stage_config.gpu_ranks.index(get_global_rank())
            ]

            # cpu buffer for all activations
            # 'num_layers -1' since we don't need to store the activation of the last layer
            cpu_buffers = [
                torch.empty(
                    buffer_shape,
                    pin_memory=True,
                    dtype=self.pipeline_config.autocast_dtype,
                )
                for _ in range((len(layers) - 1) * num_microbatches)
            ]
            # cpu buffers for gradients for next layer
            cpu_grad_buffers = [
                torch.empty(
                    buffer_shape,
                    pin_memory=True,
                    dtype=self.pipeline_config.autocast_dtype,
                )
                for _ in range(num_microbatches)
            ]

            # stores current microbatch activation
            gpu_activation_buffer = torch.empty(
                buffer_shape, device=device, dtype=self.pipeline_config.autocast_dtype
            )
            # stores gradients for next layer microbatch
            gpu_grad_buffer = torch.empty(
                buffer_shape, device=device, dtype=self.pipeline_config.autocast_dtype
            )

            state = StageState(
                stage=stage,
                cpu_buffers=cpu_buffers,
                cpu_grad_buffers=cpu_grad_buffers,
                gpu_activation_buffer=gpu_activation_buffer,
                gpu_grad_buffer=gpu_grad_buffer,
                input_chunks=[None] * num_microbatches,
                target_chunks=[None] * num_microbatches,
                layers=layers,
                num_microbatches=num_microbatches,
            )
            states.append(state)

            # connect adjacent stages
            if len(states) >= 2:
                states[-2].next_stage = stage
                states[-1].prev_stage = states[-2].stage

        return states

    def sync_memory_with_compute_stream(self):
        compute_stream = self.compute_stream
        self.mem_copy_stream.wait_stream(compute_stream)

    def load_from_cpu_buffer(self, gpu_buffer, cpu_buffer, sync_compute_stream=False):
        assert (
            gpu_buffer.dtype == cpu_buffer.dtype
        ), f"load_from_cpu_buffer device mismatch {gpu_buffer.dtype} != {cpu_buffer.dtype}"
        if sync_compute_stream:
            self.sync_memory_with_compute_stream()
        with torch.cuda.stream(self.mem_copy_stream), torch.no_grad():
            gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        return gpu_buffer

    def store_to_cpu_buffer(self, gpu_buffer, cpu_buffer, sync_compute_stream=False):
        assert (
            gpu_buffer.dtype == cpu_buffer.dtype
        ), f"store_to_cpu_buffer device mismatch {gpu_buffer.dtype} != {cpu_buffer.dtype}"
        if sync_compute_stream:
            self.sync_memory_with_compute_stream()
        with torch.cuda.stream(self.mem_copy_stream), torch.no_grad():
            cpu_buffer.copy_(gpu_buffer.data, non_blocking=True)

    def _forward_layer(
        self,
        layer: nn.Module,
        input_tensor: torch.Tensor,
        is_first_mb: bool,
        is_last_mb: bool,
        skip_reshard: bool,
    ) -> torch.Tensor:
        with torch.autocast(
            device_type="cuda", dtype=self.pipeline_config.autocast_dtype
        ):
            return layer(
                input_tensor,
                is_first_microbatch=is_first_mb,
                is_last_microbatch=is_last_mb,
                skip_reshard=skip_reshard,
                in_backwards=False,
            ).detach()

    def _backward_layer(
        self,
        layer: nn.Module,
        input_tensor: torch.Tensor,
        grad: Optional[torch.Tensor],
        is_first_mb: bool,
        is_last_mb: bool,
        skip_reshard: bool,
        is_last_layer_of_model: bool,
        target: Optional[torch.Tensor] = None,
        sync_after_recompute: bool = True,
        stage_id: Optional[int] = None,
        mb_idx: Optional[int] = None,
    ) -> None:
        with torch.autocast(
            device_type="cuda", dtype=self.pipeline_config.autocast_dtype
        ):
            output = layer(
                input_tensor,
                is_first_microbatch=is_first_mb,
                is_last_microbatch=is_last_mb,
                skip_reshard=skip_reshard,
                in_backwards=True,
            )
            if is_last_layer_of_model:
                loss = torch.nn.functional.cross_entropy(
                    output.flatten(0, 1), target.flatten(0, 1)
                )
        self.pipeline_logger.debug(
            stage_id, mb_idx, " # _backward_layer recompute forwards"
        )
        # this makes sure that unused memory is cleaned up
        # observed reduced reserved memory! less cuda malloc retries!
        if sync_after_recompute:
            self.compute_stream.synchronize()
            self.pipeline_logger.debug(
                stage_id, mb_idx, " # _backward_layer synchronize"
            )

        if is_last_layer_of_model:
            loss.backward()
        else:
            assert grad is not None
            output.backward(grad)

    def _forward_stage(self, state: StageState, global_rank: int) -> List[dist.Work]:
        sends_to_wait = []
        stage_id = state.stage.stage_id

        for li, layer in enumerate(state.layers):
            is_first_layer = li == 0
            is_last_layer = li == len(state.layers) - 1
            is_last_layer_of_model = is_last_layer and state.stage.is_last_stage()

            self.pipeline_logger.debug(stage_id, "x", f"+ Starting layer forwards {li}")

            for mb_idx in range(state.num_microbatches):
                self.pipeline_logger.debug(
                    stage_id, mb_idx, " -- Processing microbatch"
                )
                if is_first_layer:
                    if not state.stage.is_first_stage():
                        recv_work = state.stage.communicator.recv_forward(
                            state.gpu_activation_buffer.shape,
                            mb_idx=mb_idx,
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
                        # Even if we receive a single tensor in our p2p operation, the result is still returned as a list, so we have to unwrap it.
                        assert len(current_input) == 1
                        assert isinstance(current_input[0], torch.Tensor)
                        state.input_chunks[mb_idx] = current_input[0]
                    layer_input = state.input_chunks[mb_idx]
                else:
                    # otherwise, fetch directly from activations storage
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
                self.pipeline_logger.log_forward_compute(
                    state.stage.stage_id,
                    mb_idx,
                    li + state.stage.stage_config.layer_partition[0],
                )

                if not is_last_layer:
                    # use memcpy stream to offload activations to CPU buffers
                    with torch.cuda.stream(self.mem_copy_stream):
                        buffer_idx = li * state.num_microbatches + mb_idx
                        state.cpu_buffers[buffer_idx].copy_(
                            output.data, non_blocking=True
                        )

                if not is_last_layer_of_model and is_last_layer:
                    state.stage.communicator.wait_for_stream()
                    send_work = state.stage.communicator.send_forward(
                        output,
                        mb_idx=mb_idx,
                    )
                    if send_work is not None:
                        sends_to_wait.append(send_work)
                    self.pipeline_logger.log_forward_send(
                        state.stage.stage_id,
                        mb_idx,
                        state.stage.communicator.comm_plan.fwd_send_to_ranks[mb_idx],
                    )

        return sends_to_wait

    def _backward_stage(self, state: StageState, global_rank: int) -> List[dist.Work]:
        stage_id = state.stage.stage_id
        sends_to_wait = []

        for li in range(len(state.layers) - 1, -1, -1):
            layer = state.layers[li]
            is_first_layer = li == 0
            is_last_layer = li == len(state.layers) - 1
            is_last_layer_of_model = is_last_layer and state.stage.is_last_stage()

            self.pipeline_logger.debug(
                stage_id, "x", f"+ Starting layer backwards {li}"
            )

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
                        self.pipeline_logger.log_backward_recv(
                            state.stage.stage_id,
                            mb_idx,
                            state.stage.communicator.comm_plan.bwd_recv_from_ranks[
                                mb_idx
                            ],
                        )
                        grad = recv_work.result()[0]
                else:
                    with torch.cuda.stream(self.mem_copy_stream), torch.no_grad():
                        state.gpu_grad_buffer.copy_(
                            state.cpu_grad_buffers[mb_idx], non_blocking=True
                        )
                        grad = state.gpu_grad_buffer

                if is_first_layer:
                    layer_input = state.input_chunks[mb_idx]
                else:
                    # load activation from previous layer microbatch
                    with torch.cuda.stream(self.mem_copy_stream), torch.no_grad():
                        # copy_ seems to have implicit sync with compute stream
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
                    skip_reshard=False,
                    is_last_layer_of_model=is_last_layer_of_model,
                    target=state.target_chunks[mb_idx],
                    sync_after_recompute=self.pipeline_config.sync_after_recompute,
                    stage_id=stage_id,
                    mb_idx=mb_idx,
                )

                self.pipeline_logger.log_backward_compute(
                    state.stage.stage_id,
                    mb_idx,
                    li + state.stage.stage_config.layer_partition[0],
                )

                if not is_first_layer:
                    # store gradient for microbatch i (for next layer)
                    with torch.cuda.stream(self.mem_copy_stream), torch.no_grad():
                        state.cpu_grad_buffers[mb_idx].copy_(
                            layer_input.grad.detach().data, non_blocking=True
                        )

                if is_first_layer and not state.stage.is_first_stage():
                    state.stage.communicator.wait_for_stream()
                    send_work = state.stage.communicator.send_backward(
                        state.input_chunks[mb_idx].grad, mb_idx=mb_idx
                    )
                    if send_work is not None:
                        sends_to_wait.append(send_work)
                    self.pipeline_logger.log_backward_send(
                        state.stage.stage_id,
                        mb_idx,
                        state.stage.communicator.comm_plan.bwd_send_to_ranks[mb_idx],
                    )

        return sends_to_wait

    def step(
        self,
        step_idx: int,
        input_chunks: Optional[List[torch.Tensor]] = None,
        target_chunks: Optional[List[torch.Tensor]] = None,
    ) -> Optional[torch.Tensor]:
        self.pipeline_logger.set_step(step_idx)
        global_rank = get_global_rank()
        # iteration 1 onwards, set unshard patch
        # TODO: copied from old, not 100% sure we need
        if step_idx > 0:
            for state in self.stage_states:
                for layer in state.layers:
                    layer._handle._needs_pre_forward_unshard = True

        for stage_idx, state in enumerate(self.stage_states):
            if input_chunks is not None and stage_idx == 0:
                state.input_chunks = input_chunks
            if target_chunks is not None:
                state.target_chunks = target_chunks
            sends = self._forward_stage(state, global_rank=global_rank)
            for work in sends:
                work.wait()
        for _, state in enumerate(reversed(self.stage_states)):
            sends = self._backward_stage(state, global_rank=global_rank)
            for work in sends:
                work.wait()

        # manually call the final callback for each fsdp module after entire backwards completes
        # post_backward_final_callback needs to be called after all backwards are scheduled
        # since it synchronizes with post backward stream
        for state in reversed(self.stage_states):
            for layer in state.layers:
                _post_backward_final_callback(layer, None)


def build_pipeline_schedule(
    pipeline_stages: List[PipelineStage],
    pipeline_config: PipelineConfig,
    gloo_p2p: bool,
) -> PipelineScheduleGpipe:
    return PipelineScheduleGpipe(pipeline_stages, pipeline_config, gloo_p2p)
