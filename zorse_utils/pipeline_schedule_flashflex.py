# -*- coding: utf-8 -*-
"""
Standard pipeline parallelism with gradient accumulation except we use ZeRO2
FlashFlex Version: Similar to pipeline_schedule_zero2.py but without offloading
"""

# -*- coding: utf-8 -*-
from typing import List, Optional
import torch
from torch.distributed.fsdp._common_utils import _FSDPState
from torch.distributed.fsdp._flat_param import HandleTrainingState
import torch.distributed as dist

from zorse_utils.pipeline import PipelineStage
from zorse_utils.pipeline_config import PipelineConfig
from zorse_utils.pipeline_schedule_v2 import StageState, PipelineScheduleGpipe
from models.hub import get_all_layers
from utils.comm import get_global_rank
from utils.global_state import get_split_state, set_split_state
from utils.patch import _unshard_patch
from torch.distributed.fsdp._runtime_utils import _post_backward_final_callback


class PipelineScheduleGpipeFlashFlex(PipelineScheduleGpipe):
    def __init__(
        self,
        pipeline_stages: List[PipelineStage],
        pipeline_config: PipelineConfig,
        gloo_p2p: bool = False,
    ):
        super().__init__(pipeline_stages, pipeline_config, gloo_p2p=gloo_p2p)
        self.gloo_p2p = gloo_p2p
        self.recv_batch_gloo = self.gloo_p2p and pipeline_config.gloo_batch_recv
        self.alternate_nccl_gloo = pipeline_config.alternate_nccl_gloo
        set_split_state("zero2_pipeline", True)

    def irecv_forward(
        self, communicator, state: StageState, stage_id: int, mb_idx: int
    ) -> dist.Work:
        """Initiate an asynchronous forward receive operation."""
        comm_plan = communicator.comm_plan
        self.pipeline_logger.debug(
            stage_id,
            mb_idx,
            f" pre recv forwards from {comm_plan.fwd_recv_from_ranks[mb_idx]}",
        )
        recv_work = communicator.recv_forward(
            state.gpu_activation_buffer.shape, mb_idx=mb_idx
        )
        self.pipeline_logger.debug(
            stage_id,
            mb_idx,
            " post recv forwards",
        )
        return recv_work

    def wait_irecv_forward(
        self,
        recv_work: dist.Work,
        communicator,
        state: StageState,
        stage_id: int,
        mb_idx: int,
    ) -> torch.Tensor:
        """Wait for the asynchronous forward receive operation to complete and return the result."""
        comm_plan = communicator.comm_plan
        # recv_work should not be None if this function is called.
        assert recv_work is not None
        recv_work.wait()
        layer_input = recv_work.result()[0]
        assert isinstance(layer_input, torch.Tensor)
        self.pipeline_logger.log_forward_recv(
            stage_id,
            mb_idx,
            comm_plan.fwd_recv_from_ranks[mb_idx],
        )
        return layer_input

    def recv_forward(
        self, communicator, state: StageState, stage_id: int, mb_idx: int
    ) -> torch.Tensor:
        """Perform a synchronous forward receive operation."""
        recv_work = self.irecv_forward(communicator, state, stage_id, mb_idx)
        assert recv_work is not None
        layer_input = self.wait_irecv_forward(
            recv_work, communicator, state, stage_id, mb_idx
        )
        return layer_input

    def recv_batch_forwards(
        self, communicator, state: StageState, stage_id: int
    ) -> List[dist.Work]:
        recv_works = []

        for mb_idx in range(state.num_microbatches):
            recv_work = communicator.recv_forward(
                state.gpu_activation_buffer.shape, mb_idx=mb_idx
            )
            recv_works.append(recv_work)
        return recv_works

    def irecv_backward(
        self, communicator, state: StageState, stage_id: int, mb_idx: int
    ) -> dist.Work:
        """Initiate an asynchronous backward receive operation."""
        comm_plan = communicator.comm_plan
        self.pipeline_logger.debug(
            stage_id,
            mb_idx,
            f" pre recv backwards from {comm_plan.bwd_recv_from_ranks[mb_idx]}",
        )
        recv_work = communicator.recv_backward(
            state.gpu_grad_buffer.shape, mb_idx=mb_idx
        )
        self.pipeline_logger.debug(
            stage_id,
            mb_idx,
            " post recv backwards",
        )
        return recv_work

    def wait_irecv_backward(
        self,
        recv_work: dist.Work,
        communicator,
        state: StageState,
        stage_id: int,
        mb_idx: int,
    ) -> torch.Tensor:
        """Wait for the asynchronous backward receive operation to complete and return the result."""
        comm_plan = communicator.comm_plan
        # recv_work should not be None if this function is called.
        assert recv_work is not None
        recv_work.wait()
        grad = recv_work.result()[0]
        assert isinstance(grad, torch.Tensor)
        self.pipeline_logger.log_backward_recv(
            stage_id,
            mb_idx,
            comm_plan.bwd_recv_from_ranks[mb_idx],
        )
        return grad

    def recv_backward(
        self, communicator, state: StageState, stage_id: int, mb_idx: int
    ) -> torch.Tensor:
        """Perform a synchronous backward receive operation."""
        recv_work = self.irecv_backward(communicator, state, stage_id, mb_idx)
        assert recv_work is not None
        grad = self.wait_irecv_backward(
            recv_work, communicator, state, stage_id, mb_idx
        )
        return grad

    def recv_batch_backwards(
        self, communicator, state: StageState, stage_id: int
    ) -> List[dist.Work]:
        recv_works = []

        for mb_idx in range(state.num_microbatches):
            recv_work = communicator.recv_backward(
                state.gpu_grad_buffer.shape, mb_idx=mb_idx
            )
            recv_works.append(recv_work)
        return recv_works

    def send_forward(
        self,
        communicator,
        state: StageState,
        stage_id: int,
        mb_idx: int,
        output: torch.Tensor,
    ) -> dist.Work:
        comm_plan = communicator.comm_plan
        self.pipeline_logger.debug(
            stage_id,
            mb_idx,
            f" pre send forwards to {comm_plan.fwd_send_to_ranks[mb_idx]}",
        )
        send_work = communicator.send_forward(output, mb_idx=mb_idx)
        self.pipeline_logger.debug(
            stage_id,
            mb_idx,
            " post send forwards",
        )
        assert send_work is not None
        self.pipeline_logger.log_forward_send(
            stage_id,
            mb_idx,
            comm_plan.fwd_send_to_ranks[mb_idx],
        )
        return send_work

    def send_backward(
        self,
        communicator,
        state: StageState,
        stage_id: int,
        mb_idx: int,
        grad: torch.Tensor,
    ) -> dist.Work:
        comm_plan = communicator.comm_plan
        self.pipeline_logger.debug(
            stage_id,
            mb_idx,
            f" pre send backwards to {comm_plan.bwd_send_to_ranks[mb_idx]}",
        )
        send_work = communicator.send_backward(grad, mb_idx=mb_idx)
        self.pipeline_logger.debug(
            stage_id,
            mb_idx,
            " post send backwards",
        )
        assert send_work is not None
        self.pipeline_logger.log_backward_send(
            stage_id,
            mb_idx,
            comm_plan.bwd_send_to_ranks[mb_idx],
        )
        return send_work

    def _forward_stage(self, state: StageState, global_rank: int) -> None:
        sends_to_wait = []
        stage = state.stage
        stage_id = stage.stage_id
        is_first_stage_of_group = state.prev_stage is None
        is_last_stage_of_group = state.next_stage is None
        state.layer_inputs = [[] for _ in range(len(state.layers))]

        # Select communicator based on gloo_p2p and alternate_nccl_gloo flags
        recv_communicator = (
            stage.gloo_communicator if self.gloo_p2p else stage.communicator
        )
        send_communicator = (
            stage.gloo_communicator if self.gloo_p2p else stage.communicator
        )
        stage_recv_batch_gloo = self.recv_batch_gloo
        if self.gloo_p2p and self.alternate_nccl_gloo:
            stage_size = stage.stage_size()
            if not stage.is_first_stage() and stage_size == stage.prev_stage_size():
                recv_communicator = stage.communicator
                stage_recv_batch_gloo = False
            if not stage.is_last_stage() and stage_size == stage.next_stage_size():
                send_communicator = stage.communicator

        if is_first_stage_of_group:
            self.prefetch_stage_layers(stage, reverse_stages=False)
        if state.next_stage is not None:
            self.prefetch_stage_layers(state.next_stage, reverse_stages=False)

        if not stage.is_first_stage():
            # allocate cpu buffers for storing inputs
            for mb_idx in range(state.num_microbatches):
                state.input_chunks[mb_idx] = None

            if stage_recv_batch_gloo:
                recv_works = self.recv_batch_forwards(
                    recv_communicator, state, stage_id
                )

        next_mb_input_recv = None
        for mb_idx in range(state.num_microbatches):
            self.pipeline_logger.debug(stage_id, mb_idx, " -- Processing microbatch")
            # Process all microbatches for this layer
            output = None
            for li, layer in enumerate(state.layers):
                is_first_layer = li == 0
                is_last_layer = li == len(state.layers) - 1

                self.pipeline_logger.debug(
                    stage_id, "x", f"+ Starting layer forwards {li}"
                )

                if is_first_layer:
                    if not stage.is_first_stage():
                        if stage_recv_batch_gloo:
                            recv_works[mb_idx].wait()
                            layer_input = recv_works[mb_idx].result()[0]
                        elif mb_idx == 0:
                            layer_input = self.recv_forward(
                                recv_communicator, state, stage_id, mb_idx
                            )
                        else:
                            layer_input = self.wait_irecv_forward(
                                next_mb_input_recv,
                                recv_communicator,
                                state,
                                stage_id,
                                mb_idx,
                            )
                        state.input_chunks[mb_idx] = layer_input
                    else:
                        layer_input = state.input_chunks[mb_idx]
                else:
                    layer_input = output.detach()
                    state.layer_inputs[li].append(layer_input)

                output = self._forward_layer(
                    layer,
                    layer_input,
                    is_first_mb=(mb_idx == 0),
                    is_last_mb=(mb_idx == state.num_microbatches - 1),
                    skip_reshard=is_last_stage_of_group,
                )
                if (
                    is_first_layer
                    and not stage.is_first_stage()
                    and not stage_recv_batch_gloo
                    and mb_idx < state.num_microbatches - 1
                ):
                    # start receiving next microbatch so it can overlap with send
                    next_mb_input_recv = self.irecv_forward(
                        recv_communicator, state, stage_id, mb_idx + 1
                    )
                self.pipeline_logger.debug(
                    stage_id,
                    mb_idx,
                    f" layer output {output.dtype} {state.gpu_activation_buffer.dtype}",
                )

                self.pipeline_logger.log_forward_compute(
                    stage_id,
                    mb_idx,
                    li + stage.stage_config.layer_partition[0],
                )

                if not is_last_layer:
                    pass
                else:
                    if not stage.is_last_stage():
                        send_communicator.wait_for_stream()
                        send_work = self.send_forward(
                            send_communicator, state, stage_id, mb_idx, output
                        )
                        sends_to_wait.append(send_work)

        return sends_to_wait

    def _backward_stage(self, state: StageState, global_rank: int) -> None:
        sends_to_wait = []
        stage = state.stage
        stage_id = stage.stage_id

        # Select communicator based on gloo_p2p and alternate_nccl_gloo flags
        recv_communicator = (
            stage.gloo_communicator if self.gloo_p2p else stage.communicator
        )
        send_communicator = (
            stage.gloo_communicator if self.gloo_p2p else stage.communicator
        )
        stage_recv_batch_gloo = self.recv_batch_gloo
        if self.gloo_p2p and self.alternate_nccl_gloo:
            stage_size = stage.stage_size()
            if not stage.is_last_stage() and stage_size == stage.next_stage_size():
                recv_communicator = stage.communicator
                stage_recv_batch_gloo = False
            if not stage.is_first_stage() and stage_size == stage.prev_stage_size():
                send_communicator = stage.communicator

        if state.prev_stage is not None:
            self.prefetch_stage_layers(state.prev_stage, reverse_stages=True)
        if not stage.is_last_stage() and stage_recv_batch_gloo:
            recv_works = self.recv_batch_backwards(recv_communicator, state, stage_id)

        next_mb_grad_recv = None
        for mb_idx in range(state.num_microbatches - 1, -1, -1):
            self.pipeline_logger.debug(stage_id, mb_idx, " -- Processing microbatch")
            # Process all layers for this microbatch
            grad = None
            for li in range(len(state.layers) - 1, -1, -1):
                layer = state.layers[li]
                is_first_layer = li == 0
                is_last_layer = li == len(state.layers) - 1
                is_last_layer_of_model = is_last_layer and stage.is_last_stage()

                self.pipeline_logger.debug(
                    stage_id, "x", f"+ Starting layer backwards {li}"
                )

                if is_last_layer:
                    if not stage.is_last_stage():
                        if stage_recv_batch_gloo:
                            recv_works[mb_idx].wait()
                            grad = recv_works[mb_idx].result()[0]
                        elif mb_idx == state.num_microbatches - 1:
                            grad = self.recv_backward(
                                recv_communicator, state, stage_id, mb_idx
                            )
                        else:
                            # set to grad received from previous microbatch
                            grad = self.wait_irecv_backward(
                                next_mb_grad_recv,
                                recv_communicator,
                                state,
                                stage_id,
                                mb_idx,
                            )

                if is_first_layer:
                    if stage.is_first_stage():
                        layer_input = state.input_chunks[mb_idx]
                    else:
                        layer_input = state.input_chunks[mb_idx]
                        layer_input.requires_grad = True
                else:
                    layer_input = state.layer_inputs[li][mb_idx]
                    layer_input.requires_grad = True
                self.compute_stream.wait_stream(self.mem_copy_stream)

                # Should be is last stage
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
                    stage_id=stage_id,
                    mb_idx=mb_idx,
                )

                if (
                    is_last_layer
                    and not stage.is_last_stage()
                    and not stage_recv_batch_gloo
                    and mb_idx > 0
                ):
                    # start receiving next microbatch so it can overlap with send
                    next_mb_grad_recv = self.irecv_backward(
                        recv_communicator, state, stage_id, mb_idx - 1
                    )

                self.pipeline_logger.log_backward_compute(
                    stage_id,
                    mb_idx,
                    li + stage.stage_config.layer_partition[0],
                )

                if not is_first_layer:
                    grad = layer_input.grad.detach()
                    self.pipeline_logger.debug(
                        stage_id,
                        mb_idx,
                        f" layer output {layer_input.grad.dtype} {state.cpu_grad_buffers[0].dtype}",
                    )
                elif not stage.is_first_stage():
                    grad = layer_input.grad.detach()
                    self.pipeline_logger.debug(
                        stage_id,
                        mb_idx,
                        " wait for stream",
                    )
                    send_communicator.wait_for_stream()
                    self.pipeline_logger.debug(
                        stage_id,
                        mb_idx,
                        " post wait for stream",
                    )
                    send_work = self.send_backward(
                        send_communicator, state, stage_id, mb_idx, grad
                    )
                    sends_to_wait.append(send_work)

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

    def prefetch_stage_layers(self, stage: PipelineStage, reverse_stages=False):
        stage_layers = get_all_layers(stage.model)
        num_layers = len(stage_layers)
        layer_indices = (
            range(num_layers - 1, -1, -1) if reverse_stages else range(num_layers)
        )
        # _unshard_stream not initialized yet (during first iteration)
        if not hasattr(stage_layers[0], "_unshard_stream"):
            return

        self.pipeline_logger.debug(
            stage.stage_id, "x", f"+ Prefetching {num_layers} layers"
        )
        # wait until previous stage is done
        stage_layers[0]._unshard_stream.wait_stream(torch.cuda.current_stream())
        for li in layer_indices:
            stage_layer = stage_layers[li]
            _prefetch_layer(stage_layer)
        self.pipeline_logger.debug(
            stage.stage_id, "x", f"+ Finished prefetching {num_layers} layers"
        )


def build_pipeline_schedule(
    pipeline_stages: List[PipelineStage],
    pipeline_config: PipelineConfig,
    gloo_p2p: bool,
) -> PipelineScheduleGpipeFlashFlex:
    return PipelineScheduleGpipeFlashFlex(pipeline_stages, pipeline_config, gloo_p2p)


# Wrapper around the logic in torch.distributed.fsdp._runtime_utils._prefetch_handle
# Unshard the parameters of the layer passed in
def _prefetch_layer(
    state: _FSDPState,
) -> None:
    current_handle = state._handle
    if not current_handle:
        return

    target_handle_candidate = state._handle
    if (
        target_handle_candidate
        # since forwards has not registered backwards hook, this will always be false
        # and target_handle_candidate._needs_pre_backward_unshard
        and not target_handle_candidate._prefetched
    ):
        handle = target_handle_candidate
    else:
        return

    flat_param = handle.flat_param
    split_state = get_split_state()
    if hasattr(flat_param, "cpu_local_shard"):
        # Copy the CPU local shard to the compute device on separate stream so it can run in parallel
        with torch.cuda.stream(split_state["memcpy_stream"]):
            flat_param.data = torch.empty_like(
                flat_param.cpu_local_shard, device=state.compute_device
            )
            flat_param.data.copy_(flat_param.cpu_local_shard, non_blocking=True)
            flat_param._local_shard = flat_param.data

        # Wait for the memcpy_stream to finish before unsharding
        state._unshard_stream.wait_stream(split_state["memcpy_stream"])

    # Temporarily emulate the training state while calling `_unshard` to
    # ensure the correct `as_params` for `_use_unsharded_views()`
    prev_training_state = handle._training_state
    handle._training_state = HandleTrainingState.FORWARD
    # Prefetch the next set of handles without synchronizing to allow
    # the sync to happen as late as possible to maximize overlap
    _unshard_patch(state, handle, state._unshard_stream, state._pre_unshard_stream)
    handle._training_state = prev_training_state
    handle._prefetched = True
