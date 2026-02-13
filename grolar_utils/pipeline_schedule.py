# -*- coding: utf-8 -*-
from typing import List, Optional, Dict
import torch
import torch.distributed as dist
from torch.distributed.fsdp._runtime_utils import _post_backward_final_callback
from grolar_utils.pipeline_config import PipelineConfig
from grolar_utils.pipeline import PipelineStage
from utils.comm import get_global_rank, get_local_rank
from utils.logger import get_logger
from models.hub import get_all_layers


class PipelineScheduleGpipe:
    def __init__(self, pipeline_stage: PipelineStage, pipeline_config: PipelineConfig):
        self.stage = pipeline_stage
        self.num_stages = pipeline_stage.num_stages
        self.num_microbatches_per_rank = pipeline_stage.num_microbatches_per_rank
        self.gpu_ranks = pipeline_stage.stage_config.gpu_ranks
        self.pipeline_config = pipeline_config

        self.activations: Dict[int, torch.Tensor] = {}

        self.fwd_sends_to_wait: List[dist.Work] = []
        self.bwd_sends_to_wait: List[dist.Work] = []

    def step(
        self,
        step_idx: int,
        input_chunks: Optional[list[torch.Tensor]] = None,
        target_chunks: Optional[list[torch.Tensor]] = None,
    ) -> Optional[torch.Tensor]:
        logger = get_logger()
        self.activations.clear()
        self.fwd_sends_to_wait.clear()
        self.bwd_sends_to_wait.clear()
        global_rank = get_global_rank()
        global_rank_idx = self.gpu_ranks.index(global_rank)
        local_rank = get_local_rank()
        device = torch.device("cuda", local_rank)
        microbatches_for_rank = self.num_microbatches_per_rank[global_rank_idx]
        logger.debug(f"Microbatches for rank {global_rank}: {microbatches_for_rank}")
        if input_chunks is None:
            input_chunks = [None] * microbatches_for_rank  # type: ignore

        if target_chunks is None:
            target_chunks = [None] * microbatches_for_rank  # type: ignore

        stage_layers = get_all_layers(self.stage.model)
        num_layers = len(stage_layers)
        # iteration 1 onwards, set unshard patch
        # TODO: copied from old, not 100% sure we need
        if step_idx > 0:
            for layer in stage_layers:
                layer._handle._needs_pre_forward_unshard = True

        # store activations during forwards pass
        buffer_shape = torch.Size(
            [
                self.pipeline_config.microbatch_size,
                self.pipeline_config.sequence_length,
                self.stages[0].model._hidden_size,
            ]
        )
        # cpu buffer for all activations
        # 'num_layers -1' since we don't need to store the activation of the last layer
        num_buffers = (num_layers - 1) * microbatches_for_rank
        cpu_buffers = [
            torch.empty(buffer_shape, pin_memory=True) for _ in range(num_buffers)
        ]
        # cpu buffers for gradients for next layer
        cpu_grad_buffers = [
            torch.empty(buffer_shape, pin_memory=True)
            for _ in range(microbatches_for_rank)
        ]
        # stores current microbatch activation
        gpu_activation_buffer = torch.empty(buffer_shape, device=device)
        # stores gradients for next layer microbatch
        gpu_grad_buffer = torch.empty(buffer_shape, device=device)
        # mem copy stream for copying results to cpu buffer
        mem_copy_stream = torch.cuda.Stream()
        # use default stream for compute
        compute_stream = torch.cuda.current_stream()

        # Fill
        with torch.autocast(
            device_type="cuda", dtype=self.pipeline_config.autocast_dtype
        ):
            # loop over all layers
            for li, layer in enumerate(stage_layers):
                is_first_layer_of_stage = li == 0
                is_last_layer_of_stage = li == num_layers - 1
                is_last_layer_of_model = (
                    is_last_layer_of_stage and self.stage.is_last_stage()
                )
                logger.debug(f"+ f layer {li}")

                for mb_idx in range(microbatches_for_rank):
                    logger.debug(f" -- f microbatch {mb_idx}")
                    is_first_microbatch = mb_idx == 0
                    is_last_microbatch = mb_idx == microbatches_for_rank - 1

                    if is_first_layer_of_stage:
                        if not self.stage.is_first_stage():
                            logger.debug(
                                f"Microbatch {mb_idx} waiting for input rank {global_rank}"
                            )
                            recv_work = self.stage.communicator.recv_forward(
                                buffer_shape, mb_idx=mb_idx
                            )
                            recv_work.wait()
                            current_input = recv_work.result()
                            # Even if we receive a single tensor in our p2p operation, the result is still returned as a list, so we have to unwrap it.
                            assert len(current_input) == 1
                            assert isinstance(current_input[0], torch.Tensor)
                            layer_input = current_input[0]
                            input_chunks[mb_idx] = layer_input
                            logger.debug(
                                f"Microbatch {mb_idx} received input rank {global_rank}"
                            )
                        else:
                            layer_input = input_chunks[mb_idx]
                    else:
                        # otherwise, fetch directly from activations storage
                        layer_input = cpu_buffers[
                            (li - 1) * microbatches_for_rank + mb_idx
                        ]

                    logger.debug(
                        f"Microbatch {mb_idx} forwards start rank {global_rank}"
                    )
                    output = layer(
                        layer_input,
                        is_first_microbatch=is_first_microbatch,
                        is_last_microbatch=is_last_microbatch,
                        skip_reshard=is_last_layer_of_model,
                        in_backwards=False,
                    ).detach()
                    if not is_last_layer_of_stage:
                        # use memcpy stream to offload activations to CPU buffers
                        with torch.cuda.stream(mem_copy_stream):
                            buffer_idx = li * microbatches_for_rank + mb_idx
                            cpu_buffers[buffer_idx].copy_(
                                output.data, non_blocking=True
                            )
                    logger.debug(f"Microbatch {mb_idx} forwards end rank {global_rank}")

                    if is_last_layer_of_stage and not self.stage.is_last_stage():
                        self.stage.communicator.wait_for_stream()
                        send_work = self.stage.communicator.send_forward(
                            output, mb_idx=mb_idx
                        )
                        if send_work is not None:
                            self.fwd_sends_to_wait.append(send_work)

        for work in self.fwd_sends_to_wait:
            work.wait()

        # Drain
        # loop over layers (starting from last)
        for li in range(num_layers - 1, -1, -1):
            is_first_layer_of_stage = li == 0
            is_last_layer_of_stage = li == num_layers - 1
            is_last_layer_of_model = (
                is_last_layer_of_stage and self.stage.is_last_stage()
            )
            layer = stage_layers[li]
            logger.debug(f"+ b layer {li}")  # start backwards for layer li

            for mb_idx in range(microbatches_for_rank - 1, -1, -1):
                is_first_microbatch = mb_idx == microbatches_for_rank - 1
                is_last_microbatch = mb_idx == 0
                logger.debug(f" -- b microbatch {mb_idx} - forwards")

                if is_last_layer_of_stage:
                    if not self.stage.is_last_stage():
                        recv_work = self.stage.communicator.recv_backward(
                            buffer_shape, mb_idx=mb_idx
                        )
                        if recv_work is not None:
                            recv_work.wait()
                            grad = recv_work.result()[0]  # List[tensor]
                    else:
                        grad = None
                else:
                    with torch.cuda.stream(mem_copy_stream), torch.no_grad():
                        gpu_grad_buffer.copy_(
                            cpu_grad_buffers[mb_idx], non_blocking=True
                        )

                    grad = gpu_grad_buffer

                logger.debug(
                    f"Microbatch {mb_idx} recompute backwards start rank {global_rank}"
                )
                if is_first_layer_of_stage:
                    layer_input = input_chunks[mb_idx]
                else:
                    # load activation from previous layer microbatch
                    buffer_idx_curr = (li - 1) * microbatches_for_rank + mb_idx
                    # use memcpy stream, no_grad to avoid copying errors
                    with torch.cuda.stream(mem_copy_stream), torch.no_grad():
                        # copy_ seems to have implicit sync with compute stream
                        gpu_activation_buffer.copy_(
                            cpu_buffers[buffer_idx_curr],
                            non_blocking=True,
                        )

                    gpu_activation_buffer.requires_grad = True
                    layer_input = gpu_activation_buffer

                output = layer(
                    layer_input,
                    is_first_microbatch=is_first_microbatch,
                    is_last_microbatch=is_last_microbatch,
                    skip_reshard=is_last_layer_of_model,
                    in_backwards=True,
                )
                # this makes sure that unused memory is cleaned up
                # observed reduced reserved memory! less cuda malloc retries!
                # TODO: this can potentially slow the code down, so add option to skip synchronize
                compute_stream.synchronize()

                logger.debug(f"Microbatch {mb_idx} backwards start rank {global_rank}")
                if is_last_layer_of_model:
                    # compute loss
                    output = torch.nn.functional.cross_entropy(
                        output.flatten(0, 1), target_chunks[mb_idx].flatten(0, 1)
                    )
                    output.backward()
                else:
                    assert (
                        grad is not None
                    ), f"Grad is None for microbatch {mb_idx}, layer {li} rank {global_rank}"
                    output.backward(grad)  # type: ignore
                logger.debug(f"Microbatch {mb_idx} backwards end rank {global_rank}")

                if not is_first_layer_of_stage:
                    # store gradient for microbatch i (for next layer)
                    with torch.cuda.stream(mem_copy_stream), torch.no_grad():
                        cpu_grad_buffers[mb_idx].copy_(
                            layer_input.grad.detach().data,
                            non_blocking=True,
                        )

                if is_first_layer_of_stage and not self.stage.is_first_stage():
                    self.stage.communicator.wait_for_stream()
                    send_work = self.stage.communicator.send_backward(
                        input_chunks[mb_idx].grad, mb_idx=mb_idx
                    )
                    if send_work is not None:
                        self.bwd_sends_to_wait.append(send_work)

        # manually call the final callback for each fsdp module after entire backwards completes
        for layer in stage_layers:
            _post_backward_final_callback(layer, None)

        for work in self.bwd_sends_to_wait:
            work.wait()


def build_pipeline_schedule(
    pipeline_stage: PipelineStage, pipeline_config: PipelineConfig
):
    """
    This function just exists if we need some other logic here later
    Maybe different type of schedules (1F1B etc)
    """
    return PipelineScheduleGpipe(pipeline_stage, pipeline_config)
