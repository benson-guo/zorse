# -*- coding: utf-8 -*-
import contextlib
from dataclasses import dataclass
import datetime
import os
from typing import Any, Dict, List
import torch
import torch.distributed as dist
from zorse_utils.argparse import parse_args_zorse, override_args_with_config
from zorse_utils.pipeline_config import parse_pipeline_config
from zorse_utils.pipeline import PipelineStage, build_pipeline_stages
from zorse_utils.pipeline_logger import init_pipeline_logger
from zorse_utils.pipeline_schedule_v2 import (
    PipelineScheduleGpipe,
    build_pipeline_schedule,
)
from zorse_utils.pipeline_schedule_no_overlap import (
    build_pipeline_schedule as build_pipeline_schedule_sync,
)
from zorse_utils.pipeline_schedule_two_stage_sync import (
    build_pipeline_schedule as build_pipeline_schedule_two_stage_sync,
)
from zorse_utils.pipeline_schedule_zero2 import (
    build_pipeline_schedule as build_pipeline_schedule_zero2,
)
from zorse_utils.pipeline_schedule_flashflex import (
    build_pipeline_schedule as build_pipeline_schedule_flashflex,
)
from utils.global_state import configure_gradient_accumulation, set_split_state
from utils.comm import get_global_rank, dist_init
from utils.logger import get_logger, init_logger
from torch.cuda import Event

from utils.profile import get_profiler_context, print_memory_stats

from utils.patch import enable_gradient_accumulation
from utils.train_utils import get_profiler_path, print_metrics

logger = get_logger()


def get_random_tensor_on_device(pipeline_config, device):
    size = (pipeline_config.microbatch_size, pipeline_config.sequence_length)
    input_ids = torch.randint(
        low=0, high=pipeline_config.vocab_size, size=size, device=device
    )
    return input_ids


@dataclass
class TrainingState:
    input_chunks: Dict[int, List[torch.Tensor]]  # stage_id -> input chunks
    target_chunks: Dict[int, List[torch.Tensor]]  # stage_id -> target chunks
    optimizers: Dict[int, Any]  # stage_id -> optimizer


def _init_training_state(
    pipeline_stages: List[PipelineStage],
    pp_schedule: PipelineScheduleGpipe,
    device: torch.device,
    fused_optimizer: bool,
    args,
) -> TrainingState:
    input_chunks = {}
    target_chunks = {}
    optimizers = {}

    for stage in pipeline_stages:
        global_rank = get_global_rank()
        global_rank_idx = stage.stage_config.gpu_ranks.index(global_rank)
        num_microbatches = stage.num_microbatches_per_rank[global_rank_idx]

        # Prepare input tensors for first stage
        if stage.is_first_stage():
            input_chunks[stage.stage_id] = [
                get_random_tensor_on_device(pp_schedule.pipeline_config, device)
                for _ in range(num_microbatches)
            ]

        # Prepare target tensors for last stage
        if stage.is_last_stage():
            target_chunks[stage.stage_id] = [
                get_random_tensor_on_device(pp_schedule.pipeline_config, device)
                for _ in range(num_microbatches)
            ]

        if not args.optimizer_in_backwards:
            optimizers[stage.stage_id] = torch.optim.Adam(
                stage.model.parameters(), lr=0.001, fused=fused_optimizer
            )

        configure_gradient_accumulation(stage.model, num_microbatches)

    if args.offload_model_params:
        set_split_state(key="offload_model_params", value=True)
        # dedicated stream for memcpy such that offloading params doesn't block other transfers/communication
        set_split_state(key="memcpy_stream", value=torch.cuda.Stream())

    if args.optimizer_in_backwards:
        set_split_state(key="optimizer_in_backwards", value=True)
        # dedicated stream for optimizer such that it doesn't block other computation/communication
        set_split_state(key="optimizer_stream", value=torch.cuda.Stream())

    return TrainingState(
        input_chunks=input_chunks, target_chunks=target_chunks, optimizers=optimizers
    )


def train(
    args,
    pipeline_stages: List[PipelineStage],
    profiler_path: str,
    device,
    pp_schedule,
    pipeline_config,
    gloo_pg: dist.ProcessGroup = None,
):
    logger = get_logger()
    start_event = Event(enable_timing=True)
    end_event = Event(enable_timing=True)

    training_state = _init_training_state(
        pipeline_stages=pipeline_stages,
        pp_schedule=pp_schedule,
        device=device,
        fused_optimizer=pipeline_config.fused_optimizer,
        args=args,
    )

    warmup_iterations = args.warmup_iterations
    print_mem_step = warmup_iterations - 3
    total_iterations = args.warmup_iterations + args.iterations
    max_allocated_mem = 0

    # use gloo pg for barriersince it works with amd + nvidia
    if gloo_pg is None:
        gloo_pg = dist.new_group(
            ranks=list(range(args.world_size)),
            backend="gloo",
            timeout=datetime.timedelta(seconds=60),
        )
    for step_idx in range(total_iterations):
        if step_idx == warmup_iterations:
            dist.barrier(group=gloo_pg)
            start_event.record()  # type: ignore

        if (
            step_idx
            in range(warmup_iterations - args.profiling_iterations, warmup_iterations)
            and not args.skip_profile
        ):
            out_dir = os.path.join(profiler_path, f"iteration_{step_idx}")
            profiler_ctx = get_profiler_context(out_dir=out_dir, unique_gpus_only=True)
        else:
            profiler_ctx = contextlib.nullcontext()

        iteration_start = Event(enable_timing=True)
        iteration_end = Event(enable_timing=True)
        iteration_start.record()

        with profiler_ctx:
            pp_schedule.step(
                step_idx,
                input_chunks=training_state.input_chunks.get(0),  # only the first stage
                target_chunks=training_state.target_chunks.get(
                    pipeline_config.num_pipeline_stages - 1
                ),  # only the last stage
            )

            if step_idx == print_mem_step:
                stats = print_memory_stats("post-backward", all_ranks=True)
                max_allocated_mem = stats["max_allocated"]

            if not args.optimizer_in_backwards:
                # step the optimizer
                for _, optimizer in training_state.optimizers.items():
                    optimizer.step()
                    optimizer.zero_grad()

        # Record and log iteration time
        iteration_end.record()
        torch.cuda.synchronize()
        elapsed_time = iteration_start.elapsed_time(iteration_end)
        logger.info(f"Iteration {step_idx} Time {elapsed_time:.2f} ms")
        if (
            args.print_metrics_every_iteration
            and step_idx >= warmup_iterations
            and step_idx < total_iterations - 1
        ):
            print_metrics(
                args,
                max_allocated_mem,
                elapsed_time / (step_idx + 1.0 - warmup_iterations),
                profiler_path,
                pipeline_stages=pipeline_stages,
            )

    end_event.record()
    torch.cuda.synchronize()
    dist.barrier(group=gloo_pg)
    avg_iteration_time = start_event.elapsed_time(end_event) / args.iterations
    print_metrics(
        args,
        max_allocated_mem,
        avg_iteration_time,
        profiler_path,
        pipeline_stages=pipeline_stages,
        gloo_pg=gloo_pg,
    )


def main():
    args = parse_args_zorse()
    override_args_with_config(args)
    print(f"Log level: {args.log_level}")
    init_logger(name="zorse.py", log_level=args.log_level)
    world_size = dist_init()
    args.world_size = world_size
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    gloo_pg = None
    if args.gloo_p2p:
        all_ranks = list(range(world_size))
        gloo_pg = dist.new_group(
            ranks=all_ranks, backend="gloo", timeout=datetime.timedelta(seconds=300)
        )
        args.gloo_pg = gloo_pg

    pipeline_config = parse_pipeline_config(args.config_file)
    args.global_batch_size = pipeline_config.global_batch_size

    init_pipeline_logger("./zorse_logs", log_level=args.log_level)

    # build stages and schedule
    pipeline_stages = build_pipeline_stages(pipeline_config, gloo_pg=gloo_pg)
    for stage in pipeline_stages:
        stage.wrap_model_zero(args)

    # TODO: deprecate sync_comm in future because it is inefficient
    # keeping for now to debug/run evaluations
    assert not (
        args.two_stage_sync and args.sync_comm
    ), "Cannot use both two-stage-sync and sync-comm"
    if args.flashflex_pipeline:
        pp_schedule = build_pipeline_schedule_flashflex(
            pipeline_stages, pipeline_config, args.gloo_p2p
        )
    elif args.zero2_pipeline:
        pp_schedule = build_pipeline_schedule_zero2(
            pipeline_stages, pipeline_config, args.gloo_p2p
        )
    elif args.two_stage_sync:
        pp_schedule = build_pipeline_schedule_two_stage_sync(
            pipeline_stages, pipeline_config
        )
    elif args.sync_comm:
        pp_schedule = build_pipeline_schedule_sync(pipeline_stages, pipeline_config)
    else:
        pp_schedule = build_pipeline_schedule(
            pipeline_stages, pipeline_config, args.gloo_p2p
        )

    if args.offload_model_params:
        assert (
            args.optimizer_in_backwards
        ), "Offloading model params requires optimizer in backwards"

    enable_gradient_accumulation()

    profiler_path = get_profiler_path(args)
    train(
        args=args,
        pipeline_stages=pipeline_stages,
        profiler_path=profiler_path,
        device=device,
        pp_schedule=pp_schedule,
        pipeline_config=pipeline_config,
        gloo_pg=gloo_pg,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
