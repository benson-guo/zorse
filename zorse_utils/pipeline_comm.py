# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Optional, List
import torch.distributed as dist
import torch
from zorse_utils.pipeline_config import PipelineConfig
from utils.optimizer_utils import load_model_latencies_per_gpu
from utils.comm import (
    get_global_rank,
    get_gpu_name,
    get_world_size,
)
from utils.runtime_estimator import GPU_MEMORY
from utils.logger import get_logger


@dataclass
class StageCommPattern:
    """
    Communication pattern for a stage
    """

    stage_id: int
    fwd_send_to_ranks: List[Optional[int]]
    fwd_recv_from_ranks: List[Optional[int]]
    bwd_send_to_ranks: List[Optional[int]]
    bwd_recv_from_ranks: List[Optional[int]]

    fwd_send_tags: Optional[List[Optional[int]]] = None
    fwd_recv_tags: Optional[List[Optional[int]]] = None
    bwd_send_tags: Optional[List[Optional[int]]] = None
    bwd_recv_tags: Optional[List[Optional[int]]] = None

    def __repr__(self) -> str:
        return (
            f"StageCommPattern(stage={self.stage_id}, "
            f"fwd_send={self.fwd_send_to_ranks}, "
            f"fwd_recv={self.fwd_recv_from_ranks}, "
            f"bwd_send={self.bwd_send_to_ranks}, "
            f"bwd_recv={self.bwd_recv_from_ranks})"
            f"fwd_send_tags={self.fwd_send_tags}, "
            f"fwd_recv_tags={self.fwd_recv_tags}, "
            f"bwd_send_tags={self.bwd_send_tags}, "
            f"bwd_recv_tags={self.bwd_recv_tags})"
        )


@dataclass
class RankCommPlan:
    """
    Communication plan for a global rank
    """

    stage_comm_patterns: Dict[int, StageCommPattern]  # stage_id -> plan


def _validate_static_comm_plans(
    comm_plans: Dict[int, RankCommPlan], pipeline_config: PipelineConfig
):
    for rank, plan in comm_plans.items():
        for stage_id, pattern in plan.stage_comm_patterns.items():
            stage = pipeline_config.pipeline_stages[stage_id]
            rank_idx = stage.gpu_ranks.index(rank)
            num_mbs = stage.num_microbatches_per_rank[rank_idx]

            # First stage should only send forward
            if stage_id == 0:
                assert all(
                    x is None for x in pattern.fwd_recv_from_ranks
                ), f"First stage rank {rank} should not receive forward"
                assert all(
                    x is None for x in pattern.bwd_send_to_ranks
                ), f"First stage rank {rank} should not send backward"

            # Last stage should only receive forward
            if stage_id == pipeline_config.num_pipeline_stages - 1:
                assert all(
                    x is None for x in pattern.fwd_send_to_ranks
                ), f"Last stage rank {rank} should not send forward"
                assert all(
                    x is None for x in pattern.bwd_recv_from_ranks
                ), f"Last stage rank {rank} should not receive backward"

            # Middle stages should have matching sends/receives
            if 0 < stage_id < pipeline_config.num_pipeline_stages - 1:
                for mb_idx in range(num_mbs):
                    if pattern.fwd_send_to_ranks[mb_idx] is not None:
                        assert (
                            pattern.bwd_recv_from_ranks[mb_idx]
                            == pattern.fwd_send_to_ranks[mb_idx]
                        ), f"Mismatched forward/backward for rank {rank} stage {stage_id} mb {mb_idx}"
                    if pattern.fwd_recv_from_ranks[mb_idx] is not None:
                        assert (
                            pattern.bwd_send_to_ranks[mb_idx]
                            == pattern.fwd_recv_from_ranks[mb_idx]
                        ), f"Mismatched forward/backward for rank {rank} stage {stage_id} mb {mb_idx}"


def create_even_mb_comm_plan(
    pipeline_config: PipelineConfig,
) -> Dict[int, RankCommPlan]:
    def _create_stage_comm_pattern(stage_id, num_mbs):
        return StageCommPattern(
            stage_id=stage_id,
            fwd_send_to_ranks=[None] * num_mbs,
            fwd_recv_from_ranks=[None] * num_mbs,
            bwd_send_to_ranks=[None] * num_mbs,
            bwd_recv_from_ranks=[None] * num_mbs,
            fwd_send_tags=[None] * num_mbs,
            fwd_recv_tags=[None] * num_mbs,
            bwd_send_tags=[None] * num_mbs,
            bwd_recv_tags=[None] * num_mbs
        )
    
    comm_plans = {}
    for stage_idx, stage in enumerate(pipeline_config.pipeline_stages):
        for rank_idx, rank in enumerate(stage.gpu_ranks):
            num_mbs = stage.num_microbatches_per_rank[rank_idx]
            comm_plans[rank] = RankCommPlan(stage_comm_patterns={stage_idx: _create_stage_comm_pattern(stage_idx, num_mbs)})

    tag_counter = 0
    for stage_idx in range(pipeline_config.num_pipeline_stages - 1):
        cur_stage = pipeline_config.pipeline_stages[stage_idx]
        next_stage = pipeline_config.pipeline_stages[stage_idx + 1]

        for rank_idx, rank in enumerate(cur_stage.gpu_ranks):
            num_mbs = cur_stage.num_microbatches_per_rank[rank_idx]
            if stage_idx not in comm_plans[rank].stage_comm_patterns:
                comm_plans[rank].stage_comm_patterns[stage_idx] = _create_stage_comm_pattern(stage_idx, num_mbs)

        mb_idx = 0

        while mb_idx < pipeline_config.num_microbatches:
            cur_rank_idx = mb_idx % len(cur_stage.gpu_ranks)
            next_rank_idx = mb_idx % len(next_stage.gpu_ranks)

            cur_rank = cur_stage.gpu_ranks[cur_rank_idx]
            next_rank = next_stage.gpu_ranks[next_rank_idx]

            cur_local_mb = mb_idx // len(cur_stage.gpu_ranks)
            next_local_mb = mb_idx // len(next_stage.gpu_ranks)

            # Set forward comm
            tag_counter_send = tag_counter
            tag_counter += 1
            tag_counter_recv = tag_counter
            tag_counter += 1
            cur_pattern = comm_plans[cur_rank].stage_comm_patterns[stage_idx]
            cur_pattern.fwd_send_to_ranks[cur_local_mb] = next_rank
            cur_pattern.fwd_send_tags[cur_local_mb] = tag_counter_send
            cur_pattern.bwd_recv_from_ranks[cur_local_mb] = next_rank
            cur_pattern.bwd_recv_tags[cur_local_mb] = tag_counter_recv

            # Set backward comm
            if stage_idx + 1 not in comm_plans[next_rank].stage_comm_patterns:
                num_mbs = next_stage.num_microbatches_per_rank[next_rank_idx]
                next_pattern = _create_stage_comm_pattern(stage_idx + 1, num_mbs)
                comm_plans[next_rank].stage_comm_patterns[stage_idx + 1] = next_pattern
            else:
                next_pattern = comm_plans[next_rank].stage_comm_patterns[stage_idx + 1]

            next_pattern.fwd_recv_from_ranks[next_local_mb] = cur_rank
            next_pattern.fwd_recv_tags[next_local_mb] = tag_counter_send
            next_pattern.bwd_send_to_ranks[next_local_mb] = cur_rank
            next_pattern.bwd_send_tags[next_local_mb] = tag_counter_recv

            mb_idx += 1

    _validate_static_comm_plans(comm_plans=comm_plans, pipeline_config=pipeline_config)

    return comm_plans


# load layer latencies per gpu
def load_compute_times(pipeline_config: PipelineConfig) -> Dict[str, float]:
    model_name = pipeline_config.model_name
    seq_length = str(pipeline_config.sequence_length)
    dtype = str(pipeline_config.autocast_dtype).split(".")[-1]
    mbs = pipeline_config.microbatch_size
    compute_times = load_model_latencies_per_gpu(model_name, dtype, mbs, seq_length)

    return compute_times


# We use a greedy algorithm to assign cross stage communication
# This is necessary when a stage contains heterogeneous GPUs which process potentially
# different numbers of microbatches each.
# The algorithm computes the time when each microbatch from the current stage gets computed
# as well as cumulative compute time remaining at each microbatch for each GPU of the next stage
# Then it greedily assigns the ith microbatch completed to the GPU with the
# ith most work remaining. This greedy schedule will priotize sending microbatches to the GPUs
# with the most work remaining first.
def create_uneven_mb_comm_plan(
    pipeline_config: PipelineConfig,
) -> Dict[int, RankCommPlan]:
    def _create_stage_comm_pattern(stage_id, num_microbatches):
        return StageCommPattern(
            stage_id=stage_id,
            fwd_send_to_ranks=[None] * num_microbatches,
            fwd_recv_from_ranks=[None] * num_microbatches,
            bwd_send_to_ranks=[None] * num_microbatches,
            bwd_recv_from_ranks=[None] * num_microbatches,
            fwd_send_tags=[None] * num_microbatches,
            fwd_recv_tags=[None] * num_microbatches,
            bwd_send_tags=[None] * num_microbatches,
            bwd_recv_tags=[None] * num_microbatches
        )
    
    logger = get_logger()
    gpu_name = get_gpu_name()
    gpu_to_idx = {v: idx for idx, v in enumerate(list(GPU_MEMORY.keys()))}
    idx_to_gpu = {idx: v for idx, v in enumerate(list(GPU_MEMORY.keys()))}
    allocated_in = torch.tensor([gpu_to_idx[gpu_name]]).cuda()
    world_size = get_world_size()
    allocated_out = torch.tensor([0 for _ in range(world_size)]).cuda()
    dist.all_gather_into_tensor(
        allocated_out,
        allocated_in,
    )
    global_gpus = [idx_to_gpu[allocated_out[i].item()] for i in range(world_size)]
    logger.debug(f"GPUs: {global_gpus}")
    compute_times = load_compute_times(pipeline_config)
    logger.debug(f"Compute Times Per GPU: {compute_times}")

    comm_plans = {}
    world_size = get_world_size()
    for stage_idx, stage in enumerate(pipeline_config.pipeline_stages):
        for rank_idx, rank in enumerate(stage.gpu_ranks):
            num_mbs = stage.num_microbatches_per_rank[rank_idx]
            comm_plans[rank] = RankCommPlan(stage_comm_patterns={stage_idx: _create_stage_comm_pattern(stage_idx, num_mbs)})

    tag_counter = 0
    for stage_idx in range(pipeline_config.num_pipeline_stages - 1):
        cur_stage = pipeline_config.pipeline_stages[stage_idx]
        next_stage = pipeline_config.pipeline_stages[stage_idx + 1]

        cur_stage_completion_times = []
        for rank_idx, rank in enumerate(cur_stage.gpu_ranks):
            num_mbs = cur_stage.num_microbatches_per_rank[rank_idx]
            if stage_idx not in comm_plans[rank].stage_comm_patterns:
                comm_plans[rank].stage_comm_patterns[stage_idx] = _create_stage_comm_pattern(stage_idx, num_mbs)
            cur_gpu_name = global_gpus[rank]
            for i in range(num_mbs):
                layer_runtime = compute_times[cur_gpu_name]
                cur_stage_completion_times.append(
                    (layer_runtime * i, -layer_runtime, rank)
                )

        next_stage_work_remaining = []
        for rank_idx, rank in enumerate(next_stage.gpu_ranks):
            num_mbs = next_stage.num_microbatches_per_rank[rank_idx]
            next_pattern = _create_stage_comm_pattern(stage_idx + 1, num_mbs)
            comm_plans[rank].stage_comm_patterns[stage_idx + 1] = next_pattern
            cur_gpu_name = global_gpus[rank]
            for i in range(num_mbs):
                layer_runtime = compute_times[cur_gpu_name]
                next_stage_work_remaining.append(
                    (layer_runtime * (num_mbs - i), layer_runtime, rank)
                )

        # sort current stage mb completition times, next stage cumulative remaining work in descending order
        # ith completed microbatches is assigned to GPU with ith most remaining work
        cur_stage_completion_times.sort()
        next_stage_work_remaining.sort(reverse=True)
        logger.debug(
            f"Stage {stage_idx} cur_stage_completion_times: {cur_stage_completion_times}"
        )
        logger.debug(
            f"Stage {stage_idx} next_stage_work_remaining: {next_stage_work_remaining}"
        )

        cur_stage_mb_idx = {rank: 0 for rank in cur_stage.gpu_ranks}
        next_stage_mb_idx = {rank: 0 for rank in next_stage.gpu_ranks}

        for (_, _, cur_rank), (_, _, next_rank) in zip(
            cur_stage_completion_times, next_stage_work_remaining
        ):

            cur_local_mb = cur_stage_mb_idx[cur_rank]
            next_local_mb = next_stage_mb_idx[next_rank]
            cur_pattern = comm_plans[cur_rank].stage_comm_patterns[stage_idx]
            next_pattern = comm_plans[next_rank].stage_comm_patterns[stage_idx + 1]

            cur_pattern.fwd_send_to_ranks[cur_local_mb] = next_rank
            cur_pattern.fwd_send_tags[cur_local_mb] = tag_counter
            next_pattern.fwd_recv_from_ranks[next_local_mb] = cur_rank
            next_pattern.fwd_recv_tags[next_local_mb] = tag_counter

            tag_counter += 1

            cur_pattern.bwd_recv_from_ranks[cur_local_mb] = next_rank
            cur_pattern.bwd_recv_tags[cur_local_mb] = tag_counter
            next_pattern.bwd_send_to_ranks[next_local_mb] = cur_rank
            next_pattern.bwd_send_tags[next_local_mb] = tag_counter

            tag_counter += 1

            cur_stage_mb_idx[cur_rank] += 1
            next_stage_mb_idx[next_rank] += 1

    _validate_static_comm_plans(comm_plans=comm_plans, pipeline_config=pipeline_config)

    return comm_plans


def create_static_comm_plans(
    pipeline_config: PipelineConfig,
) -> Dict[int, RankCommPlan]:
    if pipeline_config.split_microbatches_evenly:
        return create_even_mb_comm_plan(pipeline_config)
    else:
        return create_uneven_mb_comm_plan(pipeline_config)


def raise_exception_with_message(func_name: str, mb_idx: int):
    global_rank = get_global_rank()
    raise Exception(
        f"{func_name} static comm plan not filled for {mb_idx} on rank {global_rank}"
    )


class PipelineCommunicator:
    def __init__(
        self,
        stage_id: int,
        num_stages: int,
        process_groups: Optional[dict],
        curr_rank: int,
        comm_plan: StageCommPattern,
        dtype: torch.dtype,
    ):
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.curr_rank = curr_rank
        self.comm_plan = comm_plan
        # We reuses streams across different PipelineCommunicators to reduce overhead
        # TODO: we may be able to share send/recv streams
        if not hasattr(PipelineCommunicator, "_shared_comm_stream_send"):
            PipelineCommunicator._shared_comm_stream_send = torch.cuda.Stream()
        self.comm_stream_send = PipelineCommunicator._shared_comm_stream_send
        if not hasattr(PipelineCommunicator, "_shared_comm_stream_recv"):
            PipelineCommunicator._shared_comm_stream_recv = torch.cuda.Stream()
        self.comm_stream_recv = PipelineCommunicator._shared_comm_stream_recv
        self.dtype = dtype

    def wait_for_stream(self, stream: Optional[torch.cuda.Stream] = None):
        sync_stream = stream if stream is not None else torch.cuda.current_stream()
        self.comm_stream_send.wait_stream(sync_stream)

    def send_forward(self, tensor: torch.Tensor, mb_idx: int) -> Optional[dist.Work]:
        dst_rank = self.comm_plan.fwd_send_to_ranks[mb_idx]
        if dst_rank is None:
            raise_exception_with_message("send_fwd", mb_idx)

        with torch.cuda.stream(self.comm_stream_send):  # type: ignore
            return dist.isend(
                tensor,
                dst=dst_rank,
            )

    def recv_forward(self, tensor_size: torch.Size, mb_idx: int) -> Optional[dist.Work]:
        src_rank = self.comm_plan.fwd_recv_from_ranks[mb_idx]
        if src_rank is None:
            raise_exception_with_message("recv_fwd", mb_idx)

        with torch.cuda.stream(self.comm_stream_recv):  # type: ignore
            buffer = torch.empty(
                tensor_size, device="cuda", dtype=self.dtype, requires_grad=True
            )
            work = dist.irecv(
                buffer,
                src=src_rank,
            )
            return work

    def send_backward(self, tensor: torch.Tensor, mb_idx: int) -> Optional[dist.Work]:
        dst_rank = self.comm_plan.bwd_send_to_ranks[mb_idx]
        if dst_rank is None:
            raise_exception_with_message("send_bwd", mb_idx)

        with torch.cuda.stream(self.comm_stream_send):  # type: ignore
            return dist.isend(
                tensor,
                dst=dst_rank,
            )

    def recv_backward(
        self, tensor_size: torch.Size, mb_idx: int
    ) -> Optional[dist.Work]:
        src_rank = self.comm_plan.bwd_recv_from_ranks[mb_idx]
        if src_rank is None:
            raise_exception_with_message("recv_bwd", mb_idx)

        with torch.cuda.stream(self.comm_stream_recv):  # type: ignore
            buffer = torch.empty(
                tensor_size, device="cuda", dtype=self.dtype, requires_grad=True
            )
            work = dist.irecv(
                buffer,
                src=src_rank,
            )
            return work
