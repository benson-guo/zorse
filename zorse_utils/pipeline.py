# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Tuple
import torch
import contextlib
from torch import distributed as dist
import torch.nn as nn
from zorse_utils.gloo_pipeline_comm import PipelineCommunicatorGloo
from zorse_utils.pipeline_config import PipelineConfig, PipelineStageConfig
from zorse_utils.pipeline_comm import PipelineCommunicator, create_static_comm_plans
from models.hub import (
    get_config_for_model,
    get_model_part_for_stage,
    get_layers,
    get_total_model_params,
    wrap_other_layers,
)
from utils.comm import (
    get_global_rank,
    is_leader,
    get_shard_process_group,
    get_replicate_process_group,
    is_local_leader,
)
from utils.logger import get_logger
from torch.distributed import fsdp
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    wrap,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed.fsdp._traversal_utils as traversal_utils


class PipelineStage(nn.Module):
    def __init__(
        self,
        stage_id: int,
        model: nn.Module,
        communicator: PipelineCommunicator,
        stage_config: PipelineStageConfig,
        pipeline_config: PipelineConfig,
        gloo_communicator: PipelineCommunicatorGloo = None,
    ):
        super().__init__()
        self.stage_id = stage_id
        self.model = model
        self.communicator = communicator
        self.num_microbatches_per_rank = stage_config.num_microbatches_per_rank
        self.stage_config = stage_config
        self.pipeline_config = pipeline_config
        self.num_stages = pipeline_config.num_pipeline_stages
        self.gpu_ranks = stage_config.gpu_ranks
        self.gloo_communicator = gloo_communicator
        self.num_gpu_groups = len(self.pipeline_config.gpu_groups)

        # initialize process groups
        # need to initialize pg that this process is not apart of too in same order
        # see: https://pytorch.org/docs/stable/distributed.html#groups
        for sc in pipeline_config.pipeline_stages:
            get_shard_process_group(hybrid_shard_groups=sc.zero_config)
            get_replicate_process_group(hybrid_shard_groups=sc.zero_config)
        shard_groups = self.stage_config.zero_config
        intra_pg = get_shard_process_group(hybrid_shard_groups=shard_groups)
        if len(shard_groups) > 1:
            # if there is more than one shard, we are hybrid sharding
            self.sharding_strategy = ShardingStrategy.HYBRID_SHARD
            inter_pg = get_replicate_process_group(hybrid_shard_groups=shard_groups)
            self.process_group = (intra_pg, inter_pg)
        else:
            self.sharding_strategy = ShardingStrategy.FULL_SHARD
            self.process_group = intra_pg
        print(f"shard groups: {shard_groups} rank: {get_global_rank()}")

        self.compute_stream = torch.cuda.Stream()

    def stage_size(self):
        return len(self.gpu_ranks)

    def next_stage_size(self):
        return len(self.pipeline_config.pipeline_stages[self.stage_id + 1].gpu_ranks)

    def prev_stage_size(self):
        return len(self.pipeline_config.pipeline_stages[self.stage_id - 1].gpu_ranks)

    def is_first_stage(self):
        return self.stage_id == 0

    def is_last_stage(self):
        return self.stage_id == self.num_stages - 1

    def even_stage(self):
        return self.stage_id % 2 == 0

    def backwards_even_stage(self):
        return (self.num_stages - 1 - self.stage_id) % 2 == 0

    def first_gpu_group(self):
        return self.stage_id % self.num_gpu_groups == 0

    def last_gpu_group(self):
        return self.stage_id % self.num_gpu_groups == self.num_gpu_groups - 1

    def wrap_model_zero(self, args):
        if is_leader():
            print("Wrapping model zorse")
        # count total parameters in model
        total_model_params = get_total_model_params(args.model_name, self.model)

        mp_config = fsdp.MixedPrecision(
            param_dtype=self.pipeline_config.autocast_dtype,
            buffer_dtype=self.pipeline_config.autocast_dtype,
            reduce_dtype=self.pipeline_config.reduce_dtype,
        )

        large_model_init = self.pipeline_config.large_model_init
        zero2 = args.zero2_pipeline or args.flashflex_pipeline
        current_device = torch.cuda.current_device()
        wrapper_kwargs = {
            "sharding_strategy": self.sharding_strategy,
            "limit_all_gathers": not zero2,
            "forward_prefetch": True,
            "mixed_precision": mp_config,
            "device_id": current_device,
            "use_orig_params": False,
            "process_group": self.process_group,
            # init on meta device
            "param_init_fn": lambda module: module.to_empty(device=current_device, recurse=False) if large_model_init else None
        }
        with enable_wrap(wrapper_cls=FSDP, **wrapper_kwargs):
            layers = get_layers(self.model)  # get layers to wrap

            # Wrap model layers
            for i in range(len(layers)):
                if is_local_leader():
                    total_layer_params = sum(
                        p.numel() for p in layers[i].parameters() if p.requires_grad
                    )
                    layer_gb = total_layer_params * 4 / 1024**3
                    print(
                        f"Layer {i} parameters: {total_layer_params}, size: {layer_gb}, FS size: {layer_gb / args.world_size} GiB"
                    )
                layers[i] = wrap(layers[i])

            # Wrap remaining model
            wrap_other_layers(args.model_name, self.model)
            self.model = wrap(self.model)

        if args.optimizer_in_backwards:
            fsdp_states, fsdp_modules = traversal_utils._get_fsdp_states_with_modules(
                self.model
            )
            param_set = set()
            for _, (_, fsdp_module) in enumerate(
                zip(reversed(fsdp_states), reversed(fsdp_modules))
            ):
                fsdp_module_params = fsdp_module.parameters()
                optimizer_params = []
                comm_hook_state = {}

                for param in fsdp_module_params:
                    if param in param_set:
                        # Param already added to previous module, skipping
                        continue
                    with torch.no_grad():
                        param_shard_tensor = param.data
                        param_shard = nn.Parameter(data=param_shard_tensor)
                        optimizer_params.append(param_shard)
                    param_set.add(param)

                if len(optimizer_params) == 0:
                    continue

                # attach optimizer to communication hook state
                optimizer = torch.optim.Adam(
                    optimizer_params,
                    fused=self.pipeline_config.fused_optimizer,
                )
                comm_hook_state["optimizer"] = optimizer
                fsdp_module._comm_hook_state = comm_hook_state

        self.model._total_params = total_model_params
        self.model._model_name = args.model_name
        if is_local_leader():
            print(
                f"Total parameters: {total_model_params} size: {total_model_params * 4 / 1024**3} GiB"
            )
            print(self.model)

        return self.model


def find_stages_for_rank(pipeline_config: PipelineConfig) -> List[int]:
    global_rank = get_global_rank()
    stage_ids = []

    for stage_config in pipeline_config.pipeline_stages:
        if global_rank in stage_config.gpu_ranks:
            stage_ids.append(stage_config.stage_id)

    return stage_ids


def get_model_parts_for_stages(
    pipeline_config: PipelineConfig, stage_ids: List[int]
) -> Dict[int, torch.nn.Module]:
    model_parts = {}
    num_stages = pipeline_config.num_pipeline_stages
    large_model_init = pipeline_config.large_model_init

    for stage_id in stage_ids:
        stage_config = pipeline_config.pipeline_stages[stage_id]
        init_on_meta = not is_local_leader() and large_model_init
        init_context = torch.device("meta") if init_on_meta else contextlib.nullcontext()
        with init_context:
            model_part = get_model_part_for_stage(
                pipeline_config.model_name,
                stage_config.layer_partition,
                is_first_stage=stage_id == 0,
                is_last_stage=stage_id == num_stages - 1,
                dtype=pipeline_config.autocast_dtype,
            )
        model_parts[stage_id] = model_part

    return model_parts


def create_communicators_for_stages(
    pipeline_config: PipelineConfig,
    stage_ids: List[int],
    gloo_pg: Optional[dist.ProcessGroup],
) -> Tuple[Dict[int, PipelineCommunicator], Dict[int, PipelineCommunicatorGloo]]:
    global_rank = get_global_rank()
    communicators = {}
    gloo_communicators = {}

    _, hidden_size, _, _ = get_config_for_model(pipeline_config.model_name)
    buffer_size = torch.Size(
        [
            pipeline_config.microbatch_size,
            pipeline_config.sequence_length,
            hidden_size,
        ]
    )

    full_static_comm_plan = create_static_comm_plans(pipeline_config)
    logger = get_logger()
    logger.info(f"Full communication plan: {full_static_comm_plan}")

    for stage_id in stage_ids:
        communicator = PipelineCommunicator(
            stage_id=stage_id,
            num_stages=pipeline_config.num_pipeline_stages,
            process_groups=None,
            curr_rank=global_rank,
            comm_plan=full_static_comm_plan[global_rank].stage_comm_patterns[stage_id],
            dtype=pipeline_config.autocast_dtype,
        )
        communicators[stage_id] = communicator
        if gloo_pg is not None:
            gloo_communicator = PipelineCommunicatorGloo(
                stage_id=stage_id,
                num_stages=pipeline_config.num_pipeline_stages,
                process_group=gloo_pg,
                curr_rank=global_rank,
                comm_plan=full_static_comm_plan[global_rank].stage_comm_patterns[
                    stage_id
                ],
                buffer_size=buffer_size,
                dtype=pipeline_config.autocast_dtype,
            )
            gloo_communicators[stage_id] = gloo_communicator

    return communicators, gloo_communicators


def build_pipeline_stages(
    pipeline_config: PipelineConfig, gloo_pg: Optional[dist.ProcessGroup]
) -> List[PipelineStage]:
    # Find which stages should run on this rank
    global_rank = get_global_rank()
    stage_ids = find_stages_for_rank(pipeline_config)
    logger = get_logger()
    logger.info(f"Building stages {stage_ids} for rank {global_rank}")

    if not stage_ids:
        raise ValueError(f"No stages found for rank {global_rank}")

    model_parts = get_model_parts_for_stages(pipeline_config, stage_ids)
    communicators, gloo_communicators = create_communicators_for_stages(
        pipeline_config, stage_ids, gloo_pg=gloo_pg
    )

    # Build pipeline stages
    pipeline_stages = []
    for stage_id in stage_ids:
        stage_config = pipeline_config.pipeline_stages[stage_id]

        stage = PipelineStage(
            stage_id=stage_id,
            model=model_parts[stage_id],
            communicator=communicators[stage_id],
            stage_config=stage_config,
            pipeline_config=pipeline_config,
            gloo_communicator=gloo_communicators.get(stage_id),
        )
        pipeline_stages.append(stage)

        logger.info(
            f"Built pipeline stage {stage_id} on rank {get_global_rank()} "
            f"with GPU ranks {stage_config.gpu_ranks}"
        )

    # not necessary but just keeping this
    pipeline_stages.sort(key=lambda x: x.stage_id)

    return pipeline_stages
