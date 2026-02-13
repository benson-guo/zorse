# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
import json
import torch


@dataclass
class PipelineStageConfig:
    stage_id: int
    gpu_ranks: List[int]
    layer_partition: Tuple[int, int]
    microbatch_size: int
    zero_config: Tuple[Tuple[int]]
    num_microbatches_per_rank: List[int]
    num_microbatches: int

    def validate(self) -> List[str]:
        errors = []

        if not self.gpu_ranks:
            errors.append("GPU ranks cannot be empty")
        if len(set(self.gpu_ranks)) != len(self.gpu_ranks):
            errors.append("Duplicate GPU ranks found")

        if len(self.layer_partition) != 2:
            errors.append("Layer partition must be (layer_start, layer_end)")
        if self.layer_partition[0] > self.layer_partition[1]:
            errors.append("Layer partition end must be greater than start")

        for group in self.zero_config:
            if len(set(group)) != len(group):
                errors.append("Duplicate GPU ranks found in ZERO config group")
            if not all(rank in self.gpu_ranks for rank in group):
                errors.append("ZERO config ranks must be subset of GPU ranks")

        if len(self.num_microbatches_per_rank) != len(self.gpu_ranks):
            errors.append(
                "Number of microbatches per rank must match number of GPU ranks"
            )
        if sum(self.num_microbatches_per_rank) != self.num_microbatches:
            errors.append("Total microbatches across ranks must equal num_microbatches")

        return errors


@dataclass
class PipelineConfig:
    model_name: str
    num_pipeline_stages: int
    gpu_groups: Set[Tuple[int]]
    global_batch_size: int
    microbatch_size: int
    sequence_length: int
    vocab_size: int
    fused_optimizer: bool
    pipeline_stages: List[PipelineStageConfig]
    num_microbatches: int
    autocast_dtype: torch.dtype
    reduce_dtype: torch.dtype
    # if split_microbatches_evenly = True, split evenly, other split proportional to GPU compute
    split_microbatches_evenly: bool
    sync_after_recompute: bool
    interleave_stage_sync: bool
    gloo_batch_recv: bool
    alternate_nccl_gloo: bool
    large_model_init: bool

    def validate(self) -> List[str]:
        errors = []

        if not self.model_name or not isinstance(self.model_name, str):
            errors.append("Model name must be a non-empty string")

        if self.num_microbatches != self.global_batch_size // self.microbatch_size:
            errors.append(
                "num_microbatches must equal global_batch_size // microbatch_size"
            )

        if self.pipeline_stages:
            # Check if stages are ordered by layer_partition
            for i in range(len(self.pipeline_stages) - 1):
                curr_stage = self.pipeline_stages[i]
                next_stage = self.pipeline_stages[i + 1]

                if curr_stage.layer_partition[1] != next_stage.layer_partition[0]:
                    errors.append(
                        f"Layer partitions must be continuous: "
                        f"Stage {curr_stage.stage_id} ends at {curr_stage.layer_partition[1]} but "
                        f"Stage {next_stage.stage_id} starts at {next_stage.layer_partition[0]}"
                    )

                if curr_stage.stage_id >= next_stage.stage_id:
                    errors.append(
                        f"Stage IDs must be strictly increasing: "
                        f"Found {curr_stage.stage_id} followed by {next_stage.stage_id}"
                    )

        for i, stage in enumerate(self.pipeline_stages):
            stage_errors = stage.validate()
            if stage_errors:
                errors.extend(
                    f"Stage {stage.stage_id}: {error}" for error in stage_errors
                )

            if stage.stage_id != i:
                errors.append(
                    f"Stage at position {i} has incorrect stage_id {stage.stage_id}"
                )

        return errors


def sort_pipeline_stages(stages: List[Dict]) -> List[Dict]:
    return sorted(stages, key=lambda x: x["layer_partition"][0])


def validate_layer_continuity(sorted_stages: List[Dict]) -> None:
    for i in range(len(sorted_stages) - 1):
        curr_stage = sorted_stages[i]
        next_stage = sorted_stages[i + 1]

        if curr_stage["layer_partition"][1] != next_stage["layer_partition"][0]:
            raise ValueError(
                f"Layer partitions must be continuous: "
                f"Stage ending at {curr_stage['layer_partition'][1]} is followed by "
                f"stage starting at {next_stage['layer_partition'][0]}"
            )


def parse_pipeline_config(config_path) -> PipelineConfig:
    if isinstance(config_path, str):
        with open(config_path) as f:
            config_dict = json.load(f)
    else:
        config_dict = config_path

    num_microbatches = (
        config_dict["global_batch_size"] // config_dict["microbatch_size"]
    )

    sorted_stages = sort_pipeline_stages(config_dict["pipeline_config"])
    validate_layer_continuity(sorted_stages)

    gpu_groups = set()
    pipeline_stages = []
    split_microbatches_evenly = config_dict.get("split_microbatches_evenly", False)
    for stage_id, stage_config in enumerate(sorted_stages):
        gpu_ranks = stage_config["gpu_ranks"]
        gpu_groups.add(tuple(gpu_ranks))
        num_ranks = len(gpu_ranks)
        base_microbatches, extra_microbatches = divmod(num_microbatches, num_ranks)

        if split_microbatches_evenly:
            num_microbatches_per_rank = [base_microbatches] * num_ranks
            for i in range(extra_microbatches):
                num_microbatches_per_rank[i] += 1
        else:
            num_microbatches_per_rank = stage_config["num_microbatches_per_rank"]

        # convert zero_config from list of lists to tuple
        zero_config = tuple(tuple(group) for group in stage_config["zero_config"])
        stage = PipelineStageConfig(
            stage_id=stage_id,
            gpu_ranks=gpu_ranks,
            layer_partition=tuple(stage_config["layer_partition"]),
            microbatch_size=config_dict["microbatch_size"],
            zero_config=zero_config,
            num_microbatches_per_rank=num_microbatches_per_rank,
            num_microbatches=num_microbatches,
        )
        pipeline_stages.append(stage)

    autocast_dtype = getattr(torch, config_dict.get("autocast_dtype", "float32"))
    reduce_dtype = getattr(torch, config_dict.get("reduce_dtype", "float32"))
    sync_after_recompute = not config_dict.get("skip_compute_sync", True)
    interleave_stage_sync = config_dict.get("interleave_stage_sync", False)
    gloo_batch_recv = config_dict.get("gloo_batch_recv", True)
    alternate_nccl_gloo = config_dict.get("alternate_nccl_gloo", True)
    large_model_init = config_dict.get("large_model_init", False)
    vocab_size = config_dict.get("vocab_size", 49152)
    config = PipelineConfig(
        model_name=config_dict["model_name"],
        num_pipeline_stages=len(pipeline_stages),
        gpu_groups=gpu_groups,
        global_batch_size=config_dict["global_batch_size"],
        microbatch_size=config_dict["microbatch_size"],
        sequence_length=config_dict["sequence_length"],
        vocab_size=vocab_size,
        fused_optimizer=config_dict["fused_optimizer"],
        pipeline_stages=pipeline_stages,
        num_microbatches=num_microbatches,
        autocast_dtype=autocast_dtype,
        reduce_dtype=reduce_dtype,
        split_microbatches_evenly=split_microbatches_evenly,
        sync_after_recompute=sync_after_recompute,
        interleave_stage_sync=interleave_stage_sync,
        gloo_batch_recv=gloo_batch_recv,
        alternate_nccl_gloo=alternate_nccl_gloo,
        large_model_init=large_model_init,
    )

    errors = config.validate()
    if errors:
        raise ValueError("Invalid configuration:\n" + "\n".join(errors))

    return config
