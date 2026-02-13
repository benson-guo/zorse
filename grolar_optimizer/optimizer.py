# -*- coding: utf-8 -*-
import argparse
from dataclasses import dataclass
import json
from typing import List, Dict, Tuple, Optional, NamedTuple

from grolar_optimizer.cluster import (
    ClusterConfigStage2 as ClusterConfig,
    ModelConfig,
    GPUGroupStage2 as GPUGroup,
)
from grolar_optimizer.utils import validate_memory_for_stage_config


@dataclass
class StagePartitionConfig:
    gpu_ranks: List[int]
    num_microbatches_per_rank: List[int]
    layer_start: int
    layer_end: int
    zero_config: List[List[int]]
    group: GPUGroup


class ModelLatencies(NamedTuple):
    forward: float
    backward: float
    total: float


class PipelineOptimizer:
    def __init__(
        self,
        cluster_config: ClusterConfig,
        model_config: ModelConfig,
        model_latencies: Dict[str, ModelLatencies],
        stage_strategy: str,
        dtype_multiplier: float,
        optimizer_multiplier: float,
        autocast_dtype: str,
        reduce_dtype: str,
        reserved_memory_gb: float,
        fix_layer_distribution: Optional[List[int]] = None,
    ):
        self.cluster_config = cluster_config
        self.model_config = model_config
        self.model_latencies = model_latencies
        self.num_microbatches = (
            model_config.global_batch_size // model_config.microbatch_size
        )
        self.stage_strategy = stage_strategy

        # autocast dtype and reduce dtype
        self.autocast_dtype = autocast_dtype
        self.reduce_dtype = reduce_dtype

        # Memory validation parameters
        self.dtype_multiplier = dtype_multiplier
        self.optimizer_multiplier = optimizer_multiplier
        self.reserved_memory_gb = reserved_memory_gb
        self.fix_layer_distribution = (
            [int(layer) for layer in fix_layer_distribution.split(",")]
            if fix_layer_distribution is not None
            else None
        )

    def calculate_raw_compute_ratios(self) -> Dict[int, List[float]]:
        group_compute_ratios = {}

        for group in self.cluster_config.groups:
            group_latencies = []
            for gpu in group.gpus:
                assert gpu.identifier in self.model_latencies
                latency = self.model_latencies[gpu.identifier].total
                group_latencies.append(1.0 / latency)

            total_compute = sum(group_latencies)
            group_compute_ratios[group.group_id] = [
                latency / total_compute for latency in group_latencies
            ]

        return group_compute_ratios

    def distribute_microbatches_within_group(self) -> Dict[int, List[int]]:
        compute_ratios = self.calculate_raw_compute_ratios()
        group_microbatches = {}

        for group_id, ratios in compute_ratios.items():
            num_mbs_per_rank = [self.num_microbatches * ratio for ratio in ratios]

            floor = [int(x) for x in num_mbs_per_rank]
            remaining = self.num_microbatches - sum(floor)

            indices = list(range(len(ratios)))
            indices.sort(key=lambda i: ratios[i], reverse=True)

            final_distribution = floor.copy()
            for i in range(remaining):
                final_distribution[indices[i % len(indices)]] += 1

            group_microbatches[group_id] = final_distribution

        return group_microbatches

    def calculate_group_layer_times(self) -> Dict[int, float]:
        group_mb_distribution = self.distribute_microbatches_within_group()
        group_layer_times = {}

        for group in self.cluster_config.groups:
            group_id = group.group_id
            microbatches_per_gpu = group_mb_distribution[group_id]

            max_layer_time = 0
            for gpu, num_mb in zip(group.gpus, microbatches_per_gpu):
                latency = self.model_latencies[gpu.identifier].total
                process_time = latency * num_mb
                max_layer_time = max(max_layer_time, process_time)

            group_layer_times[group_id] = max_layer_time

        return group_layer_times

    def calculate_cross_group_compute_ratios(self) -> List[float]:
        group_layer_times = self.calculate_group_layer_times()
        print(f"Group Layer Times: {group_layer_times}")

        group_compute_ratio = [
            1.0 / group_layer_times[group.group_id]
            for group in self.cluster_config.groups
        ]

        total_compute = sum(group_compute_ratio)
        compute_ratios = [cap / total_compute for cap in group_compute_ratio]
        print(f"Cross Group Compute Ratios: {compute_ratios}")

        return compute_ratios

    def distribute_layers(self) -> List[Tuple[int, int, int]]:
        if self.fix_layer_distribution is not None:
            final_distribution = self.fix_layer_distribution
        else:
            compute_ratios = self.calculate_cross_group_compute_ratios()
            total_layers = self.model_config.num_layers

            layer_splits = [total_layers * ratio for ratio in compute_ratios]

            floor = [int(x) for x in layer_splits]
            remaining = total_layers - sum(floor)

            indices = list(range(len(compute_ratios)))
            indices.sort(key=lambda i: compute_ratios[i], reverse=True)

            final_distribution = floor.copy()
            for i in range(remaining):
                final_distribution[indices[i % len(indices)]] += 1

        layer_assignments = []
        current_layer = 0
        for group_idx, num_layers in enumerate(final_distribution):
            if num_layers > 0:
                layer_assignments.append(
                    (
                        self.cluster_config.groups[group_idx].group_id,
                        current_layer,
                        current_layer + num_layers,
                    )
                )
            current_layer += num_layers

        return layer_assignments

    def generate_stage_configs(self) -> List[StagePartitionConfig]:
        group_mb_distribution = self.distribute_microbatches_within_group()
        layer_assignments = self.distribute_layers()

        stage_configs = []

        for group_id, layer_start, layer_end in layer_assignments:
            group = next(
                g for g in self.cluster_config.groups if g.group_id == group_id
            )
            gpu_ranks = [gpu.rank for gpu in group.gpus]
            num_microbatches = group_mb_distribution[group_id]

            zero_config = [gpu_ranks]

            stage_config = StagePartitionConfig(
                gpu_ranks=gpu_ranks,
                num_microbatches_per_rank=num_microbatches,
                layer_start=layer_start,
                layer_end=layer_end,
                zero_config=zero_config,
                group=group,
            )

            stage_configs.append(stage_config)

        return stage_configs

    def interleave_stage_configs(
        self,
        stage_configs: List[StagePartitionConfig],
        interleave_degree: int,
    ) -> Optional[List[StagePartitionConfig]]:
        stage_configs.sort(key=lambda x: x.layer_start)

        stage_total_layers = [
            config.layer_end - config.layer_start for config in stage_configs
        ]

        # Calculate base chunks and track extra layers
        stage_chunks = []
        total_extra = 0
        for stage_idx in range(len(stage_configs)):
            stage_layers = stage_total_layers[stage_idx]
            base_chunk, extra = divmod(stage_layers, interleave_degree)
            total_extra += extra
            print(
                f"Stage {stage_idx} has {stage_layers} layers, base chunk {base_chunk}, extra {extra}"
            )
            # each stage has at least one layer
            if base_chunk == 0:
                return None

            # Initialize chunks with base size
            chunks = [base_chunk] * interleave_degree
            stage_chunks.append(chunks)

        # Sort stages by their compute ratio in descending order
        stage_indices = list(range(len(stage_configs)))
        compute_ratios = self.calculate_cross_group_compute_ratios()
        # sort stage by smallest relative latency after increasing chunk size
        stage_indices.sort(key=lambda i: (stage_chunks[i][0] + 1) / compute_ratios[i])
        # total_layers = stage_configs[-1].layer_end
        # original_ratios = [layers / total_layers for layers in stage_total_layers]
        # stage_indices.sort(key=lambda i: original_ratios[i], reverse=True)
        # Distribute extra layers to chunks of stages with higher ratios
        while total_extra > 0:
            print("Before distribution:")
            for stage_idx in stage_indices:
                print(
                    f"Stage {stage_idx} has ratio {(stage_chunks[stage_idx][0]) / compute_ratios[stage_idx]}"
                )

            for stage_idx in stage_indices:
                if total_extra <= 0:
                    break

                # Try to distribute extra layer across chunks evenly
                for chunk_idx in range(interleave_degree):
                    if total_extra <= 0:
                        break
                    stage_chunks[stage_idx][chunk_idx] += 1
                    total_extra -= 1

            print("After distribution:")
            for stage_idx in stage_indices:
                print(
                    f"Stage {stage_idx} has ratio {(stage_chunks[stage_idx][0]) / compute_ratios[stage_idx]}"
                )

        interleaved_configs = []
        curr_layer = 0

        for chunk_idx in range(interleave_degree):
            for stage_idx, stage_config in enumerate(stage_configs):
                chunk_size = stage_chunks[stage_idx][chunk_idx]
                if chunk_size > 0:
                    new_config = StagePartitionConfig(
                        gpu_ranks=stage_config.gpu_ranks,
                        num_microbatches_per_rank=stage_config.num_microbatches_per_rank,
                        layer_start=curr_layer,
                        layer_end=curr_layer + chunk_size,
                        zero_config=stage_config.zero_config,
                        group=stage_config.group,
                    )
                    interleaved_configs.append(new_config)
                    curr_layer += chunk_size

        return interleaved_configs

    def calculate_pipeline_latency(self, stage_configs, interleave_degree) -> float:
        def get_gpu_ids(gpu_ranks):
            for group in self.cluster_config.groups:
                group_gpus = [g.rank for g in group.gpus]
                if set(gpu_ranks) == set(group_gpus):
                    return [g.identifier for g in group.gpus]
            raise Exception(f"Couldn't find GPUs for ranks: {gpu_ranks}")

        # Aggregate all stages for each GPU group
        stages_by_group = {}
        for s_c in stage_configs:
            g_id = s_c.group.group_id
            if g_id not in stages_by_group:
                stages_by_group[g_id] = []
            stages_by_group[g_id].append(s_c)

        num_groups = len(self.cluster_config.groups)
        total_pipeline_latency = 0

        # pipeline startup cost
        # 1 microbatch + send for the first n - 1 GPU groups
        total_startup_cost = 0
        for i, stage in enumerate(stage_configs):
            ag = stage.group.all_gather_latency
            send_recv_forward = stage.group.send_recv_latency_next  # SEND TO NEXT GROUP
            group_gpu_ids = get_gpu_ids(stage.gpu_ranks)
            num_layers = stage.layer_end - stage.layer_start

            max_forward_compute = max(
                [self.model_latencies[gpu].forward for gpu in group_gpu_ids]
            )
            # for first GPU group
            if i == 0:
                max_forward_compute = max(max_forward_compute, ag)
            # otherwise AG is overlapped
            startup_cost = max_forward_compute * num_layers
            # for all groups
            if i == num_groups:
                total_startup_cost += startup_cost
                break

            # isend cannot be overlapped for the first n-1 groups
            startup_cost += send_recv_forward
            total_startup_cost += startup_cost

        core_pipeline_latency = 0
        # pipeline for each group (n-1 microbatches for first stages)
        # n microbatches for the subsequent stages
        for _, stages in stages_by_group.items():
            sum_of_set_of_stages = 0
            for i, stage in enumerate(stages):
                ag = stage.group.all_gather_latency
                rs = stage.group.reduce_scatter_latency
                # Whatever is slower
                send_recv = max(
                    stage.group.send_recv_latency_next,
                    stage.group.send_recv_latency_prev,
                )
                group_gpu_ids = get_gpu_ids(stage.gpu_ranks)
                num_layers = stage.layer_end - stage.layer_start
                if i == 0:
                    max_forward_compute = max(
                        [
                            max(
                                self.model_latencies[gpu].forward * num_layers,
                                send_recv,
                            )
                            * (num_mbs - 1)
                            for gpu, num_mbs in zip(
                                group_gpu_ids, stage.num_microbatches_per_rank
                            )
                        ]
                    )
                else:
                    max_forward_compute = max(
                        [
                            max(
                                self.model_latencies[gpu].forward * num_layers,
                                send_recv,
                            )
                            * num_mbs
                            for gpu, num_mbs in zip(
                                group_gpu_ids, stage.num_microbatches_per_rank
                            )
                        ]
                    )

                # forwards for this stage
                forward_stage = max(max_forward_compute, ag * num_layers)

                # comm for backwards
                comm_latency = ag + rs

                # backwards for this stage
                max_backward_compute = max(
                    [
                        max(
                            (
                                self.model_latencies[gpu].backward
                                + self.model_latencies[gpu].forward
                            )
                            * num_layers,
                            send_recv,
                        )
                        * num_mbs
                        for gpu, num_mbs in zip(
                            group_gpu_ids, stage.num_microbatches_per_rank
                        )
                    ]
                )
                backward_stage = max(max_backward_compute, comm_latency * num_layers)
                sum_of_set_of_stages += forward_stage + backward_stage

            core_pipeline_latency = max(sum_of_set_of_stages, core_pipeline_latency)

        total_end_cost = 0
        # pipeline end cost
        for i, stage in enumerate(reversed(stage_configs)):
            rs = stage.group.reduce_scatter_latency
            send_recv_backward = (
                stage.group.send_recv_latency_prev
            )  # RECV FROM PREVIOUS GROUP
            group_gpu_ids = get_gpu_ids(stage.gpu_ranks)

            num_layers = stage.layer_end - stage.layer_start
            max_backward_compute = max(
                [self.model_latencies[gpu].total for gpu in group_gpu_ids]
            )
            # for the first stage (last backwards)
            if i == 0:
                max_backward_compute = max(max_backward_compute, rs)
            # otherwise AG is overlapped
            end_cost = max_backward_compute * num_layers

            if i == num_groups:
                total_end_cost += end_cost
                break

            # TODO: ADD LATER
            # end_cost += optimizer_step
            end_cost += send_recv_backward
            total_end_cost += end_cost

        total_pipeline_latency = (
            total_startup_cost + core_pipeline_latency + total_end_cost
        )

        return total_pipeline_latency

    def validate_memory_requirements(
        self, stage_configs: List[StagePartitionConfig]
    ) -> bool:
        validation_results = validate_memory_for_stage_config(
            stages_config=stage_configs,
            model_config=self.model_config,
            dtype_multiplier=self.dtype_multiplier,
            optimizer_multiplier=self.optimizer_multiplier,
            microbatch_size=self.model_config.microbatch_size,
            reserved_memory_gb=self.reserved_memory_gb,
        )
        return all(validation_results.values())

    def optimize(
        self, interleave_degree: Optional[int] = None
    ) -> tuple[List[StagePartitionConfig], int, float, bool]:
        """
        1. Generate initial stage configs
        2. Validate memory requirements
        3. Find optimal interleaving if memory requirements are met
        """
        # Generate initial stage configs
        stage_configs = self.generate_stage_configs()

        # Validate memory requirements for initial configuration
        is_valid_memory = self.validate_memory_requirements(stage_configs)

        if not is_valid_memory:
            print("Warning: Initial stage configuration exceeds memory limits")

        total_layers = self.model_config.num_layers
        # try all interleaving
        max_interleave = min(total_layers // 2, len(stage_configs)) + 2

        min_latency = float("inf")
        optimal_configs = stage_configs
        optimal_degree = 1
        interleave_range = (
            [interleave_degree]
            if interleave_degree is not None
            else range(1, max_interleave + 1)
        )

        for degree in interleave_range:
            print(f" >>>> Optimizing with interleave degree: {degree}")
            interleaved_configs = self.interleave_stage_configs(
                stage_configs=stage_configs.copy(),
                interleave_degree=degree,
            )

            if interleaved_configs is None:
                print(f"Interleave Degree: {degree}, No valid solution found")
                continue

            interleaved_valid_memory = self.validate_memory_requirements(
                interleaved_configs
            )
            if not interleaved_valid_memory and interleave_degree is None:
                print(f"Interleave Degree: {degree}, Memory validation failed")
                continue

            latency = self.calculate_pipeline_latency(interleaved_configs, degree)

            print(f"Interleave Degree: {degree}, Optimal Latency: {latency}")
            print(
                f"\tLayer splits: {[config.layer_end - config.layer_start for config in interleaved_configs]}"
            )
            print("=========================")

            if latency < min_latency:
                min_latency = latency
                optimal_configs = interleaved_configs
                optimal_degree = degree

        return optimal_configs, optimal_degree, min_latency, is_valid_memory

    def export_config(self, output_file: str, args: argparse.Namespace) -> None:
        optimal_configs, optimal_degree, min_latency, is_valid_memory = self.optimize(
            args.fix_interleave_degree
        )

        if not is_valid_memory:
            print("Warning: The generated configuration may exceed memory limits")

        if args.fix_interleave_degree is not None:
            print(
                f"Calculated Pipeline Latency is : {min_latency} with degree {optimal_degree}"
            )
        else:
            print(
                f"Calculated Pipeline Latency is : {min_latency} with optimal degree {optimal_degree}"
            )

        config = {
            "model_name": args.model_name,
            "global_batch_size": args.global_batch_size,
            "microbatch_size": args.microbatch_size,
            "sequence_length": args.sequence_length,
            "vocab_size": args.vocab_size,
            "fused_optimizer": not args.disable_fused_optimizer,
            "autocast_dtype": args.autocast_dtype,
            "reduce_dtype": args.reduce_dtype,
            "pipeline_config": [
                {
                    "gpu_ranks": config.gpu_ranks,
                    "num_microbatches_per_rank": config.num_microbatches_per_rank,
                    "layer_partition": [config.layer_start, config.layer_end],
                    "zero_config": config.zero_config,
                }
                for config in optimal_configs
            ],
        }

        with open(output_file, "w") as f:
            json.dump(config, f, indent=4)
        print(f"Pipeline config saved to {output_file}")
