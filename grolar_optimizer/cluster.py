# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import json
from typing import Dict, List


@dataclass
class GPUStage2:
    identifier: str
    rank: int


@dataclass
class GPUGroupStage2:
    all_gather_latency: float
    reduce_scatter_latency: float
    group_id: int
    gpus: List[GPUStage2]
    send_recv_latency_next: float = 20
    send_recv_latency_prev: float = 20
    machines: List[str] = field(default_factory=list)

    def get_num_gpus(self) -> int:
        return len(self.gpus)


@dataclass
class ClusterConfigStage2:
    groups: List[GPUGroupStage2]

    @classmethod
    def from_json(cls, json_file: str):
        with open(json_file, "r") as f:
            data = json.load(f)

        groups = []
        for id, group_data in enumerate(data["groups"]):
            all_gather_latency = group_data["all_gather_latency"]
            reduce_scatter_latency = group_data["reduce_scatter_latency"]
            gpus = []
            for gpu_data in group_data["gpus"]:
                gpu = GPUStage2(
                    identifier=gpu_data["identifier"], rank=gpu_data["rank"]
                )
                gpus.append(gpu)
            group = GPUGroupStage2(
                group_id=id,
                gpus=gpus,
                all_gather_latency=all_gather_latency,
                reduce_scatter_latency=reduce_scatter_latency,
            )
            groups.append(group)

        return cls(groups=groups)


@dataclass
class ModelConfig:
    name: str
    global_batch_size: int
    sequence_length: int
    microbatch_size: int
    hidden_size: int
    num_layers: int
    params_per_layer: int


@dataclass
class StageConfig:
    gpu_group: GPUGroupStage2
    layer_start: int
    layer_end: int
    num_microbatches_per_rank: List[int]  # num microbatches per rank

    def get_num_layers(self) -> int:
        return self.layer_end - self.layer_start
