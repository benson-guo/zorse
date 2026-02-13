# -*- coding: utf-8 -*-
import os
import torch
import collections
import torch.distributed as dist
from torch.distributed.fsdp._init_utils import (
    _init_intra_and_inter_node_groups,
    _get_default_group,
)
import functools as _functools
import datetime as _datetime
from typing import Optional


def dist_init(timeout=_datetime.timedelta(seconds=300), backend="nccl"):
    rank = get_global_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=timeout)
    gpu_name = get_gpu_name()
    print(f"Rank {rank} local rank {local_rank} world size {world_size} {gpu_name}")

    return world_size


def is_leader():
    """Return True if the current process is the leader."""
    return dist.get_rank() == 0


def is_rank(rank):
    """Return True if the current process is the leader."""
    return dist.get_rank() == rank


def is_local_leader():
    """Return True if the current process is the local leader."""
    return get_local_rank() == 0


def get_local_rank():
    return int(os.environ["LOCAL_RANK"])


def get_gpu_name(local_rank=None):
    local_rank = local_rank if local_rank is not None else get_local_rank()
    gpu_name = torch.cuda.get_device_name(local_rank)
    gpu_name = clean_gpu_name(gpu_name)
    # shorten names
    if gpu_name == "p100-pcie-12gb":
        gpu_name = "p100"
    elif gpu_name == "v100-pcie-16gb":
        gpu_name = "v100-pciex16"
    elif gpu_name == "v100-sxm2-16gb":
        gpu_name = "v100x16"
    elif gpu_name == "v100-sxm2-32gb":
        gpu_name = "v100x32"
    elif gpu_name == "a100-sxm4-40gb":
        gpu_name = "a100x40"
    elif gpu_name == "a100-sxm4-80gb":
        gpu_name = "a100x80"
    # rename
    elif gpu_name == "nvl":
        gpu_name = "h100-nvl"

    return gpu_name


def clean_gpu_name(gpu_name):
    gpu_name = gpu_name.lower().split(" ")[-1]
    if gpu_name == "p100-pcie-12gb":
        gpu_name = "p100"
    elif gpu_name == "v100-pcie-16gb":
        gpu_name = "v100-pciex16"
    elif gpu_name == "v100-sxm2-16gb":
        gpu_name = "v100x16"
    elif gpu_name == "v100-sxm2-32gb":
        gpu_name = "v100x32"
    elif gpu_name == "a100-sxm4-40gb":
        gpu_name = "a100x40"
    elif gpu_name == "a100-sxm4-80gb":
        gpu_name = "a100x80"
    # rename
    elif gpu_name == "nvl":
        gpu_name = "h100-nvl"

    return gpu_name


def get_node_rank():
    return int(os.environ["GROUP_RANK"])


def get_global_rank():
    return int(os.environ["RANK"])


def get_local_world_size():
    gpus_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
    return gpus_per_node


def get_world_size():
    world_size = int(os.environ["WORLD_SIZE"])
    return world_size


def get_num_nodes():
    return int(os.environ["GROUP_WORLD_SIZE"])


def comm_timeout(timeout=1800):
    return _datetime.timedelta(0, float(timeout))


@_functools.lru_cache(maxsize=1)
def _get_gloo_process_group():
    """Returns a gloo process group for torch.distributed"""
    return dist.new_group(backend=dist.Backend.GLOO, timeout=comm_timeout())


@_functools.lru_cache(maxsize=1)
def get_intra_node_process_group():
    assert dist.is_initialized()
    this_node_rank = get_node_rank()
    all_node_ranks = [None] * dist.get_world_size()
    group = _get_gloo_process_group()
    dist.all_gather_object(all_node_ranks, this_node_rank, group=group)

    node_rank2ranks = collections.defaultdict(list)
    for rank, node_rank_ in enumerate(all_node_ranks):
        node_rank2ranks[node_rank_].append(rank)

    timeout = comm_timeout()
    node_rank2pg = {}
    for node_rank_, ranks_group in node_rank2ranks.items():
        node_rank2pg[node_rank_] = dist.new_group(ranks=ranks_group, timeout=timeout)

    return node_rank2pg[this_node_rank]


@_functools.lru_cache(maxsize=1)
def get_inter_node_process_group():
    assert dist.is_initialized()
    this_local_rank = get_local_rank()
    all_local_ranks = [None] * dist.get_world_size()
    group = _get_gloo_process_group()
    dist.all_gather_object(all_local_ranks, this_local_rank, group=group)

    local_rank2ranks = collections.defaultdict(list)
    for rank_, local_rank_ in enumerate(all_local_ranks):
        local_rank2ranks[local_rank_].append(rank_)

    timeout = comm_timeout()
    local_rank2pg = {}
    for local_rank_, ranks_group in local_rank2ranks.items():
        local_rank2pg[local_rank_] = dist.new_group(ranks=ranks_group, timeout=timeout)

    return local_rank2pg[this_local_rank]


def get_default_hybrid_shard_process_groups():
    world_group = _get_default_group()
    intra_pg, inter_pg = _init_intra_and_inter_node_groups(
        world_group, get_local_world_size()
    )
    return intra_pg, inter_pg


# hybrid_shard_groups are the list of list of ranks that shard a model
@_functools.lru_cache(maxsize=1000)
def get_shard_process_group(hybrid_shard_groups=None):
    if hybrid_shard_groups is None:
        if is_leader():
            print("Using default hybrid shard sharding groups")
        return get_default_hybrid_shard_process_groups()[0]
    else:
        if is_leader():
            print(f"Using custom hybrid shard sharding groups: {hybrid_shard_groups}")

    timeout = comm_timeout()
    rank_to_group = {}
    for group in hybrid_shard_groups:
        pg = dist.new_group(ranks=group, timeout=timeout)
        for rank in group:
            rank_to_group[rank] = pg

    global_rank = get_global_rank()
    if global_rank in rank_to_group:
        print(f"rank {global_rank} rank_to_group: {rank_to_group}")
        return rank_to_group[global_rank]
    else:
        # process not part of hybrid_shard_groups
        return None


@_functools.lru_cache(maxsize=1000)
def get_replicate_process_group(
    hybrid_shard_groups=None,
) -> Optional[dist.ProcessGroup]:
    if hybrid_shard_groups is None:
        if is_leader():
            print("Using default hybrid shard replicate groups")
        return get_default_hybrid_shard_process_groups()[1]
    else:
        replicate_groups = []
        for j in range(len(hybrid_shard_groups[0])):
            next_group = []
            for i in range(len(hybrid_shard_groups)):
                next_group.append(hybrid_shard_groups[i][j])
            replicate_groups.append(next_group)
        if is_leader():
            print(f"Using custom hybrid shard replicate groups: {replicate_groups}")

    timeout = comm_timeout()
    rank_to_group = {}
    for group in replicate_groups:
        pg = dist.new_group(ranks=group, timeout=timeout)
        for rank in group:
            rank_to_group[rank] = pg

    global_rank = get_global_rank()
    if global_rank in rank_to_group:
        print(f"rank {global_rank} rank_to_group: {rank_to_group}")
        return rank_to_group[global_rank]
    else:
        # process not part of hybrid_shard_groups
        return None
