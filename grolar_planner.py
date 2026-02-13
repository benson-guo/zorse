# -*- coding: utf-8 -*-
import argparse
import json
from typing import Dict
import numpy as np
import pymetis
import time
import networkx as nx
from networkx.algorithms.connectivity import stoer_wagner
from sklearn.cluster import SpectralClustering

from comm_model import (
    estimate_collective_latency_v2,
    get_bandwidth_matrix,
    get_tensor_size_gb,
)
from grolar_optimizer.cluster import (
    ClusterConfigStage2,
    GPUStage2,
    GPUGroupStage2,
    ModelConfig,
)
from grolar_optimizer.optimizer import PipelineOptimizer
from grolar_optimizer.profiler import profile_agrs_partition
from grolar_optimizer.visualize import visualize_graph
from models.hub import get_model_stats
from utils.comm import clean_gpu_name
from utils.optimizer_utils import load_model_compute_latencies_per_gpu

MODEL_NAME = "deepspeedllama_3b"
AUTOCAST_DTYPE = "float16"
GLOBAL_BATCH_SIZE = 128
SEQUENCE_LENGTH = 512

# this is for graph construction only
MODEL_LATENCIES_PER_GPU = None


def load_json_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def compute_similarity(gpu1, gpu2):
    gpu1_name = clean_gpu_name(gpu1["name"])
    gpu2_name = clean_gpu_name(gpu2["name"])

    assert (
        gpu1_name in MODEL_LATENCIES_PER_GPU
    ), f"GPU {gpu1_name} not found in model latencies"
    assert (
        gpu2_name in MODEL_LATENCIES_PER_GPU
    ), f"GPU {gpu2_name} not found in model latencies"

    gpu1_latency = MODEL_LATENCIES_PER_GPU[gpu1_name].total
    gpu2_latency = MODEL_LATENCIES_PER_GPU[gpu2_name].total

    ratio = min(gpu1_latency, gpu2_latency) / max(gpu1_latency, gpu2_latency)

    return ratio**0.5


def memory_similarity(gpu1, gpu2, min_mem, max_mem):
    if max_mem == min_mem:
        return 1.0

    mem_diff = abs(gpu1["memory_total_mib"] - gpu2["memory_total_mib"])
    normalized_diff = mem_diff / (max_mem - min_mem)

    return 1.0 - normalized_diff


def collect_all_bandwidths(data):
    bandwidths = []

    intranode = data["bandwidth"]["intranode_bandwidth"]
    bandwidths.extend(intranode.values())

    internode = data["bandwidth"]["internode_bandwidth"]
    for source in internode.values():
        bandwidths.extend(source.values())

    return min(bandwidths), max(bandwidths)


def bandwidth_weight(gpu1, gpu2, data, bandwidth_range=None, normalize=False):
    machine1 = gpu1["machine"]
    machine2 = gpu2["machine"]

    machine1_idx = get_machine_index(data, machine1)
    machine2_idx = get_machine_index(data, machine2)

    # Get raw bandwidth value
    if machine1 == machine2:
        raw_bandwidth = data["bandwidth"]["intranode_bandwidth"][str(machine1_idx)]
    else:
        raw_bandwidth = data["bandwidth"]["internode_bandwidth"][str(machine1_idx)][
            str(machine2_idx)
        ]

    if normalize:
        min_bw, max_bw = collect_all_bandwidths(data)

        if min_bw == max_bw:
            return 1.0 if machine1 == machine2 else 0.1

        # min-max normalization
        normalized_bw = (raw_bandwidth - min_bw) / (max_bw - min_bw)
        return int(max(1, normalized_bw * 100))
    else:
        return int(max(raw_bandwidth, 1))


def get_machine_index(data, machine_name):
    for machine in data["machines"]:
        if machine["machine"] == machine_name:
            return machine["machine_index"]


def build_gpu_graph(data):
    G = nx.Graph()

    for gpu in data["gpu_list"]:
        gpu_data = {
            "id": gpu["id"],
            "name": clean_gpu_name(gpu["name"]),
            "memory_total_mib": gpu["memory_total_mib"],
            "machine": gpu["machine"],
        }
        G.add_node(gpu_data["id"], **gpu_data)

    # memories = [gpu["memory_total_mib"] for gpu in data["gpu_list"]]
    # min_mem = min(memories)
    # max_mem = max(memories)

    for i, gpu1 in enumerate(data["gpu_list"]):
        for j, gpu2 in enumerate(data["gpu_list"]):
            if i < j:
                # compute_sim = compute_similarity(gpu1, gpu2)
                # mem_sim = memory_similarity(gpu1, gpu2, min_mem, max_mem)
                bw = bandwidth_weight(gpu1, gpu2, data)
                # edge_weight = 0.1 * compute_sim + 0.2 * mem_sim + 0.7 * bw
                edge_weight = bw
                G.add_edge(gpu1["id"], gpu2["id"], weight=edge_weight)

    visualize_graph(G, data)
    return G


def partition_graph_with_pymetis(G, num_partitions):
    """
    TODO: Improve this function
    I didn't get the partitioning I was expecting always
    """
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Convert to CSR format (Compressed Sparse Row) for PyMetis
    xadj = [0]
    adjncy = []
    eweights = []

    for node in nodes:
        neighbors = list(G.neighbors(node))
        neighbor_indices = [node_to_idx[neighbor] for neighbor in neighbors]

        # Add neighbors to adjncy array
        adjncy.extend(neighbor_indices)

        # Add edge weights
        for neighbor in neighbors:
            # Convert to integer weight
            weight = int(G[node][neighbor]["weight"])
            eweights.append(weight)

        # Update xadj with current end position
        xadj.append(xadj[-1] + len(neighbors))

    # Run PyMetis k-way partitioning
    # n_cuts is the number of edges cut, parts is the partition assignment for each node
    n_cuts, parts = pymetis.part_graph(
        num_partitions, xadj=xadj, adjncy=adjncy, eweights=eweights
    )

    print(f"PyMetis partitioning complete. Cut edges: {n_cuts}")

    # Create a mapping from node ID to partition ID
    partition_map = {nodes[i]: part for i, part in enumerate(parts)}

    return partition_map, n_cuts


# Calculate average bandwidth between two groups
def calculate_avg_bandwidth(group1, group2, cluster_data):
    total_bw = 0.0
    connection_count = 0

    for machine1 in group1.machines:
        for machine2 in group2.machines:
            if machine1 == machine2:
                # Use intranode bandwidth if in same machine
                machine1_idx = get_machine_index(cluster_data, machine1)
                bw = cluster_data["bandwidth"]["intranode_bandwidth"][str(machine1_idx)]
            else:
                # Use internode bandwidth
                machine1_idx = get_machine_index(cluster_data, machine1)
                machine2_idx = get_machine_index(cluster_data, machine2)
                bw = cluster_data["bandwidth"]["internode_bandwidth"][
                    str(machine1_idx)
                ][str(machine2_idx)]

            total_bw += bw
            connection_count += 1

    # Return average bandwidth
    if connection_count > 0:
        return total_bw / connection_count
    return 0


# Calculate average latency between two groups
def calculate_avg_latency(group1, group2, cluster_data):
    total_latency = 0.0
    connection_count = 0

    for machine1 in group1.machines:
        for machine2 in group2.machines:
            if machine1 == machine2:
                # Use intranode latency if in same machine
                machine1_idx = get_machine_index(cluster_data, machine1)
                latency = cluster_data["bandwidth"]["intranode_latency"][
                    str(machine1_idx)
                ]
            else:
                # Use internode latency
                machine1_idx = get_machine_index(cluster_data, machine1)
                machine2_idx = get_machine_index(cluster_data, machine2)
                latency = cluster_data["bandwidth"]["internode_latency"][
                    str(machine1_idx)
                ][str(machine2_idx)]

            total_latency += latency
            connection_count += 1

    # Return average latency
    if connection_count > 0:
        return total_latency / connection_count
    return 0


def estimate_send_recv_latency(
    group1,
    group2,
    cluster_data,
    model_stats,
    microbatch_size,
    sequence_length,
    autocast_dtype,
):
    """
    Estimates send/receive latency using interpolation based on message size.
    Returns latency in milliseconds.
    """
    # Calculate activation tensor size in bytes
    hidden_size = model_stats["hidden_size"]
    bytes_per_element = 2 if autocast_dtype == "float16" else 4

    tensor_elements = microbatch_size * sequence_length * hidden_size
    tensor_size_bytes = tensor_elements * bytes_per_element
    tensor_size_gb = tensor_size_bytes / (1024 * 1024 * 1024)

    # Message size in bytes
    msg_size = tensor_size_bytes

    # Reference point: latency at 10^7 bytes (10MB) message size
    reference_msg_size = 10**7  # 10MB

    # Find minimum bandwidth between the two groups
    min_bandwidth = float("inf")
    connection_count = 0

    # Track if we're using intranode or internode communication
    is_intranode = False
    machine1_idx = None
    machine2_idx = None

    for machine1 in group1.machines:
        for machine2 in group2.machines:
            machine1_idx = get_machine_index(cluster_data, machine1)
            machine2_idx = get_machine_index(cluster_data, machine2)

            if machine1 == machine2:
                bw = cluster_data["bandwidth"]["intranode_bandwidth"][str(machine1_idx)]
                is_intranode = True
            else:
                bw = cluster_data["bandwidth"]["internode_bandwidth"][
                    str(machine1_idx)
                ][str(machine2_idx)]

            min_bandwidth = min(min_bandwidth, bw)
            connection_count += 1

    min_bandwidth = max(min_bandwidth, 0.0001)
    if connection_count == 0 or min_bandwidth == float("inf"):
        return 0.0001

    # Get reference latency at 10MB message size
    if is_intranode:
        # Using the last intranode connection found
        reference_latency = cluster_data["bandwidth"]["intranode_latency"][
            str(machine1_idx)
        ]
    else:
        # Using the last internode connection found
        reference_latency = cluster_data["bandwidth"]["internode_latency"][
            str(machine1_idx)
        ][str(machine2_idx)]

    # Simple linear interpolation based on message size
    # For small messages: mostly dominated by latency
    # For large messages: mostly dominated by bandwidth

    # Latency factor: how much the latency scales with message size
    # - For small messages (smaller than reference): latency doesn't change much
    # - For large messages: latency scales with message size relative to bandwidth

    if msg_size <= reference_msg_size:
        # For small messages, latency is close to reference latency
        # Small scaling factor for tiny messages
        scale_factor = 0.8 + 0.2 * (msg_size / reference_msg_size)
        latency_ms = reference_latency * scale_factor
    else:
        # For large messages, add bandwidth component
        # First, calculate the additional time due to bandwidth
        bandwidth_latency_ms = (
            ((msg_size - reference_msg_size) / (1024 * 1024 * 1024))
            * 1000
            / min_bandwidth
        )

        # Then add to the reference latency
        latency_ms = reference_latency + bandwidth_latency_ms

    print(
        f"  Activation tensor: [{microbatch_size}, {sequence_length}, {hidden_size}] = {tensor_size_gb:.4f} GB"
    )
    print(
        f"  Message size: {msg_size/(1024*1024):.2f} MB, Reference latency: {reference_latency:.6f} ms"
    )
    print(f"  Interpolated latency: {latency_ms:.6f} ms")

    return latency_ms


def run_stage2_optimizer_on_partition(*, partitions: Dict, cluster_data: Dict, args):
    print("\nRunning Stage 2 optimizer on each partition...")
    # We will create the GPU groups here
    gpu_groups = []
    for part_id, nodes in partitions.items():
        gpu_this_part = []
        gpu_ranks_this_part = []
        for node in nodes:
            node_idx = node.split("_")[-1]
            local_rank = int(cluster_data["gpu_list"][int(node_idx)]["index"])
            global_rank = int(cluster_data["gpu_list"][int(node_idx)]["global_rank"])
            gpu_name = clean_gpu_name(cluster_data["gpu_list"][int(node_idx)]["name"])
            gpu_this_part.append(GPUStage2(identifier=gpu_name, rank=global_rank))
            gpu_ranks_this_part.append((local_rank, global_rank))
        print(f"Partition {part_id}: {len(nodes)} GPUs - {gpu_this_part}")

        machines_in_group = set()
        for node in nodes:
            node_idx = int(node.split("_")[-1])
            machine = cluster_data["gpu_list"][node_idx]["machine"]
            machines_in_group.add(machine)

        part_info = {
            "group_id": part_id,
            "gpus": gpu_ranks_this_part,
            "machines": list(machines_in_group),
        }

        if args.use_agrs_comm_model:
            # Use communication model instead of profiling
            all_gather_tensor_size = get_tensor_size_gb(
                args.model_name, args.autocast_dtype
            )
            reduce_scatter_tensor_size = get_tensor_size_gb(
                args.model_name, args.reduce_dtype
            )
            global_ranks = [rank[1] for rank in gpu_ranks_this_part]
            bandwidth_matrix = get_bandwidth_matrix(
                args.cluster_info_file, global_ranks
            )
            all_gather_latency = (
                estimate_collective_latency_v2(
                    bandwidth_matrix,
                    all_gather_tensor_size,
                    "all_gather",
                    len(global_ranks),
                )
                / 1000
            )  # Convert from ms to seconds
            reduce_scatter_latency = (
                estimate_collective_latency_v2(
                    bandwidth_matrix,
                    reduce_scatter_tensor_size,
                    "reduce_scatter",
                    len(global_ranks),
                )
                / 1000
            )  # Convert from ms to seconds

            print(f"Estimated All-Gather latency: {all_gather_latency:.6f}s")
            print(f"Estimated Reduce-Scatter latency: {reduce_scatter_latency:.6f}s")

            results = {
                "allgather": {"avg_time": all_gather_latency},
                "reducescatter": {"avg_time": reduce_scatter_latency},
            }
        else:
            # Use actual profiling
            results = profile_agrs_partition(
                group_info=part_info,
                cluster_info=cluster_data,
                model_name=args.model_name,
                autocast_dtype=args.autocast_dtype,
                reduce_dtype=args.reduce_dtype,
                num_trials=args.num_trials,
                num_warmup=args.num_warmup,
                nccl_ib_disable=args.nccl_ib_disable,
                machine_file=args.machine_file,
            )
            print(f"Finished profiling partition {part_id}")

        gpu_groups.append(
            GPUGroupStage2(
                all_gather_latency=results["allgather"]["avg_time"],
                reduce_scatter_latency=results["reducescatter"]["avg_time"],
                group_id=part_id,
                gpus=gpu_this_part,
                machines=list(machines_in_group),
                send_recv_latency_prev=0,  # Updated later
                send_recv_latency_next=0,  # Updated later
            )
        )

    # Sort GPU groups by all_gather_latency (fastest first)
    gpu_groups.sort(key=lambda x: x.all_gather_latency)

    # Get the first group (fastest all_gather)
    ordered_groups = [gpu_groups[0]]
    remaining_groups = gpu_groups[1:]

    # Get model stats for activation size calculation
    model_stats = get_model_stats(args.model_name)

    # Order groups by interconnect speed and calculate send/recv latencies
    while remaining_groups:
        current_group = ordered_groups[-1]
        best_next_group = None
        best_bandwidth = -1

        for group in remaining_groups:
            avg_bw = calculate_avg_bandwidth(current_group, group, cluster_data)
            if avg_bw > best_bandwidth:
                best_bandwidth = avg_bw
                best_next_group = group

        if best_next_group:
            # For initial ordering, use a default microbatch_size of 1
            # Actual latencies will be recalculated for each specific microbatch size
            send_recv_latency = estimate_send_recv_latency(
                current_group,
                best_next_group,
                cluster_data,
                model_stats,
                1,  # Default microbatch size for initial ordering
                args.sequence_length,
                args.autocast_dtype,
            )

            print(
                f"=====> Initial Send/Recv latency between {current_group.group_id} and {best_next_group.group_id}: {send_recv_latency:.6f}s"
            )

            current_group.send_recv_latency_next = send_recv_latency
            best_next_group.send_recv_latency_prev = send_recv_latency

            ordered_groups.append(best_next_group)
            remaining_groups.remove(best_next_group)
            gpu_groups = ordered_groups

    print("\nGPU Groups Ordered by Interconnect Speed:")
    for i, group in enumerate(gpu_groups):
        print(
            f"Position {i}: Group {group.group_id} - All Gather Latency: {group.all_gather_latency:.6f}s"
        )
        if i > 0:
            avg_bw = calculate_avg_bandwidth(gpu_groups[i - 1], group, cluster_data)
            print(f"  Average Bandwidth to Previous Group: {avg_bw:.2f} GB/s")

    # We are now ready to run the Stage 2 optimizer
    best_microbatch_size = None
    best_configs = None
    best_min_latency = float("inf")
    best_degree = None
    is_valid_memory = False

    microbatch_sizes_to_try = [
        2**x for x in range(1, args.global_batch_size.bit_length() - 1)
    ]

    for microbatch_size in microbatch_sizes_to_try:
        print(f"\nTrying microbatch size: {microbatch_size}")

        # Recalculate send/recv latencies for this specific microbatch size
        for i in range(len(gpu_groups) - 1):
            send_recv_latency = estimate_send_recv_latency(
                gpu_groups[i],
                gpu_groups[i + 1],
                cluster_data,
                model_stats,
                microbatch_size,
                args.sequence_length,
                args.autocast_dtype,
            )

            print(
                f"  Updated Send/Recv latency between groups {gpu_groups[i].group_id} and {gpu_groups[i+1].group_id}: {send_recv_latency:.6f}s"
            )

            gpu_groups[i].send_recv_latency_next = send_recv_latency
            gpu_groups[i + 1].send_recv_latency_prev = send_recv_latency

        # Create cluster config with updated latencies
        cluster_config = ClusterConfigStage2(groups=gpu_groups)

        # Create model_config
        model_stats = get_model_stats(args.model_name)
        model_config = ModelConfig(
            name=args.model_name,
            global_batch_size=args.global_batch_size,
            sequence_length=args.sequence_length,
            microbatch_size=microbatch_size,
            hidden_size=model_stats["hidden_size"],
            num_layers=model_stats["num_layers"],
            params_per_layer=model_stats["parameters_per_layer"],
        )

        model_latencies = load_model_compute_latencies_per_gpu(
            model_name=args.model_name,
            autocast_dtype=args.autocast_dtype,
            batch_size=microbatch_size,
            seq_length=str(args.sequence_length),
        )

        pipeline_optimizer_stage2 = PipelineOptimizer(
            cluster_config=cluster_config,
            model_config=model_config,
            model_latencies=model_latencies,
            stage_strategy="zero2",
            dtype_multiplier=args.dtype_multiplier,
            optimizer_multiplier=args.optimizer_multiplier,
            autocast_dtype=args.autocast_dtype,
            reduce_dtype=args.reduce_dtype,
            reserved_memory_gb=args.reserved_memory_gb,
        )

        (
            optimal_configs,
            optimal_degree,
            min_latency,
            is_valid_memory,
        ) = pipeline_optimizer_stage2.optimize(interleave_degree = 1 if len(gpu_groups) == 1 else None)

        print(f"\nMicrobatch Size: {microbatch_size}, Min Latency: {min_latency:.6f}s")
        if is_valid_memory:
            print("Memory validation passed.")
            # Track best configuration if it has lower latency
            if min_latency < best_min_latency:
                best_min_latency = min_latency
                best_configs = optimal_configs
                best_microbatch_size = microbatch_size
                best_degree = optimal_degree
                print(
                    f"*** New best configuration found! Latency: {best_min_latency:.6f}s ***"
                )
        else:
            print("Memory validation failed.")
            print("Stopping.")
            break

    if best_configs:
        print("\nBest configuration for this partition:")
        print(f"  Microbatch Size: {best_microbatch_size}")
        print(f"  Interleave Degree: {best_degree}")
        print(f"  Latency: {best_min_latency:.6f}s")
        print(
            f"  Layer distribution: {[config.layer_end - config.layer_start for config in best_configs]}"
        )
    else:
        print("\nNo valid configuration found for this partition.")

    print("Stage 2 optimization complete.")

    return {
        "configs": best_configs,
        "microbatch_size": best_microbatch_size,
        "degree": best_degree,
        "latency": best_min_latency,
        "model_config": model_config if best_configs else None,
    }


def export_config(config_data, output_file, args):
    if not config_data or not config_data["configs"]:
        print("No valid configuration to export.")
        return

    configs = config_data["configs"]

    config = {
        "model_name": args.model_name,
        "global_batch_size": args.global_batch_size,
        "microbatch_size": config_data["microbatch_size"],
        "sequence_length": args.sequence_length,
        "vocab_size": args.vocab_size,
        "fused_optimizer": True,
        "interleave_degree": config_data["degree"],
        "latency": config_data["latency"],
        "autocast_dtype": args.autocast_dtype,
        "reduce_dtype": args.reduce_dtype,
        "pipeline_config": [
            {
                "gpu_ranks": config.gpu_ranks,
                "num_microbatches_per_rank": config.num_microbatches_per_rank,
                "layer_partition": [config.layer_start, config.layer_end],
                "zero_config": config.zero_config,
            }
            for config in configs
        ],
    }

    with open(output_file, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Pipeline config saved to {output_file}")


def min_k_cut_split(G: nx.Graph, k: int):
    """
    Approximate k‑cut using the ‘SPLIT’ algorithm.

    At each step:
      1. Over all current connected components, find the global minimum‑weight cut
         (via Stoer–Wagner) in that component.
      2. Remove all edges in that cut.
      3. Repeat until there are ≥ k components.

    Parameters:
        G (nx.Graph): undirected, connected, with non‑negative 'weight' on edges
        k (int): desired number of components

    Returns:
        partition_map (dict): node → component ID (1..k)
        cut_value (float): total weight of all removed edges
    """
    if not nx.is_connected(G):
        raise ValueError("Graph must be connected")

    G_work = G.copy()
    cut_value = 0

    # keep splitting until we reach k components
    while nx.number_connected_components(G_work) < k:
        best_cut_val = float("inf")
        best_S = best_T = None

        # scan each component for its lightest cut
        for comp in nx.connected_components(G_work):
            if len(comp) <= 1:
                continue
            H = G_work.subgraph(comp)
            cut_val, (S, T) = stoer_wagner(H, weight="weight")
            if cut_val < best_cut_val:
                best_cut_val = cut_val
                best_S, best_T = set(S), set(T)

        # if no cut was found (all components are singletons), break
        if best_S is None:
            break

        # collect and remove the cut edges
        cut_edges = [(u, v) for u in best_S for v in G_work[u] if v in best_T]
        for u, v in cut_edges:
            cut_value += G_work[u][v]["weight"]
        G_work.remove_edges_from(cut_edges)

    # build the final partition map
    partition_map = {}
    for part_id, comp in enumerate(nx.connected_components(G_work), start=1):
        for node in comp:
            partition_map[node] = part_id

    return partition_map, cut_value


def min_k_cut_efficient(G: nx.Graph, k: int):
    """
    Approximate k-cut using the 'EFFICIENT' algorithm.

    Parameters:
        G (nx.Graph): Undirected, connected graph with weights on edges.
        k (int): Desired number of components after cuts.

    Returns:
        partition_map (dict): node -> partition ID (1-based)
        cut_value (float): total weight of edges removed
    """
    if not nx.is_connected(G):
        raise ValueError("Graph must be connected")

    G = G.copy()  # To avoid mutating the original graph
    cut_value = 0
    edge_cut_pairs = []

    # Step 1: Compute minimum weight cut for each edge
    for u, v in G.edges():
        try:
            cut_val, partition = nx.minimum_cut(G, u, v, capacity="weight")
        except nx.NetworkXError:
            continue

        reachable, non_reachable = partition
        cut_edges = set()
        for x in reachable:
            for y in G[x]:
                if y in non_reachable:
                    cut_edges.add(frozenset((x, y)))

        edge_cut_pairs.append((cut_val, cut_edges))

    # Step 2: Sort by increasing cut weight
    edge_cut_pairs.sort(key=lambda x: x[0])

    covered_edges = set()
    components = nx.number_connected_components(G)
    for cut_val, cut in edge_cut_pairs:
        if components >= k:
            break
        if not cut.issubset(covered_edges):
            # Calculate the cut value before removing edges
            for e in cut:
                edge = tuple(e)
                if G.has_edge(*edge):
                    cut_value += G.get_edge_data(*edge)["weight"]

            # Update covered edges and remove them from the graph
            covered_edges.update(cut)
            G.remove_edges_from([tuple(e) for e in cut])
            components = nx.number_connected_components(G)

    # Step 4: Build partition map from connected components
    partition_map = {}
    for i, component in enumerate(nx.connected_components(G), start=1):
        for node in component:
            partition_map[node] = i

    return partition_map, cut_value


def partition_graph_with_clustering(G, num_partitions):
    """
    This is called "Spectral Clustering"
    """
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    n = len(nodes)
    adjacency = np.zeros((n, n))

    # Fill adjacency matrix with edge weights
    for u, v, data in G.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        weight = data.get("weight", 1.0)
        adjacency[i, j] = weight
        adjacency[j, i] = weight

    clustering = SpectralClustering(
        n_clusters=num_partitions,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=0,
    )

    similarity = adjacency
    node_clusters = clustering.fit_predict(similarity)
    partition_map = {nodes[i]: int(cluster) for i, cluster in enumerate(node_clusters)}

    # Calculate cut value (sum of weights of edges crossing partition boundaries)
    cut_value = 0
    for u, v, data in G.edges(data=True):
        if partition_map[u] != partition_map[v]:
            cut_value += data.get("weight", 1.0)

    print(f"Spectral clustering complete. Cut value: {cut_value:.2f}")

    return partition_map, cut_value


def main(args):
    cluster_data = load_json_data(args.cluster_info_file)

    print("Building GPU graph with connection weights...")
    G = build_gpu_graph(cluster_data)

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print("Graph construction complete.")

    num_machines = len(cluster_data["machines"])

    best_partition_config = None
    best_partition_latency = float("inf")
    best_num_partitions = 0
    partition_time = 0.0
    stage2_time = 0.0

    if args.partition_algorithm == "min_k_cut_efficient":
        partition_algorithm_backend = min_k_cut_efficient
    elif args.partition_algorithm == "min_k_cut_split":
        partition_algorithm_backend = min_k_cut_split
    elif args.partition_algorithm == "metis":
        partition_algorithm_backend = partition_graph_with_pymetis
    elif args.partition_algorithm == "clustering":
        partition_algorithm_backend = partition_graph_with_clustering
    else:
        # TODO
        raise ValueError(
            f"Partition algorithm {args.partition_algorithm} not supported"
        )

    optimizer_start = time.time()
    for num_partitions in range(num_machines, 0, -1):
        print(f"\nPerforming PyMetis min k-cut partitioning with k={num_partitions}...")
        start_time = time.time()
        partition_map, cut_value = partition_algorithm_backend(G, num_partitions)
        end_time = time.time()
        partition_time += end_time - start_time
        print(f"Partition time: {end_time - start_time:.2f}s")

        # Group GPUs by partition
        partitions = {}
        for node, part in partition_map.items():
            if part not in partitions:
                partitions[part] = []
            partitions[part].append(node)

        # Print partition results
        print("\nPartition Results:")
        for part_id, nodes in partitions.items():
            gpu_names = [
                clean_gpu_name(
                    cluster_data["gpu_list"][int(node.split("_")[-1])]["name"]
                )
                for node in nodes
            ]
            print(f"Partition {part_id}: {len(nodes)} GPUs - {', '.join(gpu_names)}")

        print(f"Total edge cut value: {cut_value}")

        # Run optimizer and get the best configuration for this partition
        start_time = time.time()
        partition_config = run_stage2_optimizer_on_partition(
            partitions=partitions, cluster_data=cluster_data, args=args
        )
        end_time = time.time()
        stage2_time += end_time - start_time

        if (
            partition_config
            and partition_config["configs"]
            and partition_config["latency"] < best_partition_latency
        ):
            best_partition_config = partition_config
            best_partition_latency = partition_config["latency"]
            best_num_partitions = num_partitions
            print("\n*** New global best configuration found! ***")
            print(f"  Number of partitions: {best_num_partitions}")
            print(f"  Latency: {best_partition_latency:.6f}ms")

    print(f"Total optimizer time: {time.time() - optimizer_start:.2f}s")
    print(f"Total partition time: {partition_time:.2f}s")
    print(f"Total stage2 time: {stage2_time:.2f}s")
    if best_partition_config:
        print("\nExporting best global configuration:")
        print(f"  Number of partitions: {best_num_partitions}")
        print(f"  Microbatch Size: {best_partition_config['microbatch_size']}")
        print(f"  Interleave Degree: {best_partition_config['degree']}")
        print(f"  Latency: {best_partition_latency:.6f}ms")
        export_config(best_partition_config, args.output_file, args)
    else:
        print("\nNo valid configuration found across all partitions.")


def parse_args():
    global MODEL_NAME, AUTOCAST_DTYPE, GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH, MODEL_LATENCIES_PER_GPU
    parser = argparse.ArgumentParser(description="Visualize GPU connection graph.")
    parser.add_argument("--cluster_info_file", type=str, required=True)
    parser.add_argument("--machine_file", type=str, required=True)
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of model to profile"
    )
    parser.add_argument(
        "--partition_algorithm",
        type=str,
        default="min_k_cut_split",
        help="min_k_cut_efficient, min_k_cut_split, metis, or clustering",
    )
    parser.add_argument(
        "--autocast_dtype",
        type=str,
        default="float16",
        help="Data type for autocasting (default: float16)",
    )
    parser.add_argument(
        "--reduce_dtype",
        type=str,
        default="float32",
        help="Data type for reduction operations (default: float16)",
    )
    parser.add_argument(
        "--num_trials", type=int, default=5, help="Number of profiling trials"
    )
    parser.add_argument(
        "--num_warmup", type=int, default=5, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--nccl_ib_disable", action="store_true", help="Disable InfiniBand for NCCL"
    )
    parser.add_argument("--vocab_size", type=int, default=49152)
    parser.add_argument("--dtype_multiplier", type=float, default=2.0)
    parser.add_argument("--optimizer_multiplier", type=float, default=12.0)
    parser.add_argument("--reserved_memory_gb", type=float, default=2.0)
    parser.add_argument("--global_batch_size", type=int, default=128)
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--output_file", type=str, default="pipeline_config.json")
    parser.add_argument("--use_agrs_comm_model", action="store_true")
    args = parser.parse_args()
    MODEL_NAME = args.model_name
    AUTOCAST_DTYPE = args.autocast_dtype
    GLOBAL_BATCH_SIZE = args.global_batch_size
    SEQUENCE_LENGTH = args.sequence_length
    MODEL_LATENCIES_PER_GPU = load_model_compute_latencies_per_gpu(
        model_name=MODEL_NAME,
        autocast_dtype=AUTOCAST_DTYPE,
        batch_size=GLOBAL_BATCH_SIZE,
        seq_length=str(SEQUENCE_LENGTH),
    )
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
