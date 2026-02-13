import argparse
import json
import numpy as np
import os

from models.hub import get_model_stats

MB_TO_GB = 1.0 / 1024
BYTES_TO_GB = 1.0 / (1024 * 1024 * 1024)
MS_TO_S = 1.0 / 1000

DTYPE_MULTIPLIER = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
}


def get_tensor_size_gb(model_name, dtype):
    model_stats = get_model_stats(model_name)
    total_params = model_stats["parameters_per_layer"]
    tensor_size_bytes = total_params * DTYPE_MULTIPLIER[dtype]
    tensor_size_gb = tensor_size_bytes * BYTES_TO_GB
    return tensor_size_gb

def get_bandwidth_matrix(file_path, gpu_ranks):
    with open(file_path, "r") as f:
        cluster_data_json = json.load(f)
    
    internode_matrix = cluster_data_json["bandwidth"]["internode_bandwidth"]
    intranode_bandwidth = cluster_data_json["bandwidth"]["intranode_bandwidth"]
    global_rank_map = cluster_data_json["global_rank_map"]
    
    n_gpus = len(gpu_ranks)
    bandwidth_matrix = np.zeros((n_gpus, n_gpus))
    
    for i in range(n_gpus):
        for j in range(n_gpus):
            if i == j:
                bandwidth_matrix[i, j] = float('inf')
                continue
                
            rank_i = gpu_ranks[i]
            rank_j = gpu_ranks[j]
            
            machine_i = int(global_rank_map[str(rank_i)]["machine_index"])
            machine_j = int(global_rank_map[str(rank_j)]["machine_index"])
            
            if machine_i == machine_j:
                bandwidth_matrix[i, j] = intranode_bandwidth[str(machine_i)]
            else:
                if str(machine_i) in internode_matrix and str(machine_j) in internode_matrix[str(machine_i)]:
                    bandwidth_matrix[i, j] = internode_matrix[str(machine_i)][str(machine_j)]
                elif str(machine_j) in internode_matrix and str(machine_i) in internode_matrix[str(machine_j)]:
                    bandwidth_matrix[i, j] = internode_matrix[str(machine_j)][str(machine_i)]
                else:
                    min_internode = float('inf')
                    for m1 in internode_matrix:
                        for m2 in internode_matrix[m1]:
                            if internode_matrix[m1][m2] < min_internode:
                                min_internode = internode_matrix[m1][m2]
                    
                    bandwidth_matrix[i, j] = min_internode
    
    return bandwidth_matrix

def estimate_collective_latency(bandwidth_matrix, tensor_size_gb, operation_type, n_gpus=None):
    if n_gpus is None:
        n_gpus = bandwidth_matrix.shape[0]
    
    if operation_type not in ['all_gather', 'reduce_scatter']:
        raise ValueError("Operation type must be 'all_gather' or 'reduce_scatter'")

    min_bandwidth = float('inf')
    for i in range(n_gpus):
        j = (i + 1) % n_gpus
        if bandwidth_matrix[i, j] < min_bandwidth:
            min_bandwidth = bandwidth_matrix[i, j]
    
    data_volume = (n_gpus - 1) * tensor_size_gb / n_gpus
    
    latency = data_volume / min_bandwidth
    
    latency_us = latency * 1_000
    
    return latency_us

def estimate_collective_latency_v2(bandwidth_matrix, tensor_size_gb, operation_type, n_gpus=None):
    if n_gpus is None:
        n_gpus = bandwidth_matrix.shape[0]
    
    if operation_type not in ['all_gather', 'reduce_scatter']:
        raise ValueError("Operation type must be 'all_gather' or 'reduce_scatter'")

    
    data_per_step = tensor_size_gb / n_gpus
    
    total_time = 0
    
    for step in range(n_gpus - 1):
        step_times = []
        for i in range(n_gpus):
            sender = i
            receiver = (i + 1) % n_gpus
            
            transfer_time = data_per_step / bandwidth_matrix[sender, receiver]
            
            step_times.append(transfer_time)
        
        total_time += max(step_times)
    
    total_time_us = total_time * 1_000
    
    return total_time_us * 1.15

def main():
    parser = argparse.ArgumentParser(description="Script to configure the communication model.")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the model to use.")
    parser.add_argument('--gpu_ranks', type=int, nargs='+', required=True, help="List of GPU ranks to use.")
    parser.add_argument('--cluster_info', type=str, required=True, help="Path to cluster information JSON file.")
    parser.add_argument('--autocast_dtype', type=str, default='float16', choices=['float16', 'bfloat16'], 
                       help="Data type for autocasting (used for all-gather).")
    parser.add_argument('--reduce_dtype', type=str, default='float32', choices=['float32', 'float16', 'bfloat16'], 
                       help="Data type for reduction (used for reduce-scatter).")

    args = parser.parse_args()

    print(f"Model Name: {args.model_name}")
    print(f"GPU Ranks: {args.gpu_ranks}")
    print(f"Cluster Info: {args.cluster_info}")
    print(f"Autocast Data Type: {args.autocast_dtype}")
    print(f"Reduce Data Type: {args.reduce_dtype}")
    print(f"Number of GPUs: {len(args.gpu_ranks)}")
    
    all_gather_tensor_size = get_tensor_size_gb(args.model_name, args.autocast_dtype)
    reduce_scatter_tensor_size = get_tensor_size_gb(args.model_name, args.reduce_dtype)
    
    print(f"Tensor Size (GB): {all_gather_tensor_size}")
    
    bandwidth_matrix = get_bandwidth_matrix(args.cluster_info, args.gpu_ranks)
    
    print("Bandwidth Matrix (GB/s):")
    np.set_printoptions(precision=3, suppress=True)
    print(bandwidth_matrix)
    
    print("\nMethod 1 (Basic):")
    all_gather_latency = estimate_collective_latency(
        bandwidth_matrix, all_gather_tensor_size, 'all_gather', len(args.gpu_ranks))
    reduce_scatter_latency = estimate_collective_latency(
        bandwidth_matrix, reduce_scatter_tensor_size, 'reduce_scatter', len(args.gpu_ranks))
    
    print(f"All-Gather (using {args.autocast_dtype}) estimated latency: {all_gather_latency:.2f} ms")
    print(f"Reduce-Scatter (using {args.reduce_dtype}) estimated latency: {reduce_scatter_latency:.2f} ms")
    
    print("\nMethod 2 (Advanced):")
    all_gather_latency_v2 = estimate_collective_latency_v2(
        bandwidth_matrix, all_gather_tensor_size, 'all_gather', len(args.gpu_ranks))
    reduce_scatter_latency_v2 = estimate_collective_latency_v2(
        bandwidth_matrix, reduce_scatter_tensor_size, 'reduce_scatter', len(args.gpu_ranks))
    
    print(f"All-Gather (using {args.autocast_dtype}) estimated latency: {all_gather_latency_v2:.2f} ms")
    print(f"Reduce-Scatter (using {args.reduce_dtype}) estimated latency: {reduce_scatter_latency_v2:.2f} ms")

if __name__ == "__main__":
    main()