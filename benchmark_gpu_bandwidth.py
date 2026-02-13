import torch
import torch.distributed as dist
import contextlib
import sys
import argparse
import json
import os

# add ../ to path
sys.path.append("../")
from utils.profile import (
    extract_function_runtime,
    extract_kernel_runtime,
    get_profiler_context,
)
from utils.comm import (
    dist_init,
    get_local_rank,
    get_global_rank,
    get_world_size,
    is_local_leader,
)


# Measure the send/receive bandwidth between two GPUs
def measure_bandwidth(
    rank,
    size,
    device,
    rank1,
    rank2,
    backend,
    warmup_iterations=10,
    iterations=50,
    parallel=5,
):
    is_sender = rank == rank1
    is_receiver = rank == rank2
    if not is_sender and not is_receiver:
        tensors = []
    else:
        tensors = [torch.randn(size, device=device) for _ in range(parallel)]
    torch.cuda.synchronize()
    for _ in range(warmup_iterations):
        works = []
        for tensor in tensors:
            if is_sender:
                work = dist.isend(tensor, dst=rank2)
            elif is_receiver:
                work = dist.irecv(tensor, src=rank1)
            works.append(work)
        for work in works:
            work.wait()
    if is_sender:
        profiler_ctx = get_profiler_context()
    else:
        profiler_ctx = contextlib.nullcontext()
    dist.barrier()
    torch.cuda.synchronize()
    with profiler_ctx:
        for _ in range(iterations):
            works = []
            for tensor in tensors:
                if is_sender:
                    work = dist.isend(tensor, dst=rank2)
                elif is_receiver:
                    work = dist.irecv(tensor, src=rank1)
                works.append(work)
            for work in works:
                work.wait()
        dist.barrier()
        torch.cuda.synchronize()
    if not is_sender:
        return 0.0, 0.0
    if backend == "nccl":
        avg_comm_time = extract_kernel_runtime(num_iterations=iterations * parallel)
    else:
        avg_comm_time = extract_function_runtime(
            name="gloo:send", num_iterations=iterations * parallel
        )
    print(f"Average communication time: {avg_comm_time:.2f} ms")
    print(f"Tensor size: {tensor.element_size()}", flush=True)
    if avg_comm_time == 0:
        print("Error: No communication kernels recorded", flush=True)
        bandwidth = 0.0
    else:
        bandwidth = (
            size * tensor.element_size() / 1e9 / (avg_comm_time / 1000)
        )  # in Gb/s
        latency = (avg_comm_time / 1000) / parallel  # in ms
        print(f"Bandwidth: {bandwidth:.2f} Gb/s", flush=True)
        print(f"Latency: {latency:.3f} ms", flush=True)
    return bandwidth, latency


def parse_node_ranks(ranks_string):
    """Parse a string of ranks in the format '0,1,2,3#4,5,6'
    where # separates different nodes"""
    nodes = []
    for node_str in ranks_string.split("#"):
        if node_str:
            nodes.append([int(x) for x in node_str.split(",")])
    return nodes


def get_internode_representatives(node_ranks):
    """Get the first rank from each node as representatives for internode communication"""
    return [node[0] for node in node_ranks if node]


def get_intranode_representatives(node_ranks):
    """Get pairs of ranks within each node for intranode communication"""
    intranode_pairs = []
    for node in node_ranks:
        if len(node) >= 2:
            # Create pairs within the same node
            for i in range(len(node) - 1):
                for j in range(i + 1, len(node)):
                    intranode_pairs.append((node[i], node[j]))
    return intranode_pairs


def run_internode_bandwidth_test(args, size, backend):
    world_size = get_world_size()
    rank = get_global_rank()
    local_rank = get_local_rank()
    local_leader = is_local_leader()
    device = (
        torch.device(f"cuda:{local_rank}") if backend == "nccl" else torch.device("cpu")
    )

    # Parse node ranks
    node_ranks = parse_node_ranks(args.node_ranks)

    # Get representatives for internode communication (first GPU from each node)
    representatives = get_internode_representatives(node_ranks)

    print(f"Internode representatives: {representatives}")

    # Create pairs of representatives from different nodes
    pairs = []
    for i in range(len(representatives)):
        for j in range(i + 1, len(representatives)):
            pairs.append((representatives[i], representatives[j]))

    # Initialize bandwidth results
    internode_bandwidth = {}
    internode_latency = {}
    for src, dst in pairs:
        # Find which node these ranks belong to
        src_node = -1
        dst_node = -1
        for node_idx, node in enumerate(node_ranks):
            if src in node:
                src_node = node_idx
            if dst in node:
                dst_node = node_idx

        if src_node not in internode_bandwidth:
            internode_bandwidth[src_node] = {}
        if src_node not in internode_latency:
            internode_latency[src_node] = {}

        # Measure bandwidth
        bandwidth, latency = measure_bandwidth(
            rank,
            size,
            device,
            src,
            dst,
            backend,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
            parallel=args.parallel,
        )  # type: ignore

        # All reduce bandwidth
        b_tensor = torch.tensor([bandwidth], device=device)
        l_tensor = torch.tensor([latency], device=device)
        dist.all_reduce(b_tensor)
        dist.all_reduce(l_tensor)  # Reduce latency
        internode_bandwidth[src_node][dst_node] = b_tensor[0].item()
        internode_latency[src_node][dst_node] = l_tensor[0].item()

        # Sync between measurements
        dist.barrier()

    if local_leader:
        print("\nInternode Bandwidths between nodes (Gb/s):")
        print(f"{'Source Node':>12} {'Dest Node':>12} {'Bandwidth':>12}")
        print("-" * 40)
        for src_node in sorted(internode_bandwidth.keys()):
            for dst_node, bw in sorted(internode_bandwidth[src_node].items()):
                print(f"{src_node:>12} {dst_node:>12} {bw:>12.2f}")

    dist.barrier()
    return {
        "internode_bandwidth": internode_bandwidth,
        "internode_latency": internode_latency,
    }


def run_intranode_bandwidth_test(args, size, backend):
    world_size = get_world_size()
    rank = get_global_rank()
    local_rank = get_local_rank()
    local_leader = is_local_leader()
    device = (
        torch.device(f"cuda:{local_rank}") if backend == "nccl" else torch.device("cpu")
    )

    # Parse node ranks (for intranode, we should have just one node's ranks)
    node_ranks = args.node_ranks.split(",")
    node_ranks = [int(r) for r in node_ranks]

    # Get all possible pairs within this node
    pairs = []
    for i in range(len(node_ranks)):
        for j in range(i + 1, len(node_ranks)):
            pairs.append((node_ranks[i], node_ranks[j]))

    print(f"Intranode pairs to test: {pairs}")

    # Initialize bandwidth results
    intranode_bandwidths = []
    intranode_latencies = []

    # one use one pair
    if args.fast_intra:
        pairs = [pairs[0]]

    for src, dst in pairs:
        bandwidth, latency = measure_bandwidth(
            rank,
            size,
            device,
            src,
            dst,
            backend,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
            parallel=args.parallel,
        )  # type: ignore

        # All reduce bandwidth
        b_tensor = torch.tensor([bandwidth], device=device)
        l_tensor = torch.tensor([latency], device=device)
        dist.all_reduce(b_tensor)
        dist.all_reduce(l_tensor)
        intranode_bandwidths.append(b_tensor[0].item())
        intranode_latencies.append(l_tensor[0].item())

        # Sync between measurements
        dist.barrier()

    # Calculate average intranode bandwidth
    min_intranode_bandwidth = min(intranode_bandwidths) if intranode_bandwidths else 0
    min_intranode_latency = min(intranode_latencies) if intranode_latencies else 0

    if local_leader:
        print(
            f"\nIntranode Bandwidth (minimum of {len(pairs)} pairs): {min_intranode_bandwidth:.2f} Gb/s"
        )
        print(
            f"Intranode Latency (minimum of {len(pairs)} pairs): {min_intranode_latency:.3f} ms"
        )

        # Find the node index based on the hostname or use 0 if we can't determine
        node_idx = 0

        # Create a dictionary with the machine's intranode bandwidth
        intranode_bandwidth = {node_idx: min_intranode_bandwidth}
        intranode_latency = {node_idx: min_intranode_latency}

        return {
            "intranode_bandwidth": intranode_bandwidth,
            "intranode_latency": intranode_latency,
        }

    return {"intranode_bandwidth": {}, "intranode_latency": {}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parallel", type=int, default=1, help="number of parallel communications"
    )
    parser.add_argument(
        "--send_msg_size", type=int, default=10**7, help="size of send/recv message"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="backend to use for distributed communication",
    )
    parser.add_argument(
        "--node_ranks",
        type=str,
        default="0",
        help="ranks on each node in format '0,1,2,3#4,5,6' where # separates nodes",
    )
    parser.add_argument(
        "--gpu_global_ranks",
        type=str,
        default="",
        help="alternative format for node_ranks for backward compatibility",
    )
    parser.add_argument(
        "--internode",
        action="store_true",
        help="test internode communication using GLOO backend",
    )
    parser.add_argument(
        "--intranode",
        action="store_true",
        help="test intranode communication using NCCL backend",
    )
    parser.add_argument(
        "--fast_intra",
        action="store_true",
    )
    parser.add_argument(
        "--warmup_iterations", type=int, default=10, help="number of warmup iterations"
    )
    parser.add_argument(
        "--iterations", type=int, default=50, help="number of test iterations"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="bandwidth_results.json",
        help="output file for bandwidth results",
    )
    args = parser.parse_args()

    # For compatibility with original script
    if args.gpu_global_ranks and not args.node_ranks:
        args.node_ranks = args.gpu_global_ranks

    # Set backend based on test type
    if args.internode:
        args.backend = "gloo"
    elif args.intranode:
        args.backend = "nccl"

    dist_init(backend=args.backend)
    local_leader = is_local_leader()
    local_rank = get_local_rank()

    if local_leader:
        print(f"Running bandwidth test for size {args.send_msg_size}")
        print(f"Node ranks: {args.node_ranks}")
        print(f"Backend: {args.backend}")
        print(f"Parallel communication: {args.parallel}")
        print(
            f"Test type: {'Internode' if args.internode else 'Intranode' if args.intranode else 'Custom'}"
        )
        print("=" * 60)

    # Run the appropriate test
    if args.internode:
        results = run_internode_bandwidth_test(args, args.send_msg_size, args.backend)
    elif args.intranode:
        results = run_intranode_bandwidth_test(args, args.send_msg_size, args.backend)
    else:
        # For backward compatibility - this won't be used in our case
        print("Neither --internode nor --intranode specified. Exiting.")
        results = {}

    # Save results to output file
    if local_leader and args.output_file and results:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {args.output_file}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
