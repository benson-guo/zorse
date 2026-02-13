# -*- coding: utf-8 -*-
import argparse
import functools
import torch
import pickle
import contextlib
import numpy as np
from utils.comm import is_leader, dist_init
from utils import comm_patterns
from utils.profile import (
    extract_kernel_runtime,
    get_profiler_context,
    print_memory_stats,
)

# MESSAGE_SIZES_MB = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

MESSAGE_SIZES_MB = [1] + [i for i in range(20, 1001, 20)]

CONFIG = []


def parse_args():
    parser = argparse.ArgumentParser()
    get_dtype = functools.partial(getattr, torch)
    parser.add_argument(
        "--dtype",
        type=get_dtype,
        choices=[torch.bfloat16, torch.float16, torch.float32],
        default=torch.float16,
    )
    parser.add_argument("--num_warmup_trials", type=int, default=8)
    parser.add_argument("--num_inner_trials", type=int, default=32)
    parser.add_argument("--num_outer_trials", type=int, default=1)
    parser.add_argument("--dataset_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pattern", type=str, default="ag")
    parser.add_argument("--output_file", type=str, default="/tmp/measurements_1000.pkl")
    parser.add_argument("--size_mb", type=int, default=1024)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--variable_size", action="store_true")
    return parser.parse_args()


def collect_data(args):
    measurements = {}
    for idx, (size_mb, shard_ratio) in enumerate(CONFIG):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        if is_leader():
            print(f"======================================== Iteration {idx}")
            print(f"World size: {args.world_size}")
            print(f"Message size (MiB): {size_mb}")
            print(f"Shard Ratio: {shard_ratio}")
            print(f"Dtype: {args.dtype}")
            print(f"Num inner trials: {args.num_inner_trials}")
            print(f"Num outer trials: {args.num_outer_trials}")

        if args.pattern == "ag":
            pattern_cls = comm_patterns.AllGatherUnevenRatio
            pattern = pattern_cls(size_mb, shard_ratio, dtype=args.dtype)
        elif args.pattern == "b":
            pattern_cls = comm_patterns.Broadcast
            pattern = pattern_cls(size_mb, dtype=args.dtype)
        elif args.pattern == "ag_even":
            pattern_cls = comm_patterns.AllGather
            pattern = pattern_cls(size_mb, dtype=args.dtype)
        elif args.pattern == "rs_even":
            pattern_cls = comm_patterns.ReduceScatter
            pattern = pattern_cls(size_mb, dtype=args.dtype)
        elif args.pattern == "rs":
            pattern_cls = comm_patterns.ReduceScatterUnevenRatio
            pattern = pattern_cls(size_mb, shard_ratio, dtype=args.dtype)
        else:
            raise NotImplementedError

        # warmup before profiling
        if is_leader():
            print(f"Benchmarking {pattern.name}")
        for _ in range(args.num_warmup_trials):
            pattern.execute()

        # profile time
        torch.cuda.synchronize()
        if is_leader():
            profiler_ctx = get_profiler_context()
        else:
            profiler_ctx = contextlib.nullcontext()

        allocated_out = torch.tensor([0.0 for _ in range(args.world_size)], dtype=torch.float32).cuda()

        with profiler_ctx:
            for i in range(args.num_inner_trials):
                pattern.execute()
                if i == 0:
                    # get memory stats
                    max_allocated_mem = print_memory_stats(
                        "all_gather_mem_stats", all_ranks=False
                    )["max_allocated"]
                    allocated_in = torch.tensor([max_allocated_mem], dtype=torch.float32).cuda()
                    torch.distributed.all_gather_into_tensor(
                        allocated_out,
                        allocated_in,
                    )

        torch.cuda.synchronize()

        if not is_leader():
            continue
        avg_comm_time = extract_kernel_runtime(num_iterations=args.num_inner_trials)

        measurement = {
            "size_mb": size_mb,
            "shard_ratio": shard_ratio,
            "avg_time": avg_comm_time,
            "mem_stats": allocated_out.tolist(),
        }
        if pattern.name in measurements:
            measurements[pattern.name].append(measurement)
        else:
            measurements[pattern.name] = [measurement]
        if is_leader():
            print(f"measurement: {measurement}")
            print(f"Runtime: {avg_comm_time} ms")

    if is_leader():
        with open(args.output_file, "wb") as f:
            pickle.dump(measurements, f)
        print(f"Measurements saved to {args.output_file}")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    LINEAR_RATIOS = np.linspace(0.05, 1, args.dataset_size + 1)
    global CONFIG
    world_size = dist_init()
    args.world_size = world_size
    for i in range(args.dataset_size):
        for _ in range(args.num_samples):
            random_ratio = np.random.uniform(LINEAR_RATIOS[i], LINEAR_RATIOS[i + 1])
            shard_ratio = np.random.rand(args.world_size - 1)
            available_sum = 1 - random_ratio
            shard_ratio = shard_ratio / shard_ratio.sum() * available_sum  # normalized

            # Including the random_ratio in the shard_ratio array
            shard_ratio = np.append(shard_ratio, random_ratio)

            # Randomizing the assignment of ratios to GPUs
            np.random.shuffle(shard_ratio)

            assert np.isclose(
                sum(shard_ratio), 1, 1e-5
            ), "The sum of shard_ratios must be close to 1"

            if args.variable_size:
                message_size = MESSAGE_SIZES_MB[i % len(MESSAGE_SIZES_MB)]
            else:
                message_size = args.size_mb
            CONFIG.append((message_size, shard_ratio.tolist()))

    for _ in range(args.num_outer_trials):
        collect_data(args)


if __name__ == "__main__":
    main()
