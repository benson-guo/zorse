# -*- coding: utf-8 -*-
import torch
import functools
import argparse
from torch.distributed import fsdp
from utils.comm import dist_init, get_gpu_name
from utils.runtime_estimator import LatencyEstimator


def bench_model(args, local_rank, gpu_name):
    torch.cuda.set_device(local_rank)
    model_name = args.model_name
    seq_length = args.seq_length
    vocab_size = args.vocab_size
    image_size = args.image_size
    num_shards = args.num_shards
    profile_batches = args.profile_batches
    trace_dir = args.trace_dir
    mp_config = fsdp.MixedPrecision(
        param_dtype=args.autocast_dtype,
        buffer_dtype=args.autocast_dtype,
        reduce_dtype=args.reduce_dtype,
    )
    print(
        f"Rank: {local_rank} Constructing latency estimator for {model_name} on {gpu_name}"
    )
    LatencyEstimator(
        model_name,
        gpu_type=gpu_name,
        mp_config=mp_config,
        num_shards=num_shards,
        seq_length=seq_length,
        vocab_size=vocab_size,
        image_size=image_size,
        profile_batches=profile_batches,
        trace_dir=trace_dir,
    )
    print(f"Rank: {local_rank}: Constructed latency estimator")


def main(args):
    dist_init()
    profiled_gpus = set()
    for i in range(torch.cuda.device_count()):
        gpu_name = get_gpu_name(i)
        if gpu_name in profiled_gpus:
            continue
        profiled_gpus.add(gpu_name)
        bench_model(args, i, gpu_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    get_dtype = functools.partial(getattr, torch)
    parser.add_argument(
        "--autocast_dtype",
        type=get_dtype,
        choices=[torch.bfloat16, torch.float16, torch.float32],
        default=torch.float16,
    )
    parser.add_argument(
        "--reduce_dtype",
        type=get_dtype,
        choices=[torch.bfloat16, torch.float16, torch.float32],
        default=torch.float16,
    )
    parser.add_argument("--model_name", type=str, default="gpt_85m")
    parser.add_argument("--vocab_size", type=int, default=49152)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--profile_batches", type=int, default=5)
    parser.add_argument("--trace_dir", type=str, default=None)
    args = parser.parse_args()

    main(args)
