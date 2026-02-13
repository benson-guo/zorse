# -*- coding: utf-8 -*-
import argparse
import torch
import functools
import ast
import json
from utils.argparser_utils import get_dtype_from_str


def parse_2d_array(arg):
    try:
        # Safely evaluate the string to a Python literal (list in this case)
        # convert list to tuple to make it hashable
        return tuple(tuple(lst) for lst in ast.literal_eval(arg))
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(
            "Invalid 2D array format. Example: [[1,2],[2,3]]"
        )


def string_list(raw_arg):
    return raw_arg.split(",")


def override_args_with_config(args):
    with open(args.config_file, "r") as f:
        config = json.load(f)
        for key, value in config.items():
            if "dtype" in key:
                value = get_dtype_from_str(value)
            # TODO: currently use different name for seq_length in config file
            if key == "sequence_length":
                key = "seq_length"
            setattr(args, key, value)


def parse_args_grolar():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    get_dtype = functools.partial(getattr, torch)
    parser.add_argument(
        "--autocast_dtype",
        type=get_dtype,
        choices=[torch.bfloat16, torch.float16, torch.float32],
        default=torch.float32,
    )
    parser.add_argument(
        "--reduce_dtype",
        type=get_dtype,
        choices=[torch.bfloat16, torch.float16, torch.float32],
        default=torch.float32,
    )
    parser.add_argument("--model_name", type=str, default="gpt_85m")
    parser.add_argument("--cluster", type=str, default="paper")
    parser.add_argument("--iterations", type=int, default=4, help="Training Iterations")
    parser.add_argument(
        "--warmup_iterations", type=int, default=5, help="Warmup Iterations"
    )
    parser.add_argument("--profiling_iterations", type=int, default=1)
    parser.add_argument("--vocab_size", type=int, default=49152)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--microbatches", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--full_shard_layers", type=int, default=10000)
    parser.add_argument("--recompute_layers", type=int, default=10000)
    parser.add_argument("--no_fused_optimizer", action="store_true")
    parser.add_argument("--recompute_layer", action="store_true")
    parser.add_argument("--recompute_feed_forward", action="store_true")
    parser.add_argument("--recompute_attention", action="store_true")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--offload_activations", action="store_true")
    parser.add_argument("--sync_backwards", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no_shard", action="store_true")
    parser.add_argument("--hybrid_shard", action="store_true")
    parser.add_argument(
        "--hybrid_shard_groups",
        type=parse_2d_array,
        default=None,
        help="Input 2D array in format [[0, 1],[2,3]]",
    )
    parser.add_argument("--reverse_hs", action="store_true")
    parser.add_argument("--optimizer_in_backwards", action="store_true")
    parser.add_argument("--split_uneven", action="store_true")
    parser.add_argument("--ga", action="store_true")
    parser.add_argument("--proportional_split", action="store_true")
    parser.add_argument("--skip_profile", action="store_true")
    parser.add_argument("--profile_memory", action="store_true")
    parser.add_argument("--profile_ga", action="store_true")
    parser.add_argument("--use_activation_buffers", action="store_true")
    parser.add_argument("--detailed_profiler_trace", action="store_true")
    parser.add_argument("--use_prefetch_backwards", action="store_true")
    parser.add_argument("--use_forwards_gpu_buffer", action="store_true")
    parser.add_argument("--unshard_in_compute", action="store_true")
    parser.add_argument("--async_tp", action="store_true")
    parser.add_argument("--enable_loss_parallel", action="store_true")
    parser.add_argument("--compile_transformer_blocks", action="store_true")
    parser.add_argument(
        "-sup",
        "--split_uneven_partitions",
        nargs="+",
        default=[0.25, 0.75],
        help="",
    )
    parser.add_argument(
        "-ubs",
        "--uneven_batch_sizes",
        nargs="+",
        default=[],
        help="",
    )
    parser.add_argument(
        "-um",
        "--uneven_microbatches",
        nargs="+",
        default=[],
        help="",
    )
    parser.add_argument(
        "-tol",
        type=float,
        default=0.0,
    )
    parser.add_argument("--trace_dir", type=str, default=None)
    parser.add_argument(
        "--experiment_name", type=str, default=None
    )  # name for this experiment
    parser.add_argument("--gap_threshold", type=float, default=0)
    parser.add_argument("--scale_config_bs", type=int, default=1)

    # **Added Arguments for Parallelism**
    # Pipeline Parallelism Degree
    parser.add_argument(
        "--tt_pp",
        type=int,
        default=1,
        help="Number of pipeline parallel stages (e.g., 2 for 2 stages)",
    )
    # resahrd_after_forward (Zero 2)
    parser.add_argument("--zero2", action="store_true")

    parser.add_argument(
        "--tt_pp_schedule",
        type=str,
        default="1f1b",
        help="Schedule to use for PP",
    )

    # Data Parallel Sharding Degree for FSDP
    parser.add_argument(
        "--tt_dp_shard",
        type=int,
        default=1,
        help="Data parallel sharding degree for FSDP (e.g., 2 for sharding across 2 GPUs)",
    )

    # Data Parallel Replication Degree
    parser.add_argument(
        "--tt_dp_replicate",
        type=int,
        default=1,
        help="Data parallel replication degree (set to 1 if not replicating)",
    )

    # Tensor Parallelism Degree (if applicable)
    parser.add_argument(
        "--tt_tp",
        type=int,
        default=1,
        help="Tensor parallelism degree (e.g., 1 if not using tensor parallelism)",
    )

    # Pipeline Split Points (layer indices where the pipeline is split)
    parser.add_argument(
        "--tt_pp_split_points",
        nargs="+",
        type=str,
        default=[],
        help="Layer indices where the pipeline is split into stages (e.g., 12 24 for splitting at layers 12 and 24)",
    )

    parser.add_argument(
        "--tt_pp_microbatches",
        type=int,
        default=1,
        help="PP microbatches using TT",
    )

    parser.add_argument("--use_deepcopy_for_build", action="store_true")
    parser.add_argument(
        "--mfu_gpus",
        nargs="+",
        default=[],
        help="GPUs used to calculate MFU",
    )
    parser.add_argument(
        "--config_file", type=str, default="./grolar_configs/paper103_default.json"
    )
    parser.add_argument("--sync_comm", action="store_true")
    parser.add_argument("--two_stage_sync", action="store_true")
    parser.add_argument("--zero2_pipeline", action="store_true")
    parser.add_argument("--flashflex_pipeline", action="store_true")
    parser.add_argument("--flashflex_ga", type=int, default=1)
    parser.add_argument("--gloo_p2p", action="store_true")
    parser.add_argument("--offload_model_params", action="store_true")
    parser.add_argument("--print_metrics_every_iteration", action="store_true")
    args = parser.parse_args()

    return args
