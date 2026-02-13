# -*- coding: utf-8 -*-
import json
import torch
import os
import datetime

try:
    from utils.logger import get_logger
    from utils.comm import (
        get_global_rank,
        get_gpu_name,
        is_local_leader,
        get_world_size,
    )
    from utils.argparser_utils import get_dtype_str
except ImportError:
    # needed for flashflex
    from groler.utils.logger import get_logger
    from groler.utils.comm import (
        get_global_rank,
        get_gpu_name,
        is_local_leader,
        get_world_size,
    )
    from groler.utils.argparser_utils import get_dtype_str

import numpy as np
import torch.distributed as dist
from models.hub import (
    get_config_for_model,
    get_total_model_params,
    get_model,
    get_layers,
    get_model_stats,
)

COMMUNICATION_STREAM_KERNELS = [
    "ncclKernel_Broadcast_RING_LL_Sum",
    "ncclKernel_Reduce_RING_LL_Sum_half",
    "ncclKernel_AllGather_RING_LL_Sum",
]
MEMCPY_STREAM_KERNELS = [
    "Memcpy DtoD (Device -> Device)",
    "Memcpy DtoH (Device -> Pinned)",
    "Memcpy HtoD (Pinned -> Device)",
]


def find_gaps_in_stream(events, gap_threshold=0):
    """
    Find gaps in a sorted stream/similar events
    Set gap_threshold for minimum gaps ( in us )
    """
    gaps = []
    for i in range(len(events) - 1):
        start_time = events[i]["ts"] + events[i].get("dur", 0)
        end_time = events[i + 1]["ts"]
        gap = end_time - start_time
        if gap > gap_threshold:
            # name is for the kernel exec just before
            gaps.append((start_time, end_time, gap))
    return gaps


def get_overhead_from_stream(gaps, compare_stream):
    """
    Get overhead from comparing stream
    """
    total_overhead = 0

    for gap_start, gap_end, gap in gaps:
        overhead = 0
        overlapping_ops = []

        # Find operations that overlap with the current gap
        for op in compare_stream:
            op_start = op["ts"]
            op_end = op_start + op.get("dur", 0)

            # Check if operation overlaps with the gap
            if op_start < gap_end and op_end > gap_start:
                overlapping_ops.append(op)
                # Calculate overhead if the operation starts before the gap
                if op_start <= gap_start and op_end <= gap_end:
                    overhead += op_end - gap_start
                    # if overhead > 0:
                    #     print(f"Gap (duration: {gap} us) has an overhead of {overhead} us caused by overlapping stream operations")

        # Add the duration of any operations during the gap
        for op in overlapping_ops:
            op_start = op["ts"]
            op_end = op_start + op.get("dur", 0)
            # If the operation is entirely within the gap, add its duration to the overhead
            if op_start >= gap_start and op_end <= gap_end:
                overhead += op.get("dur", 0)
                # if overhead > 0:
                #     print(f"Gap (duration: {gap} us) has an overhead of {overhead} us caused by stream operations")

        total_overhead += overhead

    return total_overhead


def get_overheads(pt_trace_file, args, compute_overheads=True):
    """
    Returns:
    1. Overhead from cudaMalloc/Free for compute stream
    2. Overhead from compute stream for communication stream
    3. Total duration for cuda Malloc/Free, compute stream, communication stream, network stream
    """
    with open(pt_trace_file, "r") as f:
        data = json.load(f)

    trace_events = data["traceEvents"]

    # setting default streams
    compute_stream = 7
    gpu_mem_copy_stream = None
    communication_stream = None

    # find streams
    for event in trace_events:
        for k in COMMUNICATION_STREAM_KERNELS:
            if k in event.get("name", ""):
                communication_stream = event.get("tid")
                break

    for event in trace_events:
        for k in MEMCPY_STREAM_KERNELS:
            if k in event.get("name", ""):
                gpu_mem_copy_stream = event.get("tid")
                break

    if gpu_mem_copy_stream is None:
        gpu_mem_copy_stream = 16
    if communication_stream is None:
        communication_stream = 27

    # cuda malloc and free events
    cuda_events = [
        e
        for e in trace_events
        if "name" in e and (e["name"] == "cudaMalloc" or e["name"] == "cudaFree")
    ]
    total_cuda_duration = sum([op.get("dur", 0) for op in cuda_events])

    # compute stream
    compute_events = [
        e
        for e in trace_events
        if e.get("tid") == compute_stream and e.get("dur", 0) > 0
    ]
    sorted_compute_events = sorted(compute_events, key=lambda x: x["ts"])
    # total_compute_duration = sum([op.get("dur", 0) for op in sorted_compute_events])
    total_compute_duration = 0
    for i in range(len(sorted_compute_events)):
        event = sorted_compute_events[i]
        total_compute_duration += event.get("dur", 0)
        if i < len(sorted_compute_events) - 1:
            start_time = event["ts"] + event.get("dur", 0)
            end_time = sorted_compute_events[i + 1]["ts"]
            gap = end_time - start_time
            if gap < 500:
                total_compute_duration += gap

    # network stream
    gpu_mem_copy_events = [
        e for e in trace_events if e.get("tid") == gpu_mem_copy_stream
    ]
    total_mem_copy_duration = sum([op.get("dur", 0) for op in gpu_mem_copy_events])

    # communication stream
    communication_events = [
        e for e in trace_events if e.get("tid") == communication_stream
    ]
    total_communication_duration = sum(
        [op.get("dur", 0) for op in communication_events]
    )

    # find overheads
    if compute_overheads:
        gaps_in_compute = find_gaps_in_stream(sorted_compute_events, args.gap_threshold)
        sorted_communication_events = sorted(
            communication_events, key=lambda x: x["ts"]
        )
        gaps_in_communication = find_gaps_in_stream(
            sorted_communication_events, args.gap_threshold
        )
        sorted_cuda_events = sorted(cuda_events, key=lambda x: x["ts"])
        cuda_overhead = get_overhead_from_stream(gaps_in_compute, sorted_cuda_events)
        compute_overhead = get_overhead_from_stream(
            gaps_in_communication, sorted_compute_events
        )
        communication_overhead = get_overhead_from_stream(
            gaps_in_compute, sorted_communication_events
        )
        sorted_gpu_mem_copy_events = sorted(gpu_mem_copy_events, key=lambda x: x["ts"])
        # gaps_in_gpu_mem_copy = find_gaps_in_stream(
        #     sorted_gpu_mem_copy_events, args.gap_threshold
        # )
        network_overhead = get_overhead_from_stream(
            gaps_in_compute, sorted_gpu_mem_copy_events
        )
    else:
        cuda_overhead = 0.0
        compute_overhead = 0.0
        communication_overhead = 0.0
        network_overhead = 0.0

    return (
        cuda_overhead,
        compute_overhead,
        communication_overhead,
        network_overhead,
        (
            total_cuda_duration,
            total_compute_duration,
            total_mem_copy_duration,
            total_communication_duration,
        ),
    )


def get_profiler_path(args):
    logger = get_logger()
    # set up profiler dirs
    gpu_id = get_global_rank()
    gpu_name = get_gpu_name()
    profiler_trace_dir = args.trace_dir
    if args.trace_dir is None:
        profiler_trace_dir = os.path.join(
            os.path.expanduser("~"), "profiler_tensorboard"
        )
    experiment_name = args.experiment_name
    if args.experiment_name is None:
        if args.split_uneven:
            experiment_name = f"{args.model_name}_uneven_{args.split_uneven_partitions}"
        else:
            experiment_name = f"{args.model_name}_even"
    profiler_path = f"{profiler_trace_dir}/{experiment_name}/GPU_{gpu_id}_{gpu_name}"
    logger.info(f"Running experiment: {experiment_name}")
    return profiler_path


GPU_TFLOPs = {
    "float32": {
        "a6000": 38.7,
        "l4": 30.30,
        "p40": 11.76,
        "p100": 9.3,
        "t4": 8.1,
        "v100x16": 14.1,
        "a10g": 31.2,
        # https://www.amd.com/en/products/graphics/workstations/radeon-pro/w6800.html
        "w6800": 35.66,
    },
    "float16": {
        "h100-nvl": 835,
        "a100x40": 312,
        "a100x80": 312,
        "l40s": 362,
        "v100": 125,
        "v100x16": 112,
        "a10g": 125,
        "l4": 120,
        "t4": 65,
        "rtx 4090": 82.6,
        "p100": 18.7,
        "p40": 0.183,
        "a6000": 155,
        "w6800": 17.83,
    },
}


def compute_mfu(
    model_name,
    model,
    avg_iteration_time,
    dtype,
    seq_length,
    global_batch_size,
    mfu_gpus,
    vocab_size=None,
    image_size=None,
):
    """
    Compute Model FLOPs Utilization (MFU)

    Args:
        model_name: Name of the model
        model: The model instance
        avg_iteration_time: Average iteration time in milliseconds
        dtype: Data type for computation (e.g., 'float16', 'float32')
        seq_length: Sequence length
        global_batch_size: Global batch size
        mfu_gpus: List of GPU types to calculate available TFLOPs (required)
        vocab_size: Vocabulary size (optional)
        image_size: Image size (for vision models, optional)

    Returns:
        float: Model FLOPs Utilization as a fraction (0.0 to 1.0)
    """
    n_layers, hidden_size, _, _ = get_config_for_model(model_name)
    model = get_model(model_name, layers=1)
    n_params = get_total_model_params(model_name, model)

    # FLOPs per token due to parameters (basic multiply-accumulate FLOPs)
    flops_per_token = 2 * n_params  # MAC -> 2 FLOPs per MAC

    # FLOPs per sequence considering the sequence length
    flops_per_seq = flops_per_token * seq_length

    # Embedding layer FLOPs
    embedding_flops_per_seq = seq_length * hidden_size

    # Self-attention mechanism FLOPs (accounting for 2 FLOPs per MAC)
    attn_flops_per_seq = n_layers * (
        2 * (3 * hidden_size * seq_length)
        + 2 * (2 * (seq_length**2) * hidden_size)  # Q, K, V projections (mult by 2)
        + 2  # Attention score calculation and applying to V
        * (hidden_size * seq_length)  # Output projection
    )

    # FFN (Feed-forward network) FLOPs (accounting for 2 FLOPs per MAC)
    ffn_flops_per_seq = n_layers * (
        2
        * (
            2 * seq_length * hidden_size * 4 * hidden_size
        )  # Multiplier of 4 for the expansion in the MLP
    )

    # Output layer FLOPs (projection to vocabulary size)
    output_flops_per_seq = seq_length * hidden_size * vocab_size

    # Total FLOPs per sequence (forward pass only)
    total_flops_per_seq = (
        flops_per_seq
        + embedding_flops_per_seq
        + attn_flops_per_seq
        + ffn_flops_per_seq
        + output_flops_per_seq
    )

    if image_size is not None and "vit" in model_name:
        total_flops_per_seq += (image_size // 16) ** 2 * (3 * hidden_size * 16**2)

    # Forward and backward passes: 3x the FLOPs of the forward pass
    total_flops_per_seq_with_bwd = 3 * total_flops_per_seq

    # Calculate throughput (sequences processed per second)
    throughput = 1.0 / (avg_iteration_time / 1000)

    # Sum GPU TFLOPs across GPUs
    total_tflops = 0
    for gpu in mfu_gpus:
        gpu = gpu.lower()
        if gpu in GPU_TFLOPs.get(dtype, {}):
            total_tflops += 1e12 * GPU_TFLOPs[dtype][gpu]  # 1 TFLOP = 1e12 FLOPs

    # Compute MFU: FLOPs utilized relative to available GPU FLOPs
    mf = total_flops_per_seq_with_bwd * global_batch_size * throughput
    if total_tflops == 0:
        return 0.0
    mfu_w_attn = mf / total_tflops

    print(f"*** MF: {mf / 1e12:.2f} TFLOPs, MFU: {100 * mfu_w_attn:.2f} %")
    return mfu_w_attn


def compute_model_flops(
    model_name, seq_length, global_batch_size, model=None, param_multiplier=1
):
    _, hidden_size, _, _ = get_config_for_model(model_name)
    if model is None:
        model_stats = get_model_stats(model_name)
        n_layers = model_stats["num_layers"]
        n_params = model_stats["total_parameters"]
    else:
        n_layers = len(get_layers(model))
        # multiply by param_multiplier to account for multiple GPUs sharding params
        if hasattr(model, "_total_params"):
            # this is computed before sharding so no need to multiply by param_multiplier
            n_params = model._total_params
        else:
            n_params = get_total_model_params(model_name, model) * param_multiplier
    flops_per_token = 2 * n_params
    flops_per_seq = flops_per_token * seq_length

    # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
    attn_flops_per_seq = n_layers * 2 * 2 * (hidden_size * (seq_length**2))

    # 1 for forwards, 2 for backwards, 1 for recomputation, 0.5 for optimizer
    multipler = 4.5
    total_flops = (
        multipler * flops_per_seq + multipler * attn_flops_per_seq
    ) * global_batch_size
    return total_flops


def compute_mfu2(
    model_name,
    avg_iteration_time,
    dtype,
    seq_length,
    global_batch_size,
    mfu_gpus,
    model=None,
):
    """
    Alternative implementation of Model FLOPs Utilization (MFU) calculation

    Args:
        model_name: Name of the model
        model: The model instance
        avg_iteration_time: Average iteration time in milliseconds
        dtype: Data type for computation (e.g., 'float16', 'float32')
        seq_length: Sequence length
        global_batch_size: Global batch size
        mfu_gpus: List of GPU types to calculate available TFLOPs (required)
        vocab_size: Vocabulary size (optional)
        image_size: Image size (optional)

    Returns:
        float: Model FLOPs Utilization as a fraction (0.0 to 1.0)
    """
    if model is None:
        n_params = get_model_stats(model_name)["total_parameters"]
    else:
        n_params = get_total_model_params(model_name, model)
    throughput = 1.0 / (avg_iteration_time / 1000)
    model_flops = compute_model_flops(
        model_name, seq_length, global_batch_size, model=model
    )
    # Calculate total available TFLOPs
    total_tflops = 0.00001
    for gpu in mfu_gpus:
        total_tflops += 1e12 * GPU_TFLOPs[dtype][gpu]
    training_flops = model_flops * throughput
    training_tflops = training_flops / 1e12
    mfu_w_attn = training_flops / total_tflops
    print(f"training flops: {training_tflops}")
    print(
        f"*** avg_iter: {avg_iteration_time} n_params {n_params} MFU: {100 * mfu_w_attn:.2f} %"
    )
    return training_tflops, mfu_w_attn


def compute_mfu_per_gpu_grolar(
    model_name,
    avg_iteration_time,
    dtype,
    seq_length,
    global_batch_size,
    mfu_gpus,
    pipeline_stages,
    pg=None,
):
    pg = pg if pg is not None else torch.distributed.group.WORLD
    global_rank = get_global_rank()
    model_flops_per_gpu = [0.0 for _ in mfu_gpus]
    throughput = 1.0 / (avg_iteration_time / 1000)
    for pipeline_stage in pipeline_stages:
        model = pipeline_stage.model
        model_flops = compute_model_flops(
            model_name,
            seq_length,
            global_batch_size,
            param_multiplier=len(pipeline_stage.gpu_ranks),
            model=model,
        )
        total_microbatches = sum(pipeline_stage.num_microbatches_per_rank)
        for idx, rank in enumerate(pipeline_stage.gpu_ranks):
            if rank != global_rank:
                continue
            model_flops_per_gpu[rank] += (
                model_flops
                * pipeline_stage.num_microbatches_per_rank[idx]
                / total_microbatches
            )
    # collect model flops per gpu across GPUs
    model_flops_per_gpu = torch.tensor(model_flops_per_gpu).cuda()
    dist.all_reduce(model_flops_per_gpu, group=pg)
    model_flops_per_gpu = model_flops_per_gpu.cpu().tolist()

    mfu_per_gpu = [0.0 for _ in mfu_gpus]
    for idx, gpu in enumerate(mfu_gpus):
        mfu_per_gpu[idx] = (
            model_flops_per_gpu[idx] * throughput / (GPU_TFLOPs[dtype][gpu] * 1e12)
        )
    if is_local_leader():
        print(f"GPUs: {', '.join(mfu_gpus)}")
        print(f"MFUs: {', '.join([f'{100 * mfu:.2f}%' for mfu in mfu_per_gpu])}")

    # Calculate average MFU per GPU type
    gpu_type_mfus = {}
    for idx, gpu in enumerate(mfu_gpus):
        if gpu not in gpu_type_mfus:
            gpu_type_mfus[gpu] = []
        gpu_type_mfus[gpu].append(mfu_per_gpu[idx])

    if is_local_leader():
        print("Average MFU by GPU type:")
        for gpu_type, mfus in gpu_type_mfus.items():
            avg_mfu = sum(mfus) / len(mfus)
            print(f"{gpu_type}: {100 * avg_mfu:.2f}%")


def gather_gpu_names(pg=None):
    """
    Gather GPU indices from all processes in the distributed setup.
    Maps GPU names to integer indices for reliable gathering.

    Returns:
        list: List of GPU names from all processes
    """
    # Create a mapping of GPU names to indices
    pg = pg if pg is not None else torch.distributed.group.WORLD
    gpu_names_list = []
    for dtype in GPU_TFLOPs:
        for gpu_name in GPU_TFLOPs[dtype]:
            if gpu_name not in gpu_names_list:
                gpu_names_list.append(gpu_name)

    # Get local GPU name and find its index
    local_gpu_name = get_gpu_name().lower()
    try:
        gpu_idx = gpu_names_list.index(local_gpu_name)
    except ValueError:
        # If GPU name not found in list, use a default index (-1)
        gpu_idx = -1

    world_size = get_world_size()

    # Create tensors to hold the gathered data
    gpu_indices_tensor = torch.tensor([float(gpu_idx)])
    all_gpu_indices = torch.zeros(world_size)

    # Gather GPU indices from all processes
    dist.all_gather_into_tensor(all_gpu_indices, gpu_indices_tensor, group=pg)

    # Convert indices back to GPU names
    gpu_names = []
    for idx in all_gpu_indices.cpu().tolist():
        idx = int(idx)
        if idx >= 0 and idx < len(gpu_names_list):
            gpu_names.append(gpu_names_list[idx])
        else:
            # Use a default GPU name if index is invalid
            raise ValueError(f"Invalid GPU index: {idx}")

    return gpu_names


# analyze_trace is not super accurate
def print_metrics(
    args,
    max_allocated_mem,
    avg_iteration_time,
    profiler_path,
    compute_overheads=True,
    pipeline_stages=None,
    analyze_trace=False,
    gloo_pg=None,
):
    logger = get_logger()
    # compute max memory stats
    max_gpu_mem = (
        torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        / 1024**3
    )
    all_ranks = list(range(args.world_size))
    if gloo_pg is None:
        # use gloo pg since it works with amd + nvidia
        gloo_pg = dist.new_group(
            ranks=all_ranks, backend="gloo", timeout=datetime.timedelta(seconds=60)
        )
    max_allocated_frac = 100 * max_allocated_mem / max_gpu_mem
    allocated_in = torch.tensor([max_allocated_mem, max_allocated_frac])
    allocated_out = torch.tensor([0.0 for _ in range(2 * args.world_size)])
    dist.all_gather_into_tensor(
        allocated_out,
        allocated_in,
        group=gloo_pg,
    )
    torch.cuda.synchronize()
    allocated_out = allocated_out.reshape(args.world_size, 2)
    max_allocated_mem_gpu = allocated_out[:, 0]
    max_allocated_frac_gpu = allocated_out[:, 1]
    aggregate_memory_usage = torch.sum(max_allocated_mem_gpu).item()
    total_cluster_mem = torch.sum(
        1 / (max_allocated_frac_gpu / 100) * max_allocated_mem_gpu
    ).item()
    aggregate_memory_util = 100 * aggregate_memory_usage / total_cluster_mem
    # parse profiler logs
    max_cuda_overhead = 0
    max_compute_overhead = 0
    max_communication_overhead = 0
    max_network_overhead = 0
    if analyze_trace:
        network_overheads = []
        compute_overheads = []
        cuda_overheads = []
        cuda_durations = []
        communication_durations = []
        compute_durations = []
        network_durations = []
        file_idx = 0
        for dirpath, _, filenames in os.walk(profiler_path):
            full_paths = [os.path.join(dirpath, filename) for filename in filenames]
            if len(full_paths) == 0:
                continue
            filename = max(full_paths, key=os.path.getmtime)
            if filename.endswith("pt.trace.json"):
                file_idx += 1
                (
                    cuda_overhead,
                    compute_overhead,
                    communication_overhead,
                    network_overhead,
                    duration_tuple,
                ) = get_overheads(
                    os.path.join(dirpath, filename),
                    args,
                    compute_overheads=compute_overheads,
                )
                cuda_overheads.append(cuda_overhead)
                compute_overheads.append(compute_overhead)
                network_overheads.append(network_overhead)
                max_cuda_overhead = max(max_cuda_overhead, cuda_overhead)
                max_compute_overhead = max(max_compute_overhead, compute_overhead)
                max_communication_overhead = max(
                    max_communication_overhead, communication_overhead
                )
                max_network_overhead = max(max_network_overhead, network_overhead)
                cuda_durations.append(duration_tuple[0])
                compute_durations.append(duration_tuple[1])
                network_durations.append(duration_tuple[2])
                communication_durations.append(duration_tuple[3])
                break

        avg_compute_duration = np.mean(np.array(compute_durations)) / 1000
        avg_network_duration = np.mean(np.array(network_durations)) / 1000
        avg_comm_duration = np.mean(np.array(communication_durations)) / 1000
        compute_in = torch.tensor([avg_compute_duration.item()]).cuda()
        compute_out = torch.tensor([0.0 for _ in range(args.world_size)]).cuda()
        dist.all_gather_into_tensor(
            compute_out,
            compute_in,
            group=gloo_pg,
        )
    if hasattr(args, "flashflex_ga"):
        args.global_batch_size = args.flashflex_ga * args.global_batch_size
    # Gather GPU names for MFU calculation
    gpu_names = gather_gpu_names(gloo_pg)
    if pipeline_stages is not None:
        compute_mfu_per_gpu_grolar(
            args.model_name,
            avg_iteration_time,
            get_dtype_str(args.autocast_dtype),
            args.seq_length,
            args.global_batch_size,
            gpu_names,
            pipeline_stages,
            pg=gloo_pg,
        )
    if is_local_leader():
        global_mean_compute = torch.mean(compute_out).item() if analyze_trace else 0
        avg_gpu_utilization = (
            100 * global_mean_compute / avg_iteration_time if analyze_trace else 0
        )
        throughput = (
            args.global_batch_size * args.seq_length / (avg_iteration_time / 1000)
        )
        tflops, mfu = compute_mfu2(
            args.model_name,
            avg_iteration_time,
            get_dtype_str(args.autocast_dtype),
            args.seq_length,
            args.global_batch_size,
            gpu_names,
        )

        logger.info("============ Training Summary ===========")
        if analyze_trace:
            logger.info(
                f"Average cuda malloc/free duration : {np.mean(np.array(cuda_durations))/1000}"
            )
            logger.info(
                f"Compute overhead for communication stream : {max_compute_overhead/1000}"
            )
            logger.info(
                f"Communication overhead for compute stream: {max_communication_overhead/1000}"
            )
            logger.info(
                f"Network overhead for compute stream: {max_network_overhead/1000}"
            )
            logger.info(
                f"Average compute duration : {global_mean_compute} ({avg_gpu_utilization}%)"
            )
            logger.info(
                f"Individual compute durations: {torch.round(100 * compute_out / avg_iteration_time, decimals=2).tolist()}%"
            )
            logger.info(
                f"Average network duration : {avg_network_duration} ({100 * avg_network_duration / avg_iteration_time}%)"
            )
            logger.info(
                f"Average communication duration : {avg_comm_duration} ({100 * avg_comm_duration / avg_iteration_time}%)"
            )
        logger.info(f"Max allocated memory: {max_allocated_mem_gpu}")
        logger.info(f"Max allocated memory frac: {max_allocated_frac_gpu}")
        logger.info(f"Aggregate memory usage: {aggregate_memory_usage} GB")
        logger.info(f"Time per step (ms): {avg_iteration_time}")
        logger.info(f"Throughput: {throughput} tokens/s")
        logger.info(f"Model FLOPs Utilization (MFU): {mfu * 100:.2f}%")
        logger.info(
            "Batch Size,Seq_Len,Min_Allocated,Max_Allocated,Latency,Throughput,Avg_Compute_Utilization,Mem_Utilization,Alloc_Overhead,MFU,TFLOPs"
        )
        global_min_allocated_mem = torch.min(max_allocated_frac_gpu).item()
        global_max_allocated_mem = torch.max(max_allocated_frac_gpu).item()
        logger.info(
            f"{args.global_batch_size},"
            + f"{args.seq_length},"
            + f"{global_min_allocated_mem:.2f}%,"
            + f"{global_max_allocated_mem:.2f}%,"
            + f"{avg_iteration_time:.2f},"
            + f"{throughput:.2f},"
            + f"{avg_gpu_utilization:.2f}%,"
            + f"{aggregate_memory_util:.2f}%,"
            + f"{max_cuda_overhead:.2f},"
            + f"{mfu * 100:.2f}%,"
            + f"{tflops:.2f}"
        )
        logger.info("==========================================")
