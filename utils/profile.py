# -*- coding: utf-8 -*-
import json
import os
import torch
import glob
import numpy as np
import contextlib

try:
    from utils.comm import is_local_leader, get_global_rank
except:
    # needed for flashflex
    from groler.utils.comm import is_local_leader, get_global_rank


def print_memory_stats(tag: str, skip=False, all_ranks=False):
    if skip or (not is_local_leader() and not all_ranks):
        return {
            "tag": tag,
            "allocated": 0,
            "max_allocated": 0,
            "max_reserved": 0,
            "cuda_malloc_retries": 0,
        }
    torch.cuda.synchronize()
    GiB = int(1024**3)
    max_memory = (
        torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / GiB
    )
    allocated = torch.cuda.memory_allocated() / GiB
    max_allocated = torch.cuda.max_memory_allocated() / GiB
    max_reserved = torch.cuda.max_memory_reserved() / GiB
    cuda_malloc_retries = torch.cuda.memory_stats().get("num_alloc_retries", 0)
    print(
        f"rank {get_global_rank()} {tag}: "
        f"{allocated:.2f} GiB allocated ({allocated / max_memory * 100:.2f}%), "
        f"{max_allocated:.2f} GiB max allocated ({max_allocated / max_memory * 100:.2f}%), "
        f"{max_reserved:.2f} GiB max reserved ({max_reserved / max_memory * 100:.2f}%), "
        f"{cuda_malloc_retries} cuda malloc retries"
    )

    return {
        "tag": tag,
        "allocated": allocated,
        "max_allocated": max_allocated,
        "max_reserved": max_reserved,
        "cuda_malloc_retries": cuda_malloc_retries,
    }


def get_profiler_context(out_dir=None, detailed_trace=False, unique_gpus_only=False):
    # if unique_gpus_only, skip profiling if not the first GPU of its type on this machine
    device_name = torch.cuda.get_device_name()
    device_id = torch.cuda.current_device()

    # Get list of all GPU device names on this machine
    device_names = []
    for i in range(torch.cuda.device_count()):
        device_names.append(torch.cuda.get_device_name(i))

    # Check if this is the first GPU of its type
    if device_names.index(device_name) != device_id and unique_gpus_only:
        return contextlib.nullcontext()

    if out_dir is None:
        out_dir = os.path.join(os.path.expanduser("~"), "profiler_tensorboard")
    os.makedirs(out_dir, exist_ok=True)

    handler = torch.profiler.tensorboard_trace_handler(out_dir)
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    return torch.profiler.profile(
        activities=activities,
        on_trace_ready=handler,
        profile_memory=detailed_trace,
        record_shapes=detailed_trace,
        with_stack=detailed_trace,
        with_modules=detailed_trace,
    )


def get_profiler_trace_path(trace_dir=None):
    if trace_dir is None:
        log_dir = os.path.join(os.path.expanduser("~"), "profiler_tensorboard/*")
    else:
        log_dir = os.path.join(trace_dir, "*")
    files = glob.glob(log_dir)
    # assume we are fetching most recent trace
    profiler_trace = max(files, key=os.path.getmtime)

    return profiler_trace


def extract_kernel_runtime(num_iterations=1, trace_dir=None):
    """
    Extracts the average duration of kernel operations from a given trace file.
    """
    file_path = get_profiler_trace_path(trace_dir=trace_dir)
    with open(file_path, "r") as file:
        profiling_data = json.load(file)

    # Filter out "Memcpy DtoH (Device -> Pinned)" events and calculate durations
    event_count = {}
    total_duration = 0
    for event in profiling_data["traceEvents"]:
        if event.get("cat") != "kernel":
            continue
        event_name = event.get("name")
        if event_name not in event_count:
            event_count[event_name] = 0
        event_count[event_name] += 1
        total_duration += (
            event.get("dur", 0) / 1000
        )  # Convert from microseconds to milliseconds

    # Calculate average duration if there are any durations collected
    avg_duration = total_duration / num_iterations

    return avg_duration


def fit_line(y_values):
    # Generate x values as sequential integers starting from 0
    x_values = np.arange(len(y_values)) + 1
    # Perform linear regression to find the best fitting line
    slope, intercept = np.polyfit(x_values, y_values, 1)

    return slope, intercept


def extract_memcpy_runtime(num_iterations=1, trace_dir=None):
    """
    Extracts the average duration of kernel operations from a given trace file.
    """
    file_path = get_profiler_trace_path(trace_dir=trace_dir)
    with open(file_path, "r") as file:
        profiling_data = json.load(file)

    # Filter out "Memcpy DtoH (Device -> Pinned)" events and calculate durations
    event_count = {}
    total_duration = 0
    for event in profiling_data["traceEvents"]:
        if event.get("cat") != "gpu_memcpy":
            continue
        event_name = event.get("name")
        if event_name not in event_count:
            event_count[event_name] = 0
        event_count[event_name] += 1
        total_duration += (
            event.get("dur", 0) / 1000
        )  # Convert from microseconds to milliseconds

    # Calculate average duration if there are any durations collected
    avg_duration = total_duration / num_iterations

    return avg_duration


def extract_function_runtime(name, num_iterations=1, trace_dir=None):
    """
    Extracts the average duration of a given function from a given trace file.
    """
    file_path = get_profiler_trace_path(trace_dir=trace_dir)
    with open(file_path, "r") as file:
        profiling_data = json.load(file)

    # Filter out "Memcpy DtoH (Device -> Pinned)" events and calculate durations
    event_count = {}
    total_duration = 0
    for event in profiling_data["traceEvents"]:
        if event.get("name") != name:
            continue
        total_duration += (
            event.get("dur", 0) / 1000
        )  # Convert from microseconds to milliseconds

    # Calculate average duration if there are any durations collected
    avg_duration = total_duration / num_iterations

    return avg_duration
