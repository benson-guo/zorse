# -*- coding: utf-8 -*-
from typing import Dict
import json

from grolar_optimizer.optimizer import ModelLatencies


# returns 1 layer forward + backward latencies for given model, dtype, batch_sze
# per gpu based on profiled linear model
def load_model_latencies_per_gpu(
    model_name, dtype, batch_size, seq_length
) -> Dict[str, float]:
    with open("data/model_latencies.json", "r") as f:
        model_latencies = json.load(f)

    compute_times = {}
    for gpu in model_latencies[model_name].keys():
        model_gpu_latencies = model_latencies[model_name][gpu]
        if (
            seq_length not in model_gpu_latencies
            or dtype not in model_gpu_latencies[seq_length]
        ):
            continue
        runtime = model_gpu_latencies[seq_length][dtype]

        # For smaller microbatches we have profiled, use runtime directly
        if isinstance(runtime, list) and len(runtime) == 4:
            runtimes_forwards = runtime[2]
            runtimes_backwards = runtime[3]
        else:
            runtimes_forwards = []
            runtimes_backwards = []
        # For larger microbatches we have not profiled, extrapolate with linear model
        if isinstance(runtime, list):
            if isinstance(runtime[0], list):
                forward_slope, forward_intercept = runtime[0]
                backward_slope, backward_intercept = runtime[1]
            else:
                forward_slope, forward_intercept = runtime[0]
                backward_slope, backward_intercept = 0, 0
        else:
            forward_slope, forward_intercept = runtime, 0
            backward_slope, backward_intercept = 0, 0
        runtime_slope = forward_slope + backward_slope
        runtime_intercept = forward_intercept + backward_intercept

        if len(runtimes_backwards) >= batch_size:
            mb_runtime = (
                runtimes_forwards[batch_size - 1] + runtimes_backwards[batch_size - 1]
            )
        else:
            mb_runtime = runtime_slope * batch_size + runtime_intercept
        compute_times[gpu] = mb_runtime

    print(f"compute_times {compute_times} for batch size {batch_size}")
    return compute_times


def load_model_compute_latencies_per_gpu(
    model_name, autocast_dtype, batch_size, seq_length
) -> Dict[str, ModelLatencies]:
    with open("data/model_latencies.json", "r") as f:
        model_latencies = json.load(f)

    compute_times = {}
    use_linear_model = True
    for gpu in model_latencies[model_name].keys():
        model_gpu_latencies = model_latencies[model_name][gpu]
        if (
            seq_length not in model_gpu_latencies
            or autocast_dtype not in model_gpu_latencies[seq_length]
        ):
            continue
        runtime = model_gpu_latencies[seq_length][autocast_dtype]

        # Initialize forward and backward times
        forward_time = 0
        backward_time = 0

        # For smaller microbatches we have profiled, use runtime directly
        if isinstance(runtime, list) and len(runtime) == 4:
            runtimes_forwards = runtime[2]
            runtimes_backwards = runtime[3]

            # Get profiled time if available for this batch size
            if (
                len(runtimes_forwards) >= batch_size
                and len(runtimes_backwards) >= batch_size
            ):
                forward_time = runtimes_forwards[batch_size - 1]
                backward_time = runtimes_backwards[batch_size - 1]
                use_linear_model = False
            else:
                # Need to extrapolate using linear model
                use_linear_model = True
        else:
            use_linear_model = True
            runtimes_forwards = []
            runtimes_backwards = []

        # For larger microbatches we have not profiled, extrapolate with linear model
        if use_linear_model:
            if isinstance(runtime, list):
                if isinstance(runtime[0], list):
                    forward_slope, forward_intercept = runtime[0]
                    backward_slope, backward_intercept = runtime[1]
                else:
                    forward_slope, forward_intercept = runtime[0]
                    backward_slope, backward_intercept = 0, 0
            else:
                forward_slope, forward_intercept = runtime, 0
                backward_slope, backward_intercept = 0, 0

            # Calculate using linear model
            forward_time = forward_slope * batch_size + forward_intercept
            backward_time = backward_slope * batch_size + backward_intercept

        # Create ModelLatencies named tuple with all values
        compute_times[gpu] = ModelLatencies(
            forward=forward_time,
            backward=backward_time,
            total=forward_time + backward_time,
        )

    return compute_times
