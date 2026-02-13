# -*- coding: utf-8 -*-
from models.hub import (
    get_model,
    get_total_model_params,
    get_embedding_layer,
    get_layers,
    get_config_for_model,
)
import torch
import time
import statistics
import torch.nn as nn
import json
import numpy as np
from utils.comm import is_leader
from utils.profile import extract_kernel_runtime, get_profiler_context, fit_line
from utils.data_loader import generate_model_data
from utils.argparser_utils import get_dtype_str

# max gpu memory in gb
MAX_MEMORY_THRESHOLD = 0.75
GPU_MEMORY = {
    "watgpu": 47.99,
    "a6000": 47.99,
    "l4": 22.50,
    "p40": 22.50,
    "t4": 15.0,
    "v100": 16.0,
    "a10g": 22.49,
    "rtx_3090": 24,
    "p100": 12,
    "v100x16": 16.0,
    "v100-pciex16": 16.0,
    "v100x32": 32.0,
    "a100x40": 40.0,
    "a100x80": 80.0,
    "h100-nvl": 94.0,
}


def get_memory_stats(tag: str, skip_print=False):
    torch.cuda.synchronize()
    GiB = int(1024**3)
    allocated = torch.cuda.memory_allocated() / GiB
    max_allocated = torch.cuda.max_memory_allocated() / GiB
    max_reserved = torch.cuda.max_memory_reserved() / GiB
    cuda_malloc_retries = torch.cuda.memory_stats().get("num_alloc_retries", 0)
    memory_stats = {
        "tag": tag,
        "allocated": allocated,
        "max_allocated": max_allocated,
        "max_reserved": max_reserved,
        "cuda_malloc_retries": cuda_malloc_retries,
    }
    if not skip_print and is_leader():
        print(json.dumps(memory_stats, indent=2))
    return memory_stats


def get_checkpoint_memory(model, model_input):
    embedding_layer = get_embedding_layer(model)
    with torch.no_grad():
        embedding = embedding_layer(model_input).detach()

    checkpoint_size = embedding.numel() * 2 / 1024**3
    print(f"Embedding shape {embedding.shape} Size {checkpoint_size} GiB")
    # assumes FP16
    return checkpoint_size


def get_activation_memory(model, model_input, warmup_iters=1, mp_config=None):
    batch_size = model_input.shape[0]
    layer = get_layers(model)[0]
    embedding_layer = get_embedding_layer(model)
    dtype = mp_config.param_dtype if mp_config is not None else torch.float16
    with torch.no_grad():
        embedding = embedding_layer(model_input).detach()

    for i in range(warmup_iters + 1):
        warmup = i < warmup_iters
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_memory_stats = get_memory_stats(
            f"Pre Batch {batch_size}", skip_print=True
        )
        initial_memory = initial_memory_stats["allocated"]
        with torch.autocast(device_type="cuda", dtype=dtype):
            output = layer(embedding)
        pseudo_loss = output[0].sum()
        pseudo_loss.backward()
        memory_stats = get_memory_stats(f"Post Batch {batch_size}", skip_print=warmup)
        activation_memory = memory_stats["max_allocated"] - initial_memory
        activation_memory_reserved = memory_stats["max_reserved"] - initial_memory
        if is_leader() and not warmup:
            print(
                f"Batch Size {batch_size} Activation Memory: {activation_memory:.3f} Activation Memory Reserved: {activation_memory_reserved:.3f}"
            )

    return activation_memory, activation_memory_reserved


def get_training_state_memory(total_params, num_shards):
    data_size = 4
    # assumes parameters, gradients, optimizer state = 4x
    training_state_size = 4 * total_params * data_size / 1024**3 / num_shards
    print(f"Total parameters: {total_params} size: {total_params * 4 / 1024**3} GiB")
    return training_state_size


def estimate_memory_usage(
    model,
    total_params,
    batch_size,
    num_shards=1,
    seq_length=512,
    vocab_size=49152,
    image_size=224,
    profile_batches=5,
    mp_config=None,
):
    # TODO refactor to use config
    num_layers = model._num_layers
    layer = get_layers(model)[0]
    embedding_layer = get_embedding_layer(model)
    layer = layer.cuda()
    embedding_layer = embedding_layer.cuda()
    prev_activation_memory = 0
    prev_activation_memory_reserved = 0
    activation_memory_increase = []
    for i in range(1, profile_batches + 1):
        model_input = generate_model_data(
            model,
            batch_size=i,
            vocab_size=vocab_size,
            seq_length=seq_length,
            image_size=image_size,
        )
        activation_memory, activation_memory_reserved = get_activation_memory(
            model, model_input, mp_config=mp_config
        )
        if prev_activation_memory > 0:
            if is_leader():
                marginal_activation_memory = activation_memory - prev_activation_memory
                activation_memory_increase.append(marginal_activation_memory)
                print(f"Activation Memory Increase: {marginal_activation_memory:.3f}")
                print(
                    f"Activation Memory Reserved Increase: {activation_memory_reserved - prev_activation_memory_reserved:.3f}"
                )
        prev_activation_memory, prev_activation_memory_reserved = (
            activation_memory,
            activation_memory_reserved,
        )
    if len(activation_memory_increase) >= 5:
        start_idx = -5
    else:
        start_idx = 0
    avg_activation_memory_increase = statistics.median(
        activation_memory_increase[start_idx:]
    )

    print(f"Average Activation Memory Increase: {avg_activation_memory_increase:.3f}")
    activation_memory = avg_activation_memory_increase * batch_size
    model_input = generate_model_data(
        model,
        batch_size=1,
        vocab_size=vocab_size,
        seq_length=seq_length,
        image_size=image_size,
    )
    checkpoint_memory_increase = get_checkpoint_memory(model, model_input) * num_layers
    total_checkpoint_memory = checkpoint_memory_increase * batch_size
    training_state_memory = get_training_state_memory(total_params, num_shards)

    memory_estimate = (
        activation_memory + total_checkpoint_memory + training_state_memory
    )
    print(f"Memory Estimate: {memory_estimate:.3f} GiB")
    return (
        memory_estimate,
        avg_activation_memory_increase,
        checkpoint_memory_increase,
        total_checkpoint_memory,
        training_state_memory,
    )


def get_compute_latency(
    model,
    batch_size=1,
    num_iterations=15,
    seq_length=512,
    vocab_size=49152,
    image_size=224,
    delay=0.05,
    trace_dir=None,
    mp_config=None,
):
    layer = get_layers(model)[0].cuda()
    embedding_layer = get_embedding_layer(model).cuda()
    input_ids = generate_model_data(
        model,
        batch_size=batch_size,
        vocab_size=vocab_size,
        seq_length=seq_length,
        image_size=image_size,
    ).cuda()
    dtype = mp_config.param_dtype if mp_config is not None else torch.float32

    with torch.autocast(device_type="cuda", dtype=dtype):
        embedding = embedding_layer(input_ids).detach()
    embedding = embedding.to(dtype)
    embedding.requires_grad = True
    # Forward pass timing
    trace_dir_forwards = trace_dir + f"/forwards/b{batch_size}/"
    profiler_ctx = get_profiler_context(out_dir=trace_dir_forwards)
    with profiler_ctx:
        with torch.autocast(device_type="cuda", dtype=dtype):
            for _ in range(num_iterations):
                output = layer(embedding)
                # Wait for GPU sync
                torch.cuda.synchronize()
                time.sleep(delay)

    # Record forward pass time
    avg_forwards_time = extract_kernel_runtime(
        num_iterations, trace_dir=trace_dir_forwards
    )
    print("--------------------")
    print(
        f"Profiled forwards pass, batch size: {batch_size}, runtime: {avg_forwards_time:.3f}"
    )

    if not isinstance(output, torch.Tensor):
        output = output[0]
    grads = torch.rand(output.shape, device=output.device, dtype=dtype)
    trace_dir_backwards = trace_dir + f"/backwards/b{batch_size}/"
    profiler_ctx = get_profiler_context(out_dir=trace_dir_backwards)
    with profiler_ctx:
        for _ in range(num_iterations):
            output.backward(grads, retain_graph=True)
            # Wait for GPU sync
            torch.cuda.synchronize()
            time.sleep(delay)

    # Record forward pass time
    avg_backwards_time = extract_kernel_runtime(
        num_iterations, trace_dir=trace_dir_backwards
    )
    print(
        f"Profiling backwards pass, batch size: {batch_size}, runtime: {avg_backwards_time:.3f}"
    )

    return avg_forwards_time, avg_backwards_time


def get_optimizer_latency(model, num_shards=1, num_iterations=50):
    print("Profiling optimizer step")
    layer_params = []
    layer = get_layers(model)[0]
    for param in layer.parameters():
        param = param.cuda()
        param_slice = nn.Parameter(param[: param.shape[0] // num_shards])
        param_slice.grad = torch.rand(*param_slice.shape, device=param.device)
        layer_params.append(param_slice)
    optimizer = torch.optim.Adam(
        layer_params,
        fused=True,
    )
    profiler_ctx = get_profiler_context()
    torch.cuda.synchronize()
    with profiler_ctx:
        for _ in range(num_iterations):
            optimizer.step()
        torch.cuda.synchronize()

    avg_optimizer_time = extract_kernel_runtime(num_iterations)
    print(f"Average Optimizer Pass Time: {avg_optimizer_time} ms")

    return avg_optimizer_time


class LatencyEstimator:
    def __init__(
        self,
        model_name,
        gpu_type="watgpu",
        mp_config=None,
        num_shards=1,
        seq_length=512,
        vocab_size=49152,
        image_size=224,
        max_nodes=1,
        max_gpus_per_node=8,
        profile_batches=5,
        network_model_path="network_model.pkl",
        trace_dir=None,
        profile_compute=True,
    ):
        model = get_model(
            model_name,
            vocab_size=vocab_size,
            seq_length=seq_length,
            image_size=image_size,
            layers=1,
            dtype=mp_config.param_dtype if mp_config is not None else torch.float32,
        )
        self.trainable_parameters = get_total_model_params(model_name, model)
        num_layers, _, _, _ = get_config_for_model(model_name)

        self.seq_length = seq_length
        self.model = model
        self.gpu_type = gpu_type
        self.mp_config = mp_config

        # profile memory usage
        self.opt_times_per_world_size = {}
        for shards in range(1, max_nodes * max_gpus_per_node + 1):
            opt_time = get_optimizer_latency(model, num_shards=shards)
            self.opt_times_per_world_size[shards] = opt_time
        (
            _,
            self.avg_activation_memory_increase,
            self.checkpoint_memory_increase,
            self.total_checkpoint_memory,
            _,
        ) = estimate_memory_usage(
            model,
            self.trainable_parameters,
            1,
            num_shards=num_shards,
            seq_length=seq_length,
            vocab_size=vocab_size,
            image_size=image_size,
            profile_batches=profile_batches,
            mp_config=mp_config,
        )

        # with open(network_model_path, "rb") as file:
        #    network_models = pickle.load(file)
        # self.network_models = network_models
        self.model = model
        self.num_layers = num_layers

        if profile_compute:
            # profile compute
            forwards_times = []
            backwards_times = []
            combined_times = []
            for bs in range(1, profile_batches + 1):
                try:
                    forwards_time, backwards_time = get_compute_latency(
                        model,
                        batch_size=bs,
                        seq_length=seq_length,
                        vocab_size=vocab_size,
                        image_size=image_size,
                        trace_dir=trace_dir,
                        mp_config=mp_config,
                    )
                    forwards_times.append(forwards_time)
                    backwards_times.append(backwards_time)
                    combined_times.append(forwards_time + backwards_time)
                # catch any error
                except Exception as e:
                    print(f"Caught CUDA OOM: {e}")
                    continue
            self.forwards_time = (forwards_times[-1] - forwards_times[0]) / (
                profile_batches - 1
            )
            self.backwards_time = (backwards_times[-1] - backwards_times[0]) / (
                profile_batches - 1
            )
            self.combined_time = self.forwards_time + self.backwards_time
            slope_forward, intercept_forward = fit_line(forwards_times)
            slope_backward, intercept_backward = fit_line(backwards_times)
            print(
                f"Forwards Time: {forwards_times} ms\n Backwards Time: {backwards_times} ms"
            )
            print(
                f"Avg (1S) Forwards Time: {self.forwards_time} ms Backwards Time: {self.backwards_time} ms"
            )
            print(
                f"Slope Forward: {slope_forward} Intercept Forward: {intercept_forward}"
            )
            print(
                f"Slope Backward: {slope_backward} Intercept Backward: {intercept_backward}"
            )
            # update model_latencies.json
            dtype = (
                get_dtype_str(mp_config.param_dtype)
                if mp_config is not None
                else "float32"
            )
            with open("data/model_latencies.json", "r") as file:
                model_latencies = json.load(file)
                if model_name not in model_latencies:
                    model_latencies[model_name] = {}
                if gpu_type not in model_latencies[model_name]:
                    model_latencies[model_name][gpu_type] = {}
                seq_length_str = str(seq_length)
                if seq_length_str not in model_latencies[model_name][gpu_type]:
                    model_latencies[model_name][gpu_type][seq_length_str] = {}

                model_latencies[model_name][gpu_type][seq_length_str][dtype] = [
                    (slope_forward, intercept_forward),
                    (slope_backward, intercept_backward),
                    forwards_times,
                    backwards_times,
                ]

            with open("data/model_latencies.json", "w") as file:
                json.dump(
                    model_latencies, file, ensure_ascii=False, indent=4, sort_keys=True
                )

    def predict_runtime(self, num_shards, batch_size):
        layers = get_layers(self.model)
        total_layer_params = sum(
            p.numel() for p in layers[0].parameters() if p.requires_grad
        )
        layer_gb = total_layer_params * 4 / 1024**3
        print(
            f"Layer parameters: {total_layer_params}, size: {layer_gb}, FS size: {layer_gb / num_shards} GiB"
        )
        layer_comm_size_mb = layer_gb * 1024
        ag_size_mb = (
            layer_comm_size_mb
            if self.mp_config and self.mp_config.param_dtype == torch.float32
            else layer_comm_size_mb / 2
        )
        rs_size_mb = (
            layer_comm_size_mb
            if self.mp_config and self.mp_config.reduce_dtype == torch.float32
            else layer_comm_size_mb / 2
        )
        # TODO: Fix fact that linear model can yield negative latency
        ag_latency = self.network_models[self.gpu_type][(1, num_shards)][
            "global_all_gather"
        ].predict(np.array([[ag_size_mb]]))
        rs_latency = self.network_models[self.gpu_type][(1, num_shards)][
            "global_reduce_scatter"
        ].predict(np.array([[rs_size_mb]]))

        total_forwards_time = (
            max(ag_latency, self.forwards_time * batch_size) * self.num_layers
        )
        total_backwards_time = (
            max(ag_latency + rs_latency, self.backwards_time * batch_size)
            * self.num_layers
        )
        total_opt_time = self.opt_times_per_world_size[num_shards] * self.num_layers

        est_runtime = total_forwards_time + total_backwards_time + total_opt_time

        activation_memory = self.avg_activation_memory_increase * batch_size
        training_state_memory = get_training_state_memory(
            self.trainable_parameters, num_shards
        )

        est_memory = (
            activation_memory + self.total_checkpoint_memory + training_state_memory
        )

        tokens_per_sec = num_shards * batch_size * self.seq_length / est_runtime * 1000

        return tokens_per_sec, est_runtime, est_memory
