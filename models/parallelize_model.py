# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.distributed._composable.fsdp._fsdp_api import MixedPrecisionPolicy
from torch.distributed.algorithms._checkpoint import checkpoint_wrapper
from torch.distributed._composable.replicate import replicate

from models.hub import get_layers

from utils.parallelism import ParallelDims
from utils.logger import get_logger

from torch.distributed._tensor import Replicate, Shard

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)


def apply_activation_checkpointing_v2(model: nn.Module):
    layers = get_layers(model)
    for layer in layers:
        layer = checkpoint_wrapper.CheckpointWrapper(
            layer,
            checkpoint_impl=checkpoint_wrapper.CheckpointImpl.NO_REENTRANT,
            preserve_rng_state=False,
        )


def apply_activation_checkpointing(model: nn.Module):
    layers = get_layers(model)
    for layer_id in range(len(layers)):
        layers[layer_id] = checkpoint_wrapper.CheckpointWrapper(
            layers[layer_id],
            checkpoint_impl=checkpoint_wrapper.CheckpointImpl.NO_REENTRANT,
            preserve_rng_state=False,
        )


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    autocast_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    tp_enabled: bool,
    pp_enabled: bool,
    zero2=False,
    recompute_layer=False,
):
    """
    Apply Fully Sharded Data Parallel (FSDP) to the model.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=autocast_dtype,
        output_dtype=autocast_dtype,
        reduce_dtype=reduce_dtype,
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    layers = get_layers(model)
    for idx, layer in enumerate(layers):
        if pp_enabled:
            if zero2:
                # For Pipeline Parallelism (PP), avoid resharding after forward to reduce overhead
                reshard_after_forward = False
            else:
                reshard_after_forward = True
        else:
            # Optimization: do not reshard after forward for the last transformer block
            reshard_after_forward = int(idx) < len(layers) - 1 and not zero2

        fully_shard(
            layer,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    logger = get_logger()
    logger.info(f"Model after FSDP wrap {model}")

    fully_shard(model, **fsdp_config, reshard_after_forward=True)


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
):
    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)


def apply_compile(model: nn.Module):
    """
    Note that this assumes that model is `Llama` model
    """
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = torch.compile(transformer_block, fullgraph=True)
        model.layers.register_module(layer_id, transformer_block)


def apply_tp_llama(
    model: nn.Module, tp_mesh: DeviceMesh, loss_parallel: bool, enable_async_tp: bool
):
    """
    Note that this assumes that model is `Llama` model
    """

    # Llama simple layer structure
    # embed_tokens -> nn.Embedding
    # norm -> RMSNorm
    # transformer blocks -> Block
    # lm_head -> nn.Linear

    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    # Transformer Block wraps
    for _, transformer_block in model.layers.named_children():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
            "feed_forward.w3": ColwiseParallel(),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    logger = get_logger()
    logger.info(f"Model after wraps : {model}")

    # this is copied over from the Torchtitan example
    if enable_async_tp:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)


def parallelize_model(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    args,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.
    """
    logger = get_logger()

    is_llama = "llama" in args.model_name
    # Note: The order of wrapping here might be important
    # TP -> AC -> Compile -> FSDP

    # TP
    if parallel_dims.tp_enabled and parallel_dims.tp > 1:
        tp_mesh = world_mesh["tp"]
        if is_llama:
            apply_tp_llama(
                model,
                tp_mesh,
                loss_parallel=parallel_dims.loss_parallel_enabled,
                enable_async_tp=args.async_tp,
            )
        elif "gpt" in args.model_name:
            raise Exception("GPT models are not support for TP yet")
            # apply_tp_gpt(
            #     model,
            #     tp_mesh,
            #     loss_parallel=parallel_dims.loss_parallel_enabled,
            #     enable_async_tp=args.async_tp
            # )

    # AC
    recompute_layer = args.recompute_layer
    if recompute_layer:
        if is_llama:
            apply_activation_checkpointing_v2(model)
        else:
            apply_activation_checkpointing(model)
        logger.info("Applied activation checkpointing to the model")

    # compile
    if args.compile_transformer_blocks:
        if is_llama:
            apply_compile(model)
            logger.info("Applied toch.compile to all transformer blocks")
        else:
            raise Exception("Compile not supported for non-llama type models")

    if parallel_dims.dp_enabled:
        if parallel_dims.dp_shard_enabled and args.tt_dp_shard > 1:
            if parallel_dims.dp_replicate_enabled:
                dp_mesh = world_mesh["dp_replicate", "dp_shard"]
            else:
                dp_mesh = world_mesh["dp"]

            apply_fsdp(
                model,
                dp_mesh,
                autocast_dtype=args.autocast_dtype,
                reduce_dtype=args.reduce_dtype,
                tp_enabled=parallel_dims.tp_enabled,
                pp_enabled=parallel_dims.pp_enabled,
                zero2=args.zero2,
                recompute_layer=args.recompute_layer,
            )
            if parallel_dims.dp_replicate_enabled:
                logger.info("Applied HSDP to the model")
            else:
                logger.info(f"Applied FSDP to the model : {dp_mesh}")
        else:
            apply_ddp(model, world_mesh)
            logger.info("Applied DDP to the model")
