# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from models.hub import (
    get_layers,
    get_model,
    get_total_model_params,
    is_llama_model,
    wrap_other_layers,
    is_vision_model,
)
from utils.global_state import get_split_state, set_split_state
from utils.comm import (
    is_leader,
    is_local_leader,
    get_shard_process_group,
    get_replicate_process_group,
)
from torch.distributed import fsdp
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp._init_utils import (
    _get_default_group,
)
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    wrap,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.algorithms._checkpoint import checkpoint_wrapper
import torch.distributed.fsdp._traversal_utils as traversal_utils


def model_init(args):
    """
    Initialize model and wrap
    """
    if is_vision_model(args.model_name):
        model = get_model(args.model_name, image_size=args.image_size)
    else:
        model = get_model(
            args.model_name, vocab_size=args.vocab_size, seq_length=args.seq_length
        )

    if is_leader():
        print("Model initialized")

    # count total parameters in model
    total_model_params = get_total_model_params(args.model_name, model)

    # set global state for total params in model
    set_split_state(key="total_model_params", value=total_model_params)

    split_state = get_split_state()
    # set global state for sharding ratios
    set_split_state(key="shard_ratio", value=[0] * len(split_state["model_partitions"]))

    # set global state for rank mask
    set_split_state(
        key="rank_mask",
        value=[
            1 if partition_size > 0.0 else 0
            for partition_size in split_state["model_partitions"]
        ],
    )

    mp_config = fsdp.MixedPrecision(
        param_dtype=args.autocast_dtype,
        buffer_dtype=args.autocast_dtype,
        reduce_dtype=args.reduce_dtype,
    )
    if args.no_shard:
        default_shard_strategy = ShardingStrategy.NO_SHARD
    elif args.hybrid_shard:
        default_shard_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        default_shard_strategy = ShardingStrategy.FULL_SHARD
    world_group = _get_default_group()
    if args.hybrid_shard:
        intra_pg = get_shard_process_group(hybrid_shard_groups=args.hybrid_shard_groups)
        inter_pg = get_replicate_process_group(hybrid_shard_groups=args.hybrid_shard_groups)
        if args.reverse_hs:
            hybrid_pg = (inter_pg, intra_pg)
        else:
            hybrid_pg = (intra_pg, inter_pg)
        default_process_group = hybrid_pg
    else:
        default_process_group = world_group
    wrapper_kwargs = {
        "sharding_strategy": default_shard_strategy,
        "limit_all_gathers": True,
        "forward_prefetch": True,
        "mixed_precision": mp_config,
        "cpu_offload": fsdp.CPUOffload(args.cpu_offload),
        "device_id": torch.cuda.current_device(),
        "use_orig_params": args.compile,
        "process_group": default_process_group,
    }
    with enable_wrap(wrapper_cls=FSDP, **wrapper_kwargs):
        layers = get_layers(model)  # get layers to wrap

        # Wrap model layers
        for i in range(len(layers)):
            if is_local_leader():
                total_layer_params = sum(
                    p.numel() for p in layers[i].parameters() if p.requires_grad
                )
                layer_gb = total_layer_params * 4 / 1024**3
                print(
                    f"Layer {i} parameters: {total_layer_params}, size: {layer_gb}, FS size: {layer_gb / args.world_size} GiB"
                )
            if i < args.recompute_layers:
                if args.recompute_layer:
                    layers[i] = checkpoint_wrapper.CheckpointWrapper(
                        layers[i],
                        checkpoint_impl=checkpoint_wrapper.CheckpointImpl.NO_REENTRANT,
                        preserve_rng_state=False,
                    )
                else:
                    if args.recompute_attention:
                        # refactor
                        if is_llama_model(args.model_name):
                            layers[i].self_attn = checkpoint_wrapper.CheckpointWrapper(
                                layers[i].self_attn,
                                checkpoint_impl=checkpoint_wrapper.CheckpointImpl.NO_REENTRANT,
                                preserve_rng_state=False,
                            )
                        else:
                            layers[i].attn = checkpoint_wrapper.CheckpointWrapper(
                                layers[i].attn,
                                checkpoint_impl=checkpoint_wrapper.CheckpointImpl.NO_REENTRANT,
                                preserve_rng_state=False,
                            )
                    if args.recompute_feed_forward:
                        layers[i].mlp = checkpoint_wrapper.CheckpointWrapper(
                            layers[i].mlp,
                            checkpoint_impl=checkpoint_wrapper.CheckpointImpl.NO_REENTRANT,
                            preserve_rng_state=False,
                        )
            sharding_strategy = (
                default_shard_strategy
                if i < args.full_shard_layers
                else ShardingStrategy.SHARD_GRAD_OP
            )
            process_group = (
                hybrid_pg
                if sharding_strategy == ShardingStrategy.HYBRID_SHARD
                else world_group
            )
            layers[i] = wrap(
                layers[i],
                sharding_strategy=sharding_strategy,
                process_group=process_group,
            )

        # Wrap remaining model
        wrap_other_layers(args.model_name, model)
        model = wrap(model)

    if args.optimizer_in_backwards:
        fsdp_states, fsdp_modules = traversal_utils._get_fsdp_states_with_modules(model)
        param_set = set()
        for _, (_, fsdp_module) in enumerate(
            zip(reversed(fsdp_states), reversed(fsdp_modules))
        ):
            fsdp_module_params = fsdp_module.parameters()
            # comm_hook_state = fsdp_state._communication_hook_state
            optimizer_params = []
            comm_hook_state = {}

            for param in fsdp_module_params:
                if param in param_set:
                    # Param already added to previous module, skipping
                    continue
                with torch.no_grad():
                    param_shard_tensor = param.data
                    param_shard = nn.Parameter(data=param_shard_tensor)
                    optimizer_params.append(param_shard)
                param_set.add(param)

            if len(optimizer_params) == 0:
                continue

            # attach optimizer to communication hook state
            optimizer = torch.optim.Adam(
                optimizer_params,
                fused=not (args.no_fused_optimizer or args.cpu_offload),
            )
            comm_hook_state["optimizer"] = optimizer
            fsdp_module._comm_hook_state = comm_hook_state

    model._total_params = total_model_params
    model._model_name = args.model_name
    if is_local_leader():
        print(
            f"Total parameters: {total_model_params} size: {total_model_params * 4 / 1024**3} GiB"
        )
        print(model)

    if args.compile:
        model = torch.compile(model)

    return model
