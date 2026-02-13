# -*- coding: utf-8 -*-


import copy
from typing import Tuple

import torch
from models.hub import (
    get_config_for_model,
    get_layers,
    get_model,
    is_vision_model,
    replace_layers,
)
from utils.logger import get_logger
from torch.distributed.pipelining import PipelineStage

from torch.distributed.pipelining import (
    Schedule1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
)

logger = get_logger()


def stage_ids_this_rank(
    pp_rank: int, pp_size: int, num_stages: int, style: str = "loop"
) -> Tuple[int]:
    """
    Compute the stage ids for the stages that will run on this pp rank for either a looped or V style schedule.
    """
    assert (
        num_stages % pp_size == 0
    ), f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size}"
    stages_per_rank = num_stages // pp_size
    if style == "loop":
        return tuple(pp_rank + s * pp_size for s in range(stages_per_rank))
    elif style == "v":
        assert (
            stages_per_rank == 2
        ), f"v schedules assume 2 stages per rank, got {stages_per_rank}"
        stage_v_pairs = list(
            zip(range(pp_size), range(num_stages - 1, pp_size - 1, -1))
        )
        return stage_v_pairs[pp_rank]
    else:
        raise ValueError(f"Unknown schedule style: {style}")


def build_pipeline_schedule(args, stages, loss_fn):
    """
    Build the pipeline schedule based on the provided arguments.
    """
    looped_schedule = False
    schedule = args.tt_pp_schedule

    if schedule == "1f1b":
        schedule_class = Schedule1F1B
    elif schedule == "gpipe":
        schedule_class = ScheduleGPipe
    elif schedule == "interleaved_1f1b":
        schedule_class = ScheduleInterleaved1F1B
        looped_schedule = True
    else:
        raise NotImplementedError(f"{schedule} is not implemented")

    logger.info(f"Using pipeline schedule {schedule}")

    n_microbatches = args.tt_pp_microbatches
    if n_microbatches is None:
        n_microbatches = args.tt_pp

    return schedule_class(
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
    )


def calculate_vit_seq_length(image_size, patch_size):
    num_patches = (image_size // patch_size) ** 2
    return num_patches + 1


def pipeline_model(model, pp_mesh, parallel_dims, device, loss_fn, args):
    """
    Split the model into pipeline stages and create a pipeline schedule.
    """
    # Pipeline configuration
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()
    tp_size = parallel_dims.tp
    microbatches = args.tt_pp_microbatches or parallel_dims.pp
    splits = args.tt_pp_split_points
    num_stages = len(splits) + 1
    stage_ids = stage_ids_this_rank(pp_rank, pp_size, num_stages, style="loop")
    logger.info(f"Rank : {pp_rank}, stage_ids : {stage_ids}", leader_only=False)

    def _build_stage(stage_idx, start_layer, stop_layer, is_first=False, is_last=False):
        """
        Build a single pipeline stage by copying relevant layers.
        """
        num_transformer_layers, hidden_size, _, _ = get_config_for_model(
            args.model_name
        )
        if args.use_deepcopy_for_build:
            model_for_stage = build_model_for_stage_v1(
                model, start_layer, stop_layer, is_first=is_first, is_last=is_last
            )
        else:
            model_for_stage = build_model_for_stage_v2(
                num_transformer_layers,
                start_layer,
                stop_layer,
                is_first=is_first,
                is_last=is_last,
                args=args,
            )

        logger.info(
            f"Num layers on rank : {pp_rank} : {len(get_layers(model_for_stage))}"
        )
        mp_dtype = args.autocast_dtype
        batch_size = args.batch_size
        vocab_size = args.vocab_size
        hidden_size = hidden_size
        num_labels = 10  # for ViT

        if is_vision_model(args.model_name):
            seq_length = calculate_vit_seq_length(
                args.image_size, model_for_stage._patch_size
            )
        else:
            seq_length = args.seq_length
        local_seq_len = int(seq_length // tp_size)

        layers_io_shape = (batch_size, local_seq_len, hidden_size)
        if is_vision_model(args.model_name):
            output_layer_shape = (batch_size, seq_length, num_labels)
        else:
            output_layer_shape = (batch_size, seq_length, vocab_size)

        if is_first:
            if is_vision_model(args.model_name):
                input_shape = (batch_size, 3, args.image_size, args.image_size)
                input = torch.randn(input_shape, dtype=mp_dtype, device=device)
            else:
                # For text models, input shape is (batch_size, seq_length) of ints
                input_shape = (batch_size, args.seq_length)
                input = torch.randint(0, vocab_size, input_shape, device=device)
        else:
            input = torch.randn(layers_io_shape, dtype=mp_dtype, device=device)

        if is_last:
            output = torch.randn(output_layer_shape, dtype=mp_dtype, device=device)
        else:
            output = torch.randn(layers_io_shape, dtype=mp_dtype, device=device)

        print(f"Module input shape: {input.chunk(microbatches)[0].shape}")
        model_for_stage.to_empty(device=device)
        # TODO: If we upgrade to newer version of pytorch we don't need this shape inference input_args/output_args
        # https://github.com/pytorch/torchtitan/commit/1629fb976566e5e6f592962890532eb492ddb828
        stage = PipelineStage(
            model_for_stage,
            stage_idx,
            num_stages,
            device,
            input_args=input.chunk(microbatches)[0],
            output_args=output.chunk(microbatches)[0],
            group=pp_mesh.get_group("pp"),
        )
        return stage, model_for_stage

    stages = []
    model_parts = []
    logger.info(f"Splits : {splits}", leader_only=False)
    for stage_idx in stage_ids:
        start_layer = splits[stage_idx - 1] if stage_idx > 0 else None
        stop_layer = splits[stage_idx] if stage_idx < num_stages - 1 else None
        is_first = stage_idx == 0
        is_last = stage_idx == num_stages - 1
        stage, model_chunk = _build_stage(
            stage_idx,
            start_layer,
            stop_layer,
            is_first=is_first,
            is_last=is_last,
        )
        logger.info(
            f"PP rank {pp_rank} is building stage_idx {stage_idx} "
            f"with start_layer {start_layer}, stop_layer {stop_layer}: model chunk \n{model_chunk}",
            leader_only=False,
        )
        stages.append(stage)
        model_parts.append(model_chunk)

    pp_schedule = build_pipeline_schedule(args, stages, loss_fn)

    return pp_schedule, model_parts


def build_model_for_stage_v1(model, start_layer, stop_layer, is_first, is_last):
    model_copy = copy.deepcopy(model)
    if not is_first:
        # Remove embedding layers from non-first stages
        # TODO: refactor/clean this logic
        model_copy.embedding = None  # gpt simple
        model_copy.embeddings = None  # Vit Simple
        model_copy.embed_tokens = None  # llama_simple
        model_copy.tok_embeddings = None  # llama_v2

    # Get all layers from the model's ModuleList
    all_layers = list(get_layers(model_copy))
    logger.info(f"Total layers before removal: {len(all_layers)}")

    # Determine the range of layers to keep
    if start_layer is None:
        start_idx = 0
    else:
        start_idx = int(start_layer.split(".")[1])  # e.g., 'layers.3' -> 3

    if stop_layer is None:
        stop_idx = len(all_layers)
    else:
        stop_idx = int(stop_layer.split(".")[1]) + 1  # Include the stop layer

    logger.info(f"Keeping layers from index {start_idx} to {stop_idx - 1}")

    # Slice the layers to keep
    kept_layers = all_layers[start_idx:stop_idx]
    dropped_layers = all_layers[:start_idx] + all_layers[stop_idx:]

    # Log which layers are being dropped
    for layer in dropped_layers:
        layer_full_name = f"layers.{all_layers.index(layer)}"
        logger.info(f"Dropping {layer_full_name}")

    # Reassign the ModuleList with only the kept layers
    new_layers = torch.nn.ModuleList(kept_layers)
    replace_layers(model_copy, new_layers)

    if not is_last:
        # TODO: refactor/clean this logic
        model_copy.ln_f = None
        model_copy.lm_head = None
        model_copy.output_layer = None
        model_copy.cls = None  # Bert

    return model_copy


def build_model_for_stage_v2(
    num_transformer_layers, start_layer, stop_layer, is_first, is_last, args
):
    if start_layer is None:
        start_idx = 0
    else:
        start_idx = int(start_layer.split(".")[1])  # e.g., 'layers.3' -> 3

    if stop_layer is None:
        stop_idx = num_transformer_layers
    else:
        stop_idx = int(stop_layer.split(".")[1])

    num_transformer_layers_for_stage = stop_idx - start_idx

    if is_vision_model(args.model_name):
        model_stage = get_model(
            args.model_name,
            image_size=args.image_size,
            layers=num_transformer_layers_for_stage,
        )
    else:
        model_stage = get_model(
            args.model_name,
            vocab_size=args.vocab_size,
            seq_length=args.seq_length,
            layers=num_transformer_layers_for_stage,
            dtype=args.autocast_dtype,
        )

    if not is_first:
        # Remove embedding layers from non-first stages
        # TODO: refactor/clean this logic
        model_stage.embedding = None  # gpt simple
        model_stage.embeddings = None  # bert
        model_stage.embed_tokens = None  # llama_simple
        model_stage.tok_embeddings = None  # llama_v2

    if not is_last:
        # TODO: refactor/clean this logic
        model_stage.ln_f = None
        model_stage.lm_head = None
        model_stage.output_layer = None
        model_stage.cls = None  # Bert

    return model_stage
