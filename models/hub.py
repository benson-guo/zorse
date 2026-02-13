# -*- coding: utf-8 -*-
from models.gpt import get_model as get_gpt_model
from models.llama import get_model as get_llama_model
from models.bert import get_model as get_bert_model
from models.opt import get_model as get_opt_model
from models.vit import get_model as get_vit_model
from models.vit_simple import get_model as get_vit_deepspeed_model
from models.bert_simple import get_model as get_bert_deepspeed_model
from models.gpt_simple import (
    get_model as get_gpt_deepspeed_model,
    get_toy_model as get_gpt_toy_model,
)
from models.llama_simple import get_model as get_llama_deepspeed_model
from models.llama_v2 import get_model as get_llama_v2_deepspeed_model
from models.mistral import get_model as get_mistral_model
from models.mistral_simple import get_model as get_mistral_deepspeed_model
from torch.distributed.fsdp.wrap import (
    wrap,
)
import torch
import os
import json


def get_model(model_name, **kwargs):
    model_name = model_name.lower()
    # we use dtype argument specifically for llama to determine if we use flash attention
    dtype = torch.float32
    if "dtype" in kwargs:
        dtype = kwargs["dtype"]
        del kwargs["dtype"]

    vision_model_args = ["image_size"]
    shared_model_args = ["layers"]
    if "insert_noop" in kwargs:
        vision_model_args.append("insert_noop")
        if not (
            model_name.startswith("deepspeedllama")
            or model_name.startswith("deepspeedgpt")
            or model_name.startswith("deepspeedvit")
        ):
            del kwargs["insert_noop"]
    if is_vision_model(model_name):
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in vision_model_args or k in shared_model_args
        }
    else:
        for k in vision_model_args:
            if k in kwargs:
                del kwargs[k]
    if is_gpt_model(model_name):
        model = get_gpt_model(model_name, **kwargs)
    elif is_llama_model(model_name):
        model = get_llama_model(model_name, **kwargs)
    elif is_bert_model(model_name):
        model = get_bert_model(model_name, **kwargs)
    elif is_opt_model(model_name):
        model = get_opt_model(model_name, **kwargs)
    elif is_vit_model(model_name):
        model = get_vit_model(model_name, **kwargs)
    elif is_gpt_deepspeed_model(model_name):
        model = get_gpt_deepspeed_model(model_name, **kwargs)
    elif is_vit_deepspeed_model(model_name):
        model = get_vit_deepspeed_model(model_name, **kwargs)
    elif is_bert_deepspeed_model(model_name):
        model = get_bert_deepspeed_model(model_name, **kwargs)
    elif is_gpt_toy_model(model_name):
        model = get_gpt_toy_model(model_name, **kwargs)
    elif is_llama_v2_deepspeed_model(model_name):
        model = get_llama_v2_deepspeed_model(model_name, **kwargs)
    elif is_llama_deepspeed_model(model_name):
        kwargs["dtype"] = dtype
        model = get_llama_deepspeed_model(model_name, **kwargs)
    elif is_mistral_model(model_name):
        model = get_mistral_model(model_name, **kwargs)
    elif is_mistral_deepspeed_model(model_name):
        model = get_mistral_deepspeed_model(model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model._model_name = model_name
    model._num_layers = len(get_layers(model))
    return model


def is_gpt_model(model_name):
    model_name = model_name.lower()
    return model_name.startswith("gpt")


def is_llama_model(model_name):
    model_name = model_name.lower()
    return model_name.startswith("llama")


def is_bert_model(model_name):
    model_name = model_name.lower()
    return model_name.startswith("bert")


def is_opt_model(model_name):
    model_name = model_name.lower()
    return model_name.startswith("opt")


def is_vit_model(model_name):
    model_name = model_name.lower()
    return model_name.startswith("vit")


def is_vision_model(model_name):
    model_name = model_name.lower()
    return "vit" in model_name


def is_gpt_deepspeed_model(model_name):
    model_name = model_name.lower()
    return model_name.startswith("deepspeedgpt")


def is_vit_deepspeed_model(model_name):
    model_name = model_name.lower()
    return model_name.startswith("deepspeedvit")


def is_bert_deepspeed_model(model_name):
    model_name = model_name.lower()
    return model_name.startswith("deepspeedbert")


def is_gpt_toy_model(model_name):
    model_name = model_name.lower()
    return model_name.startswith("toygpt")


def is_llama_deepspeed_model(model_name):
    model_name = model_name.lower()
    return model_name.startswith("deepspeedllama") and not is_llama_v2_deepspeed_model(
        model_name
    )


def is_llama_v2_deepspeed_model(model_name):
    model_name = model_name.lower()
    return model_name.startswith("deepspeedllamav2")


def is_mistral_model(model_name):
    model_name = model_name.lower()
    return model_name.startswith("mistral")


def is_mistral_deepspeed_model(model_name):
    model_name = model_name.lower()
    return model_name.startswith("deepspeedmistral")


def get_layers(model, model_name=None):
    model_name = model._model_name if model_name is None else model_name
    if is_gpt_model(model_name):
        return model.transformer.h
    elif is_llama_model(model_name):
        return model.model.layers
    elif is_bert_model(model_name):
        return model.bert.encoder.layer
    elif is_opt_model(model_name):
        return model.model.decoder.layers
    elif is_vit_model(model_name):
        return model.vit.encoder.layer
    elif is_gpt_deepspeed_model(model_name):
        return model.h
    elif is_vit_deepspeed_model(model_name):
        return model.encoder_layers
    elif is_bert_deepspeed_model(model_name):
        return model.encoder_layers
    elif is_gpt_toy_model(model_name):
        return model.h
    elif is_llama_v2_deepspeed_model(model_name):
        return model.layers
    elif is_llama_deepspeed_model(model_name):
        return model.layers
    elif is_mistral_model(model_name):
        return model.model.layers
    elif is_mistral_deepspeed_model(model_name):
        return model.layers
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def replace_layers(model, new_layers, model_name=None):
    model_name = model._model_name if model_name is None else model_name
    if is_gpt_model(model_name):
        model.transformer.h = new_layers
    elif is_llama_model(model_name):
        model.model.layers = new_layers
    elif is_bert_model(model_name):
        model.bert.encoder.layer = new_layers
    elif is_opt_model(model_name):
        model.model.decoder.layers = new_layers
    elif is_vit_model(model_name):
        model.vit.encoder.layer = new_layers
    elif is_gpt_deepspeed_model(model_name):
        model.h = new_layers
    elif is_vit_deepspeed_model(model_name):
        model.encoder_layers = new_layers
    elif is_bert_deepspeed_model(model_name):
        model.encoder_layers = new_layers
    elif is_gpt_toy_model(model_name):
        model.h = new_layers
    elif is_llama_v2_deepspeed_model(model_name):
        model.layers = new_layers
    elif is_llama_deepspeed_model(model_name):
        model.layers = new_layers
    elif is_mistral_model(model_name):
        model.model.layers = new_layers
    elif is_mistral_deepspeed_model(model_name):
        model.layers = new_layers
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def get_all_layers(model, model_name=None):
    model_name = model._model_name if model_name is None else model_name
    if is_gpt_deepspeed_model(model_name):
        return model.join_layers()
    elif is_vit_deepspeed_model(model_name):
        return model.join_layers()
    elif is_bert_deepspeed_model(model_name):
        return model.join_layers()
    elif is_gpt_toy_model(model_name):
        return model.join_layers()
    elif is_llama_v2_deepspeed_model(model_name):
        return model.join_layers()
    elif is_llama_deepspeed_model(model_name):
        return model.join_layers()
    elif is_mistral_deepspeed_model(model_name):
        return model.join_layers()
    else:
        return get_layers(model, model_name)


def get_embedding_layer(model, model_name=None):
    model_name = model._model_name if model_name is None else model_name
    if is_gpt_model(model_name):
        return model.transformer.tokens_embed
    elif is_llama_model(model_name):
        return model.model.embed_tokens
    elif is_bert_model(model_name):
        return model.bert.embeddings
    elif is_opt_model(model_name):
        return model.model.decoder.embed_tokens
    elif is_vit_model(model_name):
        return model.vit.embeddings
    elif is_gpt_deepspeed_model(model_name):
        return model.embedding
    elif is_vit_deepspeed_model(model_name):
        return model.embeddings
    elif is_bert_deepspeed_model(model_name):
        return model.embeddings
    elif is_gpt_toy_model(model_name):
        return model.embedding
    elif is_llama_v2_deepspeed_model(model_name):
        return model.tok_embeddings
    elif is_llama_deepspeed_model(model_name):
        return model.embed_tokens
    elif is_mistral_model(model_name):
        return model.model.embed_tokens
    elif is_mistral_deepspeed_model(model_name):
        return model.embed_tokens


def get_layer_attention(model_name, layer):
    if is_gpt_model(model_name):
        return layer.attn
    elif is_llama_model(model_name):
        return layer.self_attn
    elif is_gpt_deepspeed_model(model_name):
        return layer.block.attn
    elif is_vit_deepspeed_model(model_name):
        return layer.block.attention
    elif is_llama_v2_deepspeed_model(model_name):
        return layer.attention
    elif is_llama_deepspeed_model(model_name):
        return layer.block.self_attn
    elif is_mistral_model(model_name):
        return layer.self_attn
    elif is_mistral_deepspeed_model(model_name):
        return layer.block.self_attn
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def get_hidden_size(model, model_name=None):
    model_name = model._model_name if model_name is None else model_name
    if hasattr(model, "_hidden_size"):
        return model._hidden_size
    else:
        raise ValueError(f"Hidden size not implemented for: {model_name}")


def get_layer_feed_forward(model_name, layer):
    if is_gpt_model(model_name):
        return layer.mlp
    elif is_llama_model(model_name):
        return layer.mlp
    elif is_gpt_deepspeed_model(model_name):
        return layer.feed_forward
    elif is_llama_v2_deepspeed_model(model_name):
        return layer.feed_forward
    elif is_llama_deepspeed_model(model_name):
        return layer.block.mlp
    elif is_mistral_model(model_name):
        return layer.mlp
    elif is_mistral_deepspeed_model(model_name):
        return layer.block.mlp
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def wrap_other_layers(model_name, model):
    if is_gpt_model(model_name):
        model.transformer.tokens_embed = wrap(model.transformer.tokens_embed)
        model.transformer.positions_embed = wrap(model.transformer.positions_embed)
        model.lm_head = wrap(model.lm_head)

    elif is_llama_model(model_name):
        model.model.embed_tokens = wrap(model.model.embed_tokens)
        model.model.norm = wrap(model.model.norm)
        model.lm_head = wrap(model.lm_head)

    elif is_bert_model(model_name):
        model.bert.embeddings = wrap(model.bert.embeddings)
        if model.bert.pooler:
            model.bert.pooler = wrap(model.bert.pooler)
        model.cls = wrap(model.cls)

    elif is_opt_model(model_name):
        model.model.decoder.embed_tokens = wrap(model.model.decoder.embed_tokens)
        model.model.decoder.embed_positions = wrap(model.model.decoder.embed_positions)
        if model.model.decoder.project_out:
            model.model.decoder.project_out = wrap(model.model.decoder.project_out)
        if model.model.decoder.project_in:
            model.model.decoder.project_in = wrap(model.model.decoder.project_in)
        if model.model.decoder.final_layer_norm:
            model.model.decoder.final_layer_norm = wrap(
                model.model.decoder.final_layer_norm
            )
        model.lm_head = wrap(model.lm_head)

    elif is_vit_model(model_name):
        model.vit.embeddings = wrap(model.vit.embeddings)
        model.vit.layernorm = wrap(model.vit.layernorm)
        if model.vit.pooler:
            model.vit.pooler = wrap(model.vit.pooler)
        model.classifier = wrap(model.classifier)

    elif is_gpt_deepspeed_model(model_name):
        if model.embedding is not None:
            model.embedding = wrap(model.embedding)
        if model.output_layer is not None:
            model.output_layer = wrap(model.output_layer)

    elif is_gpt_toy_model(model_name):
        model.embedding = wrap(model.embedding)

    elif is_vit_deepspeed_model(model_name):
        model.embeddings = wrap(model.embeddings)
        model.output_layer = wrap(model.output_layer)

    elif is_bert_deepspeed_model(model_name):
        model.embeddings = wrap(model.embeddings)
        model.cls = wrap(model.cls)

    elif is_llama_v2_deepspeed_model(model_name):
        if model.tok_embeddings is not None:
            model.tok_embeddings = wrap(model.tok_embeddings)
        if model.output_layer is not None:
            model.output_layer = wrap(model.output_layer)

    elif is_llama_deepspeed_model(model_name):
        if model.embed_tokens is not None:
            model.embed_tokens = wrap(model.embed_tokens)
        if model.output_layer is not None:
            model.output_layer = wrap(model.output_layer)

    elif is_mistral_model(model_name):
        model.model.embed_tokens = wrap(model.model.embed_tokens)
        model.model.norm = wrap(model.model.norm)
        model.lm_head = wrap(model.lm_head)

    elif is_mistral_deepspeed_model(model_name):
        model.embed_tokens = wrap(model.embed_tokens)
        model.output_layer = wrap(model.output_layer)


def get_trainable_params(entity):
    trainable_params = sum(p.numel() for p in entity.parameters() if p.requires_grad)
    return trainable_params


def get_total_model_params(model_name, model):
    total_model_params = 0
    layers = get_layers(model, model_name=model_name)

    for i in range(len(layers)):
        total_model_params += get_trainable_params(layers[i])

    if is_gpt_model(model_name):
        total_model_params += get_trainable_params(model.transformer.tokens_embed)
        total_model_params += get_trainable_params(model.transformer.positions_embed)
        total_model_params += get_trainable_params(model.lm_head)

    elif is_llama_model(model_name):
        total_model_params += get_trainable_params(model.model.embed_tokens)
        total_model_params += get_trainable_params(model.model.norm)
        total_model_params += get_trainable_params(model.lm_head)

    elif is_bert_model(model_name):
        total_model_params += get_trainable_params(model.bert.embeddings)
        if model.bert.pooler:
            total_model_params += get_trainable_params(model.bert.pooler)

        total_model_params += get_trainable_params(model.cls)

    elif is_opt_model(model_name):
        total_model_params += get_trainable_params(model.model.decoder.embed_tokens)
        total_model_params += get_trainable_params(model.model.decoder.embed_positions)

        if model.model.decoder.project_out:
            total_model_params += get_trainable_params(model.model.decoder.project_out)
        if model.model.decoder.project_in:
            total_model_params += get_trainable_params(model.model.decoder.project_in)
        if model.model.decoder.final_layer_norm:
            total_model_params += get_trainable_params(
                model.model.decoder.final_layer_norm
            )

        total_model_params += get_trainable_params(model.lm_head)

    elif is_vit_model(model_name):
        total_model_params += get_trainable_params(model.vit.embeddings)
        total_model_params += get_trainable_params(model.vit.layernorm)
        if model.vit.pooler:
            total_model_params += get_trainable_params(model.vit.pooler)
        total_model_params += get_trainable_params(model.classifier)

    elif is_gpt_deepspeed_model(model_name):
        if model.embedding is not None:
            total_model_params += get_trainable_params(model.embedding.tokens_embed)
            total_model_params += get_trainable_params(model.embedding.positions_embed)
            total_model_params += get_trainable_params(model.embedding.drop)
        if model.ln_f is not None:
            total_model_params += get_trainable_params(model.ln_f)
        if model.output_layer is not None:
            total_model_params += get_trainable_params(model.lm_head)

    elif is_vit_deepspeed_model(model_name):
        total_model_params += get_trainable_params(model.embeddings)
        total_model_params += get_trainable_params(model.layernorm)
        total_model_params += get_trainable_params(model.classifier)

    elif is_bert_deepspeed_model(model_name):
        total_model_params += get_trainable_params(model.embeddings)
        total_model_params += get_trainable_params(model.cls)

    elif is_gpt_toy_model(model_name):
        total_model_params += get_trainable_params(model.embedding.tokens_embed)
        total_model_params += get_trainable_params(model.embedding.positions_embed)
        total_model_params += get_trainable_params(model.embedding.drop)

    elif is_llama_v2_deepspeed_model(model_name):
        if model.tok_embeddings is not None:
            total_model_params += get_trainable_params(model.tok_embeddings)
        if model.norm is not None:
            total_model_params += get_trainable_params(model.norm)
        if model.output is not None:
            total_model_params += get_trainable_params(model.output)

    elif is_llama_deepspeed_model(model_name):
        if model.embed_tokens is not None:
            total_model_params += get_trainable_params(model.embed_tokens)
        if model.norm is not None:
            total_model_params += get_trainable_params(model.norm)
        if model.lm_head is not None:
            total_model_params += get_trainable_params(model.lm_head)

    elif is_mistral_model(model_name):
        total_model_params += get_trainable_params(model.model.embed_tokens)
        total_model_params += get_trainable_params(model.model.norm)
        total_model_params += get_trainable_params(model.lm_head)

    elif is_mistral_deepspeed_model(model_name):
        total_model_params += get_trainable_params(model.embed_tokens)
        total_model_params += get_trainable_params(model.norm)
        total_model_params += get_trainable_params(model.lm_head)

    return total_model_params


SUPPORTED_MODELS = [
    # Deepspeed GPT models
    "deepspeedgpt_test",
    "deepspeedgpt_1.3b",
    "deepspeedgpt_2.7b",
    "deepspeedgpt_6.7b",
    "deepspeedgpt_13b",
    "deepspeedgpt_175b",
    # Deepspeed LLAMA models
    "deepspeedllama_tiny",
    "deepspeedllama_3b",
    "deepspeedllama_7b",
    "deepspeedllama_13b",
    "deepspeedllama_33b",
    "deepspeedllama_65b",
    # Deepspeed LLAMA v2 models
    "deepspeedllamav2_tiny",
    "deepspeedllamav2_3b",
    "deepspeedllamav2_7b",
    "deepspeedllamav2_13b",
    "deepspeedllamav2_33b",
    "deepspeedllamav2_65b",
    # Deepspeed ViT models
    "deepspeedvit_default",
    "deepspeedvit_g_small",
    "deepspeedvit_e_small",
    "deepspeedvit_g",
    "deepspeedvit_e",
    # Deepspeed BERT models
    "deepspeedbert_large",
    "deepspeedbert_xlarge",
]


def get_config_for_model(model_name):
    """
    Returns  (for now)
    1. num_transformer_layers
    2. hidden_size
    3. num_attention_heads
    4. intermediate_size
    """
    intermediate_size = None
    # TODO: improve this function
    if model_name == "deepspeedgpt_test":
        num_transformer_layers = 12
        hidden_size = 512
        num_attention_heads = 128
    elif model_name == "deepspeedgpt_1.3b":
        num_transformer_layers = 24
        hidden_size = 2048
        num_attention_heads = 128
        intermediate_size = hidden_size * 4
    elif model_name == "deepspeedgpt_2.7b":
        num_transformer_layers = 32
        hidden_size = 2560
        num_attention_heads = 80
        intermediate_size = hidden_size * 4
    elif model_name == "deepspeedgpt_6.7b":
        num_transformer_layers = 32
        hidden_size = 4096
        num_attention_heads = 128
    elif model_name == "deepspeedgpt_175b":
        num_transformer_layers = 96
        hidden_size = 12288
        num_attention_heads = 96
    elif model_name == "deepspeedllama_tiny" or model_name == "deepspeedllamav2_tiny":
        num_transformer_layers = 22
        hidden_size = 2048
        num_attention_heads = 32
        intermediate_size = 5632
    elif model_name == "deepspeedllama_3b" or model_name == "deepspeedllamav2_3b":
        num_transformer_layers = 26
        hidden_size = 3200
        num_attention_heads = 32
        intermediate_size = 8640
    elif model_name == "deepspeedllama_7b" or model_name == "deepspeedllamav2_7b":
        num_transformer_layers = 32
        hidden_size = 4096
        num_attention_heads = 32
        intermediate_size = 11008
    elif model_name == "deepspeedllama_13b" or model_name == "deepspeedllamav2_13b":
        num_transformer_layers = 40
        hidden_size = 5120
        num_attention_heads = 40
        intermediate_size = 13824
    elif model_name == "deepspeedllama_33b" or model_name == "deepspeedllamav2_33b":
        num_transformer_layers = 60
        hidden_size = 6656
        num_attention_heads = 52
        intermediate_size = 17920
    elif model_name == "deepspeedllama_65b" or model_name == "deepspeedllamav2_65b":
        num_transformer_layers = 80
        hidden_size = 8192
        num_attention_heads = 64
        intermediate_size = 22016
    elif model_name == "deepspeedvit_default":
        num_transformer_layers = 12
        hidden_size = 768
        num_attention_heads = 12
        intermediate_size = 3072
    elif model_name == "deepspeedvit_g_small":
        num_transformer_layers = 6
        hidden_size = 1664
        num_attention_heads = 16
        intermediate_size = 8192
    elif model_name == "deepspeedvit_e_small":
        num_transformer_layers = 6
        hidden_size = 1792
        num_attention_heads = 16
        intermediate_size = 15360
    elif model_name == "deepspeedvit_g":
        num_transformer_layers = 48
        hidden_size = 1664
        num_attention_heads = 16
        intermediate_size = 8192
    elif model_name == "deepspeedvit_e":
        num_transformer_layers = 56
        hidden_size = 1792
        num_attention_heads = 16
        intermediate_size = 15360
    elif model_name == "deepspeedbert_large":
        num_transformer_layers = 24
        hidden_size = 1024
        num_attention_heads = 16
        intermediate_size = hidden_size * 4
    elif model_name == "deepspeedbert_xlarge":
        num_transformer_layers = 36
        hidden_size = 1536
        num_attention_heads = 24
        intermediate_size = hidden_size * 4
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # assert intermediate_size is not None
    return num_transformer_layers, hidden_size, num_attention_heads, intermediate_size


def get_model_stats(model_name):
    stats_file = os.path.join(
        os.path.dirname(__file__), "..", "data", "model_stats.json"
    )
    with open(stats_file, "r") as f:
        stats = json.load(f)
    return stats[model_name]


def get_model_part_for_stage(
    model_name, layer_partition, is_first_stage, is_last_stage, dtype=torch.float32
):
    """
    We can refine this logic more
    """
    num_transformer_layers = layer_partition[1] - layer_partition[0]
    model_stage = get_model(model_name, layers=num_transformer_layers, dtype=dtype)
    if not is_first_stage:
        # Remove embedding layers from non-first stages
        # TODO: refactor/clean this logic
        model_stage.embedding = None  # gpt simple
        model_stage.embeddings = None  # bert
        model_stage.embed_tokens = None  # llama_simple
        model_stage.tok_embeddings = None  # llama_v2

    if not is_last_stage:
        # TODO: refactor/clean this logic
        model_stage.ln_f = None
        model_stage.lm_head = None
        model_stage.output_layer = None
        model_stage.cls = None  # Bert

    return model_stage
