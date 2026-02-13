# -*- coding: utf-8 -*-
from transformers import LlamaForCausalLM, LlamaConfig


def get_model(model_name, vocab_size=49152, seq_length=2048):
    configuration = LlamaConfig()

    configuration.vocab_size = vocab_size
    configuration.max_position_embeddings = seq_length

    if model_name == "llama_1b":
        configuration.hidden_size = 1536
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
    elif model_name == "llama_2b":
        configuration.hidden_size = 2176
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
    elif model_name == "llama_3b":
        configuration.hidden_size = 2688
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
    elif model_name == "llama_4b":
        configuration.hidden_size = 3136
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
    elif model_name == "llama_5b":
        configuration.hidden_size = 3520
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
    elif model_name == "llama_6.7b":
        configuration.hidden_size = 4096
        configuration.num_hidden_layers = 32
        configuration.num_attention_heads = 32
    elif model_name == "llama_13b":
        configuration.hidden_size = 5120
        configuration.num_hidden_layers = 40
        configuration.num_attention_heads = 40
    elif model_name == "llama_33b":
        configuration.hidden_size = 6656
        configuration.num_hidden_layers = 60
        configuration.num_attention_heads = 52
    elif model_name == "llama_65b":
        configuration.hidden_size = 8192
        configuration.num_hidden_layers = 80
        configuration.num_attention_heads = 64
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Set FFN dimension to 4x d_model
    configuration.intermediate_size = configuration.hidden_size * 4

    model = LlamaForCausalLM(configuration)
    model.hidden_size = configuration.hidden_size

    return model
