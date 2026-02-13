# -*- coding: utf-8 -*-
from transformers import OPTConfig, OPTForCausalLM


def get_model(model_name, vocab_size=49152, seq_length=2048):
    configuration = OPTConfig()

    configuration.vocab_size = vocab_size
    configuration.max_position_embeddings = seq_length

    if model_name == "opt_125m":
        configuration.hidden_size = 768
        configuration.num_hidden_layers = 12
        configuration.num_attention_heads = 12
    elif model_name == "opt_350m":
        configuration.hidden_size = 1024
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 16
    elif model_name == "opt_1b":
        configuration.hidden_size = 1824
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
    elif model_name == "opt_2b":
        configuration.hidden_size = 2624
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
    elif model_name == "opt_3b":
        configuration.hidden_size = 3232
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
    elif model_name == "opt_4b":
        configuration.hidden_size = 3712
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
    elif model_name == "opt_5b":
        configuration.hidden_size = 4160
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
    elif model_name == "opt_6.7b":
        configuration.hidden_size = 4096
        configuration.num_hidden_layers = 32
        configuration.num_attention_heads = 32
    elif model_name == "opt_13b":
        configuration.hidden_size = 5120
        configuration.num_hidden_layers = 40
        configuration.num_attention_heads = 40
    elif model_name == "opt_30b":
        configuration.hidden_size = 7168
        configuration.num_hidden_layers = 48
        configuration.num_attention_heads = 56
    elif model_name == "opt_66b":
        configuration.hidden_size = 9216
        configuration.num_hidden_layers = 64
        configuration.num_attention_heads = 72
    elif model_name == "opt_175b":
        configuration.hidden_size = 12288
        configuration.num_hidden_layers = 96
        configuration.num_attention_heads = 96
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Set FFN dimension to 4x d_model
    configuration.ffn_dim = 4 * configuration.hidden_size

    model = OPTForCausalLM(configuration)
    model._hidden_size = configuration.hidden_size

    return model
