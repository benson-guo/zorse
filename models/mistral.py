# -*- coding: utf-8 -*-
from transformers import MistralConfig, MistralForCausalLM

# Configuration taken from: https://github.com/neobundy/Deep-Dive-Into-AI-With-MLX-PyTorch/blob/master/deep-dives/001-mistral-7b/README.md#architecture-of-mistral-7b
def get_model(model_name, vocab_size=49152, seq_length=2048, layers=None):
    configuration = MistralConfig()

    # by default vocab_size is 32000, using our default value
    configuration.vocab_size = vocab_size
    # this the max sequence length that this model might ever be used with, by config: 4096*32 (but we use 2048 by default)
    configuration.max_position_embeddings = seq_length

    if model_name == "mistral_7b":
        configuration.hidden_size = 4096
        configuration.num_hidden_layers = 32
        configuration.num_attention_heads = 32
        # note intermediate to exactly match config (not *4)
        configuration.intermediate_size = 14336
    elif model_name == "mistral_toy":
        configuration.hidden_size = 4096
        configuration.num_hidden_layers = 5
        configuration.num_attention_heads = 32
        configuration.intermediate_size = 14336
    # add more configs if needed
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if layers is not None:
        configuration.num_hidden_layers = layers

    # configuration.intermediate_size = configuration.hidden_size * 4
    model = MistralForCausalLM(configuration)
    model._hidden_size = configuration.hidden_size

    return model
