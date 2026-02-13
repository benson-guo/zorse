# -*- coding: utf-8 -*-
from transformers import ViTConfig, ViTForImageClassification


def get_model(model_name, image_size=224):
    configuration = ViTConfig()

    configuration.image_size = image_size

    if model_name == "vit_default":
        configuration.hidden_size = 768
        configuration.num_hidden_layers = 12
        configuration.num_attention_heads = 12
        configuration.intermediate_size = 4 * configuration.hidden_size
    elif model_name == "vit_1b":
        configuration.hidden_size = 1824
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
        configuration.intermediate_size = 4 * configuration.hidden_size
    elif model_name == "vit_2b":
        configuration.hidden_size = 2560
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
        configuration.intermediate_size = 4 * configuration.hidden_size
    elif model_name == "vit_3b":
        configuration.hidden_size = 3172
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
        configuration.intermediate_size = 4 * configuration.hidden_size
    elif model_name == "vit_4b":
        configuration.hidden_size = 3648
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
        configuration.intermediate_size = 4 * configuration.hidden_size
    elif model_name == "vit_5b":
        configuration.hidden_size = 4096
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
        configuration.intermediate_size = 4 * configuration.hidden_size
    elif model_name == "vit_g":
        configuration.hidden_size = 1664
        configuration.num_hidden_layers = 48
        configuration.num_attention_heads = 16
        configuration.intermediate_size = 8192
    elif model_name == "vit_e":
        configuration.hidden_size = 1792
        configuration.num_hidden_layers = 56
        configuration.num_attention_heads = 16
        configuration.intermediate_size = 15360
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Set FFN dimension to 4x d_model
    # configuration.intermediate_size = 4 * configuration.hidden_size

    model = ViTForImageClassification(configuration)
    model._hidden_size = configuration.hidden_size

    return model
