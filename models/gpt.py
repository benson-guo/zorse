# -*- coding: utf-8 -*-
from transformers import OpenAIGPTConfig, OpenAIGPTLMHeadModel


# Model size reference: https://arxiv.org/pdf/2005.14165.pdf#page=8
def get_model(model_name, vocab_size=49152, seq_length=2048, layers=None):
    configuration = OpenAIGPTConfig()

    configuration.vocab_size = vocab_size
    configuration.n_positions = seq_length
    if model_name == "gpt_85m":
        configuration.n_embd = 768
        configuration.n_layer = 12
        configuration.n_head = 12
    elif model_name == "gpt_300m":
        configuration.n_embd = 1024
        configuration.n_layer = 24
        configuration.n_head = 16
    elif model_name == "gpt_1b":
        configuration.n_embd = 1824
        configuration.n_layer = 24
        configuration.n_head = 32
    elif model_name == "gpt_1b_1l":
        configuration.n_embd = 1824
        configuration.n_layer = 1
        configuration.n_head = 32
    elif model_name == "gpt_1.3b":
        # official
        configuration.n_embd = 2048
        configuration.n_layer = 24
        configuration.n_head = 128
    elif model_name == "gpt_2b":
        configuration.n_embd = 2560
        configuration.n_layer = 24
        configuration.n_head = 32
    elif model_name == "gpt_2.7b":
        # official
        configuration.n_embd = 2560
        configuration.n_layer = 32
        configuration.n_head = 80
    elif model_name == "gpt_3b":
        configuration.n_embd = 3328
        configuration.n_layer = 24
        configuration.n_head = 32
    elif model_name == "gpt_4b":
        configuration.n_embd = 3648
        configuration.n_layer = 24
        configuration.n_head = 32
    elif model_name == "gpt_5b":
        configuration.n_embd = 4096
        configuration.n_layer = 24
        configuration.n_head = 32
    elif model_name == "gpt_6b":
        configuration.n_embd = 4096
        configuration.n_layer = 32
        configuration.n_head = 32
    elif model_name == "gpt_6.7b":
        # official
        configuration.n_embd = 4096
        configuration.n_layer = 32
        configuration.n_head = 128
    elif model_name == "gpt_10b":
        configuration.n_embd = 4680
        configuration.n_layer = 36
        configuration.n_head = 36
    elif model_name == "gpt_13b":
        # official
        configuration.n_embd = 5140
        configuration.n_layer = 40
        configuration.n_head = 128
    elif model_name == "gpt_20b":
        configuration.n_embd = 6144
        configuration.n_layer = 44
        configuration.n_head = 48
    elif model_name == "gpt_25b":
        configuration.n_embd = 6656
        configuration.n_layer = 46
        configuration.n_head = 52
    elif model_name == "gpt_30b":
        configuration.n_embd = 7168
        configuration.n_layer = 48
        configuration.n_head = 56
    elif model_name == "gpt_35b":
        configuration.n_embd = 7424
        configuration.n_layer = 53
        configuration.n_head = 58
    elif model_name == "gpt_40b":
        configuration.n_embd = 7808
        configuration.n_layer = 54
        configuration.n_head = 61
    elif model_name == "gpt_45b":
        configuration.n_embd = 8192
        configuration.n_layer = 56
        configuration.n_head = 64
    elif model_name == "gpt_50b":
        configuration.n_embd = 8448
        configuration.n_layer = 58
        configuration.n_head = 66
    elif model_name == "gpt_55b":
        configuration.n_embd = 8704
        configuration.n_layer = 61
        configuration.n_head = 68
    elif model_name == "gpt_60b":
        configuration.n_embd = 8960
        configuration.n_layer = 62
        configuration.n_head = 70
    elif model_name == "gpt_65b":
        configuration.n_embd = 9216
        configuration.n_layer = 64
        configuration.n_head = 72
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if layers is not None:
        configuration.n_layer = layers

    model = OpenAIGPTLMHeadModel(configuration)
    model._hidden_size = configuration.n_embd

    return model
