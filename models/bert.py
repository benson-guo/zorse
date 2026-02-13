# -*- coding: utf-8 -*-
from transformers import BertConfig, BertLMHeadModel


def get_model(model_name, vocab_size=49152, seq_length=2048):
    # Initializing a BERT bert-base-uncased style configuration
    configuration = BertConfig()

    configuration.vocab_size = vocab_size
    configuration.max_position_embeddings = seq_length
    configuration.is_decoder = True

    if model_name == "bert_110m":
        configuration.hidden_size = 768
        configuration.num_hidden_layers = 12
        configuration.num_attention_heads = 12
        configuration.intermediate_size = 4 * configuration.hidden_size
    elif model_name == "bert_330m":
        configuration.hidden_size = 1024
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 16
        configuration.intermediate_size = 4 * configuration.hidden_size
    elif model_name == "bert_1b":
        configuration.hidden_size = 1824
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
        configuration.intermediate_size = 4 * configuration.hidden_size
    elif model_name == "bert_2b":
        configuration.hidden_size = 2560
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
        configuration.intermediate_size = 4 * configuration.hidden_size
    elif model_name == "bert_3b":
        configuration.hidden_size = 3172
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
        configuration.intermediate_size = 4 * configuration.hidden_size
    elif model_name == "bert_4b":
        configuration.hidden_size = 3648
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
        configuration.intermediate_size = 4 * configuration.hidden_size
    elif model_name == "bert_5b":
        configuration.hidden_size = 4096
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
        configuration.intermediate_size = 4 * configuration.hidden_size
    elif model_name == "bert_large":
        # 340M
        configuration.hidden_size = 1024
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 16
        configuration.intermediate_size = 4096
    elif model_name == "bert_xlarge":
        # 1.2B
        configuration.hidden_size = 1536
        configuration.num_hidden_layers = 36
        configuration.num_attention_heads = 24
        configuration.intermediate_size = 6144
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model = BertLMHeadModel(configuration)
    model._hidden_size = configuration.hidden_size

    return model
