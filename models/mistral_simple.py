# -*- coding: utf-8 -*-
from models.common import NoOp
import torch
import torch.nn as nn
from transformers import MistralConfig
from transformers.models.mistral.modeling_mistral import (
    MistralRMSNorm,
    MistralDecoderLayer,
)
from models.common import _cast_buffers_to_dtype_and_device_patch


class Block(nn.Module):
    """
    Wrapper around Mistral Decoder layer
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.block = MistralDecoderLayer(config, layer_idx)
        self.register_buffer(
            "cached_position_ids", None
        )  # Register a persistent buffer for cached position IDs

    def forward(self, x):
        seq_length = x.size(1)  # Get the current sequence length
        if (
            self.cached_position_ids is None
            or self.cached_position_ids.size(1) != seq_length
        ):
            # Recompute position IDs if cache is empty or sequence length has changed
            self.cached_position_ids = torch.arange(
                0, seq_length, device=x.device, dtype=torch.long
            ).unsqueeze(0)

        # Use cached position IDs
        decoder_output = self.block(
            x, attention_mask=None, position_ids=self.cached_position_ids
        )
        x = decoder_output[0]
        return x


class DeepspeedMistral(nn.Module):
    def __init__(
        self,
        vocab_size=49152,
        seq_length=2048,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=22,
        num_attention_heads=32,
        insert_noop=False,
    ):
        super().__init__()
        # patch for FSDP
        torch.distributed.fsdp._runtime_utils._cast_buffers_to_dtype_and_device = (
            _cast_buffers_to_dtype_and_device_patch
        )
        # initialize config
        config = MistralConfig()
        config.vocab_size = vocab_size
        config.max_position_embeddings = seq_length
        config.hidden_size = hidden_size
        config.num_hidden_layers = num_hidden_layers
        config.num_attention_heads = num_attention_heads
        config.intermediate_size = intermediate_size

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        layer_list = []
        for layer_idx in range(config.num_hidden_layers):
            layer_list.append(Block(config, layer_idx))
            if insert_noop:
                layer_list.append(NoOp())
        self.layers = nn.ModuleList(layer_list)
        self._attn_implementation = config._attn_implementation
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.output_layer = nn.Sequential(self.norm, self.lm_head)

    def forward(self, x, labels=None):
        # embedding
        x = self.embed_tokens(x)

        # transformer blocks
        for _, block in enumerate(self.layers):
            x = block(x)

        # Apply final layer normalization
        x = self.output_layer(x)

        return x

    def join_layers(self):
        return [self.embed_tokens] + [i for i in self.layers] + [self.output_layer]


def get_model(
    model_name, vocab_size=49152, seq_length=2048, layers=None, insert_noop=False
):
    hidden_size = 4096
    intermediate_size = 14336
    num_hidden_layers = 32
    num_attention_heads = 32

    if model_name == "deepspeedmistral_7b":
        hidden_size = 4096
        intermediate_size = 14336
        num_hidden_layers = 32
        num_attention_heads = 32
    elif model_name == "deepspeedmistral_toy":
        hidden_size = 4096
        intermediate_size = 14336
        num_hidden_layers = 5
        num_attention_heads = 32
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if layers:
        num_hidden_layers = layers

    model = DeepspeedMistral(
        vocab_size=vocab_size,
        seq_length=seq_length,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        insert_noop=insert_noop,
    )
    model__hidden_size = hidden_size

    return model
