# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import math
import torch
import torch.nn as nn
from transformers.pytorch_utils import (
    Conv1D,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)
from models.common import NoOp
from transformers.activations import silu, gelu_new


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Using OpenAI GPT Attention and Block Implementation
class Attention(nn.Module):
    def __init__(self, nx, n_positions, config, scale=False):
        super().__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implementation]
        if n_state % config.n_head != 0:
            raise ValueError(
                f"Attention n_state shape: {n_state} must be divisible by config.n_head {config.n_head}"
            )
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(n_positions, n_positions)).view(
                1, 1, n_positions, n_positions
            ),
            persistent=False,
        )
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_head, self.split_size // self.n_head, self.pruned_heads
        )
        index_attn = torch.cat(
            [index, index + self.split_size, index + (2 * self.split_size)]
        )
        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(
        self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False
    ):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        # w = w * self.bias + -1e9 * (1 - self.bias)  # TF implementation method: mask_attn_weights
        # XD: self.b may be larger than w, so we need to crop it
        b = self.bias[:, :, : w.size(-2), : w.size(-1)]
        w = w * b + -1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.functional.softmax(w, dim=-1)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implementation: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implementation: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, attention_mask=None, head_mask=None, output_attentions=False):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        attn_outputs = self._attn(
            query, key, value, attention_mask, head_mask, output_attentions
        )
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a] + attn_outputs[1:]
        return outputs  # a, (attentions)


ACT_FNS = {"relu": nn.ReLU, "silu": silu, "gelu": gelu_new, "swish": silu}


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT_FNS[config.afn]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_positions, config, scale=False):
        super().__init__()
        nx = config.n_embd
        self.attn = Attention(nx, n_positions, config, scale)
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)

    def forward(self, x, attention_mask=None, head_mask=None, output_attentions=False):
        attn_outputs = self.attn(
            x,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        a = attn_outputs[0]

        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)

        outputs = [h] + attn_outputs[1:]
        return outputs


# Embedding Layer v2 using nn.Parameter
# class EmbeddingLayer(nn.Module):
#     """
#     Initial embedding layer for GPT
#     token embedding + position embedding -> Dropout
#     """
#     def __init__(self, config):
#         super().__init__()
#         self.token_embeddings = nn.Parameter(torch.randn(config.vocab_size, config.n_embd))
#         self.position_embeddings = nn.Parameter(torch.randn(config.n_positions, config.n_embd))

#         # Initialize the dropout layer
#         self.drop = nn.Dropout(config.embd_pdrop)

#     def forward(self, x):
#         batch_size, seq_length = x.size()
#         position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device).unsqueeze(0).expand(batch_size, -1)
#         tokens_embeddings = torch.embedding(self.token_embeddings, x)
#         positions_embeddings = torch.embedding(self.position_embeddings, position_ids)
#         x = tokens_embeddings + positions_embeddings
#         x = self.drop(x)
#         return x

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Using OpenAI GPT Attention and Block Implementation


@dataclass
class DeepspeedGPTConfig:
    """
    Configure vocab_size, n_embd, n_head, n_positions (seq_length), n_layer, dropout
    """

    vocab_size: int
    n_embd: int
    n_head: int
    n_positions: int
    n_layer: int
    embd_pdrop: float
    attn_pdrop: float = field(default=0.1)
    resid_pdrop: float = field(default=0.1)
    afn: str = field(default="gelu")
    layer_norm_epsilon: float = field(default=1e-05)


class TransformerBlock(nn.Module):
    """
    Wrapper around OpenAI GPT Block
    """

    def __init__(self, config):
        super().__init__()
        self.block = Block(config.n_positions, config, scale=True)

    def forward(self, x):
        block_output = self.block(
            x, attention_mask=None, head_mask=None, output_attentions=False
        )
        x = block_output[0]
        return x


class EmbeddingLayer(nn.Module):
    """
    Initial embedding layer for GPT
    token embedding + position embedding -> Dropout
    """

    def __init__(self, config):
        super().__init__()
        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

    def forward(self, x):
        tokens_embeddings = self.tokens_embed(x)
        positions_embeddings = self.positions_embed(
            torch.arange(0, x.size(1), device=x.device, dtype=torch.long)
        ).unsqueeze(0)
        x = tokens_embeddings + positions_embeddings
        x = self.drop(x)
        return x


class DeepspeedGPT(nn.Module):
    def __init__(
        self,
        vocab_size=49152,
        seq_length=512,
        n_layer=8,
        n_embd=512,
        n_head=8,
        dropout=0.1,
        insert_noop=False,
    ):
        super().__init__()
        config = DeepspeedGPTConfig(
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_head=n_head,
            n_positions=seq_length,
            n_layer=n_layer,
            embd_pdrop=dropout,
        )
        self.config = config
        self.embedding = EmbeddingLayer(config)
        h_list = []
        for _ in range(config.n_layer):
            h_list.append(TransformerBlock(config))
            if insert_noop:
                h_list.append(NoOp())
        self.h = nn.ModuleList(h_list)
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.output_layer = nn.Sequential(self.ln_f, self.lm_head)

    def forward(self, x, labels=None):
        # embedding
        if self.embedding:
            x = self.embedding(x)

        # transformer blocks
        for _, block in enumerate(self.h):
            x = block(x)

        # Apply final layer normalization
        if self.output_layer:
            x = self.output_layer(x)

        return x

    def join_layers(self):
        model_layers = []
        if self.embedding:
            model_layers.append(self.embedding)
        model_layers.extend([i for i in self.h])
        if self.output_layer:
            model_layers.append(self.output_layer)
        return model_layers


def get_model(
    model_name, vocab_size=49152, seq_length=2048, layers=None, insert_noop=False
):
    n_layer = 8
    n_embd = 512
    n_head = 8
    dropout = 0.1
    if model_name == "deepspeedgpt_test":
        n_layer = 12
        n_embd = 512
        n_head = 128
        dropout = 0
    elif model_name == "deepspeedgpt_1.3b":
        n_layer = 24
        n_embd = 2048
        n_head = 128
        dropout = 0
    elif model_name == "deepspeedgpt_2.7b":
        n_layer = 32
        n_embd = 2560
        n_head = 80
        dropout = 0
    elif model_name == "deepspeedgpt_6.7b":
        n_layer = 32
        n_embd = 4096
        n_head = 128
        dropout = 0
    elif model_name == "deepspeedgpt_13b":
        n_layer = 40
        n_embd = 5140
        n_head = 128
        dropout = 0
    elif model_name == "deepspeedgpt_175b":
        n_layer = 96
        n_embd = 12288
        n_head = 96
        dropout = 0
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if layers:
        n_layer = layers

    model = DeepspeedGPT(
        vocab_size=vocab_size,
        seq_length=seq_length,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        dropout=dropout,
        insert_noop=insert_noop,
    )
    model._hidden_size = n_embd

    return model


class ToyGPT(nn.Module):
    def __init__(
        self,
        vocab_size=49152,
        seq_length=2048,
        n_layer=8,
        n_embd=512,
        n_head=8,
        dropout=0.1,
    ):
        super().__init__()
        config = DeepspeedGPTConfig(
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_head=n_head,
            n_positions=seq_length,
            n_layer=n_layer,
            embd_pdrop=dropout,
        )
        self.embedding = EmbeddingLayer(config)
        self.h = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )

    def forward(self, x, labels=None):

        # embedding
        x = self.embedding(x)
        # transformer blocks
        for _, block in enumerate(self.h):
            x = block(x)

        return x

    def join_layers(self):
        return [self.embedding] + [i for i in self.h]


def get_toy_model(model_name, vocab_size=49152, seq_length=2048, layers=None):
    if model_name == "toygpt_test":
        n_layer = 6
        n_embd = 768
        n_head = 12
        dropout = 0
    elif model_name == "toygpt_1.3b":
        n_layer = 5
        n_embd = 2048
        n_head = 128
        dropout = 0
    elif model_name == "toygpt_2.7b":
        n_layer = 5
        n_embd = 2560
        n_head = 80
        dropout = 0
    elif model_name == "toygpt_6.7b":
        n_layer = 5
        n_embd = 4096
        n_head = 128
        dropout = 0

    return ToyGPT(
        vocab_size=vocab_size,
        seq_length=seq_length,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        dropout=dropout,
    )