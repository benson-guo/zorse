# -*- coding: utf-8 -*-
import collections
from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from models.common import NoOp
from transformers.activations import ACT2FN

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Used from huggingface transformers


class ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = (
            nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            if use_mask_token
            else None
        )
        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(
            1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        assert (
            int(h0) == patch_pos_embed.shape[-2]
            and int(w0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, height, width
            )
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class ViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )

        # changed indexing here
        num_patches = (image_size[1] // patch_size[1]) * (
            image_size[0] // patch_size[0]
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class ViTSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.key = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.value = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


class ViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class ViTAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class ViTIntermediate(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class ViTOutput(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


class ViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config) -> None:
        super().__init__()
        self.seq_len_dim = 1
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.layernorm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(
                hidden_states
            ),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Used from huggingface transformers


@dataclass
class DeepspeedViTConfig:
    """
    Configure image_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads
    """

    image_size = 224
    hidden_size = 768
    intermediate_size = 3072
    num_hidden_layers = 12
    num_attention_heads = 12
    hidden_act = "gelu"
    hidden_dropout_prob = 0.0
    attention_probs_dropout_prob = 0.0
    initializer_range = 0.02
    layer_norm_eps = 1e-12
    patch_size = 16
    num_channels = 3
    qkv_bias = True
    encoder_stride = 16
    num_labels = 10


class VitLayerBlock(nn.Module):
    """
    Wrapper around VitEncoderLayer
    """

    def __init__(self, config):
        super().__init__()
        self.block = ViTLayer(config)

    def forward(self, x):
        block_output = self.block(x, head_mask=None, output_attentions=False)
        x = block_output[0]
        return x


class DeepspeedViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = ViTEmbeddings(config, use_mask_token=False)
        layer_list = []
        for _ in range(config.num_hidden_layers):
            layer_list.append(VitLayerBlock(config))
            if config.insert_noop:
                layer_list.append(NoOp())
        self.encoder_layers = nn.ModuleList(layer_list)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.output_layer = nn.Sequential(self.layernorm, self.classifier)

    def forward(self, pixel_values, labels=None):
        if self.embeddings:
            x = self.embeddings(pixel_values)
        else:
            x = pixel_values

        for layer in self.encoder_layers:
            x = layer(x)

        if self.output_layer:
            x = self.output_layer(x)
        return x

    def join_layers(self):
        return (
            [self.embeddings] + [i for i in self.encoder_layers] + [self.output_layer]
        )


def get_model(model_name, image_size=224, layers=None, insert_noop=False):
    configuration = DeepspeedViTConfig()

    configuration.image_size = image_size

    if model_name == "deepspeedvit_default":
        configuration.hidden_size = 768
        configuration.num_hidden_layers = 12
        configuration.num_attention_heads = 12
        configuration.intermediate_size = 4 * configuration.hidden_size
    elif model_name == "deepspeedvit_1b":
        configuration.hidden_size = 1824
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
        configuration.intermediate_size = 4 * configuration.hidden_size
    elif model_name == "deepspeedvit_2b":
        configuration.hidden_size = 2560
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
        configuration.intermediate_size = 4 * configuration.hidden_size
    elif model_name == "deepspeedvit_3b":
        configuration.hidden_size = 3172
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
        configuration.intermediate_size = 4 * configuration.hidden_size
    elif model_name == "deepspeedvit_4b":
        configuration.hidden_size = 3648
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
        configuration.intermediate_size = 4 * configuration.hidden_size
    elif model_name == "deepspeedvit_5b":
        configuration.hidden_size = 4096
        configuration.num_hidden_layers = 24
        configuration.num_attention_heads = 32
        configuration.intermediate_size = 4 * configuration.hidden_size
    elif model_name == "deepspeedvit_h":
        configuration.hidden_size = 1280
        configuration.num_hidden_layers = 32
        configuration.num_attention_heads = 16
        configuration.intermediate_size = 5120
    elif model_name == "deepspeedvit_g_small":
        configuration.hidden_size = 1664
        configuration.num_hidden_layers = 12
        configuration.num_attention_heads = 16
        configuration.intermediate_size = 8192
    elif model_name == "deepspeedvit_g":
        configuration.hidden_size = 1664
        configuration.num_hidden_layers = 6
        configuration.num_attention_heads = 16
        configuration.intermediate_size = 8192
    elif model_name == "deepspeedvit_e":
        configuration.hidden_size = 1792
        configuration.num_hidden_layers = 56
        configuration.num_attention_heads = 16
        configuration.intermediate_size = 15360
    elif model_name == "deepspeedvit_e_toy":
        configuration.hidden_size = 1792
        configuration.num_hidden_layers = 8
        configuration.num_attention_heads = 16
        configuration.intermediate_size = 15360
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if layers:
        configuration.num_hidden_layers = layers

    # Set FFN dimension to 4x d_model
    # configuration.intermediate_size = 4 * configuration.hidden_size

    configuration.insert_noop = insert_noop
    model = DeepspeedViT(configuration)
    model._hidden_size = configuration.hidden_size
    model._patch_size = configuration.patch_size

    return model
