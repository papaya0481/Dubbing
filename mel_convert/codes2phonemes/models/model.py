import torch
import torch.nn as nn
import math
from dataclasses import dataclass, field

from typing import Optional, Tuple, Any
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3PreTrainedModel,
    Qwen3RotaryEmbedding,
)
from transformers.utils import logging

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.utils import can_return_tuple, auto_docstring
from transformers.utils.doc import add_start_docstrings_to_model_forward
from transformers.modeling_outputs import BaseModelOutputWithPast

logger = logging.get_logger(__name__)

class Qwen3DecodeLayerForAlignment(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if (
            config.sliding_window and config._attn_implementation != "flash_attention_2"
        ):  # diff with Llama is this warning
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        
        # Fobidden is_causal
        self.self_attn.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Optional[Any],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
    
class Qwen3ModelForAlignment(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecodeLayerForAlignment(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Optional[Any]
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # if self.gradient_checkpointing and self.training and use_cache:
        #     logger.warning_once(
        #         "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        #     )
        #     use_cache = False

        # # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        # if not isinstance(past_key_values, (type(None), Cache)):
        #     raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            raise ValueError("inputs_embeds should not be None for alignment model.")

        # if use_cache and past_key_values is None:
        #     past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = attention_mask

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        

@dataclass
class ConvTransformerAlignerConfig:
    # Vocabulary
    token_vocab_size: int = 8194
    phoneme_vocab_size: int = 60

    # Shared dimension
    hidden_dim: int = 256

    # ConvNet
    conv_input_dim: int = 256
    conv_kernel_size: int = 5
    conv_stride: int = 1
    conv_padding: int = 2
    conv_num_layers: int = 2
    conv_dropout: float = 0.1

    # Transformer config
    transformer_config: Qwen3Config = None


class ConvNet(nn.Module):
    def __init__(self, config: ConvTransformerAlignerConfig):
        super().__init__()
        layers = []
        in_dim = config.conv_input_dim
        for _ in range(config.conv_num_layers):
            layers += [
                nn.Conv1d(in_dim, config.hidden_dim,
                          kernel_size=config.conv_kernel_size,
                          stride=config.conv_stride,
                          padding=config.conv_padding),
                nn.Mish(),  # activation f(x) = x * tanh(softplus(x))
                nn.BatchNorm1d(config.hidden_dim),
                nn.Dropout(config.conv_dropout),
            ]
            in_dim = config.hidden_dim
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [Batch, Time, Dim]
        x = x.transpose(1, 2) # [Batch, Dim, Time]
        x = self.conv(x)
        x = x.transpose(1, 2) # [Batch, Time, Dim]
        return x
        

class ConvTransformerAligner(nn.Module):
    def __init__(self, config: ConvTransformerAlignerConfig):
        super().__init__()
        
        # 2. Convolutional Pre-Net (局部平滑)
        self.conv_prenet = ConvNet(config)
        self.transformer_encoder = Qwen3ModelForAlignment(config.transformer_config)
        
        # 4. 预测输出层
        self.classifier = nn.Linear(config.hidden_dim, config.phoneme_vocab_size)

    def forward(self, x):
        # x shape: [Batch, Time, Dim]
        # 过 Conv 层需要 [Batch, Channel, Time]
        x = self.transformer_encoder(inputs_embeds=x) # [Batch, Time, Dim]
        x = x.last_hidden_state # [Batch, Time, Dim]
        
        x = self.conv_prenet(x)
        
        # 输出 Logits
        logits = self.classifier(x) # [Batch, Time, Vocab_Size]
        return logits
    