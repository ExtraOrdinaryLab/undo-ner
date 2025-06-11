from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn

from transformers.modeling_outputs import TokenClassifierOutput, BaseModelOutputWithPast
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3PreTrainedModel,
    Qwen3Config,
    Qwen3Model,
    Qwen3RMSNorm,
    Qwen3DecoderLayer,
    Qwen3Attention,
    BaseModelOutputWithPast,
    TokenClassifierOutput,
    Cache,
    FlashAttentionKwargs,
    Unpack,
    Qwen3RotaryEmbedding,
    Qwen3MLP,
    apply_rotary_pos_emb,
    can_return_tuple,
    eager_attention_forward
)


def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        if not isinstance(num_items_in_batch, torch.Tensor):
            num_items_in_batch = torch.tensor(num_items_in_batch, device=loss.device, dtype=loss.dtype)
        elif num_items_in_batch.device != loss.device:
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss


class UnmaskingQwen3Attention(Qwen3Attention):
    """Multi-headed attention without causal mask for bidirectional attention"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Use eager attention as default
        attention_interface: Callable = eager_attention_forward
        
        # Remove causal mask by setting attention_mask to None or creating a non-causal mask
        # For bidirectional attention, we don't want any masking except padding
        if attention_mask is not None and 0.0 in attention_mask:
            # Keep only padding mask if it exists, remove causal part
            # This allows tokens to attend to future tokens
            pass
        else:
            # If there's no padding, we can set attention_mask to None for full attention
            attention_mask = None

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class UnmaskingQwen3DecoderLayer(Qwen3DecoderLayer):

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super(Qwen3DecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = UnmaskingQwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class UnmaskingQwen3Model(Qwen3Model):

    def __init__(self, config: Qwen3Config):
        super(Qwen3PreTrainedModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [UnmaskingQwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        # Override the causal mask creation to create a non-causal mask
        # This allows bidirectional attention
        if attention_mask is None:
            # If no attention mask is provided, return None to allow full attention
            return None
            
        # If attention_mask is provided, it's likely for padding
        # Convert it to the right format but without the causal constraint
        dtype = input_tensor.dtype
        min_dtype = torch.finfo(dtype).min
        batch_size = input_tensor.shape[0]
        sequence_length = input_tensor.shape[1]
        
        if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2:
            # Convert 2D padding mask to 4D attention mask
            expanded_attn_mask = attention_mask[:, None, None, :]
            expanded_attn_mask = expanded_attn_mask.to(dtype=dtype)
            expanded_attn_mask = (1.0 - expanded_attn_mask) * min_dtype
            return expanded_attn_mask
        
        # If it's already 4D, return as is
        return attention_mask


class UnmaskingQwen3ForTokenClassification(Qwen3PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = UnmaskingQwen3Model(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        num_items_in_batch: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> TokenClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.view(-1, self.num_labels)
            labels = labels.view(-1).to(logits.device)
            logits = logits.float()
            loss = fixed_cross_entropy(logits, labels, num_items_in_batch, ignore_index)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
