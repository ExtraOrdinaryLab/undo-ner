from typing import Optional

import torch
import torch.nn as nn

from transformers.modeling_outputs import TokenClassifierOutput, BaseModelOutputWithPast
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2PreTrainedModel,
    Qwen2Model,
    SlidingWindowCache, 
    StaticCache
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


class UnmaskingQwen2Model(Qwen2Model):
    """
    UnmaskingQwen2Model is a modified version of Qwen2Model that removes the causal mask,
    allowing bidirectional attention similar to BERT-like models.
    """

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        """
        Override the causal mask creation to create a non-causal (bidirectional) mask.
        This allows each token to attend to all tokens in the sequence.
        """
        # For flash attention, just return None or the padding mask
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        
        # For flex attention, keep the same behavior but without causality
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                # We don't convert to causal mask here
                return attention_mask
            return attention_mask

        # For other attention implementations, create a non-causal mask
        batch_size = input_tensor.shape[0]
        sequence_length = input_tensor.shape[1]
        dtype = input_tensor.dtype
        
        # For SlidingWindowCache or StaticCache
        if isinstance(past_key_values, (SlidingWindowCache, StaticCache)):
            target_length = past_key_values.get_max_cache_shape()
        else:
            # For DynamicCache or no cache
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_key_values.get_seq_length() + sequence_length + 1
                if past_key_values is not None
                else sequence_length
            )
        
        # Create a non-causal mask (all zeros, allowing full attention)
        # Instead of using min_dtype to mask out future tokens, we use zeros to allow attention to all positions
        non_causal_mask = torch.zeros(
            (batch_size, 1, sequence_length, target_length),
            dtype=dtype,
            device=input_tensor.device,
        )
        
        # If there's a padding attention mask, apply it
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Convert 2D attention mask to 4D
                expanded_mask = attention_mask[:, None, None, :].expand(
                    batch_size, 1, sequence_length, attention_mask.shape[-1]
                ).to(non_causal_mask.device)
                
                # Apply padding mask (0 for tokens to attend to, large negative for padded positions)
                min_dtype = torch.finfo(dtype).min
                padding_mask = expanded_mask == 0
                non_causal_mask = non_causal_mask.masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                # If already 4D, use as is
                non_causal_mask = attention_mask
        
        return non_causal_mask


class UnmaskingQwen2ForTokenClassification(Qwen2PreTrainedModel):
    """
    Qwen2 model with a token classification head on top, but with bidirectional attention.
    This is achieved by using the UnmaskingQwen2Model which removes the causal mask.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # Use the UnmaskingQwen2Model instead of the standard Qwen2Model
        self.model = UnmaskingQwen2Model(config)
        
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
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> TokenClassifierOutput:
        """
        Forward pass for token classification with bidirectional attention.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past key values for efficient generation
            inputs_embeds: Pre-computed input embeddings
            labels: Token classification labels
            use_cache: Whether to use cache for efficient generation
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            flash_attn_kwargs: Additional arguments for flash attention
            
        Returns:
            TokenClassifierOutput with loss, logits, and optional hidden states and attentions
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
            **flash_attn_kwargs,
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
