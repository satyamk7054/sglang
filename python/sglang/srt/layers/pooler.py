# adapted from
# https://github.com/vllm-project/vllm/blob/82a1b1a82b1fbb454c82a9ef95730b929c9b270c/vllm/model_executor/layers/pooler.py

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import PoolerConfig

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from sglang.srt.layers.activation import get_cross_encoder_activation_function
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class PoolingType(IntEnum):
    LAST = 0
    CLS = 1
    MEAN = 2


@dataclass
class EmbeddingPoolerOutput:
    # Pooler can return list[tensor] instead of tensor if the dimension of each tensor in the batch is different
    # due to different per-request matryoshka dim truncation
    embeddings: torch.Tensor | list[torch.Tensor]


class Pooler(nn.Module):
    """A layer that pools specific information from hidden states.
    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.
    Attributes:
        pooling_type: The type of pooling to use (LAST, AVERAGE, MAX).
        normalize: Whether to normalize the pooled data.
    """

    def __init__(self, pooling_type: PoolingType, normalize: bool):
        super().__init__()
        self.pooling_type = pooling_type
        self.normalize = normalize

    @staticmethod
    def from_pooler_config(
        pooler_config: Optional["PoolerConfig"],
        default_type: PoolingType,
        default_normalize: bool,
    ) -> "Pooler":
        if pooler_config is not None:
            return Pooler(
                pooling_type=pooler_config.pooling_type,
                normalize=pooler_config.normalize,
            )
        return Pooler(pooling_type=default_type, normalize=default_normalize)

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> EmbeddingPoolerOutput:

        if self.pooling_type == PoolingType.LAST:
            last_token_indices = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            pooled_data = hidden_states[last_token_indices]
        elif self.pooling_type == PoolingType.CLS:
            prompt_lens = forward_batch.extend_seq_lens
            first_token_flat_indices = torch.zeros_like(prompt_lens)
            first_token_flat_indices[1:] += torch.cumsum(prompt_lens, dim=0)[:-1]
            pooled_data = hidden_states[first_token_flat_indices]
        elif self.pooling_type == PoolingType.MEAN:
            seq_lens = forward_batch.extend_seq_lens
            num_seqs = seq_lens.shape[0]
            hidden_dim = hidden_states.shape[-1]
            # Build segment IDs for each token
            segment_ids = torch.arange(
                num_seqs, device=seq_lens.device
            ).repeat_interleave(seq_lens)
            # Accumulate hidden states per segment in float32 to prevent overflow
            pooled_data = torch.zeros(
                num_seqs, hidden_dim, dtype=torch.float32, device=hidden_states.device
            )
            pooled_data.index_add_(0, segment_ids, hidden_states.to(torch.float32))
            # Divide by sequence lengths to get mean
            pooled_data = pooled_data / seq_lens.unsqueeze(1).to(torch.float32)
            pooled_data = pooled_data.to(hidden_states.dtype)
        else:
            raise ValueError(f"Invalid pooling type: {self.pooling_type}")

        if forward_batch.dimensions is not None:
            all_same_dimensions = len(set(forward_batch.dimensions)) == 1
            if all_same_dimensions:
                pooled_data = pooled_data[..., : forward_batch.dimensions[0]]
            else:
                pooled_data = [
                    tensor[..., :dim]
                    for tensor, dim in zip(pooled_data, forward_batch.dimensions)
                ]

        if self.normalize:
            if isinstance(pooled_data, list):
                pooled_data = [
                    nn.functional.normalize(tensor, p=2, dim=-1)
                    for tensor in pooled_data
                ]
            else:
                pooled_data = nn.functional.normalize(pooled_data, p=2, dim=-1)

        return EmbeddingPoolerOutput(embeddings=pooled_data)


class CrossEncodingPooler(nn.Module):
    """A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `EmbeddingPoolerOutput`.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        classifier: nn.Module,
        pooler: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.classifier = classifier
        self.pooler = pooler
        self.default_activation_function = get_cross_encoder_activation_function(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> EmbeddingPoolerOutput:
        """Pools sentence pair scores from the hidden_states."""

        prompt_lens = forward_batch.extend_seq_lens

        offset = 0
        pooled_data_lst = []
        for prompt_len in prompt_lens:
            pooled_data_i = hidden_states[offset : offset + prompt_len]

            if self.pooler is not None:
                final_shape_tensor = self.pooler(pooled_data_i, forward_batch)
            else:
                final_shape_tensor = self.classifier(pooled_data_i)

            pooled_data_lst.append(final_shape_tensor)
            offset += prompt_len

        pooled_output = torch.stack(pooled_data_lst)

        if self.pooler is not None:
            # apply classifier once on the full batch if possible
            pooled_output = self.classifier(pooled_output)

        scores = self.default_activation_function(pooled_output).squeeze(-1)

        return EmbeddingPoolerOutput(embeddings=scores)
