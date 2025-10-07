# adapted from
# https://github.com/vllm-project/vllm/blob/82a1b1a82b1fbb454c82a9ef95730b929c9b270c/vllm/model_executor/layers/pooler.py

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from sglang.srt.layers.activation import get_cross_encoder_activation_function
from sglang.srt.model_executor.model_runner import ForwardBatch


class PoolingType(IntEnum):
    LAST = 0
    CLS = 1
    MEAN = 2


class PoolerConfig:
    def __init__(
        self, pooling_type: PoolingType | None = None, normalize: bool | None = None
    ):
        self.pooling_type = pooling_type

        # None is different from False because different models have different defaults
        # for unset 'normalize' config to maintain backward compatibility
        self.normalize = normalize

    @staticmethod
    def from_server_args(server_args: ServerArgs):
        return PoolerConfig(
            pooling_type=(
                PoolingType[server_args.pooling_type]
                if server_args.pooling_type
                else None
            ),
        )

    def merge_with_defaults(
        self, pooling_type: PoolingType, normalize: bool
    ) -> "PoolerConfig":
        """Method to merge with model-specific defaults if the config(s) are not passed by the user"""

        self.pooling_type = self.pooling_type or pooling_type
        self.normalize = self.normalize if self.normalize is not None else normalize

        return self


@dataclass
class EmbeddingPoolerOutput:
    embeddings: torch.Tensor


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
    def from_pooler_config(config: PoolerConfig) -> "Pooler":
        return Pooler(
            pooling_type=config.pooling_type,
            normalize=config.normalize,
        )

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
            extend_seq_lens = forward_batch.extend_seq_lens

            num_seqs = extend_seq_lens.numel()
            hidden_size = hidden_states.size(1)
            num_tokens = forward_batch.seq_lens_sum

            # Build a segment id per token: [0,0,...,1,1,...,2,2,...]
            segment_ids = torch.arange(
                num_seqs, device=hidden_states.device
            ).repeat_interleave(extend_seq_lens, output_size=num_tokens)

            # Use fp32 for mean pooling otherwise overflow is possible
            hidden_states = hidden_states.to(torch.float32)
            sums = torch.zeros(
                num_seqs,
                hidden_size,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

            sums.index_add_(0, segment_ids, hidden_states)

            pooled_data = sums / extend_seq_lens.unsqueeze(1)
        else:
            raise ValueError(f"Invalid pooling type: {self.pooling_type}")

        if self.normalize:
            pooled_data = nn.functional.normalize(pooled_data, p=2, dim=1)

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
