# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Compact SID projection with a streamed full-vocabulary (or, opt-in,
active-codebook) log-softmax normalizer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import msgspec
import torch

from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

_DEFAULT_NORMALIZER_CHUNK_SIZE = 8192
_NORMALIZER_TAIL_ALIGNMENT = 64
logger = logging.getLogger(__name__)


class BeamTrieHeadOutput(msgspec.Struct, frozen=True):
    logits: torch.Tensor
    # [rows, 2] fp32 (max, logsumexp - max); None scores over the top-k only.
    normalizer: Optional[torch.Tensor]


def use_topk_logprob() -> bool:
    return envs.SGLANG_OPT_BEAM_TRIE_TOPK_LOGPROB.get()


def use_codebook_normalizer() -> bool:
    enabled = envs.SGLANG_OPT_BEAM_TRIE_CODEBOOK_NORMALIZER.get()
    if enabled and use_topk_logprob():
        raise ValueError(
            "SGLANG_OPT_BEAM_TRIE_CODEBOOK_NORMALIZER and "
            "SGLANG_OPT_BEAM_TRIE_TOPK_LOGPROB are mutually exclusive"
        )
    return enabled


def row_log_normalizer(logits: torch.Tensor) -> torch.Tensor:
    """Per-row ``[rows, 2]`` fp32 ``(max, logsumexp - max)`` of 2D logits."""
    row_max, row_log_sum = _row_lse_pair(logits)
    return torch.stack((row_max, row_log_sum), dim=1)


def _row_lse_pair(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.is_cuda:
        from sglang.srt.layers.logsumexp import row_logsumexp

        return row_logsumexp(logits)
    logits = logits.float()
    row_max = logits.max(dim=-1).values
    row_log_sum = torch.log(torch.exp(logits - row_max[:, None]).sum(dim=-1))
    return row_max, row_log_sum


def configure_model_trie_output_heads(
    *,
    model: torch.nn.Module,
    config: Optional[TrieOutputHeadConfig],
    device: str,
    tp_size: int,
    quantization: Optional[str],
    lora_enabled: bool,
    rl_on_policy_target: Optional[str],
) -> bool:
    from sglang.srt.layers.logits_processor import LogitsProcessor

    enabled = config is not None and (
        device == "cuda"
        and tp_size == 1
        and quantization is None
        and not lora_enabled
        and rl_on_policy_target is None
    )
    if config is not None and not enabled:
        logger.info(
            "Using dense LM-head scoring for model-bundled beam trie; "
            "the compact output head requires CUDA, TP=1, an unquantized "
            "LM head, and no LoRA or RL target."
        )
    configured = 0
    for module in model.modules():
        if isinstance(module, LogitsProcessor):
            module.configure_beam_trie(config if enabled else None)
            configured += 1
    if enabled and configured == 0:
        raise RuntimeError(
            "A model-bundled beam trie was discovered, but the loaded "
            "generation model has no compatible LogitsProcessor"
        )
    return enabled


def build_trie_row_levels(
    batch: ScheduleBatch, model_runner: ModelRunner
) -> Optional[torch.Tensor]:
    """Per-row SID codebook depths of an all-beam decode batch on the compact
    head; None runs the dense LM head."""
    if (
        batch.forward_mode.is_decode()
        and batch.beam_tail is not None
        and model_runner.beam_trie_compact_enabled
        and batch.reqs
        and all(req.beam_group is not None for req in batch.reqs)
    ):
        beam_trie_config = model_runner.model_config.beam_trie_config
        assert beam_trie_config is not None
        max_level = beam_trie_config.num_codebooks - 1
        row_levels = [
            min(req.beam_group.num_generated, max_level) for req in batch.reqs
        ]
        if batch.beam_tail is not None:
            for entry in batch.beam_tail.entries:
                level = min(entry.group.num_generated, max_level)
                row_levels.extend([level] * (entry.end - entry.start))
        assert len(row_levels) == len(batch.seq_lens)
        return torch.tensor(
            row_levels,
            dtype=torch.int64,
            device=batch.device,
        )
    return None


class BeamTrieOutputHead:
    """Projects the active SID codebook and optional full-vocabulary scores."""

    def __init__(
        self,
        config: TrieOutputHeadConfig,
        vocab_size: int,
        chunk_size: int = _DEFAULT_NORMALIZER_CHUNK_SIZE,
    ):
        self.config = config
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size

    def project_mixed(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        depths: torch.Tensor,
        project: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        transform: Callable[[torch.Tensor], torch.Tensor],
        use_grouped_projection: bool = False,
    ) -> BeamTrieHeadOutput:
        start = self.config.token_start
        end = self.config.token_end
        logits = None
        if use_grouped_projection:
            from sglang.kernels.ops.gemm.codebook_projection import (
                project_active_codebook_logits,
            )

            logits = project_active_codebook_logits(
                hidden_states,
                weight[start:end],
                depths,
                self.config.codebook_size,
            )
        if logits is None:
            all_logits = project(hidden_states, weight[start:end]).view(
                hidden_states.shape[0],
                self.config.num_codebooks,
                self.config.codebook_size,
            )
            row_indices = torch.arange(
                hidden_states.shape[0], device=hidden_states.device
            )
            logits = all_logits[row_indices, depths]
        # Match the dense path: logit_scale runs in the LM-head dtype, then
        # scoring and softcapping consume FP32 logits.
        logits = transform(logits).float()
        if use_codebook_normalizer():
            # Approximates log Z by the active codebook alone; the error per
            # row is log(1 + outside_mass / codebook_mass).
            normalizer = row_log_normalizer(logits)
        elif use_topk_logprob():
            normalizer = None
        else:
            normalizer = self._stream_normalizer(
                hidden_states,
                weight,
                project,
                transform,
            )
        return BeamTrieHeadOutput(logits, normalizer)

    def _stream_normalizer(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        project: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        if self.vocab_size > weight.shape[0]:
            raise RuntimeError(
                f"Model vocabulary ({self.vocab_size}) exceeds LM-head rows "
                f"({weight.shape[0]})"
            )

        # LM-head rows are padded past the vocabulary (VocabParallelEmbedding
        # pads to a multiple of 64 and zero-fills). Projecting the tail chunk
        # through the padded end keeps its GEMM width aligned; an unaligned
        # width drops cuBLAS to a several-times-slower kernel. The padded
        # columns are sliced off before the logsumexp.
        aligned_vocab_end = min(
            -(-self.vocab_size // _NORMALIZER_TAIL_ALIGNMENT)
            * _NORMALIZER_TAIL_ALIGNMENT,
            weight.shape[0],
        )
        row_max = None
        row_log_sum = None
        for start in range(0, self.vocab_size, self.chunk_size):
            logical_end = min(start + self.chunk_size, self.vocab_size)
            physical_end = (
                aligned_vocab_end if logical_end == self.vocab_size else logical_end
            )
            chunk_logits = project(hidden_states, weight[start:physical_end])
            chunk_logits = transform(chunk_logits)[:, : logical_end - start]
            chunk_max, chunk_log_sum = _row_lse_pair(chunk_logits)

            if row_max is None:
                row_max = chunk_max
                row_log_sum = chunk_log_sum
            else:
                next_max = torch.maximum(row_max, chunk_max)
                row_log_sum = torch.logaddexp(
                    row_log_sum + row_max - next_max,
                    chunk_log_sum + chunk_max - next_max,
                )
                row_max = next_max

        assert row_max is not None and row_log_sum is not None
        return torch.stack((row_max, row_log_sum), dim=1)


class NormalizerGapStats:
    """Windowed per-depth stats of ``exact_lse - codebook_lse`` per SID row.

    The gap is the codebook normalizer's error, log(1 + outside / codebook
    mass). Recording copies each gap to the host, so it is debug-only.
    """

    def __init__(self, interval: int):
        self.interval = interval
        self.records = 0
        self.gaps: dict[int, list[torch.Tensor]] = {}

    def record(self, depth: int, exact_lse: torch.Tensor, codebook_lse: torch.Tensor):
        if exact_lse.is_cuda and torch.cuda.is_current_stream_capturing():
            return
        gap = (exact_lse.float() - codebook_lse.float()).reshape(-1)
        self.gaps.setdefault(depth, []).append(gap.detach().cpu())
        self.records += 1
        if self.records % self.interval == 0:
            for line in self.summarize():
                logger.info(line)
            self.gaps.clear()

    def summarize(self) -> list[str]:
        lines = []
        for depth in sorted(self.gaps):
            gaps = torch.cat(self.gaps[depth])
            gaps = gaps[torch.isfinite(gaps)]
            if gaps.numel() == 0:
                continue
            ordered = gaps.sort().values
            n = ordered.numel()

            def quantile(q: float) -> float:
                return ordered[min(n - 1, int(q * (n - 1) + 0.5))].item()

            outside_fraction = -torch.expm1(-gaps)
            lines.append(
                f"Beam trie normalizer gap depth={depth} rows={n} "
                f"mean={gaps.mean().item():.4g} p50={quantile(0.5):.4g} "
                f"p99={quantile(0.99):.4g} max={ordered[-1].item():.4g} "
                f"outside_mass_mean={outside_fraction.mean().item():.4g}"
            )
        return lines


_normalizer_gap_stats: Optional[NormalizerGapStats] = None


def normalizer_gap_stats() -> Optional[NormalizerGapStats]:
    """The process-wide gap recorder, or None when the diagnostic is off or
    the scores are already approximate."""
    global _normalizer_gap_stats
    interval = envs.SGLANG_DEBUG_BEAM_TRIE_NORMALIZER_GAP_INTERVAL.get()
    if interval <= 0 or use_topk_logprob() or use_codebook_normalizer():
        return None
    if _normalizer_gap_stats is None or _normalizer_gap_stats.interval != interval:
        _normalizer_gap_stats = NormalizerGapStats(interval)
    return _normalizer_gap_stats
