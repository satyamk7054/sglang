import sys

import pytest
import torch

from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
from sglang.srt.beam_search.trie_constraint import BeamTrieConstraint
from sglang.srt.beam_search.trie_output_head import (
    BeamTrieOutputHead,
    NormalizerGapStats,
    normalizer_gap_stats,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _config():
    return TrieOutputHeadConfig(
        tensor_path="/unused/trie_output_head.safetensors",
        token_start=8,
        codebook_size=4,
        num_codebooks=2,
    )


def test_full_projection_preserves_scores():
    torch.manual_seed(0)
    hidden_states = torch.randn(3, 5)
    weight = torch.randn(16, 5)
    depths = torch.tensor([0, 1, 0])

    def project(states, selected_weight):
        return torch.matmul(states, selected_weight.T)

    def transform(logits):
        return logits

    chunked = BeamTrieOutputHead(_config(), vocab_size=16, chunk_size=3)
    full = BeamTrieOutputHead(_config(), vocab_size=16, chunk_size=16)

    chunked_output = chunked.project_mixed(
        hidden_states,
        weight,
        depths,
        project,
        transform,
    )
    full_output = full.project_mixed(
        hidden_states,
        weight,
        depths,
        project,
        transform,
    )

    assert torch.equal(full_output.logits, chunked_output.logits)
    assert torch.allclose(
        full_output.normalizer,
        chunked_output.normalizer,
        rtol=1e-6,
        atol=1e-6,
    )


def _naive_normalizer(hidden_states, weight, vocab_size, transform):
    logits = transform(torch.matmul(hidden_states, weight[:vocab_size].T)).float()
    row_max = logits.max(dim=-1).values
    return torch.stack(
        (row_max, torch.logsumexp(logits, dim=-1) - row_max),
        dim=1,
    )


@pytest.mark.parametrize(
    "vocab_size,weight_rows,chunk_size,expected_widths",
    [
        # Unaligned tail projects through the padded rows: 100 -> 128.
        (100, 128, 32, [32, 32, 32, 32]),
        # Single unaligned chunk.
        (45, 64, 8192, [64]),
        # Padding shorter than the alignment is used as far as it goes.
        (70, 72, 64, [64, 8]),
        # Unpadded weight keeps the logical tail width.
        (45, 45, 16, [16, 16, 13]),
        # Aligned vocabulary is unchanged.
        (128, 192, 64, [64, 64]),
    ],
)
def test_normalizer_matches_naive_logsumexp_with_padded_tail(
    vocab_size, weight_rows, chunk_size, expected_widths
):
    torch.manual_seed(0)
    hidden_states = torch.randn(4, 6)
    weight = torch.randn(weight_rows, 6)
    # Padding rows hold large values so any leak into the normalizer shows.
    weight[vocab_size:] = 50.0
    projected_widths = []

    def project(states, selected_weight):
        projected_widths.append(selected_weight.shape[0])
        return torch.matmul(states, selected_weight.T)

    def transform(logits):
        return 2.0 * logits + 3.0

    head = BeamTrieOutputHead(_config(), vocab_size=vocab_size, chunk_size=chunk_size)
    normalizer = head._stream_normalizer(hidden_states, weight, project, transform)

    assert projected_widths == expected_widths
    assert torch.allclose(
        normalizer,
        _naive_normalizer(hidden_states, weight, vocab_size, transform),
        rtol=1e-5,
        atol=1e-5,
    )


def test_topk_logprob_skips_full_vocabulary_normalizer():
    torch.manual_seed(0)
    hidden_states = torch.randn(3, 5)
    weight = torch.randn(16, 5)
    depths = torch.tensor([0, 1, 0])
    projected_widths = []

    def project(states, selected_weight):
        projected_widths.append(selected_weight.shape[0])
        return torch.matmul(states, selected_weight.T)

    with envs.SGLANG_OPT_BEAM_TRIE_TOPK_LOGPROB.override(True):
        output = BeamTrieOutputHead(_config(), vocab_size=16).project_mixed(
            hidden_states,
            weight,
            depths,
            project,
            lambda logits: logits,
        )

    assert output.normalizer is None
    assert projected_widths == [8]


def _compact_inputs(seed=0):
    torch.manual_seed(seed)
    hidden_states = torch.randn(3, 5)
    weight = torch.randn(16, 5)
    depths = torch.tensor([0, 1, 0])
    return hidden_states, weight, depths


def _compact_reference(hidden_states, weight, depths, transform):
    config = _config()
    sid = transform(
        torch.matmul(hidden_states, weight[config.token_start : config.token_end].T)
    ).float()
    sid = sid.view(-1, config.num_codebooks, config.codebook_size)
    return sid[torch.arange(hidden_states.shape[0]), depths]


def _project(hidden_states, weight, depths, transform, projected_widths):
    def project(states, selected_weight):
        projected_widths.append(selected_weight.shape[0])
        return torch.matmul(states, selected_weight.T)

    return BeamTrieOutputHead(_config(), vocab_size=16, chunk_size=4).project_mixed(
        hidden_states, weight, depths, project, transform
    )


def _lse(normalizer):
    return normalizer[:, 0] + normalizer[:, 1]


def _transform(logits):
    return 2.0 * logits + 3.0


def test_codebook_normalizer_is_active_codebook_logsumexp():
    hidden_states, weight, depths = _compact_inputs()
    projected_widths = []
    with envs.SGLANG_OPT_BEAM_TRIE_CODEBOOK_NORMALIZER.override(True):
        output = _project(hidden_states, weight, depths, _transform, projected_widths)

    compact = _compact_reference(hidden_states, weight, depths, _transform)
    # Only the SID rows are projected: no full-vocabulary chunk runs.
    assert projected_widths == [8]
    assert torch.equal(output.logits, compact)
    assert output.normalizer.shape == (3, 2)
    assert output.normalizer.dtype == torch.float32
    assert torch.allclose(
        _lse(output.normalizer), torch.logsumexp(compact, dim=-1), atol=1e-5
    )
    assert torch.equal(output.normalizer[:, 0], compact.max(dim=-1).values)


def test_codebook_normalizer_off_keeps_exact_full_vocabulary_scores():
    hidden_states, weight, depths = _compact_inputs()
    projected_widths = []
    with envs.SGLANG_OPT_BEAM_TRIE_CODEBOOK_NORMALIZER.override(False):
        output = _project(hidden_states, weight, depths, _transform, projected_widths)

    assert projected_widths == [8, 4, 4, 4, 4]
    assert torch.equal(
        output.logits, _compact_reference(hidden_states, weight, depths, _transform)
    )
    assert torch.allclose(
        output.normalizer,
        _naive_normalizer(hidden_states, weight, 16, _transform),
        rtol=1e-5,
        atol=1e-5,
    )


def test_codebook_normalizer_matches_exact_when_mass_is_in_codebook():
    hidden_states, weight, _ = _compact_inputs()
    hidden_states = hidden_states.abs() + 0.1
    weight = weight.abs()
    config = _config()
    active_end = config.token_start + config.codebook_size
    # Every row outside the active codebook 0 scores ~-1000 below it.
    weight[: config.token_start] = -100.0
    weight[active_end:] = -100.0
    depths = torch.zeros(3, dtype=torch.int64)

    exact = _project(hidden_states, weight, depths, _transform, [])
    with envs.SGLANG_OPT_BEAM_TRIE_CODEBOOK_NORMALIZER.override(True):
        approx = _project(hidden_states, weight, depths, _transform, [])

    assert torch.equal(exact.logits, approx.logits)
    assert torch.allclose(_lse(exact.normalizer), _lse(approx.normalizer), atol=1e-5)


def test_codebook_and_topk_logprob_are_mutually_exclusive():
    hidden_states, weight, depths = _compact_inputs()
    with envs.SGLANG_OPT_BEAM_TRIE_CODEBOOK_NORMALIZER.override(
        True
    ), envs.SGLANG_OPT_BEAM_TRIE_TOPK_LOGPROB.override(True):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _project(hidden_states, weight, depths, _transform, [])


def _constraint():
    config = TrieOutputHeadConfig(
        tensor_path="", token_start=8, codebook_size=4, num_codebooks=3
    )
    root = torch.tensor([True, True, False, True])
    level1 = torch.ones(4, 4, dtype=torch.bool)
    level1[:, 0] = False
    level2 = torch.ones(16, 4, dtype=torch.bool)
    return BeamTrieConstraint(config, root, level1, level2, None, (4, 4, 4))


def test_dense_scores_use_codebook_normalizer_when_enabled():
    torch.manual_seed(0)
    constraint = _constraint()
    dense = torch.randn(3, 24)
    prefixes = torch.tensor([[0], [1], [3]])
    codebook = dense[:, 12:16]

    exact_v, exact_t = constraint.topk_logprobs([dense], prefixes, 1, 2)
    with envs.SGLANG_OPT_BEAM_TRIE_CODEBOOK_NORMALIZER.override(True):
        approx_v, approx_t = constraint.topk_logprobs([dense], prefixes, 1, 2)

    assert torch.equal(exact_t, approx_t)
    raw = torch.gather(codebook, 1, approx_t - 12)
    assert torch.allclose(
        approx_v, raw - torch.logsumexp(codebook, dim=-1, keepdim=True)
    )
    assert torch.allclose(exact_v, raw - torch.logsumexp(dense, dim=-1, keepdim=True))


def test_compact_and_dense_codebook_scores_agree():
    torch.manual_seed(1)
    constraint = _constraint()
    dense = torch.randn(3, 24)
    prefixes = torch.tensor([[0], [1], [3]])
    compact = dense[:, 12:16].clone()
    with envs.SGLANG_OPT_BEAM_TRIE_CODEBOOK_NORMALIZER.override(True):
        from sglang.srt.beam_search.trie_output_head import row_log_normalizer

        dense_v, dense_t = constraint.topk_logprobs([dense], prefixes, 1, 2)
        compact_v, compact_t = constraint.topk_logprobs(
            [compact], prefixes, 1, 2, [row_log_normalizer(compact)]
        )
    assert torch.equal(dense_t, compact_t)
    assert torch.allclose(dense_v, compact_v, atol=1e-6)


def test_normalizer_gap_diagnostic_records_exact_minus_codebook_lse():
    torch.manual_seed(2)
    constraint = _constraint()
    dense = torch.randn(3, 24)
    prefixes = torch.tensor([[0], [1], [3]])
    with envs.SGLANG_DEBUG_BEAM_TRIE_NORMALIZER_GAP_INTERVAL.override(1000):
        stats = normalizer_gap_stats()
        assert stats is not None
        stats.gaps.clear()
        constraint.topk_logprobs([dense], prefixes, 1, 2)
        with envs.SGLANG_OPT_BEAM_TRIE_CODEBOOK_NORMALIZER.override(True):
            assert normalizer_gap_stats() is None
    gap = torch.cat(stats.gaps[1])
    expected = torch.logsumexp(dense, dim=-1) - torch.logsumexp(dense[:, 12:16], dim=-1)
    assert torch.allclose(gap, expected, atol=1e-6)
    assert (gap > 0).all()
    [line] = stats.summarize()
    assert "depth=1 rows=3" in line
    stats.gaps.clear()


def test_normalizer_gap_diagnostic_is_off_by_default():
    assert envs.SGLANG_DEBUG_BEAM_TRIE_NORMALIZER_GAP_INTERVAL.get() == 0
    assert normalizer_gap_stats() is None


def test_normalizer_gap_stats_logs_and_resets_every_interval():
    stats = NormalizerGapStats(interval=2)
    stats.record(0, torch.tensor([1.0, 2.0]), torch.tensor([0.5, 2.0]))
    assert len(stats.gaps[0]) == 1
    stats.record(0, torch.tensor([1.0]), torch.tensor([1.0]))
    assert stats.gaps == {}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
