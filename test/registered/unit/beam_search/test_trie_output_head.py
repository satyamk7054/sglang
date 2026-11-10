"""The streamed trie-head normalizer projects its tail chunk through the padded
LM-head rows (for an aligned GEMM width) and must slice the padding off before
the logsumexp, so the result equals an unpadded full-vocabulary logsumexp."""

import unittest

import torch

from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
from sglang.srt.beam_search.trie_constraint import BeamTrieConstraint
from sglang.srt.beam_search.trie_output_head import (
    BeamTrieOutputHead,
    NormalizerGapStats,
    normalizer_gap_stats,
    row_log_normalizer,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _config():
    return TrieOutputHeadConfig(
        tensor_path="/unused/trie_output_head.safetensors",
        token_start=8,
        codebook_size=4,
        num_codebooks=2,
    )


def _naive_normalizer(hidden_states, weight, vocab_size, transform):
    logits = transform(torch.matmul(hidden_states, weight[:vocab_size].T)).float()
    row_max = logits.max(dim=-1).values
    return torch.stack((row_max, torch.logsumexp(logits, dim=-1) - row_max), dim=1)


class TestNormalizerPaddedTail(CustomTestCase):
    def test_matches_naive_logsumexp(self):
        cases = [
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
        ]
        for vocab_size, weight_rows, chunk_size, expected_widths in cases:
            with self.subTest(vocab_size=vocab_size, weight_rows=weight_rows):
                torch.manual_seed(0)
                hidden_states = torch.randn(4, 6)
                weight = torch.randn(weight_rows, 6)
                # Large padding values make any leak into the normalizer show.
                weight[vocab_size:] = 50.0
                projected_widths = []

                def project(states, selected_weight):
                    projected_widths.append(selected_weight.shape[0])
                    return torch.matmul(states, selected_weight.T)

                def transform(logits):
                    return 2.0 * logits + 3.0

                head = BeamTrieOutputHead(
                    _config(), vocab_size=vocab_size, chunk_size=chunk_size
                )
                normalizer = head._stream_normalizer(
                    hidden_states, weight, project, transform
                )

                self.assertEqual(projected_widths, expected_widths)
                torch.testing.assert_close(
                    normalizer,
                    _naive_normalizer(hidden_states, weight, vocab_size, transform),
                    rtol=1e-5,
                    atol=1e-5,
                )


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


def _project(hidden_states, weight, depths, projected_widths):
    def project(states, selected_weight):
        projected_widths.append(selected_weight.shape[0])
        return torch.matmul(states, selected_weight.T)

    return BeamTrieOutputHead(_config(), vocab_size=16, chunk_size=4).project_mixed(
        hidden_states, weight, depths, project, _transform
    )


def _lse(normalizer):
    return normalizer[:, 0] + normalizer[:, 1]


def _transform(logits):
    return 2.0 * logits + 3.0


def _constraint():
    config = TrieOutputHeadConfig(
        tensor_path="", token_start=8, codebook_size=4, num_codebooks=3
    )
    root = torch.tensor([True, True, False, True])
    level1 = torch.ones(4, 4, dtype=torch.bool)
    level1[:, 0] = False
    level2 = torch.ones(16, 4, dtype=torch.bool)
    return BeamTrieConstraint(config, root, level1, level2, None, (4, 4, 4))


class TestCodebookNormalizer(CustomTestCase):
    def test_is_active_codebook_logsumexp(self):
        hidden_states, weight, depths = _compact_inputs()
        projected_widths = []
        with envs.SGLANG_OPT_BEAM_TRIE_CODEBOOK_NORMALIZER.override(True):
            output = _project(hidden_states, weight, depths, projected_widths)

        compact = _compact_reference(hidden_states, weight, depths, _transform)
        # Only the SID rows are projected: no full-vocabulary chunk runs.
        self.assertEqual(projected_widths, [8])
        self.assertTrue(torch.equal(output.logits, compact))
        self.assertEqual(output.normalizer.shape, (3, 2))
        self.assertEqual(output.normalizer.dtype, torch.float32)
        torch.testing.assert_close(
            _lse(output.normalizer), torch.logsumexp(compact, dim=-1)
        )
        self.assertTrue(
            torch.equal(output.normalizer[:, 0], compact.max(dim=-1).values)
        )

    def test_off_keeps_exact_full_vocabulary_scores(self):
        hidden_states, weight, depths = _compact_inputs()
        projected_widths = []
        output = _project(hidden_states, weight, depths, projected_widths)

        self.assertEqual(projected_widths, [8, 4, 4, 4, 4])
        self.assertTrue(
            torch.equal(
                output.logits,
                _compact_reference(hidden_states, weight, depths, _transform),
            )
        )
        torch.testing.assert_close(
            output.normalizer,
            _naive_normalizer(hidden_states, weight, 16, _transform),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_matches_exact_when_mass_is_in_codebook(self):
        hidden_states, weight, _ = _compact_inputs()
        hidden_states = hidden_states.abs() + 0.1
        weight = weight.abs()
        config = _config()
        active_end = config.token_start + config.codebook_size
        # Every row outside the active codebook 0 scores ~-1000 below it.
        weight[: config.token_start] = -100.0
        weight[active_end:] = -100.0
        depths = torch.zeros(3, dtype=torch.int64)

        exact = _project(hidden_states, weight, depths, [])
        with envs.SGLANG_OPT_BEAM_TRIE_CODEBOOK_NORMALIZER.override(True):
            approx = _project(hidden_states, weight, depths, [])

        self.assertTrue(torch.equal(exact.logits, approx.logits))
        torch.testing.assert_close(_lse(exact.normalizer), _lse(approx.normalizer))

    def test_dense_scores_use_codebook_normalizer_when_enabled(self):
        torch.manual_seed(0)
        constraint = _constraint()
        dense = torch.randn(3, 24)
        prefixes = torch.tensor([[0], [1], [3]])
        codebook = dense[:, 12:16]

        exact_v, exact_t = constraint.topk_logprobs([dense], prefixes, 1, 2)
        with envs.SGLANG_OPT_BEAM_TRIE_CODEBOOK_NORMALIZER.override(True):
            approx_v, approx_t = constraint.topk_logprobs([dense], prefixes, 1, 2)

        self.assertTrue(torch.equal(exact_t, approx_t))
        raw = torch.gather(codebook, 1, approx_t - 12)
        torch.testing.assert_close(
            approx_v, raw - torch.logsumexp(codebook, dim=-1, keepdim=True)
        )
        torch.testing.assert_close(
            exact_v, raw - torch.logsumexp(dense, dim=-1, keepdim=True)
        )

    def test_compact_and_dense_codebook_scores_agree(self):
        torch.manual_seed(1)
        constraint = _constraint()
        dense = torch.randn(3, 24)
        prefixes = torch.tensor([[0], [1], [3]])
        compact = dense[:, 12:16].clone()
        with envs.SGLANG_OPT_BEAM_TRIE_CODEBOOK_NORMALIZER.override(True):
            dense_v, dense_t = constraint.topk_logprobs([dense], prefixes, 1, 2)
            compact_v, compact_t = constraint.topk_logprobs(
                [compact], prefixes, 1, 2, [row_log_normalizer(compact)]
            )
        self.assertTrue(torch.equal(dense_t, compact_t))
        torch.testing.assert_close(dense_v, compact_v, atol=1e-6, rtol=0)


class TestNormalizerGapStats(CustomTestCase):
    def test_records_exact_minus_codebook_lse(self):
        torch.manual_seed(2)
        constraint = _constraint()
        dense = torch.randn(3, 24)
        prefixes = torch.tensor([[0], [1], [3]])
        with envs.SGLANG_DEBUG_BEAM_TRIE_NORMALIZER_GAP_INTERVAL.override(1000):
            stats = normalizer_gap_stats()
            self.assertIsNotNone(stats)
            stats.gaps.clear()
            constraint.topk_logprobs([dense], prefixes, 1, 2)
            with envs.SGLANG_OPT_BEAM_TRIE_CODEBOOK_NORMALIZER.override(True):
                self.assertIsNone(normalizer_gap_stats())
        gap = torch.cat(stats.gaps[1])
        expected = torch.logsumexp(dense, dim=-1) - torch.logsumexp(
            dense[:, 12:16], dim=-1
        )
        torch.testing.assert_close(gap, expected, atol=1e-6, rtol=0)
        self.assertTrue((gap > 0).all())
        [line] = stats.summarize()
        self.assertIn("depth=1 rows=3", line)
        stats.gaps.clear()

    def test_logs_and_resets_every_interval(self):
        stats = NormalizerGapStats(interval=2)
        stats.record(0, torch.tensor([1.0, 2.0]), torch.tensor([0.5, 2.0]))
        self.assertEqual(len(stats.gaps[0]), 1)
        stats.record(0, torch.tensor([1.0]), torch.tensor([1.0]))
        self.assertEqual(stats.gaps, {})


if __name__ == "__main__":
    unittest.main()
