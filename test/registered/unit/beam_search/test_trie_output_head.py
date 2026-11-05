"""The streamed trie-head normalizer projects its tail chunk through the padded
LM-head rows (for an aligned GEMM width) and must slice the padding off before
the logsumexp, so the result equals an unpadded full-vocabulary logsumexp."""

import unittest

import torch

from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
from sglang.srt.beam_search.trie_output_head import BeamTrieOutputHead
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


if __name__ == "__main__":
    unittest.main()
