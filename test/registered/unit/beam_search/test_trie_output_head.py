import sys

import pytest
import torch

from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
from sglang.srt.beam_search.trie_output_head import BeamTrieOutputHead
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
