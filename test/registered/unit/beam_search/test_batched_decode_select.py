"""Batched trie decode selection: selecting same-shape beam groups together
must reproduce the per-group path bit for bit (tokens, parents, scores, tie
order, prefixes, KV remap, relayed tokens), and fall back for groups that
cannot share a launch."""

import random
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.beam_search import BeamGroup
from sglang.srt.beam_search.batch_tail import BeamTail, BeamTailEntry
from sglang.srt.beam_search.batched_select import rank_candidates_batched
from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.beam_search.joint_select import FinalSelect, _ranked_candidates
from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
from sglang.srt.beam_search.trie_constraint import BeamTrieConstraint
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CODEBOOK = 8
TOKEN_START = 100
NUM_CODEBOOKS = 3
MAX_LEN = 16


def make_trie(seed):
    gen = torch.Generator().manual_seed(seed)
    config = TrieOutputHeadConfig(
        tensor_path="",
        token_start=TOKEN_START,
        codebook_size=CODEBOOK,
        num_codebooks=NUM_CODEBOOKS,
    )

    def masks(rows):
        mask = torch.rand(rows, CODEBOOK, generator=gen) < 0.6
        mask[torch.arange(rows), torch.randint(CODEBOOK, (rows,), generator=gen)] = True
        return mask

    return BeamTrieConstraint(
        config,
        masks(1)[0],
        masks(CODEBOOK),
        masks(CODEBOOK * CODEBOOK),
        None,
        (CODEBOOK, CODEBOOK, CODEBOOK),
    )


class StashRecorder:
    def __init__(self):
        self.calls = []

    def stash(self, rows, payload):
        self.calls.append((rows.clone(), payload.bonus_tokens.clone()))

    def relayed(self):
        out = {}
        for rows, tokens in self.calls:
            out.update(zip(rows.tolist(), tokens.tolist()))
        return out


def quantized(gen, *shape, ties):
    # Coarse values make many exact ties inside and across rows.
    if ties:
        return torch.randint(-3, 1, shape, generator=gen).float()
    return torch.randn(*shape, generator=gen)


def build_case(spec, seed, *, ties, dense):
    """spec: per-group dicts with beam_width, depth, prompt_len and optional
    retired / device_history; returns (coordinator, batch, logits_output)."""
    gen = torch.Generator().manual_seed(seed)
    trie = make_trie(seed)
    num_rows = sum(s["beam_width"] for s in spec)
    req_to_token = torch.randperm(num_rows * MAX_LEN, generator=gen).view(
        num_rows, MAX_LEN
    )
    coordinator = BeamCoordinator(
        model_config=SimpleNamespace(beam_trie_config=None),
        spec_algorithm=None,
        dllm_enabled=False,
        max_req_len=0,
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token, device="cpu"),
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        future_map=StashRecorder(),
    )
    coordinator.trie_constraint = trie

    entries, next_row, tail_rows = [], 0, 0
    for gi, s in enumerate(spec):
        k, depth = s["beam_width"], s["depth"]
        group = BeamGroup(
            beam_width=k,
            stop_token_ids=[2],
            max_new_tokens=s.get("max_new_tokens", NUM_CODEBOOKS),
            device_history=s.get("device_history", True),
        )
        group.num_generated = depth
        group.prompt_len = s["prompt_len"]
        group.all_rows = torch.arange(next_row, next_row + k)
        next_row += k
        group.frontier_cum_logprobs = quantized(gen, k, ties=ties)
        group.trie_prefix_codes = valid_prefixes(trie, gen, k, depth)
        group.leader = SimpleNamespace(
            kv=SimpleNamespace(kv_committed_len=s["prompt_len"] + depth),
            output_ids=[],
        )
        group.retired = s.get("retired", False)
        entries.append(BeamTailEntry(group, gi, tail_rows, tail_rows + k - 1))
        tail_rows += k - 1

    width = TOKEN_START + NUM_CODEBOOKS * CODEBOOK + 5 if dense else CODEBOOK
    capture = SimpleNamespace(
        leader_logits=quantized(gen, len(spec), width, ties=ties),
        tail_logits=quantized(gen, tail_rows, width, ties=ties),
        leader_normalizer=None if dense else quantized(gen, len(spec), 2, ties=ties),
        tail_normalizer=None if dense else quantized(gen, tail_rows, 2, ties=ties),
    )
    batch = SimpleNamespace(
        beam_tail=BeamTail(num_base_rows=len(spec), entries=entries),
        forward_iter=7,
    )
    return coordinator, batch, SimpleNamespace(beam=capture)


def valid_prefixes(trie, gen, k, depth):
    rows = []
    for _ in range(k):
        codes = []
        for d in range(depth):
            mask = trie.valid_child_mask(
                torch.tensor([codes], dtype=torch.int64).view(1, d), d
            )[0]
            choices = mask.nonzero().flatten()
            codes.append(int(choices[torch.randint(len(choices), (1,), generator=gen)]))
        rows.append(codes)
    return torch.tensor(rows, dtype=torch.int64).view(k, depth)


def run_select(spec, seed, *, batched, ties=True, dense=False):
    coordinator, batch, logits_output = build_case(spec, seed, ties=ties, dense=dense)
    coordinator.batched_decode_selection = batched
    with mock.patch.object(
        BeamCoordinator,
        "_select_bucket",
        autospec=True,
        side_effect=BeamCoordinator._select_bucket,
    ) as bucket_spy:
        coordinator.select_and_relay_decode(batch, logits_output)
    groups = [entry.group for entry in batch.beam_tail.entries]
    return coordinator, groups, bucket_spy.call_count


def staged_tensors(group):
    out = []
    for tick, sel in group._pending_steps:
        if isinstance(sel, FinalSelect):
            out.append((tick, sel.tokens, sel.parent_idx, sel.cum_logprobs))
        else:
            out.append(
                (
                    tick,
                    sel.next_tokens,
                    sel.parent_idx,
                    sel.new_cum_logprobs,
                    sel.num_survivors,
                    sel.num_finished,
                )
            )
    return out


class TestBatchedDecodeSelectParity(CustomTestCase):
    def assert_same(self, a, b, msg):
        self.assertEqual(type(a), type(b), msg)
        if torch.is_tensor(a):
            self.assertEqual(a.dtype, b.dtype, msg)
            self.assertTrue(torch.equal(a, b), f"{msg}: {a} != {b}")
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b), msg)
            for x, y in zip(a, b):
                self.assert_same(x, y, msg)
        else:
            self.assertEqual(a, b, msg)

    def check_parity(self, spec, seed, *, expect_buckets, **kwargs):
        ref, ref_groups, ref_buckets = run_select(spec, seed, batched=False, **kwargs)
        new, new_groups, new_buckets = run_select(spec, seed, batched=True, **kwargs)
        self.assertEqual(ref_buckets, 0)
        self.assertEqual(new_buckets, expect_buckets, f"seed={seed}")
        msg = f"seed={seed} spec={spec}"
        self.assertEqual(new.future_map.relayed(), ref.future_map.relayed(), msg)
        self.assert_same(
            new.req_to_token_pool.req_to_token, ref.req_to_token_pool.req_to_token, msg
        )
        for gi, (g_new, g_ref) in enumerate(zip(new_groups, ref_groups)):
            gmsg = f"{msg} group={gi}"
            self.assertEqual(g_new.num_generated, g_ref.num_generated, gmsg)
            self.assert_same(staged_tensors(g_new), staged_tensors(g_ref), gmsg)
            self.assert_same(
                g_new.frontier_cum_logprobs, g_ref.frontier_cum_logprobs, gmsg
            )
            self.assert_same(g_new.trie_prefix_codes, g_ref.trie_prefix_codes, gmsg)
            self.assertEqual(g_new.leader.output_ids, g_ref.leader.output_ids, gmsg)
            self.assert_same(
                [(e.tick, e.old_mapping, e.new_mapping) for e in g_new.pending_orphans],
                [(e.tick, e.old_mapping, e.new_mapping) for e in g_ref.pending_orphans],
                gmsg,
            )

    def test_uniform_groups_with_ties(self):
        spec = [dict(beam_width=4, depth=1, prompt_len=5) for _ in range(5)]
        for seed in range(20):
            self.check_parity(spec, seed, expect_buckets=1)

    def test_no_ties(self):
        spec = [dict(beam_width=4, depth=1, prompt_len=3 + g) for g in range(4)]
        for seed in range(10):
            self.check_parity(spec, seed, expect_buckets=1, ties=False)

    def test_final_step(self):
        spec = [dict(beam_width=3, depth=2, prompt_len=4) for _ in range(4)]
        for seed in range(10):
            self.check_parity(spec, seed, expect_buckets=1)

    def test_dense_logits_without_normalizers(self):
        spec = [dict(beam_width=4, depth=1, prompt_len=5) for _ in range(3)]
        for seed in range(10):
            self.check_parity(spec, seed, expect_buckets=1, dense=True)

    def test_mixed_groups_bucket_and_fall_back(self):
        spec = [
            dict(beam_width=4, depth=1, prompt_len=5),
            dict(beam_width=3, depth=1, prompt_len=5),  # singleton width
            dict(beam_width=4, depth=1, prompt_len=5, retired=True),
            dict(beam_width=4, depth=2, prompt_len=4),  # final bucket
            dict(beam_width=4, depth=1, prompt_len=6),
            dict(beam_width=4, depth=1, prompt_len=5, device_history=False),
            dict(beam_width=4, depth=2, prompt_len=4),
            dict(beam_width=4, depth=1, prompt_len=5, max_new_tokens=4),
            dict(beam_width=4, depth=2, prompt_len=4, max_new_tokens=4),
        ]
        for seed in range(20):
            # Buckets: depth-1 non-final (groups 0, 4, 7) and depth-2 final
            # (3, 6); the rest select per group.
            self.check_parity(spec, seed, expect_buckets=2)

    def test_random_specs(self):
        rng = random.Random(0)
        for seed in range(40):
            spec = [
                dict(
                    beam_width=rng.choice([2, 4]),
                    depth=rng.choice([1, 2]),
                    prompt_len=rng.choice([3, 5]),
                    retired=rng.random() < 0.15,
                )
                for _ in range(rng.randint(1, 7))
            ]
            ref = run_select(spec, seed, batched=True)
            self.check_parity(spec, seed, expect_buckets=ref[2], ties=True)

    def test_kill_switch(self):
        for disabled in (False, True):
            with envs.SGLANG_DISABLE_BEAM_BATCHED_DECODE_SELECTION.override(disabled):
                coordinator, _, _ = build_case(
                    [dict(beam_width=2, depth=1, prompt_len=3)],
                    0,
                    ties=True,
                    dense=False,
                )
            self.assertEqual(coordinator.batched_decode_selection, not disabled)


class TestRankCandidatesBatched(CustomTestCase):
    def test_matches_per_group_with_ties_and_neg_inf(self):
        gen = torch.Generator().manual_seed(0)
        for trial in range(50):
            num_groups, k, c = 4, 5, 6
            values = torch.randint(-2, 1, (num_groups, k, c), generator=gen).float()
            values[torch.rand(values.shape, generator=gen) < 0.2] = -torch.inf
            tokens = torch.randint(0, 50, (num_groups, k, c), generator=gen)
            cum = torch.randint(-2, 1, (num_groups, k), generator=gen).float()
            scores, parent, out_tokens = rank_candidates_batched(
                cum,
                values[:, 0],
                values[:, 1:],
                tokens[:, 0],
                tokens[:, 1:],
                num_out=k,
            )
            for g in range(num_groups):
                ref = _ranked_candidates(cum[g], values[g], tokens[g], k)
                for actual, expected in zip((scores[g], parent[g], out_tokens[g]), ref):
                    self.assertTrue(torch.equal(actual, expected), f"trial={trial}")


if __name__ == "__main__":
    unittest.main()
