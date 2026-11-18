"""Stop trimming must use each beam's own finish reason, not the leader's.

A group's returned beams mix stop-finished and length-finished ones, so a shared
reason either drops a real token from a length-finished beam or leaks a stop
token into a matched one.
"""

import json
import pickle
import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import msgspec
import torch

from sglang.srt.beam_search import BeamGroup, joint_select, select_final_topk
from sglang.srt.beam_search.beam_group import BeamGroupState
from sglang.srt.beam_search.output import (
    decode_beam_search_output,
    pack_beam_search_output,
    try_build_beam_search_out_dict,
)
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.incremental_detokenize import DetokenizeHeader
from sglang.srt.managers.io_struct import (
    BatchStrOutput,
    BatchTokenIDOutput,
    BeamSearchOutput,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

STOP_ID = 99


def _trim(output, finished_reason, no_stop_trim):
    stub = SimpleNamespace(
        is_tool_call_parser_gpt_oss=False, tool_close_token_ids=frozenset()
    )
    return DetokenizerManager.trim_matched_stop(
        stub, output, finished_reason, no_stop_trim
    )


def _header(*, no_stop_trim):
    return DetokenizeHeader(
        prompt_tail_ids=array("q"),
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        no_stop_trim=no_stop_trim,
        http_worker_ipc=None,
    )


class _IdTokenizer:
    """Renders the token list it was handed, so assertions read the trim result."""

    def decode(self, tokens, **kwargs):
        return ",".join(str(t) for t in tokens)

    def batch_decode(self, token_lists, **kwargs):
        return [self.decode(t) for t in token_lists]


def _select(cum, logprobs, tokens, k):
    return joint_select(
        torch.tensor(cum, dtype=torch.float32),
        torch.tensor(logprobs, dtype=torch.float32),
        torch.tensor(tokens, dtype=torch.int64),
        torch.tensor([STOP_ID], dtype=torch.int64),
        k,
    )


def _mixed_group(*, stop_wins: bool) -> BeamGroup:
    """A finished group with one stop-matched and one length-finished beam;
    stop_wins picks which of the two scores higher, i.e. the leader's reason."""
    group = BeamGroup(beam_width=2, stop_token_ids=[STOP_ID], max_new_tokens=3)
    group.advance(_select([0.0], [[-0.1, -0.2, -0.3, -0.4]], [[1, 2, 3, 4]], 2))

    # Bounded on both sides: low enough that the 3-token length beam outscores it
    # once normalized (-0.12), high enough to stay inside the examined window --
    # a stop candidate ranked past k survivors never finishes at all.
    stop_logprob = -0.05 if stop_wins else -0.15
    group.advance(
        _select(
            [-0.1, -0.2],
            [[stop_logprob, -0.2, -0.5, -0.9], [-0.11, -0.4, -0.8, -1.2]],
            [[STOP_ID, 5, 6, 7], [8, 9, 10, 11]],
            2,
        )
    )
    assert len(group.completed) == 1, "the stop candidate should have finished"

    # max_new_tokens: the surviving frontier finishes by length.
    group.advance_final(
        select_final_topk(
            group.frontier_cum_logprobs,
            torch.tensor([[-0.1, -0.9], [-0.05, -0.9]], dtype=torch.float32),
            torch.tensor([[12, 13], [14, 15]], dtype=torch.int64),
            2,
        )
    )
    group.final_results = group.finalize()
    return group


def _decode(group, *, disable_batch_decode):
    packed = pack_beam_search_output(SimpleNamespace(beam_group=group))
    recv_obj = SimpleNamespace(
        beam_search_output=[packed],
        # The leader's reason, which the trim must not consult.
        finished_reasons=[{"type": "stop", "matched": STOP_ID}],
    )
    decode_beam_search_output(
        recv_obj,
        [_header(no_stop_trim=False)],
        tokenizer=_IdTokenizer(),
        disable_batch_decode=disable_batch_decode,
        trim_matched_stop=_trim,
    )
    return {tuple(s.tokens): s.text for s in packed.sequences}


class TestDecodeBeamSearchOutput(CustomTestCase):
    def _assert_mixed_group_trims(self, *, stop_wins, disable_batch_decode):
        group = _mixed_group(stop_wins=stop_wins)
        texts = _decode(group, disable_batch_decode=disable_batch_decode)
        matched = [r for r in group.final_results if r.matched_token is not None]
        length = [r for r in group.final_results if r.matched_token is None]
        self.assertEqual(len(matched), 1, group.final_results)
        self.assertEqual(len(length), 1, group.final_results)

        # The stop token is trimmed off the matched beam...
        stop_tokens = tuple(matched[0].tokens)
        self.assertEqual(stop_tokens[-1], STOP_ID)
        self.assertEqual(texts[stop_tokens], ",".join(map(str, stop_tokens[:-1])))
        # ...and the length-finished beam keeps every token.
        len_tokens = tuple(length[0].tokens)
        self.assertEqual(texts[len_tokens], ",".join(map(str, len_tokens)))

    def test_leader_matched_does_not_trim_the_length_beam(self):
        for disable_batch_decode in (True, False):
            with self.subTest(disable_batch_decode=disable_batch_decode):
                self._assert_mixed_group_trims(
                    stop_wins=True, disable_batch_decode=disable_batch_decode
                )

    def test_leader_length_still_trims_the_matched_beam(self):
        for disable_batch_decode in (True, False):
            with self.subTest(disable_batch_decode=disable_batch_decode):
                self._assert_mixed_group_trims(
                    stop_wins=False, disable_batch_decode=disable_batch_decode
                )

    def test_no_stop_trim_keeps_the_stop_token(self):
        group = _mixed_group(stop_wins=True)
        packed = pack_beam_search_output(SimpleNamespace(beam_group=group))
        recv_obj = SimpleNamespace(
            beam_search_output=[packed],
            finished_reasons=[{"type": "stop", "matched": STOP_ID}],
        )
        decode_beam_search_output(
            recv_obj,
            [_header(no_stop_trim=True)],
            tokenizer=_IdTokenizer(),
            disable_batch_decode=True,
            trim_matched_stop=_trim,
        )
        for seq in packed.sequences:
            self.assertEqual(seq.text, ",".join(map(str, seq.tokens)))

    def test_non_beam_item_in_a_mixed_batch_is_skipped(self):
        group = _mixed_group(stop_wins=True)
        packed = pack_beam_search_output(SimpleNamespace(beam_group=group))
        recv_obj = SimpleNamespace(
            beam_search_output=[None, packed],
            finished_reasons=[None, {"type": "stop", "matched": STOP_ID}],
        )
        decode_beam_search_output(
            recv_obj,
            [_header(no_stop_trim=False), _header(no_stop_trim=False)],
            tokenizer=_IdTokenizer(),
            disable_batch_decode=False,
            trim_matched_stop=_trim,
        )
        self.assertTrue(all(s.text is not None for s in packed.sequences))

    def test_beam_row_without_detokenize_state_is_skipped(self):
        """A row with no header gets no beam text instead of failing the whole batch."""
        for disable_batch_decode in (True, False):
            with self.subTest(disable_batch_decode=disable_batch_decode):
                group = _mixed_group(stop_wins=True)
                packed = pack_beam_search_output(SimpleNamespace(beam_group=group))
                recv_obj = SimpleNamespace(
                    beam_search_output=[packed],
                    finished_reasons=[{"type": "stop", "matched": STOP_ID}],
                )
                decode_beam_search_output(
                    recv_obj,
                    [None],
                    tokenizer=_IdTokenizer(),
                    disable_batch_decode=disable_batch_decode,
                    trim_matched_stop=_trim,
                )
                self.assertTrue(all(s.text is None for s in packed.sequences))


def _finished_trie_group(cum_logprobs, *, num_return, length_penalty=1.0, seed=0):
    """A device-history group finished by length with the given final cums."""
    k, length = len(cum_logprobs), 3
    gen = torch.Generator().manual_seed(seed)
    group = BeamGroup(
        beam_width=k,
        max_new_tokens=length,
        num_return=num_return,
        length_penalty=length_penalty,
        device_history=True,
    )
    group._device_final_tokens = torch.randint(0, 1000, (k, length), generator=gen)
    group._device_final_cum_logprobs = torch.tensor(cum_logprobs, dtype=torch.float32)
    group.num_committed = length
    group.state = BeamGroupState.FINISHED
    return group


def _out_dict(group, *, columnar, recv_cls, ipc):
    """The beam out dict a finished group yields at the tokenizer manager."""
    if columnar:
        group.final_columnar = group.finalize_columnar()
    else:
        group.final_results = group.finalize()
    packed = ipc(pack_beam_search_output(SimpleNamespace(beam_group=group)))
    if recv_cls is BatchStrOutput:
        decode_beam_search_output(
            SimpleNamespace(beam_search_output=[packed]),
            [_header(no_stop_trim=False)],
            tokenizer=_IdTokenizer(),
            disable_batch_decode=False,
            trim_matched_stop=_trim,
        )
    recv_obj = mock.Mock(spec=recv_cls)
    recv_obj.beam_search_output = [packed]
    recv_obj.finished_reasons = [{"type": "length", "length": 3}]
    meta_info = {"id": "r", "prompt_tokens": 7, "completion_tokens": 3}
    return try_build_beam_search_out_dict(recv_obj, 0, meta_info)


def _pickle_ipc(obj):
    return pickle.loads(pickle.dumps(obj))


def _msgpack_ipc(obj):
    return msgspec.msgpack.decode(msgspec.msgpack.encode(obj), type=BeamSearchOutput)


class TestColumnarBeamOutput(CustomTestCase):
    """The columnar form of a trie group must reach clients byte-identical to
    the per-sequence form."""

    CASES = {
        # Ties (incl. -inf) keep row order; num_return trims after the sort.
        "ties": ([-1.5, -0.5, -1.5, -float("inf"), -0.5, -2.0], 4, 1.0),
        "unsorted": ([-3.25, -0.125, -7.0, -0.1, -2.2], 5, 1.0),
        "length_penalty": ([-0.3, -0.7, -0.1, -0.9], 2, 0.6),
        "random": (
            torch.randn(64, generator=torch.Generator().manual_seed(0)).tolist(),
            50,
            1.3,
        ),
    }

    def test_out_dict_is_byte_identical_to_per_sequence_form(self):
        for name, (cums, num_return, lp) in self.CASES.items():
            for recv_cls in (BatchStrOutput, BatchTokenIDOutput):
                for ipc in (_pickle_ipc, _msgpack_ipc):
                    with self.subTest(case=name, recv=recv_cls.__name__, ipc=ipc):
                        dicts = [
                            _out_dict(
                                _finished_trie_group(
                                    cums, num_return=num_return, length_penalty=lp
                                ),
                                columnar=columnar,
                                recv_cls=recv_cls,
                                ipc=ipc,
                            )
                            for columnar in (False, True)
                        ]
                        legacy, columnar = (json.dumps(d) for d in dicts)
                        self.assertEqual(columnar, legacy)
                        self.assertEqual(
                            len(dicts[1]["beam_results"]), min(len(cums), num_return)
                        )

    def test_staged_copy_matches_unstaged(self):
        cums = [-1.0, -0.25, -0.5]
        staged = _finished_trie_group(cums, num_return=3)
        staged.stage_final_results()
        a = staged.finalize_columnar()
        b = _finished_trie_group(cums, num_return=3).finalize_columnar()
        self.assertEqual(a.tokens.tolist(), b.tokens.tolist())
        self.assertEqual(a.beam_scores.tolist(), b.beam_scores.tolist())
        self.assertIsNone(staged._host_final)


if __name__ == "__main__":
    unittest.main()
