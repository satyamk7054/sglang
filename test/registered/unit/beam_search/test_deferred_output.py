"""Deferred beam output: a finished group frees its rows at commit, while its
finalize/pack/send is held until the next decode launch (or a fallback point).
The held path must produce exactly the inline path's results."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.beam_search import BeamGroup, joint_select, select_final_topk
from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, FINISH_MATCHED_TOKEN
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def T(data, dtype):
    return torch.tensor(data, dtype=dtype)


class FakeLeader:
    def __init__(self, group):
        self.beam_group = group
        self.defer_output = False
        self.finished_reason = None
        group.leader = self

    def finish_outside_step(self, reason):
        self.finished_reason = reason


def make_coordinator():
    return BeamCoordinator(
        model_config=SimpleNamespace(beam_trie_config=None),
        spec_algorithm=None,
        dllm_enabled=False,
        max_req_len=0,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        future_map=None,
    )


def run_to_length_finish(group):
    """Two joint-select steps then a final top-k; the group finishes by length."""
    group.advance(
        joint_select(
            T([0.0], torch.float32),
            T([[-0.1, -0.2, -0.3, -0.4]], torch.float32),
            T([[1, 2, 3, 4]], torch.int64),
            T([], torch.int64),
            2,
        )
    )
    group.advance(
        joint_select(
            group.frontier_cum_logprobs,
            T([[-0.3, -0.1], [-0.05, -0.4]], torch.float32),
            T([[5, 6], [7, 8]], torch.int64),
            T([], torch.int64),
            2,
        )
    )
    final = select_final_topk(
        group.frontier_cum_logprobs,
        T([[-0.2, -0.4], [-0.1, -0.5]], torch.float32),
        T([[9, 10], [11, 12]], torch.int64),
        2,
    )
    group.advance_final_frontier(final)
    assert group.commit_pending()


def results_of(results):
    return [(r.tokens, r.cum_logprob, r.beam_score, r.matched_token) for r in results]


class TestStagedFinalize(CustomTestCase):
    def test_staged_finalize_matches_direct(self):
        direct = BeamGroup(beam_width=2, max_new_tokens=3, device_history=True)
        staged = BeamGroup(beam_width=2, max_new_tokens=3, device_history=True)
        run_to_length_finish(direct)
        run_to_length_finish(staged)

        staged.stage_final_results()
        self.assertIsNotNone(staged._host_final)
        self.assertEqual(results_of(staged.finalize()), results_of(direct.finalize()))
        self.assertIsNone(staged._host_final)


@mock.patch.object(BeamCoordinator, "_free_member_rows")
class TestCoordinatorDeferredFinish(CustomTestCase):
    def _finish(self, coordinator, group):
        leader = FakeLeader(group)
        coordinator._num_live_groups = 1
        coordinator._finish_group(group)
        return leader

    def test_device_history_defers_finalize(self, free_rows):
        inline = BeamGroup(beam_width=2, max_new_tokens=3, device_history=True)
        held = BeamGroup(beam_width=2, max_new_tokens=3, device_history=True)
        run_to_length_finish(inline)
        run_to_length_finish(held)

        inline_leader = self._finish(make_coordinator(), inline)

        coordinator = make_coordinator()
        coordinator.defer_finished_outputs = True
        leader = self._finish(coordinator, held)

        # Rows are freed and the group retired at commit, before any output.
        self.assertEqual(free_rows.call_count, 2)
        self.assertTrue(held.retired)
        self.assertEqual(coordinator._num_live_groups, 0)
        self.assertIsNone(held.final_results)
        self.assertTrue(leader.defer_output)
        self.assertTrue(coordinator.has_deferred_outputs())
        self.assertIsInstance(leader.finished_reason, FINISH_LENGTH)
        self.assertEqual(
            leader.finished_reason.to_json(), inline_leader.finished_reason.to_json()
        )

        self.assertEqual(coordinator.take_deferred_outputs(), [leader])
        self.assertFalse(leader.defer_output)
        self.assertFalse(coordinator.has_deferred_outputs())
        self.assertEqual(
            results_of(held.final_results), results_of(inline.final_results)
        )

    def test_host_history_keeps_stop_finish_reason(self, free_rows):
        group = BeamGroup(beam_width=2, stop_token_ids=[99], max_new_tokens=3)
        group.advance(
            joint_select(
                T([0.0], torch.float32),
                T([[-0.1, -0.2, -0.3, -0.4]], torch.float32),
                T([[99, 2, 3, 4]], torch.int64),
                T([99], torch.int64),
                2,
            )
        )
        group.advance_final(
            select_final_topk(
                group.frontier_cum_logprobs,
                T([[-2.0, -3.0], [-2.0, -3.0]], torch.float32),
                T([[9, 10], [11, 12]], torch.int64),
                2,
            )
        )
        coordinator = make_coordinator()
        coordinator.defer_finished_outputs = True
        leader = self._finish(coordinator, group)

        self.assertIsInstance(leader.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(group.final_results[0].tokens, [99])
        self.assertTrue(leader.defer_output)
        self.assertEqual(coordinator.take_deferred_outputs(), [leader])
        self.assertEqual(group.final_results[0].tokens, [99])

    def test_inline_finish_holds_nothing(self, free_rows):
        group = BeamGroup(beam_width=2, max_new_tokens=3, device_history=True)
        run_to_length_finish(group)
        coordinator = make_coordinator()
        leader = self._finish(coordinator, group)
        self.assertFalse(leader.defer_output)
        self.assertFalse(coordinator.has_deferred_outputs())
        self.assertEqual(len(group.final_results), 2)


class TestSchedulerFlush(CustomTestCase):
    def test_flush_streams_held_leaders_once(self):
        group = BeamGroup(beam_width=2, max_new_tokens=3, device_history=True)
        run_to_length_finish(group)
        coordinator = make_coordinator()
        coordinator.defer_finished_outputs = True
        with mock.patch.object(BeamCoordinator, "_free_member_rows"):
            coordinator._num_live_groups = 1
            leader = FakeLeader(group)
            coordinator._finish_group(group)

        streamer = mock.Mock()
        scheduler = SimpleNamespace(
            beam_coordinator=coordinator, output_streamer=streamer
        )
        Scheduler.flush_deferred_beam_outputs(scheduler)
        streamer.stream_output.assert_called_once_with([leader], False)
        self.assertEqual(len(group.final_results), 2)

        Scheduler.flush_deferred_beam_outputs(scheduler)
        streamer.stream_output.assert_called_once()

    def test_enable_respects_kill_switch(self):
        from sglang.srt.environ import envs

        for disabled in (False, True):
            coordinator = make_coordinator()
            streamer = SimpleNamespace(defer_outputs=False)
            scheduler = SimpleNamespace(
                beam_coordinator=coordinator, output_streamer=streamer
            )
            with envs.SGLANG_DISABLE_BEAM_DEFERRED_OUTPUT.override(disabled):
                Scheduler.enable_beam_deferred_output(scheduler)
            self.assertEqual(coordinator.defer_finished_outputs, not disabled)
            self.assertEqual(streamer.defer_outputs, not disabled)


def held_coordinator(num_live_groups):
    """A coordinator holding one finished group's output."""
    group = BeamGroup(beam_width=2, max_new_tokens=3, device_history=True)
    run_to_length_finish(group)
    coordinator = make_coordinator()
    coordinator.defer_finished_outputs = True
    with mock.patch.object(BeamCoordinator, "_free_member_rows"):
        coordinator._num_live_groups = 1
        leader = FakeLeader(group)
        coordinator._finish_group(group)
    coordinator._num_live_groups = num_live_groups
    return coordinator, leader


def fake_batch(decode):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: decode), beam_tail=None
    )


class TestFlushPolicy(CustomTestCase):
    def test_nothing_held_never_flushes(self):
        coordinator = make_coordinator()
        coordinator._num_live_groups = 0
        self.assertFalse(coordinator.should_flush_after_launch(is_decode=True))
        self.assertFalse(coordinator.should_flush_after_launch(is_decode=False))

    def test_decode_launch_flushes(self):
        coordinator, _ = held_coordinator(num_live_groups=3)
        self.assertTrue(coordinator.should_flush_after_launch(is_decode=True))

    def test_prefill_launch_holds_while_decode_is_due(self):
        coordinator, _ = held_coordinator(num_live_groups=3)
        bound = coordinator.max_deferred_output_hold_launches
        for _ in range(bound - 1):
            self.assertFalse(coordinator.should_flush_after_launch(is_decode=False))
        self.assertTrue(coordinator.should_flush_after_launch(is_decode=False))
        coordinator.take_deferred_outputs()
        self.assertEqual(coordinator._deferred_hold_launches, 0)

    def test_prefill_launch_flushes_without_live_groups(self):
        coordinator, _ = held_coordinator(num_live_groups=0)
        self.assertTrue(coordinator.should_flush_after_launch(is_decode=False))

    def test_after_launch_fallback(self):
        for decode, live, flushed in (
            (False, 2, False),
            (False, 0, True),
            (True, 2, True),
        ):
            coordinator, leader = held_coordinator(num_live_groups=live)
            streamer = mock.Mock()
            scheduler = SimpleNamespace(
                beam_coordinator=coordinator, output_streamer=streamer
            )
            scheduler.flush_deferred_beam_outputs = lambda s=scheduler: (
                Scheduler.flush_deferred_beam_outputs(s)
            )
            Scheduler.maybe_flush_deferred_beam_outputs_after_launch(
                scheduler, fake_batch(decode)
            )
            self.assertEqual(streamer.stream_output.called, flushed)
            self.assertEqual(coordinator.has_deferred_outputs(), not flushed)


class TestRelayFlush(CustomTestCase):
    def _relay(self, decode):
        calls = mock.Mock()
        scheduler = SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_ngram=lambda: False),
            uocr_runtime=None,
            future_map=SimpleNamespace(stash=lambda *args: None),
            beam_coordinator=SimpleNamespace(
                maybe_select_and_relay=calls.select_and_relay
            ),
            flush_deferred_beam_outputs=calls.flush,
            chunked_req=None,
        )
        batch_result = SimpleNamespace(
            next_draft_input=None,
            has_sampled_token_ids=True,
            next_token_ids=torch.zeros(1, dtype=torch.int64),
        )
        Scheduler._relay_forward_payload(
            scheduler, fake_batch(decode), torch.zeros(1), batch_result
        )
        return [c[0] for c in calls.mock_calls]

    def test_decode_flushes_before_selection(self):
        self.assertEqual(self._relay(decode=True), ["flush", "select_and_relay"])

    def test_prefill_does_not_flush(self):
        self.assertEqual(self._relay(decode=False), ["select_and_relay"])


if __name__ == "__main__":
    unittest.main()
