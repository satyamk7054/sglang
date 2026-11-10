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
"""Trie decode selection for G same-shape beam groups in one launch sequence.

Pure tensor functions; slice g of every output equals the per-group path
(trie_constraint.topk_logprobs + joint_select / select_final_topk +
advance_prefixes + remap_kv_mapping) applied to group g, bit for bit:

- every op is either elementwise, integer, or a per-row reduction whose row
  contents and order match the per-group call, so float rounding is unchanged;
- the joint top-k runs per group (dim=1 of [G, k * C]) over the same flat
  candidate order as the per-group reshape(-1), so torch.topk resolves ties
  the same way.

Trie candidates are SID tokens and trie groups reject SID stop tokens, so the
joint step is the no-stop fast path: the top k candidates all survive.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from sglang.srt.beam_search.joint_select import FinalSelect, SelectResult


def regroup_trie_rows(values: torch.Tensor, num_groups: int):
    """Split rows laid out leaders-then-tails ([G] + [G * (k-1)]) into
    ([G, C] leader, [G, k-1, C] member) views, no copy."""
    leaders = values[:num_groups]
    members = values[num_groups:].view(num_groups, -1, values.shape[-1])
    return leaders, members


def rank_candidates_batched(
    cum_logprobs: torch.Tensor,  # [G, k] float32
    leader_values: torch.Tensor,  # [G, C] float32, frontier row 0
    member_values: torch.Tensor,  # [G, k-1, C] float32, frontier rows 1..k-1
    leader_tokens: torch.Tensor,  # [G, C] int64
    member_tokens: torch.Tensor,  # [G, k-1, C] int64
    num_out: int,
):
    """Batched _ranked_candidates: (scores, parent, tokens), each [G, num_out]."""
    num_groups, beam_width = cum_logprobs.shape
    num_candidates = leader_values.shape[-1]
    # Written in frontier-row order so row g flattens exactly like the
    # per-group [k, C] score matrix.
    scores = cum_logprobs.new_empty((num_groups, beam_width, num_candidates))
    torch.add(cum_logprobs[:, :1, None], leader_values[:, None, :], out=scores[:, :1])
    torch.add(cum_logprobs[:, 1:, None], member_values, out=scores[:, 1:])
    cand_scores, cand_idx = scores.view(num_groups, -1).topk(
        num_out, dim=1, sorted=True
    )
    parent = cand_idx // num_candidates
    column = cand_idx - parent * num_candidates
    group_idx = torch.arange(num_groups, device=cand_idx.device).unsqueeze(1)
    tokens = torch.where(
        parent == 0,
        leader_tokens.gather(1, column),
        member_tokens[group_idx, (parent - 1).clamp_min(0), column],
    )
    return cand_scores, parent, tokens


def advance_prefixes_batched(
    prefix_codes: torch.Tensor,  # [G, k, depth] int64
    parent_idx: torch.Tensor,  # [G, k] int64
    next_tokens: torch.Tensor,  # [G, k] int64
    token_start: int,
) -> torch.Tensor:
    """Batched BeamTrieConstraint.advance_prefixes: [G, k, depth + 1]."""
    depth = prefix_codes.shape[2]
    inherited = prefix_codes.gather(1, parent_idx.unsqueeze(2).expand(-1, -1, depth))
    return torch.cat([inherited, (next_tokens - token_start).unsqueeze(2)], dim=2)


def remap_kv_mapping_batched(
    req_to_token: torch.Tensor,
    rows: torch.Tensor,  # [G, k] int64, disjoint across groups
    parent_idx: torch.Tensor,  # [G, k] int64
    prefix_lens: torch.Tensor,  # [G] int64, on req_to_token's device
    window: int,
):
    """Batched remap_kv_mapping over equal-length decode windows starting at
    each group's own prefix length; returns ([G, k, window] old, new)."""
    cols = prefix_lens.view(-1, 1, 1) + torch.arange(
        window, device=prefix_lens.device
    ).view(1, 1, -1)
    row_index = rows.unsqueeze(2)
    old_mapping = req_to_token[row_index, cols]
    new_mapping = old_mapping.gather(1, parent_idx.unsqueeze(2).expand(-1, -1, window))
    req_to_token[row_index, cols] = new_mapping
    return old_mapping, new_mapping


def split_final_select(
    scores: torch.Tensor, parent: torch.Tensor, tokens: torch.Tensor
) -> List[FinalSelect]:
    return [
        FinalSelect(tokens=t, parent_idx=p, cum_logprobs=s)
        for s, p, t in zip(scores.unbind(0), parent.unbind(0), tokens.unbind(0))
    ]


def split_no_stop_select(
    scores: torch.Tensor, parent: torch.Tensor, tokens: torch.Tensor
) -> List[SelectResult]:
    """Per-group joint_select fast-path results (num_candidates == k, no stop
    tokens); the finished-pool fields are shared zero views."""
    num_groups, beam_width = tokens.shape
    survivors = torch.full(
        (num_groups,), beam_width, dtype=torch.int64, device=tokens.device
    ).unbind(0)
    finished = torch.zeros(num_groups, dtype=torch.int64, device=tokens.device)
    zero_ids = torch.zeros_like(tokens)
    zero_scores = torch.zeros_like(scores)
    return [
        SelectResult(
            next_tokens=tokens[g],
            parent_idx=parent[g],
            new_cum_logprobs=scores[g],
            num_survivors=survivors[g],
            fin_tokens=zero_ids[g],
            fin_parent_idx=zero_ids[g],
            fin_cum_logprobs=zero_scores[g],
            num_finished=finished[g],
        )
        for g in range(num_groups)
    ]


def gather_rows(
    tensor: Optional[torch.Tensor],
    ranges: Sequence[tuple],
    index_to_device,
) -> Optional[torch.Tensor]:
    """Rows [start, end) of each range, concatenated; a view when the ranges
    are back to back."""
    if tensor is None:
        return None
    if all(ranges[i][1] == ranges[i + 1][0] for i in range(len(ranges) - 1)):
        return tensor[ranges[0][0] : ranges[-1][1]]
    index = torch.cat([torch.arange(start, end) for start, end in ranges])
    return tensor.index_select(0, index_to_device(index))
