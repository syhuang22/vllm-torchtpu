# Copyright 2026 Google LLC
#
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
"""Per-request ``SamplingParams.thinking_token_budget`` tracker and step adapters.

Builds on ``RequestSlotTable`` for request-ID-keyed ``[max_num_reqs + 1, 7]``
device-slot pooling, wave-swap slot release, and cached ``slot_ids`` H2D
uploads, while delegating KMP transitions and forcing kernels to
``vllm_torchtpu.layers.adapter.sample.thinking_budget``.
"""

from __future__ import annotations

from collections.abc import Callable, Container, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm_torchtpu.layers.adapter.sample import thinking_budget as tb_kernels
from vllm_torchtpu.layers.adapter.sample.request_slot_table import RequestSlotTable
from vllm_torchtpu.layers.adapter.sample.thinking_budget import (
    INT32_MAX,
    NO_FORCED_TOKEN,
    NUM_STATE_COLS,
    UNTRACKED_STATE,
    StateTuple,
    ThinkingBudgetMarkers,
    advance_sequence_state,
    compute_sequence_state,
)
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm import SamplingParams
    from vllm.config.reasoning import ReasoningConfig
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

logger = init_logger(__name__)


def _prefix_1d(values: torch.Tensor, num_rows: int) -> torch.Tensor:
    if values.shape[0] == num_rows:
        return values
    return tb_kernels.slice_rows_1d(values, num_rows)


def _compute_request_state(
    req_state: CachedRequestState | None,
    budget: int,
    markers: ThinkingBudgetMarkers,
) -> StateTuple:
    """Reconstruct KMP state from ``req_state``'s prompt and committed outputs.

    Placeholder invariants:
    * ``on_resumed`` runs in ``TPUModelRunner._update_states`` after
      ``super()._update_states`` aligns ``req_state.output_token_ids`` to the
      scheduler's committed ``num_output_tokens``; negative scheduler
      placeholders are skipped by ``out_arr >= 0``.
    * ``prepare_step`` only calls ``_compute_request_state`` on a cold reseed
      when a budgeted request has neither an allocated slot nor a staged dirty
      entry; ``TPUModelRunner.sample_tokens`` settles any in-flight
      ``_pre_async_results`` placeholders via ``_modify_prev_results`` whenever
      ``needs_output_reseed`` is true.
    """
    if req_state is None:
        return compute_sequence_state((), budget, markers)
    state = compute_sequence_state(req_state.prompt_token_ids or (), budget, markers)
    out_ids = req_state.output_token_ids
    if out_ids:
        out_arr = np.asarray(out_ids, dtype=np.int64)
        state = advance_sequence_state(state, out_arr[out_arr >= 0], markers)
    return state


class ThinkingBudgetTracker:
    """Request-ID-keyed ``thinking_token_budget`` lifecycle and device state."""

    def __init__(
        self,
        reasoning_config: ReasoningConfig | None,
        max_num_reqs: int,
    ) -> None:
        self._markers = ThinkingBudgetMarkers.from_reasoning_config(reasoning_config)
        self._slots: RequestSlotTable[StateTuple] = RequestSlotTable(
            max_num_reqs=max_num_reqs,
            num_cols=NUM_STATE_COLS,
            sentinel_row=UNTRACKED_STATE,
        )
        self._budgets: dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        return self._markers is not None

    @property
    def max_num_reqs(self) -> int:
        return self._slots.max_num_reqs

    @property
    def markers(self) -> ThinkingBudgetMarkers | None:
        return self._markers

    @property
    def requests(self) -> Mapping[str, int]:
        return self._budgets

    @property
    def slots(self) -> RequestSlotTable[StateTuple]:
        return self._slots

    def __bool__(self) -> bool:
        return self._markers is not None and bool(self._budgets)

    def commit_state_table(self, state_table: torch.Tensor) -> None:
        self._slots.commit_table(state_table)

    def ensure_state_table(self, device: torch.device) -> torch.Tensor:
        return self._slots.ensure_table(device)

    def discard(self, req_ids: Iterable[str]) -> None:
        if not self._budgets:
            return
        req_list = list(req_ids)
        for req_id in req_list:
            self._budgets.pop(req_id, None)
        self._slots.release(req_list)

    def on_preempted(self, req_ids: Iterable[str] | None) -> None:
        if not req_ids or not self._budgets:
            return
        self._slots.release(r for r in req_ids if r in self._budgets)

    def release_unscheduled(self, active_req_ids: Container[str]) -> None:
        """Free slots for budgeted requests no longer in ``input_batch``."""
        self._slots.release_unscheduled(active_req_ids)

    def add(
        self,
        req_id: str,
        sampling_params: SamplingParams | None,
        prompt_token_ids: Sequence[int] | None,
    ) -> None:
        raw_budget = (
            None if sampling_params is None else sampling_params.thinking_token_budget
        )
        if raw_budget is None or raw_budget < 0:
            if self._budgets:
                self.discard((req_id,))
            return
        if self._markers is None:
            logger.warning_once(
                "Ignoring thinking_token_budget because reasoning start/end "
                "token IDs are not configured on the model runner."
            )
            return
        clamped_budget = min(int(raw_budget), INT32_MAX)
        self._budgets[req_id] = clamped_budget
        self._slots.mark_dirty(
            req_id,
            compute_sequence_state(
                prompt_token_ids or (), clamped_budget, self._markers
            ),
        )

    def needs_output_reseed(
        self,
        batch_req_ids: Iterable[str | None],
        device: torch.device | None = None,
    ) -> bool:
        """Return True if ``prepare_step`` will read ``output_token_ids``."""
        if self._markers is None or not self._budgets:
            return False
        return self._slots.needs_cold_reseed(batch_req_ids, self._budgets, device)

    def step_state(
        self,
        device: torch.device,
        batch_req_ids: Sequence[str | None],
        requests: Mapping[str, CachedRequestState],
        discard_req_ids: AbstractSet[str],
    ) -> ThinkingBudgetStepState | None:
        """Prepare this step's device KMP state for ``sample_tokens``."""
        if not self:
            return None
        markers = self._markers
        assert markers is not None
        state_table = self.prepare_step(device, batch_req_ids, requests)
        if state_table is None:
            return None
        return ThinkingBudgetStepState(self, state_table, markers, discard_req_ids)

    def on_resumed(
        self,
        resumed_req_ids: Iterable[str],
        requests: Mapping[str, CachedRequestState],
    ) -> None:
        if self._markers is None or not self._budgets:
            return
        for req_id in resumed_req_ids:
            budget = self._budgets.get(req_id)
            req_state = None if budget is None else requests.get(req_id)
            if budget is None or req_state is None:
                continue
            self._slots.mark_dirty(
                req_id, _compute_request_state(req_state, budget, self._markers)
            )

    def prepare_step(
        self,
        device: torch.device,
        batch_req_ids: Sequence[str | None],
        requests: Mapping[str, CachedRequestState],
    ) -> torch.Tensor | None:
        """Ensure all scheduled budgeted requests own seeded slots on ``device``."""
        if self._markers is None or not self._budgets:
            return None
        markers = self._markers
        return self._slots.sync_step(
            device=device,
            batch_req_ids=batch_req_ids,
            tracked_req_ids=self._budgets,
            compute_cold_row=lambda req_id: _compute_request_state(
                requests.get(req_id), self._budgets[req_id], markers
            ),
        )

    def build_chunk_slot_ids(
        self,
        device: torch.device,
        batch_req_ids: Sequence[str | None],
        start_idx: int,
        end_idx: int,
        padded_num_reqs: int | None = None,
        discard_req_ids: AbstractSet[str] | None = None,
    ) -> torch.Tensor | None:
        """Build ``[padded_num_reqs]`` int32 slot IDs for ``[start_idx, end_idx)``."""
        return self._slots.build_chunk_slot_ids(
            device=device,
            batch_req_ids=batch_req_ids,
            start_idx=start_idx,
            end_idx=end_idx,
            padded_num_reqs=padded_num_reqs,
            discard_req_ids=discard_req_ids,
        )

    def precompile(
        self,
        device: torch.device,
        num_reqs_paddings: Sequence[int],
        verify_logits_paddings: Sequence[int],
        max_draft_tokens: int,
        dummy_logits_fn: Callable[[int], torch.Tensor],
        sync_fn: Callable[..., None],
        precompile_top_tokens: bool = False,
    ) -> None:
        """Pre-warm all thinking-budget subgraphs across compiled bucket shapes."""
        if self._markers is None:
            return
        markers = self._markers
        kernel_args = markers.kernel_args()
        max_reqs = self.max_num_reqs
        state_table = self._slots.precompile(device, sync_fn)

        slot_ids_full = torch.zeros(max_reqs, dtype=torch.int32).to(device)
        forced_full = tb_kernels.query_forced_tokens(
            state_table, slot_ids_full, markers.end_ids
        )
        sync_fn(forced_full)

        has_spec_verify = max_draft_tokens > 0 and bool(verify_logits_paddings)
        for num_reqs in num_reqs_paddings:
            forced_req = _prefix_1d(forced_full, num_reqs)
            selected = torch.zeros((num_reqs, 1), dtype=torch.int64).to(device)
            masked_plain, _ = tb_kernels.force_logits_rows(
                dummy_logits_fn(num_reqs), forced_req
            )
            state_table = tb_kernels.advance_plain_states(
                state_table, slot_ids_full, selected, *kernel_args
            )
            sync_fn([masked_plain, state_table])
            if precompile_top_tokens:
                out_plain, state_table = tb_kernels.override_and_advance_plain(
                    state_table, slot_ids_full, selected, *kernel_args
                )
                if has_spec_verify:
                    out_bonus_toks, _ = tb_kernels.force_token_ids(
                        torch.zeros(num_reqs, dtype=torch.int64).to(device),
                        forced_req,
                    )
                    sync_fn([out_plain, out_bonus_toks, state_table])
                else:
                    sync_fn([out_plain, state_table])
            if has_spec_verify:
                state_table = tb_kernels.advance_spec_states(
                    state_table,
                    slot_ids_full,
                    torch.full(
                        (num_reqs, max_draft_tokens + 1),
                        NO_FORCED_TOKEN,
                        dtype=torch.int32,
                    ).to(device),
                    max_draft_tokens,
                    *kernel_args,
                )
                sync_fn(state_table)
            logger.info("  -- plain num_seqs: %d", num_reqs)

        if has_spec_verify:
            for num_tokens in verify_logits_paddings:
                draft_ids = torch.zeros(num_tokens, dtype=torch.int32).to(device)
                seg_ids = torch.full((num_tokens,), max_reqs, dtype=torch.int64).to(
                    device
                )
                grp_ids = torch.zeros(num_tokens, dtype=torch.int32).to(device)
                forced_tgt, forced_bonus = tb_kernels.prepare_spec_verify_drafts(
                    state_table,
                    slot_ids_full,
                    draft_ids,
                    seg_ids,
                    grp_ids,
                    max_draft_tokens,
                    *kernel_args,
                )
                out_logits, did_force = tb_kernels.force_logits_rows(
                    dummy_logits_fn(num_tokens), forced_tgt
                )
                eff_drafts = tb_kernels.invalidate_forced_drafts(
                    draft_ids, forced_tgt, did_force
                )
                sync_fn([forced_bonus, out_logits, eff_drafts])
                if precompile_top_tokens:
                    out_toks, did_force_toks = tb_kernels.force_token_ids(
                        torch.zeros(num_tokens, dtype=torch.int64).to(device),
                        forced_tgt,
                    )
                    eff_top_drafts = tb_kernels.invalidate_forced_drafts(
                        draft_ids, forced_tgt, did_force_toks
                    )
                    sync_fn([out_toks, eff_top_drafts])

        self._slots.commit_table(state_table)


@dataclass
class ThinkingBudgetStepState:
    """Per-step thinking-budget state threaded through ``sample_tokens``."""

    tracker: ThinkingBudgetTracker
    state_table: torch.Tensor
    markers: ThinkingBudgetMarkers
    discard_req_ids: AbstractSet[str]

    def commit(self) -> None:
        self.tracker.commit_state_table(self.state_table)

    def chunk_slot_ids(
        self,
        device: torch.device,
        batch_req_ids: Sequence[str | None],
        start_idx: int,
        end_idx: int,
        padded_num_reqs: int | None = None,
    ) -> torch.Tensor | None:
        return self.tracker.build_chunk_slot_ids(
            device=device,
            batch_req_ids=batch_req_ids,
            start_idx=start_idx,
            end_idx=end_idx,
            padded_num_reqs=padded_num_reqs,
            discard_req_ids=self.discard_req_ids,
        )

    def mask_plain_logits(
        self, logits: torch.Tensor, slot_ids: torch.Tensor
    ) -> torch.Tensor:
        forced_full = tb_kernels.query_forced_tokens(
            self.state_table, slot_ids, self.markers.end_ids
        )
        forced_rows = _prefix_1d(forced_full, logits.shape[0])
        masked_logits, _ = tb_kernels.force_logits_rows(logits, forced_rows)
        return masked_logits

    def advance_plain(
        self, slot_ids: torch.Tensor, selected_token_ids: torch.Tensor
    ) -> None:
        self.state_table = tb_kernels.advance_plain_states(
            self.state_table,
            slot_ids,
            selected_token_ids,
            *self.markers.kernel_args(),
        )
        self.commit()

    def override_and_advance_plain(
        self,
        device: torch.device,
        batch_req_ids: Sequence[str | None],
        start_idx: int,
        end_idx: int,
        selected_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        slot_ids = self.chunk_slot_ids(device, batch_req_ids, start_idx, end_idx)
        if slot_ids is None:
            return selected_token_ids
        out, self.state_table = tb_kernels.override_and_advance_plain(
            self.state_table,
            slot_ids,
            selected_token_ids,
            *self.markers.kernel_args(),
        )
        self.commit()
        return out

    def spec_chunk(
        self,
        device: torch.device,
        batch_req_ids: Sequence[str | None],
        start_idx: int,
        end_idx: int,
    ) -> ThinkingBudgetSpecChunk | None:
        slot_ids = self.chunk_slot_ids(device, batch_req_ids, start_idx, end_idx)
        if slot_ids is None:
            return None
        return ThinkingBudgetSpecChunk(self, slot_ids)


@dataclass(frozen=True)
class ThinkingBudgetSpecChunk:
    """Speculative verification forcing/advance handle for one batch chunk."""

    step: ThinkingBudgetStepState
    slot_ids: torch.Tensor

    def prepare_logits(
        self,
        target_logits: torch.Tensor,
        bonus_logits: torch.Tensor,
        draft_token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        group_indices: torch.Tensor,
        max_draft_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        forced_tgt, forced_bonus_full = tb_kernels.prepare_spec_verify_drafts(
            self.step.state_table,
            self.slot_ids,
            draft_token_ids,
            segment_ids,
            group_indices,
            max_draft_tokens,
            *self.step.markers.kernel_args(),
        )
        target_logits, did_force = tb_kernels.force_logits_rows(
            target_logits, forced_tgt
        )
        effective_drafts = tb_kernels.invalidate_forced_drafts(
            draft_token_ids, forced_tgt, did_force
        )
        forced_bonus = _prefix_1d(forced_bonus_full, bonus_logits.shape[0])
        bonus_logits, _ = tb_kernels.force_logits_rows(bonus_logits, forced_bonus)
        return target_logits, bonus_logits, effective_drafts

    def prepare_tokens(
        self,
        target_argmax: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        draft_token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        group_indices: torch.Tensor,
        max_draft_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        forced_tgt, forced_bonus_full = tb_kernels.prepare_spec_verify_drafts(
            self.step.state_table,
            self.slot_ids,
            draft_token_ids,
            segment_ids,
            group_indices,
            max_draft_tokens,
            *self.step.markers.kernel_args(),
        )
        target_argmax, did_force = tb_kernels.force_token_ids(target_argmax, forced_tgt)
        effective_drafts = tb_kernels.invalidate_forced_drafts(
            draft_token_ids, forced_tgt, did_force
        )
        forced_bonus = _prefix_1d(forced_bonus_full, bonus_token_ids.shape[0])
        bonus_token_ids, _ = tb_kernels.force_token_ids(bonus_token_ids, forced_bonus)
        return target_argmax, bonus_token_ids, effective_drafts

    def finish(self, next_tokens: torch.Tensor, max_draft_tokens: int) -> None:
        self.step.state_table = tb_kernels.advance_spec_states(
            self.step.state_table,
            self.slot_ids,
            next_tokens,
            max_draft_tokens,
            *self.step.markers.kernel_args(),
        )
        self.step.commit()
