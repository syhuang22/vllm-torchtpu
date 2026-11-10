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
"""Reusable request-ID-keyed device-slot table for stateful sampling adapters.

``TPUModelRunner`` reorders batch rows mid-step in ``_reorder_batch_for_rpa``
and under ``--async-scheduling`` leaves unsettled output placeholders in flight
when ``sample_tokens()`` begins. Stateful sampling adapters (such as
``thinking_token_budget``, token penalties, or ``bad_words``) can own a
``RequestSlotTable`` to maintain fixed-shape ``[max_num_reqs + 1, num_cols]``
per-request state on device (with slot ``0`` reserved as an untracked sentinel),
release slots across unscheduled waves, and reuse cached ``slot_ids`` H2D
uploads across steady-state decode steps.
"""

from collections.abc import Callable, Container, Iterable, Sequence
from collections.abc import Set as AbstractSet

import numpy as np
import torch

from vllm_torchtpu.layers.adapter.sample.scatter_ops import broadcast_amax_scatter


def same_target_device(actual: torch.device, requested: torch.device) -> bool:
    """Match e.g. ``device('tpu', 0)`` against unindexed ``device('tpu')``."""
    if actual.type != requested.type:
        return False
    if requested.index is not None:
        return actual.index == requested.index
    if actual.index is None:
        return True
    default_idx = (
        torch.accelerator.current_device_index()
        if torch.accelerator.is_available()
        and (acc := torch.accelerator.current_accelerator()) is not None
        and acc.type == requested.type
        else 0
    )
    return actual.index == default_idx


def scatter_slot_rows(
    table: torch.Tensor, slot_ids: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    """Return ``table`` with ``out[slot_ids[i]] = rows[i]`` for ``slot_ids > 0``."""
    return broadcast_amax_scatter(table, slot_ids, rows, slot_ids > 0)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def seed_slot_rows(
    table: torch.Tensor,
    seed_slots: torch.Tensor,
    seed_values: torch.Tensor,
) -> torch.Tensor:
    """Scatter ``seed_values`` into ``table`` at active ``seed_slots > 0``."""
    return scatter_slot_rows(table, seed_slots, seed_values)


class RequestSlotTable[RowT: Sequence[int]]:
    """Request-ID-keyed ``[max_num_reqs + 1, num_cols]`` int32 device-slot pool.

    Slot ``0`` is a permanent untracked sentinel row; slots ``1..max_num_reqs``
    are leased to active requests in ``input_batch`` and released when requests
    finish, are preempted, or rotate out in unscheduled waves.
    """

    def __init__(
        self,
        max_num_reqs: int,
        num_cols: int,
        sentinel_row: Sequence[int] | None = None,
    ) -> None:
        self._max_num_reqs = max(int(max_num_reqs), 1)
        self._num_cols = int(num_cols)
        if self._num_cols <= 0:
            raise ValueError(f"num_cols must be positive, got {num_cols}")
        default_row = (
            tuple(int(v) for v in sentinel_row)
            if sentinel_row is not None
            else (0,) * self._num_cols
        )
        if len(default_row) != self._num_cols:
            raise ValueError(
                f"sentinel_row length {len(default_row)} != num_cols {self._num_cols}"
            )
        self._sentinel_row: tuple[int, ...] = default_row
        self._capacity = self._max_num_reqs + 1
        self._slot_by_req_id: dict[str, int] = {}
        self._free_slots: list[int] = list(range(self._capacity - 1, 0, -1))
        self._dirty_rows: dict[str, RowT] = {}
        self._table: torch.Tensor | None = None
        self._last_slot_ids: tuple[tuple[int, bytes], torch.Tensor] | None = None

    @property
    def max_num_reqs(self) -> int:
        return self._max_num_reqs

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def num_cols(self) -> int:
        return self._num_cols

    @property
    def table(self) -> torch.Tensor | None:
        return self._table

    def slot_for(self, req_id: str) -> int | None:
        return self._slot_by_req_id.get(req_id)

    def has_dirty(self, req_id: str) -> bool:
        return req_id in self._dirty_rows

    def allocated_req_ids(self) -> AbstractSet[str]:
        return self._slot_by_req_id.keys()

    def commit_table(self, table: torch.Tensor) -> None:
        self._table = table

    def ensure_table(self, device: torch.device) -> torch.Tensor:
        if self._table is None or not same_target_device(self._table.device, device):
            if self._table is not None:
                self._slot_by_req_id.clear()
                self._free_slots = list(range(self._capacity - 1, 0, -1))
                self._last_slot_ids = None
            sentinel_np = np.tile(
                np.asarray(self._sentinel_row, dtype=np.int32), (self._capacity, 1)
            )
            self._table = torch.from_numpy(sentinel_np).to(device)
        return self._table

    def mark_dirty(self, req_id: str, row: RowT) -> None:
        self._dirty_rows[req_id] = row

    def _release_slot(self, req_id: str) -> None:
        slot = self._slot_by_req_id.pop(req_id, None)
        if slot is not None:
            self._free_slots.append(slot)

    def release(self, req_ids: Iterable[str] | None) -> None:
        """Free slots and pending dirty rows for ``req_ids``."""
        if not req_ids:
            return
        for req_id in req_ids:
            self._dirty_rows.pop(req_id, None)
            self._release_slot(req_id)

    def release_unscheduled(self, active_req_ids: Container[str]) -> None:
        """Free slots and dirty rows for requests absent from ``active_req_ids``."""
        if not self._slot_by_req_id and not self._dirty_rows:
            return
        for req_id in list(self._slot_by_req_id):
            if req_id not in active_req_ids:
                self._dirty_rows.pop(req_id, None)
                self._release_slot(req_id)
        for req_id in list(self._dirty_rows):
            if req_id not in active_req_ids:
                self._dirty_rows.pop(req_id, None)

    def needs_cold_reseed(
        self,
        batch_req_ids: Iterable[str | None],
        tracked_req_ids: Container[str],
        device: torch.device | None = None,
    ) -> bool:
        """Return True if ``sync_step`` will invoke ``compute_cold_row``."""
        slots_valid = self._table is None or (
            device is None or same_target_device(self._table.device, device)
        )
        return any(
            req_id is not None
            and req_id in tracked_req_ids
            and (not slots_valid or req_id not in self._slot_by_req_id)
            and req_id not in self._dirty_rows
            for req_id in batch_req_ids
        )

    def _allocate_slot(self, req_id: str) -> int:
        existing = self._slot_by_req_id.get(req_id)
        if existing is not None:
            return existing
        assert self._free_slots, (
            f"RequestSlotTable pool exhausted "
            f"(max_num_reqs={self._max_num_reqs}, capacity={self._capacity})"
        )
        slot = self._free_slots.pop()
        self._slot_by_req_id[req_id] = slot
        return slot

    def sync_step(
        self,
        device: torch.device,
        batch_req_ids: Sequence[str | None],
        tracked_req_ids: Container[str],
        compute_cold_row: Callable[[str], RowT],
    ) -> torch.Tensor | None:
        """Allocate slots and seed dirty/cold rows for ``batch_req_ids``."""
        batch_tracked = list(
            dict.fromkeys(
                req_id
                for req_id in batch_req_ids
                if req_id is not None and req_id in tracked_req_ids
            )
        )
        if not batch_tracked:
            return None

        self.release_unscheduled(set(batch_tracked))
        table = self.ensure_table(device)
        to_seed: list[tuple[int, RowT]] = []
        for req_id in batch_tracked:
            had_slot = req_id in self._slot_by_req_id
            slot = self._allocate_slot(req_id)
            dirty = self._dirty_rows.pop(req_id, None)
            if dirty is None and not had_slot:
                dirty = compute_cold_row(req_id)
            if dirty is not None:
                to_seed.append((slot, dirty))

        if to_seed:
            seed_width = self._max_num_reqs
            sentinel_arr = np.asarray(self._sentinel_row, dtype=np.int32)
            for start in range(0, len(to_seed), seed_width):
                chunk = to_seed[start : start + seed_width]
                slots_np = np.zeros(seed_width, dtype=np.int32)
                vals_np = np.tile(sentinel_arr, (seed_width, 1))
                for idx, (slot, row_vals) in enumerate(chunk):
                    slots_np[idx] = slot
                    vals_np[idx] = row_vals
                # Synchronous .to(device): sample_tokens() runs after the
                # runner's input-staging fence, so async H2D from temporary CPU
                # buffers would risk reading freed host memory in TorchTPU.
                table = seed_slot_rows(
                    table,
                    torch.from_numpy(slots_np).to(device),
                    torch.from_numpy(vals_np).to(device),
                )
            self._table = table
        return table

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
        width = self._max_num_reqs if padded_num_reqs is None else int(padded_num_reqs)
        upper = min(end_idx, len(batch_req_ids), start_idx + width)
        slots_np = np.zeros(width, dtype=np.int32)
        seen_slots: set[int] = set()
        any_active = False
        for batch_i in range(start_idx, upper):
            req_id = batch_req_ids[batch_i]
            if req_id is None:
                continue
            slot = self._slot_by_req_id.get(req_id)
            if slot is None or (
                discard_req_ids is not None and req_id in discard_req_ids
            ):
                continue
            assert slot not in seen_slots, (
                f"Duplicate slot {slot} for req_id={req_id!r} in chunk"
            )
            seen_slots.add(slot)
            slots_np[batch_i - start_idx] = slot
            any_active = True
        if not any_active:
            return None
        key = (width, slots_np.tobytes())
        prev = self._last_slot_ids
        if (
            prev is not None
            and prev[0] == key
            and same_target_device(prev[1].device, device)
        ):
            return prev[1]
        dev = torch.from_numpy(slots_np).to(device)
        self._last_slot_ids = (key, dev)
        return dev

    def precompile(
        self, device: torch.device, sync_fn: Callable[..., None]
    ) -> torch.Tensor:
        """Warm ``seed_slot_rows`` at ``[max_num_reqs]`` width."""
        table = self.ensure_table(device)
        seed_slots = torch.zeros(self._max_num_reqs, dtype=torch.int32).to(device)
        seed_vals = np.tile(
            np.asarray(self._sentinel_row, dtype=np.int32), (self._max_num_reqs, 1)
        )
        table = seed_slot_rows(
            table, seed_slots, torch.from_numpy(seed_vals).to(device)
        )
        sync_fn(table)
        self._table = table
        return table
