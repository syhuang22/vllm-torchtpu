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
"""Per-concurrency TPOT floor, interactivity ceiling, and throughput ceiling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .modules import Parallel, StepShape
from .step import spec_cycle_floor, step_floor

if TYPE_CHECKING:
    from .chips import ChipSpec
    from .geometry import ModelGeometry
    from .modules import Dtypes


@dataclass(frozen=True)
class ConcurrencyRow:
    """Serving ceiling row for one decode concurrency."""

    c: int
    rows: int
    binding: str
    hard_floor_ms: float
    serial_floor_ms: float
    tpot_floor_ms: float
    interactivity_ceil_tok_s: float
    output_tput_ceil_per_chip: float
    flag: str


def plan(
    geom: ModelGeometry,
    chip: ChipSpec,
    par: Parallel,
    dt: Dtypes,
    *args: int | Sequence[int],
    **kwargs: float | Sequence[int] | None,
) -> list[ConcurrencyRow]:
    """Compute ConcurrencyRow ceilings across a list of concurrencies."""
    context: int = (
        int(args[0])
        if len(args) > 0 and isinstance(args[0], int)
        else int(kwargs.get("context", 0) or 0)
    )
    raw_conc = (
        args[1]
        if len(args) > 1 and isinstance(args[1], Sequence)
        else kwargs.get("concurrencies", ())
    )
    concurrencies: Sequence[int] = raw_conc if isinstance(raw_conc, Sequence) else ()

    spec_window: int = int(kwargs.get("spec_window", 0) or 0)
    raw_al = kwargs.get("acceptance_length")
    acceptance_length: float | None = float(raw_al) if raw_al is not None else None
    raw_cl = kwargs.get("collective_latency_us")
    collective_latency_us: float | None = float(raw_cl) if raw_cl is not None else None
    raw_tcp = kwargs.get("tensor_cores_per_chip", 2)
    tensor_cores_per_chip: int = int(raw_tcp) if raw_tcp is not None else 2

    chips = par.tp / float(tensor_cores_per_chip)
    if spec_window == 0:
        tokens = 1.0
    elif acceptance_length is None:
        msg = "acceptance_length is required when spec_window > 0"
        raise ValueError(msg)
    else:
        tokens = float(acceptance_length)
    lat_s = (collective_latency_us or 0.0) * 1e-6
    rows_out: list[ConcurrencyRow] = []
    for c in concurrencies:
        if spec_window > 0:
            sf = spec_cycle_floor(
                geom,
                chip,
                par,
                c,
                context,
                spec_window,
                dt,
                collective_latency_s=lat_s,
            )
        else:
            sf = step_floor(
                geom,
                chip,
                par,
                StepShape(batch=c, rows_per_seq=1, context=context),
                dt,
                collective_latency_s=lat_s,
            )
        hard_ms = sf.hard_floor_s * 1e3
        serial_ms = sf.serial_floor_s * 1e3
        tpot_ms = hard_ms / tokens
        interactivity = 1000.0 / tpot_ms
        output_tput = (c * tokens) / sf.hard_floor_s / chips
        flag = "MEASURE" if sf.binding == "chain" else "-"
        rows_out.append(
            ConcurrencyRow(
                c=c,
                rows=c * (spec_window + 1),
                binding=sf.binding,
                hard_floor_ms=hard_ms,
                serial_floor_ms=serial_ms,
                tpot_floor_ms=tpot_ms,
                interactivity_ceil_tok_s=interactivity,
                output_tput_ceil_per_chip=output_tput,
                flag=flag,
            )
        )
    return rows_out
