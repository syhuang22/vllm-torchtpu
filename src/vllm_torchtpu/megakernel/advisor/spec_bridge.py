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
"""Bridge declarative ModelSpec + ExecSpec schedules via planner.plan."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

try:
    import jax.numpy as jnp

    from ..core import planner
except (ImportError, RuntimeError):
    jnp = None
    planner = None

if TYPE_CHECKING:
    from ..core import exec_spec, spec

from ..advisor.models import ModelProfile
from ..core.errors import MegakernelConfigError
from .specs import AdvisorReport, MoEMode, WorkloadSpec
from .verdict import advise

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .chips import ChipSpec


class MegakernelPlannerUnavailableError(MegakernelConfigError):
    """Raised when JAX or the planner is unavailable."""

    def __init__(self, message: str = "Megakernel planner requires JAX.") -> None:
        """Initialize the exception with an informative message."""
        super().__init__(message)


def _try_plan_spec(
    model_spec: spec.ModelSpec, ex: exec_spec.ExecSpec
) -> tuple[str, bool, Mapping[str, int], int, str, str, str]:
    """Isolate planner.plan and return parsed weights, ring slots, and dtypes."""
    if planner is None or jnp is None:
        raise MegakernelPlannerUnavailableError
    plan_err = ""
    ar_slots = 2
    is_periodic = True
    mlc: Mapping[str, int] = {}
    wd, ed, kv_dt = "fp8", "", "fp8"
    try:
        plan_obj = planner.plan(model_spec, ex)
        wds = tuple(plan_obj.weight_decls.values())
        mlc = plan_obj.module_layer_counts
        ar_slots = sum(x is not None for x in plan_obj.slot_ar_index)
        rings = ([w for w in wds if w.stream_ring == r] for r in ("static", "dynamic"))
        kv_states = [
            s
            for k, s in plan_obj.state_decls.items()
            if "dn" not in plan_obj.state_owners[k]
        ]
        wd, ed, kv_dt = [
            ("fp8" if jnp.dtype(x[0].dtype).itemsize == 1 else "bf16") if x else d
            for x, d in zip((*rings, kv_states), ("fp8", "", "fp8"), strict=True)
        ]
    except MegakernelConfigError as exc:
        plan_err, is_periodic = str(exc), False
    return plan_err, is_periodic, mlc, ar_slots, wd, ed, kv_dt


def advise_from_spec(
    model_spec: spec.ModelSpec,
    ex: exec_spec.ExecSpec,
    chip: ChipSpec | str = "tpu7x",
    moe_mode: MoEMode = "masked_on_chip",
    op_overhead_us: float = 0.0,
) -> AdvisorReport:
    """Bridge declarative ModelSpec + ExecSpec schedules via planner.plan."""
    plan_err, is_periodic, mlc, ar_slots, wd, ed, kv_dt = _try_plan_spec(model_spec, ex)

    cfg = dataclasses.asdict(model_spec.config)
    num_layers = int(cfg["L"])
    hidden = int(cfg["H"])
    att_every = int(cfg.get("att_every") or 1)
    num_experts = int(cfg.get("E") or 0)
    num_attn = sum(v for k, v in mlc.items() if "gqa" in k or "att" in k)
    num_attn = min(
        num_layers,
        num_attn or (num_layers // att_every if att_every > 1 else num_layers),
    )
    prof = ModelProfile(
        name=model_spec.name,
        num_layers=num_layers,
        hidden_size=hidden,
        intermediate_size=int(cfg.get("I") or 4 * hidden),
        num_attention_heads=int(cfg.get("NH") or ex.R),
        num_kv_heads=int(cfg.get("NKV") or ex.R),
        head_dim=int(cfg.get("HD") or 128),
        num_linear_layers=max(0, num_layers - num_attn),
        linear_key_heads=int(cfg.get("KH") or 0),
        linear_value_heads=int(cfg.get("VH") or 0),
        num_experts=num_experts,
        top_k=int(cfg.get("TOPK") or 0) * (num_experts > 0),
        moe_intermediate_size=int(cfg.get("I") or 0),
        shared_expert_intermediate_size=int(cfg.get("IS") or 0),
        num_moe_layers=num_layers * (num_experts > 0),
        num_attention_layers=num_attn,
        linear_key_dim=int(cfg.get("KD") or 128),
        linear_value_dim=int(cfg.get("VD") or 128),
        is_periodic=is_periodic,
        ar_slots_per_layer=ar_slots,
        weight_dtype=wd,
        expert_dtype=ed or wd,
        kv_dtype=kv_dt,
    )
    eff_ep = ex.R if (num_experts > 0 and not bool(cfg.get("moe_tp", True))) else 1
    wl = WorkloadSpec(
        phase="verify" if ex.T > 1 else "decode",
        batch=ex.NS if ex.T > 1 else ex.B,
        isl=ex.ctx,
        osl=0,
        context=ex.ctx,
        spec_tokens=max(0, ex.T - 1),
        tp=ex.R,
        ep=eff_ep,
        moe_mode=moe_mode,
        tiles=ex.tiles,
        op_overhead_us=op_overhead_us,
    )
    return advise(prof, wl, chip, plan_err)
