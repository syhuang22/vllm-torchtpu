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
"""Megakernel core capability limits and default-deny support gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .models import DEPLOYMENT_PRESETS, DeploymentPreset, ModelProfile
from .specs import (
    SUPPORTED_ONCHIP_FAMILIES,
    FusionScope,
    SupportStatus,
    WorkloadSpec,
)

if TYPE_CHECKING:
    from .chips import ChipSpec
    from .roofline import _ScopeContext


@dataclass(frozen=True)
class CoreLimits:
    """Verified limits mirrored from `megakernel/core/exec_spec.py` and `profile.py`."""

    max_batch_rows: int = 128
    verified_ranks: tuple[int, ...] = (1, 2, 4, 8, 16)
    chip_kinds: tuple[str, ...] = ("TPU7x",)
    max_ar_slots_per_layer: int = 2
    nbd: int = 14
    nbe: int = 10
    bank: int = 2 << 20
    ebank: int = 1 << 20
    vmem_mib: int = 62


CORE_LIMITS = CoreLimits()

_CHIP_KIND_MAP: dict[str, str] = {
    "tpu7x": "TPU7x",
}


def _find_deployment_preset(profile_name: str) -> DeploymentPreset | None:
    """Resolve a DeploymentPreset by exact or normalized model preset name."""
    dep = DEPLOYMENT_PRESETS.get(profile_name)
    if dep is not None:
        return dep
    norm = profile_name.lower().replace(".", "_").replace("-", "_")
    for key, val in DEPLOYMENT_PRESETS.items():
        if key.lower().replace(".", "_").replace("-", "_") == norm:
            return val
    return None


def _is_declared_in_preset(
    profile_name: str,
    scope: FusionScope,
    ctx: _ScopeContext,
) -> bool:
    """Return True if deployments.json declares megakernel support for scope."""
    dep = _find_deployment_preset(profile_name)
    if dep is None:
        return False
    for layout in (dep.decode, dep.prefill, dep.saturation):
        mk = layout.megakernel if layout is not None else None
        if not mk:
            continue
        scopes_raw = mk.get("scopes", ())
        scopes = (
            tuple(str(s) for s in scopes_raw)
            if isinstance(scopes_raw, (list, tuple))
            else ()
        )
        if (
            str(mk.get("source", ""))
            and scope in scopes
            and ctx.active_tokens <= int(str(mk.get("max_rows", 0)))
            and ctx.eff_tp == int(str(mk.get("ranks", 0)))
        ):
            return True
    return False


def evaluate_support(
    scope: FusionScope,
    profile: ModelProfile,
    workload: WorkloadSpec,
    chip_spec: ChipSpec,
    ctx: _ScopeContext,
) -> tuple[SupportStatus, tuple[str, ...]]:
    """Return `(support_status, blockers)` for `scope` under `CORE_LIMITS`."""
    if scope == "per_op" or workload.batch <= 0:
        return ("baseline", ())

    has_unmodeled = any(
        fam not in SUPPORTED_ONCHIP_FAMILIES for fam in profile.layer_families
    )
    planner_bad = (
        bool(ctx.plan_error)
        or not profile.is_periodic
        or profile.ar_slots_per_layer > CORE_LIMITS.max_ar_slots_per_layer
    )
    checks = (
        (
            scope in ("single_layer", "layer_group"),
            "scope_without_core_counterpart",
        ),
        (
            ctx.active_tokens > CORE_LIMITS.max_batch_rows,
            "rows_exceed_core_max_batch_rows",
        ),
        (
            ctx.eff_tp not in CORE_LIMITS.verified_ranks,
            "rank_count_outside_core_grid",
        ),
        (
            _CHIP_KIND_MAP.get(chip_spec.name.lower()) not in CORE_LIMITS.chip_kinds,
            "chip_without_core_budget",
        ),
        (ctx.ep_dp > 1, "all_to_all_not_in_core"),
        (has_unmodeled, "unmodeled_layer_family"),
        (profile.architecture_unverified, "model_type_unrecognized"),
        (planner_bad, "planner_rejected"),
    )
    blockers = tuple(tag for cond, tag in checks if cond)
    if blockers:
        return ("hypothetical", blockers)
    if _is_declared_in_preset(profile.name, scope, ctx):
        return ("declared", ())
    return ("within_core_limits", ())
