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
"""Advise decision engine and margin-sensitivity evaluation for TPU advisor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import ceil
from typing import TYPE_CHECKING

from ..advisor.models import (
    DEPLOYMENT_PRESETS,
    ModelProfile,
    ParallelLayout,
    resolve_model_profile,
)
from ..core import errors
from . import roofline, vmem
from .chips import ChipSpec, resolve_chip_spec
from .roofline import (
    _BOUNDARY_SYNC_HOP_FACTOR,
    _PER_OP_COLLECTIVE_CALL_FACTOR,
    _build_scope_context,
    _evaluate_scopes,
    _is_calibrated_trace,
    _resolve_calibrated_priors,
)
from .specs import (
    AdvisorReport,
    FusionScope,
    MeasurementsData,
    ProvenanceMetric,
    ProvenanceTag,
    ScopeEstimate,
    WorkloadSpec,
    load_measurements,
)
from .vmem import _mesh_divisibility_reasons

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from .roofline import _ScopeContext

_DOT_FP8 = "fp8"
_DOT_BF16 = "bf16"
_AB_KNOB = "ab_step_time"
_SETTLE_CMD = "Pass --measurements measurements.json (schema advisor.measurements.v1)"
_CORNER_AXES: Mapping[str, int] = {
    "launch_gap_us": 0,
    "collective_latency_us": 1,
    "prefill_attn_dot_dtype": 2,
}
_Corner = tuple[float, float, str]


@dataclass(frozen=True)
class _AssumedBox:
    """Low and high value of each step-time input shared by every scope."""

    launch_gap_us: tuple[float, float]
    collective_s: tuple[float, float]
    attn_dots: tuple[str, ...]


def _assumed_box(
    ctx: _ScopeContext,
    workload: WorkloadSpec,
    priors: tuple[
        tuple[float, float, float, ProvenanceTag, str],
        tuple[float, float, float, ProvenanceTag, str],
        tuple[float, list[float], ProvenanceTag, str],
    ],
) -> _AssumedBox:
    """Build the input box: measured values when present, else the assumed band."""
    (_, lg_lo, lg_hi, lg_tag, _), (c_val, _, _, c_tag, _), _ = priors
    # Uncalibrated floor keeps the zero-launch roofline; a trace floor uses p50.
    lg_box = (
        (lg_lo, lg_hi)
        if lg_tag in ("MEASURED", "DERIVED")
        or workload.dispatch_regime in ("unfused_eager", "compiled_xla")
        else (workload.op_overhead_us, ctx.launch_gap_upper_us)
    )
    coll_box = (
        (c_val * 1e-6, c_val * 1e-6)
        if c_tag == "MEASURED"
        else (ctx.lat_coll, ctx.lat_coll_upper)
    )
    dot_differs = workload.phase in ("prefill", "mixed") and ctx.flops_kv > 0
    dots = (_DOT_FP8, _DOT_BF16) if dot_differs else (_DOT_FP8,)
    return _AssumedBox(lg_box, coll_box, dots)


def _apply_box(
    ctx: _ScopeContext, workload: WorkloadSpec, box: _AssumedBox
) -> tuple[_ScopeContext, WorkloadSpec]:
    """Price scope floors at the box low corner and band tops at its high corner."""
    new_ar_base_lat_s = ctx.ar_intra_hops * box.collective_s[0] + ctx.ar_inter_latency_s
    new_ctx = replace(
        ctx,
        lat_coll=box.collective_s[0],
        lat_coll_upper=box.collective_s[1],
        ar_base_lat_s=new_ar_base_lat_s,
        launch_gap_upper_us=box.launch_gap_us[1],
    )
    return new_ctx, replace(workload, op_overhead_us=box.launch_gap_us[0])


def _evaluate_corners(
    ctx: _ScopeContext,
    profile: ModelProfile,
    workload: WorkloadSpec,
    box: _AssumedBox,
) -> dict[_Corner, dict[FusionScope, float]]:
    """Step time of every feasible scope at every corner of the assumed box.

    Every scope at a corner sees the same launch gap, collective latency, and
    attention dtype. Collective latency moves only the all-reduce and cross-DP
    MoE hop term, as in `step_upper_us`, so the all-low corner equals
    `step_bound_us` and the all-high corner equals `step_upper_us`.
    """
    table: dict[_Corner, dict[FusionScope, float]] = {}
    for lg in dict.fromkeys(box.launch_gap_us):
        for coll in dict.fromkeys(box.collective_s):
            ar_lat = ctx.ar_intra_hops * coll + ctx.ar_inter_latency_s
            for dot in box.attn_dots:
                est, _ = _evaluate_scopes(
                    replace(
                        ctx,
                        lat_coll=coll,
                        lat_coll_upper=coll,
                        ar_base_lat_s=ar_lat,
                        launch_gap_upper_us=lg,
                    ),
                    profile,
                    replace(workload, op_overhead_us=lg),
                    bf16_attention_upper=dot == _DOT_BF16,
                )
                table[(lg, round(coll * 1e6, 3), dot)] = {
                    e.scope: e.step_upper_us for e in est if e.feasible
                }
    return table


def _corner_winners(
    corners: Mapping[_Corner, Mapping[FusionScope, float]],
    ops: Mapping[FusionScope, int],
) -> dict[_Corner, FusionScope]:
    """Fastest scope at each corner; fewer launches break exact ties."""
    return {
        c: min(steps, key=lambda s: (steps[s], ops[s]))
        for c, steps in corners.items()
        if steps
    }


def _margin_swing(
    corners: Mapping[_Corner, Mapping[FusionScope, float]],
    best: FusionScope,
    challenger: FusionScope,
    axis: str,
) -> float:
    """Largest shift in the challenger-minus-best margin across one input's band."""
    idx = _CORNER_AXES[axis]
    pair = {best, challenger}
    swing = 0.0
    for c_a, steps_a in corners.items():
        for c_b, steps_b in corners.items():
            moved_only_axis = all(c_a[k] == c_b[k] for k in range(3) if k != idx)
            if c_a == c_b or not moved_only_axis:
                continue
            if pair <= steps_a.keys() and pair <= steps_b.keys():
                d_a = steps_a[challenger] - steps_a[best]
                d_b = steps_b[challenger] - steps_b[best]
                swing = max(swing, abs(d_a - d_b))
    return round(swing, 2)


def _decisive_input(
    winners: Mapping[_Corner, FusionScope], candidates: Sequence[str]
) -> str:
    """First candidate input that leaves one winner when held at either band end."""
    for name in candidates:
        idx = _CORNER_AXES[name]
        for value in {c[idx] for c in winners}:
            if len({w for c, w in winners.items() if c[idx] == value}) == 1:
                return name
    return candidates[0] if candidates else _AB_KNOB


def _collect_unresolved_assumptions(
    workload: WorkloadSpec,
    chip_spec: ChipSpec,
    ctx: _ScopeContext,
    bounds: tuple[
        int,
        float,
        tuple[float, float, ProvenanceTag],
        tuple[float, float, ProvenanceTag],
        ProvenanceTag,
    ],
    contest: tuple[
        Mapping[_Corner, Mapping[FusionScope, float]],
        FusionScope,
        FusionScope | None,
    ],
) -> list[dict[str, object]]:
    """Collect and sort unresolved ASSUMED inputs by step-time sensitivity.

    `sensitivity_us` of a scope input is how far its band moves the margin
    between the best scope and its challenger (the corner winner on a tie,
    else the runner-up). `affects` is `scope` for inputs that can change the
    recommended scope and `gain` for `step_coverage`, which scales gain only.
    """
    d_ops, b_val, (lg_lo, lg_hi, lg_tag), (coll_lo, coll_hi, coll_tag), cov_tag = bounds
    corners, best_scope, challenger = contest

    def scope_sens(axis: str, fallback_us: float) -> float:
        """Margin swing vs the challenger, else the per_op-relative estimate."""
        if challenger is None:
            return round(fallback_us, 2)
        return _margin_swing(corners, best_scope, challenger, axis)

    unresolved: list[dict[str, object]] = []
    if lg_tag == "ASSUMED":
        unresolved.append(
            {
                "name": "launch_gap_us",
                "band": [lg_lo, lg_hi],
                "unit": "us",
                "sensitivity_us": scope_sens("launch_gap_us", d_ops * (lg_hi - lg_lo)),
                "affects": "scope",
                "how_to_settle": _SETTLE_CMD,
            }
        )
    if coll_tag == "ASSUMED" and ctx.eff_tp * workload.ep > 1:
        unresolved.append(
            {
                "name": "collective_latency_us",
                "band": [coll_lo, coll_hi],
                "unit": "us/hop",
                "sensitivity_us": scope_sens(
                    "collective_latency_us",
                    ctx.num_ar * ctx.ar_hops * max(0.0, coll_hi - coll_lo),
                ),
                "affects": "scope",
                "how_to_settle": _SETTLE_CMD,
            }
        )
    if (
        workload.phase in ("prefill", "mixed")
        and chip_spec.native_ocp_fp8
        and ctx.flops_kv > 0
    ):
        attn_sens_us = (ctx.flops_kv / ctx.bf16_peak - ctx.flops_kv / ctx.peak_w) * 1e6
        unresolved.append(
            {
                "name": "prefill_attn_dot_dtype",
                "band": [
                    round(ctx.bf16_peak / 1e12, 1),
                    round(ctx.peak_w / 1e12, 1),
                ],
                "unit": "TFLOP/s",
                "sensitivity_us": scope_sens(
                    "prefill_attn_dot_dtype", max(0.0, attn_sens_us)
                ),
                "affects": "scope",
                "how_to_settle": "Inspect XProf HLO attention dot precision",
            }
        )
    if cov_tag == "ASSUMED":
        unresolved.append(
            {
                "name": "step_coverage",
                "band": [0.0, 1.0],
                "unit": "ratio",
                "sensitivity_us": round(b_val, 2),
                "affects": "gain",
                "how_to_settle": _SETTLE_CMD,
            }
        )
    unresolved.sort(key=lambda item: float(str(item["sensitivity_us"])), reverse=True)
    return unresolved


def _build_provenance_and_verdict(
    profile: ModelProfile,
    workload: WorkloadSpec,
    chip_spec: ChipSpec,
    ctx: _ScopeContext,
    info: tuple[
        list[ScopeEstimate],
        ScopeEstimate,
        int,
        bool,
        MeasurementsData | None,
        Mapping[_Corner, Mapping[FusionScope, float]],
        float,
    ],
) -> tuple[
    dict[str, ProvenanceMetric],
    tuple[str, ...],
    str,
    str,
    tuple[dict[str, object], ...],
    tuple[_Corner, FusionScope] | None,
]:
    """Assemble provenance-tagged metrics, flags, confidence, and verdict."""
    estimates, best, ridge_tokens, hbm_fits, meas, corners, lg_floor = info
    per_op_sc = estimates[0]
    d_ops = max(0, per_op_sc.ops_per_step - best.ops_per_step)
    (
        (lg_val, lg_lo, lg_hi, lg_tag, lg_src),
        (coll_val, coll_lo, coll_hi, coll_tag, coll_src),
        (cov_val, cov_band, cov_tag, cov_src),
    ) = _resolve_calibrated_priors(workload, chip_spec, ctx, meas)

    # step_bound_us already carries ops * lg_floor; strip it before re-pricing.
    base_per_op_us = per_op_sc.step_bound_us - per_op_sc.ops_per_step * lg_floor
    base_best_us = best.step_bound_us - best.ops_per_step * lg_floor
    pure_diff_us = max(0.0, base_per_op_us - base_best_us)
    pp_zero_time = (
        ctx.token_residual_bytes / chip_spec.ici_link_bw_bytes_per_s
        if workload.pp > 1
        else 0.0
    )
    est_zero, _ = _evaluate_scopes(
        replace(
            ctx,
            lat_coll=0.0,
            lat_coll_upper=0.0,
            ar_base_lat_s=0.0,
            ar_inter_latency_s=0.0,
            pp_stage_time=pp_zero_time,
            launch_gap_upper_us=0.0,
        ),
        profile,
        replace(workload, op_overhead_us=0.0),
    )
    zero_map = {e.scope: e for e in est_zero}
    fact_diff_us = round(
        max(
            0.0,
            zero_map["per_op"].step_bound_us - zero_map[best.scope].step_bound_us,
        ),
        2,
    )
    pure_diff_round_us = round(pure_diff_us, 2)
    assumed_diff_us = round(max(0.0, pure_diff_round_us - fact_diff_us), 2)
    c_gain_tag: ProvenanceTag = (
        "ASSUMED" if (assumed_diff_us > 0.0 and coll_tag != "MEASURED") else "DERIVED"
    )
    b_val = round(d_ops * lg_val + pure_diff_us, 2)
    b_lo = round(d_ops * lg_lo + pure_diff_us, 2)
    b_hi = round(
        d_ops * lg_hi
        + pure_diff_us
        + ctx.num_ar * ctx.ar_hops * max(0.0, coll_hi - coll_lo),
        2,
    )
    b_tag: ProvenanceTag = (
        "MEASURED" if lg_tag == "MEASURED" and coll_tag == "MEASURED" else "ASSUMED"
    )

    denom_us = max(1e-6, base_per_op_us + per_op_sc.ops_per_step * lg_val)
    fg_val = round(min(1.0, max(0.0, b_val / denom_us)), 4)
    fg_lo = round(min(fg_val, max(0.0, b_lo / denom_us)), 4)
    fg_hi = round(max(fg_val, min(1.0, b_hi / denom_us)), 4)
    eg_val, eg_lo, eg_hi = (
        round(fg_val * cov_val, 4),
        round(fg_lo * cov_band[0], 4),
        round(fg_hi * cov_band[1], 4),
    )

    # Tokens finished per step across the slice, divided by physical chips.
    # active_tokens is per core: PCP splits the window, TP and DCP replicate it,
    # and DP cancels (each replica owns its own tokens and devices).
    eff_pcp = workload.pcp if workload.phase in ("prefill", "mixed") else 1
    tokens_per_chip = (
        ctx.active_tokens
        * eff_pcp
        * chip_spec.devices_per_chip
        / max(1, ctx.eff_tp * workload.pp)
    )
    tput_hi = round(tokens_per_chip / max(1e-9, best.step_bound_us * 1e-6), 2)
    tput_lo = round(
        tokens_per_chip / max(1e-9, max(best.step_bound_us, best.step_upper_us) * 1e-6),
        2,
    )

    metrics: dict[str, ProvenanceMetric] = {
        "fits_hbm": {
            "value": hbm_fits,
            "unit": "bool",
            "tag": "DERIVED",
            "source": "hbm_used <= chip_spec.hbm_capacity_bytes",
            "band": [float(hbm_fits), float(hbm_fits)],
        },
        "tokens_per_core": {
            "value": ctx.active_tokens,
            "unit": "tok",
            "tag": "DERIVED",
            "source": "WorkloadSpec batch * phase tokens / (dp * pcp)",
            "band": [float(ctx.active_tokens), float(ctx.active_tokens)],
        },
        "onchip_ceiling": {
            "value": best.t_vmem_max,
            "unit": "tok",
            "tag": "DERIVED",
            "source": "(vmem_budget - static_vmem_bytes) // per_token_vmem_bytes",
            "band": [float(best.t_vmem_max), float(best.t_vmem_max)],
        },
        "compute_bound_at": {
            "value": ridge_tokens,
            "unit": "tok",
            "tag": "DERIVED",
            "source": "(w_step + state_bytes) * ridge_flop_per_byte / flop_per_tok",
            "band": [float(ridge_tokens), float(ridge_tokens)],
        },
        "launch_gap_us": {
            "value": lg_val,
            "unit": "us",
            "tag": lg_tag,
            "source": lg_src,
            "band": [lg_lo, lg_hi],
        },
        "collective_latency_us": {
            "value": coll_val,
            "unit": "us/hop",
            "tag": coll_tag,
            "source": coll_src,
            "band": [coll_lo, coll_hi],
        },
        "step_floor_us": {
            "value": best.step_bound_us,
            "unit": "us",
            "tag": "DERIVED",
            "source": f"roofline floor for scope={best.scope}",
            "band": [best.step_bound_us, max(best.step_bound_us, best.step_upper_us)],
        },
        "throughput_tok_s_per_chip": {
            "value": tput_hi,
            "unit": "tok/s/chip",
            "tag": "DERIVED",
            "source": "tokens/core * pcp * devices_per_chip / (tp * pp) / floor",
            "band": [min(tput_lo, tput_hi), max(tput_lo, tput_hi)],
        },
        "boundary_cost_us": {
            "value": b_val,
            "unit": "us",
            "tag": b_tag,
            "source": "(per_op_ops - scope_ops) * launch_gap_us + collective_saving_us",
            "band": [min(b_lo, b_hi), max(b_lo, b_hi)],
        },
        "gain_fact_us": {
            "value": fact_diff_us,
            "unit": "us",
            "tag": "DERIVED",
            "source": (
                "per_op_zero_hop_us - best_zero_hop_us "
                "(HBM weight/activation/staging save)"
            ),
            "band": [fact_diff_us, fact_diff_us],
        },
        "gain_assumed_us": {
            "value": assumed_diff_us,
            "unit": "us",
            "tag": (
                "MEASURED"
                if coll_tag == "MEASURED"
                else ("ASSUMED" if assumed_diff_us > 0.0 else "DERIVED")
            ),
            "source": (
                "compiler_proof_gain_us - gain_fact_us "
                "(collective hop & call-multiplier save)"
            ),
            "band": [
                assumed_diff_us,
                round(
                    assumed_diff_us
                    + ctx.num_ar * ctx.ar_hops * max(0.0, coll_hi - coll_lo),
                    2,
                ),
            ],
        },
        "compiler_proof_gain_us": {
            "value": round(pure_diff_us, 2),
            "unit": "us",
            "tag": c_gain_tag,
            "source": "per_op_physics_us - best_scope_physics_us (0 us boundary gap)",
            "band": [
                round(pure_diff_us, 2),
                round(
                    pure_diff_us
                    + ctx.num_ar * ctx.ar_hops * max(0.0, coll_hi - coll_lo),
                    2,
                ),
            ],
        },
        "compiler_proof_gain_pct": {
            "value": round(100.0 * pure_diff_us / max(1e-6, base_per_op_us), 2),
            "unit": "pct",
            "tag": c_gain_tag,
            "source": "100 * compiler_proof_gain_us / per_op_physics_us",
            "band": [
                round(100.0 * pure_diff_us / max(1e-6, base_per_op_us), 2),
                round(
                    100.0
                    * (
                        pure_diff_us
                        + ctx.num_ar * ctx.ar_hops * max(0.0, coll_hi - coll_lo)
                    )
                    / max(1e-6, base_per_op_us),
                    2,
                ),
            ],
        },
        "achievable_gain_pct": {
            "value": round(
                100.0
                * (per_op_sc.step_achievable_us - best.step_achievable_us)
                / max(1e-6, per_op_sc.step_achievable_us),
                2,
            ),
            "unit": "pct",
            "tag": c_gain_tag,
            "source": "100 * achievable_saving_us / per_op_achievable_us",
            "band": [
                round(
                    100.0
                    * (per_op_sc.step_achievable_us - best.step_achievable_us)
                    / max(1e-6, per_op_sc.step_achievable_us),
                    2,
                ),
                round(
                    100.0
                    * (per_op_sc.step_achievable_us - best.step_achievable_us)
                    / max(1e-6, per_op_sc.step_achievable_us),
                    2,
                ),
            ],
        },
        "dispatch_bubble_gain_us": {
            "value": round(d_ops * lg_val, 2),
            "unit": "us",
            "tag": lg_tag,
            "source": "(per_op_ops - best_scope_ops) * launch_gap_us",
            "band": [round(d_ops * lg_lo, 2), round(d_ops * lg_hi, 2)],
        },
        "fusion_gain_max": {
            "value": fg_val,
            "unit": "ratio",
            "tag": "DERIVED",
            "source": "boundary_cost_us / per_op_step_us",
            "band": [fg_lo, fg_hi],
        },
        "step_coverage": {
            "value": cov_val,
            "unit": "ratio",
            "tag": cov_tag,
            "source": cov_src,
            "band": cov_band,
        },
        "expected_gain_max": {
            "value": eg_val,
            "unit": "ratio",
            "tag": "DERIVED",
            "source": "fusion_gain_max * step_coverage",
            "band": [eg_lo, eg_hi],
        },
        "eta_hbm": {
            "value": workload.efficiency.eta_hbm,
            "unit": "ratio",
            "tag": "ASSUMED",
            "source": "WorkloadSpec.efficiency.eta_hbm",
            "band": list(workload.efficiency.BRACKETS["eta_hbm"]),
        },
        "eta_dense": {
            "value": workload.efficiency.eta_dense,
            "unit": "ratio",
            "tag": "ASSUMED",
            "source": "WorkloadSpec.efficiency.eta_dense",
            "band": list(workload.efficiency.BRACKETS["eta_dense"]),
        },
        "eta_moe_grouped": {
            "value": workload.efficiency.eta_moe_grouped,
            "unit": "ratio",
            "tag": "ASSUMED",
            "source": "WorkloadSpec.efficiency.eta_moe_grouped",
            "band": list(workload.efficiency.BRACKETS["eta_moe_grouped"]),
        },
        "eta_ici": {
            "value": workload.efficiency.eta_ici,
            "unit": "ratio",
            "tag": "ASSUMED",
            "source": "WorkloadSpec.efficiency.eta_ici",
            "band": list(workload.efficiency.BRACKETS["eta_ici"]),
        },
        "per_op_collective_call_factor": {
            "value": float(_PER_OP_COLLECTIVE_CALL_FACTOR),
            "unit": "x",
            "tag": "ASSUMED",
            "source": "roofline._PER_OP_COLLECTIVE_CALL_FACTOR",
            "band": [1.0, float(_PER_OP_COLLECTIVE_CALL_FACTOR)],
        },
        "boundary_sync_hop_factor": {
            "value": float(_BOUNDARY_SYNC_HOP_FACTOR),
            "unit": "hops",
            "tag": "ASSUMED",
            "source": "roofline._BOUNDARY_SYNC_HOP_FACTOR",
            "band": [0.0, float(_BOUNDARY_SYNC_HOP_FACTOR)],
        },
    }

    hbm_pool_overflow = (
        ctx.hbm_used > workload.hbm_utilization_cap * chip_spec.hbm_capacity_bytes
    )
    flags_list = [
        flag
        for cond, flag in (
            (profile.architecture_unverified, "architecture_unverified"),
            (best.n_splits > 1, "multi_call_unmodeled"),
            (hbm_pool_overflow, "hbm_pool_overflow"),
            (True, "core_limits_pinned"),
        )
        if cond
    ]
    feasible_others = [e for e in estimates if e.feasible and e.scope != best.scope]
    runner_up = (
        min(feasible_others, key=lambda e: (e.step_bound_us, e.ops_per_step))
        if feasible_others
        else None
    )
    # Every scope is priced at the same corner; a tie needs the winner to change.
    ops = {e.scope: e.ops_per_step for e in estimates}
    winners = _corner_winners(corners, ops) if hbm_fits else {}
    flip = next(((c, w) for c, w in winners.items() if w != best.scope), None)
    challenger = flip[1] if flip else (runner_up.scope if runner_up else None)
    unresolved = _collect_unresolved_assumptions(
        workload,
        chip_spec,
        ctx,
        (
            d_ops,
            b_val,
            (lg_lo, lg_hi, lg_tag),
            (coll_lo, coll_hi, coll_tag),
            cov_tag,
        ),
        (corners, best.scope, challenger),
    )
    scope_inputs = [str(u["name"]) for u in unresolved if u["affects"] == "scope"]
    is_tie = hbm_fits and flip is not None
    verdict = (
        f"tie(measure: {_decisive_input(winners, scope_inputs)})"
        if is_tie
        else _scope_verdict(best.scope if hbm_fits else "per_op")
    )
    confidence = (
        "validated_end_to_end"
        if meas is not None and meas.step_time_us is not None
        else ("bounded" if _is_calibrated_trace(meas) else "prior_only")
    )
    return metrics, tuple(flags_list), verdict, confidence, tuple(unresolved), flip


def _format_advise_reason(
    best: ScopeEstimate,
    estimates: Sequence[ScopeEstimate],
    ridge_tokens: int,
    used_gb: float,
    cap_gb: float,
) -> str:
    """Format rationale: best scope, runner-up, and per_op."""
    if used_gb > cap_gb:
        return f"Model+cache ({used_gb:.2f} GB) > HBM ({cap_gb:.2f} GB)"
    others = [e for e in estimates if e.feasible and e.scope != best.scope]
    text = f"{best.scope} ({best.step_bound_us:.1f} us)"
    if others:
        ru = min(others, key=lambda e: (e.step_bound_us, e.ops_per_step))
        delta = ru.step_bound_us - best.step_bound_us
        pct = 100.0 * delta / max(1e-9, best.step_bound_us)
        text += (
            f" vs runner-up {ru.scope} ({ru.step_bound_us:.1f} us, "
            f"+{delta:.1f} us, {pct:.1f}%)"
        )
        if best.scope != "per_op" and ru.scope != "per_op":
            text += f"; per_op {estimates[0].step_bound_us:.1f} us"
    if best.scope == "per_op":
        text += f"; compute-bound at T={ridge_tokens}"
    return text


def _scope_verdict(scope: FusionScope) -> str:
    """Name the build decision: multi-layer megakernel, per-layer kernel, or ops."""
    if scope in ("whole_step", "layer_group"):
        return "megakernel_likely"
    if scope == "single_layer":
        return "fused_layer_likely"
    return "separate_likely"


def _flip_note(flip: tuple[_Corner, FusionScope] | None) -> str:
    """Name the scope that wins at the corner where the verdict flips."""
    if flip is None:
        return ""
    (lg, coll, dot), alt = flip
    return (
        f"; {alt} wins at launch gap {lg:.1f} us, collective "
        f"{coll:.2f} us/hop, {dot} attention"
    )


def advise(
    profile: ModelProfile | str | Mapping[str, object],
    workload: WorkloadSpec,
    chip: ChipSpec | str = "tpu7x",
    plan_error: str = "",
    **extra: object,
) -> AdvisorReport:
    """Evaluate fusion scopes from first-principles hardware and model bounds."""
    prof = resolve_model_profile(profile)
    chip_spec_path = extra.get("chip_spec_path")
    cs_path = str(chip_spec_path) if chip_spec_path is not None else None
    chip_spec = resolve_chip_spec(chip, chip_spec_path=cs_path)
    meas_arg = extra.get("measurements")
    meas = (
        load_measurements(meas_arg, chip_name=chip_spec.name)
        if meas_arg is not None
        else None
    )
    eff_wl = workload
    if meas is not None and meas.ring_plan is not None and eff_wl.ring_plan is None:
        eff_wl = replace(eff_wl, ring_plan=meas.ring_plan)

    vmem.validate_advise_inputs(prof, eff_wl)
    max_usable_mib = chip_spec.vmem_usable_bytes / (1 << 20)
    if eff_wl.vmem_budget_mib is not None and eff_wl.vmem_budget_mib > max_usable_mib:
        phys_mib = chip_spec.vmem_capacity_bytes / (1 << 20)
        msg = (
            f"vmem_budget_mib ({eff_wl.vmem_budget_mib}) exceeds usable VMEM "
            f"ceiling ({max_usable_mib:.1f} MiB usable out of {phys_mib:.1f} MiB "
            f"physical) on chip {chip_spec.name!r}"
        )
        raise errors.MegakernelConfigError(msg)
    ctx = _build_scope_context(
        prof, eff_wl, chip_spec, plan_error=plan_error, meas=meas
    )
    box = _assumed_box(
        ctx, eff_wl, _resolve_calibrated_priors(eff_wl, chip_spec, ctx, meas)
    )
    ctx, eval_wl = _apply_box(ctx, eff_wl, box)
    estimates, raw_step_map = _evaluate_scopes(ctx, prof, eval_wl)
    corners = _evaluate_corners(ctx, prof, eval_wl, box)
    return _assemble_advisor_report(
        prof,
        eff_wl,
        chip_spec,
        ctx,
        (estimates, raw_step_map, corners, meas, eval_wl),
    )


def _assemble_advisor_report(
    prof: ModelProfile,
    eff_wl: WorkloadSpec,
    chip_spec: ChipSpec,
    ctx: _ScopeContext,
    eval_bundle: tuple[
        list[ScopeEstimate],
        Mapping[FusionScope, float],
        Mapping[_Corner, Mapping[FusionScope, float]],
        MeasurementsData | None,
        WorkloadSpec,
    ],
) -> AdvisorReport:
    """Assemble final AdvisorReport from roofline bounds and calibrated priors."""
    estimates, raw_step_map, corners, meas, eval_wl = eval_bundle
    hbm_used = ctx.hbm_used
    hbm_fits = hbm_used <= chip_spec.hbm_capacity_bytes
    candidates = [e for e in estimates if e.feasible] if hbm_fits else [estimates[0]]
    best = min(candidates, key=lambda e: (raw_step_map[e.scope], e.ops_per_step))
    supported_candidates = [
        e
        for e in candidates
        if e.support in ("baseline", "within_core_limits", "declared")
    ] or [estimates[0]]
    recommended = (
        min(
            supported_candidates,
            key=lambda e: (raw_step_map[e.scope], e.ops_per_step),
        )
        if hbm_fits
        else estimates[0]
    )
    used_gb, cap_gb = hbm_used / 1e9, chip_spec.hbm_capacity_bytes / 1e9
    ridge = ctx.peak_w / ctx.bw_hbm
    b_wa = best.w_step_gb * 1e9 * best.n_splits + best.act_hbm_mb * 1e6
    state_bytes = ctx.kv_cache_read_bytes + ctx.recurrent_state_rw_bytes
    kv_lora, qk_rope = prof.kv_lora_rank, prof.qk_rope_head_dim
    nh_local, q_dim = ctx.nh_local, ctx.q_dim
    v_q = nh_local * (2 * kv_lora + qk_rope) if kv_lora else 2.0 * q_dim
    tot_bytes = max(1.0, b_wa + state_bytes)
    active_tokens = ctx.active_tokens
    eff_ctx = ctx.eff_ctx
    l_moe, l_attn = ctx.l_moe, ctx.l_attn
    flops_dense, flops_moe = ctx.flops_dense, ctx.flops_moe
    v_p = (
        ctx.base_dense_params
        + l_moe * prof.hidden_size * prof.num_experts
        + flops_moe / (2.0 * active_tokens)
    )
    v_flops = 2.0 * ctx.batch_local * ctx.verify_tokens * (v_p + l_attn * eff_ctx * v_q)
    flop_per_tok = max(1.0, (flops_dense + flops_moe) / max(1, active_tokens))
    ridge_tokens = max(
        1,
        ceil((estimates[0].w_step_gb * 1e9 + state_bytes) * ridge / flop_per_tok),
    )
    fastest_scope: FusionScope = best.scope if hbm_fits else "per_op"
    rec_scope: FusionScope = recommended.scope if hbm_fits else "per_op"
    step_ai = round((flops_dense + best.moe_flop_factor * flops_moe) / tot_bytes, 2)
    asymp_ai = round(v_flops / state_bytes, 2) if state_bytes else None
    if l_moe == 0:
        rec_moe = "dense_mlp"
    elif fastest_scope == "per_op":
        rec_moe = "grouped_in_hbm"
    else:
        can_group_rec = roofline.can_group_moe_in_scope(
            fastest_scope, ctx.eff_tp, ctx.ep_dp, eff_wl.moe_mode
        )
        t_per_ep = ceil(active_tokens / max(1, best.n_splits)) * ctx.ep_dp
        rec_moe = (
            "grouped_on_chip"
            if can_group_rec
            and (
                eff_wl.moe_mode == "grouped_on_chip"
                or t_per_ep >= chip_spec.lhs_row_floor(prof.expert_dtype)
            )
            else "masked_on_chip"
        )

    metrics, flags, verdict, confidence, unresolved, flip = (
        _build_provenance_and_verdict(
            prof,
            eff_wl,
            chip_spec,
            ctx,
            (
                estimates,
                best,
                ridge_tokens,
                hbm_fits,
                meas,
                corners,
                eval_wl.op_overhead_us,
            ),
        )
    )
    base_rationale = _format_advise_reason(
        best, estimates, ridge_tokens, used_gb, cap_gb
    ) + _flip_note(flip)
    (
        verdict,
        fastest_verdict,
        rec_scope,
        recommended,
        flags,
        rationale,
        div_reasons,
    ) = _resolve_throughput_and_divisibility(
        prof,
        eff_wl,
        ctx,
        (
            metrics,
            flags,
            verdict,
            rec_scope,
            fastest_scope,
            recommended,
            estimates,
            base_rationale,
        ),
    )

    hypothetical_best: dict[str, object] | None = (
        {
            "scope": best.scope,
            "verdict": fastest_verdict,
            "step_bound_us": best.step_bound_us,
            "step_achievable_us": best.step_achievable_us,
            "support": best.support,
            "blockers": list(best.blockers),
        }
        if (not div_reasons and rec_scope != fastest_scope)
        else None
    )

    return AdvisorReport(
        model=prof.name,
        chip=chip_spec.name,
        phase=eff_wl.phase,
        active_tokens=active_tokens,
        effective_context=eff_ctx,
        vmem_budget_mib=round(ctx.vmem_budget / (1 << 20), 3),
        hbm_fits=hbm_fits,
        hbm_used_gb=round(used_gb, 3),
        hbm_capacity_gb=round(cap_gb, 3),
        ridge_dense_flop_per_byte=round(ridge, 2),
        step_arithmetic_intensity=step_ai,
        asymptotic_decode_intensity=asymp_ai,
        recommended_scope=rec_scope,
        build_megakernel=(not div_reasons) and verdict == "megakernel_likely",
        rationale=rationale,
        scopes=tuple(estimates),
        per_token_vmem_bytes=ctx.per_token_fused_bytes,
        ridge_tokens=ridge_tokens,
        recommended_moe_mode=rec_moe,
        dma_buffers=eff_wl.dma_buffers,
        verdict=verdict,
        confidence=confidence,
        flags=flags,
        metrics=metrics,
        unresolved_assumptions=unresolved,
        fastest_scope=fastest_scope,
        fastest_verdict=fastest_verdict,
        evidence=recommended.support,
        hypothetical_best=hypothetical_best,
        coverage=(
            "decoder layer stack; excludes embedding, LM head, sampling, "
            "draft heads and modality towers"
        ),
        unmodeled_features=prof.unmodeled_features,
    )


def _resolve_throughput_and_divisibility(
    prof: ModelProfile,
    eff_wl: WorkloadSpec,
    ctx: _ScopeContext,
    state: tuple[
        Mapping[str, ProvenanceMetric],
        tuple[str, ...],
        str,
        FusionScope,
        FusionScope,
        ScopeEstimate,
        Sequence[ScopeEstimate],
        str,
    ],
) -> tuple[
    str,
    str,
    FusionScope,
    ScopeEstimate,
    tuple[str, ...],
    str,
    tuple[str, ...],
]:
    """Apply mesh divisibility check and throughput layout comparison to verdict."""
    (
        metrics,
        flags,
        verdict,
        rec_scope,
        fastest_scope,
        recommended,
        estimates,
        rationale,
    ) = state
    chip_spec = ctx.chip_spec
    eff_ctx = ctx.eff_ctx
    div_reasons: tuple[str, ...] = tuple(_mesh_divisibility_reasons(prof, eff_wl))
    fastest_verdict = verdict
    if rec_scope != fastest_scope:
        verdict = _scope_verdict(rec_scope)
    if div_reasons:
        inv_verdict = f"invalid_config({','.join(div_reasons)})"
        return (
            inv_verdict,
            inv_verdict,
            "per_op",
            estimates[0],
            (*flags, *div_reasons),
            rationale,
            div_reasons,
        )
    if eff_wl.objective in ("throughput", "both"):
        tput_val = float(metrics["throughput_tok_s_per_chip"]["value"])
        rationale += f"; throughput={tput_val:.1f} tok/s/chip"
        dep = DEPLOYMENT_PRESETS.get(prof.name)
        if dep is not None and dep.saturation is not None and eff_wl.phase == "decode":
            cur_layout = ParallelLayout(
                tp=eff_wl.tp,
                ep=eff_wl.ep,
                dp=eff_wl.dp,
                pp=eff_wl.pp,
                dcp=eff_wl.dcp,
                pcp=eff_wl.pcp,
                layer_groups=eff_wl.layer_groups,
            )
            if cur_layout.label() != dep.saturation.label():
                comp = compare_parallelism_candidates(
                    prof,
                    (cur_layout, dep.saturation),
                    batch=max(1, eff_wl.batch // max(1, eff_wl.dp)),
                    context=eff_ctx,
                    chip=chip_spec,
                    objective=eff_wl.objective,
                )
                comp_verdict = str(comp["verdict"])
                if eff_wl.objective == "both" and comp_verdict.startswith(
                    "tie(objective_split:"
                ):
                    split_flag: str = "objective_split"
                    flags = (*flags, split_flag)
                    fastest_verdict = comp_verdict
                    if rec_scope == fastest_scope:
                        verdict = comp_verdict
                    rationale += f"; {comp_verdict}"
                elif eff_wl.objective == "throughput":
                    rationale += (
                        f"; best_throughput_layout={comp['best_throughput_layout']}"
                    )
    return (
        verdict,
        fastest_verdict,
        rec_scope,
        recommended,
        flags,
        rationale,
        div_reasons,
    )


def compare_parallelism_candidates(
    profile: ModelProfile | str | Mapping[str, object],
    layouts: Sequence[ParallelLayout],
    *,
    batch: int = 16,
    chip: ChipSpec | str = "tpu7x",
    **opts: object,
) -> dict[str, object]:
    """Compare ParallelLayouts under step-latency and chip-throughput objectives."""
    prof = resolve_model_profile(profile)
    chip_spec = resolve_chip_spec(chip)
    context = int(str(opts.get("context", 4096)))
    objective = str(opts.get("objective", "both"))
    rows: list[dict[str, object]] = []
    for layout in layouts:
        wl = WorkloadSpec(
            phase="decode",
            batch=batch * layout.dp,
            isl=context,
            osl=0,
            context=context,
            tp=layout.tp,
            ep=layout.ep,
            dp=layout.dp,
            pp=layout.pp,
            dcp=layout.dcp,
            pcp=layout.pcp,
            layer_groups=layout.layer_groups,
            moe_mode="grouped_on_chip" if prof.num_experts > 0 else "dense_mlp",
        )
        rep = advise(prof, wl, chip_spec)
        step_us = float(rep.metrics["step_floor_us"]["value"])
        tput = float(rep.metrics["throughput_tok_s_per_chip"]["value"])
        rows.append(
            {
                "layout": layout.label(),
                "step_floor_us": step_us,
                "throughput_tok_s_per_chip": tput,
                "recommended_scope": rep.recommended_scope,
                "hbm_fits": rep.hbm_fits,
            }
        )
    feasible_rows = [r for r in rows if bool(r["hbm_fits"])] or rows
    best_step = min(feasible_rows, key=lambda r: float(str(r["step_floor_us"])))
    best_tput = max(
        feasible_rows, key=lambda r: float(str(r["throughput_tok_s_per_chip"]))
    )
    step_lbl = str(best_step["layout"])
    tput_lbl = str(best_tput["layout"])
    if objective == "step":
        verdict = step_lbl
    elif objective == "throughput":
        verdict = tput_lbl
    elif step_lbl != tput_lbl:
        verdict = f"tie(objective_split: step={step_lbl}, throughput={tput_lbl})"
    else:
        verdict = step_lbl
    return {
        "model": prof.name,
        "chip": chip_spec.name,
        "objective": objective,
        "best_step_layout": step_lbl,
        "best_throughput_layout": tput_lbl,
        "verdict": verdict,
        "candidates": rows,
    }


def compute_scope_raw_step_us(
    model: str | Path | ModelProfile,
    workload: WorkloadSpec,
    *,
    chip: str = "tpu7x",
    measurements: str | Path | MeasurementsData | None = None,
) -> dict[str, float]:
    """Compute unconstrained/raw step latency per scope before boundary padding."""
    prof = resolve_model_profile(model)
    chip_spec = resolve_chip_spec(chip)
    meas = (
        load_measurements(measurements, chip_name=chip_spec.name)
        if measurements is not None
        else None
    )
    eff_wl = workload
    if meas is not None and meas.ring_plan is not None and eff_wl.ring_plan is None:
        eff_wl = replace(eff_wl, ring_plan=meas.ring_plan)
    ctx = _build_scope_context(prof, eff_wl, chip_spec, meas=meas)
    box = _assumed_box(
        ctx,
        eff_wl,
        _resolve_calibrated_priors(eff_wl, chip_spec, ctx, meas),
    )
    ctx, eval_wl = _apply_box(ctx, eff_wl, box)
    _, raw_step_map = _evaluate_scopes(ctx, prof, eval_wl)
    return {str(s): float(v) for s, v in raw_step_map.items()}
