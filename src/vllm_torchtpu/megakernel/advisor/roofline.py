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
"""First-principles roofline, scope assessment, and runtime bounds for TPU advisor."""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import ceil, gcd
from typing import TYPE_CHECKING

from ..advisor.models import ModelProfile, resolve_model_profile
from .chips import ChipSpec, resolve_chip_spec
from .specs import (
    FusionScope,
    MeasurementsData,
    ProvenanceTag,
    ScopeEstimate,
    WorkloadSpec,
    default_double_buffer_ring_plan,
)
from .support import evaluate_support
from .vmem import (
    _ATTN_TILE_TOKENS,
    _CHIPS_PER_HOST,
    _DENSE_FFN_TILE_CAP,
    _DENSE_SLOT_DEFAULT_BYTES,
    _EXPERT_SLOT_DEFAULT_BYTES,
    _GDN_STREAMED_HEADS_CAP,
    _mesh_divisibility_reasons,
    _scope_vmem_and_ops,
    compute_all_reduce_transfer_cost,
    compute_per_token_vmem_bytes,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

_GROUPED_MOE_MODES = frozenset({"grouped_on_chip"})
_DEFAULT_LAUNCH_GAP_FLOOR_US = 1.0
_DEFAULT_LAUNCH_GAP_VAL_US = 2.0
_DEFAULT_LAUNCH_GAP_UPPER_US = 8.0
_COMPILED_XLA_GAP_P50_US = 1.0
_COMPILED_XLA_GAP_P90_US = 1.5
_UNFUSED_EAGER_GAP_US = 6.0
# ASSUMED: per-op XLA collective call multiplier (2x hops/staging vs fused ring).
_PER_OP_COLLECTIVE_CALL_FACTOR = 2
# ASSUMED: barrier synchronization hop multiplier per kernel launch boundary.
_BOUNDARY_SYNC_HOP_FACTOR = 2.0
_FP32_BYTES = 4
_DTYPE_BYTES: Mapping[str, float] = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "fp8_e4m3fn": 1.0,
    "fp8_e5m2": 1.0,
    "int8": 1.0,
    "int4": 0.5,
    "nvfp4": 0.5625,
    "mxfp4": 0.53125,
    "fp4": 0.5625,
}


def validate_workload(
    profile: ModelProfile | str | Mapping[str, object],
    chip: ChipSpec | str,
    workload: WorkloadSpec,
) -> tuple[str, ...]:
    """Validate sharding divisibility and serving HBM pool fit without raising."""
    prof = resolve_model_profile(profile)
    chip_spec = resolve_chip_spec(chip)
    pos_vals = (
        prof.num_layers,
        prof.hidden_size,
        prof.num_attention_heads,
        prof.head_dim,
        workload.batch,
        workload.tp,
        workload.pp,
        workload.dp,
        workload.ep,
        workload.kv_split,
        workload.dcp,
        workload.pcp,
        workload.isl,
    )
    nonneg_vals = (
        workload.osl,
        workload.context,
        workload.spec_tokens,
        workload.chunk_size,
        workload.microbatches,
        workload.segment_layers,
    )
    if min(*pos_vals) < 1 or min(*nonneg_vals) < 0 or workload.op_overhead_us < 0.0:
        return ("invalid_dimensions",)
    reasons = _mesh_divisibility_reasons(prof, workload)
    ctx = _build_scope_context(prof, workload, chip_spec, plan_error="")
    if ctx.hbm_used > workload.hbm_utilization_cap * chip_spec.hbm_capacity_bytes:
        reasons.append("hbm_pool_overflow")
    return tuple(reasons)


def _experts_hit(
    tokens_ep: int,
    top_k: int,
    num_experts: int,
    experts_local: int,
) -> float:
    """Return expected distinct local experts hit by `tokens_ep` tokens."""
    miss = (1.0 - min(1.0, top_k / num_experts)) ** tokens_ep
    return max(1.0, experts_local * (1.0 - miss))


def _strict_moe_weights(ctx: _ScopeContext, profile: ModelProfile) -> float:
    """Return strict lower-bound step weight bytes under perfectly balanced routing."""
    if ctx.experts_local == 0:
        return ctx.w_static
    e_strict = min(
        float(ctx.experts_local),
        max(1.0, profile.top_k * ctx.experts_local / max(1, profile.num_experts)),
    )
    return ctx.w_static + ctx.l_moe * e_strict * ctx.exp_bytes


def _moe_weights_and_padding(
    ctx: _ScopeContext,
    profile: ModelProfile,
    tile_tokens: int,
    *,
    tiled: bool,
    grouped: bool,
) -> tuple[float, float]:
    """Return weight bytes read per step and the MoE matmul padding factor."""
    experts_local, ep_dp = ctx.experts_local, ctx.ep_dp
    if experts_local == 0:
        return ctx.w_static, 1.0

    t_ep = tile_tokens * ep_dp
    e_hit = _experts_hit(t_ep, profile.top_k, profile.num_experts, experts_local)
    pairs = max(1.0, t_ep * profile.top_k * experts_local / profile.num_experts)
    avg_m = max(1.0, pairs / e_hit)
    hit = (
        _experts_hit(
            ctx.active_tokens * ep_dp,
            profile.top_k,
            profile.num_experts,
            experts_local,
        )
        if tiled
        else e_hit
    )
    w_step = ctx.w_static + ctx.l_moe * hit * ctx.exp_bytes
    if grouped:
        return w_step, ceil(avg_m / ctx.row_floor) * ctx.row_floor / avg_m
    rup_16 = ((t_ep + 15) // 16) * 16
    return w_step, max(1.0, (rup_16 * e_hit) / pairs)


def can_group_moe_in_scope(
    scope: FusionScope,
    eff_tp: int,
    ep_dp: int,
    moe_mode: str,
) -> bool:
    """Return whether `scope` can group routed expert tokens on chip or in HBM."""
    return (
        scope not in ("whole_step", "layer_group")
        or ep_dp == 1
        or (eff_tp * ep_dp <= _CHIPS_PER_HOST and moe_mode in _GROUPED_MOE_MODES)
    )


def _resolve_scope_moe_and_acts(
    scope: FusionScope,
    ctx: _ScopeContext,
    profile: ModelProfile,
    workload: WorkloadSpec,
    n_splits: int,
) -> tuple[float, float, float, int, bool]:
    """Return w_step, phi, extra activation bytes, tile_tokens, and grouped_moe."""
    tiled = scope in ("whole_step", "layer_group") and n_splits > 1
    tile_tokens = ceil(ctx.active_tokens / max(1, n_splits))
    can_group_moe = can_group_moe_in_scope(
        scope, ctx.eff_tp, ctx.ep_dp, workload.moe_mode
    )
    grouped_moe = can_group_moe and (
        scope == "per_op"
        or workload.moe_mode in _GROUPED_MOE_MODES
        or tile_tokens * ctx.ep_dp >= ctx.row_floor
    )
    moe_tokens = (
        ctx.active_tokens if grouped_moe and scope != "single_layer" else tile_tokens
    )
    w_step, phi = _moe_weights_and_padding(
        ctx,
        profile,
        moe_tokens,
        tiled=tiled,
        grouped=grouped_moe,
    )
    extra_act = (
        2.0
        * max(1, profile.residual_streams)
        * ctx.layers_local
        * ctx.token_residual_bytes
        if tiled
        else 0.0
    )
    return w_step, phi, extra_act, tile_tokens, grouped_moe


def _compute_scope_collective_times(
    scope: FusionScope,
    ctx: _ScopeContext,
    n_splits: int,
) -> tuple[float, float, int]:
    """Compute sync stage time, comm time, and xla ar factor in seconds."""
    num_ar = ctx.num_ar
    lat_coll = ctx.lat_coll
    l_moe = ctx.l_moe
    ep_dp = ctx.ep_dp
    pp_stage_time = ctx.pp_stage_time
    a2a_time = ctx.a2a_time
    bw_hbm = ctx.bw_hbm

    xla_ar_factor = (
        _PER_OP_COLLECTIVE_CALL_FACTOR if scope == "per_op" and ctx.eff_tp > 1 else 1
    )
    xla_hbm_coll_bytes = (
        2.0 * num_ar * ctx.token_residual_bytes
        if scope == "per_op" and ctx.eff_tp > 1
        else 0.0
    )
    is_fused_multi = scope in ("whole_step", "layer_group")
    overlapped_comm = is_fused_multi and n_splits == 1
    ar_and_pcp_s = (
        num_ar * (ctx.ar_transfer_time_s + n_splits * xla_ar_factor * ctx.ar_base_lat_s)
        + xla_hbm_coll_bytes / bw_hbm
        + ctx.pcp_ring_time
    )
    a2a_sync_s = (
        l_moe * (a2a_time + 2 * n_splits * xla_ar_factor * lat_coll)
        if l_moe > 0 and ep_dp > 1
        else 0.0
    )
    dcn_sync_s = num_ar * n_splits * xla_ar_factor * ctx.ar_inter_latency_s
    sync_stage_s = (
        (dcn_sync_s + pp_stage_time + a2a_sync_s)
        if overlapped_comm
        else (ar_and_pcp_s + pp_stage_time + a2a_sync_s)
        if is_fused_multi
        else 0.0
    )
    comm_s = (
        (ar_and_pcp_s - dcn_sync_s)
        if overlapped_comm
        else 0.0
        if is_fused_multi
        else (ar_and_pcp_s + pp_stage_time + a2a_sync_s)
    )
    return sync_stage_s, comm_s, xla_ar_factor


def _compute_scope_matmuls(
    ctx: _ScopeContext,
    workload: WorkloadSpec,
    phi: float,
    *,
    bf16_attention_upper: bool,
) -> tuple[float, float]:
    """Compute baseline and upper-bound matmul times in seconds."""
    t_matmul = ctx.flops_dense / ctx.peak_w + (phi * ctx.flops_moe) / ctx.peak_e
    t_matmul_upper = (
        (ctx.flops_dense - ctx.flops_kv) / ctx.peak_w
        + ctx.flops_kv / ctx.bf16_peak
        + (phi * ctx.flops_moe) / ctx.peak_e
        if bf16_attention_upper
        and workload.phase in ("prefill", "mixed")
        and ctx.flops_kv > 0
        else t_matmul
    )
    return t_matmul, t_matmul_upper


def _compute_ach_upper_and_strict(
    ctx: _ScopeContext,
    profile: ModelProfile,
    workload: WorkloadSpec,
    flags: tuple[bool, bool, bool, bool, int, int],
    times: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ],
) -> tuple[float, float, float]:
    """Return step_ach_upper_us, w_step_strict, and step_floor_strict_us."""
    (
        tiled,
        is_fused_multi,
        overlapped_comm,
        bf16_attention_upper,
        n_splits,
        n_ops,
    ) = flags
    (
        phi,
        eta_moe,
        t_comp_ach,
        t_hbm_ach,
        comm_s,
        ar_extra_s,
        a2a_extra_s,
        sync_comm_extra_s,
        pipeline_fill_drain_s,
        repass_spill_us,
        step_ach_us,
        step_upper_us,
        act_bytes,
        eff_state_bytes,
        base_ai,
    ) = times
    eff = workload.efficiency
    t_comp_upper_ach = (
        (
            ((ctx.flops_dense - ctx.flops_kv) / ctx.peak_w) / max(1e-3, eff.eta_dense)
            + (ctx.flops_kv / ctx.bf16_peak) / max(1e-3, eff.eta_dense)
            + ((phi * ctx.flops_moe) / ctx.peak_e) / max(1e-3, eta_moe)
        )
        if bf16_attention_upper
        and workload.phase in ("prefill", "mixed")
        and ctx.flops_kv > 0
        else t_comp_ach
    )
    t_comm_upper_ach = (comm_s + ar_extra_s) / max(1e-3, eff.eta_ici)
    ach_upper_core_s = (
        (
            max(t_hbm_ach, t_comp_upper_ach + t_comm_upper_ach)
            + sync_comm_extra_s
            + a2a_extra_s
        )
        if overlapped_comm
        else (
            max(t_comp_upper_ach, t_hbm_ach)
            + t_comm_upper_ach
            + sync_comm_extra_s
            + a2a_extra_s
        )
    )
    step_ach_upper_us = max(
        step_ach_us,
        step_upper_us,
        (ach_upper_core_s + pipeline_fill_drain_s) * 1e6
        + repass_spill_us
        + n_ops * ctx.launch_gap_upper_us,
    )
    w_step_strict = _strict_moe_weights(ctx, profile)
    t_matmul_strict = ctx.flops_dense / ctx.peak_w + ctx.flops_moe / ctx.peak_e
    hbm_int_strict = (1 if tiled else n_splits) * w_step_strict + eff_state_bytes
    hbm_time_strict_s = (hbm_int_strict + act_bytes) / ctx.bw_hbm
    roof_core_strict_s = (
        max(hbm_time_strict_s, t_matmul_strict + comm_s)
        if is_fused_multi
        else (max(t_matmul_strict, hbm_time_strict_s) + comm_s)
    )
    repass_spill_strict_us = (
        (act_bytes + (n_splits - 1) * w_step_strict) / ctx.bw_hbm * 1e6
        if base_ai >= ctx.peak_w / ctx.bw_hbm and n_splits > 1 and not tiled
        else 0.0
    )
    step_floor_strict_us = (
        (roof_core_strict_s + pipeline_fill_drain_s) * 1e6
        + repass_spill_strict_us
        + n_ops * workload.op_overhead_us
    )
    return step_ach_upper_us, w_step_strict, step_floor_strict_us


def _evaluate_single_scope(
    scope: FusionScope,
    ctx: _ScopeContext,
    profile: ModelProfile,
    workload: WorkloadSpec,
    scope_eval_cfg: tuple[float, bool],
) -> tuple[ScopeEstimate, float, float]:
    """Evaluate one FusionScope against VMEM, HBM, matmul, and collective bounds."""
    sub_pure_us, bf16_attention_upper = scope_eval_cfg
    layers_local, l_moe = ctx.layers_local, ctx.l_moe
    token_residual_bytes, ep_dp = ctx.token_residual_bytes, ctx.ep_dp
    num_ar = ctx.num_ar
    lat_coll = ctx.lat_coll
    flops_dense, flops_moe = ctx.flops_dense, ctx.flops_moe
    peak_w, peak_e, bw_hbm = ctx.peak_w, ctx.peak_e, ctx.bw_hbm
    kv_cache_read_bytes, recurrent_state_rw_bytes, first_slot_bytes = (
        ctx.kv_cache_read_bytes,
        ctx.recurrent_state_rw_bytes,
        ctx.first_slot_bytes,
    )

    (
        static_vmem_bytes,
        act_bytes,
        n_ops,
        t_max,
        n_splits,
        feasible,
        reason,
        sel_nbd,
        sel_nbe,
    ) = _scope_vmem_and_ops(scope, ctx, profile, workload)
    w_step, phi, extra_act, tile_tokens, grouped_moe = _resolve_scope_moe_and_acts(
        scope, ctx, profile, workload, n_splits
    )
    tiled = scope in ("whole_step", "layer_group") and n_splits > 1
    act_bytes += extra_act

    is_fused_multi = scope in ("whole_step", "layer_group")
    overlapped_comm = is_fused_multi and n_splits == 1
    sync_stage_s, comm_s, xla_ar_factor = _compute_scope_collective_times(
        scope, ctx, n_splits
    )
    eff_state_bytes = kv_cache_read_bytes + recurrent_state_rw_bytes
    pipeline_fill_drain_s = (n_ops - 1) * (
        first_slot_bytes / bw_hbm + _BOUNDARY_SYNC_HOP_FACTOR * lat_coll
    ) + sync_stage_s
    streams = max(1, profile.residual_streams)
    if tiled:
        tile_residual_bytes = streams * tile_tokens * profile.hidden_size * 2.0
        pipeline_fill_drain_s += (
            (n_splits - 1)
            * 2
            * layers_local
            * (tile_residual_bytes / bw_hbm + _BOUNDARY_SYNC_HOP_FACTOR * lat_coll)
        )
    t_matmul, t_matmul_upper = _compute_scope_matmuls(
        ctx, workload, phi, bf16_attention_upper=bf16_attention_upper
    )
    act_saved, hbm_int = (
        4.0 * streams * layers_local * token_residual_bytes - act_bytes,
        (1 if tiled else n_splits) * w_step + eff_state_bytes,
    )
    hbm_time_s, d_ops = (hbm_int + act_bytes) / bw_hbm, 2 * layers_local - n_ops
    roof_core_s = (
        max(hbm_time_s, t_matmul + comm_s)
        if is_fused_multi
        else (max(t_matmul, hbm_time_s) + comm_s)
    )
    roof_us = (roof_core_s + pipeline_fill_drain_s) * 1e6
    base_ai = (flops_dense + flops_moe) / max(
        1.0, w_step + kv_cache_read_bytes + recurrent_state_rw_bytes
    )
    repass_spill_us = (
        (act_bytes + (n_splits - 1) * w_step) / bw_hbm * 1e6
        if base_ai >= peak_w / bw_hbm and n_splits > 1 and not tiled
        else 0.0
    )
    pure_us = roof_us + repass_spill_us
    new_sub_pure_us, step_us = (
        (pure_us if scope == "per_op" else sub_pure_us),
        pure_us + n_ops * workload.op_overhead_us,
    )

    eff = workload.efficiency
    eta_moe = eff.eta_moe_grouped if grouped_moe and l_moe > 0 else eff.eta_dense
    t_comp_ach = (flops_dense / peak_w) / max(1e-3, eff.eta_dense) + (
        (phi * flops_moe) / peak_e
    ) / max(1e-3, eta_moe)
    t_hbm_ach, t_comm_ach = (
        hbm_time_s / max(1e-3, eff.eta_hbm),
        comm_s / max(1e-3, eff.eta_ici),
    )
    sync_comm_extra_s = (
        sync_stage_s * (1.0 / max(1e-3, eff.eta_ici) - 1.0) if is_fused_multi else 0.0
    )
    ach_core_s = (
        (max(t_hbm_ach, t_comp_ach + t_comm_ach) + sync_comm_extra_s)
        if is_fused_multi
        else (max(t_comp_ach, t_hbm_ach) + t_comm_ach)
    )
    step_ach_us = (
        (ach_core_s + pipeline_fill_drain_s) * 1e6
        + repass_spill_us
        + n_ops * workload.op_overhead_us
    )

    ar_extra_s = (num_ar * n_splits * xla_ar_factor * ctx.ar_intra_hops) * max(
        0.0, ctx.lat_coll_upper - lat_coll
    )
    a2a_extra_s = (
        (2 * l_moe * n_splits * xla_ar_factor) if l_moe > 0 and ep_dp > 1 else 0
    ) * max(0.0, ctx.lat_coll_upper - lat_coll)
    roof_upper_core_s = (
        (max(hbm_time_s, t_matmul_upper + comm_s + ar_extra_s) + a2a_extra_s)
        if overlapped_comm
        else (max(t_matmul_upper, hbm_time_s) + comm_s + ar_extra_s + a2a_extra_s)
    )
    roof_upper_us = (roof_upper_core_s + pipeline_fill_drain_s) * 1e6
    step_upper_us = max(
        step_us,
        roof_upper_us + repass_spill_us + n_ops * ctx.launch_gap_upper_us,
    )
    step_ach_upper_us, w_step_strict, step_floor_strict_us = (
        _compute_ach_upper_and_strict(
            ctx,
            profile,
            workload,
            (
                tiled,
                is_fused_multi,
                overlapped_comm,
                bf16_attention_upper,
                n_splits,
                n_ops,
            ),
            (
                phi,
                eta_moe,
                t_comp_ach,
                t_hbm_ach,
                comm_s,
                ar_extra_s,
                a2a_extra_s,
                sync_comm_extra_s,
                pipeline_fill_drain_s,
                repass_spill_us,
                step_ach_us,
                step_upper_us,
                act_bytes,
                eff_state_bytes,
                base_ai,
            ),
        )
    )
    support_status, blockers = evaluate_support(
        scope, profile, workload, ctx.chip_spec, ctx
    )

    d_bytes = bw_hbm * max(0, d_ops) * workload.op_overhead_us * 1e-6
    break_even = (
        round(max(0.0, (pure_us - new_sub_pure_us) / d_ops), 3) if d_ops > 0 else 0.0
    )
    w_crit = (
        round(((act_saved + d_bytes) / (n_splits - 1)) / 1e9, 4)
        if n_splits > 1
        else None
    )
    est = ScopeEstimate(
        scope=scope,
        feasible=feasible,
        reason=reason,
        ops_per_step=n_ops,
        vmem_static_mib=round(static_vmem_bytes / (1 << 20), 3),
        t_vmem_max=t_max,
        n_splits=n_splits,
        w_step_gb=round(w_step / 1e9, 4),
        w_crit_gb=w_crit,
        act_hbm_mb=round(act_bytes / 1e6, 3),
        moe_flop_factor=round(phi, 3),
        roof_floor_us=round(roof_us, 2),
        step_bound_us=round(step_us, 2),
        selected_nbd=sel_nbd,
        selected_nbe=sel_nbe,
        step_achievable_us=round(step_ach_us, 2),
        break_even_op_overhead_us=break_even,
        step_upper_us=round(step_upper_us, 2),
        compute_us=round(t_matmul * 1e6, 2),
        hbm_us=round(hbm_time_s * 1e6, 2),
        comm_us=round(comm_s * 1e6, 2),
        fill_drain_us=round(pipeline_fill_drain_s * 1e6, 2),
        repass_spill_us=round(repass_spill_us, 2),
        launch_us=round(n_ops * workload.op_overhead_us, 2),
        step_achievable_upper_us=round(step_ach_upper_us, 2),
        support=support_status,
        blockers=blockers,
        w_step_strict_gb=round(w_step_strict / 1e9, 5),
        step_floor_strict_us=round(step_floor_strict_us, 2),
    )
    return est, step_us, new_sub_pure_us


def _evaluate_scopes(
    ctx: _ScopeContext,
    profile: ModelProfile,
    workload: WorkloadSpec,
    *,
    bf16_attention_upper: bool = True,
) -> tuple[list[ScopeEstimate], dict[FusionScope, float]]:
    """Evaluate every FusionScope against VMEM, HBM, and matmul bounds."""
    scopes_order: tuple[FusionScope, ...] = (
        "per_op",
        "single_layer",
        "layer_group",
        "whole_step",
    )
    estimates: list[ScopeEstimate] = []
    raw_step_map: dict[FusionScope, float] = {}
    sub_pure_us = 0.0
    for scope in scopes_order:
        est, step_us, sub_pure_us = _evaluate_single_scope(
            scope,
            ctx,
            profile,
            workload,
            (sub_pure_us, bf16_attention_upper),
        )
        estimates.append(est)
        raw_step_map[scope] = step_us
    return estimates, raw_step_map


@dataclass(frozen=True)
class _ScopeContext:
    """Per-core weight, state, FLOP, collective, and VMEM terms for one workload."""

    vmem_budget: int
    single_layer_dense_staging_bytes: int
    single_layer_expert_staging_bytes: int
    first_slot_bytes: float
    gqa_scratch: int
    gdn_scratch: int
    ffn_scratch: int
    layers_local: int
    num_segments: int
    num_layer_groups: int
    l_moe: int
    l_attn: int
    l_lin: int
    active_tokens: int
    eff_ctx: int
    batch_local: int
    verify_tokens: int
    nh_local: int
    q_dim: int
    base_dense_params: float
    per_token_fused_bytes: int
    token_residual_bytes: float
    w_static: float
    exp_bytes: float
    experts_local: int
    ep_dp: int
    eff_tp: int
    row_floor: int
    num_ar: int
    ar_hops: int
    ar_intra_hops: int
    ar_inter_latency_s: float
    dcn_bytes_per_s: float
    ar_vol: float
    ar_transfer_time_s: float
    ar_base_lat_s: float
    egress_bw: float
    lat_coll: float
    lat_coll_upper: float
    launch_gap_upper_us: float
    pp_stage_time: float
    pcp_ring_time: float
    a2a_time: float
    flops_dense: float
    flops_kv: float
    flops_moe: float
    peak_w: float
    peak_e: float
    bf16_peak: float
    kv_cache_read_bytes: float
    recurrent_state_rw_bytes: float
    bw_hbm: float
    hbm_used: float
    no_whole_step: bool
    plan_error: str
    chip_spec: ChipSpec


def _vmem_budget_bytes(workload: WorkloadSpec, chip_spec: ChipSpec) -> int:
    """Return usable VMEM bytes: user budget, Tiles budget, or chip default."""
    vmem_usable = chip_spec.vmem_usable_bytes
    tiles = workload.tiles
    rp = workload.ring_plan or default_double_buffer_ring_plan(workload.dma_buffers)
    base_vmem_bytes = (
        int(workload.vmem_budget_mib * (1 << 20))
        if workload.vmem_budget_mib is not None
        else (tiles.vmem_mib << 20 if tiles else vmem_usable)
    )
    vmem_raw = (
        base_vmem_bytes - rp.compiler_spill_reserve_bytes * workload.vmem_fit_headroom
    )
    return min(vmem_usable, max(0, vmem_raw))


def _first_slot_bytes(workload: WorkloadSpec, layer_dense_weight_bytes: int) -> float:
    """Return first weight DMA slot size in bytes for pipeline fill/drain."""
    tiles = workload.tiles
    if tiles is not None:
        dense_slot_bytes = tiles.bank
    else:
        rp = workload.ring_plan or default_double_buffer_ring_plan(workload.dma_buffers)
        dense_slot_bytes = rp.dense_slot_bytes
    return float(min(dense_slot_bytes, max(1 << 18, layer_dense_weight_bytes // 4)))


@dataclass(frozen=True)
class _LayerGeometry:
    """Request and layer geometry terms for layer weight accumulation."""

    eff_tp: int
    kv_split: int
    l_attn: int
    l_moe: int
    l_lin: int
    layers_local: int
    active_tokens: int
    chunk_tokens: int
    batch_local: int
    eff_ctx: int


@dataclass(frozen=True)
class _LayerWeightsAndFlops:
    """Layer static weights, expert bytes, state bytes, and FLOP terms."""

    w_static: float
    exp_bytes: float
    kv_cache_read_bytes: float
    recurrent_state_rw_bytes: float
    layer_dense_weight_bytes: int
    flops_kv: float
    flops_dense: float
    flops_moe: float
    base_dense_params: float
    nh_local: int
    q_dim: int
    experts_local: int
    ep_dp: int
    kv_bytes_per_tok: float
    gqa_scratch: int
    gdn_scratch: int
    ffn_scratch: int


def _compute_layer_weights_and_flops(
    profile: ModelProfile,
    workload: WorkloadSpec,
    geom: _LayerGeometry,
    dtypes_bw: tuple[float, float, float, float],
) -> _LayerWeightsAndFlops:
    """Compute per-layer static weights, cache state bytes, and FLOP counts."""
    bw_w, bw_e, bw_kv, bw_st = dtypes_bw
    hidden = profile.hidden_size
    head_dim = profile.head_dim
    kv_lora = profile.kv_lora_rank
    q_lora = profile.q_lora_rank
    qk_rope = profile.qk_rope_head_dim
    eff_tp = geom.eff_tp
    kv_split = geom.kv_split
    l_attn = geom.l_attn
    l_moe = geom.l_moe
    l_lin = geom.l_lin
    layers_local = geom.layers_local
    active_tokens = geom.active_tokens
    chunk_tokens = geom.chunk_tokens
    batch_local = geom.batch_local
    eff_ctx = geom.eff_ctx
    ep = workload.ep
    num_experts = profile.num_experts
    top_k = profile.top_k
    qk_nope = profile.qk_nope_head_dim or head_dim
    kd, vd = profile.linear_key_dim, profile.linear_value_dim
    nh_local, kvh_local = (
        max(1, ceil(profile.num_attention_heads / eff_tp)),
        max(1, ceil(profile.num_kv_heads / max(1, eff_tp // kv_split))),
    )
    q_dim, q_tot = nh_local * head_dim, nh_local * (qk_nope + qk_rope)
    mla_proj_params = (
        q_lora * (hidden + q_tot) if q_lora else hidden * q_tot
    ) + hidden * (kv_lora + qk_rope + q_dim)
    attn_gate_params = (
        hidden * q_dim if profile.has_attn_output_gate and not kv_lora else 0
    )
    attn_params = (
        mla_proj_params + kv_lora * nh_local * (qk_nope + head_dim)
        if kv_lora
        else hidden * 2 * (q_dim + kvh_local * head_dim) + attn_gate_params
    )
    kh_local = max(1, ceil(profile.linear_key_heads / eff_tp)) if l_lin > 0 else 0
    vh_local = max(1, ceil(profile.linear_value_heads / eff_tp)) if l_lin > 0 else 0
    z_dim, qkv_lin, ep_tp = (
        vh_local * vd,
        2 * kh_local * kd + vh_local * vd,
        gcd(ep, eff_tp),
    )
    linear_attn_params, experts_local = (
        hidden * (qkv_lin + 2 * z_dim) * (l_lin > 0),
        (num_experts // ep) * (l_moe > 0),
    )
    dense_ffn_params, shared_expert_params, ep_dp = (
        3 * hidden * max(1, ceil(profile.intermediate_size / eff_tp)),
        3 * hidden * ceil(profile.shared_expert_intermediate_size / eff_tp),
        ep // ep_tp,
    )
    i_exp_local = ceil(profile.moe_intermediate_size / (eff_tp // ep_tp)) * (l_moe > 0)
    routed_expert_params = (
        3 * (profile.routed_expert_hidden_size or hidden) * i_exp_local
    )
    base_dense_params = (
        l_attn * attn_params
        + l_lin * linear_attn_params
        + (layers_local - l_moe) * dense_ffn_params
        + l_moe * shared_expert_params
    )
    w_static, exp_bytes = (
        base_dense_params * bw_w + l_moe * hidden * num_experts * 2.0,
        routed_expert_params * bw_e,
    )

    kv_bytes_per_tok = (
        ((kv_lora + qk_rope) if kv_lora else 2.0 * kvh_local * head_dim)
        * bw_kv
        * profile.kv_compress_factor
        / kv_split
    )
    kv_cache_read_bytes = l_attn * batch_local * eff_ctx * kv_bytes_per_tok
    recurrent_state_rw_bytes = (
        2.0
        * l_lin
        * batch_local
        * (z_dim * kd + qkv_lin * profile.linear_conv_kernel)
        * bw_st
    )
    layer_dense_weight_bytes = int(
        (
            max(attn_params, linear_attn_params)
            + (shared_expert_params if l_moe > 0 else dense_ffn_params)
        )
        * bw_w
    )
    t_prefill_part = {"prefill": active_tokens, "mixed": chunk_tokens}.get(
        workload.phase, 0
    )
    d_mla = (active_tokens - t_prefill_part) * (
        2 * kv_lora + qk_rope
    ) + t_prefill_part * (qk_nope + qk_rope + head_dim)
    # Causal prefill attention: each of T new tokens attends to prefix + (1..T),
    # giving mean context prefix + (T + 1) / 2 instead of rectangular eff_ctx.
    prefix_ctx = max(0, eff_ctx - t_prefill_part)
    attn_ctx = (
        prefix_ctx + (t_prefill_part + 1) / 2.0
        if workload.phase == "prefill" and t_prefill_part > 1
        else float(eff_ctx)
    ) * profile.kv_compress_factor
    flops_kv = (
        2.0
        * l_attn
        * attn_ctx
        * (nh_local * d_mla if kv_lora else 2.0 * active_tokens * q_dim)
    )
    flops_dense = (
        2.0 * active_tokens * (base_dense_params + l_moe * hidden * num_experts)
        + flops_kv
    )
    flops_moe = (
        2.0
        * active_tokens
        * ep_dp
        * l_moe
        * top_k
        * routed_expert_params
        * experts_local
        / num_experts
        if experts_local > 0
        else 0.0
    )

    gqa_scratch = (
        2 * _ATTN_TILE_TOKENS * (_FP32_BYTES * max(head_dim, kv_lora))
        + 2 * _ATTN_TILE_TOKENS * _ATTN_TILE_TOKENS * _FP32_BYTES
        if l_attn > 0
        else 0
    )
    gdn_scratch = (
        _FP32_BYTES * min(vh_local, _GDN_STREAMED_HEADS_CAP) * kd * vd
        if l_lin > 0
        else 0
    )
    ffn_scratch = (
        (2 * _ATTN_TILE_TOKENS * i_exp_local)
        if l_moe > 0
        else (2 * _ATTN_TILE_TOKENS * _DENSE_FFN_TILE_CAP * _FP32_BYTES)
    )

    return _LayerWeightsAndFlops(
        w_static=w_static,
        exp_bytes=exp_bytes,
        kv_cache_read_bytes=kv_cache_read_bytes,
        recurrent_state_rw_bytes=recurrent_state_rw_bytes,
        layer_dense_weight_bytes=layer_dense_weight_bytes,
        flops_kv=flops_kv,
        flops_dense=flops_dense,
        flops_moe=flops_moe,
        base_dense_params=base_dense_params,
        nh_local=nh_local,
        q_dim=q_dim,
        experts_local=experts_local,
        ep_dp=ep_dp,
        kv_bytes_per_tok=kv_bytes_per_tok,
        gqa_scratch=gqa_scratch,
        gdn_scratch=gdn_scratch,
        ffn_scratch=ffn_scratch,
    )


def _extract_slot_banks(workload: WorkloadSpec) -> tuple[int, int]:
    """Return dense and expert weight slot byte sizes for pipeline buffers."""
    tiles = workload.tiles
    if tiles is not None:
        return tiles.bank, tiles.ebank
    rp = workload.ring_plan
    if rp is not None:
        return rp.dense_slot_bytes, rp.expert_slot_bytes
    return _DENSE_SLOT_DEFAULT_BYTES, _EXPERT_SLOT_DEFAULT_BYTES


def _compute_ar_hops_and_latency(eff_tp: int) -> tuple[int, float]:
    """Return single-slice ICI all-reduce hops and zero inter-host latency."""
    if eff_tp <= 1:
        return 0, 0.0
    return round(math.log2(eff_tp)), 0.0


def _build_scope_context(
    profile: ModelProfile,
    workload: WorkloadSpec,
    chip_spec: ChipSpec,
    plan_error: str = "",
    meas: MeasurementsData | None = None,
) -> _ScopeContext:
    """Assemble per-core weight, state, FLOP, and VMEM scope context."""
    bw_w = _DTYPE_BYTES.get(profile.weight_dtype.lower(), 1.0)
    bw_e = _DTYPE_BYTES.get(profile.expert_dtype.lower(), 1.0)
    bw_kv = _DTYPE_BYTES.get(profile.kv_dtype.lower(), 1.0)
    bw_st = _DTYPE_BYTES.get(profile.state_dtype.lower(), 2.0)
    num_layers, hidden = profile.num_layers, profile.hidden_size
    tp, pp, dp = workload.tp, workload.pp, workload.dp
    eff_tp, kv_split, spec_tok = (
        max(tp, workload.dcp) if tp == 1 else tp,
        max(workload.kv_split, workload.dcp),
        workload.spec_tokens,
    )
    eff_pcp = workload.pcp if workload.phase in ("prefill", "mixed") else 1

    uncached_isl = max(1, round(workload.isl * (1.0 - workload.prefix_hit_rate)))
    batch_local, chunk_tokens, layers_local = (
        ceil(workload.batch / dp),
        ceil((workload.chunk_size or uncached_isl) / eff_pcp),
        ceil(num_layers / pp),
    )
    eff_ctx = workload.context or (
        workload.isl + (0 if workload.phase == "prefill" else workload.osl // 2)
    )
    verify_tokens = (
        max(1, spec_tok) + 1
        if workload.phase == "verify"
        else (spec_tok + 1 if spec_tok > 0 else 1)
    )
    prefill_tokens = (
        ceil(workload.chunk_size / eff_pcp)
        if workload.chunk_size
        else ceil(
            batch_local * uncached_isl / max(1, (workload.microbatches or 1) * eff_pcp)
        )
    )
    decode_tokens = batch_local * (1 if workload.phase == "decode" else verify_tokens)
    active_tokens = (
        prefill_tokens
        if workload.phase == "prefill"
        else decode_tokens + chunk_tokens * (workload.phase == "mixed")
    )
    l_attn = min(
        layers_local, ceil(profile.num_attention_layers * layers_local / num_layers)
    )
    l_moe = min(layers_local, ceil(profile.num_moe_layers * layers_local / num_layers))
    l_lin = min(
        layers_local - l_attn,
        ceil(profile.num_linear_layers * layers_local / num_layers),
    )
    token_residual_bytes = active_tokens * hidden * 2.0

    geom = _LayerGeometry(
        eff_tp=eff_tp,
        kv_split=kv_split,
        l_attn=l_attn,
        l_moe=l_moe,
        l_lin=l_lin,
        layers_local=layers_local,
        active_tokens=active_tokens,
        chunk_tokens=chunk_tokens,
        batch_local=batch_local,
        eff_ctx=eff_ctx,
    )
    wf = _compute_layer_weights_and_flops(
        profile, workload, geom, (bw_w, bw_e, bw_kv, bw_st)
    )
    w_static = wf.w_static
    exp_bytes = wf.exp_bytes
    kv_cache_read_bytes = wf.kv_cache_read_bytes
    recurrent_state_rw_bytes = wf.recurrent_state_rw_bytes
    layer_dense_weight_bytes = wf.layer_dense_weight_bytes
    flops_kv = wf.flops_kv
    flops_dense = wf.flops_dense
    flops_moe = wf.flops_moe
    base_dense_params = wf.base_dense_params
    nh_local = wf.nh_local
    q_dim = wf.q_dim
    experts_local = wf.experts_local
    ep_dp = wf.ep_dp
    kv_bytes_per_tok = wf.kv_bytes_per_tok
    gqa_scratch = wf.gqa_scratch
    gdn_scratch = wf.gdn_scratch
    ffn_scratch = wf.ffn_scratch
    vmem_budget = _vmem_budget_bytes(workload, chip_spec)
    num_segments = (
        ceil(layers_local / workload.segment_layers)
        if 0 < workload.segment_layers < layers_local
        else 1
    )
    buf_slots = workload.dma_buffers
    first_slot_bytes = _first_slot_bytes(workload, layer_dense_weight_bytes)
    dcn_bw_gbps = (
        meas.dcn_bw_gbps
        if meas is not None and meas.dcn_bw_gbps is not None
        else chip_spec.dcn_bw_gbps
    )
    dcn_bytes_per_s = (dcn_bw_gbps / 8.0) * 1e9
    ar_transfer_time_s, ar_base_lat_s, ar_hops, ar_vol = (
        compute_all_reduce_transfer_cost(
            token_residual_bytes,
            eff_tp,
            chip_spec,
            dcn_bw_bytes_per_s=dcn_bytes_per_s,
        )
    )
    ar_intra_hops, ar_inter_latency_s = _compute_ar_hops_and_latency(eff_tp)
    dense_bank, expert_bank = _extract_slot_banks(workload)
    pcp_ring_time = (
        (eff_pcp - 1)
        * l_attn
        * chunk_tokens
        * (kv_bytes_per_tok * kv_split)
        / chip_spec.ici_link_bw_bytes_per_s
        if eff_pcp > 1 and l_attn > 0
        else 0.0
    )
    return _ScopeContext(
        vmem_budget=vmem_budget,
        single_layer_dense_staging_bytes=min(
            buf_slots * dense_bank, buf_slots * layer_dense_weight_bytes
        ),
        single_layer_expert_staging_bytes=min(
            buf_slots * expert_bank, int(buf_slots * exp_bytes)
        ),
        first_slot_bytes=first_slot_bytes,
        gqa_scratch=gqa_scratch,
        gdn_scratch=gdn_scratch,
        ffn_scratch=ffn_scratch,
        layers_local=layers_local,
        num_segments=num_segments,
        num_layer_groups=(
            ceil(len(workload.layer_groups) * layers_local / num_layers)
            if workload.layer_groups
            else 0
        ),
        l_moe=l_moe,
        l_attn=l_attn,
        l_lin=l_lin,
        active_tokens=active_tokens,
        eff_ctx=eff_ctx,
        batch_local=batch_local,
        verify_tokens=verify_tokens,
        nh_local=nh_local,
        q_dim=q_dim,
        base_dense_params=base_dense_params,
        per_token_fused_bytes=compute_per_token_vmem_bytes(profile, workload)[
            "total_per_token_bytes"
        ],
        token_residual_bytes=token_residual_bytes,
        w_static=w_static,
        exp_bytes=exp_bytes,
        experts_local=experts_local,
        ep_dp=ep_dp,
        eff_tp=eff_tp,
        row_floor=chip_spec.lhs_row_floor(
            profile.expert_dtype if l_moe > 0 else profile.weight_dtype
        ),
        num_ar=profile.ar_slots_per_layer * layers_local,
        ar_hops=ar_hops,
        ar_intra_hops=ar_intra_hops,
        ar_inter_latency_s=ar_inter_latency_s,
        dcn_bytes_per_s=dcn_bytes_per_s,
        ar_vol=ar_vol,
        ar_transfer_time_s=ar_transfer_time_s,
        ar_base_lat_s=ar_base_lat_s,
        egress_bw=chip_spec.ici_egress_bw_bytes_per_s,
        lat_coll=chip_spec.collective_latency_s,
        lat_coll_upper=chip_spec.collective_latency_us_bracket[1] * 1e-6,
        launch_gap_upper_us=max(workload.op_overhead_us, _DEFAULT_LAUNCH_GAP_UPPER_US),
        pp_stage_time=(
            (
                token_residual_bytes / chip_spec.ici_link_bw_bytes_per_s
                + chip_spec.collective_latency_s
            )
            if pp > 1
            else 0.0
        ),
        pcp_ring_time=pcp_ring_time,
        a2a_time=(
            2.0
            * (ep_dp - 1)
            / ep_dp
            * min(max(1, profile.top_k), ep_dp)
            * token_residual_bytes
            / chip_spec.ici_egress_bw_bytes_per_s
            if ep_dp > 1
            else 0.0
        ),
        flops_dense=flops_dense,
        flops_kv=flops_kv,
        flops_moe=flops_moe,
        peak_w=chip_spec.peak_flops(profile.weight_dtype),
        peak_e=chip_spec.peak_flops(profile.expert_dtype),
        bf16_peak=chip_spec.bf16_flops_per_s,
        kv_cache_read_bytes=kv_cache_read_bytes,
        recurrent_state_rw_bytes=recurrent_state_rw_bytes,
        bw_hbm=chip_spec.hbm_bw_bytes_per_s,
        hbm_used=(
            w_static
            + l_moe * experts_local * exp_bytes
            + kv_cache_read_bytes
            + 0.5 * recurrent_state_rw_bytes
        ),
        no_whole_step=(
            pp > 1
            or num_segments > 1
            or len(workload.layer_groups) > 1
            or num_layers == 1
            or bool(plan_error)
        ),
        plan_error=plan_error,
        chip_spec=chip_spec,
    )


def _is_calibrated_trace(meas: MeasurementsData | None) -> bool:
    """Return True if meas represents a calibrated empirical trace."""
    return meas is not None and meas.source_kind != "architectural_prior"


def _resolve_launch_gap_prior(
    workload: WorkloadSpec,
    meas: MeasurementsData | None,
) -> tuple[float, float, float, ProvenanceTag, str]:
    """Resolve launch gap bounds and provenance tag across dispatch regimes."""
    if workload.dispatch_regime == "zero_floor":
        return (
            0.0,
            0.0,
            0.0,
            "DERIVED",
            "zero_floor compiler-proof physics regime (0.0 us/boundary)",
        )
    if workload.dispatch_regime == "unfused_eager":
        lg_val = (
            workload.op_overhead_us
            if workload.op_overhead_us > 0.0
            else _UNFUSED_EAGER_GAP_US
        )
        return (
            lg_val,
            lg_val,
            max(lg_val, _DEFAULT_LAUNCH_GAP_UPPER_US),
            "ASSUMED",
            "unfused_eager exposed dispatch sensitivity regime",
        )
    if meas is not None and meas.launch_gap_p50_us is not None:
        lg_lo = meas.launch_gap_p50_us
        lg_hi = max(lg_lo, meas.launch_gap_p90_us or lg_lo)
        is_cal = _is_calibrated_trace(meas)
        tag: ProvenanceTag = "MEASURED" if is_cal else "ASSUMED"
        n_sfx = f" (n={meas.launch_gap_n})" if is_cal else ""
        return (lg_lo, lg_lo, lg_hi, tag, f"{meas.source_desc}{n_sfx}")
    if workload.dispatch_regime == "compiled_xla":
        return (
            _COMPILED_XLA_GAP_P50_US,
            _COMPILED_XLA_GAP_P50_US,
            _COMPILED_XLA_GAP_P90_US,
            "ASSUMED",
            (
                "compiled-XLA sublayer boundary prior "
                "(uncalibrated; run advisor calibrate)"
            ),
        )
    if workload.op_overhead_us > 0.0:
        lg_val = workload.op_overhead_us
        return (
            lg_val,
            lg_val,
            max(lg_val, _DEFAULT_LAUNCH_GAP_UPPER_US),
            "ASSUMED",
            "caller WorkloadSpec.op_overhead_us override",
        )
    return (
        _DEFAULT_LAUNCH_GAP_VAL_US,
        _DEFAULT_LAUNCH_GAP_FLOOR_US,
        _DEFAULT_LAUNCH_GAP_UPPER_US,
        "ASSUMED",
        "uncalibrated TPU host/device launch gap prior [1.0, 8.0] us",
    )


def _resolve_calibrated_priors(
    workload: WorkloadSpec,
    chip_spec: ChipSpec,
    ctx: _ScopeContext,
    meas: MeasurementsData | None,
) -> tuple[
    tuple[float, float, float, ProvenanceTag, str],
    tuple[float, float, float, ProvenanceTag, str],
    tuple[float, list[float], ProvenanceTag, str],
]:
    """Return calibrated or assumed priors for launch gap, collectives, and coverage."""
    lg_info = _resolve_launch_gap_prior(workload, meas)

    if meas is not None and meas.collective_hop_us is not None:
        c_val = round(meas.collective_hop_us, 3)
        is_cal = _is_calibrated_trace(meas)
        c_tag: ProvenanceTag = "MEASURED" if is_cal else "ASSUMED"
        n_sfx = f" (n={meas.collective_n})" if is_cal else ""
        coll_info = (
            c_val,
            c_val,
            c_val,
            c_tag,
            f"{meas.source_desc}{n_sfx}",
        )
    else:
        c_lo, c_hi = chip_spec.collective_latency_us_bracket
        c_src = (chip_spec.sources or {}).get(
            "collective_latency_us_bracket", "uncalibrated ICI hop latency bracket"
        )
        coll_info = (
            round(chip_spec.collective_latency_s * 1e6, 3),
            c_lo,
            c_hi,
            "ASSUMED",
            c_src,
        )

    if meas is not None and meas.step_coverage is not None:
        cov_val = round(meas.step_coverage, 4)
        is_cal = _is_calibrated_trace(meas)
        cov_tag: ProvenanceTag = "MEASURED" if is_cal else "ASSUMED"
        cov_info = (
            cov_val,
            [cov_val, cov_val],
            cov_tag,
            f"{meas.source_desc} step_coverage",
        )
    elif workload.phase == "mixed":
        dec_tok = ctx.batch_local * max(1, ctx.verify_tokens)
        cov_val = round(min(1.0, dec_tok / max(1, ctx.active_tokens)), 4)
        cov_info = (
            cov_val,
            [0.0, 1.0],
            "ASSUMED",
            "decode token share on mixed step (settle via --measurements)",
        )
    else:
        cov_info = (
            1.0,
            [0.0, 1.0],
            "ASSUMED",
            "single-phase step coverage prior (settle via --measurements)",
        )
    return lg_info, coll_info, cov_info
