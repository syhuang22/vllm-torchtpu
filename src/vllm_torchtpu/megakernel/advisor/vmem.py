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
"""VMEM floorplan, transfer cost, and per-token memory modeling for TPU advisor."""

from __future__ import annotations

from math import ceil, gcd, log2
from typing import TYPE_CHECKING

from ..core import errors
from .specs import FusionScope, RingPlan, WorkloadSpec, default_double_buffer_ring_plan
from .support import CORE_LIMITS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..advisor.models import ModelProfile
    from ..core.exec_spec import Tiles
    from .chips import ChipSpec
    from .roofline import _ScopeContext

_MIN_VMEM_MIB = 2
_DEFAULT_USABLE_VMEM_MIB = CORE_LIMITS.vmem_mib
_DOUBLE_BUFFER_SLOTS = 2
_GROUPED_MOE_MODES: frozenset[str] = frozenset({"grouped_on_chip"})
_VALID_DMA_BUFFERS = (2, 3)
_WIDE_DENSE_RING_SLOTS = CORE_LIMITS.nbd
_WIDE_EXPERT_RING_SLOTS = CORE_LIMITS.nbe
_DENSE_SLOT_DEFAULT_BYTES = CORE_LIMITS.bank
_EXPERT_SLOT_DEFAULT_BYTES = CORE_LIMITS.ebank
_ARENA_ALIGN_RESERVE_BYTES = 2 << 20
_ATTN_TILE_TOKENS = 128
_GDN_STREAMED_HEADS_CAP = 8
_DENSE_FFN_TILE_CAP = 1024
_FP32_BYTES = 4
_CHIPS_PER_HOST: int = 8
_DEFAULT_DCN_HOP_LATENCY_S: float = 5.0e-6
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


def fit_vmem_budget_bytes(
    tiles: Tiles | None = None,
    ring_plan: RingPlan | None = None,
    chip_spec: ChipSpec | None = None,
) -> int:
    """Return usable scratchpad budget bytes after subtracting RingPlan reserve.

    Args:
        tiles: Optional execution tiles specifying vmem_mib.
        ring_plan: Optional RingPlan reserve to deduct.
        chip_spec: Optional target ChipSpec to derive usable VMEM limit.

    Returns:
        Usable VMEM budget in bytes.
    """
    rp = ring_plan or default_double_buffer_ring_plan()
    if tiles is not None and tiles.vmem_mib > 0:
        vmem_mib = tiles.vmem_mib
    elif chip_spec is not None:
        vmem_mib = chip_spec.vmem_usable_bytes >> 20
    else:
        vmem_mib = _DEFAULT_USABLE_VMEM_MIB
    return max(0, (vmem_mib << 20) - rp.compiler_spill_reserve_bytes)


def compute_all_reduce_transfer_cost(
    message_bytes: float,
    effective_tp: int,
    chip_spec: ChipSpec,
    dcn_bw_bytes_per_s: float | None = None,
    collective_latency_s: float | None = None,
) -> tuple[float, float, int, float]:
    """Computes transfer time, latency, hop count, and volume for All-Reduce.

    Models single-slice optical ICI ring/torus reduce-scatter + all-gather
    across `effective_tp` ranks (`effective_tp > 1`).

    Args:
        message_bytes: Payload size transferred in the collective, in bytes.
        effective_tp: Effective tensor-parallel rank count (e.g. tp // pp).
        chip_spec: Hardware chip specification providing ICI bandwidth and base latency.
        dcn_bw_bytes_per_s: Unused legacy argument kept for call-site compatibility.
        collective_latency_s: Optional collective base latency in seconds.

    Returns:
        A tuple of (transfer_time_s, base_latency_s, hop_count, nominal_volume_bytes).
    """
    if (
        effective_tp <= 1
        or message_bytes <= 0.0
        or (dcn_bw_bytes_per_s is not None and dcn_bw_bytes_per_s < 0.0)
    ):
        return 0.0, 0.0, 0, 0.0

    nominal_volume_bytes = 2.0 * (effective_tp - 1) / effective_tp * message_bytes
    hop_count = round(log2(effective_tp))
    coll_lat = (
        chip_spec.collective_latency_s
        if collective_latency_s is None
        else collective_latency_s
    )
    transfer_time_s = nominal_volume_bytes / chip_spec.ici_egress_bw_bytes_per_s
    base_latency_s = hop_count * coll_lat
    return transfer_time_s, base_latency_s, hop_count, nominal_volume_bytes


def compute_per_token_vmem_bytes(
    profile: ModelProfile, workload: WorkloadSpec
) -> dict[str, int]:
    """Compute per-token live VMEM bytes from tensor lifetimes and TP/EP sharding."""
    hidden = profile.hidden_size
    tp = max(1, workload.tp, workload.dcp if workload.tp == 1 else workload.tp)
    ep = max(1, workload.ep)
    kv_split = max(1, workload.kv_split, workload.dcp)

    # 1. On-chip residual (F32 4*streams*H), norm (BF16 2*H), staging (6*H)
    streams = max(1, profile.residual_streams)
    residual_buffer_bytes = (4 * streams + 2) * hidden + (6 * hidden if tp > 1 else 0)

    # 2. Attention per_op per-token scratch (sharded across TP ranks)
    nh_local = max(1, ceil(profile.num_attention_heads / tp))
    if profile.kv_lora_rank > 0:
        attn_scratch_bytes = 2 * (
            profile.q_lora_rank + profile.kv_lora_rank + profile.qk_rope_head_dim
        ) + 6 * (nh_local * profile.head_dim)
    else:
        nkv_local = max(1, ceil(profile.num_kv_heads / max(1, tp // kv_split)))
        attn_scratch_bytes = 2 * (2 * nh_local + 2 * nkv_local) * profile.head_dim + (
            4 * nh_local * profile.head_dim
        )

    # 3. Linear attention (GDN / KDA) per-token scratch (sharded across TP ranks)
    if profile.num_linear_layers > 0 and profile.linear_value_heads > 0:
        kh_local = max(1, ceil(profile.linear_key_heads / tp))
        vh_local = max(1, ceil(profile.linear_value_heads / tp))
        linear_attn_scratch_bytes = 2 * (
            2 * kh_local * profile.linear_key_dim
            + 2 * vh_local * profile.linear_value_dim
        ) + 4 * (vh_local * profile.linear_value_dim)
    else:
        linear_attn_scratch_bytes = 0

    # 4. MoE / FFN per_op per-token scratch (sharded across TP and EP ranks)
    expert_hidden = profile.routed_expert_hidden_size or hidden
    if profile.num_moe_layers > 0 and profile.num_experts > 0:
        ep_tp = gcd(ep, tp)
        inter_local = ceil(profile.moe_intermediate_size / max(1, tp // ep_tp))
        shared_local = (
            ceil(profile.shared_expert_intermediate_size / tp)
            if profile.shared_expert_intermediate_size > 0
            else 0
        )
        router_scratch_bytes = 4 * profile.num_experts + 8 * profile.top_k
        is_grouped = workload.moe_mode in _GROUPED_MOE_MODES
        if is_grouped:
            # Streaming grouped MoE holds active expert FFN intermediates in fixed
            # tile scratch; at most _DOUBLE_BUFFER_SLOTS * ep streamed expert slices
            # (or 1 on single-rank EP=1) are live per token at once.
            eff_top_k = min(profile.top_k, _DOUBLE_BUFFER_SLOTS * ep) if ep > 1 else 1
            expert_tile_bytes = (
                8 * expert_hidden
                + ceil(eff_top_k * (6 * expert_hidden + 8 * inter_local) / ep)
                + 8 * shared_local
            )
        else:
            expert_tile_bytes = 16 * expert_hidden + 8 * (inter_local + shared_local)
        moe_scratch_bytes = (
            max(router_scratch_bytes + 2 * expert_hidden, expert_tile_bytes)
            if workload.vmem_alias
            else (router_scratch_bytes + expert_tile_bytes)
        )
    else:
        inter_local = ceil(profile.intermediate_size / tp)
        moe_scratch_bytes = 6 * hidden + 8 * inter_local

    per_op_scratch_bytes = (
        max(attn_scratch_bytes, linear_attn_scratch_bytes, moe_scratch_bytes)
        if workload.vmem_alias
        else (attn_scratch_bytes + linear_attn_scratch_bytes + moe_scratch_bytes)
    )
    total_per_token_bytes = residual_buffer_bytes + per_op_scratch_bytes
    return {
        "residual_buffer_bytes": residual_buffer_bytes,
        "attn_scratch_bytes": attn_scratch_bytes,
        "linear_attn_scratch_bytes": linear_attn_scratch_bytes,
        "moe_scratch_bytes": moe_scratch_bytes,
        "total_per_token_bytes": total_per_token_bytes,
        "b_spine": residual_buffer_bytes,
        "b_attn": attn_scratch_bytes,
        "b_lin": linear_attn_scratch_bytes,
        "b_moe": moe_scratch_bytes,
        "b_tok": total_per_token_bytes,
    }


def validate_advise_inputs(profile: ModelProfile, workload: WorkloadSpec) -> None:
    """Validate ModelProfile and WorkloadSpec field ranges and constraints."""
    num_layers = profile.num_layers
    tp, pp, dp, ep = workload.tp, workload.pp, workload.dp, workload.ep
    kv_split, dcp, pcp = workload.kv_split, workload.dcp, workload.pcp
    eff_tp = max(tp, dcp) if tp == 1 else tp
    pos_fields = (
        num_layers,
        profile.hidden_size,
        profile.num_attention_heads,
        profile.head_dim,
        workload.batch,
        tp,
        pp,
        dp,
        ep,
        kv_split,
        dcp,
        pcp,
        workload.isl,
    )
    nonneg_fields = (
        profile.num_attention_layers,
        profile.num_linear_layers,
        profile.num_moe_layers,
        workload.osl,
        workload.context,
        workload.spec_tokens,
        workload.chunk_size,
        profile.kv_lora_rank,
        workload.microbatches,
        workload.segment_layers,
        profile.q_lora_rank,
        profile.qk_rope_head_dim,
        profile.qk_nope_head_dim,
    )
    dtypes = (
        profile.weight_dtype,
        profile.expert_dtype,
        profile.kv_dtype,
        profile.state_dtype,
    )
    if min(*pos_fields) < 1 or workload.op_overhead_us < 0.0:
        msg = "Dimensions and parallelism must be >= 1"
        raise errors.MegakernelConfigError(msg)
    if workload.dma_buffers not in _VALID_DMA_BUFFERS:
        msg = f"dma_buffers must be 2 or 3, got {workload.dma_buffers!r}"
        raise errors.MegakernelConfigError(msg)
    if (
        workload.vmem_budget_mib is not None
        and workload.vmem_budget_mib < _MIN_VMEM_MIB
    ):
        msg = f"vmem_budget_mib must be >= {_MIN_VMEM_MIB}"
        raise errors.MegakernelConfigError(msg)
    parallel_ok = (
        min(*nonneg_fields) >= 0
        and (eff_tp * dp * pcp) % ep == 0
        and eff_tp % kv_split == 0
        and eff_tp % dcp == 0
        and pp <= num_layers
    )
    expert_ok = (
        profile.top_k == 0
        if profile.num_experts == 0
        else (
            1 <= profile.top_k <= profile.num_experts and profile.num_experts % ep == 0
        )
    )
    tiles = workload.tiles
    tiles_ok = 0.0 <= workload.prefix_hit_rate < 1.0 and not (
        tiles and tiles.vmem_mib < _MIN_VMEM_MIB
    )
    layers_ok = (
        profile.num_attention_layers + profile.num_linear_layers <= num_layers
        and profile.num_moe_layers <= num_layers
        and (
            not workload.layer_groups
            or (
                sum(workload.layer_groups) == num_layers
                and min(workload.layer_groups) >= 1
            )
        )
    )
    if not (
        parallel_ok
        and expert_ok
        and tiles_ok
        and layers_ok
        and all(d.lower() in _DTYPE_BYTES for d in dtypes)
    ):
        msg = "Invalid parallelism, layers, or dtype"
        raise errors.MegakernelConfigError(msg)


def _mesh_divisibility_reasons(prof: ModelProfile, workload: WorkloadSpec) -> list[str]:
    """Return non-raising sharding divisibility violation codes."""
    reasons: list[str] = []
    tp, pp, dp, ep = workload.tp, workload.pp, workload.dp, workload.ep
    kv_split, dcp, pcp = workload.kv_split, workload.dcp, workload.pcp
    eff_tp = max(tp, dcp) if tp == 1 else tp
    if (eff_tp & (eff_tp - 1)) != 0:
        reasons.append("tp_not_power_of_two")
    if prof.num_attention_heads % eff_tp != 0:
        reasons.append("attn_heads_not_divisible_by_tp")
    kv_tp = max(1, eff_tp // max(1, kv_split))
    if eff_tp % kv_split != 0 or (
        prof.num_kv_heads % kv_tp != 0 and kv_tp % prof.num_kv_heads != 0
    ):
        reasons.append("kv_heads_not_divisible_by_tp")
    if eff_tp % dcp != 0:
        reasons.append("dcp_not_dividing_tp")
    if prof.num_experts > 0 and (
        prof.num_experts % ep != 0 or not (1 <= prof.top_k <= prof.num_experts)
    ):
        reasons.append("experts_not_divisible_by_ep")
    if (eff_tp * dp * pcp) % ep != 0:
        reasons.append("ep_exceeds_slice_ranks")
    if pp > prof.num_layers:
        reasons.append("pp_exceeds_layers")
    return reasons


def _compute_scope_ring_and_splits(
    scope: FusionScope,
    ctx: _ScopeContext,
    workload: WorkloadSpec,
    mul: int,
    scratch_op_bytes: int,
) -> tuple[int, float, int, int, int, bool, str, int, int]:
    """Compute static VMEM, activation bytes, and splits for fused scopes."""
    is_single_layer = scope == "single_layer"
    is_single_or_preset_group = is_single_layer or (
        scope == "layer_group" and bool(ctx.num_layer_groups)
    )
    vmem_budget = ctx.vmem_budget
    single_layer_dense_staging = ctx.single_layer_dense_staging_bytes
    single_layer_expert_staging = ctx.single_layer_expert_staging_bytes
    layers_local = ctx.layers_local
    preset_groups = ctx.num_layer_groups
    l_moe = ctx.l_moe
    l_attn = ctx.l_attn
    l_lin = ctx.l_lin
    active_tokens = ctx.active_tokens
    per_token_fused_bytes = ctx.per_token_fused_bytes
    token_residual_bytes = ctx.token_residual_bytes

    if is_single_or_preset_group:
        sel_nbd = workload.dma_buffers
        sel_nbe = workload.dma_buffers if l_moe > 0 else 0
        static_vmem_bytes = (
            single_layer_dense_staging
            + single_layer_expert_staging * (l_moe > 0)
            + scratch_op_bytes
        )
        min_static_bytes = static_vmem_bytes
    else:
        sel_nbd, sel_nbe, ring_bytes, min_ring_bytes = _select_adaptive_ring(
            ctx, workload, scratch_op_bytes + _ARENA_ALIGN_RESERVE_BYTES
        )
        static_vmem_bytes = ring_bytes + scratch_op_bytes + _ARENA_ALIGN_RESERVE_BYTES
        min_static_bytes = (
            min_ring_bytes + scratch_op_bytes + _ARENA_ALIGN_RESERVE_BYTES
        )
    act_bytes = 2.0 * (layers_local if is_single_layer else mul) * token_residual_bytes
    n_ops = (
        layers_local
        if is_single_layer
        else mul
        if scope == "layer_group" and preset_groups
        else (mul * (3 if ctx.eff_tp * workload.ep > 1 and l_attn + l_lin else 1))
    )
    alpha = per_token_fused_bytes
    t_max = (
        0
        if min_static_bytes + alpha > vmem_budget
        else max(1, (vmem_budget - min_static_bytes) // alpha)
    )
    n_splits = active_tokens if t_max == 0 else ceil(active_tokens / t_max)
    feasible = True
    reason = "ok"
    if t_max == 0:
        feasible, reason = False, "Weight buffers exceed VMEM"
    return (
        static_vmem_bytes,
        act_bytes,
        n_ops,
        t_max,
        n_splits,
        feasible,
        reason,
        sel_nbd,
        sel_nbe,
    )


def _select_adaptive_ring(
    ctx: _ScopeContext,
    workload: WorkloadSpec,
    fixed_non_ring_bytes: int,
) -> tuple[int, int, int, int]:
    """Select widest (nbd, nbe) fitting active_tokens in VMEM before splitting."""
    has_moe = ctx.l_moe > 0
    if workload.tiles is not None:
        nbd = workload.tiles.nbd
        nbe = workload.tiles.nbe if has_moe else 0
        ring_b = nbd * workload.tiles.bank + nbe * workload.tiles.ebank
        return nbd, nbe, ring_b, ring_b
    if workload.ring_plan is not None:
        rp = workload.ring_plan
        nbd = rp.dense_slots
        nbe = rp.expert_slots if has_moe else 0
        ring_b = nbd * rp.dense_slot_bytes + nbe * rp.expert_slot_bytes
        return nbd, nbe, ring_b, ring_b
    min_nbd = workload.dma_buffers
    extra = min_nbd - _DOUBLE_BUFFER_SLOTS
    max_nbd = _WIDE_DENSE_RING_SLOTS + extra
    min_nbe = min_nbd if has_moe else 0
    max_nbe = (_WIDE_EXPERT_RING_SLOTS + extra) if has_moe else 0
    min_ring_b = (
        min_nbd * _DENSE_SLOT_DEFAULT_BYTES + min_nbe * _EXPERT_SLOT_DEFAULT_BYTES
    )
    avail = (
        ctx.vmem_budget
        - fixed_non_ring_bytes
        - ctx.per_token_fused_bytes * ctx.active_tokens
    )
    if avail < min_ring_b:
        return min_nbd, min_nbe, min_ring_b, min_ring_b
    best_key = (-1, -1, -1)
    best_choice = (min_nbd, min_nbe, min_ring_b)
    for nbd in range(min_nbd, max_nbd + 1):
        for nbe in range(min_nbe, max_nbe + 1):
            ring_b = nbd * _DENSE_SLOT_DEFAULT_BYTES + nbe * _EXPERT_SLOT_DEFAULT_BYTES
            if ring_b <= avail:
                bal = (
                    (nbd - min_nbd)
                    if not has_moe
                    else min(nbd - min_nbd, nbe - min_nbe)
                )
                cand_key = (ring_b, bal, nbd)
                if cand_key > best_key:
                    best_key = cand_key
                    best_choice = (nbd, nbe, ring_b)
    return best_choice[0], best_choice[1], best_choice[2], min_ring_b


def _scope_vmem_and_ops(
    scope: FusionScope,
    ctx: _ScopeContext,
    profile: ModelProfile,
    workload: WorkloadSpec,
) -> tuple[int, float, int, int, int, bool, str, int, int]:
    """Compute static VMEM, activation HBM bytes, ops, splits, and ring slots."""
    vmem_budget = ctx.vmem_budget
    single_layer_dense_staging = ctx.single_layer_dense_staging_bytes
    gqa_scratch = ctx.gqa_scratch
    gdn_scratch = ctx.gdn_scratch
    ffn_scratch = ctx.ffn_scratch
    layers_local = ctx.layers_local
    num_segments = ctx.num_segments
    active_tokens = ctx.active_tokens
    token_residual_bytes = ctx.token_residual_bytes
    no_whole_step = ctx.no_whole_step
    plan_error = ctx.plan_error

    feasible, reason = True, "ok"
    preset_groups = ctx.num_layer_groups
    if scope == "layer_group" and preset_groups == 0 and workload.pp == 1:
        if num_segments == 1:
            feasible = False
            reason = (
                "No layer grouping: needs a model preset, pp > 1, or segment_layers"
            )
    elif scope == "whole_step" and no_whole_step:
        feasible = False
        reason = (
            plan_error
            or "Bounded by layer groups, single layers, or unmodeled layer family"
        )

    if scope == "per_op":
        static_vmem_bytes = max(1 << 20, single_layer_dense_staging)
        alpha = _FP32_BYTES * profile.hidden_size
        n_splits = 1
        t_max = max(active_tokens, (vmem_budget - static_vmem_bytes) // alpha)
        n_ops = 2 * layers_local
        act_bytes = (
            4.0 * max(1, profile.residual_streams) * layers_local * token_residual_bytes
        )
        return (
            static_vmem_bytes,
            act_bytes,
            n_ops,
            t_max,
            n_splits,
            feasible,
            reason,
            0,
            0,
        )

    mul = (preset_groups or num_segments) if scope == "layer_group" else 1
    scratch_op_bytes = (
        max(gqa_scratch, gdn_scratch, ffn_scratch)
        if workload.vmem_alias
        else (gqa_scratch + gdn_scratch + ffn_scratch)
    )
    res = _compute_scope_ring_and_splits(
        scope,
        ctx,
        workload,
        mul,
        scratch_op_bytes,
    )
    if not res[5]:  # feasible
        feasible, reason = False, res[6]
    return (
        res[0],
        res[1] * max(1, profile.residual_streams),
        res[2],
        res[3],
        res[4],
        feasible,
        reason,
        res[7],
        res[8],
    )
