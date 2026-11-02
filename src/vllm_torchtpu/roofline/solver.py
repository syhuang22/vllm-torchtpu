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
"""Unconstrained single-pool mixed and xP/yD disaggregated TPU mesh solver."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from statistics import NormalDist
from typing import TYPE_CHECKING, Literal

from .chips import TOPOLOGIES, ChipSpec, SliceTopology
from .collectives import submesh_for_group
from .modules import Dtypes, Parallel, StepShape
from .step import StepFloor, _combine_modules, spec_cycle_floor, step_floor

_STD_NORM = NormalDist()

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .geometry import ModelGeometry

# vLLM v0.32.0 CacheConfig.gpu_memory_utilization default (https://github.com/vllm-project/vllm):
DEFAULT_GPU_MEMORY_UTILIZATION: float = 0.92
EXAMPLE_CONCURRENCIES: tuple[int, ...] = (1, 4, 8, 16, 32, 64, 128, 256)
_HEAD_DIM_64: int = 64
_MAX_CTX_TOKENS_FOR_SMALL_PAGE: int = 8192
_MIN_MOE_INTERMEDIATE_ALIGNMENT: int = 128
_BYTE_SCALE_SPARSE_MLA: float = 4.0
_ROUND_PACKING_NUMERATOR: float = 4.0
_MIN_PACKING_DIVISOR: float = 0.5
_PAGE_TABLE_BYTE_BUDGET: int = 524_288
_BYTES_PER_PAGE_ENTRY: int = 4


@dataclass(frozen=True)
class TrafficScenario:
    """User-selected workload distribution and optional p90 SLO bounds."""

    name: str
    isl: int
    osl: int
    prefix_hit_rate: float
    accept_len: float
    think_time_s: float
    concurrencies: tuple[int, ...]
    kv_working_set_turns: float
    prefill_chunk: int = 2048
    ttft_slo_ms: float | None = None
    tpot_slo_ms: float | None = None
    spec_tokens: int | None = None
    kv_offload_gb: float = 0.0
    page_size: int | None = None
    max_num_seqs: int | None = None

    def __post_init__(self) -> None:
        """Validate input sequence lengths, hit rates, and concurrency parameters."""
        if self.isl <= 0 or self.osl <= 0:
            msg = f"isl and osl must be > 0, got {self.isl}, {self.osl}"
            raise ValueError(msg)
        if not 0.0 <= self.prefix_hit_rate <= 1.0:
            msg = f"prefix_hit_rate must be in [0, 1], got {self.prefix_hit_rate}"
            raise ValueError(msg)
        if self.accept_len <= 0.0:
            msg = f"accept_len must be > 0, got {self.accept_len}"
            raise ValueError(msg)
        if self.think_time_s < 0.0:
            msg = f"think_time_s must be >= 0, got {self.think_time_s}"
            raise ValueError(msg)
        if not self.concurrencies or any(c <= 0 for c in self.concurrencies):
            msg = f"concurrencies must be positive ints, got {self.concurrencies}"
            raise ValueError(msg)
        if self.page_size is not None and self.page_size <= 0:
            msg = f"page_size must be > 0, got {self.page_size}"
            raise ValueError(msg)
        if self.max_num_seqs is not None and self.max_num_seqs <= 0:
            msg = f"max_num_seqs must be > 0, got {self.max_num_seqs}"
            raise ValueError(msg)


EXAMPLE_SCENARIOS: dict[str, TrafficScenario] = {
    "chat-8k1k": TrafficScenario(
        name="example:chat-8k1k",
        isl=8192,
        osl=1024,
        concurrencies=(1, 4, 8, 16, 32, 64, 128, 256),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        prefill_chunk=2048,
    ),
    "reasoning-1k8k": TrafficScenario(
        name="example:reasoning-1k8k",
        isl=1024,
        osl=8192,
        concurrencies=(1, 4, 8, 16, 32, 64, 128, 256),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        prefill_chunk=2048,
    ),
    "coding-agent": TrafficScenario(
        name="example:coding-agent",
        isl=16384,
        osl=1024,
        concurrencies=(1, 4, 8, 16, 32, 64, 128),
        prefix_hit_rate=0.92,
        accept_len=3.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        prefill_chunk=2048,
        spec_tokens=3,
    ),
    "custom": TrafficScenario(
        name="example:custom",
        isl=4096,
        osl=1024,
        concurrencies=(1, 4, 16, 64),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        prefill_chunk=2048,
    ),
}

EXAMPLE_SLO_TIERS: dict[str, tuple[float | None, float | None]] = {
    "none": (None, None),
    "interactive": (5000.0, 33.33),
    "throughput": (5000.0, 66.67),
}


def build_example_scenario(
    preset: str = "chat-8k1k",
    *,
    slo_tier: str = "none",
    **kwargs: float | tuple[int, ...] | str | None,
) -> TrafficScenario:
    """Resolve a TrafficScenario from an explicitly labeled example scenario."""
    if preset not in EXAMPLE_SCENARIOS:
        msg = (
            f"Unknown example scenario preset {preset!r}; "
            f"known: {sorted(EXAMPLE_SCENARIOS)}"
        )
        raise ValueError(msg)
    if slo_tier not in EXAMPLE_SLO_TIERS:
        msg = f"Unknown SLO tier {slo_tier!r}; known: {sorted(EXAMPLE_SLO_TIERS)}"
        raise ValueError(msg)
    base = EXAMPLE_SCENARIOS[preset]
    tier_ttft, tier_tpot = EXAMPLE_SLO_TIERS[slo_tier]
    isl = kwargs.get("isl")
    osl = kwargs.get("osl")
    concurrencies = kwargs.get("concurrencies")
    prefix_hit_rate = kwargs.get("prefix_hit_rate")
    accept_len = kwargs.get("accept_len")
    think_time_s = kwargs.get("think_time_s")
    prefill_chunk = kwargs.get("prefill_chunk")
    ttft_slo_ms = kwargs.get("ttft_slo_ms")
    tpot_slo_ms = kwargs.get("tpot_slo_ms")
    kv_working_set_turns = kwargs.get("kv_working_set_turns")
    return replace(
        base,
        isl=int(isl) if isinstance(isl, (int, float, str)) else base.isl,
        osl=int(osl) if isinstance(osl, (int, float, str)) else base.osl,
        concurrencies=(
            concurrencies if isinstance(concurrencies, tuple) else base.concurrencies
        ),
        prefix_hit_rate=(
            float(prefix_hit_rate)
            if isinstance(prefix_hit_rate, (int, float, str))
            else base.prefix_hit_rate
        ),
        accept_len=(
            float(accept_len)
            if isinstance(accept_len, (int, float, str))
            else base.accept_len
        ),
        think_time_s=(
            float(think_time_s)
            if isinstance(think_time_s, (int, float, str))
            else base.think_time_s
        ),
        prefill_chunk=(
            int(prefill_chunk)
            if isinstance(prefill_chunk, (int, float, str))
            else base.prefill_chunk
        ),
        ttft_slo_ms=(
            float(ttft_slo_ms)
            if isinstance(ttft_slo_ms, (int, float, str))
            else (tier_ttft if tier_ttft is not None else base.ttft_slo_ms)
        ),
        tpot_slo_ms=(
            float(tpot_slo_ms)
            if isinstance(tpot_slo_ms, (int, float, str))
            else (tier_tpot if tier_tpot is not None else base.tpot_slo_ms)
        ),
        kv_working_set_turns=(
            float(kv_working_set_turns)
            if isinstance(kv_working_set_turns, (int, float, str))
            else base.kv_working_set_turns
        ),
    )


@dataclass(frozen=True)
class HbmBreakdown:
    """Per-TensorCore HBM bytes split into weights, KV cache, and recurrent state."""

    weights: float
    kv: float
    state: float
    total: float


@dataclass(frozen=True)
class FloorTimes:
    """Decode-step resource times (ms) behind a point's binding unit."""

    hbm_ms: float
    matmul_ms: float
    ici_ms: float
    chain_ms: float
    hop_latency_ms: float
    hard_floor_ms: float
    binding: str


@dataclass(frozen=True)
class SearchStats:
    """Search funnel: (layout, concurrency) tuples inside the chip budget."""

    enumerated: int
    hbm_fit: int
    slo_met: int
    frontier: int


@dataclass(frozen=True)
class PoolMeshPlan:
    """Homogeneous instance pool for mixed, prefill-only, or decode-only role."""

    role: Literal["mixed", "prefill", "decode"]
    instances: int
    topology: str
    chips_per_instance: int
    tp: int
    dp: int
    ep: int
    pp: int
    pcp: int
    kv_split: int
    mesh_axes: tuple[int, ...]
    hbm_used_gb: float
    hbm_limit_gb: float
    moe_dispatch: Literal["allreduce", "a2a_push", "ag_ep", "fused_rs"] = "allreduce"
    hbm_weights_gb: float = 0.0
    hbm_kv_gb: float = 0.0
    hbm_state_gb: float = 0.0
    max_num_seqs: int | None = None
    max_model_len: int | None = None
    pp_layer_partition: tuple[int, ...] | None = None
    kv_offload_gb: float = 0.0


@dataclass(frozen=True)
class ParetoPoint:
    """Evaluated serving configuration with latency, throughput, and SLO state."""

    mode: Literal["mixed", "disagg"]
    concurrency: int
    total_chips: int
    prefill_pool: PoolMeshPlan | None
    decode_pool: PoolMeshPlan
    p90_ttft_ms: float
    p90_tpot_ms: float
    p90_interactivity_tok_s: float
    output_throughput_per_chip: float
    token_throughput_per_chip: float
    p_mixed: float
    binding: str
    slo_met: bool
    decode_floor: FloorTimes | None = None
    kv_cache_capacity_tokens: int | None = None
    effective_prefix_hit_rate: float | None = None


@dataclass(frozen=True)
class ServingPlan:
    """Full Pareto frontier and recommended serving configuration for a scenario."""

    model_layers: int
    chip: str
    scenario: TrafficScenario
    mode: str
    recommended: ParetoPoint | None
    frontier: tuple[ParetoPoint, ...]
    warnings: tuple[str, ...]
    candidates: tuple[ParetoPoint, ...] = ()
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    search: SearchStats | None = None
    sweep_candidates: tuple[ParetoPoint, ...] = ()
    winner_prefix_hit_range: tuple[float, float] | None = None
    prefix_hit_crossover_rate: float | None = None
    spec_tokens: int = 0
    mtp_layers: int = 0


def _count_layer_types(
    geom: ModelGeometry,
) -> tuple[int, int, int, int, int, int, float, float, float]:
    """Count attention and recurrent layer types and KV retention factors."""
    n_gqa = 0
    n_swa = 0
    n_mla = 0
    n_dsa = 0
    n_gdn = 0
    n_kda = 0
    kv_n_gqa = 0.0
    kv_n_swa = 0.0
    kv_n_mla = 0.0
    for idx, tok in enumerate(geom.layer_types):
        r_i = geom.compress_ratios[idx] if idx < len(geom.compress_ratios) else 1.0
        kv_ret = (1.0 / r_i) if r_i > 1.0 else r_i
        if tok in ("mla", "mla_attention"):
            n_mla += 1
            kv_n_mla += kv_ret
        elif tok in ("dsa", "dsa_attention", "sparse_mla"):
            n_mla += 1
            n_dsa += 1
            kv_n_mla += kv_ret
        elif tok in ("sliding_attention", "swa", "sliding_window_attention"):
            n_swa += 1
            kv_n_swa += kv_ret
        elif tok in ("kda", "kimi_linear", "kda_attention"):
            n_kda += 1
        elif tok in ("linear_attention", "gdn", "gated_deltanet", "mamba"):
            n_gdn += 1
        elif tok == "ffn_only":
            continue
        elif geom.kv_lora_rank > 0:
            n_mla += 1
            kv_n_mla += kv_ret
            if geom.index_topk > 0:
                n_dsa += 1
        else:
            n_gqa += 1
            kv_n_gqa += kv_ret
    eff_n_gqa = kv_n_gqa if geom.compress_ratios else float(n_gqa)
    eff_n_swa = kv_n_swa if geom.compress_ratios else float(n_swa)
    eff_n_mla = kv_n_mla if geom.compress_ratios else float(n_mla)
    return (
        n_gqa,
        n_swa,
        n_mla,
        n_dsa,
        n_gdn,
        n_kda,
        eff_n_gqa,
        eff_n_swa,
        eff_n_mla,
    )


def _compute_w_attn(
    geom: ModelGeometry,
    dt: Dtypes,
    par: Parallel,
    counts: tuple[int, int, int, int, int, int],
) -> float:
    """Compute per-core attention projection and query-key-value weight bytes."""
    n_gqa, n_swa, n_mla, n_dsa, _, _ = counts
    tp = max(1, par.tp)
    pp = max(1, par.pp)
    q_mult = 2 if geom.attn_output_gate else 1
    kv_div = min(tp, max(1, geom.num_kv_heads))
    w_attn = 0.0
    if n_gqa > 0:
        gqa_hd = geom.global_head_dim if geom.global_head_dim > 0 else geom.head_dim
        w_q_o_gqa = (
            float(n_gqa)
            * float(
                geom.hidden * geom.num_heads * gqa_hd * q_mult
                + geom.num_heads * gqa_hd * geom.hidden,
            )
            * dt.w_dense
            / (tp * pp)
        )
        w_kv_gqa = (
            float(n_gqa)
            * float(2 * geom.hidden * geom.num_kv_heads * gqa_hd)
            * dt.w_dense
            / (kv_div * pp)
        )
        w_attn += w_q_o_gqa + w_kv_gqa
    if n_swa > 0:
        swa_hd = geom.head_dim
        w_q_o_swa = (
            float(n_swa)
            * float(
                geom.hidden * geom.num_heads * swa_hd * q_mult
                + geom.num_heads * swa_hd * geom.hidden,
            )
            * dt.w_dense
            / (tp * pp)
        )
        w_kv_swa = (
            float(n_swa)
            * float(2 * geom.hidden * geom.num_kv_heads * swa_hd)
            * dt.w_dense
            / (kv_div * pp)
        )
        w_attn += w_q_o_swa + w_kv_swa
    if n_mla > 0:
        if geom.kv_lora_rank <= 0:
            msg = "MLA layer requires config field 'kv_lora_rank'"
            raise ValueError(msg)
        if geom.qk_nope_head_dim <= 0:
            msg = "MLA layer requires config field 'qk_nope_head_dim'"
            raise ValueError(msg)
        if geom.qk_rope_head_dim < 0:
            msg = "MLA layer requires config field 'qk_rope_head_dim'"
            raise ValueError(msg)
        kv_lora = geom.kv_lora_rank
        nope_d = geom.qk_nope_head_dim
        rope_d = geom.qk_rope_head_dim
        v_d = geom.v_head_dim or nope_d
        if geom.q_lora_rank > 0:
            q_term_hidden = float(geom.hidden * geom.q_lora_rank) / pp
            q_term_proj = float(
                geom.q_lora_rank * geom.num_heads * (nope_d + rope_d),
            ) / (tp * pp)
        else:
            q_term_hidden = 0.0
            q_term_proj = float(geom.hidden * geom.num_heads * (nope_d + rope_d)) / (
                tp * pp
            )
        w_mla = (
            float(n_mla)
            * (
                q_term_hidden
                + float(geom.hidden * (kv_lora + rope_d)) / pp
                + q_term_proj
                + float(
                    kv_lora * geom.num_heads * (nope_d + v_d)
                    + geom.num_heads * v_d * geom.hidden,
                )
                / (tp * pp)
            )
            * dt.w_dense
        )
        w_attn += w_mla
        if n_dsa > 0 and geom.index_n_heads > 0 and geom.index_head_dim > 0:
            w_attn += (
                float(n_dsa)
                * float(
                    geom.hidden
                    * max(1, geom.index_n_heads // tp)
                    * geom.index_head_dim,
                )
                * dt.w_dense
                / pp
            )
    return w_attn


def _compute_w_rec(
    geom: ModelGeometry,
    dt: Dtypes,
    par: Parallel,
    counts: tuple[int, int, int, int, int, int],
) -> float:
    """Compute per-core recurrent weight bytes for linear attention layers."""
    _, _, _, _, n_gdn, n_kda = counts
    tp = max(1, par.tp)
    pp = max(1, par.pp)
    w_rec = 0.0
    if n_gdn > 0:
        kh = max(1, geom.linear_k_heads // tp)
        vh = max(1, geom.linear_v_heads // tp)
        qkv_r = 2 * kh * geom.linear_k_dim + vh * geom.linear_v_dim
        z_r = vh * geom.linear_v_dim
        w_rec += (
            float(n_gdn)
            * (
                float(geom.hidden * (qkv_r + z_r) + z_r * geom.hidden) * dt.w_dense
                + float(geom.hidden * 2 * vh + qkv_r * geom.conv_kernel) * dt.w_aux
            )
            / pp
        )
    if n_kda > 0:
        kh = max(1, (geom.linear_k_heads or geom.num_heads) // tp)
        vh = max(1, (geom.linear_v_heads or geom.num_heads) // tp)
        kd = geom.linear_k_dim or geom.head_dim
        vd = geom.linear_v_dim or geom.head_dim
        w_rec += (
            float(n_kda)
            * float(geom.hidden * (2 * kh * kd + 2 * vh * vd) + vh * vd * geom.hidden)
            * dt.w_dense
            / pp
        )
    return w_rec


def _compute_w_ffn(
    geom: ModelGeometry,
    dt: Dtypes,
    par: Parallel,
) -> tuple[float, float]:
    """Compute per-core dense and routed MoE feed-forward network weight bytes."""
    tp = max(1, par.tp)
    pp = max(1, par.pp)
    ep = par.ep if (par.moe == "ep" and par.ep > 1) else (tp if par.moe == "ep" else 1)
    w_ffn = 0.0
    w_dense_ffn = 0.0
    n_moe_ffn, n_dense_ffn = geom.ffn_layer_counts()
    if n_dense_ffn > 0 and geom.dense_inter > 0:
        w_dense_ffn = (
            float(n_dense_ffn * 3 * geom.hidden * geom.dense_inter)
            * dt.w_dense
            / (tp * pp)
        )
        w_ffn += w_dense_ffn
    if n_moe_ffn > 0:
        routed_div = ep * pp if par.moe == "ep" else tp * pp
        w_ffn += (
            float(n_moe_ffn * geom.num_experts * 3 * geom.hidden * geom.moe_inter)
            * dt.w_expert
            / routed_div
        )
        if geom.shared_inter > 0:
            isr_eff = (
                geom.shared_inter
                if par.residual == "sp"
                else max(1, geom.shared_inter // tp)
            )
            w_ffn += float(n_moe_ffn * 3 * geom.hidden * isr_eff) * dt.w_dense / pp
        w_ffn += float(n_moe_ffn * geom.hidden * geom.num_experts) * dt.router / pp
    return w_ffn, w_dense_ffn


def _compute_w_mtp(
    geom: ModelGeometry,
    dt: Dtypes,
    par: Parallel,
    w_attn_data: tuple[float, int],
    w_ffn_terms: tuple[float, float],
) -> float:
    """Compute per-core weight bytes for multi-token prediction layers."""
    if geom.mtp_layers <= 0:
        return 0.0
    tp = max(1, par.tp)
    pp = max(1, par.pp)
    w_attn, n_attn_total = w_attn_data
    w_ffn, w_dense_ffn = w_ffn_terms
    mtp_fc = float(2 * geom.hidden * geom.hidden) * dt.w_aux / tp
    mtp_att = (w_attn / float(max(1, n_attn_total))) * float(pp)
    if geom.mtp_inter > 0:
        mtp_ffn = float(3 * geom.hidden * geom.mtp_inter) * dt.w_dense / tp
    elif geom.num_experts > 0:
        n_moe_ffn, _ = geom.ffn_layer_counts()
        n_moe_layers = max(1, n_moe_ffn)
        mtp_ffn = ((w_ffn - w_dense_ffn) / float(n_moe_layers)) * float(pp)
    else:
        mtp_ffn = (w_ffn / float(max(1, geom.num_layers))) * float(pp)
    return float(geom.mtp_layers) * (mtp_fc + mtp_att + mtp_ffn) / float(pp)


def _compute_kv_state_bytes(
    geom: ModelGeometry,
    dt: Dtypes,
    par: Parallel,
    seq_layout: tuple[int, int, bool],
    counts: tuple[int, int, int, int, int, int, float, float, float],
) -> tuple[float, float]:
    """Compute active KV cache and linear recurrent state bytes per core."""
    seqs_per_dp, context, physical_kv_layout = seq_layout
    _, _, n_mla, _, n_gdn, n_kda, eff_n_gqa, eff_n_swa, eff_n_mla = counts
    tp = max(1, par.tp)
    pp = max(1, par.pp)
    kvh = max(1, geom.num_kv_heads // tp) if geom.num_kv_heads > 0 else 0
    if n_mla > 0:
        if geom.kv_lora_rank <= 0:
            msg = "MLA layer requires config field 'kv_lora_rank'"
            raise ValueError(msg)
        if geom.qk_rope_head_dim < 0:
            msg = "MLA layer requires config field 'qk_rope_head_dim'"
            raise ValueError(msg)
    raw_latent_w = geom.kv_lora_rank + geom.qk_rope_head_dim
    gqa_hd = geom.global_head_dim if geom.global_head_dim > 0 else geom.head_dim
    swa_hd = geom.head_dim
    if physical_kv_layout:
        kv_packing = max(
            1,
            round(_ROUND_PACKING_NUMERATOR / max(_MIN_PACKING_DIVISOR, dt.kv)),
        )
        padded_gqa_hd = math.ceil(gqa_hd / 128) * 128 if gqa_hd > 0 else 0
        kx2_gqa = kvh if gqa_hd == _HEAD_DIM_64 else 2 * kvh
        padded_kx2_gqa = (
            math.ceil(kx2_gqa / kv_packing) * kv_packing if kx2_gqa > 0 else 0
        )
        gqa_kv_b_per_tok = float(padded_kx2_gqa * padded_gqa_hd) * dt.kv

        padded_swa_hd = math.ceil(swa_hd / 128) * 128 if swa_hd > 0 else 0
        kx2_swa = kvh if swa_hd == _HEAD_DIM_64 else 2 * kvh
        padded_kx2_swa = (
            math.ceil(kx2_swa / kv_packing) * kv_packing if kx2_swa > 0 else 0
        )
        swa_kv_b_per_tok = float(padded_kx2_swa * padded_swa_hd) * dt.kv

        mla_latent_b = float(math.ceil(raw_latent_w / 128) * 128) * dt.kv
        dsa_idx_b = (
            float(geom.index_head_dim) * dt.kv
            + (_BYTE_SCALE_SPARSE_MLA if dt.kv <= 1.0 else 0.0)
            if (geom.index_topk > 0 and geom.index_head_dim > 0)
            else 0.0
        )
    else:
        gqa_kv_b_per_tok = float(2 * kvh * gqa_hd) * dt.kv
        swa_kv_b_per_tok = float(2 * kvh * swa_hd) * dt.kv
        mla_latent_b = float(raw_latent_w) * dt.kv
        dsa_idx_b = (
            float(geom.index_head_dim) * dt.kv
            if (geom.index_topk > 0 and geom.index_head_dim > 0)
            else 0.0
        )
    kv_div = par.kv_split * par.pcp * pp
    kv_gqa = float(seqs_per_dp * context) * eff_n_gqa * gqa_kv_b_per_tok / kv_div
    eff_swa_ctx = (
        min(context, geom.sliding_window) if geom.sliding_window > 0 else context
    )
    kv_swa = float(seqs_per_dp * eff_swa_ctx) * eff_n_swa * swa_kv_b_per_tok / kv_div
    kv_mla = (
        float(seqs_per_dp * context) * eff_n_mla * (mla_latent_b + dsa_idx_b) / kv_div
    )
    st_bytes = 0.0
    if n_gdn > 0:
        kh = max(1, geom.linear_k_heads // tp)
        vh = max(1, geom.linear_v_heads // tp)
        qkv_r = 2 * kh * geom.linear_k_dim + vh * geom.linear_v_dim
        st_bytes += (
            float(seqs_per_dp * n_gdn)
            * (
                float(vh * geom.linear_k_dim * geom.linear_v_dim) * dt.state
                + float(qkv_r * max(0, geom.conv_kernel - 1)) * dt.conv
            )
            / pp
        )
    if n_kda > 0:
        vh = max(1, (geom.linear_v_heads or geom.num_heads) // tp)
        kd = geom.linear_k_dim or geom.head_dim
        vd = geom.linear_v_dim or geom.head_dim
        st_bytes += float(seqs_per_dp * n_kda * vh * kd * vd) * dt.state / pp
    return kv_gqa + kv_swa + kv_mla, st_bytes


@lru_cache(maxsize=32768)
def hbm_breakdown_per_core_bytes(
    geom: ModelGeometry,
    dt: Dtypes,
    par: Parallel,
    seqs_per_dp: int,
    context: int = 0,
    *,
    physical_kv_layout: bool = True,
) -> HbmBreakdown:
    """Compute static model weights plus active KV/recurrent state bytes on one core."""
    counts = _count_layer_types(geom)
    n_gqa, n_swa, n_mla = counts[0], counts[1], counts[2]
    tp = max(1, par.tp)
    pp = max(1, par.pp)
    w_attn = _compute_w_attn(geom, dt, par, counts[:6])
    w_rec = _compute_w_rec(geom, dt, par, counts[:6])
    w_ffn, w_dense_ffn = _compute_w_ffn(geom, dt, par)
    w_head = float(2 * geom.vocab * geom.hidden) * dt.lm_head / (tp * pp)
    w_mtp = _compute_w_mtp(
        geom,
        dt,
        par,
        (w_attn, n_gqa + n_swa + n_mla),
        (w_ffn, w_dense_ffn),
    )
    kv, state = _compute_kv_state_bytes(
        geom,
        dt,
        par,
        (seqs_per_dp, context, physical_kv_layout),
        counts,
    )
    weights = w_attn + w_rec + w_ffn + w_head + w_mtp
    total = w_attn + w_rec + w_ffn + w_head + w_mtp + kv + state
    return HbmBreakdown(
        weights=weights,
        kv=kv,
        state=state,
        total=total,
    )


def hbm_footprint_per_core_bytes(
    geom: ModelGeometry,
    dt: Dtypes,
    par: Parallel,
    seqs_per_dp: int,
    *args: int,
    **kwargs: bool,
) -> float:
    """Static model weights plus active KV/recurrent state bytes on one TensorCore."""
    return hbm_breakdown_per_core_bytes(
        geom, dt, par, seqs_per_dp, *args, **kwargs
    ).total


def _floor_times(sf: StepFloor) -> FloorTimes:
    """Decode-step resource times (ms) behind a point's binding unit."""
    return FloorTimes(
        hbm_ms=sf.hbm_s * 1e3,
        matmul_ms=sf.mm_s * 1e3,
        ici_ms=sf.ici_s * 1e3,
        chain_ms=sf.chain_s * 1e3,
        hop_latency_ms=sf.lat_s * 1e3,
        hard_floor_ms=sf.hard_floor_s * 1e3,
        binding=sf.binding,
    )


def _default_rpa_page_size(ctx_tokens: int) -> int:
    """Determine default RPA page size based on sequence token budget."""
    if ctx_tokens > _MAX_CTX_TOKENS_FOR_SMALL_PAGE:
        return 16
    pow2 = 1 << (max(1, ctx_tokens) - 1).bit_length()
    return min(256, max(16, pow2 // 16))


def _smem_max_seqs(ctx_tokens: int, kv_split: int, page_size: int) -> int:
    """Estimate maximum concurrent sequences fitting inside SMEM page tables."""
    pages_per_req = max(
        1,
        math.ceil((ctx_tokens / max(1, kv_split)) / max(1, page_size)),
    )
    return _PAGE_TABLE_BYTE_BUDGET // pages_per_req // _BYTES_PER_PAGE_ENTRY


def _valid_head_tp(geom: ModelGeometry, tp: int) -> bool:
    """Check whether TP divides attention heads or evenly groups them."""
    if geom.num_heads > 0 and (tp > geom.num_heads or geom.num_heads % tp != 0):
        return False
    if geom.num_kv_heads <= 0:
        return True
    return geom.num_kv_heads % tp == 0 or tp % geom.num_kv_heads == 0


def _has_gdn_layers(geom: ModelGeometry) -> bool:
    """Check whether any layer type in geometry belongs to gated delta net."""
    return any(
        tok in ("linear_attention", "gdn", "gated_deltanet", "mamba")
        for tok in geom.layer_types
    )


def _valid_gdn_degree(geom: ModelGeometry, deg: int) -> bool:
    """Check legality of GDN degree across key and value heads."""
    if deg <= 1 or not _has_gdn_layers(geom):
        return True
    if geom.linear_v_heads <= 0 or geom.linear_k_heads <= 0:
        return True
    if geom.linear_v_heads % deg != 0:
        return False
    uq = min(geom.linear_k_heads, deg)
    if geom.linear_k_heads % uq != 0 or deg % uq != 0:
        return False
    return (geom.linear_v_heads // deg) % (geom.linear_k_heads // uq) == 0


def _valid_raiden_gdn_pair(
    geom: ModelGeometry,
    par_p: Parallel,
    par_d: Parallel,
) -> bool:
    """Check whether prefill and decode parallelisms form a legal GDN pair."""
    if not _has_gdn_layers(geom):
        return True
    if geom.linear_v_heads <= 0 or geom.linear_k_heads <= 0:
        return True
    deg_p = par_p.tp * par_p.pcp
    deg_d = par_d.tp * par_d.pcp
    if not (_valid_gdn_degree(geom, deg_p) and _valid_gdn_degree(geom, deg_d)):
        return False
    if par_p.pp > 1 and deg_p > 1 and deg_d > deg_p and deg_d % deg_p == 0:
        return True
    kq = geom.linear_k_heads
    if max(1, deg_p // kq) != max(1, deg_d // kq):
        return False
    return deg_p == 1 or deg_p % deg_d == 0


_NON_FULL_ATTN_TOKENS: frozenset[str] = frozenset(
    {
        "sliding_attention",
        "swa",
        "sliding_window_attention",
        "linear_attention",
        "gdn",
        "gated_deltanet",
        "mamba",
        "kda",
        "kimi_linear",
        "kda_attention",
        "ffn_only",
    }
)


_EP_PAIR: int = 2
_MESH_AXES_LEN: int = 4


def _count_full_attn_layers(geom: ModelGeometry) -> int:
    """Count full-attention layers in the model geometry."""
    return sum(1 for tok in geom.layer_types if tok not in _NON_FULL_ATTN_TOKENS)


@lru_cache(maxsize=256)
def _ep_z_unit(ep: int) -> float:
    """Return the expected maximum of standard normal variables under EP."""
    return (
        1.0 / math.sqrt(math.pi)
        if ep == _EP_PAIR
        else _STD_NORM.inv_cdf(1.0 - 1.0 / float(ep))
    )


def max_ep_load_ratio(total_assignments: int, ep: int) -> float:
    """Expected busiest-rank token-load multiplier under uniform EP routing."""
    if ep <= 1 or total_assignments <= 0:
        return 1.0
    z = _ep_z_unit(ep)
    ratio = 1.0 + z * math.sqrt(float(ep - 1) / float(max(1, total_assignments)))
    return min(float(ep), max(1.0, ratio))


def _scale_ep_moe_floor(
    sf: StepFloor,
    chip: ChipSpec,
    geom: ModelGeometry,
    par: Parallel,
    global_tokens: int,
) -> StepFloor:
    """Scale EP MoE module flops and latency by the max EP load ratio."""
    if par.ep <= 1 or geom.num_experts <= 0 or global_tokens <= 0:
        return sf
    ratio = max_ep_load_ratio(global_tokens * geom.top_k, par.ep)
    if ratio <= 1.0:
        return sf
    scaled = tuple(
        replace(m, fp8_flops=m.fp8_flops * ratio, bf16_flops=m.bf16_flops * ratio)
        if m.name in ("moe", "draft_moe")
        else m
        for m in sf.modules
    )
    mm_s = (
        sum(m.fp8_flops for m in scaled) / chip.fp8_flops
        + sum(m.bf16_flops for m in scaled) / chip.bf16_flops
    )
    use_torus = par.mesh_axes is not None
    resource_bound_s = max(sf.hbm_s, mm_s, sf.ici_s)
    hard_floor_s = max(resource_bound_s, sf.lat_s, sf.chain_s)
    serial_floor_s = (
        sum(
            max(
                m.hbm_bytes / chip.hbm_bw,
                m.fp8_flops / chip.fp8_flops + m.bf16_flops / chip.bf16_flops,
            )
            for m in scaled
        )
        + sf.chain_s
    )
    if hard_floor_s == sf.hbm_s:
        binding = "hbm"
    elif hard_floor_s == mm_s:
        binding = "matmul"
    elif hard_floor_s > sf.ici_s or use_torus or sf.lat_s > sf.ici_s:
        binding = "chain"
    else:
        binding = "ici"
    return StepFloor(
        modules=scaled,
        hbm_s=sf.hbm_s,
        mm_s=mm_s,
        ici_s=sf.ici_s,
        chain_s=sf.chain_s,
        hard_floor_s=hard_floor_s,
        serial_floor_s=serial_floor_s,
        binding=binding,
        lat_s=sf.lat_s,
    )


def _scaled_ep_serial_s(
    sf: StepFloor,
    chip: ChipSpec,
    geom: ModelGeometry,
    par: Parallel,
    global_tokens: int,
) -> float:
    """Compute serial execution latency of step floor scaled by EP routing imbalance."""
    if par.ep <= 1 or geom.num_experts <= 0 or global_tokens <= 0:
        return sf.serial_floor_s
    ratio = max_ep_load_ratio(global_tokens * geom.top_k, par.ep)
    if ratio <= 1.0:
        return sf.serial_floor_s
    mod_s = 0.0
    for m in sf.modules:
        mul = ratio if m.name in ("moe", "draft_moe") else 1.0
        hb_s = m.hbm_bytes / chip.hbm_bw
        mm_s = (m.fp8_flops * mul) / chip.fp8_flops + (
            m.bf16_flops * mul
        ) / chip.bf16_flops
        mod_s += max(hb_s, mm_s)
    return mod_s + sf.chain_s


def _scale_pp_stage_floor(
    sf: StepFloor,
    pp_part: tuple[int, ...] | None,
    chip: ChipSpec,
    par: Parallel,
    topology: SliceTopology | None = None,
) -> StepFloor:
    """Scale step floor collectives and execution time by uneven PP stage ratio."""
    if not pp_part or len(pp_part) <= 1:
        return sf
    num_layers = sum(pp_part)
    pp = len(pp_part)
    even_max = max(1, math.ceil(num_layers / pp))
    actual_max = max(pp_part)
    if actual_max == even_max:
        return sf
    ratio = float(actual_max) / float(even_max)
    scaled = []
    for m in sf.modules:
        if m.name in ("embed", "lm_head", "p2p", "pp_p2p"):
            scaled.append(m)
            continue
        colls = [c for c in m.collectives if c.name.startswith("pp")]
        non_pp = [c for c in m.collectives if not c.name.startswith("pp")]
        if non_pp:
            target_n = max(1, round(len(non_pp) * ratio))
            colls.extend(non_pp[idx % len(non_pp)] for idx in range(target_n))
        scaled.append(
            replace(
                m,
                hbm_bytes=m.hbm_bytes * ratio,
                fp8_flops=m.fp8_flops * ratio,
                bf16_flops=m.bf16_flops * ratio,
                collectives=tuple(colls),
                unique_collectives=None,
            )
        )
    return _combine_modules(tuple(scaled), chip, topology=topology, par=par)


def _peak_prefill_logit_bytes(
    geom: ModelGeometry,
    dtypes: Dtypes,
    par: Parallel,
    chunk_rows: int,
    *,
    sampled_rows: int = 1,
) -> float:
    """Return estimated peak HBM buffer bytes for prefill logit and activation."""
    if (chunk_rows <= 0 and sampled_rows <= 0) or geom.vocab <= 0:
        return 0.0
    tp = max(1, par.tp)
    rows_local = math.ceil(max(0, chunk_rows) / max(1, par.pcp))
    n_samp = max(1, sampled_rows) if chunk_rows > 0 else max(0, sampled_rows)
    act_b = 2.0 if dtypes.act == "a16" else 1.0
    logit_b = max(act_b, dtypes.lm_head)
    return (
        float(n_samp) * (float(geom.vocab) / float(tp)) * logit_b
        + float(rows_local) * float(geom.hidden) * act_b
    )


def _disagg_kv_xfer_s(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    topos: tuple[SliceTopology, SliceTopology],
    uncached_tokens: int,
) -> float:
    """Return transfer time in seconds to migrate KV cache over DCN between pools."""
    if chip.dcn_bw <= 0.0 or not math.isfinite(chip.dcn_bw):
        return 0.0
    bd_one = hbm_breakdown_per_core_bytes(
        geom,
        dtypes,
        Parallel(tp=1, pp=1),
        1,
        max(0, uncached_tokens),
        physical_kv_layout=False,
    )
    wire_b = bd_one.kv + bd_one.state
    topo_p, topo_d = topos
    denom = float(max(1, min(topo_p.num_chips, topo_d.num_chips))) * chip.dcn_bw
    return wire_b / denom


def _pp_layer_prefixes(
    geom: ModelGeometry,
    dtypes: Dtypes,
    par1: Parallel,
    context: int,
    seqs_per_dp: int,
) -> tuple[
    tuple[list[float], list[float], list[float], list[int]], tuple[float, float]
]:
    """Compute prefix sums of weights, KV, state, and full-attention counts."""
    n_l = geom.num_layers
    tp = max(1, par1.tp)
    w_embed = float(geom.vocab * geom.hidden) * dtypes.lm_head / float(tp)
    if geom.mtp_layers > 0:
        bd_with_mtp = hbm_breakdown_per_core_bytes(geom, dtypes, par1, 0, 0)
        bd_no_mtp = hbm_breakdown_per_core_bytes(
            replace(geom, mtp_layers=0), dtypes, par1, 0, 0
        )
        w_mtp_total = max(0.0, bd_with_mtp.weights - bd_no_mtp.weights)
    else:
        w_mtp_total = 0.0
    w_lm = w_embed + w_mtp_total
    w_head_both = 2.0 * w_embed

    layer_w: list[float] = []
    layer_kv: list[float] = []
    layer_st: list[float] = []
    is_full: list[int] = []
    for idx in range(n_l):
        cr_i = (geom.compress_ratios[idx],) if idx < len(geom.compress_ratios) else ()
        f_kind = (
            geom.ffn_kinds[idx]
            if idx < len(geom.ffn_kinds)
            else (
                "moe"
                if geom.num_experts > 0 and idx >= geom.first_k_dense_replace
                else "dense"
            )
        )
        g_i = replace(
            geom,
            num_layers=1,
            layer_types=(geom.layer_types[idx],),
            compress_ratios=cr_i,
            ffn_kinds=(f_kind,),
            n_moe_layers=1 if f_kind == "moe" else 0,
            n_dense_layers=1 if f_kind == "dense" else 0,
            first_k_dense_replace=0 if f_kind == "moe" else 1,
            mtp_layers=0,
        )
        bd_i = hbm_breakdown_per_core_bytes(g_i, dtypes, par1, seqs_per_dp, context)
        layer_w.append(max(0.0, bd_i.weights - w_head_both))
        layer_kv.append(bd_i.kv)
        layer_st.append(bd_i.state)
        is_full.append(0 if geom.layer_types[idx] in _NON_FULL_ATTN_TOKENS else 1)

    pref_w = [0.0] * (n_l + 1)
    pref_kv = [0.0] * (n_l + 1)
    pref_st = [0.0] * (n_l + 1)
    pref_full = [0] * (n_l + 1)
    for i in range(n_l):
        pref_w[i + 1] = pref_w[i] + layer_w[i]
        pref_kv[i + 1] = pref_kv[i] + layer_kv[i]
        pref_st[i + 1] = pref_st[i] + layer_st[i]
        pref_full[i + 1] = pref_full[i] + is_full[i]
    return ((pref_w, pref_kv, pref_st, pref_full), (w_embed, w_lm))


def _check_uniform_pp(
    unif: tuple[int, ...],
    pref: tuple[list[float], list[float], list[float], list[int]],
    weights: tuple[float, float],
    min_full: int,
    target_cost: float,
) -> bool:
    """Check if uniform layer partition is legal and cost-competitive."""
    pref_w, pref_kv, pref_st, pref_full = pref
    w_embed, w_lm = weights
    pp = len(unif)
    u_max = 0.0
    pos = 0
    for s, cnt in enumerate(unif):
        nxt = pos + cnt
        if min_full > 0 and (pref_full[nxt] - pref_full[pos]) < min_full:
            return False
        ex = (w_embed if s == 0 else 0.0) + (w_lm if s == pp - 1 else 0.0)
        u_max = max(
            u_max,
            (pref_w[nxt] - pref_w[pos])
            + (pref_kv[nxt] - pref_kv[pos])
            + (pref_st[nxt] - pref_st[pos])
            + ex,
        )
        pos = nxt
    return u_max <= target_cost * (1.0 + 1e-6)


def _relax_stage_dp(
    stage_info: tuple[int, float, int],
    cfg: tuple[int, int, int | None],
    pref: tuple[list[float], list[float], list[float], list[int]],
    dp: tuple[list[list[float]], list[list[int]]],
) -> None:
    """Relax DP transitions for a single pipeline stage index s."""
    s, extra, min_full = stage_info
    pp, n_l, max_stage_layers = cfg
    pref_w, pref_kv, pref_st, pref_full = pref
    dp_cost, dp_prev = dp
    inf = math.inf
    rem_stages = pp - s
    for b in range(s, n_l - rem_stages + 1):
        best_c = inf
        best_a = -1
        a_min = (
            max(s - 1, b - max_stage_layers)
            if max_stage_layers is not None
            else (s - 1)
        )
        for a in range(a_min, b):
            if dp_cost[s - 1][a] == inf:
                continue
            if min_full > 0 and (pref_full[b] - pref_full[a]) < min_full:
                continue
            seg = (
                (pref_w[b] - pref_w[a])
                + (pref_kv[b] - pref_kv[a])
                + (pref_st[b] - pref_st[a])
                + extra
            )
            cand = max(dp_cost[s - 1][a], seg)
            if cand < best_c:
                best_c = cand
                best_a = a
        dp_cost[s][b] = best_c
        dp_prev[s][b] = best_a


def _dp_solve_pp(
    cfg: tuple[int, int, int | None],
    pref: tuple[list[float], list[float], list[float], list[int]],
    weights: tuple[float, float],
    min_full: int = 0,
) -> tuple[tuple[int, ...], bool]:
    """Solve DP partitioning for contiguous pipeline stages."""
    pp, n_l, _ = cfg
    w_embed, w_lm = weights
    inf = math.inf
    dp_cost = [[inf] * (n_l + 1) for _ in range(pp + 1)]
    dp_prev = [[-1] * (n_l + 1) for _ in range(pp + 1)]
    dp_cost[0][0] = 0.0

    for s in range(1, pp + 1):
        extra = (w_embed if s == 1 else 0.0) + (w_lm if s == pp else 0.0)
        _relax_stage_dp((s, extra, min_full), cfg, pref, (dp_cost, dp_prev))

    if dp_prev[pp][n_l] < 0:
        base, rem = n_l // pp, n_l % pp
        return (tuple(base + (1 if s < rem else 0) for s in range(pp)), False)

    rev: list[int] = []
    cur = n_l
    for s in range(pp, 0, -1):
        prv = dp_prev[s][cur]
        rev.append(cur - prv)
        cur = prv
    part = tuple(reversed(rev))

    if n_l % pp == 0:
        unif = (n_l // pp,) * pp
        if _check_uniform_pp(unif, pref, weights, min_full, dp_cost[pp][n_l]):
            part = unif
    return (part, True)


def _pp_part_breakdown(
    part: tuple[int, ...],
    pref_3: tuple[list[float], list[float], list[float]],
    weights: tuple[float, float],
    pp: int,
) -> tuple[HbmBreakdown, float]:
    """Calculate the peak stage HBM breakdown and last-stage footprint."""
    pref_w, pref_kv, pref_st = pref_3
    w_embed, w_lm = weights
    best_bd = HbmBreakdown(0.0, 0.0, 0.0, -1.0)
    last_tot = 0.0
    pos = 0
    for s, cnt in enumerate(part):
        nxt = pos + cnt
        ex_w = (w_embed if s == 0 else 0.0) + (w_lm if s == pp - 1 else 0.0)
        sw = (pref_w[nxt] - pref_w[pos]) + ex_w
        skv = pref_kv[nxt] - pref_kv[pos]
        sst = pref_st[nxt] - pref_st[pos]
        stot = sw + skv + sst
        if s == pp - 1:
            last_tot = stot
        if stot > best_bd.total:
            best_bd = HbmBreakdown(weights=sw, kv=skv, state=sst, total=stot)
        pos = nxt
    return (best_bd, last_tot)


def _pp_stage_breakdowns(
    geom: ModelGeometry,
    dtypes: Dtypes,
    par: Parallel,
    workload: tuple[int, int] = (0, 1),
    options: tuple[bool, int | None] = (False, None),
) -> tuple[tuple[int, ...], HbmBreakdown, float, bool]:
    """Dynamic-programming contiguous PP layer partition and HBM breakdown."""
    context, seqs_per_dp = workload
    disagg, max_stage_layers = options
    pp = max(1, par.pp)
    n_l = geom.num_layers
    if pp <= 1 or n_l <= 0:
        bd = hbm_breakdown_per_core_bytes(geom, dtypes, par, seqs_per_dp, context)
        return ((n_l,), bd, bd.total, True)
    par1 = replace(par, pp=1)
    pref, weights = _pp_layer_prefixes(geom, dtypes, par1, context, seqs_per_dp)
    req_full = disagg and (0 < pref[3][n_l] < n_l) and pp <= pref[3][n_l]
    part, feasible = _dp_solve_pp(
        (pp, n_l, max_stage_layers), pref, weights, min_full=1 if req_full else 0
    )
    best_bd, last_tot = _pp_part_breakdown(part, pref[:3], weights, pp)
    return (part, best_bd, last_tot, feasible)


pp_stage_breakdowns = _pp_stage_breakdowns


def partition_pp_layers(
    geom: ModelGeometry,
    dtypes: Dtypes,
    par: Parallel,
    workload: tuple[int, int] = (0, 1),
    *,
    disagg: bool = False,
) -> tuple[int, ...]:
    """Dynamic-programming contiguous PP layer partition minimizing max-stage HBM."""
    part, _, _, _ = _pp_stage_breakdowns(
        geom, dtypes, par, workload=workload, options=(disagg, None)
    )
    return part


def _kv_capacity_and_hit_rate(
    geom: ModelGeometry,
    dtypes: Dtypes,
    par: Parallel,
    b_res: int,
    b_ws: int,
    ctx_full: int,
    hbm_limit_b: float,
    w_base_b: float,
    b_kv_phys_core: float,
    offload_b_per_core: float,
    scenario: TrafficScenario,
) -> tuple[float, float]:
    """Compute physical KV token capacity and clamped prefix cache hit rate."""
    st_b = hbm_footprint_per_core_bytes(geom, dtypes, par, b_res, 0) - w_base_b
    cap_tok = (
        (max(0.0, hbm_limit_b - w_base_b - st_b) + offload_b_per_core) / b_kv_phys_core
        if b_kv_phys_core > 0.0
        else math.inf
    )
    w_tok = max(0.0, scenario.kv_working_set_turns) * float(b_ws * ctx_full)
    h_eff = (
        scenario.prefix_hit_rate * min(1.0, cap_tok / max(1.0, w_tok))
        if scenario.prefix_hit_rate > 0.0 and w_tok > 0.0 and b_kv_phys_core > 0.0
        else scenario.prefix_hit_rate
    )
    return cap_tok, h_eff


def _divisors(n: int) -> tuple[int, ...]:
    """Return all integer divisors of n in ascending order."""
    return tuple(d for d in range(1, n + 1) if n % d == 0)


def _candidate_decode_parallels(
    geom: ModelGeometry,
    topo: SliceTopology,
) -> list[Parallel]:
    """Enumerate legal decode parallelism configurations for a given topology."""
    cores = topo.num_cores
    out: list[Parallel] = []
    for tp in _divisors(cores):
        if not _valid_head_tp(geom, tp) or not _valid_gdn_degree(geom, tp):
            continue
        dp = cores // tp
        kv_group = (
            max(1, tp // geom.num_kv_heads)
            if geom.num_kv_heads > 0 and geom.kv_lora_rank == 0
            else tp
        )
        kv_splits = _divisors(kv_group) if kv_group > 1 else (1,)
        moe_mode: Literal["tp_slice", "ep"] = (
            "ep"
            if geom.num_experts > 0 and cores > 1 and geom.num_experts >= cores
            else "tp_slice"
        )
        if geom.num_experts > 0 and moe_mode == "tp_slice" and dp > 1:
            continue
        ep_val = cores if moe_mode == "ep" else 1
        mesh_ax = submesh_for_group(tp, topo, topo.cores_per_chip)
        modes: list[
            tuple[
                Literal["tp_slice", "ep"],
                int,
                Literal["allreduce", "ag_ep", "a2a_push"],
            ]
        ]
        if moe_mode == "ep" and dp == 1:
            modes = [
                ("ep", ep_val, "allreduce"),
                ("ep", ep_val, "ag_ep"),
                ("ep", ep_val, "a2a_push"),
            ]
            if (
                geom.moe_inter // tp >= _MIN_MOE_INTERMEDIATE_ALIGNMENT
                and (geom.moe_inter // tp) % _MIN_MOE_INTERMEDIATE_ALIGNMENT == 0
            ):
                modes.append(("tp_slice", 1, "allreduce"))
        elif moe_mode == "ep":
            modes = [
                ("ep", ep_val, "ag_ep"),
                ("ep", ep_val, "a2a_push"),
            ]
        else:
            modes = [("tp_slice", 1, "allreduce")]
        for kvs in kv_splits:
            for m_mode, m_ep, m_disp in modes:
                out.append(
                    Parallel(
                        tp=tp,
                        kv_split=kvs,
                        moe=m_mode,
                        moe_dispatch=m_disp,
                        ranks_per_chip=topo.cores_per_chip,
                        dp=dp,
                        ep=m_ep,
                        pp=1,
                        pcp=1,
                        mesh_axes=mesh_ax,
                    ),
                )
    return out


def _candidate_prefill_parallels(
    geom: ModelGeometry,
    topo: SliceTopology,
    *,
    for_disagg: bool = False,
) -> list[Parallel]:
    """Enumerate legal prefill parallelism configurations for a given topology."""
    cores = topo.num_cores
    n_full_attn = _count_full_attn_layers(geom)
    out: list[Parallel] = []
    for pp in _divisors(cores):
        if geom.num_layers < pp:
            continue
        if for_disagg and 0 < n_full_attn < geom.num_layers and pp > n_full_attn:
            continue
        rem = cores // pp
        for pcp in _divisors(rem):
            tp = rem // pcp
            if tp < 1 or (pp > 1 and tp > 1):
                continue
            if not _valid_head_tp(geom, tp) or not _valid_gdn_degree(geom, tp * pcp):
                continue
            if pcp > 1 and geom.num_experts > 0:
                ep_span = tp * pcp
                if geom.num_experts < ep_span or geom.num_experts % ep_span != 0:
                    continue
                moe_mode: Literal["tp_slice", "ep"] = "ep"
                ep_val = ep_span
            else:
                moe_mode = (
                    "ep"
                    if geom.num_experts > 0 and tp > 1 and geom.num_experts >= tp
                    else "tp_slice"
                )
                ep_val = tp if moe_mode == "ep" else 1
            mesh_ax = submesh_for_group(tp, topo, topo.cores_per_chip)
            dispatches: tuple[Literal["allreduce", "ag_ep"], ...] = (
                ("ag_ep",)
                if (moe_mode == "ep" and tp == 1)
                else (("allreduce", "ag_ep") if moe_mode == "ep" else ("allreduce",))
            )
            out.extend(
                Parallel(
                    tp=tp,
                    kv_split=1,
                    moe=moe_mode,
                    moe_dispatch=disp,
                    ranks_per_chip=topo.cores_per_chip,
                    dp=1,
                    ep=ep_val,
                    pp=pp,
                    pcp=pcp,
                    mesh_axes=mesh_ax,
                )
                for disp in dispatches
            )
    return out


def _pareto_filter(points: Sequence[ParetoPoint]) -> tuple[ParetoPoint, ...]:
    """Filter points to non-dominated Pareto frontier sorted by concurrency."""
    if not points:
        return ()
    ordered = sorted(
        points,
        key=lambda p: (
            -p.p90_interactivity_tok_s,
            -p.token_throughput_per_chip,
            p.total_chips,
        ),
    )
    kept: list[ParetoPoint] = []
    best_tput = -1.0
    for pt in ordered:
        if pt.token_throughput_per_chip > best_tput * (1.0 + 1e-9):
            kept.append(pt)
            best_tput = pt.token_throughput_per_chip
    return tuple(kept)


@dataclass
class _SolveContext:
    """Internal solver context capturing model, hardware, and scenario constants."""

    geom: ModelGeometry
    dtypes: Dtypes
    chip: ChipSpec
    scenario: TrafficScenario
    mode: str
    max_chips: int | None
    max_instances: int | None
    gpu_memory_utilization: float
    hbm_limit_b: float
    hbm_limit_gb: float
    eff_al: float
    spec_k: int
    ctx_full: int
    ctx_mid: int
    eff_isl: int
    n_chunks: int
    page_size: int
    c_max: int
    enumerated: int = 0
    hbm_fit: int = 0


@dataclass(frozen=True)
class _PrefillOption:
    """Feasible prefill configuration for disaggregated serving."""

    topo: SliceTopology
    par: Parallel
    ttft_ms: float
    rate_p: float
    hbm_gb: float
    hbm_bd: HbmBreakdown
    pp_part: tuple[int, ...] | None = None


@dataclass(frozen=True)
class _DisaggDecodeState:
    """Decode configuration and memory bounds for disaggregated serving."""

    topo: SliceTopology
    par: Parallel
    w_base_b: float
    b_hbm_max: int
    b_smem_max: int
    b_kv_phys: float
    offload_b_per_core: float


def _find_prefill_options(
    ctx: _SolveContext,
    topos: list[SliceTopology],
) -> tuple[list[_PrefillOption], float]:
    """Find fastest legal prefill configurations grouped by GDN compatibility class."""
    prefill_options: list[_PrefillOption] = []
    min_req_ttft_ms = math.inf
    pre_chunk_rows = min(ctx.eff_isl, ctx.scenario.prefill_chunk)
    for topo in topos:
        best_by_class: dict[tuple[int, bool], _PrefillOption] = {}
        for par in _candidate_prefill_parallels(ctx.geom, topo, for_disagg=True):
            logit_peak_b = _peak_prefill_logit_bytes(
                ctx.geom, ctx.dtypes, par, pre_chunk_rows, sampled_rows=1
            )
            if par.pp > 1:
                even_cap = max(1, math.ceil(ctx.geom.num_layers / par.pp))
                pp_part, hbm_bd, last_stage_b, pp_feas = _pp_stage_breakdowns(
                    ctx.geom,
                    ctx.dtypes,
                    par,
                    (ctx.scenario.isl, 1),
                    (True, even_cap),
                )
                if (
                    not pp_feas
                    or hbm_bd.total > ctx.hbm_limit_b
                    or (last_stage_b + logit_peak_b) > ctx.hbm_limit_b
                ):
                    pp_part, hbm_bd, last_stage_b, pp_feas = _pp_stage_breakdowns(
                        ctx.geom,
                        ctx.dtypes,
                        par,
                        (ctx.scenario.isl, 1),
                        (True, None),
                    )
                if not pp_feas:
                    continue
                pp_part_opt = pp_part if len(set(pp_part)) > 1 else None
            else:
                hbm_bd = hbm_breakdown_per_core_bytes(
                    ctx.geom,
                    ctx.dtypes,
                    par,
                    1,
                    ctx.scenario.isl,
                )
                last_stage_b = hbm_bd.total
                pp_part_opt = None
            hbm_b = hbm_bd.total
            if (
                hbm_b > ctx.hbm_limit_b
                or (last_stage_b + logit_peak_b) > ctx.hbm_limit_b
            ):
                continue
            sf_p_raw = step_floor(
                ctx.geom,
                ctx.chip,
                par,
                StepShape(
                    batch=0,
                    rows_per_seq=0,
                    context=0,
                    state_writes=0,
                    prefill_tokens=ctx.scenario.isl,
                    prefix_hit_rate=ctx.scenario.prefix_hit_rate,
                ),
                ctx.dtypes,
                topology=topo,
            )
            sf_p = _scale_pp_stage_floor(
                sf_p_raw,
                pp_part_opt,
                ctx.chip,
                par,
                topo,
            )
            ttft_ms = (
                sf_p.serial_floor_s
                * (1.0 + float(par.pp - 1) / float(max(1, ctx.n_chunks)))
                * 1e3
            )
            if ctx.mode in ("auto", "mixed"):
                min_req_ttft_ms = min(min_req_ttft_ms, ttft_ms)
            else:
                xfer_min_ms = (
                    _disagg_kv_xfer_s(
                        ctx.geom, ctx.dtypes, ctx.chip, (topo, topo), ctx.eff_isl
                    )
                    * 1e3
                )
                min_req_ttft_ms = min(min_req_ttft_ms, ttft_ms + xfer_min_ms)
            rate_p = 1.0 / max(1e-9, sf_p.serial_floor_s)
            cls_key = (par.tp * par.pcp, par.pp > 1)
            prev = best_by_class.get(cls_key)
            if prev is None or rate_p > prev.rate_p:
                best_by_class[cls_key] = _PrefillOption(
                    topo=topo,
                    par=par,
                    ttft_ms=ttft_ms,
                    rate_p=rate_p,
                    hbm_gb=hbm_b / 1e9,
                    hbm_bd=hbm_bd,
                    pp_part=pp_part_opt,
                )
        prefill_options.extend(best_by_class.values())
    return prefill_options, min_req_ttft_ms


def _evaluate_mixed_point(
    ctx: _SolveContext,
    topo: SliceTopology,
    par: Parallel,
    c: int,
    bounds: tuple[float, float, int, int, StepFloor],
) -> ParetoPoint | None:
    """Evaluate one concurrency target for a single-pool mixed serving configuration."""
    w_base_b, b_kv_phys_core, b_hbm_max, b_smem_max, sf_ttft = bounds
    if b_hbm_max < 1 or b_smem_max < 1:
        return None
    dec_steps_per_req = max(1.0, ctx.scenario.osl / ctx.eff_al)
    offload_b_per_core = (
        float(ctx.scenario.kv_offload_gb)
        * float(1 << 30)
        / float(max(1, par.tp * par.pp))
    )
    b_local = max(1, math.ceil(c / max(1, par.dp)))
    if ctx.scenario.think_time_s > 0.0:
        for _ in range(2):
            sf_dec_tmp = (
                spec_cycle_floor(
                    ctx.geom,
                    ctx.chip,
                    par,
                    b_local,
                    ctx.ctx_mid,
                    ctx.spec_k,
                    ctx.dtypes,
                    topology=topo,
                )
                if ctx.spec_k > 0
                else step_floor(
                    ctx.geom,
                    ctx.chip,
                    par,
                    StepShape(batch=b_local, context=ctx.ctx_mid),
                    ctx.dtypes,
                    topology=topo,
                )
            )
            w_srv = (
                sf_ttft.serial_floor_s + dec_steps_per_req * sf_dec_tmp.serial_floor_s
            )
            c_act = float(c) * w_srv / (w_srv + ctx.scenario.think_time_s)
            b_local = max(1, math.ceil(c_act / max(1, par.dp)))
    b_res = min(b_local, b_hbm_max, b_smem_max)
    if ctx.scenario.max_num_seqs is not None:
        b_res = min(b_res, ctx.scenario.max_num_seqs)
    hbm_bd = hbm_breakdown_per_core_bytes(
        ctx.geom,
        ctx.dtypes,
        par,
        b_res,
        ctx.ctx_full,
    )
    hbm_b = hbm_bd.total
    pre_chunk_rows = min(ctx.eff_isl, ctx.scenario.prefill_chunk)
    logit_peak_mix_b = _peak_prefill_logit_bytes(
        ctx.geom,
        ctx.dtypes,
        par,
        pre_chunk_rows,
        sampled_rows=b_res * max(1, ctx.spec_k + 1),
    )
    while b_res > 1 and (hbm_b + logit_peak_mix_b) > ctx.hbm_limit_b:
        b_res -= 1
        hbm_bd = hbm_breakdown_per_core_bytes(
            ctx.geom,
            ctx.dtypes,
            par,
            b_res,
            ctx.ctx_full,
        )
        hbm_b = hbm_bd.total
        logit_peak_mix_b = _peak_prefill_logit_bytes(
            ctx.geom,
            ctx.dtypes,
            par,
            pre_chunk_rows,
            sampled_rows=b_res * max(1, ctx.spec_k + 1),
        )
    if (hbm_b + logit_peak_mix_b) > ctx.hbm_limit_b:
        return None
    ctx.hbm_fit += 1
    cap_tok, h_eff = _kv_capacity_and_hit_rate(
        ctx.geom,
        ctx.dtypes,
        par,
        b_res,
        b_local,
        ctx.ctx_full,
        ctx.hbm_limit_b,
        w_base_b,
        b_kv_phys_core,
        offload_b_per_core,
        ctx.scenario,
    )
    cap_tok_int = None if math.isinf(cap_tok) else int(cap_tok)
    sf_ttft_c = (
        step_floor(
            ctx.geom,
            ctx.chip,
            par,
            StepShape(
                batch=0,
                rows_per_seq=0,
                context=0,
                state_writes=0,
                prefill_tokens=ctx.scenario.isl,
                prefix_hit_rate=h_eff,
            ),
            ctx.dtypes,
            topology=topo,
        )
        if h_eff != ctx.scenario.prefix_hit_rate
        else sf_ttft
    )
    _, h_hbm_only = _kv_capacity_and_hit_rate(
        ctx.geom,
        ctx.dtypes,
        par,
        b_res,
        b_local,
        ctx.ctx_full,
        ctx.hbm_limit_b,
        w_base_b,
        b_kv_phys_core,
        0.0,
        scenario=ctx.scenario,
    )
    offload_read_b = (
        max(0.0, h_eff - h_hbm_only) * float(ctx.scenario.isl) * b_kv_phys_core
    )
    pcie_core_bw = ctx.chip.pcie_bw / float(max(1, ctx.chip.cores_per_chip))
    pcie_read_s = (
        offload_read_b / pcie_core_bw
        if offload_read_b > 0.0 and pcie_core_bw > 0.0 and math.isfinite(pcie_core_bw)
        else 0.0
    )
    pre_ttft_s = sf_ttft_c.serial_floor_s + pcie_read_s
    eff_isl_c = max(1, math.ceil((1.0 - h_eff) * ctx.scenario.isl))
    n_chunks_c = max(1, math.ceil(eff_isl_c / max(1, ctx.scenario.prefill_chunk)))
    chunk_tok_c = min(eff_isl_c, ctx.scenario.prefill_chunk)
    sf_dec = (
        spec_cycle_floor(
            ctx.geom,
            ctx.chip,
            par,
            b_res,
            ctx.ctx_mid,
            ctx.spec_k,
            ctx.dtypes,
            topology=topo,
        )
        if ctx.spec_k > 0
        else step_floor(
            ctx.geom,
            ctx.chip,
            par,
            StepShape(batch=b_res, context=ctx.ctx_mid),
            ctx.dtypes,
            topology=topo,
        )
    )
    sf_ver_only = (
        step_floor(
            ctx.geom,
            ctx.chip,
            par,
            StepShape(
                batch=b_res,
                rows_per_seq=ctx.spec_k + 1,
                context=ctx.ctx_mid,
                state_writes=1,
            ),
            ctx.dtypes,
            topology=topo,
        )
        if ctx.spec_k > 0
        else sf_dec
    )
    draft_serial_s = max(0.0, sf_dec.serial_floor_s - sf_ver_only.serial_floor_s)
    sf_mix = step_floor(
        ctx.geom,
        ctx.chip,
        par,
        StepShape(
            batch=b_res,
            rows_per_seq=ctx.spec_k + 1,
            context=ctx.ctx_mid,
            state_writes=1,
            prefill_tokens=chunk_tok_c,
            prefix_hit_rate=0.0,
        ),
        ctx.dtypes,
        topology=topo,
    )
    mix_serial_s = sf_mix.serial_floor_s + draft_serial_s
    total_req_work_s = (
        dec_steps_per_req * sf_dec.serial_floor_s
        + float(b_res * n_chunks_c) * mix_serial_s
    )
    res_cap = float(b_res * max(1, par.dp))
    c_served = min(float(c), res_cap)
    server_cap_rate = res_cap / max(1e-9, total_req_work_s)
    w_srv_final = pre_ttft_s + dec_steps_per_req * sf_dec.serial_floor_s
    rate_req_s = (
        min(
            server_cap_rate,
            float(c) / max(1e-9, w_srv_final + ctx.scenario.think_time_s),
        )
        if ctx.scenario.think_time_s > 0.0
        else c_served / max(1e-9, total_req_work_s)
    )
    p_mix = min(
        1.0,
        float(b_res * n_chunks_c)
        / float(max(1.0, dec_steps_per_req + float(b_res * n_chunks_c))),
    )
    p90_step_s = (1.0 - p_mix) * sf_dec.serial_floor_s + p_mix * mix_serial_s
    p90_tpot_ms = p90_step_s * 1e3 / ctx.eff_al
    c_demand = min(float(c), float(b_local * max(1, par.dp)))
    queue_wait_s = max(0.0, (c_demand / max(1e-9, res_cap)) - 1.0) * w_srv_final
    p90_ttft_ms = (pre_ttft_s + queue_wait_s) * 1e3 + (
        sf_mix.serial_floor_s * 1e3 if c > 1 else 0.0
    )
    interactivity = 1000.0 / max(1e-6, p90_tpot_ms)
    out_tput = rate_req_s * float(ctx.scenario.osl) / float(topo.num_chips)
    tot_tput = (
        rate_req_s * float(ctx.scenario.isl + ctx.scenario.osl) / float(topo.num_chips)
    )
    slo_ok = (
        ctx.scenario.ttft_slo_ms is None or p90_ttft_ms <= ctx.scenario.ttft_slo_ms
    ) and (ctx.scenario.tpot_slo_ms is None or p90_tpot_ms <= ctx.scenario.tpot_slo_ms)
    max_seqs_flag = b_res if b_res < b_local else ctx.scenario.max_num_seqs
    pool = PoolMeshPlan(
        role="mixed",
        instances=1,
        topology=topo.name,
        chips_per_instance=topo.num_chips,
        tp=par.tp,
        dp=par.dp,
        ep=par.ep,
        pp=par.pp,
        pcp=par.pcp,
        kv_split=par.kv_split,
        mesh_axes=(
            par.mesh_axes
            if par.mesh_axes is not None
            else submesh_for_group(par.tp, topo, topo.cores_per_chip)
        ),
        hbm_used_gb=hbm_b / 1e9,
        hbm_limit_gb=ctx.hbm_limit_gb,
        moe_dispatch=par.moe_dispatch,
        hbm_weights_gb=hbm_bd.weights / 1e9,
        hbm_kv_gb=hbm_bd.kv / 1e9,
        hbm_state_gb=hbm_bd.state / 1e9,
        max_num_seqs=max_seqs_flag,
        max_model_len=ctx.ctx_full,
        kv_offload_gb=ctx.scenario.kv_offload_gb,
    )
    binding = (
        f"dec:{sf_dec.binding},mix:{sf_mix.binding}" if p_mix > 0.0 else sf_dec.binding
    )
    return ParetoPoint(
        mode="mixed",
        concurrency=c,
        total_chips=topo.num_chips,
        prefill_pool=None,
        decode_pool=pool,
        p90_ttft_ms=p90_ttft_ms,
        p90_tpot_ms=p90_tpot_ms,
        p90_interactivity_tok_s=interactivity,
        output_throughput_per_chip=out_tput,
        token_throughput_per_chip=tot_tput,
        p_mixed=p_mix,
        binding=binding,
        slo_met=slo_ok,
        decode_floor=_floor_times(sf_dec),
        kv_cache_capacity_tokens=cap_tok_int,
        effective_prefix_hit_rate=h_eff,
    )


def _evaluate_mixed_candidates(
    ctx: _SolveContext,
    topos: list[SliceTopology],
) -> tuple[list[ParetoPoint], float, float]:
    """Evaluate all single-pool mixed candidate parallelisms across topologies."""
    candidates: list[ParetoPoint] = []
    min_c1_tpot_ms = math.inf
    min_req_ttft_ms = math.inf
    for topo in topos:
        for par in _candidate_decode_parallels(ctx.geom, topo):
            w_base_b = hbm_footprint_per_core_bytes(ctx.geom, ctx.dtypes, par, 0, 0)
            hbm_c1 = hbm_footprint_per_core_bytes(
                ctx.geom,
                ctx.dtypes,
                par,
                1,
                ctx.ctx_full,
            )
            per_seq_b = hbm_c1 - w_base_b
            b_hbm_max = (
                int((ctx.hbm_limit_b - w_base_b) // per_seq_b)
                if (hbm_c1 <= ctx.hbm_limit_b and per_seq_b > 0.0)
                else (ctx.c_max if hbm_c1 <= ctx.hbm_limit_b else 0)
            )
            b_smem_max = _smem_max_seqs(ctx.ctx_full, par.kv_split, ctx.page_size)
            if hbm_c1 <= ctx.hbm_limit_b and b_smem_max >= 1:
                sf_c1 = (
                    spec_cycle_floor(
                        ctx.geom,
                        ctx.chip,
                        par,
                        1,
                        ctx.ctx_mid,
                        ctx.spec_k,
                        ctx.dtypes,
                        topology=topo,
                    )
                    if ctx.spec_k > 0
                    else step_floor(
                        ctx.geom,
                        ctx.chip,
                        par,
                        StepShape(batch=1, context=ctx.ctx_mid),
                        ctx.dtypes,
                        topology=topo,
                    )
                )
                min_c1_tpot_ms = min(
                    min_c1_tpot_ms,
                    sf_c1.serial_floor_s * 1e3 / ctx.eff_al,
                )
            if ctx.mode not in ("auto", "mixed"):
                continue
            sf_ttft = step_floor(
                ctx.geom,
                ctx.chip,
                par,
                StepShape(
                    batch=0,
                    rows_per_seq=0,
                    context=0,
                    state_writes=0,
                    prefill_tokens=ctx.scenario.isl,
                    prefix_hit_rate=ctx.scenario.prefix_hit_rate,
                ),
                ctx.dtypes,
                topology=topo,
            )
            min_req_ttft_ms = min(min_req_ttft_ms, sf_ttft.serial_floor_s * 1e3)
            b_kv_phys_core = hbm_footprint_per_core_bytes(
                ctx.geom,
                ctx.dtypes,
                par,
                1,
                1,
            ) - hbm_footprint_per_core_bytes(ctx.geom, ctx.dtypes, par, 1, 0)
            bounds = (w_base_b, b_kv_phys_core, b_hbm_max, b_smem_max, sf_ttft)
            for c in ctx.scenario.concurrencies:
                ctx.enumerated += 1
                pt = _evaluate_mixed_point(ctx, topo, par, c, bounds)
                if pt is not None:
                    candidates.append(pt)
    return candidates, min_c1_tpot_ms, min_req_ttft_ms


def _build_disagg_pool_plans(
    pref: _PrefillOption,
    dec: _DisaggDecodeState,
    instances: tuple[int, int],
    hbm_limits: tuple[float, float, int | None, int],
    kv_offload_gb: float = 0.0,
) -> tuple[PoolMeshPlan, PoolMeshPlan]:
    """Construct prefill and decode PoolMeshPlan instances for disaggregated serving."""
    x_p, y_d = instances
    hbm_d, hbm_limit_gb, max_seqs_d, ctx_full = hbm_limits
    p_pool = PoolMeshPlan(
        role="prefill",
        instances=x_p,
        topology=pref.topo.name,
        chips_per_instance=pref.topo.num_chips,
        tp=pref.par.tp,
        dp=pref.par.dp,
        ep=pref.par.ep,
        pp=pref.par.pp,
        pcp=pref.par.pcp,
        kv_split=pref.par.kv_split,
        mesh_axes=(
            pref.par.mesh_axes
            if pref.par.mesh_axes is not None
            else submesh_for_group(pref.par.tp, pref.topo, pref.topo.cores_per_chip)
        ),
        hbm_used_gb=pref.hbm_gb,
        hbm_limit_gb=hbm_limit_gb,
        moe_dispatch=pref.par.moe_dispatch,
        max_model_len=ctx_full,
        pp_layer_partition=pref.pp_part,
    )
    d_pool = PoolMeshPlan(
        role="decode",
        instances=y_d,
        topology=dec.topo.name,
        chips_per_instance=dec.topo.num_chips,
        tp=dec.par.tp,
        dp=dec.par.dp,
        ep=dec.par.ep,
        pp=dec.par.pp,
        pcp=dec.par.pcp,
        kv_split=dec.par.kv_split,
        mesh_axes=(
            dec.par.mesh_axes
            if dec.par.mesh_axes is not None
            else submesh_for_group(dec.par.tp, dec.topo, dec.topo.cores_per_chip)
        ),
        hbm_used_gb=hbm_d / 1e9,
        hbm_limit_gb=hbm_limit_gb,
        moe_dispatch=dec.par.moe_dispatch,
        max_num_seqs=max_seqs_d,
        max_model_len=ctx_full,
        kv_offload_gb=kv_offload_gb,
    )
    return p_pool, d_pool


def _compute_disagg_metrics(
    ctx: _SolveContext,
    dec: _DisaggDecodeState,
    pref: _PrefillOption,
    rates_in: tuple[int, int, int, int],
    batch_in: tuple[int, int, float, float, float],
    sf_d: StepFloor,
) -> tuple[float, float, float, float, float]:
    """Compute serving latencies and throughputs for disaggregated configuration."""
    c, x_p, y_d, total_chips = rates_in
    b_res, b_local, ttft_eff_ms, rate_p_eff, h_eff = batch_in
    dec_steps_per_req = max(1.0, ctx.scenario.osl / ctx.eff_al)
    p90_tpot_ms = sf_d.serial_floor_s * 1e3 / ctx.eff_al
    interactivity = 1000.0 / max(1e-6, p90_tpot_ms)
    res_cap_d = float(b_res * max(1, y_d * dec.par.dp))
    c_served_d = min(float(c), res_cap_d)
    rate_d_cap = c_served_d / (dec_steps_per_req * max(1e-9, sf_d.serial_floor_s))
    c_demand_d = min(float(c), float(b_local * max(1, y_d * dec.par.dp)))
    queue_wait_d_ms = (
        max(0.0, (c_demand_d / max(1e-9, res_cap_d)) - 1.0)
        * (dec_steps_per_req * sf_d.serial_floor_s)
        * 1e3
    )
    uncached_d_tok = max(1, math.ceil((1.0 - h_eff) * ctx.scenario.isl))
    kv_xfer_ms = (
        _disagg_kv_xfer_s(
            ctx.geom,
            ctx.dtypes,
            ctx.chip,
            (pref.topo, dec.topo),
            uncached_d_tok,
        )
        * 1e3
    )
    _, h_hbm_d = _kv_capacity_and_hit_rate(
        ctx.geom,
        ctx.dtypes,
        dec.par,
        b_res,
        b_local,
        ctx.ctx_full,
        ctx.hbm_limit_b,
        dec.w_base_b,
        dec.b_kv_phys,
        0.0,
        scenario=ctx.scenario,
    )
    offload_read_d_b = (
        max(0.0, h_eff - h_hbm_d) * float(ctx.scenario.isl) * dec.b_kv_phys
    )
    pcie_d_bw = ctx.chip.pcie_bw / float(max(1, ctx.chip.cores_per_chip))
    pcie_d_ms = (
        (offload_read_d_b / pcie_d_bw) * 1e3
        if offload_read_d_b > 0.0 and pcie_d_bw > 0.0 and math.isfinite(pcie_d_bw)
        else 0.0
    )
    p90_ttft_d_ms = ttft_eff_ms + pcie_d_ms + kv_xfer_ms + queue_wait_d_ms
    if ctx.scenario.think_time_s > 0.0:
        client_rate = float(c) / max(
            1e-9,
            ttft_eff_ms * 1e-3
            + dec_steps_per_req * sf_d.serial_floor_s
            + ctx.scenario.think_time_s,
        )
        coupled_rate = min(x_p * rate_p_eff, rate_d_cap, client_rate)
    else:
        coupled_rate = min(x_p * rate_p_eff, rate_d_cap)
    out_tput = coupled_rate * float(ctx.scenario.osl) / float(total_chips)
    tot_tput = (
        coupled_rate * float(ctx.scenario.isl + ctx.scenario.osl) / float(total_chips)
    )
    return p90_ttft_d_ms, p90_tpot_ms, interactivity, out_tput, tot_tput


def _evaluate_disagg_instance_pair(
    ctx: _SolveContext,
    dec: _DisaggDecodeState,
    pref: _PrefillOption,
    target: tuple[int, int, int],
    cache: dict[tuple[str, Parallel, float], tuple[float, float]],
) -> ParetoPoint | None:
    """Evaluate one instance ratio pair (x_p, y_d) for disaggregated serving."""
    c, x_p, y_d = target
    total_chips = x_p * pref.topo.num_chips + y_d * dec.topo.num_chips
    if ctx.max_chips is not None and total_chips > ctx.max_chips:
        return None
    if dec.b_hbm_max < 1 or dec.b_smem_max < 1:
        return None
    dec_steps_per_req = max(1.0, ctx.scenario.osl / ctx.eff_al)
    b_local = max(1, math.ceil(c / max(1, y_d * dec.par.dp)))
    if ctx.scenario.think_time_s > 0.0:
        for _ in range(2):
            sf_d_tmp = (
                spec_cycle_floor(
                    ctx.geom,
                    ctx.chip,
                    dec.par,
                    b_local,
                    ctx.ctx_mid,
                    ctx.spec_k,
                    ctx.dtypes,
                    topology=dec.topo,
                )
                if ctx.spec_k > 0
                else step_floor(
                    ctx.geom,
                    ctx.chip,
                    dec.par,
                    StepShape(batch=b_local, context=ctx.ctx_mid),
                    ctx.dtypes,
                    topology=dec.topo,
                )
            )
            w_srv = pref.ttft_ms * 1e-3 + dec_steps_per_req * sf_d_tmp.serial_floor_s
            c_act = float(c) * w_srv / (w_srv + ctx.scenario.think_time_s)
            b_local = max(1, math.ceil(c_act / max(1, y_d * dec.par.dp)))
    b_res = min(b_local, dec.b_hbm_max, dec.b_smem_max)
    if ctx.scenario.max_num_seqs is not None:
        b_res = min(b_res, ctx.scenario.max_num_seqs)
    hbm_bd_d = hbm_breakdown_per_core_bytes(
        ctx.geom,
        ctx.dtypes,
        dec.par,
        b_res,
        ctx.ctx_full,
    )
    hbm_d = hbm_bd_d.total
    while b_res > 1 and hbm_d > ctx.hbm_limit_b:
        b_res -= 1
        hbm_bd_d = hbm_breakdown_per_core_bytes(
            ctx.geom,
            ctx.dtypes,
            dec.par,
            b_res,
            ctx.ctx_full,
        )
        hbm_d = hbm_bd_d.total
    if hbm_d > ctx.hbm_limit_b:
        return None
    ctx.hbm_fit += 1
    cap_tok, h_eff = _kv_capacity_and_hit_rate(
        ctx.geom,
        ctx.dtypes,
        dec.par,
        b_res,
        b_local,
        ctx.ctx_full,
        ctx.hbm_limit_b,
        dec.w_base_b,
        dec.b_kv_phys,
        dec.offload_b_per_core,
        ctx.scenario,
    )
    cap_tok_int = None if math.isinf(cap_tok) else int(cap_tok)
    if h_eff < ctx.scenario.prefix_hit_rate:
        p_key = (pref.topo.name, pref.par, round(h_eff, 6))
        cached_p = cache.get(p_key)
        if cached_p is None:
            eff_isl_p = max(1, math.ceil((1.0 - h_eff) * ctx.scenario.isl))
            n_chunks_p = max(
                1,
                math.ceil(eff_isl_p / max(1, ctx.scenario.prefill_chunk)),
            )
            sf_p_eff_raw = step_floor(
                ctx.geom,
                ctx.chip,
                pref.par,
                StepShape(
                    batch=0,
                    rows_per_seq=0,
                    context=0,
                    state_writes=0,
                    prefill_tokens=ctx.scenario.isl,
                    prefix_hit_rate=h_eff,
                ),
                ctx.dtypes,
                topology=pref.topo,
            )
            sf_p_eff = _scale_pp_stage_floor(
                sf_p_eff_raw,
                pref.pp_part,
                ctx.chip,
                pref.par,
                pref.topo,
            )
            ttft_p_eff = (
                sf_p_eff.serial_floor_s
                * (1.0 + float(pref.par.pp - 1) / float(max(1, n_chunks_p)))
                * 1e3
            )
            rate_p_eff = 1.0 / max(1e-9, sf_p_eff.serial_floor_s)
            cached_p = (ttft_p_eff, rate_p_eff)
            cache[p_key] = cached_p
        ttft_eff_ms, rate_p_eff = cached_p
    else:
        ttft_eff_ms = pref.ttft_ms
        rate_p_eff = pref.rate_p
    sf_d = (
        spec_cycle_floor(
            ctx.geom,
            ctx.chip,
            dec.par,
            b_res,
            ctx.ctx_mid,
            ctx.spec_k,
            ctx.dtypes,
            topology=dec.topo,
        )
        if ctx.spec_k > 0
        else step_floor(
            ctx.geom,
            ctx.chip,
            dec.par,
            StepShape(batch=b_res, context=ctx.ctx_mid),
            ctx.dtypes,
            topology=dec.topo,
        )
    )
    p90_ttft_d_ms, p90_tpot_ms, interactivity, out_tput, tot_tput = (
        _compute_disagg_metrics(
            ctx,
            dec,
            pref,
            (c, x_p, y_d, total_chips),
            (b_res, b_local, ttft_eff_ms, rate_p_eff, h_eff),
            sf_d,
        )
    )
    slo_ok = (
        ctx.scenario.ttft_slo_ms is None or p90_ttft_d_ms <= ctx.scenario.ttft_slo_ms
    ) and (ctx.scenario.tpot_slo_ms is None or p90_tpot_ms <= ctx.scenario.tpot_slo_ms)
    max_seqs_d = b_res if b_res < b_local else ctx.scenario.max_num_seqs
    p_pool = PoolMeshPlan(
        role="prefill",
        instances=x_p,
        topology=pref.topo.name,
        chips_per_instance=pref.topo.num_chips,
        tp=pref.par.tp,
        dp=pref.par.dp,
        ep=pref.par.ep,
        pp=pref.par.pp,
        pcp=pref.par.pcp,
        kv_split=pref.par.kv_split,
        mesh_axes=(
            pref.par.mesh_axes
            if pref.par.mesh_axes is not None
            else submesh_for_group(pref.par.tp, pref.topo, pref.topo.cores_per_chip)
        ),
        hbm_used_gb=pref.hbm_gb,
        hbm_limit_gb=ctx.hbm_limit_gb,
        moe_dispatch=pref.par.moe_dispatch,
        hbm_weights_gb=pref.hbm_bd.weights / 1e9,
        hbm_kv_gb=pref.hbm_bd.kv / 1e9,
        hbm_state_gb=pref.hbm_bd.state / 1e9,
        max_model_len=ctx.ctx_full,
        pp_layer_partition=pref.pp_part,
    )
    d_pool = PoolMeshPlan(
        role="decode",
        instances=y_d,
        topology=dec.topo.name,
        chips_per_instance=dec.topo.num_chips,
        tp=dec.par.tp,
        dp=dec.par.dp,
        ep=dec.par.ep,
        pp=dec.par.pp,
        pcp=dec.par.pcp,
        kv_split=dec.par.kv_split,
        mesh_axes=(
            dec.par.mesh_axes
            if dec.par.mesh_axes is not None
            else submesh_for_group(dec.par.tp, dec.topo, dec.topo.cores_per_chip)
        ),
        hbm_used_gb=hbm_d / 1e9,
        hbm_limit_gb=ctx.hbm_limit_gb,
        moe_dispatch=dec.par.moe_dispatch,
        hbm_weights_gb=hbm_bd_d.weights / 1e9,
        hbm_kv_gb=hbm_bd_d.kv / 1e9,
        hbm_state_gb=hbm_bd_d.state / 1e9,
        max_num_seqs=max_seqs_d,
        max_model_len=ctx.ctx_full,
        kv_offload_gb=ctx.scenario.kv_offload_gb,
    )
    return ParetoPoint(
        mode="disagg",
        concurrency=c,
        total_chips=total_chips,
        prefill_pool=p_pool,
        decode_pool=d_pool,
        p90_ttft_ms=p90_ttft_d_ms,
        p90_tpot_ms=p90_tpot_ms,
        p90_interactivity_tok_s=interactivity,
        output_throughput_per_chip=out_tput,
        token_throughput_per_chip=tot_tput,
        p_mixed=0.0,
        binding=sf_d.binding,
        slo_met=slo_ok,
        decode_floor=_floor_times(sf_d),
        kv_cache_capacity_tokens=cap_tok_int,
        effective_prefix_hit_rate=h_eff,
    )


def _search_disagg_instance_ratios(
    ctx: _SolveContext,
    dec: _DisaggDecodeState,
    pref: _PrefillOption,
    c: int,
    cache: dict[tuple[str, Parallel, float], tuple[float, float]],
) -> list[ParetoPoint]:
    """Search instance multiplier ratios (x_p, y_d) for one prefill-decode pair."""
    dec_steps_per_req = max(1.0, ctx.scenario.osl / ctx.eff_al)
    b_one = math.ceil(c / max(1, dec.par.dp))
    sf_d_one = (
        spec_cycle_floor(
            ctx.geom,
            ctx.chip,
            dec.par,
            b_one,
            ctx.ctx_mid,
            ctx.spec_k,
            ctx.dtypes,
            topology=dec.topo,
        )
        if ctx.spec_k > 0
        else step_floor(
            ctx.geom,
            ctx.chip,
            dec.par,
            StepShape(batch=b_one, context=ctx.ctx_mid),
            ctx.dtypes,
            topology=dec.topo,
        )
    )
    rate_d_one = float(c) / (dec_steps_per_req * max(1e-9, sf_d_one.serial_floor_s))
    eff_max_inst = ctx.max_instances if ctx.max_instances is not None else 16
    if ctx.max_chips is None:
        ratio_xp = math.ceil(rate_d_one / max(1e-9, pref.rate_p)) + 1
        ratio_yd = math.ceil(pref.rate_p / max(1e-9, rate_d_one)) + 1
        max_xp = min(eff_max_inst, max(1, ratio_xp))
        max_yd = min(eff_max_inst, max(1, ratio_yd))
    else:
        max_xp = min(eff_max_inst, max(1, ctx.max_chips // pref.topo.num_chips))
        max_yd = min(eff_max_inst, max(1, ctx.max_chips // dec.topo.num_chips))
    points: list[ParetoPoint] = []
    for x_p in range(1, max_xp + 1):
        for y_d in range(1, max_yd + 1):
            total_chips = x_p * pref.topo.num_chips + y_d * dec.topo.num_chips
            if ctx.max_chips is not None and total_chips > ctx.max_chips:
                continue
            ctx.enumerated += 1
            pt = _evaluate_disagg_instance_pair(ctx, dec, pref, (c, x_p, y_d), cache)
            if pt is not None:
                points.append(pt)
    return points


def _evaluate_disaggregated_candidates(
    ctx: _SolveContext,
    topos: list[SliceTopology],
    prefill_options: list[_PrefillOption],
) -> list[ParetoPoint]:
    """Evaluate all unconstrained disaggregated serving candidate configurations."""
    if ctx.mode not in ("auto", "disagg") or not prefill_options:
        return []
    candidates: list[ParetoPoint] = []
    cache: dict[tuple[str, Parallel, float], tuple[float, float]] = {}
    for topo_d in topos:
        for par_d in _candidate_decode_parallels(ctx.geom, topo_d):
            offload_b = (
                float(ctx.scenario.kv_offload_gb)
                * float(1 << 30)
                / float(max(1, par_d.tp * par_d.pp))
            )
            w_base_d = hbm_footprint_per_core_bytes(ctx.geom, ctx.dtypes, par_d, 0, 0)
            hbm_d1 = hbm_footprint_per_core_bytes(
                ctx.geom,
                ctx.dtypes,
                par_d,
                1,
                ctx.ctx_full,
            )
            per_seq_d = hbm_d1 - w_base_d
            b_hbm_max_d = (
                int((ctx.hbm_limit_b - w_base_d) // per_seq_d)
                if (hbm_d1 <= ctx.hbm_limit_b and per_seq_d > 0.0)
                else (ctx.c_max if hbm_d1 <= ctx.hbm_limit_b else 0)
            )
            b_smem_max_d = _smem_max_seqs(ctx.ctx_full, par_d.kv_split, ctx.page_size)
            b_kv_phys_d = hbm_footprint_per_core_bytes(
                ctx.geom,
                ctx.dtypes,
                par_d,
                1,
                1,
            ) - hbm_footprint_per_core_bytes(ctx.geom, ctx.dtypes, par_d, 1, 0)
            dec_state = _DisaggDecodeState(
                topo=topo_d,
                par=par_d,
                w_base_b=w_base_d,
                b_hbm_max=b_hbm_max_d,
                b_smem_max=b_smem_max_d,
                b_kv_phys=b_kv_phys_d,
                offload_b_per_core=offload_b,
            )
            for c in ctx.scenario.concurrencies:
                for pref in prefill_options:
                    if not _valid_raiden_gdn_pair(ctx.geom, pref.par, par_d):
                        continue
                    candidates.extend(
                        _search_disagg_instance_ratios(ctx, dec_state, pref, c, cache),
                    )
    return candidates


def solve_serving_plan(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    *args: object,
    **kwargs: object,
) -> ServingPlan:
    """Search single-pool mixed and unconstrained xP/yD disagg plans."""
    arg_iter = iter(args)
    mode_val = next(arg_iter, kwargs.get("mode", "auto"))
    mode = str(mode_val)
    max_chips_val = next(arg_iter, kwargs.get("max_chips"))
    max_chips = (
        int(max_chips_val) if isinstance(max_chips_val, (int, float, str)) else None
    )
    max_inst_val = kwargs.get("max_instances")
    max_instances = (
        int(max_inst_val) if isinstance(max_inst_val, (int, float, str)) else None
    )
    gpu_util_val = kwargs.get("gpu_memory_utilization", DEFAULT_GPU_MEMORY_UTILIZATION)
    gpu_memory_utilization = (
        float(gpu_util_val)
        if isinstance(gpu_util_val, (int, float, str))
        else DEFAULT_GPU_MEMORY_UTILIZATION
    )

    if not (0.0 < gpu_memory_utilization <= 1.0):
        msg = f"gpu_memory_utilization must be in (0, 1], got {gpu_memory_utilization}"
        raise ValueError(msg)
    hbm_limit_b = float(gpu_memory_utilization) * float(chip.hbm_bytes)
    eff_al = max(1.0, scenario.accept_len)
    spec_k = (
        scenario.spec_tokens
        if scenario.spec_tokens is not None
        else (max(geom.mtp_layers, math.ceil(eff_al) - 1) if eff_al > 1.0 else 0)
    )
    ctx_full = scenario.isl + scenario.osl
    eff_isl = max(1, math.ceil((1.0 - scenario.prefix_hit_rate) * scenario.isl))
    n_chunks = max(1, math.ceil(eff_isl / max(1, scenario.prefill_chunk)))
    page_size = (
        scenario.page_size
        if scenario.page_size is not None
        else _default_rpa_page_size(ctx_full)
    )

    ctx = _SolveContext(
        geom=geom,
        dtypes=dtypes,
        chip=chip,
        scenario=scenario,
        mode=mode,
        max_chips=max_chips,
        max_instances=max_instances,
        gpu_memory_utilization=gpu_memory_utilization,
        hbm_limit_b=hbm_limit_b,
        hbm_limit_gb=hbm_limit_b / 1e9,
        eff_al=eff_al,
        spec_k=spec_k,
        ctx_full=ctx_full,
        ctx_mid=scenario.isl + scenario.osl // 2,
        eff_isl=eff_isl,
        n_chunks=n_chunks,
        page_size=page_size,
        c_max=max(scenario.concurrencies),
    )

    topos = [
        t for t in TOPOLOGIES.values() if max_chips is None or t.num_chips <= max_chips
    ]
    prefill_options, min_prefill_ttft_ms = _find_prefill_options(ctx, topos)
    mixed_candidates, min_c1_tpot_ms, min_mixed_ttft_ms = _evaluate_mixed_candidates(
        ctx,
        topos,
    )
    disagg_candidates = _evaluate_disaggregated_candidates(ctx, topos, prefill_options)
    candidates = mixed_candidates + disagg_candidates

    warnings: list[str] = []
    min_req_ttft_ms = min(min_prefill_ttft_ms, min_mixed_ttft_ms)
    if (
        scenario.tpot_slo_ms is not None
        and math.isfinite(min_c1_tpot_ms)
        and scenario.tpot_slo_ms < min_c1_tpot_ms
    ):
        warnings.append(
            f"INFEASIBLE_SLO: tpot_slo_ms={scenario.tpot_slo_ms:.2f} < "
            f"single-request floor {min_c1_tpot_ms:.2f} ms",
        )
    if (
        scenario.ttft_slo_ms is not None
        and math.isfinite(min_req_ttft_ms)
        and scenario.ttft_slo_ms < min_req_ttft_ms
    ):
        warnings.append(
            f"INFEASIBLE_SLO: ttft_slo_ms={scenario.ttft_slo_ms:.2f} < "
            f"single-request prefill floor {min_req_ttft_ms:.2f} ms",
        )

    slo_candidates = [p for p in candidates if p.slo_met]
    pool_for_frontier = slo_candidates if slo_candidates else candidates
    if candidates and not slo_candidates and not warnings:
        warnings.append("SLO_UNMET: no feasible topology met both TTFT and TPOT bounds")

    frontier = _pareto_filter(pool_for_frontier)
    recommended = (
        max(frontier, key=lambda p: p.token_throughput_per_chip) if frontier else None
    )
    pruned_sweep = _prune_sweep_candidates(geom, dtypes, chip, scenario, candidates)
    base_plan = ServingPlan(
        model_layers=geom.num_layers,
        chip=chip.name,
        scenario=scenario,
        mode=mode,
        recommended=recommended,
        frontier=frontier,
        warnings=tuple(warnings),
        candidates=tuple(pool_for_frontier),
        gpu_memory_utilization=gpu_memory_utilization,
        search=SearchStats(
            enumerated=ctx.enumerated,
            hbm_fit=ctx.hbm_fit,
            slo_met=len(slo_candidates),
            frontier=len(frontier),
        ),
        sweep_candidates=pruned_sweep,
        spec_tokens=spec_k,
        mtp_layers=geom.mtp_layers,
    )
    win_range, cross_rate = sweep_prefix_hit_crossovers(geom, dtypes, chip, base_plan)
    return replace(
        base_plan,
        winner_prefix_hit_range=win_range,
        prefix_hit_crossover_rate=cross_rate,
    )


def _pool_sig(pool: PoolMeshPlan | None) -> tuple | None:
    """Return canonical tuple signature for a pool mesh plan."""
    if pool is None:
        return None
    return (
        pool.topology,
        pool.instances,
        pool.tp,
        pool.dp,
        pool.ep,
        pool.pp,
        pool.pcp,
        pool.kv_split,
        pool.moe_dispatch,
        pool.pp_layer_partition,
    )


def _layout_sig(pt: ParetoPoint) -> tuple:
    """Return canonical layout signature for a candidate point."""
    return (pt.mode, _pool_sig(pt.prefill_pool), _pool_sig(pt.decode_pool))


def _pool_parallel(pool: PoolMeshPlan) -> tuple[SliceTopology, Parallel]:
    """Construct SliceTopology and Parallel objects from a pool mesh plan."""
    topo = TOPOLOGIES[pool.topology]
    moe_m: Literal["tp_slice", "ep"] = "ep" if pool.ep > 1 else "tp_slice"
    mesh_axes = (
        (
            pool.mesh_axes[0],
            pool.mesh_axes[1],
            pool.mesh_axes[2],
            pool.mesh_axes[3],
        )
        if len(pool.mesh_axes) == _MESH_AXES_LEN
        else None
    )
    return (
        topo,
        Parallel(
            tp=pool.tp,
            dp=pool.dp,
            ep=pool.ep,
            pp=pool.pp,
            pcp=pool.pcp,
            kv_split=pool.kv_split,
            moe=moe_m,
            moe_dispatch=pool.moe_dispatch,
            ranks_per_chip=topo.cores_per_chip,
            mesh_axes=mesh_axes,
        ),
    )


def _calc_sweep_kv_ratios(
    pt: ParetoPoint,
    scen: TrafficScenario,
    b_local: int,
    d_pool: PoolMeshPlan,
    _get_b_kv: Callable[[PoolMeshPlan], float],
) -> tuple[float, float, float]:
    """Calculate KV cache capacity ratio, HBM ratio, and PCIe slope for sweep point."""
    cap_tok = pt.kv_cache_capacity_tokens
    ctx_full = scen.isl + scen.osl
    w_turns = max(0.0, scen.kv_working_set_turns)
    need_ttft = scen.think_time_s > 0.0 or (scen.ttft_slo_ms is not None)
    has_offload = scen.kv_offload_gb > 0.0
    offload_bytes = float(scen.kv_offload_gb) * float(1 << 30)

    if w_turns > 0.0 and cap_tok is not None:
        w_tok = w_turns * float(b_local * ctx_full)
        cap_t = float(cap_tok)
        r_c = cap_t / max(1.0, w_tok)
        cap_ratio = min(1.0, r_c)
        if has_offload and need_ttft:
            b_kv = _get_b_kv(d_pool)
            tp_pp = d_pool.tp * d_pool.pp
            off_tok = (
                (offload_bytes / float(max(1, tp_pp))) / b_kv if b_kv > 0.0 else 0.0
            )
            hbm_cap_t = max(0.0, cap_t - off_tok)
            r_h = hbm_cap_t / max(1.0, w_tok)
            hbm_ratio = min(1.0, r_h)
            pcie_k = b_kv if cap_ratio > hbm_ratio else 0.0
            return (cap_ratio, hbm_ratio, pcie_k)
        return (cap_ratio, cap_ratio, 0.0)
    return (1.0, 1.0, 0.0)


def _eval_mixed_sweep_cand(
    pt: ParetoPoint,
    scen: TrafficScenario,
    _get_b_kv: Callable[[PoolMeshPlan], float],
) -> tuple[tuple, float, float]:
    """Evaluate mixed serving candidate constraints and sort keys for sweep pruning."""
    d_pool = pt.decode_pool
    d_dp = max(1, d_pool.dp)
    d_idp = d_pool.instances * d_dp
    b_local = max(1, math.ceil(pt.concurrency / max(1, d_idp)))
    b_res = d_pool.max_num_seqs if d_pool.max_num_seqs is not None else b_local
    cap_tok = pt.kv_cache_capacity_tokens
    ctx_full = scen.isl + scen.osl
    w_turns = max(0.0, scen.kv_working_set_turns)
    need_ttft = scen.think_time_s > 0.0 or (scen.ttft_slo_ms is not None)
    has_offload = scen.kv_offload_gb > 0.0
    offload_bytes = float(scen.kv_offload_gb) * float(1 << 30)

    if w_turns > 0.0 and cap_tok is not None:
        w_tok = w_turns * float(b_local * ctx_full)
        cap_t = float(cap_tok)
        r_c = cap_t / max(1.0, w_tok)
        cap_ratio = min(1.0, r_c)
        if has_offload and need_ttft:
            b_kv = _get_b_kv(d_pool)
            tp_pp = d_pool.tp * d_pool.pp
            off_tok = (
                (offload_bytes / float(max(1, tp_pp))) / b_kv if b_kv > 0.0 else 0.0
            )
            hbm_cap_t = max(0.0, cap_t - off_tok)
            r_h = hbm_cap_t / max(1.0, w_tok)
            hbm_ratio = min(1.0, r_h)
            pcie_k = b_kv if cap_ratio > hbm_ratio else 0.0
        else:
            hbm_ratio = cap_ratio
            pcie_k = 0.0
    else:
        cap_ratio = 1.0
        hbm_ratio = 1.0
        pcie_k = 0.0

    res_cap = float(b_res * d_dp)
    conc_f = float(pt.concurrency)
    c_srv = min(conc_f, res_cap)
    if scen.ttft_slo_ms is not None:
        dem_cap = float(b_local * d_dp)
        c_dem = min(conc_f, dem_cap)
        over = max(0.0, (c_dem / max(1e-9, res_cap)) - 1.0)
        mix_pen = 1 if pt.concurrency > 1 else 0
    else:
        over = 0.0
        mix_pen = 0

    mkey = (
        d_pool.topology,
        d_pool.instances,
        d_pool.tp,
        d_pool.dp,
        d_pool.ep,
        d_pool.pp,
        d_pool.pcp,
        d_pool.kv_split,
        d_pool.moe_dispatch,
        d_pool.pp_layer_partition,
        pt.total_chips,
        b_res,
        cap_ratio,
        hbm_ratio,
        pcie_k,
        mix_pen,
    )
    primary = conc_f if scen.think_time_s > 0.0 else c_srv
    return (mkey, primary, over)


def _calc_disagg_ttft_dkey(
    pt: ParetoPoint,
    scen: TrafficScenario,
    chip: ChipSpec,
    timing: tuple[int, int, float, float],
    ratios: tuple[float, float, float, bool],
) -> tuple[tuple, float]:
    """Calculate composite key and concurrency capacity for disaggregated candidate."""
    b_local, d_idp, dec_dur_s, res_cap_d = timing
    cap_ratio, hbm_ratio, pcie_k, tpot_ok = ratios
    p_pool = pt.prefill_pool
    if p_pool is None:
        msg = "Disaggregated point missing prefill_pool"
        raise RuntimeError(msg)
    d_pool = pt.decode_pool
    conc_f = float(pt.concurrency)
    chips_f = float(pt.total_chips)
    seq_tok = float(scen.isl + scen.osl)
    c_cap = conc_f * seq_tok / chips_f if scen.think_time_s > 0.0 else 0.0
    if scen.ttft_slo_ms is not None:
        dem_cap_d = float(b_local * d_idp)
        c_dem_d = min(conc_f, dem_cap_d)
        q_wait_d_ms = max(0.0, (c_dem_d / max(1e-9, res_cap_d)) - 1.0) * dec_dur_s * 1e3
    else:
        q_wait_d_ms = 0.0
    think_dur_k = dec_dur_s if scen.think_time_s > 0.0 else 0.0
    need_dcn_split = chip.dcn_bw > 0.0 and math.isfinite(chip.dcn_bw)
    min_c = (
        max(
            1,
            min(
                TOPOLOGIES[p_pool.topology].num_chips,
                TOPOLOGIES[d_pool.topology].num_chips,
            ),
        )
        if need_dcn_split
        else 0
    )
    dkey = (
        p_pool.topology,
        p_pool.tp,
        p_pool.dp,
        p_pool.ep,
        p_pool.pp,
        p_pool.pcp,
        p_pool.kv_split,
        p_pool.moe_dispatch,
        p_pool.pp_layer_partition,
        min_c,
        cap_ratio,
        hbm_ratio,
        pcie_k,
        tpot_ok,
        q_wait_d_ms,
        think_dur_k,
    )
    return (dkey, c_cap)


def _eval_disagg_sweep_cand(
    pt: ParetoPoint,
    scen: TrafficScenario,
    chip: ChipSpec,
    _get_b_kv: Callable[[PoolMeshPlan], float],
) -> tuple[tuple, float, float, float, float]:
    """Evaluate disaggregated candidate constraints and sort keys for sweep pruning."""
    p_pool = pt.prefill_pool
    if p_pool is None:
        msg = "Disaggregated point missing prefill_pool"
        raise RuntimeError(msg)
    d_pool = pt.decode_pool
    d_dp = max(1, d_pool.dp)
    d_idp = max(1, d_pool.instances * d_dp)
    b_res = (
        d_pool.max_num_seqs
        if d_pool.max_num_seqs is not None
        else max(1, math.ceil(pt.concurrency / d_idp))
    )
    b_local = max(1, math.ceil(pt.concurrency / d_idp))
    cap_ratio, hbm_ratio, pcie_k = _calc_sweep_kv_ratios(
        pt, scen, b_local, d_pool, _get_b_kv
    )
    ctx_full = scen.isl + scen.osl

    eff_al = max(1.0, scen.accept_len)
    sf_dec_s = pt.p90_tpot_ms * (eff_al * 1e-3)
    dec_dur_s = max(1.0, scen.osl / eff_al) * sf_dec_s
    res_cap_d = float(b_res * d_idp)
    conc_f = float(pt.concurrency)
    rate_d = min(conc_f, res_cap_d) / max(1e-09, dec_dur_s)
    tpot_h = sf_dec_s * (1e3 / eff_al)
    inter_h = 1000.0 / max(1e-06, tpot_h)
    tpot_ok = scen.tpot_slo_ms is None or tpot_h <= scen.tpot_slo_ms
    chips_f = float(pt.total_chips)
    seq_tok = float(ctx_full)
    p_cap = float(p_pool.instances) * seq_tok / chips_f
    tput_d_cap = rate_d * seq_tok / chips_f

    need_ttft = scen.think_time_s > 0.0 or (scen.ttft_slo_ms is not None)
    if not need_ttft:
        c_cap = 0.0
        dkey: tuple = (
            p_pool.topology,
            p_pool.tp,
            p_pool.dp,
            p_pool.ep,
            p_pool.pp,
            p_pool.pcp,
            p_pool.kv_split,
            p_pool.moe_dispatch,
            p_pool.pp_layer_partition,
            cap_ratio,
            tpot_ok,
        )
    else:
        dkey, c_cap = _calc_disagg_ttft_dkey(
            pt,
            scen,
            chip,
            (b_local, d_idp, dec_dur_s, res_cap_d),
            (cap_ratio, hbm_ratio, pcie_k, tpot_ok),
        )
    return (dkey, p_cap, tput_d_cap, c_cap, inter_h)


def _filter_mixed_skyline(
    mixed_groups: dict,
) -> list[tuple[int, ParetoPoint]]:
    """Filter mixed candidates by minimum latency overhead."""
    kept: list[tuple[int, ParetoPoint]] = []
    for m_items in mixed_groups.values():
        m_items.sort()
        min_over = math.inf
        for _neg_prim, over, idx, pt in m_items:
            if over < min_over:
                min_over = over
                kept.append((idx, pt))
    return kept


def _filter_disagg_2d_skyline(
    d_items: list,
) -> list[tuple[int, ParetoPoint]]:
    """Filter disaggregated candidates across (prefill, decode, inter) skyline."""
    strict_band = 1.0 + 1e-6
    skyline_2d: list[tuple[int, float, float, float]] = []
    kept: list[tuple[int, ParetoPoint]] = []
    s_ptr = 0
    max_dk_strict = -1.0
    for neg_p, neg_d, _neg_c, neg_i, chips, idx, pt in d_items:
        p_cap = -neg_p
        tput_d = -neg_d
        inter = -neg_i
        p_req = p_cap * strict_band
        n_sky = len(skyline_2d)
        while s_ptr < n_sky and skyline_2d[s_ptr][1] > p_req:
            dk_s = skyline_2d[s_ptr][2]
            max_dk_strict = max(max_dk_strict, dk_s)
            s_ptr += 1
        if max_dk_strict > tput_d * strict_band:
            continue
        start_k = 0 if max_dk_strict >= tput_d else s_ptr
        dominated = False
        for k in range(start_k, n_sky):
            ck, pk, dk, ik = skyline_2d[k]
            if pk >= p_cap and dk >= tput_d and ik >= inter and ck <= chips:
                dominated = True
                break
        if not dominated:
            kept.append((idx, pt))
            skyline_2d.append((chips, p_cap, tput_d, inter))
    return kept


def _filter_disagg_3d_skyline(
    d_items: list,
) -> list[tuple[int, ParetoPoint]]:
    """Filter disaggregated candidates across 4D skyline."""
    skyline_3d: list[tuple[int, float, float, float, float]] = []
    kept: list[tuple[int, ParetoPoint]] = []
    for neg_p, neg_d, neg_c, neg_i, chips, idx, pt in d_items:
        p_cap = -neg_p
        tput_d = -neg_d
        c_cap = -neg_c
        inter = -neg_i
        dominated = False
        for ck, pk, dk, c_k, ik in skyline_3d:
            if (
                pk >= p_cap
                and dk >= tput_d
                and c_k >= c_cap
                and ik >= inter
                and ck <= chips
            ):
                dominated = True
                break
        if not dominated:
            kept.append((idx, pt))
            skyline_3d.append((chips, p_cap, tput_d, c_cap, inter))
    return kept


def _filter_sweep_skyline(
    mixed_groups: dict,
    disagg_groups: dict,
    *,
    has_think: bool,
) -> tuple[ParetoPoint, ...]:
    """Filter candidate groups using multi-dimensional skyline."""
    kept = _filter_mixed_skyline(mixed_groups)
    for d_items in disagg_groups.values():
        d_items.sort()
        if not has_think:
            kept.extend(_filter_disagg_2d_skyline(d_items))
        else:
            kept.extend(_filter_disagg_3d_skyline(d_items))
    kept.sort()
    return tuple(pt for _idx, pt in kept)


def _prune_sweep_candidates(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scen: TrafficScenario,
    cands: list[ParetoPoint] | tuple[ParetoPoint, ...],
) -> tuple[ParetoPoint, ...]:
    """Prune candidates strictly dominated across every prefix hit rate h in [0, 1]."""
    if len(cands) <= 1:
        return tuple(cands)
    kv_b_cache: dict[tuple[int, int, int, int], float] = {}

    def _get_b_kv(pool: PoolMeshPlan) -> float:
        """Calculate single-token KV cache bytes per core for a pool."""
        k = (pool.tp, pool.pp, pool.pcp, pool.kv_split)
        val = kv_b_cache.get(k)
        if val is None:
            par_k = Parallel(
                tp=pool.tp, pp=pool.pp, pcp=pool.pcp, kv_split=pool.kv_split
            )
            val = hbm_footprint_per_core_bytes(
                geom, dtypes, par_k, 1, 1
            ) - hbm_footprint_per_core_bytes(geom, dtypes, par_k, 1, 0)
            kv_b_cache[k] = val
        return val

    mixed_groups: dict[tuple, list[tuple[float, float, int, ParetoPoint]]] = {}
    disagg_groups: dict[
        tuple, list[tuple[float, float, float, float, int, int, ParetoPoint]]
    ] = {}
    for idx, pt in enumerate(cands):
        if pt.mode == "mixed":
            mkey, primary, over = _eval_mixed_sweep_cand(pt, scen, _get_b_kv)
            mixed_groups.setdefault(mkey, []).append((-primary, over, idx, pt))
        else:
            dkey, p_cap, tput_d, c_cap, inter = _eval_disagg_sweep_cand(
                pt, scen, chip, _get_b_kv
            )
            disagg_groups.setdefault(dkey, []).append(
                (-p_cap, -tput_d, -c_cap, -inter, pt.total_chips, idx, pt)
            )

    return _filter_sweep_skyline(
        mixed_groups, disagg_groups, has_think=scen.think_time_s > 0.0
    )


def _best_frontier_winner(
    scored: list[tuple[float, float, int, ParetoPoint]],
) -> ParetoPoint:
    """Pick best throughput winner from Pareto candidates."""
    ordered = sorted(scored, key=lambda x: (-x[1], -x[0], -x[2]))
    best_tput = -1.0
    best_pt = ordered[0][3]
    for tput, _inter, _neg_c, pt in ordered:
        if tput > best_tput * (1.0 + 1e-9):
            best_tput = tput
            best_pt = pt
    return best_pt


@dataclass(frozen=True, slots=True)
class _GridEvalContext:
    """Context and memoized caches for prefix hit rate sweep evaluation."""

    pre_cache: dict[tuple[int, float], float]
    mix_cache: dict[tuple[int, int, int], float]
    pool_plans: list[tuple[SliceTopology | None, Parallel, tuple[int, ...] | None]]
    compute_pre_sf: Callable[[int, PoolMeshPlan, float], float]
    compute_mix_sf: Callable[[int, int, int], float]
    get_wire_b: Callable[[int], float]
    need_ttft: bool
    need_wire: bool
    prefill_chunk: int
    eff_al_s: float
    inv_al_ms: float
    dec_steps: float
    think_s: float
    has_think: bool
    ttft_slo: float | None
    tpot_slo: float | None


class _CandidateState:
    """Accumulator for Pareto candidates evaluated at a specific prefix hit rate."""

    def __init__(self) -> None:
        """Initialize empty candidate state."""
        self.near_slo: list[tuple[float, float, int, ParetoPoint]] = []
        self.near_all: list[tuple[float, float, int, ParetoPoint]] = []
        self.max_slo: float = -1.0
        self.max_all: float = -1.0


def _calc_mix_ttft_and_rate(
    d_pid: int,
    d_pool: PoolMeshPlan,
    h_and_pcie: tuple[float, float],
    mix_state: tuple[
        float, float, tuple[float, float, float, float, bool, float], _GridEvalContext
    ],
) -> tuple[float, float]:
    """Calculate mixed candidate TTFT and served throughput rate."""
    h_eff, pcie_s = h_and_pcie
    denom_w, over, mix_info, ctx = mix_state
    dec_dur_s, sf_mix_s, res_cap, c_srv, has_c_gt1, conc_f = mix_info
    if not ctx.need_ttft:
        return (0.0, c_srv / denom_w)
    sf_pre_s = ctx.pre_cache.get((d_pid, h_eff))
    if sf_pre_s is None:
        sf_pre_s = ctx.compute_pre_sf(d_pid, d_pool, h_eff)
    pre_ttft_s = sf_pre_s + pcie_s
    w_srv_final = pre_ttft_s + dec_dur_s
    if ctx.has_think:
        wt = w_srv_final + ctx.think_s
        rate_s = min(res_cap / denom_w, conc_f / max(1e-09, wt))
    else:
        rate_s = c_srv / denom_w
    ttft_h = (pre_ttft_s + over * w_srv_final) * 1e3 + (
        sf_mix_s * 1e3 if has_c_gt1 else 0.0
    )
    return (ttft_h, rate_s)


def _eval_grid_mix_candidates(
    h: float,
    scen: TrafficScenario,
    mix_meta: list,
    ctx: _GridEvalContext,
    state: _CandidateState,
) -> None:
    """Evaluate mixed candidate throughput and latency under prefix hit rate h."""
    tol_band = 1.0 + 1e-7
    eff_isl_1 = max(1, math.ceil((1.0 - h) * scen.isl))
    n_chk_1 = max(1, math.ceil(eff_isl_1 / ctx.prefill_chunk))
    chunk_tok_1 = min(ctx.prefill_chunk, eff_isl_1)

    for (
        pt,
        d_pool,
        d_pid,
        b_res,
        b_res_f,
        cap_ratio,
        pcie_slope,
        sf_dec_s,
        dec_dur_s,
        draft_s,
        res_cap,
        c_srv,
        conc_f,
        has_c_gt1,
        over,
        inv_chips,
        neg_chips,
    ) in mix_meta:
        if cap_ratio == 1.0 or h == 0.0:
            h_eff = h
            n_chk = n_chk_1
            chunk_tok_c = chunk_tok_1
        else:
            h_eff = h * cap_ratio
            eff_isl_h = max(1, math.ceil((1.0 - h_eff) * scen.isl))
            n_chk = max(1, math.ceil(eff_isl_h / ctx.prefill_chunk))
            chunk_tok_c = min(ctx.prefill_chunk, eff_isl_h)
        pcie_s = h * pcie_slope if pcie_slope > 0.0 else 0.0
        sf_mix_s = ctx.mix_cache.get((d_pid, b_res, chunk_tok_c))
        if sf_mix_s is None:
            sf_mix_s = ctx.compute_mix_sf(d_pid, b_res, chunk_tok_c)
        mix_serial_s = sf_mix_s + draft_s
        bn = b_res_f * float(n_chk)
        tot_work_s = dec_dur_s + bn * mix_serial_s
        denom_w = max(1e-09, tot_work_s)
        ttft_h, rate_s = _calc_mix_ttft_and_rate(
            d_pid,
            d_pool,
            (h_eff, pcie_s),
            (
                denom_w,
                over,
                (dec_dur_s, sf_mix_s, res_cap, c_srv, has_c_gt1, conc_f),
                ctx,
            ),
        )
        denom_pm = ctx.dec_steps + bn
        p_mix = min(1.0, bn / max(1.0, denom_pm))
        p90_step_s = (1.0 - p_mix) * sf_dec_s + p_mix * mix_serial_s
        tpot_h = p90_step_s * ctx.inv_al_ms
        inter_h = 1000.0 / max(1e-06, tpot_h)
        tput = rate_s * inv_chips
        slo_ok = (ctx.ttft_slo is None or ttft_h <= ctx.ttft_slo) and (
            ctx.tpot_slo is None or tpot_h <= ctx.tpot_slo
        )
        if slo_ok:
            if tput * tol_band >= state.max_slo:
                state.max_slo = max(state.max_slo, tput)
                state.near_slo.append((tput, inter_h, neg_chips, pt))
        elif not state.near_slo and tput * tol_band >= state.max_all:
            state.max_all = max(state.max_all, tput)
            state.near_all.append((tput, inter_h, neg_chips, pt))


def _eval_grid_disagg_fast(
    h: float,
    dis_fast_groups: dict,
    ctx: _GridEvalContext,
    state: _CandidateState,
) -> None:
    """Evaluate fast disaggregated candidate throughput and latency under hit rate h."""
    tol_band = 1.0 + 1e-7
    for (p_pid, cap_ratio), (p_pool, grp_f) in dis_fast_groups.items():
        h_eff = h if (cap_ratio == 1.0 or h == 0.0) else h * cap_ratio
        sf_pre_s = ctx.pre_cache.get((p_pid, h_eff))
        if sf_pre_s is None:
            sf_pre_s = ctx.compute_pre_sf(p_pid, p_pool, h_eff)
        denom_pre = max(1e-09, sf_pre_s)
        for p_inst_f, rate_d, inv_chips, tpot_ok, inter_d, neg_chips, pt in grp_f:
            rate_p = p_inst_f / denom_pre
            tput = min(rate_p, rate_d) * inv_chips
            if tpot_ok:
                if tput * tol_band >= state.max_slo:
                    state.max_slo = max(state.max_slo, tput)
                    state.near_slo.append((tput, inter_d, neg_chips, pt))
            elif not state.near_slo and tput * tol_band >= state.max_all:
                state.max_all = max(state.max_all, tput)
                state.near_all.append((tput, inter_d, neg_chips, pt))


def _eval_grid_disagg_ttft(
    h: float,
    scen: TrafficScenario,
    dis_ttft_groups: dict,
    ctx: _GridEvalContext,
    state: _CandidateState,
) -> None:
    """Evaluate TTFT-constrained disaggregated candidates under prefix hit rate h."""
    tol_band = 1.0 + 1e-7
    eff_isl_1 = max(1, math.ceil((1.0 - h) * scen.isl))
    n_chk_1 = max(1, math.ceil(eff_isl_1 / ctx.prefill_chunk))
    wire_b_1 = ctx.get_wire_b(eff_isl_1) if ctx.need_wire else 0.0

    for (p_pid, cap_ratio), (p_pool, bub_num, grp_t) in dis_ttft_groups.items():
        if cap_ratio == 1.0 or h == 0.0:
            h_eff = h
            n_chk = n_chk_1
            wire_b_h = wire_b_1
        else:
            h_eff = h * cap_ratio
            eff_isl_h = max(1, math.ceil((1.0 - h_eff) * scen.isl))
            n_chk = max(1, math.ceil(eff_isl_h / ctx.prefill_chunk))
            wire_b_h = ctx.get_wire_b(eff_isl_h) if ctx.need_wire else 0.0
        sf_pre_s = ctx.pre_cache.get((p_pid, h_eff))
        if sf_pre_s is None:
            sf_pre_s = ctx.compute_pre_sf(p_pid, p_pool, h_eff)
        denom_pre = max(1e-09, sf_pre_s)
        bub = 1.0 + bub_num / float(n_chk) if bub_num > 0.0 else 1.0
        pre_bub_s = sf_pre_s * bub
        pre_bub_ms = pre_bub_s * 1e3
        for (
            p_inst_f,
            rate_d,
            inv_chips,
            tpot_ok,
            inter_d,
            neg_chips,
            pcie_slope,
            inv_min_c_dcn_ms,
            q_wait_d_ms,
            dec_dur_s,
            conc_f,
            pt,
        ) in grp_t:
            rate_p = p_inst_f / denom_pre
            coupled_cap = min(rate_p, rate_d)
            pcie_s = h * pcie_slope if pcie_slope > 0.0 else 0.0
            xfer_ms = wire_b_h * inv_min_c_dcn_ms
            ttft_h = pre_bub_ms + pcie_s * 1e3 + xfer_ms + q_wait_d_ms
            if ctx.has_think:
                wt_d = pre_bub_s + pcie_s + xfer_ms * 1e-3 + dec_dur_s + ctx.think_s
                c_rate_d = conc_f / max(1e-09, wt_d)
                coupled = min(coupled_cap, c_rate_d)
            else:
                coupled = coupled_cap
            tput = coupled * inv_chips
            slo_ok = tpot_ok and (ctx.ttft_slo is None or ttft_h <= ctx.ttft_slo)
            if slo_ok:
                if tput * tol_band >= state.max_slo:
                    state.max_slo = max(state.max_slo, tput)
                    state.near_slo.append((tput, inter_d, neg_chips, pt))
            elif not state.near_slo and tput * tol_band >= state.max_all:
                state.max_all = max(state.max_all, tput)
                state.near_all.append((tput, inter_d, neg_chips, pt))


def _eval_grid_h(
    h: float,
    scen: TrafficScenario,
    pools: tuple[list, dict, dict],
    ctx: _GridEvalContext,
) -> tuple[float, tuple] | None:
    """Evaluate candidate serving pools at a prefix hit rate and return best layout."""
    mix_meta, dis_fast_groups, dis_ttft_groups = pools
    state = _CandidateState()
    _eval_grid_mix_candidates(h, scen, mix_meta, ctx, state)
    if not ctx.need_ttft:
        _eval_grid_disagg_fast(h, dis_fast_groups, ctx, state)
    else:
        _eval_grid_disagg_ttft(h, scen, dis_ttft_groups, ctx, state)
    tol_band = 1.0 + 1e-7
    pool_h = (
        [x for x in state.near_slo if x[0] * tol_band >= state.max_slo]
        if state.near_slo
        else [x for x in state.near_all if x[0] * tol_band >= state.max_all]
    )
    if not pool_h:
        return None
    best_pt = _best_frontier_winner(pool_h)
    return (h, _layout_sig(best_pt))


def _derive_winner_range_and_crossover(
    scen: TrafficScenario,
    plan: ServingPlan,
    winners: list[tuple[float, tuple]],
) -> tuple[tuple[float, float] | None, float | None]:
    """Derive recommended winner hit rate range and closest crossover point."""
    if plan.recommended is None:
        return (None, None)
    h0 = round(scen.prefix_hit_rate, 4)
    idx0 = min(range(len(winners)), key=lambda i: abs(winners[i][0] - h0))
    sig0 = _layout_sig(plan.recommended)
    winners[idx0] = (winners[idx0][0], sig0)
    lo = idx0
    while lo > 0 and winners[lo - 1][1] == sig0:
        lo -= 1
    hi = idx0
    while hi + 1 < len(winners) and winners[hi + 1][1] == sig0:
        hi += 1
    win_range = (winners[lo][0], winners[hi][0])
    cross: float | None = None
    best_dist = math.inf
    for i in range(len(winners) - 1):
        if winners[i][1] != winners[i + 1][1]:
            mid = round(0.5 * (winners[i][0] + winners[i + 1][0]), 4)
            dist = abs(mid - h0)
            if dist < best_dist:
                best_dist = dist
                cross = mid
    return (win_range, cross)


def _setup_crossover_mixed_cand(
    pt: ParetoPoint,
    scen_chip: tuple[TrafficScenario, ChipSpec],
    d_pid: int,
    helpers: tuple,
    mix_meta: list,
) -> None:
    """Precompute step floors and metadata for a mixed candidate in crossover sweep."""
    scen, chip = scen_chip
    d_pool = pt.decode_pool
    d_dp = max(1, d_pool.dp)
    d_idp = d_pool.instances * d_dp
    b_local = max(1, math.ceil(pt.concurrency / max(1, d_idp)))
    b_res = d_pool.max_num_seqs if d_pool.max_num_seqs is not None else b_local
    get_b_kv, get_dec_sf = helpers
    cap_ratio, _, pcie_k = _calc_sweep_kv_ratios(pt, scen, b_local, d_pool, get_b_kv)
    pcie_slope = 0.0
    if pcie_k > 0.0 and chip.pcie_bw > 0.0 and math.isfinite(chip.pcie_bw):
        pcie_core_bw = chip.pcie_bw / float(max(1, chip.cores_per_chip))
        pcie_slope = pcie_k / pcie_core_bw
    sf_dec_s, draft_s = get_dec_sf(d_pid, b_res)
    eff_al = max(1.0, scen.accept_len)
    dec_dur_s = max(1.0, scen.osl / eff_al) * sf_dec_s
    res_cap = float(b_res * d_dp)
    conc_f = float(pt.concurrency)
    c_srv = min(conc_f, res_cap)
    if scen.ttft_slo_ms is not None:
        dem_cap = float(b_local * d_dp)
        c_dem = min(conc_f, dem_cap)
        over = max(0.0, (c_dem / max(1e-9, res_cap)) - 1.0)
    else:
        over = 0.0
    inv_chips = 1.0 / float(pt.total_chips)
    mix_meta.append(
        (
            pt,
            d_pool,
            d_pid,
            b_res,
            float(b_res),
            cap_ratio,
            pcie_slope,
            sf_dec_s,
            dec_dur_s,
            draft_s,
            res_cap,
            c_srv,
            conc_f,
            pt.concurrency > 1,
            over,
            inv_chips,
            -pt.total_chips,
        )
    )


def _setup_crossover_disagg_cand(
    pt: ParetoPoint,
    scen_chip: tuple[TrafficScenario, ChipSpec],
    d_pid: int,
    helpers: tuple,
    groups: tuple[dict, dict],
) -> None:
    """Precompute step floors and group entries for a disaggregated candidate."""
    scen, chip = scen_chip
    p_pool = pt.prefill_pool
    if p_pool is None:
        msg = "Disaggregated point missing prefill_pool"
        raise RuntimeError(msg)
    d_pool = pt.decode_pool
    d_dp = max(1, d_pool.dp)
    d_idp = max(1, d_pool.instances * d_dp)
    b_local = max(1, math.ceil(pt.concurrency / d_idp))
    b_res = d_pool.max_num_seqs if d_pool.max_num_seqs is not None else b_local
    get_b_kv, get_dec_sf, get_pool_id = helpers
    p_pid = get_pool_id(p_pool)
    cap_ratio, _, pcie_k = _calc_sweep_kv_ratios(pt, scen, b_local, d_pool, get_b_kv)
    pcie_slope = 0.0
    if pcie_k > 0.0 and chip.pcie_bw > 0.0 and math.isfinite(chip.pcie_bw):
        pcie_core_bw = chip.pcie_bw / float(max(1, chip.cores_per_chip))
        pcie_slope = pcie_k / pcie_core_bw
    sf_dec_s, _draft_s = get_dec_sf(d_pid, b_res)
    eff_al = max(1.0, scen.accept_len)
    inv_al_ms = 1e3 / eff_al
    dec_dur_s = max(1.0, scen.osl / eff_al) * sf_dec_s
    res_cap_d = float(b_res * d_idp)
    conc_f = float(pt.concurrency)
    rate_d = min(conc_f, res_cap_d) / max(1e-09, dec_dur_s)
    tpot_d = sf_dec_s * inv_al_ms
    inter_d = 1000.0 / max(1e-06, tpot_d)
    tpot_ok = scen.tpot_slo_ms is None or tpot_d <= scen.tpot_slo_ms
    p_inst_f = float(p_pool.instances)
    inv_chips = 1.0 / float(pt.total_chips)
    gkey = (p_pid, cap_ratio)
    dis_fast_groups, dis_ttft_groups = groups
    need_ttft = scen.think_time_s > 0.0 or (scen.ttft_slo_ms is not None)
    if not need_ttft:
        g_fast = dis_fast_groups.setdefault(gkey, (p_pool, []))
        g_fast[1].append(
            (p_inst_f, rate_d, inv_chips, tpot_ok, inter_d, -pt.total_chips, pt)
        )
    else:
        dem_cap_d = float(b_local * d_idp)
        over_d = (
            max(0.0, (min(conc_f, dem_cap_d) / max(1e-9, res_cap_d)) - 1.0)
            if scen.ttft_slo_ms is not None
            else 0.0
        )
        q_wait_d_ms = over_d * dec_dur_s * 1e3
        has_dcn = chip.dcn_bw > 0.0 and math.isfinite(chip.dcn_bw)
        min_c = float(max(1, min(p_pool.chips_per_instance, d_pool.chips_per_instance)))
        inv_min_c_dcn_ms = 1e3 / (min_c * chip.dcn_bw) if has_dcn else 0.0
        g_ttft = dis_ttft_groups.setdefault(gkey, (p_pool, float(p_pool.pp - 1), []))
        g_ttft[2].append(
            (
                p_inst_f,
                rate_d,
                inv_chips,
                tpot_ok,
                inter_d,
                -pt.total_chips,
                pcie_slope,
                inv_min_c_dcn_ms,
                q_wait_d_ms,
                dec_dur_s,
                conc_f,
                pt,
            )
        )


class _CrossoverPlanner:
    """Precompute candidate pool metadata and step floor caches for crossover sweep."""

    def __init__(
        self,
        geom: ModelGeometry,
        dtypes: Dtypes,
        chip: ChipSpec,
        scen: TrafficScenario,
    ) -> None:
        """Initialize cache containers and compute static parameters."""
        self.geom = geom
        self.dtypes = dtypes
        self.chip = chip
        self.scen = scen
        self.ctx_mid: int = scen.isl + scen.osl // 2
        self.eff_al = max(1.0, scen.accept_len)
        self.spec_k = (
            scen.spec_tokens
            if scen.spec_tokens is not None
            else (
                max(geom.mtp_layers, math.ceil(self.eff_al) - 1)
                if self.eff_al > 1.0
                else 0
            )
        )
        self.dec_steps = max(1.0, scen.osl / self.eff_al)
        has_dcn = chip.dcn_bw > 0.0 and math.isfinite(chip.dcn_bw)
        self.think_s = scen.think_time_s
        self.has_think = self.think_s > 0.0
        self.need_ttft = self.has_think or (scen.ttft_slo_ms is not None)
        self.need_wire = has_dcn and self.need_ttft

        self.pool_ids: dict[tuple, int] = {}
        self.pool_plans: list[
            tuple[SliceTopology | None, Parallel, tuple[int, ...] | None]
        ] = []
        self.pre_cache: dict[tuple[int, float], float] = {}
        self.pre_step_cache: dict[tuple[int, int, int, int], float] = {}
        self.dec_cache: dict[tuple[int, int], tuple[float, float]] = {}
        self.mix_cache: dict[tuple[int, int, int], float] = {}
        self.kv_b_cache: dict[tuple[int, int, int, int], float] = {}
        self._init_wire()

    def _init_wire(self) -> None:
        """Initialize KV cache wire parameters if DCN transfer is required."""
        geom = self.geom
        sw_w = geom.sliding_window
        self.sw_w = sw_w
        if not self.need_wire:
            self.wire_state_b = 0.0
            self.wire_kv_step1 = 0.0
            self.wire_kv_at_sw = 0.0
            self.wire_kv_post_sw = 0.0
            return
        par_w = Parallel(tp=1, pp=1)
        dtypes = self.dtypes
        bd_w0 = hbm_breakdown_per_core_bytes(
            geom, dtypes, par_w, 1, 0, physical_kv_layout=False
        )
        bd_w1 = hbm_breakdown_per_core_bytes(
            geom, dtypes, par_w, 1, 1, physical_kv_layout=False
        )
        self.wire_state_b, self.wire_kv_step1 = bd_w0.state, bd_w1.kv
        if sw_w > 0:
            bd_w_sw = hbm_breakdown_per_core_bytes(
                geom, dtypes, par_w, 1, sw_w, physical_kv_layout=False
            )
            bd_w_sw1 = hbm_breakdown_per_core_bytes(
                geom, dtypes, par_w, 1, sw_w + 1, physical_kv_layout=False
            )
            self.wire_kv_at_sw = bd_w_sw.kv
            self.wire_kv_post_sw = bd_w_sw1.kv - bd_w_sw.kv
        else:
            self.wire_kv_at_sw, self.wire_kv_post_sw = 0.0, 0.0

    def get_wire_b(self, uncached_tok: int) -> float:
        """Compute KV cache wire transfer bytes for uncached prefix tokens."""
        tok_k = max(0, uncached_tok)
        if self.sw_w > 0 and tok_k > self.sw_w:
            kv_w = self.wire_kv_at_sw + float(tok_k - self.sw_w) * self.wire_kv_post_sw
        else:
            kv_w = float(tok_k) * self.wire_kv_step1
        return self.wire_state_b + kv_w

    def get_b_kv(self, pool: PoolMeshPlan) -> float:
        """Calculate single-token KV cache bytes per core for a pool."""
        k = (pool.tp, pool.pp, pool.pcp, pool.kv_split)
        val = self.kv_b_cache.get(k)
        if val is None:
            par_k = Parallel(
                tp=pool.tp, pp=pool.pp, pcp=pool.pcp, kv_split=pool.kv_split
            )
            val = hbm_footprint_per_core_bytes(
                self.geom, self.dtypes, par_k, 1, 1
            ) - hbm_footprint_per_core_bytes(self.geom, self.dtypes, par_k, 1, 0)
            self.kv_b_cache[k] = val
        return val

    def get_pool_id(self, pool: PoolMeshPlan) -> int:
        """Retrieve or allocate unique integer index for a pool layout."""
        pk = (
            pool.topology,
            pool.tp,
            pool.dp,
            pool.ep,
            pool.pp,
            pool.pcp,
            pool.kv_split,
            pool.moe_dispatch,
            pool.mesh_axes,
            pool.pp_layer_partition,
        )
        pid = self.pool_ids.get(pk)
        if pid is None:
            pid = len(self.pool_plans)
            self.pool_ids[pk] = pid
            topo, par = _pool_parallel(pool)
            max_p = max(1, math.ceil(self.geom.num_layers / pool.pp))
            even_pp = (
                pool.pp <= 1
                or not pool.pp_layer_partition
                or max(pool.pp_layer_partition) == max_p
            )
            self.pool_plans.append(
                (topo, par, None if even_pp else pool.pp_layer_partition)
            )
        return pid

    def compute_pre_sf(self, pid: int, pool: PoolMeshPlan, h_val: float) -> float:
        """Calculate prefill step floor latency under prefix caching."""
        h_clamped = max(0.0, min(1.0, h_val))
        eff_tok = (
            max(1, math.ceil((1.0 - h_clamped) * self.scen.isl))
            if self.scen.isl > 0
            else 0
        )
        eff_pre = (
            max(
                1,
                math.ceil(self.scen.isl * max(0.0, 1.0 - h_clamped) / max(1, pool.pcp)),
            )
            if self.scen.isl > 0
            else 0
        )
        cached_pre = round(self.scen.isl * h_clamped) if self.scen.isl > 0 else 0
        skey = (pid, eff_tok, eff_pre, cached_pre)
        sf_s = self.pre_step_cache.get(skey)
        if sf_s is None:
            topo, par, pp_part = self.pool_plans[pid]
            shape = StepShape(
                batch=0,
                rows_per_seq=0,
                context=0,
                state_writes=0,
                prefill_tokens=self.scen.isl,
                prefix_hit_rate=h_clamped,
            )
            sf_obj = step_floor(
                self.geom, self.chip, par, shape, dtypes=self.dtypes, topology=topo
            )
            if par.ep > 1:
                sf_obj = _scale_ep_moe_floor(sf_obj, self.chip, self.geom, par, eff_pre)
            if par.pp > 1:
                sf_obj = _scale_pp_stage_floor(sf_obj, pp_part, self.chip, par, topo)
            sf_s = sf_obj.serial_floor_s
            self.pre_step_cache[skey] = sf_s
        return sf_s

    def get_dec_sf(self, pid: int, b_res: int) -> tuple[float, float]:
        """Retrieve or compute decode step floor latency."""
        key = (pid, b_res)
        val = self.dec_cache.get(key)
        if val is None:
            topo, par, pp_part = self.pool_plans[pid]
            if self.spec_k <= 0:
                sf = step_floor(
                    self.geom,
                    self.chip,
                    par,
                    StepShape(batch=b_res, rows_per_seq=1, context=self.ctx_mid),
                    dtypes=self.dtypes,
                    topology=topo,
                )
                if par.pp > 1:
                    sf = _scale_pp_stage_floor(sf, pp_part, self.chip, par, topo)
                val = (_scaled_ep_serial_s(sf, self.chip, self.geom, par, b_res), 0.0)
            else:
                sf_dec = spec_cycle_floor(
                    self.geom,
                    self.chip,
                    par,
                    b_res,
                    self.ctx_mid,
                    self.spec_k,
                    self.dtypes,
                    topology=topo,
                )
                sf_ver_only = step_floor(
                    self.geom,
                    self.chip,
                    par,
                    StepShape(
                        batch=b_res,
                        rows_per_seq=self.spec_k + 1,
                        context=self.ctx_mid,
                        state_writes=1,
                    ),
                    dtypes=self.dtypes,
                    topology=topo,
                )
                if par.pp > 1:
                    sf_dec = _scale_pp_stage_floor(
                        sf_dec, pp_part, self.chip, par, topo
                    )
                    sf_ver_only = _scale_pp_stage_floor(
                        sf_ver_only, pp_part, self.chip, par, topo
                    )
                if par.ep > 1:
                    tot_tok = b_res * (self.spec_k + 1)
                    sf_dec = _scale_ep_moe_floor(
                        sf_dec, self.chip, self.geom, par, tot_tok
                    )
                    sf_ver_only = _scale_ep_moe_floor(
                        sf_ver_only, self.chip, self.geom, par, tot_tok
                    )
                draft_s = max(0.0, sf_dec.serial_floor_s - sf_ver_only.serial_floor_s)
                val = (sf_dec.serial_floor_s, draft_s)
            self.dec_cache[key] = val
        return val

    def compute_mix_sf(self, pid: int, b_res: int, chunk_tok: int) -> float:
        """Compute mixed prefill-decode step floor latency."""
        key = (pid, b_res, chunk_tok)
        val = self.mix_cache.get(key)
        if val is None:
            topo, par, pp_part = self.pool_plans[pid]
            shape = StepShape(
                batch=b_res,
                rows_per_seq=self.spec_k + 1,
                context=self.ctx_mid,
                state_writes=1,
                prefill_tokens=chunk_tok,
                prefix_hit_rate=0.0,
            )
            sf = step_floor(
                self.geom, self.chip, par, shape, dtypes=self.dtypes, topology=topo
            )
            if par.pp > 1:
                sf = _scale_pp_stage_floor(sf, pp_part, self.chip, par, topo)
            val = _scaled_ep_serial_s(
                sf, self.chip, self.geom, par, b_res * (self.spec_k + 1) + chunk_tok
            )
            self.mix_cache[key] = val
        return val


def _setup_crossover_sweep(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scen: TrafficScenario,
    cands: tuple[ParetoPoint, ...],
) -> tuple[tuple[list, dict, dict], _GridEvalContext]:
    """Precompute candidate pool metadata and step floor caches for crossover sweep."""
    planner = _CrossoverPlanner(geom, dtypes, chip, scen)
    mix_meta: list = []
    dis_fast_groups: dict = {}
    dis_ttft_groups: dict = {}
    helpers = (planner.get_b_kv, planner.get_dec_sf, planner.get_pool_id)
    scen_chip = (scen, chip)

    for pt in cands:
        d_pool = pt.decode_pool
        d_pid = planner.get_pool_id(d_pool)
        if pt.mode == "mixed":
            _setup_crossover_mixed_cand(
                pt,
                scen_chip,
                d_pid,
                (planner.get_b_kv, planner.get_dec_sf),
                mix_meta,
            )
        else:
            _setup_crossover_disagg_cand(
                pt, scen_chip, d_pid, helpers, (dis_fast_groups, dis_ttft_groups)
            )

    ctx = _GridEvalContext(
        pre_cache=planner.pre_cache,
        mix_cache=planner.mix_cache,
        pool_plans=planner.pool_plans,
        compute_pre_sf=planner.compute_pre_sf,
        compute_mix_sf=planner.compute_mix_sf,
        get_wire_b=planner.get_wire_b,
        need_ttft=planner.need_ttft,
        need_wire=planner.need_wire,
        prefill_chunk=scen.prefill_chunk,
        eff_al_s=planner.eff_al * 1e-3,
        inv_al_ms=1e3 / planner.eff_al,
        dec_steps=planner.dec_steps,
        think_s=planner.think_s,
        has_think=planner.has_think,
        ttft_slo=scen.ttft_slo_ms,
        tpot_slo=scen.tpot_slo_ms,
    )
    return ((mix_meta, dis_fast_groups, dis_ttft_groups), ctx)


def sweep_prefix_hit_crossovers(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    plan: ServingPlan,
    *,
    hit_rates: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95),
) -> tuple[tuple[float, float] | None, float | None]:
    """Sweep candidate HBM-fit pools across prefix hit rates in under 50 ms."""
    if plan.recommended is None:
        return (None, None)
    raw_cands = (
        plan.sweep_candidates
        if plan.sweep_candidates
        else (plan.candidates if plan.candidates else plan.frontier)
    )
    if not raw_cands:
        return (None, None)
    scen = plan.scenario
    cands = (
        raw_cands
        if plan.sweep_candidates
        else _prune_sweep_candidates(geom, dtypes, chip, scen, raw_cands)
    )
    grid = tuple(
        sorted(
            {*(round(float(h), 4) for h in hit_rates), round(scen.prefix_hit_rate, 4)}
        )
    )
    pools, ctx = _setup_crossover_sweep(geom, dtypes, chip, scen, cands)
    winners: list[tuple[float, tuple]] = []
    for h in grid:
        entry = _eval_grid_h(h, scen, pools, ctx)
        if entry is not None:
            winners.append(entry)
    if not winners:
        return (None, None)
    return _derive_winner_range_and_crossover(scen, plan, winners)
