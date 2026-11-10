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
"""Megakernel advisor data schemas, constants, and workload specifications."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from math import log2
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, TypedDict

from ..advisor.models import ModelProfile, ParallelLayout
from ..core import errors
from . import chips

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..core.exec_spec import Tiles
    from .chips import ChipSpec

FusionScope = Literal["per_op", "single_layer", "layer_group", "whole_step"]
Phase = Literal["decode", "verify", "prefill", "mixed"]
# Where routed tokens are grouped by expert and where the data lives.
MoEMode = Literal["masked_on_chip", "grouped_on_chip", "grouped_in_hbm", "dense_mlp"]
ProvenanceTag = Literal["DERIVED", "ASSUMED", "MEASURED"]
ObjectiveMode = Literal["step", "throughput", "both"]
DispatchRegime = Literal["auto", "zero_floor", "compiled_xla", "unfused_eager"]
SupportStatus = Literal["baseline", "within_core_limits", "declared", "hypothetical"]

_MEASUREMENTS_SCHEMA = "advisor.measurements.v1"

SUPPORTED_ONCHIP_FAMILIES: frozenset[str] = frozenset(
    {"gqa", "mla", "gdn", "kda", "dense_mlp", "moe"}
)

_DTYPE_BYTES: Mapping[str, float] = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "nvfp4": 0.5625,
    "mxfp4": 0.53125,
    "fp4": 0.5625,
}


class ProvenanceMetric(TypedDict):
    """Single output metric value paired with its provenance tag, source, and band."""

    value: float | int | bool
    unit: str
    tag: ProvenanceTag
    source: str
    band: list[float]


@dataclass(frozen=True)
class RingPlan:
    """On-chip VMEM ring-buffer slot counts and bank widths for weight streaming."""

    dense_slots: int = 2
    expert_slots: int = 2
    dense_slot_bytes: int = 2 << 20
    expert_slot_bytes: int = 1 << 20
    compiler_spill_reserve_bytes: int = 0


def default_double_buffer_ring_plan(dma_buffers: int = 2) -> RingPlan:
    """Return the default n-slot weight-streaming RingPlan."""
    return RingPlan(dense_slots=dma_buffers, expert_slots=dma_buffers)


@dataclass(frozen=True)
class MeasurementsData:
    """Calibrated runtime measurements loaded from an `advisor.measurements.v1` file."""

    source_desc: str
    source_kind: str = "trace"
    launch_gap_p50_us: float | None = None
    launch_gap_p90_us: float | None = None
    launch_gap_n: int = 0
    collective_hop_us: float | None = None
    collective_n: int = 0
    step_coverage: float | None = None
    step_time_us: float | None = None
    dcn_bw_gbps: float | None = None
    ring_plan: RingPlan | None = None


def _parse_collectives_hop_us(coll_raw: object) -> tuple[float | None, int]:
    """Extract minimum per-hop collective latency and count from collectives."""
    if type(coll_raw) is not list or not coll_raw:
        return None, 0
    hops_list: list[float] = []
    coll_n = 0
    for item in coll_raw:
        if type(item) is dict and "latency_us" in item:
            grp = max(2, int(item.get("group_size", 2)))
            hops = max(1, round(log2(grp)))
            hops_list.append(float(item["latency_us"]) / hops)
            coll_n += int(item.get("n", 1))
    return (min(hops_list) if hops_list else None), coll_n


def _parse_ring_plan(rp_raw: object) -> RingPlan | None:
    """Parse an optional RingPlan mapping from measurements JSON."""
    if type(rp_raw) is not dict:
        return None
    return RingPlan(
        dense_slots=int(rp_raw.get("dense_slots", 2)),
        expert_slots=int(rp_raw.get("expert_slots", 2)),
        dense_slot_bytes=int(rp_raw.get("dense_slot_bytes", 2 << 20)),
        expert_slot_bytes=int(rp_raw.get("expert_slot_bytes", 1 << 20)),
        compiler_spill_reserve_bytes=int(rp_raw.get("compiler_spill_reserve_bytes", 0)),
    )


def load_measurements(
    target: object,
    *,
    chip_name: str = "tpu7x",
) -> MeasurementsData:
    """Load and validate an `advisor.measurements.v1` JSON file or mapping."""
    if type(target) is MeasurementsData:
        return target
    if type(target) is dict:
        payload: Mapping[str, object] = target
        file_label = "inline_mapping"
    else:
        target_str = str(target).strip()
        if target_str in ("bundled", "auto"):
            bundled_path = chips.public_calibration_path(chip_name)
            if bundled_path is None or not bundled_path.is_file():
                msg = (
                    "No default bundled architectural priors found for chip "
                    f"{chip_name!r}"
                )
                raise errors.MegakernelConfigError(msg)
            path = bundled_path
        else:
            path = Path(target_str)
            if not path.is_file():
                msg = f"Missing measurements file: {path}"
                raise errors.MegakernelConfigError(msg)
        payload = json.loads(path.read_text("utf-8"))
        file_label = path.name
    if payload.get("schema") != _MEASUREMENTS_SCHEMA:
        msg = (
            f"Unsupported measurements schema {payload.get('schema')!r}; "
            f"expected {_MEASUREMENTS_SCHEMA!r}"
        )
        raise errors.MegakernelConfigError(msg)
    src_raw = payload.get("source")
    src_map: Mapping[str, object] = src_raw if type(src_raw) is dict else {}
    src_kind = str(src_map.get("kind", "trace"))
    src_desc = f"{src_kind}:{src_map.get('file', file_label)}"

    lg_raw = payload.get("launch_gap_us")
    lg_map: Mapping[str, object] = lg_raw if type(lg_raw) is dict else {}
    lg_p50 = float(str(lg_map["p50"])) if "p50" in lg_map else None
    lg_p90 = (
        float(str(lg_map.get("p90", lg_p50 or 0.0))) if lg_p50 is not None else None
    )
    lg_n = int(str(lg_map.get("n", 0))) if lg_p50 is not None else 0

    coll_hop, coll_n = _parse_collectives_hop_us(payload.get("collectives"))

    cov_raw = payload.get("step_coverage")
    cov_map: Mapping[str, object] = cov_raw if type(cov_raw) is dict else {}
    cov_val = (
        float(str(cov_map["fused_eligible_fraction"]))
        if "fused_eligible_fraction" in cov_map
        else None
    )

    st_raw = payload.get("step_time_us")
    st_map: Mapping[str, object] = st_raw if type(st_raw) is dict else {}
    st_val = (
        float(str(st_map["p50"]))
        if "p50" in st_map
        else (float(str(st_raw)) if type(st_raw) in (int, float) else None)
    )
    dcn_raw = payload.get("dcn_bw_gbps")
    dcn_val = float(str(dcn_raw)) if type(dcn_raw) in (int, float) else None

    return MeasurementsData(
        source_desc=src_desc,
        source_kind=src_kind,
        launch_gap_p50_us=lg_p50,
        launch_gap_p90_us=lg_p90,
        launch_gap_n=lg_n,
        collective_hop_us=coll_hop,
        collective_n=coll_n,
        step_coverage=cov_val,
        step_time_us=st_val,
        dcn_bw_gbps=dcn_val,
        ring_plan=_parse_ring_plan(payload.get("ring_plan")),
    )


@dataclass(frozen=True)
class EfficiencySpec:
    """ASSUMED achievable unit utilization priors and brackets vs peak rates."""

    BRACKETS: ClassVar[Mapping[str, tuple[float, float]]] = {
        "eta_hbm": (0.70, 0.90),
        "eta_dense": (0.60, 0.85),
        "eta_moe_grouped": (0.35, 0.70),
        "eta_ici": (0.60, 0.90),
    }

    eta_hbm: float = 0.82
    eta_dense: float = 0.72
    eta_moe_grouped: float = 0.50
    eta_ici: float = 0.75


@dataclass(frozen=True)
class WorkloadSpec:
    """Serving request geometry, VMEM budget, and multi-buffering configuration."""

    phase: Phase = "decode"
    batch: int = 16
    isl: int = 4096
    osl: int = 1024
    context: int = 0
    prefix_hit_rate: float = 0.0
    spec_tokens: int = 0
    chunk_size: int = 0
    microbatches: int = 0
    tp: int = 8
    pp: int = 1
    dp: int = 1
    ep: int = 1
    kv_split: int = 1
    dcp: int = 1
    pcp: int = 1
    segment_layers: int = 0
    # Layers fused per kernel call for the layer_group scope, in call order
    # (empty: the model has no shipped layer grouping). Must sum to num_layers.
    layer_groups: tuple[int, ...] = ()
    moe_mode: MoEMode = "masked_on_chip"
    vmem_alias: bool = True
    vmem_fit_headroom: bool = False
    tiles: Tiles | None = None
    op_overhead_us: float = 0.0
    vmem_budget_mib: float | None = None
    dma_buffers: int = 2
    ring_plan: RingPlan | None = None
    objective: ObjectiveMode = "step"
    hbm_utilization_cap: float = 0.80
    dispatch_regime: DispatchRegime = "auto"
    efficiency: EfficiencySpec = EfficiencySpec()


@dataclass(frozen=True)
class ScopeEstimate:
    """Roofline and VMEM capacity estimate for one fusion scope."""

    scope: FusionScope
    feasible: bool
    reason: str
    ops_per_step: int
    vmem_static_mib: float
    t_vmem_max: int
    n_splits: int
    w_step_gb: float
    w_crit_gb: float | None
    act_hbm_mb: float
    moe_flop_factor: float
    roof_floor_us: float
    step_bound_us: float
    selected_nbd: int = 0
    selected_nbe: int = 0
    step_achievable_us: float = 0.0
    break_even_op_overhead_us: float = 0.0
    step_upper_us: float = 0.0
    compute_us: float = 0.0
    hbm_us: float = 0.0
    comm_us: float = 0.0
    fill_drain_us: float = 0.0
    repass_spill_us: float = 0.0
    launch_us: float = 0.0
    step_achievable_upper_us: float = 0.0
    support: SupportStatus = "hypothetical"
    blockers: tuple[str, ...] = ()
    w_step_strict_gb: float = 0.0
    step_floor_strict_us: float = 0.0

    @property
    def step_ach_upper_us(self) -> float:
        """Alias for step_achievable_upper_us."""
        return self.step_achievable_upper_us

    @property
    def max_vmem_tokens(self) -> int:
        """Return maximum active tokens per step fitting in a single VMEM pass."""
        return self.t_vmem_max

    @property
    def num_vmem_splits(self) -> int:
        """Return the number of VMEM token-dimension passes per step."""
        return self.n_splits


@dataclass(frozen=True)
class AdvisorReport:
    """First-principles fusion-scope recommendation for a single workload."""

    model: str
    chip: str
    phase: Phase
    active_tokens: int
    effective_context: int
    vmem_budget_mib: float
    hbm_fits: bool
    hbm_used_gb: float
    hbm_capacity_gb: float
    ridge_dense_flop_per_byte: float
    step_arithmetic_intensity: float
    asymptotic_decode_intensity: float | None
    recommended_scope: FusionScope
    build_megakernel: bool
    rationale: str
    scopes: tuple[ScopeEstimate, ...]
    per_token_vmem_bytes: int = 0
    ridge_tokens: int = 0
    recommended_moe_mode: str = "dense_mlp"
    dma_buffers: int = 2
    verdict: str = "separate_likely"
    confidence: str = "prior_only"
    flags: tuple[str, ...] = ()
    metrics: dict[str, ProvenanceMetric] = field(default_factory=dict)
    unresolved_assumptions: tuple[dict[str, object], ...] = ()
    fastest_scope: FusionScope = "per_op"
    fastest_verdict: str = "separate_likely"
    evidence: SupportStatus = "baseline"
    hypothetical_best: dict[str, object] | None = None
    coverage: str = (
        "decoder layer stack; excludes embedding, LM head, sampling, "
        "draft heads and modality towers"
    )
    unmodeled_features: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize AdvisorReport to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ParallelismRecommendation:
    """Auto-derived balanced TPU parallelism layout for bare-minimum queries."""

    tp: int
    ep: int
    dp: int
    pp: int
    topology: str
    hbm_used_gb: float
    hbm_capacity_gb: float
    w_step_gb: float
    auto_selected: bool = True
    dcp: int = 1
    pcp: int = 1
    # "deployment preset" when taken from DEPLOYMENT_PRESETS, else empty.
    source: str = ""
    layer_groups: tuple[int, ...] = ()

    def label(self) -> str:
        """Return the layout as `TP8 x DCP4 x EP8`, omitting degree-1 axes."""
        return ParallelLayout(
            tp=self.tp, ep=self.ep, dp=self.dp, pp=self.pp, dcp=self.dcp, pcp=self.pcp
        ).label()


@dataclass(frozen=True)
class RegimeEntry:
    """Evaluation summary for one canonical serving request regime."""

    label: str
    phase: Phase
    active_tokens: int
    effective_context: int
    step_arithmetic_intensity: float
    max_vmem_tokens: int
    num_vmem_splits: int
    recommended_scope: FusionScope
    recommended_moe_mode: str
    step_bound_us: float
    rationale: str
    layout: str = ""
    step_upper_us: float = 0.0
    verdict: str = "separate_likely"
    confidence: str = "prior_only"
    flags: tuple[str, ...] = ()
    metrics: dict[str, ProvenanceMetric] = field(default_factory=dict)
    unresolved_assumptions: tuple[dict[str, object], ...] = ()
    # Floor, upper, launches, and passes of every scope, for the report ledger.
    scopes: tuple[dict[str, object], ...] = ()
    workload: dict[str, object] = field(default_factory=dict)
    fastest_scope: FusionScope = "per_op"
    fastest_verdict: str = "separate_likely"
    evidence: SupportStatus = "baseline"
    step_achievable_us: float = 0.0
    step_achievable_upper_us: float = 0.0
    step_floor_strict_us: float = 0.0
    blockers: tuple[str, ...] = ()
    hypothetical_best: dict[str, object] | None = None

    @property
    def step_ach_upper_us(self) -> float:
        """Alias for step_achievable_upper_us."""
        return self.step_achievable_upper_us


@dataclass(frozen=True)
class DisaggHandoffSpec:
    """Per-request KV and recurrent state wire payload and DCN transfer floor."""

    isl: int
    kv_wire_mb: float
    state_wire_mb: float
    total_wire_mb: float
    dcn_bw_gbps: float
    dcn_bw_gbps_bracket: tuple[float, float]
    dcn_xfer_ms: float
    dcn_xfer_bracket_ms: tuple[float, float]
    dcn_provenance: Literal["ASSUMED", "MEASURED"] = "ASSUMED"


@dataclass(frozen=True)
class AdvisorSweepReport:
    """Multi-regime crossover report across request regimes."""

    model: str
    chip: str
    parallelism: ParallelismRecommendation
    vmem_budget_mib: float
    vmem_static_mib: float
    per_token_vmem_bytes: int
    vmem_breakdown_bytes: dict[str, int]
    max_vmem_tokens: int
    ridge_dense_flop_per_byte: float
    ridge_tokens: int
    regimes: tuple[RegimeEntry, ...]
    tailoring_guide: tuple[str, ...]
    dma_buffers: int = 2
    single_layer_vmem_tokens: int = 0
    prefill_parallelism: ParallelismRecommendation | None = None
    saturation_parallelism: ParallelismRecommendation | None = None
    disagg_handoff: DisaggHandoffSpec | None = None
    chip_sources: Mapping[str, str] = field(default_factory=dict)
    model_sources: Mapping[str, str] = field(default_factory=dict)
    chip_spec: ChipSpec | None = None
    model_profile: ModelProfile | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize AdvisorSweepReport to a JSON-compatible dictionary."""
        return asdict(self)


BUCKET_SCHEDULE_SCHEMA = "advisor.bucket_schedule.v1"
BUCKET_CONSUMER_STATUS = "advisory_offline_v1"


@dataclass(frozen=True)
class BucketScheduleRule:
    """Contiguous active-token interval sharing one scope and MoE grouping mode."""

    phase: str
    layout: str
    min_active_tokens: int
    max_active_tokens: int
    effective_context: int
    recommended_scope: str
    recommended_moe_mode: str
    max_vmem_tokens: int
    num_vmem_splits: int
    step_bound_us: float
    verdict: str


@dataclass(frozen=True)
class BucketSchedulePayload:
    """Offline advisory token-bucket schedule derived from a regime sweep."""

    schema: str
    consumer_status: str
    model: str
    chip: str
    default_layout: str
    vmem_budget_mib: float
    max_vmem_tokens: int
    ridge_tokens: int
    buckets: tuple[BucketScheduleRule, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize BucketSchedulePayload to a JSON-compatible dictionary."""
        return asdict(self)
