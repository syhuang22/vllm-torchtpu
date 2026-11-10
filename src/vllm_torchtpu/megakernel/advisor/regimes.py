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
"""Regime sweep, auto-parallelism recommendation, and cross-regime advisor matrix."""

from __future__ import annotations

from dataclasses import replace
from math import gcd
from typing import TYPE_CHECKING, Literal

from ..advisor.models import (
    DEPLOYMENT_PRESETS,
    MODEL_PRESET_SOURCES,
    DeploymentPreset,
    ModelProfile,
    ParallelLayout,
    resolve_model_profile,
)
from .chips import ChipSpec, resolve_chip_spec
from .specs import (
    AdvisorSweepReport,
    DisaggHandoffSpec,
    DispatchRegime,
    MeasurementsData,
    ObjectiveMode,
    ParallelismRecommendation,
    RegimeEntry,
    WorkloadSpec,
    load_measurements,
)
from .verdict import AdvisorReport, ScopeEstimate, advise
from .vmem import _DTYPE_BYTES, compute_per_token_vmem_bytes

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


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
    context = _int_opt(opts, "context", 4096)
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
                "fastest_scope": rep.fastest_scope,
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


def recommend_parallelism(
    profile: ModelProfile,
    chip: ChipSpec | str = "tpu7x",
    *,
    dp: int = 1,
    pp: int = 1,
) -> ParallelismRecommendation:
    """Auto-derive the minimum balanced TPU parallelism mesh for a model and chip."""
    chip_spec = resolve_chip_spec(chip)
    fallback: ParallelismRecommendation | None = None
    for tp_cand in (1, 2, 4, 8, 16, 32, 64):
        if tp_cand * dp * pp < chip_spec.min_slice_cores:
            continue
        if profile.num_attention_heads % tp_cand != 0:
            continue
        if (
            profile.num_linear_layers > 0
            and profile.linear_value_heads > 0
            and profile.linear_value_heads % tp_cand != 0
        ):
            continue
        ep_cand = (
            gcd(tp_cand * dp, profile.num_experts) if profile.num_experts > 0 else 1
        )
        wl = WorkloadSpec(
            phase="decode", batch=16, tp=tp_cand, ep=ep_cand, dp=dp, pp=pp
        )
        rep = advise(profile, wl, chip_spec)
        if not rep.hbm_fits:
            continue
        sub = rep.scopes[0]
        total_cores = tp_cand * dp * pp
        topo = chip_spec.topologies.get(total_cores, f"{total_cores} cores")
        cand = ParallelismRecommendation(
            tp=tp_cand,
            ep=ep_cand,
            dp=dp,
            pp=pp,
            topology=topo,
            hbm_used_gb=rep.hbm_used_gb,
            hbm_capacity_gb=rep.hbm_capacity_gb,
            w_step_gb=sub.w_step_gb,
            auto_selected=True,
        )
        fallback = cand
        # Verify HBM capacity holds at the high-concurrency 32K regime (B=64, 32K)
        # and that the reference B=16 decode batch fits in one VMEM pass.
        rep_high = advise(
            profile,
            replace(wl, batch=64, isl=32768, context=32768),
            chip_spec,
        )
        whole_sc = next(s for s in rep.scopes if s.scope == "whole_step")
        if rep_high.hbm_fits and whole_sc.n_splits == 1:
            return cand
    if fallback is not None:
        return fallback
    return ParallelismRecommendation(
        tp=8,
        ep=gcd(8 * dp, profile.num_experts) if profile.num_experts > 0 else 1,
        dp=dp,
        pp=pp,
        topology=chip_spec.topologies.get(8 * dp * pp, f"{8 * dp * pp} cores"),
        hbm_used_gb=0.0,
        hbm_capacity_gb=round(chip_spec.hbm_capacity_bytes / 1e9, 3),
        w_step_gb=0.0,
        auto_selected=True,
    )


def _int_opt(opts: Mapping[str, object], key: str, default: int) -> int:
    """Extract an integer option from opts or return default."""
    val = opts.get(key, default)
    return int(float(str(val))) if type(val) in (int, float, str) else default


def _float_opt(opts: Mapping[str, object], key: str, default: float) -> float:
    """Extract a float option from opts or return default."""
    val = opts.get(key, default)
    return float(str(val)) if type(val) in (int, float, str) else default


def _layout_recommendation(
    prof: ModelProfile,
    chip_spec: ChipSpec,
    layout: ParallelLayout,
    base: WorkloadSpec,
    source: str = "deployment preset",
) -> ParallelismRecommendation:
    """Evaluate a fixed layout HBM fit at the reference decode workload."""
    base_rep = advise(
        prof,
        replace(
            base,
            phase="decode",
            batch=16,
            tp=layout.tp,
            ep=layout.ep,
            dp=layout.dp,
            pp=layout.pp,
            dcp=layout.dcp,
            pcp=layout.pcp,
            layer_groups=layout.layer_groups,
        ),
        chip_spec,
    )
    total_cores = layout.tp * layout.dp * layout.pp
    return ParallelismRecommendation(
        tp=layout.tp,
        ep=layout.ep,
        dp=layout.dp,
        pp=layout.pp,
        topology=chip_spec.topologies.get(total_cores, f"{total_cores} cores"),
        hbm_used_gb=base_rep.hbm_used_gb,
        hbm_capacity_gb=base_rep.hbm_capacity_gb,
        w_step_gb=base_rep.scopes[0].w_step_gb,
        auto_selected=False,
        dcp=layout.dcp,
        pcp=layout.pcp,
        source=source,
        layer_groups=layout.layer_groups,
    )


def _select_layouts(
    prof: ModelProfile,
    chip_spec: ChipSpec,
    base: WorkloadSpec,
    requested: tuple[int | None, int | None, int, int, int, int],
    dep: DeploymentPreset | None,
) -> tuple[
    ParallelismRecommendation,
    ParallelismRecommendation | None,
    ParallelismRecommendation | None,
]:
    """Return decode, prefill, and saturation layouts: preset, user, or auto-pick."""
    tp, ep, dp, pp, dcp, pcp = requested
    if dep is not None:
        sat_rec = (
            _layout_recommendation(prof, chip_spec, dep.saturation, base)
            if dep.saturation is not None
            else None
        )
        dec_rec = (
            sat_rec
            if base.objective == "throughput" and sat_rec is not None
            else _layout_recommendation(prof, chip_spec, dep.decode, base)
        )
        return (
            dec_rec,
            _layout_recommendation(prof, chip_spec, dep.prefill, base),
            sat_rec,
        )
    if tp is not None:
        ep_val = (
            ep
            if ep is not None
            else (gcd(tp * dp, prof.num_experts) if prof.num_experts > 0 else 1)
        )
        layout = ParallelLayout(tp=tp, ep=ep_val, dp=dp, pp=pp, dcp=dcp, pcp=pcp)
        return (
            _layout_recommendation(prof, chip_spec, layout, base, source=""),
            None,
            None,
        )
    par = recommend_parallelism(prof, chip_spec, dp=dp, pp=pp)
    if ep is not None or dcp > 1 or pcp > 1:
        par = replace(par, ep=ep if ep is not None else par.ep, dcp=dcp, pcp=pcp)
    return par, None, None


def _compute_disagg_handoff(
    prof: ModelProfile,
    chip_spec: ChipSpec,
    dep: DeploymentPreset | None,
    meas: MeasurementsData | None,
    isl: int = 4096,
) -> DisaggHandoffSpec | None:
    """Compute per-request KV + recurrent state wire bytes and DCN transfer floor."""
    if dep is None:
        return None
    bw_kv = _DTYPE_BYTES.get(prof.kv_dtype.lower(), 1.0)
    bw_st = _DTYPE_BYTES.get(prof.state_dtype.lower(), 2.0)
    kv_per_tok = (
        (prof.kv_lora_rank + prof.qk_rope_head_dim) * bw_kv
        if prof.kv_lora_rank
        else 2.0 * prof.num_kv_heads * prof.head_dim * bw_kv
    )
    kv_wire_bytes = (
        prof.num_attention_layers * isl * kv_per_tok * prof.kv_compress_factor
    )
    qkv_lin_full = (
        2 * prof.linear_key_heads * prof.linear_key_dim
        + prof.linear_value_heads * prof.linear_value_dim
    )
    z_dim_full = prof.linear_value_heads * prof.linear_value_dim
    state_wire_bytes = (
        prof.num_linear_layers
        * (z_dim_full * prof.linear_key_dim + qkv_lin_full * prof.linear_conv_kernel)
        * bw_st
    )
    total_wire_bytes = kv_wire_bytes + state_wire_bytes
    if meas is not None and meas.dcn_bw_gbps is not None:
        dcn_gbps = meas.dcn_bw_gbps
        dcn_bracket = (dcn_gbps, dcn_gbps)
        dcn_prov: Literal["ASSUMED", "MEASURED"] = "MEASURED"
    else:
        dcn_gbps = chip_spec.dcn_bw_gbps
        dcn_bracket = chip_spec.dcn_bw_gbps_bracket
        dcn_prov = "ASSUMED"
    dcn_bytes_per_s = (dcn_gbps / 8.0) * 1e9
    dcn_lo_bytes_per_s = (dcn_bracket[0] / 8.0) * 1e9
    dcn_hi_bytes_per_s = (dcn_bracket[1] / 8.0) * 1e9
    xfer_ms = round((total_wire_bytes / max(1.0, dcn_bytes_per_s)) * 1e3, 2)
    xfer_lo_ms = round((total_wire_bytes / max(1.0, dcn_hi_bytes_per_s)) * 1e3, 2)
    xfer_hi_ms = round((total_wire_bytes / max(1.0, dcn_lo_bytes_per_s)) * 1e3, 2)
    return DisaggHandoffSpec(
        isl=isl,
        kv_wire_mb=round(kv_wire_bytes / 1e6, 2),
        state_wire_mb=round(state_wire_bytes / 1e6, 2),
        total_wire_mb=round(total_wire_bytes / 1e6, 2),
        dcn_bw_gbps=dcn_gbps,
        dcn_bw_gbps_bracket=dcn_bracket,
        dcn_xfer_ms=xfer_ms,
        dcn_xfer_bracket_ms=(xfer_lo_ms, xfer_hi_ms),
        dcn_provenance=dcn_prov,
    )


def _with_layout(wl: WorkloadSpec, layout: ParallelLayout) -> WorkloadSpec:
    """Copy `wl` onto `layout`'s parallelism degrees and layer groups."""
    return replace(
        wl,
        tp=layout.tp,
        ep=layout.ep,
        dp=layout.dp,
        pp=layout.pp,
        dcp=layout.dcp,
        pcp=layout.pcp,
        layer_groups=layout.layer_groups,
    )


def _with_preset_prefill(
    workloads: tuple[tuple[str, WorkloadSpec], ...],
    dep: DeploymentPreset | None,
) -> tuple[tuple[str, WorkloadSpec], ...]:
    """Move prefill and saturated decode regimes onto preset layouts and add windows."""
    if dep is None:
        return workloads
    pre = dep.prefill
    sat = dep.saturation
    decode_base = _with_layout(workloads[0][1], pre)
    windows = tuple(
        (
            f"Preset Prefill Window (W={win})",
            replace(
                decode_base,
                phase="prefill",
                batch=1,
                isl=win,
                chunk_size=win,
                moe_mode="grouped_on_chip",
            ),
        )
        for win in dep.prefill_windows
    )
    sat_batch = min(600, 256 * sat.dp) if sat is not None else 256
    sat_k = dep.decode.spec_tokens
    moved_list: list[tuple[str, WorkloadSpec]] = []
    for lbl, wl in workloads:
        if wl.phase == "prefill":
            moved_list.append((lbl, _with_layout(wl, pre)))
        elif sat is not None and lbl.startswith("Decode Saturated"):
            moved_list.append(
                (
                    f"Decode Saturated (B={sat_batch} over DP{sat.dp}, T=1, 32K)",
                    replace(_with_layout(wl, sat), batch=min(600, wl.batch * sat.dp)),
                )
            )
        else:
            moved_list.append((lbl, wl))
            if sat is not None and sat_k > 0 and lbl.startswith("Spec Verify High"):
                sat_lbl = (
                    f"Spec Verify Saturated (B={sat_batch} over DP{sat.dp}, "
                    f"K={sat_k}, 32K)"
                )
                moved_list.append(
                    (
                        sat_lbl,
                        replace(
                            _with_layout(wl, sat),
                            phase="verify",
                            batch=sat_batch,
                            spec_tokens=sat_k,
                        ),
                    )
                )
    return tuple(moved_list) + windows


def _build_reference_workloads(
    base: WorkloadSpec, k_spec: int
) -> tuple[tuple[str, WorkloadSpec], ...]:
    """Build the standard 12 reference request regimes."""
    return (
        (
            "Decode Small (B=4, T=1, 4K)",
            replace(base, phase="decode", batch=4, isl=3072, osl=2048),
        ),
        (
            "Decode Low (B=8, T=1, 4K)",
            replace(base, phase="decode", batch=8, isl=3072, osl=2048),
        ),
        (
            "Decode Med (B=16, T=1, 4K)",
            replace(base, phase="decode", batch=16, isl=3072, osl=2048),
        ),
        (
            "Decode High (B=64, T=1, 32K)",
            replace(
                base,
                phase="decode",
                batch=64,
                isl=32768,
                osl=2048,
                context=32768,
                moe_mode="grouped_on_chip",
            ),
        ),
        (
            "Decode Saturated (B=256, T=1, 32K)",
            replace(
                base,
                phase="decode",
                batch=256,
                isl=32768,
                osl=2048,
                context=32768,
                moe_mode="grouped_on_chip",
            ),
        ),
        (
            f"Spec Verify Small (B=4, K={k_spec})",
            replace(
                base, phase="verify", batch=4, spec_tokens=k_spec, isl=3072, osl=2048
            ),
        ),
        (
            f"Spec Verify Low (B=8, K={k_spec})",
            replace(
                base,
                phase="verify",
                batch=8,
                spec_tokens=k_spec,
                isl=3072,
                osl=2048,
                moe_mode="grouped_on_chip",
            ),
        ),
        (
            f"Spec Verify Med (B=16, K={k_spec})",
            replace(
                base,
                phase="verify",
                batch=16,
                spec_tokens=k_spec,
                isl=3072,
                osl=2048,
                moe_mode="grouped_on_chip",
            ),
        ),
        (
            f"Spec Verify High (B=32, K={k_spec})",
            replace(
                base,
                phase="verify",
                batch=32,
                spec_tokens=k_spec,
                isl=32768,
                osl=2048,
                context=32768,
                moe_mode="grouped_on_chip",
            ),
        ),
        (
            "Mixed Decode+Prefill (B=16, C=256)",
            replace(
                base,
                phase="mixed",
                batch=16,
                isl=4096,
                osl=2048,
                chunk_size=256,
                prefix_hit_rate=0.90,
                moe_mode="grouped_on_chip",
            ),
        ),
        (
            "Chunked Prefill (B=1, T=256)",
            replace(
                base,
                phase="prefill",
                batch=1,
                isl=4096,
                chunk_size=256,
                prefix_hit_rate=0.9375,
                moe_mode="grouped_on_chip",
            ),
        ),
        (
            "Long Prefill (B=1, T=4096)",
            replace(
                base, phase="prefill", batch=1, isl=4096, moe_mode="grouped_on_chip"
            ),
        ),
    )


def _evaluate_sweep_workloads(
    prof: ModelProfile,
    chip_spec: ChipSpec,
    workloads: tuple[tuple[str, WorkloadSpec], ...],
    meas_opt: object,
) -> tuple[list[RegimeEntry], list[AdvisorReport]]:
    """Evaluate workloads across all regimes and assemble report entries."""
    entries: list[RegimeEntry] = []
    reports: list[AdvisorReport] = []
    for label, wl in workloads:
        rep = advise(prof, wl, chip_spec, measurements=meas_opt)
        reports.append(rep)
        best_sc = next(s for s in rep.scopes if s.scope == rep.fastest_scope)
        entries.append(
            RegimeEntry(
                label=label,
                phase=wl.phase,
                active_tokens=rep.active_tokens,
                effective_context=rep.effective_context,
                step_arithmetic_intensity=rep.step_arithmetic_intensity,
                max_vmem_tokens=best_sc.t_vmem_max,
                num_vmem_splits=best_sc.n_splits,
                recommended_scope=rep.recommended_scope,
                recommended_moe_mode=rep.recommended_moe_mode,
                step_bound_us=best_sc.step_bound_us,
                rationale=rep.rationale,
                layout=ParallelLayout(
                    tp=wl.tp, ep=wl.ep, dp=wl.dp, pp=wl.pp, dcp=wl.dcp, pcp=wl.pcp
                ).label(),
                step_upper_us=best_sc.step_upper_us,
                verdict=rep.verdict,
                confidence=rep.confidence,
                flags=rep.flags,
                metrics=rep.metrics,
                unresolved_assumptions=rep.unresolved_assumptions,
                scopes=tuple(
                    {
                        "scope": s.scope,
                        "feasible": s.feasible,
                        "reason": s.reason,
                        "step_bound_us": s.step_bound_us,
                        "step_achievable_us": s.step_achievable_us,
                        "step_achievable_upper_us": s.step_achievable_upper_us,
                        "step_upper_us": s.step_upper_us,
                        "roof_floor_us": s.roof_floor_us,
                        "compute_us": s.compute_us,
                        "hbm_us": s.hbm_us,
                        "comm_us": s.comm_us,
                        "fill_drain_us": s.fill_drain_us,
                        "repass_spill_us": s.repass_spill_us,
                        "launch_us": s.launch_us,
                        "ops_per_step": s.ops_per_step,
                        "n_splits": s.n_splits,
                        "t_vmem_max": s.t_vmem_max,
                        "selected_nbd": s.selected_nbd,
                        "selected_nbe": s.selected_nbe,
                        "vmem_static_mib": s.vmem_static_mib,
                        "w_step_gb": s.w_step_gb,
                        "act_hbm_mb": s.act_hbm_mb,
                        "moe_flop_factor": s.moe_flop_factor,
                        "w_step_strict_gb": s.w_step_strict_gb,
                        "step_floor_strict_us": s.step_floor_strict_us,
                        "support": s.support,
                        "blockers": list(s.blockers),
                    }
                    for s in rep.scopes
                ),
                fastest_scope=rep.fastest_scope,
                fastest_verdict=rep.fastest_verdict,
                evidence=rep.evidence,
                hypothetical_best=rep.hypothetical_best,
                workload={
                    "phase": wl.phase,
                    "batch": wl.batch,
                    "isl": wl.isl,
                    "osl": wl.osl,
                    "context": rep.effective_context,
                    "spec_tokens": wl.spec_tokens,
                    "chunk_size": wl.chunk_size,
                    "prefix_hit_rate": wl.prefix_hit_rate,
                    "microbatches": wl.microbatches,
                    "segment_layers": wl.segment_layers,
                    "tp": wl.tp,
                    "ep": wl.ep,
                    "dp": wl.dp,
                    "pp": wl.pp,
                    "dcp": wl.dcp,
                    "pcp": wl.pcp,
                    "kv_split": wl.kv_split,
                    "layer_groups": list(wl.layer_groups),
                    "moe_mode": wl.moe_mode,
                    "vmem_alias": wl.vmem_alias,
                    "vmem_fit_headroom": wl.vmem_fit_headroom,
                    "op_overhead_us": wl.op_overhead_us,
                    "dma_buffers": wl.dma_buffers,
                    "dispatch_regime": wl.dispatch_regime,
                    "hbm_utilization_cap": wl.hbm_utilization_cap,
                },
            )
        )
    return entries, reports


def advise_regimes(
    profile: ModelProfile | str | Mapping[str, object],
    chip: ChipSpec | str = "tpu7x",
    **opts: object,
) -> AdvisorSweepReport:
    """Evaluate the 12-regime Batch Size x Phase & Context matrix."""
    raw_tp = opts.get("tp")
    raw_ep = opts.get("ep")
    raw_vmem = opts.get("vmem_budget_mib")
    tp = int(raw_tp) if type(raw_tp) is int else None
    ep = int(raw_ep) if type(raw_ep) is int else None
    vmem_budget_mib = (
        float(str(raw_vmem)) if type(raw_vmem) in (int, float, str) else None
    )
    dp = _int_opt(opts, "dp", 1)
    pp = _int_opt(opts, "pp", 1)
    dcp = _int_opt(opts, "dcp", 1)
    pcp = _int_opt(opts, "pcp", 1)
    meas_opt = opts.get("measurements")
    cs_path_opt = opts.get("chip_spec_path")
    raw_obj = str(opts.get("objective", "step"))
    obj_mode: ObjectiveMode = (
        raw_obj if raw_obj in ("step", "throughput", "both") else "step"
    )
    raw_dr = str(opts.get("dispatch_regime", "auto"))
    dr_mode: DispatchRegime = (
        raw_dr
        if raw_dr in ("auto", "zero_floor", "compiled_xla", "unfused_eager")
        else "auto"
    )
    base = WorkloadSpec(
        vmem_alias=bool(opts.get("vmem_alias", True)),
        vmem_fit_headroom=bool(opts.get("vmem_fit_headroom", False)),
        op_overhead_us=_float_opt(opts, "op_overhead_us", 0.0),
        vmem_budget_mib=vmem_budget_mib,
        dma_buffers=_int_opt(opts, "dma_buffers", 2),
        objective=obj_mode,
        dispatch_regime=dr_mode,
    )
    dma_buffers = base.dma_buffers

    prof = resolve_model_profile(profile)
    chip_spec = resolve_chip_spec(
        chip,
        chip_spec_path=str(cs_path_opt) if cs_path_opt is not None else None,
    )
    explicit_layout = tp is not None or ep is not None or (dp, pp, dcp, pcp) != (1,) * 4
    dep = (
        DEPLOYMENT_PRESETS.get(prof.name)
        if bool(opts.get("deployment_preset", True)) and not explicit_layout
        else None
    )
    par, prefill_par, sat_par = _select_layouts(
        prof, chip_spec, base, (tp, ep, dp, pp, dcp, pcp), dep
    )
    meas_data = (
        load_measurements(meas_opt, chip_name=chip_spec.name)
        if meas_opt is not None
        else None
    )
    disagg_handoff = _compute_disagg_handoff(prof, chip_spec, dep, meas_data)
    base = replace(
        base, tp=par.tp, ep=par.ep, dp=par.dp, pp=par.pp, dcp=par.dcp, pcp=par.pcp
    )
    k_spec = dep.decode.spec_tokens if dep is not None and dep.decode.spec_tokens else 3
    workloads = _build_reference_workloads(base, k_spec)
    workloads = _with_preset_prefill(workloads, dep)

    entries, reports = _evaluate_sweep_workloads(prof, chip_spec, workloads, meas_opt)
    decode_rep = reports[2]
    vmem_parts = compute_per_token_vmem_bytes(prof, workloads[2][1])
    cross_sc = next(s for s in decode_rep.scopes if s.scope == "whole_step")
    one_l_sc = next(s for s in decode_rep.scopes if s.scope == "single_layer")
    ref_sc = cross_sc if cross_sc.feasible else one_l_sc

    t_ceil = ref_sc.t_vmem_max
    t_ridge = decode_rep.ridge_tokens
    step_mode = primary_step_mode(cross_sc)
    slope_kib = vmem_parts["total_per_token_bytes"] / 1024
    ridge_fb = decode_rep.ridge_dense_flop_per_byte
    guide = (
        f"Small/Med Decode & Spec Verify (T <= {t_ceil} active tokens): "
        f"use scope='{step_mode}' with moe_mode='masked_on_chip' (T < 32) "
        f"or 'grouped_on_chip' (T >= 32) to keep [T, H] in VMEM "
        f"({ref_sc.vmem_static_mib:.1f} MiB {dma_buffers}-slot weight staging + "
        f"{slope_kib:.1f} KiB/tok).",
        f"High-Batch Decode & Chunked Prefill ({t_ceil} < T <= {t_ridge} tokens): "
        "use scope='single_layer' or 'layer_group' "
        f"(single-layer VMEM ceiling T_max={one_l_sc.t_vmem_max}) "
        "so weights are not re-streamed across multi-pass token splits.",
        f"Saturated Decode & Compute-Bound Prefill (T > {t_ridge} tokens, "
        f"AI >= {ridge_fb:.0f} FLOP/B): use scope='per_op' "
        "with moe_mode='grouped_in_hbm': tokens are grouped by expert in HBM "
        "and each operator is its own call, so no VMEM is held for "
        "multi-layer weight staging.",
    )
    return AdvisorSweepReport(
        model=prof.name,
        chip=chip_spec.name,
        parallelism=par,
        vmem_budget_mib=decode_rep.vmem_budget_mib,
        vmem_static_mib=ref_sc.vmem_static_mib,
        per_token_vmem_bytes=vmem_parts["total_per_token_bytes"],
        vmem_breakdown_bytes=vmem_parts,
        max_vmem_tokens=t_ceil,
        ridge_dense_flop_per_byte=decode_rep.ridge_dense_flop_per_byte,
        ridge_tokens=t_ridge,
        regimes=tuple(entries),
        tailoring_guide=guide,
        dma_buffers=dma_buffers,
        single_layer_vmem_tokens=one_l_sc.t_vmem_max,
        prefill_parallelism=prefill_par,
        saturation_parallelism=sat_par,
        disagg_handoff=disagg_handoff,
        chip_sources=chip_spec.sources if chip_spec.sources is not None else {},
        model_sources=_build_model_sources(prof),
        chip_spec=chip_spec,
        model_profile=prof,
    )


def _build_model_sources(prof: ModelProfile) -> dict[str, str]:
    """Extract provenance sources for model architecture facts."""
    base_src = MODEL_PRESET_SOURCES.get(prof.name, f"Model config: {prof.name}")
    sources = {
        "config": base_src,
        "num_layers": f"{prof.num_layers} layers ({base_src})",
        "hidden_size": f"H={prof.hidden_size} ({base_src})",
        "attention_heads": (
            f"NH={prof.num_attention_heads}, NKV={prof.num_kv_heads}, "
            f"HD={prof.head_dim}"
        ),
        "dtypes": (
            f"weight={prof.weight_dtype}, expert={prof.expert_dtype}, "
            f"kv={prof.kv_dtype}"
        ),
    }
    if prof.num_experts > 0:
        sources["moe"] = (
            f"E={prof.num_experts}, TOPK={prof.top_k}, "
            f"I={prof.moe_intermediate_size} ({base_src})"
        )
    return sources


def primary_step_mode(cross_sc: ScopeEstimate) -> str:
    """Return whole_step if whole-step fusion is feasible, else single_layer."""
    return "whole_step" if cross_sc.feasible else "single_layer"
