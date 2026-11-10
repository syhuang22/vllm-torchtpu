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
"""Differential parity suite between Python advisor physics and JS workbench engine."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import signal
import time
from pathlib import Path
from typing import TypedDict

import pytest

from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import models, workbench_engine

_TWO_DECIMAL_ABS_TOL = 0.015
_RAW_STEP_REL_TOL = 1e-6
_MODEL_SAMPLE_COUNT = 5
_EXPECTED_SCENARIO_COUNT = 80
_TP_BASE_WIDE = 8
_TP_BASE_MID = 4
_EP_WIDE = 16


class _ScopeParityEntry(TypedDict):
    """Typed schema for one scope parity entry."""

    feasible: bool
    t_vmem_max: int
    n_splits: int
    selected_nbd: int
    selected_nbe: int
    raw_step_us: float
    step_bound_us: float
    step_floor_strict_us: float
    step_achievable_us: float
    step_achievable_upper_us: float
    w_step_strict_gb: float
    act_hbm_mb: float
    comm_us: float
    support: str
    blockers: list[str]


class _EvalParityResult(TypedDict):
    """Typed schema for model evaluation parity result."""

    invalid_reasons: list[str]
    fastest_scope: str
    recommended_scope: str
    scopes: dict[str, _ScopeParityEntry]


class _BatchScenarioItem(TypedDict):
    """Typed schema for one input scenario in parity batch."""

    only_validate: bool
    model_name: str
    wl: advisor.WorkloadSpec


def _select_five_models() -> tuple[str, ...]:
    """Return 5 distinct model presets spanning GQA, GDN, MLA, MoE, and dense FFN."""
    return (
        "qwen3.5-397b",
        "kimi-k3",
        "qwen3.8-2.4t",
        "deepseek-v3",
        "llama-3.1-405b",
    )


def _valid_workloads_for_model(
    prof: advisor.ModelProfile,
) -> list[advisor.WorkloadSpec]:
    """Build 15 diverse valid WorkloadSpec regimes for one model profile."""
    has_moe = prof.num_moe_layers > 0 and prof.num_experts > 0
    tp_base = (
        _TP_BASE_WIDE if prof.num_attention_heads % _TP_BASE_WIDE == 0 else _TP_BASE_MID
    )
    ep_valid = (
        _TP_BASE_WIDE
        if (
            has_moe
            and prof.num_experts % _TP_BASE_WIDE == 0
            and tp_base >= _TP_BASE_WIDE
        )
        else 1
    )
    dcp_val = _TP_BASE_MID if tp_base >= _TP_BASE_MID else 1
    return [
        advisor.WorkloadSpec(
            phase="decode", batch=4, isl=3072, osl=1024, tp=tp_base, ep=ep_valid
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=16,
            isl=4096,
            osl=512,
            tp=tp_base,
            ep=ep_valid,
            dispatch_regime="zero_floor",
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=32,
            isl=8192,
            osl=512,
            tp=tp_base,
            ep=ep_valid,
            dispatch_regime="compiled_xla",
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=64,
            isl=16384,
            osl=512,
            tp=tp_base,
            ep=ep_valid,
            dispatch_regime="unfused_eager",
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=8,
            isl=2048,
            osl=256,
            tp=tp_base,
            ep=ep_valid,
            moe_mode="grouped_on_chip",
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=16,
            isl=4096,
            osl=1024,
            tp=tp_base,
            ep=ep_valid,
            dispatch_regime="compiled_xla",
        ),
        advisor.WorkloadSpec(
            phase="verify",
            batch=24,
            spec_tokens=5,
            isl=8192,
            osl=512,
            tp=tp_base,
            ep=ep_valid,
            moe_mode="grouped_on_chip",
        ),
        advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=4096,
            osl=512,
            chunk_size=512,
            prefix_hit_rate=0.5,
            tp=tp_base,
            ep=ep_valid,
        ),
        advisor.WorkloadSpec(
            phase="prefill",
            batch=2,
            isl=8192,
            osl=512,
            pcp=2,
            tp=tp_base,
            ep=ep_valid,
            moe_mode="grouped_on_chip",
        ),
        advisor.WorkloadSpec(
            phase="mixed",
            batch=16,
            isl=4096,
            osl=2048,
            chunk_size=256,
            prefix_hit_rate=0.90,
            tp=tp_base,
            ep=ep_valid,
            moe_mode="grouped_on_chip",
        ),
        advisor.WorkloadSpec(
            phase="mixed",
            batch=12,
            isl=8192,
            osl=1024,
            pcp=2,
            dcp=dcp_val,
            tp=1 if dcp_val > 1 else tp_base,
            ep=1,
            moe_mode="grouped_on_chip",
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=16,
            isl=4096,
            osl=512,
            tp=tp_base,
            pp=2,
            ep=ep_valid,
            vmem_alias=False,
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=20,
            isl=4096,
            osl=512,
            tp=tp_base,
            ep=ep_valid,
            vmem_budget_mib=48.0,
            dma_buffers=3,
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=4,
            isl=1024,
            osl=128,
            tp=tp_base,
            ep=ep_valid,
            vmem_fit_headroom=0.95,
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=8,
            isl=4096,
            osl=512,
            tp=tp_base,
            ep=ep_valid,
            moe_mode="masked_on_chip",
        ),
    ]


def _invalid_sharding_scenarios() -> list[tuple[str, advisor.WorkloadSpec]]:
    """Build 5 invalid sharding scenarios spanning distinct divisibility checks."""
    return [
        ("qwen3.5-397b", advisor.WorkloadSpec(phase="decode", batch=8, tp=6, ep=1)),
        ("qwen3.5-397b", advisor.WorkloadSpec(phase="decode", batch=8, tp=8, ep=7)),
        ("kimi-k3", advisor.WorkloadSpec(phase="decode", batch=8, tp=2, dp=1, ep=8)),
        ("qwen3.5-397b", advisor.WorkloadSpec(phase="decode", batch=8, tp=8, dcp=3)),
        ("qwen3.5-397b", advisor.WorkloadSpec(phase="decode", batch=8, tp=8, pp=512)),
    ]


def _python_reference_eval(
    model_name: str,
    wl: advisor.WorkloadSpec,
    chip_spec: advisor.ChipSpec,
) -> _EvalParityResult:
    """Compute authoritative Python scopes and reasons using public advisor APIs."""
    prof = models.resolve_model_profile(model_name)
    reasons = list(advisor.validate_workload(prof, chip_spec, wl))
    rep = advisor.advise(prof, wl, chip_spec)
    raw_steps = advisor.compute_scope_raw_step_us(prof, wl, chip=chip_spec.name)
    scopes_out: dict[str, _ScopeParityEntry] = {}
    for sc in rep.scopes:
        scopes_out[sc.scope] = {
            "feasible": sc.feasible,
            "t_vmem_max": sc.t_vmem_max,
            "n_splits": sc.n_splits,
            "selected_nbd": sc.selected_nbd,
            "selected_nbe": sc.selected_nbe,
            "raw_step_us": raw_steps.get(sc.scope, 0.0),
            "step_bound_us": sc.step_bound_us,
            "step_floor_strict_us": sc.step_floor_strict_us,
            "step_achievable_us": sc.step_achievable_us,
            "step_achievable_upper_us": sc.step_achievable_upper_us,
            "w_step_strict_gb": sc.w_step_strict_gb,
            "act_hbm_mb": sc.act_hbm_mb,
            "comm_us": sc.comm_us,
            "support": sc.support,
            "blockers": list(sc.blockers),
        }
    return {
        "invalid_reasons": reasons,
        "fastest_scope": rep.fastest_scope,
        "recommended_scope": rep.recommended_scope,
        "scopes": scopes_out,
    }


def test_workbench_engine_asset_and_seed_contract() -> None:
    """Verify static JS engine asset carries cleanroom physics API and license."""
    engine_js = workbench_engine.render_workbench_engine_js()
    assert "// Copyright 2026 Google LLC" in engine_js
    assert "Licensed under the Apache License, Version 2.0" in engine_js
    assert "createAdvisorEngine" in engine_js
    assert "__MK_CREATE_ADVISOR_ENGINE" in engine_js

    for symbol in (
        "computePerTokenVmem",
        "evaluateWorkload",
        "allReduceCost",
        "expertsHit",
        "selectAdaptiveRing",
        "validateWorkload",
        "roundEven",
    ):
        assert symbol in engine_js, f"Missing symbol: {symbol}"

    # Verify no disallowed remote schemes or hardcoded constants in engine body
    # (excluding the standard Apache license URL in the header).
    body_lines = [
        line
        for line in engine_js.splitlines()
        if "http://www.apache.org/licenses/LICENSE-2.0" not in line
    ]
    body_js = "\n".join(body_lines)
    for banned in (
        "3.7e12",
        "3e11",
        "1e11",
        "2.3e15",
        "1.155e15",
        "103e9",
        "https://",
        "http://",
    ):
        assert banned not in body_js, f"Banned literal: {banned}"


def _build_parity_payloads(
    chip_spec: advisor.ChipSpec,
) -> tuple[
    list[_BatchScenarioItem],
    list[_EvalParityResult],
    list[dict[str, object]],
]:
    """Assemble reference evaluations and payload items for parity run."""
    chip_dict = dataclasses.asdict(chip_spec)
    models_sample = _select_five_models()
    batch_items: list[_BatchScenarioItem] = []
    expected_refs: list[_EvalParityResult] = []
    payload_items: list[dict[str, object]] = []

    for model_name in models_sample:
        prof = models.resolve_model_profile(model_name)
        prof_dict = dataclasses.asdict(prof)
        for wl in _valid_workloads_for_model(prof):
            batch_items.append(
                {"only_validate": False, "model_name": model_name, "wl": wl}
            )
            expected_refs.append(_python_reference_eval(model_name, wl, chip_spec))
            payload_items.append(
                {
                    "params": {
                        "only_validate": False,
                        "profile": prof_dict,
                        "chip": chip_dict,
                        "wl": dataclasses.asdict(wl),
                    }
                }
            )

    for model_name, bad_wl in _invalid_sharding_scenarios():
        prof = models.resolve_model_profile(model_name)
        prof_dict = dataclasses.asdict(prof)
        inv_reasons = list(advisor.validate_workload(prof, chip_spec, bad_wl))
        assert len(inv_reasons) >= 1
        batch_items.append(
            {"only_validate": True, "model_name": model_name, "wl": bad_wl}
        )
        expected_refs.append(
            {
                "invalid_reasons": inv_reasons,
                "fastest_scope": "per_op",
                "recommended_scope": "per_op",
                "scopes": {},
            }
        )
        payload_items.append(
            {
                "params": {
                    "only_validate": True,
                    "profile": prof_dict,
                    "chip": chip_dict,
                    "wl": dataclasses.asdict(bad_wl),
                }
            }
        )
    return batch_items, expected_refs, payload_items


def _run_engine_runner(
    js_runtime: str,
    runner_path: Path,
    in_json: Path,
    out_json: Path,
    timeout_s: float = 60.0,
) -> None:
    """Execute advisor engine runner script using POSIX spawn without shell."""
    argv = [js_runtime, str(runner_path), str(in_json), str(out_json)]
    pid = os.posix_spawnp(js_runtime, argv, os.environ)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        wpid, status = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            rc = os.waitstatus_to_exitcode(status)
            if rc != 0:
                msg = f"Engine runner failed with return code {rc}"
                raise RuntimeError(msg)
            return
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    msg = f"Engine runner timed out after {timeout_s}s"
    raise TimeoutError(msg)


def _verify_parity_records(
    batch_items: list[_BatchScenarioItem],
    expected_refs: list[_EvalParityResult],
    out_records: list[dict[str, object]],
) -> None:
    """Verify each output record against authoritative reference evaluation."""
    for idx, (item, exp, rec) in enumerate(
        zip(batch_items, expected_refs, out_records)
    ):
        label = f"scenario[{idx}] model={item['model_name']}"
        got = rec["eval"]
        if item["only_validate"]:
            assert list(got["invalid_reasons"]) == list(exp["invalid_reasons"]), label
            continue

        assert list(got["invalid_reasons"]) == list(exp["invalid_reasons"]), label
        full_got = got["full"]
        assert full_got["fastestScope"] == exp["fastest_scope"], label
        assert full_got["recScope"] == exp["recommended_scope"], label
        for scope_name, exp_sc in exp["scopes"].items():
            got_sc = got["scopes"][scope_name]
            sc_lbl = f"{label} scope={scope_name}"
            assert got_sc["feasible"] is exp_sc["feasible"], sc_lbl
            assert got_sc["t_vmem_max"] == exp_sc["t_vmem_max"], sc_lbl
            assert got_sc["n_splits"] == exp_sc["n_splits"], sc_lbl
            assert got_sc["selected_nbd"] == exp_sc["selected_nbd"], sc_lbl
            assert got_sc["selected_nbe"] == exp_sc["selected_nbe"], sc_lbl
            assert got_sc["support"] == exp_sc["support"], sc_lbl
            assert list(got_sc["blockers"]) == list(exp_sc["blockers"]), sc_lbl
            assert got_sc["rawStepUs"] == pytest.approx(
                exp_sc["raw_step_us"], rel=_RAW_STEP_REL_TOL, abs=1e-9
            ), f"{sc_lbl} field=rawStepUs"
            assert got_sc["w_step_strict_gb"] == pytest.approx(
                exp_sc["w_step_strict_gb"], abs=1e-4
            ), f"{sc_lbl} field=w_step_strict_gb"
            for disp_key in (
                "step_bound_us",
                "step_floor_strict_us",
                "step_achievable_us",
                "step_achievable_upper_us",
                "act_hbm_mb",
                "comm_us",
            ):
                assert got_sc[disp_key] == pytest.approx(
                    exp_sc[disp_key], abs=_TWO_DECIMAL_ABS_TOL
                ), f"{sc_lbl} field={disp_key}"
            assert got_sc["step_floor_strict_us"] <= got_sc["step_bound_us"]
            assert (
                got_sc["step_bound_us"]
                <= got_sc["step_achievable_us"]
                <= got_sc["step_achievable_upper_us"]
            )


def test_js_engine_runner_executes_static_asset_parity(tmp_path: Path) -> None:
    """Verify exact Python vs JS parity across 80 scenarios via static engine runner."""
    js_runtime = shutil.which("gjs") or shutil.which("node")
    if js_runtime is None:
        pytest.skip("Neither gjs nor node runtime found")

    runner_path = Path(__file__).with_name("advisor_engine_runner.js")
    assert runner_path.is_file(), f"Missing runner at {runner_path}"

    engine_path = Path(models.__file__).parent / "assets" / "workbench_engine.js"
    assert engine_path.is_file(), f"Missing engine at {engine_path}"

    chip_spec = advisor.resolve_chip_spec("tpu7x")
    batch_items, expected_refs, payload_items = _build_parity_payloads(chip_spec)
    assert len(batch_items) == _EXPECTED_SCENARIO_COUNT

    in_json = tmp_path / "parity_in.json"
    out_json = tmp_path / "parity_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": {
                    "chip_spec": dataclasses.asdict(chip_spec),
                    **getattr(
                        workbench_engine, "build_engine_seed_defaults", dict
                    )(),
                },
                "items": payload_items,
            }
        ),
        encoding="utf-8",
    )

    _run_engine_runner(js_runtime, runner_path, in_json, out_json)

    out_records = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(out_records) == len(expected_refs)
    _verify_parity_records(batch_items, expected_refs, out_records)


def test_js_engine_intra_host_dp_and_pp_prefill_parity(tmp_path: Path) -> None:
    """Verify JS engine parity on intra-host DPExchange, DCN EP, and PP prefill."""
    js_runtime = shutil.which("gjs") or shutil.which("node")
    if js_runtime is None:
        pytest.skip("Neither gjs nor node runtime found")

    runner_path = Path(__file__).with_name("advisor_engine_runner.js")
    engine_path = Path(models.__file__).parent / "assets" / "workbench_engine.js"
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    chip_dict = dataclasses.asdict(chip_spec)
    prof = models.resolve_model_profile("qwen3.5-397b")
    prof_dict = dataclasses.asdict(prof)

    workloads = [
        advisor.WorkloadSpec(
            phase="decode",
            batch=384,
            context=32768,
            tp=2,
            dp=4,
            ep=8,
            moe_mode="grouped_on_chip",
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=384,
            context=32768,
            tp=2,
            dp=4,
            ep=8,
            moe_mode="masked_on_chip",
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=768,
            context=32768,
            tp=2,
            dp=8,
            ep=16,
            moe_mode="grouped_on_chip",
        ),
        advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=4096,
            tp=1,
            pp=8,
            ep=1,
            chunk_size=0,
            microbatches=0,
        ),
    ]
    items = [
        {
            "params": {
                "only_validate": False,
                "profile": prof_dict,
                "chip": chip_dict,
                "wl": dataclasses.asdict(wl),
            }
        }
        for wl in workloads
    ]
    in_json = tmp_path / "dp_pp_in.json"
    out_json = tmp_path / "dp_pp_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": {
                    "chip_spec": chip_dict,
                    **getattr(
                        workbench_engine, "build_engine_seed_defaults", dict
                    )(),
                },
                "items": items,
            }
        ),
        encoding="utf-8",
    )
    _run_engine_runner(js_runtime, runner_path, in_json, out_json)
    records = json.loads(out_json.read_text(encoding="utf-8"))

    for wl, rec in zip(workloads, records, strict=True):
        rep = advisor.advise(prof, wl, chip_spec)
        full = rec["eval"]["full"]
        assert full["activeTokens"] == rep.active_tokens
        assert full["recScope"] == rep.recommended_scope
        assert full["moeModeRec"] == rep.recommended_moe_mode
        assert full["compilerProofGainPct"] == pytest.approx(
            float(rep.metrics["compiler_proof_gain_pct"]["value"]),
            abs=_TWO_DECIMAL_ABS_TOL,
        )
        assert full["achievableGainPct"] == pytest.approx(
            float(rep.metrics["achievable_gain_pct"]["value"]),
            abs=_TWO_DECIMAL_ABS_TOL,
        )


def test_js_engine_support_gating_parity(tmp_path: Path) -> None:
    """Verify JS support gating, blockers, strict floor, and recScope parity."""
    js_runtime = shutil.which("gjs") or shutil.which("node")
    if js_runtime is None:
        pytest.skip("Neither gjs nor node runtime found")

    seed_defaults = workbench_engine.build_engine_seed_defaults()
    assert "core_limits" in seed_defaults
    assert "workload_defaults" in seed_defaults

    runner_path = Path(__file__).with_name("advisor_engine_runner.js")
    engine_path = Path(models.__file__).parent / "assets" / "workbench_engine.js"
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    chip_dict = dataclasses.asdict(chip_spec)

    qwen_prof = models.resolve_model_profile("qwen3.5-397b")
    kimi_prof = models.resolve_model_profile("kimi-k3")
    unv_prof = dataclasses.replace(
        qwen_prof,
        name="custom-unverified",
        model_type="future_arch",
        architecture_unverified=True,
    )

    cases = [
        (qwen_prof, advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=1)),
        (qwen_prof, advisor.WorkloadSpec(phase="decode", batch=256, tp=8, ep=1)),
        (kimi_prof, advisor.WorkloadSpec(phase="decode", batch=64, tp=4, dp=8, ep=32)),
        (unv_prof, advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=1)),
        (qwen_prof, advisor.WorkloadSpec(phase="decode", batch=16, tp=6, ep=1)),
    ]
    items = [
        {
            "params": {
                "only_validate": False,
                "profile": dataclasses.asdict(prof),
                "chip": chip_dict,
                "wl": dataclasses.asdict(wl),
            }
        }
        for prof, wl in cases
    ]
    in_json = tmp_path / "support_in.json"
    out_json = tmp_path / "support_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": {"chip_spec": chip_dict, **seed_defaults},
                "items": items,
            }
        ),
        encoding="utf-8",
    )
    _run_engine_runner(js_runtime, runner_path, in_json, out_json)
    records = json.loads(out_json.read_text(encoding="utf-8"))

    for (prof, wl), rec in zip(cases, records, strict=True):
        rep = advisor.advise(prof, wl, chip_spec)
        got = rec["eval"]
        full = got["full"]
        assert full["fastestScope"] == rep.fastest_scope
        assert full["recScope"] == rep.recommended_scope
        for sc in rep.scopes:
            got_sc = got["scopes"][sc.scope]
            assert got_sc["support"] == sc.support
            assert list(got_sc["blockers"]) == list(sc.blockers)
            assert got_sc["step_floor_strict_us"] == pytest.approx(
                sc.step_floor_strict_us, abs=_TWO_DECIMAL_ABS_TOL
            )
            assert got_sc["step_achievable_upper_us"] == pytest.approx(
                sc.step_achievable_upper_us, abs=_TWO_DECIMAL_ABS_TOL
            )


def test_sparse_csa_and_hyper_connection_engine_parity(tmp_path: Path) -> None:
    """Verify JS parity for deepseek-v4 (CSA + mHC) and qwen4-exp (4x residual)."""
    js_runtime = shutil.which("gjs") or shutil.which("node")
    if js_runtime is None:
        pytest.skip("Neither gjs nor node runtime found")

    seed_defaults = workbench_engine.build_engine_seed_defaults()
    runner_path = Path(__file__).with_name("advisor_engine_runner.js")
    engine_path = Path(models.__file__).parent / "assets" / "workbench_engine.js"
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    chip_dict = dataclasses.asdict(chip_spec)

    cases: list[tuple[advisor.ModelProfile, advisor.WorkloadSpec]] = []
    for model_name in ("deepseek-v4", "qwen4-exp"):
        prof = models.resolve_model_profile(model_name)
        cases.extend(
            [
                (prof, advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=1)),
                (
                    prof,
                    advisor.WorkloadSpec(
                        phase="prefill",
                        batch=1,
                        isl=4096,
                        chunk_size=512,
                        tp=8,
                        ep=1,
                    ),
                ),
            ]
        )

    items = [
        {
            "params": {
                "only_validate": False,
                "profile": dataclasses.asdict(prof),
                "chip": chip_dict,
                "wl": dataclasses.asdict(wl),
            }
        }
        for prof, wl in cases
    ]
    in_json = tmp_path / "csa_mhc_in.json"
    out_json = tmp_path / "csa_mhc_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": {"chip_spec": chip_dict, **seed_defaults},
                "items": items,
            }
        ),
        encoding="utf-8",
    )
    _run_engine_runner(js_runtime, runner_path, in_json, out_json)
    records = json.loads(out_json.read_text(encoding="utf-8"))

    for (prof, wl), rec in zip(cases, records, strict=True):
        rep = advisor.advise(prof, wl, chip_spec)
        got = rec["eval"]
        full = got["full"]
        assert full["fastestScope"] == rep.fastest_scope
        assert full["recScope"] == rep.recommended_scope
        for sc in rep.scopes:
            got_sc = got["scopes"][sc.scope]
            assert got_sc["step_bound_us"] == pytest.approx(
                sc.step_bound_us, abs=_TWO_DECIMAL_ABS_TOL
            )
            assert got_sc["step_floor_strict_us"] == pytest.approx(
                sc.step_floor_strict_us, abs=_TWO_DECIMAL_ABS_TOL
            )
            assert got_sc["step_achievable_us"] == pytest.approx(
                sc.step_achievable_us, abs=_TWO_DECIMAL_ABS_TOL
            )
            assert got_sc["step_achievable_upper_us"] == pytest.approx(
                sc.step_achievable_upper_us, abs=_TWO_DECIMAL_ABS_TOL
            )
            assert got_sc["act_hbm_mb"] == pytest.approx(
                sc.act_hbm_mb, abs=_TWO_DECIMAL_ABS_TOL
            )

