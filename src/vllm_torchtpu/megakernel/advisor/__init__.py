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
"""First-principles fusion-scope advisor for TPU megakernel evaluation."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..advisor.chips import (
        BUILTIN_CHIP_SPECS,
        REQUIRED_CHIP_SPEC_KEYS,
        TPU6E_TOPOLOGIES,
        TPU7X_TOPOLOGIES,
        ChipSpec,
        FieldSource,
        detect_tpu_from_sysfs,
        load_chip_spec_file,
        load_public_calibration,
        public_calibration_path,
        resolve_chip_spec,
        verify_chip_spec_against_jax,
    )
    from ..advisor.cli import (
        build_arg_parser,
        main,
    )
    from ..advisor.models import (
        DEPLOYMENT_PRESET_SOURCES,
        DEPLOYMENT_PRESETS,
        MODEL_PRESET_SOURCES,
        MODEL_PRESETS,
        RECOGNIZED_MODEL_TYPES,
        REGISTERED_MODEL_ALIASES,
        SUPPORTED_MODEL_TYPES,
        DeploymentPreset,
        ModelProfile,
        ParallelLayout,
        UnsupportedArchitectureError,
        model_profile_from_hf_config,
        resolve_hf_snapshot_config,
        resolve_model_profile,
        supports_hf_config,
    )
    from ..advisor.regimes import (
        advise_regimes,
        compare_parallelism_candidates,
        primary_step_mode,
        recommend_parallelism,
    )
    from ..advisor.report import (
        format_advisor_summary,
        format_regime_table,
        render_advisor_html,
        render_regime_dot_plot_svg,
        svg_text_width,
        write_advisor_html,
    )
    from ..advisor.roofline import (
        validate_workload,
    )
    from ..advisor.spec_bridge import (
        advise_from_spec,
    )
    from ..advisor.specs import (
        BUCKET_CONSUMER_STATUS,
        BUCKET_SCHEDULE_SCHEMA,
        SUPPORTED_ONCHIP_FAMILIES,
        AdvisorReport,
        AdvisorSweepReport,
        BucketSchedulePayload,
        BucketScheduleRule,
        DisaggHandoffSpec,
        DispatchRegime,
        EfficiencySpec,
        FusionScope,
        MeasurementsData,
        MoEMode,
        ObjectiveMode,
        ParallelismRecommendation,
        Phase,
        ProvenanceMetric,
        ProvenanceTag,
        RegimeEntry,
        RingPlan,
        ScopeEstimate,
        SupportStatus,
        WorkloadSpec,
        default_double_buffer_ring_plan,
        load_measurements,
    )
    from ..advisor.support import (
        CORE_LIMITS,
        CoreLimits,
        evaluate_support,
    )
    from ..advisor.verdict import (
        advise,
        compute_scope_raw_step_us,
    )
    from ..advisor.vmem import (
        compute_all_reduce_transfer_cost,
        compute_per_token_vmem_bytes,
        fit_vmem_budget_bytes,
        validate_advise_inputs,
    )

_SUBMODULES = frozenset(
    {
        "assets",
        "chips",
        "cli",
        "jax_bridge",
        "models",
        "regimes",
        "report",
        "roofline",
        "spec_bridge",
        "specs",
        "support",
        "verdict",
        "vmem",
    }
)

_LAZY = {
    # chips
    "BUILTIN_CHIP_SPECS": ".chips",
    "ChipSpec": ".chips",
    "FieldSource": ".chips",
    "REQUIRED_CHIP_SPEC_KEYS": ".chips",
    "TPU6E_TOPOLOGIES": ".chips",
    "TPU7X_TOPOLOGIES": ".chips",
    "detect_tpu_from_sysfs": ".chips",
    "load_chip_spec_file": ".chips",
    "load_public_calibration": ".chips",
    "public_calibration_path": ".chips",
    "resolve_chip_spec": ".chips",
    "verify_chip_spec_against_jax": ".chips",
    # cli
    "build_arg_parser": ".cli",
    "main": ".cli",
    # models
    "DEPLOYMENT_PRESETS": ".models",
    "DEPLOYMENT_PRESET_SOURCES": ".models",
    "DeploymentPreset": ".models",
    "MODEL_PRESETS": ".models",
    "MODEL_PRESET_SOURCES": ".models",
    "ModelProfile": ".models",
    "ParallelLayout": ".models",
    "RECOGNIZED_MODEL_TYPES": ".models",
    "REGISTERED_MODEL_ALIASES": ".models",
    "SUPPORTED_MODEL_TYPES": ".models",
    "UnsupportedArchitectureError": ".models",
    "model_profile_from_hf_config": ".models",
    "resolve_hf_snapshot_config": ".models",
    "resolve_model_profile": ".models",
    "supports_hf_config": ".models",
    # regimes
    "advise_regimes": ".regimes",
    "compare_parallelism_candidates": ".regimes",
    "primary_step_mode": ".regimes",
    "recommend_parallelism": ".regimes",
    # report
    "format_advisor_summary": ".report",
    "format_regime_table": ".report",
    "render_advisor_html": ".report",
    "render_regime_dot_plot_svg": ".report",
    "svg_text_width": ".report",
    "write_advisor_html": ".report",
    # roofline
    "validate_workload": ".roofline",
    # spec_bridge
    "advise_from_spec": ".spec_bridge",
    # specs
    "AdvisorReport": ".specs",
    "AdvisorSweepReport": ".specs",
    "BUCKET_CONSUMER_STATUS": ".specs",
    "BUCKET_SCHEDULE_SCHEMA": ".specs",
    "BucketSchedulePayload": ".specs",
    "BucketScheduleRule": ".specs",
    "DisaggHandoffSpec": ".specs",
    "DispatchRegime": ".specs",
    "EfficiencySpec": ".specs",
    "FusionScope": ".specs",
    "MeasurementsData": ".specs",
    "MoEMode": ".specs",
    "ObjectiveMode": ".specs",
    "ParallelismRecommendation": ".specs",
    "Phase": ".specs",
    "ProvenanceMetric": ".specs",
    "ProvenanceTag": ".specs",
    "RegimeEntry": ".specs",
    "RingPlan": ".specs",
    "SUPPORTED_ONCHIP_FAMILIES": ".specs",
    "ScopeEstimate": ".specs",
    "SupportStatus": ".specs",
    "WorkloadSpec": ".specs",
    "default_double_buffer_ring_plan": ".specs",
    "load_measurements": ".specs",
    # support
    "CORE_LIMITS": ".support",
    "CoreLimits": ".support",
    "evaluate_support": ".support",
    # verdict
    "advise": ".verdict",
    "compute_scope_raw_step_us": ".verdict",
    # vmem
    "compute_all_reduce_transfer_cost": ".vmem",
    "compute_per_token_vmem_bytes": ".vmem",
    "fit_vmem_budget_bytes": ".vmem",
    "validate_advise_inputs": ".vmem",
}


def __getattr__(name: str) -> object:
    """Resolve top-level advisor symbols lazily upon first attribute access."""
    if name in _SUBMODULES:
        mod = importlib.import_module(f".{name}", __package__)
        globals()[name] = mod
        return mod
    submod = _LAZY.get(name)
    if submod is not None:
        mod = importlib.import_module(submod, __package__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """Return available public attributes and submodules."""
    return sorted(set(globals()) | set(_SUBMODULES) | set(_LAZY))


__all__ = [
    "BUCKET_CONSUMER_STATUS",
    "BUCKET_SCHEDULE_SCHEMA",
    "BUILTIN_CHIP_SPECS",
    "CORE_LIMITS",
    "DEPLOYMENT_PRESETS",
    "DEPLOYMENT_PRESET_SOURCES",
    "MODEL_PRESETS",
    "MODEL_PRESET_SOURCES",
    "RECOGNIZED_MODEL_TYPES",
    "REGISTERED_MODEL_ALIASES",
    "REQUIRED_CHIP_SPEC_KEYS",
    "SUPPORTED_MODEL_TYPES",
    "SUPPORTED_ONCHIP_FAMILIES",
    "TPU6E_TOPOLOGIES",
    "TPU7X_TOPOLOGIES",
    "AdvisorReport",
    "AdvisorSweepReport",
    "BucketSchedulePayload",
    "BucketScheduleRule",
    "ChipSpec",
    "CoreLimits",
    "DeploymentPreset",
    "DisaggHandoffSpec",
    "DispatchRegime",
    "EfficiencySpec",
    "FieldSource",
    "FusionScope",
    "MeasurementsData",
    "MoEMode",
    "ModelProfile",
    "ObjectiveMode",
    "ParallelLayout",
    "ParallelismRecommendation",
    "Phase",
    "ProvenanceMetric",
    "ProvenanceTag",
    "RegimeEntry",
    "RingPlan",
    "ScopeEstimate",
    "SupportStatus",
    "UnsupportedArchitectureError",
    "WorkloadSpec",
    "advise",
    "advise_from_spec",
    "advise_regimes",
    "build_arg_parser",
    "compare_parallelism_candidates",
    "compute_all_reduce_transfer_cost",
    "compute_per_token_vmem_bytes",
    "compute_scope_raw_step_us",
    "default_double_buffer_ring_plan",
    "detect_tpu_from_sysfs",
    "evaluate_support",
    "fit_vmem_budget_bytes",
    "format_advisor_summary",
    "format_regime_table",
    "load_chip_spec_file",
    "load_measurements",
    "load_public_calibration",
    "main",
    "model_profile_from_hf_config",
    "primary_step_mode",
    "public_calibration_path",
    "recommend_parallelism",
    "render_advisor_html",
    "render_regime_dot_plot_svg",
    "resolve_chip_spec",
    "resolve_hf_snapshot_config",
    "resolve_model_profile",
    "supports_hf_config",
    "svg_text_width",
    "validate_advise_inputs",
    "validate_workload",
    "verify_chip_spec_against_jax",
    "write_advisor_html",
]
