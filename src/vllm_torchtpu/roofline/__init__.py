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
"""Analytical TPU serving roofline, step floor, mesh solver, and CLI emitter."""

from __future__ import annotations

from .chips import (
    CHIPS,
    TOPOLOGIES,
    ChipSpec,
    ConstantSpec,
    SliceTopology,
    chip_catalog,
)
from .collectives import (
    Collective,
    collective_cut_bound_s,
    collective_ici_hops,
    submesh_for_group,
    torus_collective_s,
)
from .emit import (
    SOLVE_SCHEMA,
    build_solve_payload,
    emit_ceiling_rows,
    emit_launch_args,
    point_launch_args,
    pool_cli_args,
    top_layouts,
)
from .geometry import (
    LAYER_FAMILIES,
    ModelGeometry,
    UnsupportedArchitectureError,
    from_hf_config,
    layer_ffn_kinds,
    supports,
)
from .hf_loader import dtypes_from_hf_config, load_hf_config
from .modules import (
    Dtypes,
    ModuleCost,
    Parallel,
    StepShape,
    ep_busiest_experts,
    experts_touched,
    kv_bytes_per_token_per_layer,
    pass_costs,
)
from .report import MAX_EMBEDDED_CANDIDATES, render_html, write_html
from .serving import ConcurrencyRow, plan
from .solver import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    EXAMPLE_CONCURRENCIES,
    EXAMPLE_SCENARIOS,
    EXAMPLE_SLO_TIERS,
    FloorTimes,
    HbmBreakdown,
    ParetoPoint,
    PoolMeshPlan,
    SearchStats,
    ServingPlan,
    TrafficScenario,
    build_example_scenario,
    hbm_breakdown_per_core_bytes,
    hbm_footprint_per_core_bytes,
    max_ep_load_ratio,
    partition_pp_layers,
    solve_serving_plan,
    sweep_prefix_hit_crossovers,
)
from .step import StepFloor, spec_cycle_floor, step_floor

__all__ = [
    "CHIPS",
    "DEFAULT_GPU_MEMORY_UTILIZATION",
    "EXAMPLE_CONCURRENCIES",
    "EXAMPLE_SCENARIOS",
    "EXAMPLE_SLO_TIERS",
    "LAYER_FAMILIES",
    "MAX_EMBEDDED_CANDIDATES",
    "SOLVE_SCHEMA",
    "TOPOLOGIES",
    "ChipSpec",
    "Collective",
    "ConcurrencyRow",
    "ConstantSpec",
    "Dtypes",
    "FloorTimes",
    "HbmBreakdown",
    "ModelGeometry",
    "ModuleCost",
    "Parallel",
    "ParetoPoint",
    "PoolMeshPlan",
    "SearchStats",
    "ServingPlan",
    "SliceTopology",
    "StepFloor",
    "StepShape",
    "TrafficScenario",
    "UnsupportedArchitectureError",
    "build_example_scenario",
    "build_solve_payload",
    "chip_catalog",
    "collective_cut_bound_s",
    "collective_ici_hops",
    "dtypes_from_hf_config",
    "emit_ceiling_rows",
    "emit_launch_args",
    "ep_busiest_experts",
    "experts_touched",
    "from_hf_config",
    "hbm_breakdown_per_core_bytes",
    "hbm_footprint_per_core_bytes",
    "kv_bytes_per_token_per_layer",
    "layer_ffn_kinds",
    "load_hf_config",
    "max_ep_load_ratio",
    "partition_pp_layers",
    "pass_costs",
    "plan",
    "point_launch_args",
    "pool_cli_args",
    "render_html",
    "solve_serving_plan",
    "spec_cycle_floor",
    "step_floor",
    "submesh_for_group",
    "supports",
    "sweep_prefix_hit_crossovers",
    "top_layouts",
    "torus_collective_s",
    "write_html",
]
