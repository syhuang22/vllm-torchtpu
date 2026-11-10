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
"""Optional JAX Pallas TPU probe and verification bridge."""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from jax.experimental.pallas import tpu as pltpu
except (ImportError, RuntimeError):
    pltpu = None

if TYPE_CHECKING:
    from .chips import ChipSpec


def probe_jax_tpu_info() -> tuple[str, int] | None:
    """Probe TPU device version and core count from JAX Pallas if available."""
    if pltpu is None:
        return None
    try:
        info = pltpu.get_tpu_info()
        return str(info.chip_version.value).lower(), int(info.num_cores)
    except (AttributeError, RuntimeError, ValueError):
        return None


def verify_chip_spec_against_jax(spec: ChipSpec) -> list[str]:
    """Compare a built-in ChipSpec against jax.experimental.pallas.tpu if available."""
    if pltpu is None:
        return []
    base_name = spec.name.removesuffix("_megacore")
    cv_str = "7x" if base_name == "tpu7x" else base_name
    try:
        info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion(cv_str), 1)
    except (ValueError, KeyError, RuntimeError, AttributeError):
        return []
    cores = spec.cores_per_device
    mismatches: list[str] = []
    fp8_raw = float(getattr(info, "fp8_ops_per_second", 0) or 0) * cores
    int8_raw = float(getattr(info, "int8_ops_per_second", 0) or 0) * cores
    exp_fp8 = fp8_raw if fp8_raw > 0 else float(spec.bf16_flops_per_s)
    exp_int8 = int8_raw if int8_raw > 0 else float(spec.bf16_flops_per_s)
    checks = (
        (
            "vmem_capacity_bytes",
            spec.vmem_capacity_bytes,
            int(info.vmem_capacity_bytes) * cores,
        ),
        (
            "hbm_capacity_bytes",
            spec.hbm_capacity_bytes,
            int(info.hbm_capacity_bytes) * cores,
        ),
        (
            "hbm_bw_bytes_per_s",
            spec.hbm_bw_bytes_per_s,
            float(info.mem_bw_bytes_per_second) * cores,
        ),
        (
            "bf16_flops_per_s",
            spec.bf16_flops_per_s,
            float(info.bf16_ops_per_second) * cores,
        ),
        (
            "fp8_flops_per_s",
            spec.fp8_flops_per_s,
            exp_fp8,
        ),
        (
            "int8_flops_per_s",
            spec.int8_flops_per_s,
            exp_int8,
        ),
    )
    for field_name, actual, expected in checks:
        if actual != expected:
            mismatches.append(f"{spec.name}.{field_name}: {actual} != {expected}")
    sources = spec.sources or {}
    fp8_src = str(sources.get("fp8_flops_per_s", ""))
    int8_src = str(sources.get("int8_flops_per_s", ""))
    if fp8_raw == 0 and not fp8_src.startswith("DERIVED:"):
        mismatches.append(
            f"{spec.name}.fp8_flops_per_s: missing DERIVED: source prefix"
        )
    if int8_raw == 0 and not int8_src.startswith("DERIVED:"):
        mismatches.append(
            f"{spec.name}.int8_flops_per_s: missing DERIVED: source prefix"
        )
    return mismatches
