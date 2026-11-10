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
"""Per-core hardware specifications, constants catalog, and 3D slice topologies."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

try:
    import jax as _jax
    from jax.experimental.pallas import tpu as _pltpu
except ImportError:
    pltpu: Any = None
    JAX_VERSION: str = "stdlib-fallback"
else:
    pltpu = _pltpu
    JAX_VERSION = str(_jax.__version__)

_MIN_TORUS_DIM: int = 4
DEFAULT_CORES_PER_CHIP: int = 2
TPU7X_ICI_LINKS_PER_CHIP: int = 6
TPU7X_ICI_BIDIR_LINK_BW: float = 200.0e9
DEFAULT_ICI_LINK_BW: float = TPU7X_ICI_BIDIR_LINK_BW / 2.0
DEFAULT_D2D_BW: float = float(TPU7X_ICI_LINKS_PER_CHIP) * DEFAULT_ICI_LINK_BW
DEFAULT_ICI_BW: float = DEFAULT_D2D_BW / float(DEFAULT_CORES_PER_CHIP)
DEFAULT_ICI_HOP_LATENCY_S: float = 1.0e-6
DEFAULT_MXU_M_TILE: int = 256
DEFAULT_DCN_HOST_GBIT_PER_S: float = 400.0
DEFAULT_CHIPS_PER_HOST: int = 4
BITS_PER_BYTE: float = 8.0
DEFAULT_DCN_BW: float = (DEFAULT_DCN_HOST_GBIT_PER_S * 1.0e9 / BITS_PER_BYTE) / float(
    DEFAULT_CHIPS_PER_HOST
)
DEFAULT_PCIE_BW: float = 64.0e9
PAGE_TABLE_BYTE_BUDGET: int = 512 * 1024
RPA_PAGE_SIZE_TOKENS: int = 64
DEFAULT_GPU_MEMORY_UTILIZATION: float = 0.92


@dataclass(frozen=True)
class ConstantSpec:
    """Metadata for one hardware or solver constant in the roofline catalog."""

    value: float | int
    unit: str
    scope: str
    kind: str
    source: str
    bracket: tuple[float, float] | None = None


@dataclass(frozen=True)
class ChipSpec:
    """Hardware ceiling parameters for one TensorCore and its parent chip/host."""

    name: str
    hbm_bytes: int
    hbm_bw: float
    bf16_flops: float
    fp8_flops: float
    vmem_bytes: int
    ici_bw: float = math.inf
    cores_per_chip: int = DEFAULT_CORES_PER_CHIP
    ici_link_bw: float = DEFAULT_ICI_LINK_BW
    d2d_bw: float = DEFAULT_D2D_BW
    ici_hop_latency_s: float = DEFAULT_ICI_HOP_LATENCY_S
    mxu_m_tile: int = DEFAULT_MXU_M_TILE
    dcn_host_gbit_per_s: float = DEFAULT_DCN_HOST_GBIT_PER_S
    chips_per_host: int = DEFAULT_CHIPS_PER_HOST
    dcn_bw: float = DEFAULT_DCN_BW
    pcie_bw: float = DEFAULT_PCIE_BW

    def catalog(self) -> dict[str, dict[str, Any]]:
        """Return the constants catalog for this ChipSpec instance."""
        return chip_catalog(self)


def chip_catalog(chip: ChipSpec) -> dict[str, dict[str, Any]]:
    """Build structured constants catalog with value, unit, scope, and source."""
    dcn_low = (200.0 * 1.0e9 / BITS_PER_BYTE) / float(max(1, chip.chips_per_host))
    dcn_high = (800.0 * 1.0e9 / BITS_PER_BYTE) / float(max(1, chip.chips_per_host))
    entries: dict[str, ConstantSpec] = {
        "hbm_bytes": ConstantSpec(
            value=chip.hbm_bytes,
            unit="B",
            scope="core",
            kind="fact",
            source="jax._src.tpu_info (ChipVersion.TPU_7X hbm_capacity_bytes)",
        ),
        "hbm_bw": ConstantSpec(
            value=chip.hbm_bw,
            unit="B/s",
            scope="core",
            kind="fact",
            source="jax._src.tpu_info (ChipVersion.TPU_7X mem_bw_bytes_per_second)",
        ),
        "bf16_flops": ConstantSpec(
            value=chip.bf16_flops,
            unit="FLOP/s",
            scope="core",
            kind="fact",
            source="jax._src.tpu_info (ChipVersion.TPU_7X bf16_ops_per_second)",
        ),
        "fp8_flops": ConstantSpec(
            value=chip.fp8_flops,
            unit="FLOP/s",
            scope="core",
            kind="fact",
            source="jax._src.tpu_info (ChipVersion.TPU_7X fp8_ops_per_second)",
        ),
        "vmem_bytes": ConstantSpec(
            value=chip.vmem_bytes,
            unit="B",
            scope="core",
            kind="fact",
            source="jax._src.tpu_info (ChipVersion.TPU_7X vmem_capacity_bytes)",
        ),
        "cores_per_chip": ConstantSpec(
            value=chip.cores_per_chip,
            unit="cores",
            scope="chip",
            kind="fact",
            source="jax._src.tpu_info (ChipVersion.TPU_7X tensor_cores_per_chip)",
        ),
        "ici_link_bw": ConstantSpec(
            value=chip.ici_link_bw,
            unit="B/s",
            scope="link",
            kind="derived",
            source=(
                "200 GB/s bidirectional link / 2 directions "
                "(cloud.google.com/tpu/docs/tpu7x)"
            ),
        ),
        "d2d_bw": ConstantSpec(
            value=chip.d2d_bw,
            unit="B/s",
            scope="chip",
            kind="derived",
            source=(
                "6 links * ici_link_bw across chip (cloud.google.com/tpu/docs/tpu7x)"
            ),
        ),
        "ici_bw": ConstantSpec(
            value=chip.ici_bw,
            unit="B/s",
            scope="core",
            kind="derived",
            source=(
                "d2d_bw / cores_per_chip per-core ICI share "
                "(cloud.google.com/tpu/docs/tpu7x)"
            ),
        ),
        "dcn_bw": ConstantSpec(
            value=chip.dcn_bw,
            unit="B/s",
            scope="chip",
            kind="derived",
            source=(
                "dcn_host_gbit_per_s * 1e9 / 8 / chips_per_host "
                "(400 Gbit/s / 8 / 4 chips)"
            ),
            bracket=(dcn_low, dcn_high),
        ),
        "ici_hop_latency_s": ConstantSpec(
            value=chip.ici_hop_latency_s,
            unit="s",
            scope="link",
            kind="assumed",
            source=(
                "Scaling Book TPU v5e proxy "
                "(jax-ml.github.io/scaling-book/tpus); no TPU7x value published"
            ),
            bracket=(0.5e-6, 2.0e-6),
        ),
        "pcie_bw": ConstantSpec(
            value=chip.pcie_bw,
            unit="B/s",
            scope="chip",
            kind="assumed",
            source=(
                "PCIe 5.0 x16 raw link rate (64 GB/s per chip); "
                "host-to-device effective rate uncalibrated"
            ),
            bracket=(32.0e9, 64.0e9),
        ),
        "mxu_m_tile": ConstantSpec(
            value=chip.mxu_m_tile,
            unit="rows",
            scope="core",
            kind="assumed",
            source=(
                "Assumed Pallas MoE kernel M-tile multiple on 256x256 MXU "
                "(jax._src.tpu_info mxu_column_size=256)"
            ),
            bracket=(128.0, 256.0),
        ),
        "dcn_host_gbit_per_s": ConstantSpec(
            value=chip.dcn_host_gbit_per_s,
            unit="Gbit/s",
            scope="host",
            kind="assumed",
            source=(
                "Public machine-type search summary (400 Gbit/s per 4-chip host); "
                "not verified offline"
            ),
            bracket=(200.0, 800.0),
        ),
        "chips_per_host": ConstantSpec(
            value=chip.chips_per_host,
            unit="chips",
            scope="host",
            kind="assumed",
            source="Public machine-type search summary (4 TPU7x chips per host)",
            bracket=(4.0, 4.0),
        ),
        "page_table_byte_budget": ConstantSpec(
            value=PAGE_TABLE_BYTE_BUDGET,
            unit="B",
            scope="core",
            kind="assumed",
            source=(
                "Assumed VMEM page-table scratch budget "
                "(512 KiB per core; vllm_torchtpu.roofline.solver)"
            ),
            bracket=(262144.0, 1048576.0),
        ),
        "rpa_page_size_tokens": ConstantSpec(
            value=RPA_PAGE_SIZE_TOKENS,
            unit="tokens",
            scope="core",
            kind="assumed",
            source=(
                "Assumed RPA KV page size for KV head counts <= 8 "
                "(64 tokens; 32 above 8; vllm_torchtpu.roofline.solver)"
            ),
            bracket=(16.0, 64.0),
        ),
        "default_gpu_memory_utilization": ConstantSpec(
            value=DEFAULT_GPU_MEMORY_UTILIZATION,
            unit="ratio",
            scope="core",
            kind="config",
            source=(
                "Roofline default --gpu-memory-utilization "
                "(vLLM CacheConfig default is 0.90)"
            ),
            bracket=(0.80, 0.95),
        ),
    }
    return {k: asdict(v) for k, v in entries.items()}


_FALLBACK_TPU_SPECS: dict[str, tuple[int, float, float, float, int]] = {
    # (hbm_bytes, hbm_bw, bf16_flops, fp8_flops, vmem_bytes) per TensorCore
    # matching jax._src.tpu_info for ChipVersion.TPU_7X.
    "tpu7x": (103_000_000_000, 3.70e12, 1.155e15, 2.30e15, 67_108_864),
}


def from_jax(  # noqa: PLR0913
    name: str,
    version: object,
    *,
    cores_per_chip: int = DEFAULT_CORES_PER_CHIP,
    ici_link_bw: float = DEFAULT_ICI_LINK_BW,
    d2d_bw: float | None = None,
    ici_bw: float | None = None,
    ici_hop_latency_s: float = DEFAULT_ICI_HOP_LATENCY_S,
    mxu_m_tile: int = DEFAULT_MXU_M_TILE,
    dcn_host_gbit_per_s: float = DEFAULT_DCN_HOST_GBIT_PER_S,
    chips_per_host: int = DEFAULT_CHIPS_PER_HOST,
    dcn_bw: float | None = None,
    pcie_bw: float = DEFAULT_PCIE_BW,
) -> ChipSpec:
    """Construct ChipSpec from JAX TPU info table or stdlib fallback table."""
    eff_d2d_bw = (
        d2d_bw if d2d_bw is not None else float(TPU7X_ICI_LINKS_PER_CHIP) * ici_link_bw
    )
    eff_ici_bw = (
        ici_bw if ici_bw is not None else eff_d2d_bw / float(max(1, cores_per_chip))
    )
    eff_dcn_bw = (
        dcn_bw
        if dcn_bw is not None
        else (dcn_host_gbit_per_s * 1.0e9 / BITS_PER_BYTE)
        / float(max(1, chips_per_host))
    )
    if pltpu is None or type(version) is str:
        hbm_b, hbm_bw, bf16, fp8, vmem_b = _FALLBACK_TPU_SPECS[name]
        return ChipSpec(
            name=name,
            hbm_bytes=hbm_b,
            hbm_bw=hbm_bw,
            bf16_flops=bf16,
            fp8_flops=fp8,
            vmem_bytes=vmem_b,
            cores_per_chip=cores_per_chip,
            ici_link_bw=ici_link_bw,
            d2d_bw=eff_d2d_bw,
            ici_bw=eff_ici_bw,
            ici_hop_latency_s=ici_hop_latency_s,
            mxu_m_tile=mxu_m_tile,
            dcn_host_gbit_per_s=dcn_host_gbit_per_s,
            chips_per_host=chips_per_host,
            dcn_bw=eff_dcn_bw,
            pcie_bw=pcie_bw,
        )
    info = pltpu.get_tpu_info_for_chip(version, 1)
    return ChipSpec(
        name=name,
        hbm_bytes=info.hbm_capacity_bytes,
        hbm_bw=float(info.mem_bw_bytes_per_second),
        bf16_flops=float(info.bf16_ops_per_second),
        fp8_flops=float(info.fp8_ops_per_second),
        vmem_bytes=info.vmem_capacity_bytes,
        cores_per_chip=cores_per_chip,
        ici_link_bw=ici_link_bw,
        d2d_bw=eff_d2d_bw,
        ici_bw=eff_ici_bw,
        ici_hop_latency_s=ici_hop_latency_s,
        mxu_m_tile=mxu_m_tile,
        dcn_host_gbit_per_s=dcn_host_gbit_per_s,
        chips_per_host=chips_per_host,
        dcn_bw=eff_dcn_bw,
        pcie_bw=pcie_bw,
    )


@dataclass(frozen=True)
class SliceTopology:
    """Physical 3D chip slice topology and torus wraparound rules."""

    name: str
    chip_axes: tuple[int, int, int]
    cores_per_chip: int = DEFAULT_CORES_PER_CHIP

    @property
    def num_chips(self) -> int:
        """Total number of physical chips in the 3D topology."""
        a, b, c = self.chip_axes
        return a * b * c

    @property
    def num_cores(self) -> int:
        """Total number of TensorCores across all chips in the topology."""
        return self.num_chips * self.cores_per_chip

    @property
    def has_torus_wraparound(self) -> bool:
        """True when all physical chip axes are multiples of 4."""
        return all(
            d >= _MIN_TORUS_DIM and d % _MIN_TORUS_DIM == 0 for d in self.chip_axes
        )

    def axis_wrapped(self, axis: int, submesh_extent: int) -> bool:
        """True when collective spans full extent of wrapped physical axis."""
        return (
            self.has_torus_wraparound
            and submesh_extent == self.chip_axes[axis]
            and submesh_extent >= _MIN_TORUS_DIM
        )


CHIPS: dict[str, ChipSpec] = {
    "tpu7x": from_jax(
        "tpu7x",
        pltpu.ChipVersion.TPU_7X if pltpu is not None else "tpu7x",
    ),
}

TOPOLOGIES: dict[str, SliceTopology] = {
    "1x1x1": SliceTopology("1x1x1", (1, 1, 1)),
    "1x2x1": SliceTopology("1x2x1", (1, 2, 1)),
    "2x2x1": SliceTopology("2x2x1", (2, 2, 1)),
    "2x2x2": SliceTopology("2x2x2", (2, 2, 2)),
    "2x2x4": SliceTopology("2x2x4", (2, 2, 4)),
    "2x4x4": SliceTopology("2x4x4", (2, 4, 4)),
    "4x4x4": SliceTopology("4x4x4", (4, 4, 4)),
}
