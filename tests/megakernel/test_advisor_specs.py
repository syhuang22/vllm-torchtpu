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
"""Specifications and profile tests for the TPU megakernel advisor."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from vllm_torchtpu.megakernel.advisor import chips, models, specs
from vllm_torchtpu.megakernel.core import errors

_TPU7X_CORES_PER_DEVICE = 2
_SINGLE_CORE = 1
_BYTES_PER_MIB = 1 << 20
_BYTES_PER_GIB = 1 << 30
_TPU7X_USABLE_MIB = 62
_TPU7X_PHYSICAL_MIB = 64
_CUSTOM_VMEM_USABLE_MIB = 58
_CUSTOM_VMEM_CAP_MIB = 64
_MIN_PRESET_COUNT = 10
_DEEPSEEK_LAYERS = 61
_DEEPSEEK_MOE_LAYERS = 58
_DEEPSEEK_EXPERTS = 256
_DEEPSEEK_TOP_K = 8
_DEEPSEEK_Q_LORA_RANK = 1536
_DEEPSEEK_KV_LORA_RANK = 512
_QWEN35_LAYERS = 60
_QWEN35_ATTN_LAYERS = 15
_QWEN35_LINEAR_LAYERS = 45
_QWEN35_EXPERTS = 512
_QWEN35_TOP_K = 10
_LLAMA_LAYERS = 32
_KIMI_LAYERS = 93
_KIMI_LINEAR_LAYERS = 69
_KIMI_ROUTED_HIDDEN = 3584
_GLM5_LAYERS = 78
_GLM5_MOE_LAYERS = 75
_GLM5_EXPERTS = 256
_TEST_MODEL_LAYERS = 16

_DEEPSEEK_V3_CFG: dict[str, object] = {
    "model_type": "deepseek_v3",
    "num_hidden_layers": 61,
    "first_k_dense_replace": 3,
    "hidden_size": 7168,
    "intermediate_size": 18432,
    "moe_intermediate_size": 2048,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "quantization_config": {"quant_method": "fp8"},
}

_QWEN35_397B_CFG: dict[str, object] = {
    "model_type": "qwen3_5_moe",
    "num_hidden_layers": 60,
    "hidden_size": 4096,
    "intermediate_size": 1024,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "full_attention_interval": 4,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 64,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 1024,
    "shared_expert_intermediate_size": 1024,
    "quantization_config": {"quant_method": "fp8"},
}

_KIMI_K3_PRO_CFG: dict[str, object] = {
    "model_type": "kimi_linear",
    "num_hidden_layers": 93,
    "first_k_dense_replace": 1,
    "hidden_size": 7168,
    "intermediate_size": 33792,
    "moe_intermediate_size": 3072,
    "shared_expert_intermediate_size": 6144,
    "routed_expert_hidden_size": 3584,
    "num_attention_heads": 96,
    "num_key_value_heads": 96,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "num_experts": 896,
    "num_experts_per_token": 16,
    "linear_attn_config": {
        "kda_layers": list(range(69)),
        "num_heads": 96,
        "head_k_dim": 128,
        "head_v_dim": 128,
        "short_conv_kernel_size": 4,
    },
    "quantization_config": {"quant_method": "fp8", "expert_dtype": "mxfp4"},
}


def test_tpu7x_dual_core_defaults() -> None:
    """Verify TPU v7x resolves to dual-core defaults."""
    chip_spec = chips.resolve_chip_spec("tpu7x")
    assert chip_spec.name == "tpu7x"
    assert chip_spec.devices_per_chip == _TPU7X_CORES_PER_DEVICE
    assert chip_spec.cores_per_device == _SINGLE_CORE
    assert chip_spec.vmem_usable_bytes == _TPU7X_USABLE_MIB * _BYTES_PER_MIB
    assert chip_spec.vmem_capacity_bytes == _TPU7X_PHYSICAL_MIB * _BYTES_PER_MIB


def test_single_core_escape_hatch() -> None:
    """Verify single-core escape hatch preserves 1-core spec on TPU v7x."""
    chip_spec = chips.resolve_chip_spec("tpu7x", megacore=False)
    assert chip_spec.name == "tpu7x"
    assert chip_spec.cores_per_device == _SINGLE_CORE
    assert chip_spec.vmem_usable_bytes == _TPU7X_USABLE_MIB * _BYTES_PER_MIB
    assert chip_spec.vmem_capacity_bytes == _TPU7X_PHYSICAL_MIB * _BYTES_PER_MIB


def test_all_preset_chips_load() -> None:
    """Verify all built-in chip specs load and have valid physical parameters."""
    for name in ("tpu7x", "v6e", "v5p", "v5e", "v4"):
        chip_spec = chips.resolve_chip_spec(name)
        assert chip_spec.vmem_capacity_bytes > 0
        assert chip_spec.hbm_capacity_bytes > 0
        assert chip_spec.hbm_bw_bytes_per_s > 0
        assert chip_spec.bf16_flops_per_s > 0


def test_chip_spec_json_override(tmp_path: Path) -> None:
    """Verify loading a custom chip spec JSON overrides hardware parameters."""
    custom_json = tmp_path / "custom_chip.json"
    payload = {
        "name": "custom_v7",
        "cores_per_device": _SINGLE_CORE,
        "vmem_capacity_bytes": _CUSTOM_VMEM_CAP_MIB * _BYTES_PER_MIB,
        "vmem_usable_bytes": _CUSTOM_VMEM_USABLE_MIB * _BYTES_PER_MIB,
        "hbm_capacity_bytes": 128 * _BYTES_PER_GIB,
        "hbm_bw_bytes_per_s": 2400e9,
        "bf16_flops_per_s": 1150e12,
        "fp8_flops_per_s": 2300e12,
        "ici_egress_bw_bytes_per_s": 450e9,
        "ici_link_bw_bytes_per_s": 150e9,
        "collective_latency_us_bracket": [1.2, 1.2],
        "sources": {"spec": "custom user chip spec JSON"},
    }
    custom_json.write_text(json.dumps(payload), encoding="utf-8")
    loaded = chips.resolve_chip_spec("custom", chip_spec_path=custom_json)
    assert loaded.name == "custom_v7"
    assert loaded.vmem_usable_bytes == _CUSTOM_VMEM_USABLE_MIB * _BYTES_PER_MIB
    assert loaded.cores_per_device == _SINGLE_CORE


def test_resolve_chip_spec_unknown_raises() -> None:
    """Verify resolving an unknown chip name raises MegakernelConfigError."""
    with pytest.raises(errors.MegakernelConfigError, match="Unknown chip="):
        chips.resolve_chip_spec("nonexistent_chip_xyz")


def test_all_preset_models_resolve() -> None:
    """Verify all registered model presets resolve to valid ModelProfile."""
    for slug in models.MODEL_PRESETS:
        profile = models.resolve_model_profile(slug)
        assert profile.name == slug
        assert profile.num_layers > 0
        assert profile.hidden_size > 0
        assert profile.num_attention_heads > 0


def test_hf_config_autodiscovery_deepseek_v3() -> None:
    """Verify DeepSeek-V3 HF config auto-discovery sets MLA parameters."""
    profile = models.model_profile_from_hf_config(_DEEPSEEK_V3_CFG)
    assert profile.num_layers == _DEEPSEEK_LAYERS
    assert profile.num_moe_layers == _DEEPSEEK_MOE_LAYERS
    assert profile.num_experts == _DEEPSEEK_EXPERTS
    assert profile.top_k == _DEEPSEEK_TOP_K
    assert profile.q_lora_rank == _DEEPSEEK_Q_LORA_RANK
    assert profile.kv_lora_rank == _DEEPSEEK_KV_LORA_RANK


def test_hf_config_autodiscovery_qwen3_next_hybrid() -> None:
    """Verify Qwen3.5 hybrid linear-attention config autodiscovery."""
    profile = models.model_profile_from_hf_config(_QWEN35_397B_CFG)
    assert profile.num_layers == _QWEN35_LAYERS
    assert profile.num_attention_layers == _QWEN35_ATTN_LAYERS
    assert profile.num_linear_layers == _QWEN35_LINEAR_LAYERS
    assert profile.num_experts == _QWEN35_EXPERTS
    assert profile.top_k == _QWEN35_TOP_K


def test_hf_config_autodiscovery_dense_llama() -> None:
    """Verify dense LLaMA style config autodiscovery."""
    cfg = {
        "model_type": "llama",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    profile = models.model_profile_from_hf_config(cfg)
    assert profile.num_layers == _LLAMA_LAYERS
    assert profile.num_attention_layers == _LLAMA_LAYERS
    assert profile.num_linear_layers == 0
    assert profile.num_experts == 0


def test_hf_config_autodiscovery_kimi_k25_native_int4() -> None:
    """Verify Kimi-K3 nested linear_attn_config and mxfp4 expert quantization."""
    profile = models.model_profile_from_hf_config(_KIMI_K3_PRO_CFG)
    assert profile.num_layers == _KIMI_LAYERS
    assert profile.num_linear_layers == _KIMI_LINEAR_LAYERS
    assert profile.routed_expert_hidden_size == _KIMI_ROUTED_HIDDEN
    assert profile.expert_dtype == "mxfp4"


def test_hf_config_autodiscovery_glm5_mla_dsa() -> None:
    """Verify GLM-5 MLA/DSA architecture config autodiscovery."""
    cfg = {
        "model_type": "glm_moe",
        "num_hidden_layers": 78,
        "first_k_dense_replace": 3,
        "hidden_size": 6144,
        "intermediate_size": 12288,
        "moe_intermediate_size": 2048,
        "num_attention_heads": 96,
        "num_key_value_heads": 96,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "num_experts": 256,
        "num_experts_per_tok": 8,
    }
    profile = models.model_profile_from_hf_config(cfg)
    assert profile.num_layers == _GLM5_LAYERS
    assert profile.num_moe_layers == _GLM5_MOE_LAYERS
    assert profile.num_experts == _GLM5_EXPERTS


def test_preset_models_json_schema() -> None:
    """Verify models.json and deployments.json load and conform to schema."""
    presets_dir = Path(models.__file__).resolve().parent / "data" / "presets"
    models_json = presets_dir / "models.json"
    deployments_json = presets_dir / "deployments.json"
    assert models_json.is_file()
    assert deployments_json.is_file()
    models_data = json.loads(models_json.read_text(encoding="utf-8"))
    assert "presets" in models_data
    assert len(models_data["presets"]) >= _MIN_PRESET_COUNT
    deployments_data = json.loads(deployments_json.read_text(encoding="utf-8"))
    assert "deployments" in deployments_data


def test_detect_sysfs_and_hf_snapshot_accept_injected_paths(tmp_path: Path) -> None:
    """Verify public sysfs and HF snapshot path-injection helpers exist and work."""
    pci_dev = tmp_path / "pci" / "0000:00:04.0"
    pci_dev.mkdir(parents=True)
    (pci_dev / "vendor").write_text("0x1ae0\n", encoding="utf-8")
    (pci_dev / "device").write_text("0x0076\n", encoding="utf-8")
    detected = chips.detect_tpu_from_sysfs(tmp_path / "pci")
    assert detected == "tpu7x"
    chip_spec = chips.resolve_chip_spec(detected)
    assert chip_spec.name == "tpu7x"
    assert chip_spec.devices_per_chip == _TPU7X_CORES_PER_DEVICE

    hf_home = tmp_path / "hf_home"
    repo_dir = hf_home / "hub" / "models--testorg--testmodel"
    snap_dir = repo_dir / "snapshots" / "abc123rev"
    snap_dir.mkdir(parents=True)
    cfg_file = snap_dir / "config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "model_type": "llama",
                "num_hidden_layers": 16,
                "hidden_size": 2048,
                "num_attention_heads": 16,
            }
        ),
        encoding="utf-8",
    )
    ref_main = repo_dir / "refs" / "main"
    ref_main.parent.mkdir(parents=True, exist_ok=True)
    ref_main.write_text("abc123rev\n", encoding="utf-8")

    resolved = models.resolve_hf_snapshot_config("testorg/testmodel", hf_home=hf_home)
    assert resolved == cfg_file
    profile = models.resolve_model_profile(resolved)
    assert profile.num_layers == _TEST_MODEL_LAYERS


_TPU7X_INT8_BF16_FLOPS = 1.155e15
_DEEPSEEK_V4_COMPRESS_RATIO = 0.325
_RESIDUAL_STREAMS_4 = 4
_DENSE_PREFIX_MOE_LAYERS_3 = 3
_SOURCE_PREFIXES = ("FACT:", "DERIVED:", "ASSUMED:", "CHOSEN:")


def test_tpu7x_int8_peak_equals_bf16_rate() -> None:
    """Verify tpu7x int8 peak FLOP/s equals the BF16 rate (1155 TFLOP/s)."""
    c7x = chips.resolve_chip_spec("tpu7x")
    assert c7x.int8_flops_per_s == c7x.bf16_flops_per_s
    assert c7x.peak_flops("int8") == _TPU7X_INT8_BF16_FLOPS


def test_chip_sources_cover_every_field() -> None:
    """Verify every built-in ChipSpec value field has a prefixed source entry."""
    skip = {"name", "topologies", "sources", "tpu_info"}
    value_fields = {f.name for f in dataclasses.fields(chips.ChipSpec)} - skip
    for name, spec in chips.BUILTIN_CHIP_SPECS.items():
        assert spec.sources is not None, f"{name} missing sources"
        assert value_fields <= set(spec.sources.keys()), (
            f"{name} missing field sources: {value_fields - set(spec.sources.keys())}"
        )
        for key in value_fields:
            src_val = spec.sources[key]
            assert src_val.startswith(_SOURCE_PREFIXES), (
                f"{name}.{key} source missing class prefix: {src_val!r}"
            )
    for legacy in ("v5p", "v5e", "v4"):
        assert chips.BUILTIN_CHIP_SPECS[legacy].topologies == {}


def test_jax_bridge_checks_every_exposed_field() -> None:
    """Verify verify_chip_spec_against_jax checks all 6 tpu_info fields."""
    for name in ("tpu7x", "v6e", "v5p", "v5e", "v4"):
        cs = chips.resolve_chip_spec(name)
        assert chips.verify_chip_spec_against_jax(cs) == []

    bad_7x = dataclasses.replace(
        chips.resolve_chip_spec("tpu7x"), int8_flops_per_s=2.31e15
    )
    mismatches = chips.verify_chip_spec_against_jax(bad_7x)
    assert any("int8_flops_per_s" in m for m in mismatches)


def test_dense_prefix_predicate_matches_non_homogeneous_reasons() -> None:
    """Verify dense_prefix_moe is derived when 0 < moe_layers < num_layers."""
    cfg = {
        **_DEEPSEEK_V3_CFG,
        "num_hidden_layers": _RESIDUAL_STREAMS_4,
        "first_k_dense_replace": 1,
    }
    prof = models.model_profile_from_hf_config(cfg)
    assert prof.num_layers == _RESIDUAL_STREAMS_4
    assert prof.num_moe_layers == _DENSE_PREFIX_MOE_LAYERS_3
    assert "dense_prefix_moe" in prof.layer_families


def test_load_measurements_bundled_uses_requested_chip() -> None:
    """Verify load_measurements('bundled', chip_name=...) uses requested chip."""
    if chips.public_calibration_path("tpu7x") is not None:
        m7x = specs.load_measurements("bundled", chip_name="tpu7x")
        assert m7x.source_kind == "architectural_prior"
    with pytest.raises(errors.MegakernelConfigError, match="v6e"):
        specs.load_measurements("bundled", chip_name="v6e")


def test_hf_config_autodiscovery_deepseek_v4_csa_and_mhc() -> None:
    """Verify deepseek_v4 CSA layer token, compress_ratios, and mhc_num_streams."""
    cfg: dict[str, object] = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 4,
        "hidden_size": 7168,
        "intermediate_size": 18432,
        "moe_intermediate_size": 2048,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "layer_types": ["mla", "csa", "csa", "csa"],
        "compress_ratios": [1.0, 0.1, 0.1, 0.1],
        "mhc_num_streams": _RESIDUAL_STREAMS_4,
    }
    prof = models.model_profile_from_hf_config(cfg)
    assert prof.architecture_unverified is False
    assert prof.kv_compress_factor == pytest.approx(_DEEPSEEK_V4_COMPRESS_RATIO)
    assert prof.residual_streams == _RESIDUAL_STREAMS_4
    assert "sparse_csa" in prof.layer_families
    assert "hyper_connection" in prof.layer_families


def test_hf_config_autodiscovery_qwen4_exp_hyper_connections() -> None:
    """Verify qwen4_exp HCA layer token and hc_count residual streams."""
    cfg: dict[str, object] = {
        "model_type": "qwen4_exp",
        "num_hidden_layers": 4,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "layer_types": ["hca", "gdn", "gdn", "gdn"],
        "hc_count": _RESIDUAL_STREAMS_4,
    }
    prof = models.model_profile_from_hf_config(cfg)
    assert prof.architecture_unverified is False
    assert prof.residual_streams == _RESIDUAL_STREAMS_4
    assert "hyper_connection" in prof.layer_families
