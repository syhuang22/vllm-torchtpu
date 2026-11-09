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
"""CPU unit tests for Muse Spark speculative decoding draft models."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from vllm.config import VllmConfig
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP

from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.muse_spark.configuration import (
    MuseSparkDSparkConfig,
    register_muse_spark_configs,
)
from vllm_torchtpu.models.vllm.muse_spark.dflash import MuseSparkDFlash2Model
from vllm_torchtpu.models.vllm.muse_spark.dspark import (
    MuseSparkDSparkModel,
    _remap_dspark_name,
)
from vllm_torchtpu.models.vllm.muse_spark.text_model import (
    MuseSparkForCausalLM,
    TPUMuseSparkConfig,
)


@pytest.fixture(autouse=True)
def _cpu_model_tests(monkeypatch):
    tpu = getattr(torch, "tpu", None)
    if tpu is None:
        return
    monkeypatch.setattr(tpu, "is_available", lambda: False)
    monkeypatch.setattr(tpu, "current_device", lambda: 0)
    monkeypatch.setattr(tpu, "device_count", lambda: 0)


@pytest.fixture
def _single_rank_dist(monkeypatch):
    import socket

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_name", "cpu", raising=False)
    monkeypatch.setattr(current_platform, "device_type", "cpu", raising=False)
    if not torch.distributed.is_initialized():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="gloo",
        )
    with set_current_vllm_config(VllmConfig()):
        ensure_model_parallel_initialized(1, 1)


def test_spec_architectures_are_registered() -> None:
    register_models()
    register_muse_spark_configs()

    assert CONFIG_MAPPING["muse_spark_dspark"] is MuseSparkDSparkConfig
    assert (
        ModelRegistry.models["MuseSparkDSparkModel"].load_model_cls()
        is MuseSparkDSparkModel
    )
    assert (
        ModelRegistry.models["MuseSparkDFlash2Model"].load_model_cls()
        is MuseSparkDFlash2Model
    )
    assert MODELS_CONFIG_MAP["MuseSparkDSparkModel"] is TPUMuseSparkConfig


def test_muse_spark_dspark_config_defaults() -> None:
    cfg = MuseSparkDSparkConfig()
    assert cfg.model_type == "muse_spark_dspark"
    assert cfg.hidden_size == 8192
    assert cfg.intermediate_size == 16384
    assert cfg.num_hidden_layers == 5
    assert cfg.num_attention_heads == 128
    assert cfg.num_key_value_heads == 16
    assert cfg.head_dim == 64
    assert cfg.sliding_window == 2048
    assert cfg.target_layer_ids == [13, 29, 45, 57, 61]
    assert cfg.block_size == 7
    assert cfg.markov_rank == 512
    assert cfg.sample_from_anchor is True
    assert cfg.dflash_config["mask_token_id"] == 200018
    assert cfg.dflash_config["target_layer_ids"] == [13, 29, 45, 57, 61]
    assert cfg.dflash_config["causal"] is True


def test_remap_dspark_weights() -> None:
    assert _remap_dspark_name("fc.weight") == "model.fc.weight"
    assert _remap_dspark_name("hidden_norm.weight") == "model.hidden_norm.weight"
    assert _remap_dspark_name("norm.weight") == "model.norm.weight"
    assert _remap_dspark_name("lm_head.weight") == "lm_head.weight"
    assert _remap_dspark_name("d2t") == "model.d2t"
    assert (
        _remap_dspark_name("markov_head.markov_w1.weight")
        == "model.markov_head.markov_w1.weight"
    )
    assert _remap_dspark_name("confidence_head.proj.weight") is None

    # Layer weights
    assert (
        _remap_dspark_name("layers.0.self_attn.pre_qkv_proj_norm.weight")
        == "model.layers.0.input_layernorm.weight"
    )
    assert (
        _remap_dspark_name("layers.0.self_attn.q_proj.weight")
        == "model.layers.0.self_attn.q_proj.weight"
    )
    assert (
        _remap_dspark_name("layers.0.add_attn.gate")
        == "model.layers.0.post_attention_residual_gate.gate"
    )
    assert (
        _remap_dspark_name("layers.0.pre_ffn_norm.weight")
        == "model.layers.0.pre_feedforward_layernorm.weight"
    )
    assert (
        _remap_dspark_name("layers.0.feed_forward.gate_proj.weight")
        == "model.layers.0.feed_forward.gate_proj.weight"
    )
    assert (
        _remap_dspark_name("layers.0.post_ffn_norm.weight")
        == "model.layers.0.post_feedforward_layernorm.weight"
    )
    assert (
        _remap_dspark_name("layers.0.add_ffn.gate")
        == "model.layers.0.post_feedforward_residual_gate.gate"
    )


def test_set_aux_hidden_state_layers_normalization() -> None:
    mock_model = mock.MagicMock()
    mock_model.layers = [mock.MagicMock() for _ in range(62)]
    causal_lm = mock.MagicMock(spec=MuseSparkForCausalLM)
    causal_lm.model = mock_model
    MuseSparkForCausalLM.set_aux_hidden_state_layers(causal_lm, (14, 30, 46, 58, 62))
    assert mock_model.aux_hidden_state_layers == (14, 30, 46, 58, 62)

    MuseSparkForCausalLM.set_aux_hidden_state_layers(causal_lm, [13, 29, 45, 57, 61])
    assert mock_model.aux_hidden_state_layers == (14, 30, 46, 58, 62)

    MuseSparkForCausalLM.set_aux_hidden_state_layers(causal_lm, [3, 19, 31, 47, 59])
    assert mock_model.aux_hidden_state_layers == (4, 20, 32, 48, 60)


def test_muse_spark_dspark_model_instantiation_and_forward(
    _single_rank_dist,
) -> None:
    cfg = MuseSparkDSparkConfig(
        vocab_size=256,
        draft_vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        target_hidden_size=64,
        num_target_layers=2,
        target_layer_ids=[0, 1],
        markov_rank=16,
        block_size=4,
    )
    vllm_config = VllmConfig()
    vllm_config.speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(hf_config=cfg, quantization=None),
        num_speculative_tokens=4,
        enable_adaptive_verification=False,
    )
    vllm_config.model_config = SimpleNamespace(
        max_model_len=128,
        get_vocab_size=lambda: 256,
        dtype=torch.float32,
        is_mm_prefix_lm=False,
        head_dtype=None,
        rswa_window=None,
    )

    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(vllm_config):
        model = MuseSparkDSparkModel(vllm_config=vllm_config)
        assert model.owns_context_kv is True
        assert model.has_own_embed_tokens is False
        assert model.has_own_lm_head is True
        assert model.get_draft_attn_causal() == [True, True]
        assert len(model.get_draft_kv_cache_layer_names()) == 2

        target_embed = torch.nn.Embedding(256, 64)
        model.model.embed_tokens = target_embed

        input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        embeds = model.embed_input_ids(input_ids)
        assert embeds.shape == (4, 64)

        target_aux = [torch.randn(4, 64), torch.randn(4, 64)]
        combined = model.combine_hidden_states(target_aux)
        assert combined.shape == (4, 64)

        markov_emb = model.markov_embed(torch.tensor([5, 10], dtype=torch.int32))
        assert markov_emb.shape == (2, 16)

        bias = model.markov_bias(markov_emb)
        assert bias.shape == (2, 256)

        mapped = model.map_draft_to_target(torch.tensor([10, 20], dtype=torch.int32))
        assert torch.equal(mapped, torch.tensor([10, 20], dtype=torch.int32))


def test_dspark_config_resolution_uses_target_type_not_directory_name() -> None:
    from vllm_torchtpu.models.vllm.muse_spark.configuration import MuseSparkConfig

    updated = []
    draft_hf = SimpleNamespace(model_type="qwen3_dspark", architectures=["Qwen3DSpark"])
    draft_cfg = SimpleNamespace(
        model="/checkpoints/custom-finetune-draft-v2",
        hf_config=draft_hf,
        quantization="nvfp4",
        quantization_config={"quant_method": "modelopt"},
    )
    vllm_cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=MuseSparkConfig()),
        cache_config=SimpleNamespace(block_size=128, user_specified_block_size=True),
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        speculative_config=SimpleNamespace(
            method="dspark",
            draft_model_config=draft_cfg,
            update_arch_=lambda: updated.append(True),
        ),
    )
    TPUMuseSparkConfig.verify_and_update_config(vllm_cfg)
    assert draft_hf.model_type == "muse_spark_dspark"
    assert draft_hf.architectures == ["MuseSparkDSparkModel"]
    assert draft_cfg.quantization is None
    assert draft_cfg.quantization_config is None
    assert updated == [True]
