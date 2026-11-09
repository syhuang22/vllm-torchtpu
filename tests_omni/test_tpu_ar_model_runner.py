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
"""Unit tests for src/vllm_torchtpu/omni/worker/tpu_ar_model_runner.py."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm_omni.config import OmniModelConfig
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVTransferManager,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
)

from vllm_torchtpu.omni.worker.tpu_ar_model_runner import TPUARModelRunner
from vllm_torchtpu.omni.worker.tpu_model_runner import OmniTPUModelRunner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUModelRunnerOutput

pytestmark = pytest.mark.omni

# The connector fields needs_omni_connector() looks at, all off.
_SINGLE_STAGE = dict(
    requires_full_payload_input=False,
    custom_process_next_stage_input_func=None,
    stage_connector_config=None,
)


def _fake_base_init(self, vllm_config, *args, **kwargs):
    """Stand-in for TPUModelRunner.__init__: only the config plumbing."""
    self.vllm_config = vllm_config
    self.model_config = vllm_config.model_config


def _stage_config(**fields) -> OmniModelConfig:
    config = OmniModelConfig.__new__(OmniModelConfig)
    for name, value in {**_SINGLE_STAGE, **fields}.items():
        object.__setattr__(config, name, value)
    return config


def _build(model_config) -> TPUARModelRunner:
    with patch.object(TPUModelRunner, "__init__", _fake_base_init):
        return TPUARModelRunner(SimpleNamespace(model_config=model_config), "tpu")


def _async_output(inner) -> AsyncTPUModelRunnerOutput:
    out = AsyncTPUModelRunnerOutput.__new__(AsyncTPUModelRunnerOutput)
    out._model_runner_output = inner
    return out


def test_tpu_ar_model_runner_inheritance():
    # Same shape as upstream GPUARModelRunner: the connector mixin sits on the
    # leaf runner, not on the shared OmniTPUModelRunner base.
    assert TPUARModelRunner.__bases__ == (
        OmniTPUModelRunner,
        OmniConnectorModelRunnerMixin,
    )
    assert OmniConnectorModelRunnerMixin not in OmniTPUModelRunner.__mro__


@pytest.mark.parametrize(("stage", "stage_id"), [("thinker", 0), ("talker", 1)])
def test_supported_stages_are_accepted(stage: str, stage_id: int):
    runner = _build(_stage_config(model_stage=stage, stage_id=stage_id))
    assert runner.model_stage == stage
    assert runner.stage_id == stage_id
    assert runner.model_intermediate_buffer == {}


def test_plain_vllm_config_is_accepted():
    # No stage metadata at all: a single-stage deployment outside vLLM-Omni.
    runner = _build(SimpleNamespace())
    assert runner.model_stage is None
    assert runner.stage_id is None


def test_unsupported_stage_is_refused_at_construction():
    with pytest.raises(NotImplementedError, match="code2wav"):
        _build(_stage_config(model_stage="code2wav", stage_id=2))


def test_multi_stage_runner_initializes_connectors_in_init():
    cfg = _stage_config(
        model_stage="thinker",
        stage_id=0,
        custom_process_next_stage_input_func="pkg.mod.thinker2talker",
    )
    fake_kv_mgr = object()
    with (
        patch.object(
            OmniKVTransferManager, "from_vllm_config", return_value=fake_kv_mgr
        ) as mock_from_cfg,
        patch.object(TPUARModelRunner, "init_omni_connectors") as mock_init_conn,
    ):
        runner = _build(cfg)

    mock_from_cfg.assert_called_once_with(runner.vllm_config, cfg)
    mock_init_conn.assert_called_once_with(
        model_config=cfg,
        kv_transfer_manager=fake_kv_mgr,
    )
    assert runner.kv_transfer_manager is fake_kv_mgr


def test_update_states_cleans_finished_requests_and_populates_buffer():
    runner = _build(_stage_config(model_stage="talker", stage_id=1))
    runner._omni_connector_initialized = True
    runner.cleanup_finished_request = MagicMock()
    runner.model = SimpleNamespace(gpu_resident_buffer_keys={("hidden_states", "last")})
    runner.model_intermediate_buffer = {"req-done": {"meta": {"step": 1}}}

    new_req = SimpleNamespace(
        req_id="req-new",
        additional_information={"speaker": "ethan"},
        model_intermediate_buffer={
            "hidden_states": {"last": torch.ones(2, 4)},
            "embed": {"prefill": torch.zeros(3, 4)},
        },
    )
    cached_reqs = SimpleNamespace(
        req_ids=["req-new"],
        additional_information={"req-new": {"meta": {"decode_flag": True}}},
    )
    sched_out = SimpleNamespace(
        finished_req_ids=["req-done"],
        scheduled_new_reqs=[new_req],
        scheduled_cached_reqs=cached_reqs,
    )

    with patch.object(TPUModelRunner, "_update_states", return_value=True):
        assert runner._update_states(sched_out) is True

    runner.cleanup_finished_request.assert_called_once_with("req-done")
    assert "req-done" not in runner.model_intermediate_buffer
    buf = runner.model_intermediate_buffer["req-new"]
    assert buf["speaker"] == "ethan"
    assert buf["meta"] == {"decode_flag": True}
    assert torch.equal(buf["hidden_states"]["last"], torch.ones(2, 4))
    assert torch.equal(buf["embed"]["prefill"], torch.zeros(3, 4))


def test_tpu_ar_model_runner_mixed_dim_mm_slicing():
    runner = _build(_stage_config(model_stage="thinker", stage_id=0))
    runner.supports_mm_inputs = True

    vid_emb, aud_emb = torch.ones(6, 32), torch.ones(4, 8)
    vid_emb.modality, aud_emb.modality = "video", "audio"
    mask = torch.ones(10, dtype=torch.bool)

    with patch.object(
        OmniTPUModelRunner,
        "_gather_mm_embeddings",
        return_value=([vid_emb, aud_emb], mask),
    ):
        idx_list, out_mask = runner._gather_mm_embeddings("sched_out")
    assert len(idx_list) == 1 and torch.equal(idx_list[0], torch.arange(10))
    assert out_mask is mask

    captured_mm = []

    def _dummy_embed_input_ids(ids, **kwargs):
        captured_mm.append(kwargs.get("multimodal_embeddings"))
        return torch.zeros(ids.shape[0], 8)

    runner.model = SimpleNamespace(embed_input_ids=_dummy_embed_input_ids)
    _, embeds_out = runner._get_model_inputs(
        torch.zeros(5, dtype=torch.long),
        ([torch.arange(4, 9, dtype=torch.int64)], torch.ones(5, dtype=torch.bool)),
    )
    assert embeds_out.shape == (5, 8) and len(captured_mm[0]) == 2
    assert captured_mm[0][0].shape == (2, 32)
    assert captured_mm[0][0].modality == "video"
    assert captured_mm[0][1].shape == (3, 8)
    assert captured_mm[0][1].modality == "audio"
    assert len(runner._omni_mm_embeds_list) == 2

    # Slicing through the final embedding row clears _omni_mm_embeds_list.
    runner._get_model_inputs(
        torch.zeros(1, dtype=torch.long),
        ([torch.arange(9, 10, dtype=torch.int64)], torch.ones(1, dtype=torch.bool)),
    )
    assert runner._omni_mm_embeds_list == []


def test_tpu_ar_model_runner_capture_model_precompiles_mm_buckets():
    runner = _build(_stage_config(model_stage="thinker", stage_id=0))
    runner.device = torch.device("cpu")
    runner.enforce_eager = False
    runner.supports_mm_inputs = True
    runner._pp_is_first = True
    runner.num_tokens_paddings = [16, 64]
    runner._update_num_xla_graphs = MagicMock()

    seen_modalities: list[str] = []
    seen_merges: list[tuple[int, tuple[tuple[int, ...], str] | None]] = []
    reset_cache = MagicMock()
    runner.mm_budget = SimpleNamespace(
        mm_max_toks_per_item={"video": 65536, "audio": 390, "image": 4096},
        reset_cache=reset_cache,
    )
    runner._get_mm_dummy_batch = lambda modality, count: {"modality": modality}

    def _embed_multimodal(**kwargs):
        modality = kwargs["modality"]
        seen_modalities.append(modality)
        width = 8 if modality == "audio" else 32
        return (torch.ones(4, width),)

    def _embed_input_ids(input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        mm_info = (
            None
            if multimodal_embeddings is None
            else (
                tuple(multimodal_embeddings[0].shape),
                multimodal_embeddings[0].modality,
            )
        )
        seen_merges.append((int(input_ids.shape[0]), mm_info))
        return torch.zeros(input_ids.shape[0], 8)

    runner.model = SimpleNamespace(
        embed_multimodal=_embed_multimodal,
        embed_input_ids=_embed_input_ids,
    )

    with patch.object(OmniTPUModelRunner, "capture_model") as mock_super_capture:
        runner.capture_model()
        mock_super_capture.assert_called_once_with()
        assert seen_modalities == ["audio", "image"]
        reset_cache.assert_called_once_with()
        assert seen_merges == [
            (16, ((1, 8), "audio")),
            (64, ((1, 8), "audio")),
            (16, ((1, 32), "image")),
            (64, ((1, 32), "image")),
            (16, None),
        ]

        # Non-MM, enforce_eager, non-first PP, or talker stage skips MM precompilation
        for attr in ("supports_mm_inputs", "_pp_is_first"):
            seen_modalities.clear()
            seen_merges.clear()
            setattr(runner, attr, False)
            runner.capture_model()
            assert seen_modalities == []
            assert seen_merges == []
            setattr(runner, attr, True)

        seen_modalities.clear()
        seen_merges.clear()
        runner.enforce_eager = True
        runner.capture_model()
        assert seen_modalities == []
        assert seen_merges == []
        runner.enforce_eager = False

        seen_modalities.clear()
        seen_merges.clear()
        runner.model_stage = "talker"
        runner.capture_model()
        assert seen_modalities == []
        assert seen_merges == []


def test_tpu_ar_model_runner_forward_model_unwraps_omni_output():
    runner = _build(_stage_config(model_stage="thinker", stage_id=0))
    runner.model = SimpleNamespace()
    expected_hs = torch.ones(3, 8)
    omni_out = OmniOutput(text_hidden_states=expected_hs)
    with patch.object(
        OmniTPUModelRunner, "forward_model", return_value=(omni_out, None)
    ):
        hs, aux = runner.forward_model(
            torch.zeros(3, dtype=torch.long), torch.arange(3)
        )
    assert hs is expected_hs
    assert aux is None


def test_tpu_ar_model_runner_talker_preprocess_mtp_postprocess_and_output():
    runner = _build(_stage_config(model_stage="talker", stage_id=1))
    runner.supports_mm_inputs = False
    runner.device = torch.device("cpu")
    runner.input_batch = SimpleNamespace(
        req_ids=["req-0"],
        num_reqs=1,
        num_computed_tokens_cpu=np.array([4], dtype=np.int32),
    )
    runner.query_start_loc_np = np.array([0, 1], dtype=np.int32)

    class DummyTalkerStage:
        has_preprocess = True
        has_postprocess = True
        gpu_resident_buffer_keys = {("hidden_states", "last"), ("codes", "audio")}

        def preprocess(self, input_ids, input_embeds=None, **req_info):
            assert req_info["_omni_num_computed_tokens"] == 4
            assert req_info["_omni_is_prefill"] is False
            return (
                input_ids,
                torch.ones(1, 8),
                {"mtp_inputs": (torch.full((1, 8), 2.0), torch.full((1, 8), 3.0))},
            )

        def talker_mtp(self, input_ids, input_embeds, last_hidden, text_step):
            return input_embeds + last_hidden + text_step, torch.arange(16).unsqueeze(0)

        def postprocess(self, hidden_states, **kwargs):
            return {"hidden_states": {"last": hidden_states[-1].detach()}}

        def make_omni_output(self, model_outputs, model_intermediate_buffer=None):
            codes = [b["codes"]["audio"] for b in model_intermediate_buffer]
            return OmniOutput(
                text_hidden_states=model_outputs,
                multimodal_outputs={"codes": {"audio": torch.cat(codes, dim=0)}},
            )

    runner.model = DummyTalkerStage()
    out_ids, out_embeds = runner._get_model_inputs(
        torch.tensor([42], dtype=torch.long), None
    )
    assert out_ids is None
    assert torch.equal(out_embeds, torch.full((1, 8), 6.0))
    assert torch.equal(
        runner.model_intermediate_buffer["req-0"]["codes"]["audio"],
        torch.arange(16).unsqueeze(0),
    )

    hidden = torch.full((1, 8), 9.0)
    with patch.object(OmniTPUModelRunner, "forward_model", return_value=(hidden, None)):
        fwd_out, fwd_aux = runner.forward_model(None, torch.arange(1), out_embeds)
    assert torch.equal(fwd_out, hidden)
    assert fwd_aux is None
    assert torch.equal(
        runner.model_intermediate_buffer["req-0"]["hidden_states"]["last"],
        torch.full((8,), 9.0),
    )

    inner_out = SimpleNamespace(req_ids=["req-0"], inter_stage_outputs=None)
    runner._omni_connector_initialized = True
    runner._should_accumulate_full_payload_output = MagicMock(return_value=False)
    runner.attach_omni_connector_output = MagicMock(side_effect=lambda x: x)
    with patch.object(
        OmniTPUModelRunner, "sample_tokens", return_value=_async_output(inner_out)
    ):
        runner.sample_tokens()
    assert len(inner_out.inter_stage_outputs) == 1
    assert torch.equal(
        inner_out.inter_stage_outputs[0]["codes"]["audio"],
        torch.arange(16).unsqueeze(0),
    )
    runner.attach_omni_connector_output.assert_called_once_with(inner_out)


def test_tpu_ar_model_runner_full_payload_connector_lifecycle():
    runner = _build(_stage_config(model_stage="talker", stage_id=1, async_chunk=False))
    runner._omni_connector_initialized = True
    runner._pending_full_payload_send = {"req-done", "req-evicted"}
    runner.requests = {"req-done": SimpleNamespace(prompt_token_ids=[1, 2, 3])}
    runner.register_chunk_recv = MagicMock()
    runner.recv_full_payload_inputs = MagicMock()
    runner.flush_full_payload_outputs = MagicMock()
    runner.attach_omni_connector_output = MagicMock(side_effect=lambda x: x)

    sched_out = SimpleNamespace(
        pending_input_registrations=["reg-0"],
        finished_req_ids={"req-done"},
    )
    with patch.object(OmniTPUModelRunner, "execute_model", return_value=None):
        assert runner.execute_model(sched_out) is None

    runner.register_chunk_recv.assert_called_once_with("reg-0")
    runner.recv_full_payload_inputs.assert_called_once_with(sched_out)
    runner.flush_full_payload_outputs.assert_called_once_with(
        {"req-done", "req-evicted"}
    )


def test_sync_local_stage_payloads_moves_ready_active_payloads():
    runner = _build(_stage_config(model_stage="talker", stage_id=1, async_chunk=False))
    runner.model = SimpleNamespace(gpu_resident_buffer_keys=set())
    runner.requests = {"req-ready": SimpleNamespace(), "req-pending": SimpleNamespace()}
    runner._local_stage_payload_cache = {
        "req-ready": {"embed": {"prefill": torch.ones(2, 4)}},
        "req-pending": {"embed": {"prefill": torch.zeros(2, 4)}},
        "req-unknown": {"embed": {"prefill": torch.ones(1, 4)}},
    }
    runner._full_payload_pending_broadcast_req_ids = {"req-pending"}

    runner._sync_local_stage_payloads()

    assert "req-ready" not in runner._local_stage_payload_cache
    assert "req-pending" in runner._local_stage_payload_cache
    assert "req-unknown" in runner._local_stage_payload_cache
    assert torch.equal(
        runner.model_intermediate_buffer["req-ready"]["embed"]["prefill"],
        torch.ones(2, 4),
    )
    assert "req-pending" not in runner.model_intermediate_buffer
