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
"""Registry gating and wiring of the Qwen3-Omni Thinker patches.

The on-TPU behaviour of the patched audio encoder (exact window multiples, token
counts, CPU parity) is covered by test_qwen3_omni_audio_encoder_tpu.py; these
tests check the callback contract the patch registry relies on and the
static-shape bucketing, frame-chunking, and multimodal merge wrappers on CPU.
"""

import importlib.util
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import vllm.model_executor.models.utils as vllm_utils
from vllm.model_executor.models import qwen3_omni_moe_thinker as vllm_thinker_mod
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder as VllmQwen3OmniMoeAudioEncoder,
)
from vllm.multimodal.utils import set_mm_embedding_modality
from vllm_omni.model_executor.models.qwen3_omni import (
    qwen3_omni_moe_thinker as thinker_mod,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3Omni_VisionTransformer,
    Qwen3OmniMoeAudioEncoder,
)

import vllm_torchtpu
from vllm_torchtpu.omni.patches.qwen3omni_thinker import (
    _AUDIO_CHUNK_LENGTHS_PATCHED,
    _select_bucket,
    maybe_patch_qwen3_omni_thinker,
)

pytestmark = pytest.mark.omni


def _model_config(*architectures: str) -> SimpleNamespace:
    """The only ModelConfig field the callback reads."""
    return SimpleNamespace(architectures=list(architectures))


OMNI_THINKER = _model_config("Qwen3OmniMoeForConditionalGeneration")


def _isolate_thinker_patch_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore every attribute rebound by maybe_patch_qwen3_omni_thinker."""
    monkeypatch.setattr(
        vllm_torchtpu, "_patch_vllm_merge_multimodal_embeddings", lambda: None
    )
    for cls in (
        vllm_thinker_mod.Qwen3OmniMoeAudioEncoder,
        thinker_mod.Qwen3OmniMoeAudioEncoder,
        thinker_mod.Qwen3Omni_VisionTransformer,
    ):
        monkeypatch.setattr(cls, "forward", cls.forward)
        monkeypatch.setattr(
            cls,
            "_tpu_orig_forward",
            getattr(cls, "_tpu_orig_forward", None),
            raising=False,
        )
    monkeypatch.setattr(
        thinker_mod.Qwen3OmniMoeThinkerForConditionalGeneration,
        "embed_input_ids",
        thinker_mod.Qwen3OmniMoeThinkerForConditionalGeneration.embed_input_ids,
    )
    monkeypatch.setattr(
        vllm_utils,
        "_merge_multimodal_embeddings",
        vllm_utils._merge_multimodal_embeddings,
    )
    monkeypatch.setattr(
        thinker_mod,
        "_merge_multimodal_embeddings",
        thinker_mod._merge_multimodal_embeddings,
    )
    monkeypatch.setattr(
        thinker_mod,
        "check_interleaved_audio_video",
        thinker_mod.check_interleaved_audio_video,
    )
    monkeypatch.setattr(
        thinker_mod,
        "merge_interleaved_embeddings",
        thinker_mod.merge_interleaved_embeddings,
    )


class _DummyLinear(nn.Module):
    def forward(self, x):
        return x, None


class _DummyEncoderLayer(nn.Module):
    def __init__(self, seen_seq_lens: list[int] | None = None):
        super().__init__()
        self.seen_seq_lens = seen_seq_lens

    def forward(self, hs, cu_seqlens, max_seqlen=None):
        assert cu_seqlens.shape == (128,)
        assert max_seqlen is None
        if self.seen_seq_lens is not None:
            self.seen_seq_lens.append(int(hs.shape[0]))
        return hs


class _DummyAudioEncoder(nn.Module):
    num_mel_bins = 16
    n_window = 50
    n_window_infer = 400
    conv_chunksize = 16

    def __init__(self, seen_seq_lens: list[int] | None = None):
        super().__init__()
        self.conv2d1 = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)
        self.conv2d2 = nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1)
        self.conv2d3 = nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1)
        self.conv_out = _DummyLinear()
        self.positional_embedding = SimpleNamespace(
            positional_embedding=torch.zeros(64, 8)
        )
        self.layers = nn.ModuleList([_DummyEncoderLayer(seen_seq_lens)])
        self.ln_post = nn.Identity()
        self.proj1 = _DummyLinear()
        self.act = F.gelu
        self.proj2 = _DummyLinear()


class _DummyViTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_shapes: list[int] = []

    def forward(
        self,
        hs,
        cu_seqlens=None,
        rotary_pos_emb_cos=None,
        rotary_pos_emb_sin=None,
        max_seqlen=None,
        sequence_lengths=None,
    ):
        self.seen_shapes.append(int(hs.shape[0]))
        cos = rotary_pos_emb_cos.unsqueeze(1)
        sin = rotary_pos_emb_sin.unsqueeze(1)
        return hs * cos + sin


class _DummyViT(nn.Module):
    apply_vit_abs_pos_embed = True
    spatial_merge_unit = 4
    deepstack_visual_indexes = [0]

    # Bind the real upstream Qwen3Omni_VisionTransformer.forward_with_encoder_metadata
    # and prepare_encoder_metadata so the test exercises the real upstream
    # execution path and compares _padded_vit_forward against _tpu_orig_forward.
    forward_with_encoder_metadata = (
        Qwen3Omni_VisionTransformer.forward_with_encoder_metadata
    )
    prepare_encoder_metadata = Qwen3Omni_VisionTransformer.prepare_encoder_metadata

    def __init__(self):
        super().__init__()
        self.patch_embed = SimpleNamespace(proj=nn.Linear(8, 8, bias=False))
        nn.init.eye_(self.patch_embed.proj.weight)
        self.blk = _DummyViTBlock()
        self.blocks = nn.ModuleList([self.blk])
        self.merger = lambda hs: hs.squeeze(1)[::4]
        self.merger_list = [lambda hs: hs.squeeze(1)[::4] * 2.0]
        self.attn_backend = None

    def patch_embed_call(self, x):
        return self.patch_embed.proj(x)

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self):
        return self.patch_embed.proj.weight.device

    def compute_attn_mask_seqlen(self, cu_seqlens):
        return int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

    def rot_pos_emb(self, g):
        assert isinstance(g, torch.Tensor) and g.device.type == "cpu"
        rows = []
        for t, h, w in g.tolist():
            n_frame = int(h * w)
            idx = (
                torch.arange(n_frame, dtype=torch.float32)
                .repeat(int(t))
                .unsqueeze(1)
                .expand(int(t) * n_frame, 8)
            )
            rows.append(idx)
        cat = torch.cat(rows, dim=0)
        return torch.ones_like(cat), cat * 0.01

    def fast_pos_embed_interpolate(self, g):
        g_list = g.tolist() if isinstance(g, torch.Tensor) else g
        assert isinstance(g_list, list)
        rows = []
        for t, h, w in g_list:
            n_frame = int(h * w)
            pos = (
                torch.arange(n_frame, dtype=torch.float32)
                .repeat(int(t))
                .unsqueeze(1)
                .expand(int(t) * n_frame, 8)
                * 0.1
            )
            rows.append(pos)
        return torch.cat(rows, dim=0)


_DummyViT.patch_embed = nn.Linear(8, 8, bias=False)


def _make_dummy_vit() -> _DummyViT:
    vit = _DummyViT()
    vit.patch_embed = nn.Linear(8, 8, bias=False)
    vit.patch_embed.proj = vit.patch_embed
    nn.init.eye_(vit.patch_embed.weight)
    return vit


def test_non_omni_config_is_not_consumed():
    # ``False`` (not ``None``) tells the registry to retry at model_load. The
    # match is the literal architecture: a substring such as the thinker's own
    # class name does not count.
    before = Qwen3OmniMoeAudioEncoder.forward
    assert maybe_patch_qwen3_omni_thinker(None) is False
    assert maybe_patch_qwen3_omni_thinker(_model_config("LlamaForCausalLM")) is False
    assert (
        maybe_patch_qwen3_omni_thinker(
            _model_config("Qwen3OmniMoeThinkerForConditionalGeneration")
        )
        is False
    )
    assert Qwen3OmniMoeAudioEncoder.forward is before


def test_qwen3_omni_audio_chunk_lengths_patch(monkeypatch):
    _isolate_thinker_patch_targets(monkeypatch)

    def _orig_forward(self, input_features, feature_lens, aftercnn_lens):
        return input_features

    monkeypatch.setattr(Qwen3OmniMoeAudioEncoder, "forward", _orig_forward)

    assert maybe_patch_qwen3_omni_thinker(OMNI_THINKER) is True
    patched = Qwen3OmniMoeAudioEncoder.forward
    assert patched is not _orig_forward
    assert getattr(patched, _AUDIO_CHUNK_LENGTHS_PATCHED, False)
    assert Qwen3OmniMoeAudioEncoder._tpu_orig_forward is _orig_forward
    assert getattr(
        VllmQwen3OmniMoeAudioEncoder.forward, _AUDIO_CHUNK_LENGTHS_PATCHED, False
    )

    class TrackedTensor(torch.Tensor):
        cpu_called: bool = False

        def cpu(self, *args, **kwargs):
            self.cpu_called = True
            return super().cpu(*args, **kwargs)

    enc = _DummyAudioEncoder()
    feat_lens = torch.tensor([800]).as_subclass(TrackedTensor)
    cnn_lens = torch.tensor([104]).as_subclass(TrackedTensor)
    out = patched(enc, torch.zeros(16, 800), feat_lens, cnn_lens)
    assert feat_lens.cpu_called is True
    assert cnn_lens.cpu_called is True
    assert out.shape == (104, 8)

    # Idempotent: a second application keeps the same wrapper and orig_forward.
    assert maybe_patch_qwen3_omni_thinker(OMNI_THINKER) is True
    assert Qwen3OmniMoeAudioEncoder.forward is patched
    assert Qwen3OmniMoeAudioEncoder._tpu_orig_forward is _orig_forward


def test_qwen3_omni_patch_without_vllm_omni_installed(monkeypatch):
    """Plain-vLLM Thinker-only startup without vllm-omni patches vLLM's encoder."""
    _isolate_thinker_patch_targets(monkeypatch)

    def _vllm_forward(self, input_features, feature_lens, aftercnn_lens):
        return input_features

    monkeypatch.setattr(VllmQwen3OmniMoeAudioEncoder, "forward", _vllm_forward)
    orig_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **kw: (
            None if name == "vllm_omni" else orig_find_spec(name, *a, **kw)
        ),
    )

    omni_before = Qwen3OmniMoeAudioEncoder.forward
    assert maybe_patch_qwen3_omni_thinker(OMNI_THINKER) is True
    assert getattr(
        VllmQwen3OmniMoeAudioEncoder.forward, _AUDIO_CHUNK_LENGTHS_PATCHED, False
    )
    assert VllmQwen3OmniMoeAudioEncoder._tpu_orig_forward is _vllm_forward
    assert Qwen3OmniMoeAudioEncoder.forward is omni_before


def test_qwen3_omni_audio_encoder_static_bucketing(monkeypatch):
    _isolate_thinker_patch_targets(monkeypatch)
    seen_layer_seq_lens: list[int] = []

    class DummyAudioEncoder(_DummyAudioEncoder):
        def __init__(self):
            super().__init__(seen_layer_seq_lens)

    monkeypatch.setattr(thinker_mod, "Qwen3OmniMoeAudioEncoder", DummyAudioEncoder)
    assert maybe_patch_qwen3_omni_thinker(OMNI_THINKER) is True

    with patch.object(
        nn.Module,
        "cpu",
        side_effect=AssertionError("Unexpected Module.cpu() host mutation"),
    ):
        enc = DummyAudioEncoder()
        for flen, alen in ((220, 29), (310, 41), (670, 87), (800, 104)):
            out = enc(
                torch.zeros(16, flen),
                torch.tensor([flen], dtype=torch.long),
                torch.tensor([alen], dtype=torch.long),
            )
            assert out.shape == (alen, 8)
        assert seen_layer_seq_lens == [512, 512, 512, 512]
        assert _select_bucket(5, (8, 16), align=8) == 8
        assert _select_bucket(19, (8, 16), align=8) == 24


@pytest.mark.parametrize(
    ("grid", "expected_chunks"),
    [
        ([[1, 20, 30]], [16384]),
        ([[5, 40, 100]], [16384, 16384]),
    ],
)
def test_qwen3_omni_vit_frame_chunking_and_parity(
    monkeypatch, grid: list[list[int]], expected_chunks: list[int]
):
    """Verify ViT bucketing, frame chunking, and parity with upstream forward."""
    _isolate_thinker_patch_targets(monkeypatch)
    assert maybe_patch_qwen3_omni_thinker(OMNI_THINKER) is True
    assert hasattr(Qwen3Omni_VisionTransformer, "_tpu_orig_forward")

    total_patches = sum(t * h * w for t, h, w in grid)
    x = torch.arange(total_patches * 8, dtype=torch.float32).reshape(total_patches, 8)
    grid_tensor = torch.tensor(grid, dtype=torch.int32)

    vit_patched = _make_dummy_vit()
    vit_patched.forward = MethodType(Qwen3Omni_VisionTransformer.forward, vit_patched)

    vit_ref = _make_dummy_vit()
    vit_ref.forward = MethodType(Qwen3Omni_VisionTransformer._tpu_orig_forward, vit_ref)

    with patch.object(
        nn.Module,
        "cpu",
        side_effect=AssertionError("Unexpected Module.cpu() host mutation"),
    ):
        out_patched = vit_patched(x, grid_tensor)

    out_ref = vit_ref(x, grid_tensor)
    assert out_patched.shape == (total_patches // 4, 16)
    assert vit_patched.blk.seen_shapes == expected_chunks
    assert torch.allclose(out_patched, out_ref)


def test_qwen3_omni_embed_input_ids_patch(monkeypatch):
    """Verify embed_input_ids clears deepstack state and keeps embeddings on device."""
    _isolate_thinker_patch_targets(monkeypatch)

    class DummyMixin:
        use_deepstack = True
        deepstack_input_embeds_num_tokens = 12
        deepstack_input_embeds = [torch.zeros(16, 8)]

        def __init__(self):
            self.cleared_tokens: list[int] = []

        def _clear_deepstack_input_embeds(self, n: int):
            self.cleared_tokens.append(n)
            self.deepstack_input_embeds_num_tokens = 0

        def embed_input_ids(
            self, ids, multimodal_embeddings=None, *, is_multimodal=None
        ):
            return multimodal_embeddings, is_multimodal

    monkeypatch.setattr(
        thinker_mod, "Qwen3OmniMoeThinkerForConditionalGeneration", DummyMixin
    )
    assert maybe_patch_qwen3_omni_thinker(OMNI_THINKER) is True

    vid_emb = torch.ones(6, 32)
    vid_emb.modality = "video"
    mixin = DummyMixin()
    # Both the embeddings and the already-on-CPU mask are forwarded intact.
    kw_embeds, kw_mask = mixin.embed_input_ids(
        input_ids=torch.zeros(5, dtype=torch.long),
        multimodal_embeddings=(vid_emb,),
        is_multimodal=torch.ones(5, dtype=torch.bool),
    )
    assert kw_embeds[0] is vid_emb
    assert kw_embeds[0].modality == "video"
    assert torch.equal(kw_mask, torch.ones(5, dtype=torch.bool))
    assert mixin.cleared_tokens == [12]
    assert mixin.deepstack_input_embeds_num_tokens == 0


def test_qwen3_omni_thinker_interleaved_mm_merge(monkeypatch):
    """Test _check_interleaved, _merge_interleaved, and _omni_merge."""
    _isolate_thinker_patch_targets(monkeypatch)
    seen_padded_rows: list[int] = []

    def _fake_static_merge(inputs_embeds, multimodal_embeddings, is_multimodal):
        flat = vllm_utils._flatten_embeddings(multimodal_embeddings)
        seen_padded_rows.append(int(flat.shape[0]))
        idx = torch.cumsum(is_multimodal, dim=0, dtype=torch.int32) - 1
        idx = torch.clamp(idx, 0, flat.shape[0] - 1)
        return torch.where(is_multimodal.unsqueeze(-1), flat[idx], inputs_embeds)

    monkeypatch.setattr(vllm_utils, "_merge_multimodal_embeddings", _fake_static_merge)
    assert maybe_patch_qwen3_omni_thinker(OMNI_THINKER) is True

    # 1. _check_interleaved: short-circuits when either count is 0 and detects
    # interleaved video+audio spans on CPU.
    is_video = torch.tensor([False, True, False, True, False, False])
    is_audio = torch.tensor([False, False, True, False, False, False])
    is_image = torch.tensor([False, False, False, False, True, False])
    is_mm = is_video | is_audio | is_image
    assert thinker_mod.check_interleaved_audio_video(is_video, is_audio, 0, 1) is False
    assert thinker_mod.check_interleaved_audio_video(is_video, is_audio, 2, 1) is True

    # 2. _merge_interleaved + _omni_merge: pads each modality group to
    # inputs_embeds.shape[0] and matches boolean-indexing reference.
    inputs_embeds = torch.zeros(6, 4)
    vid_emb = set_mm_embedding_modality(torch.tensor([[1.0] * 4, [2.0] * 4]), "video")
    aud_emb = set_mm_embedding_modality(torch.tensor([[3.0] * 4]), "audio")
    img_emb = set_mm_embedding_modality(torch.tensor([[4.0] * 4]), "image")

    merged = thinker_mod.merge_interleaved_embeddings(
        inputs_embeds.clone(),
        [vid_emb, aud_emb, img_emb],
        is_video,
        is_audio,
        is_mm,
    )
    expected = inputs_embeds.clone()
    expected[is_video] = vid_emb
    expected[is_audio] = aud_emb
    expected[is_image] = img_emb
    assert torch.equal(merged, expected)
    # _omni_merge padded all three modality tensors up to inputs_embeds.shape[0] = 6.
    assert seen_padded_rows == [6, 6, 6]

    # 3. Missing .modality raises ValueError.
    untagged = torch.ones(2, 4)
    with pytest.raises(ValueError, match="Missing modality"):
        thinker_mod.merge_interleaved_embeddings(
            inputs_embeds.clone(),
            [untagged, aud_emb, img_emb],
            is_video,
            is_audio,
            is_mm,
        )

    # 4. Non-empty modality mask with mismatched token count raises ValueError.
    with pytest.raises(ValueError, match="Attempted to assign 0 audio tokens to 1"):
        thinker_mod.merge_interleaved_embeddings(
            inputs_embeds.clone(),
            [vid_emb, img_emb],
            is_video,
            is_audio,
            is_mm,
        )
