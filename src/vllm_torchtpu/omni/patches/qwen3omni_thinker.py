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
"""TPU patches for the Qwen3-Omni Thinker: audio encoder chunk-length and bucketing.

``Qwen3OmniMoeAudioEncoder.forward`` (in both upstream vLLM and vLLM-Omni's
thinker subclass) splits the mel features into windows of
``n_window * 2 = 100`` frames with

    chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

and then calls ``input_features.T.split(chunk_lengths)`` with one entry per
window. On TPU the float division lowers to a multiplication by the
reciprocal of 100, which float32 cannot represent exactly
(0.01000000074505806), so every exact multiple of the window rounds up:
800 * 0.01000000074505806 = 8.000001, ``torch.ceil`` yields 9 windows for an
8 s clip, the window lengths add up to 900 frames instead of 800, and
``split_with_sizes`` raises. On TPU v6e every multiple of 100 frames up to
100000 is affected and no other length is. CPU evaluates the same expression
exactly.

In addition, unpadded chunk and token counts trigger an XLA recompilation for
every distinct audio clip length. The patched ``forward`` moves
``feature_lens`` and ``aftercnn_lens`` to CPU, computes window counts with
exact integer arithmetic, pads chunk counts to ``_AUDIO_CHUNK_BUCKETS`` and
token counts to multiples of 128, pads ``cu_seqlens`` to a fixed ``(128,)``
shape, and slices back to ``cnn_len`` valid tokens per clip.

Installed through ``patch_registry.PATCHES`` at ``platform_activation`` and
``model_load`` for ``Qwen3OmniMoeForConditionalGeneration`` only. The thinker
stage runs in a ``TPUWorker`` (``TPUARWorker``), which runs both the
``platform_activation`` and the ``model_load`` patch-registry stages; the
callback returns ``False`` for every other architecture so the registry
retries it once a model config is known.

TODO: remove the CPU ``feature_lens`` hop and the canary
``test_tpu_float_ceil_still_inexact_for_window_multiples`` in
``tests_omni/test_qwen3_omni_audio_encoder_tpu.py`` once vLLM and vLLM-Omni
compute ``chunk_num`` with integer ceil-division,
``(feature_lens + window - 1) // window``, in
https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py
or once TPU float division is exact for these lengths; the static-shape
bucketing in ``_padded_encoder_forward`` remains needed until upstream buckets
audio encoder shapes.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import align_to, synchronize_tensors

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_QWEN3_OMNI_ARCH = "Qwen3OmniMoeForConditionalGeneration"

# Idempotence marker on the wrapped ``forward``. The registry runs a callback
# once per process; the marker covers forks and direct calls from tests.
_AUDIO_CHUNK_LENGTHS_PATCHED = "_tpu_audio_chunk_lengths_patched"

# Each 100-frame window (n_window * 2) is 1 s of 16 kHz audio. A single
# 32-window bucket (32 s -> 512 padded tokens) covers
# WhisperFeatureExtractor.chunk_length (30 s) as well as 32 s clips, so one
# dummy audio warmup in TPUARModelRunner._precompile_mm_inputs compiles the
# bucket used by all standard clips. On TPU v6e-8, one 32-window (512-token)
# pass takes ~4.5 ms on device even for a 1 s clip (and N * ~4.5 ms for N
# clips), vs ~15-30 s for an XLA recompile per distinct clip length. Clips
# longer than 32 windows fall back to 8-window alignment with a warning.
_AUDIO_CHUNK_BUCKETS: tuple[int, ...] = (32,)


def _select_bucket(
    size: int,
    buckets: tuple[int, ...],
    align: int,
) -> int:
    for bucket in buckets:
        if size <= bucket:
            return bucket
    aligned = align_to(max(1, size), align)
    logger.warning_once(
        "Qwen3-Omni Thinker encoder input size %d exceeds max bucket %d; "
        "falling back to %d-aligned size %d (may compile a new XLA graph).",
        size,
        buckets[-1],
        align,
        aligned,
    )
    return aligned


def _pad_rows(tensor: torch.Tensor, padded_rows: int) -> torch.Tensor:
    if tensor.shape[0] >= padded_rows:
        return tensor
    return F.pad(tensor, (0, 0, 0, padded_rows - tensor.shape[0]))


def _unwrap_linear(out: torch.Tensor | tuple[torch.Tensor, Any]) -> torch.Tensor:
    """Unwrap ``(tensor, bias)`` tuples from ``ReplicatedLinear`` or ``nn.Linear``."""
    return out[0] if isinstance(out, tuple) else out


def _build_padded_cu_seqlens(
    seqlens: list[int], padded_total: int, device: torch.device
) -> torch.Tensor:
    # TpuMMEncoderAttention builds segment IDs via (positions >= cu_seqlens).sum(0)
    # over positions in [0, padded_total). Padding cu_seqlens with padded_total + 1
    # keeps a fixed (128,) shape while contributing 0 to every position, so trailing
    # padded rows share one isolated segment ID and never attend to real tokens.
    cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    pad_len = (-cu_seqlens.numel()) % 128
    return F.pad(cu_seqlens, (0, pad_len), value=padded_total + 1).to(device)


# Adapted from Qwen3OmniMoeAudioEncoder.forward in
# vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py
# (vllm-omni==0.32.0rc1) and vllm/model_executor/models/qwen3_omni_moe_thinker.py
# (vllm==0.33.0). Changes from upstream:
# 1. Moves feature_lens and aftercnn_lens to CPU and computes per-clip
#    chunk counts with integer ceil-division instead of float division
#    on device.
# 2. Runs one encoder pass per audio clip, padding chunk counts to
#    _AUDIO_CHUNK_BUCKETS and token counts to multiples of 128.
# 3. Keeps all num_chunks * tokens_per_chunk rows through the conv and
#    transformer layers instead of boolean-indexing with
#    padded_mask_after_cnn, pads cu_seqlens to (128,) with
#    padded_tokens + 1, passes max_seqlen=None (unused by
#    TpuMMEncoderAttention), and slices hidden_states[:cnn_len] at the end.
def _padded_encoder_forward(
    self: Any,
    input_features: torch.Tensor,
    feature_lens: torch.Tensor,
    aftercnn_lens: torch.Tensor,
) -> torch.Tensor:
    """Bucketed audio encoder forward with ``feature_lens`` on CPU."""
    feature_lens = feature_lens.cpu()
    aftercnn_lens = aftercnn_lens.cpu()
    window_size = self.n_window * 2
    tokens_per_chunk = (((window_size + 1) // 2 + 1) // 2 + 1) // 2
    window_aftercnn = tokens_per_chunk * (self.n_window_infer // window_size)
    device = self.conv2d1.weight.device
    dtype = self.conv2d1.weight.dtype
    feat_lens = feature_lens.tolist()
    cnn_lens = aftercnn_lens.tolist()
    feature_list = input_features.to(device=device, dtype=dtype).split(feat_lens, dim=1)
    pos_emb = (
        self.positional_embedding.positional_embedding[:tokens_per_chunk]
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
    )
    outputs = []
    for feature, feat_len, cnn_len in zip(feature_list, feat_lens, cnn_lens):
        num_chunks = _select_bucket(
            align_to(feat_len, window_size) // window_size,
            _AUDIO_CHUNK_BUCKETS,
            align=8,
        )
        padded_tokens = align_to(num_chunks * tokens_per_chunk, 128)
        chunk_seqlens = [window_aftercnn] * (cnn_len // window_aftercnn)
        if cnn_len % window_aftercnn:
            chunk_seqlens.append(cnn_len % window_aftercnn)
        cu_seqlens = _build_padded_cu_seqlens(chunk_seqlens, padded_tokens, device)
        padded_feature = (
            F.pad(feature, (0, num_chunks * window_size - feat_len))
            .T.reshape(num_chunks, window_size, -1)
            .transpose(1, 2)
            .unsqueeze(1)
            .contiguous()
        )
        if device.type == "tpu":
            synchronize_tensors(padded_feature, wait=False)
        conv_chunks = [
            F.gelu(self.conv2d3(F.gelu(self.conv2d2(F.gelu(self.conv2d1(c))))))
            for c in padded_feature.split(self.conv_chunksize, dim=0)
        ]
        conv_out_in = (
            torch.cat(conv_chunks, dim=0)
            .permute(0, 3, 1, 2)
            .contiguous()
            .view(num_chunks, tokens_per_chunk, -1)
        )
        hidden_states = _unwrap_linear(self.conv_out(conv_out_in))
        hidden_states = _pad_rows(
            (hidden_states + pos_emb).reshape(num_chunks * tokens_per_chunk, -1),
            padded_tokens,
        )
        # In vllm==0.33.0 and vllm-omni==0.32.0rc1, Qwen3OmniMoeAudioEncoderLayer
        # passes max_seqlen only to Qwen3OmniMoeAudioAttention -> MMEncoderAttention.
        # Because num_heads == num_kv_heads (is_gqa is False), TPU always takes
        # TpuMMEncoderAttention._flash_attention, which ignores max_seqlen and
        # masks segments solely from cu_seqlens.
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, cu_seqlens, max_seqlen=None)
        hidden_states = _unwrap_linear(self.proj1(self.ln_post(hidden_states)))
        hidden_states = _unwrap_linear(self.proj2(self.act(hidden_states)))
        if device.type == "tpu":
            # Cut the encoder XLA graph at the static (padded_tokens, D)
            # shape so the 32-layer encoder compiles once per bucket;
            # the [:cnn_len] slice below is padded back to the static
            # token bucket in _omni_merge before the LM graph cut.
            synchronize_tensors(hidden_states, wait=False)
        outputs.append(hidden_states[:cnn_len])
    return torch.cat(outputs, dim=0)


setattr(_padded_encoder_forward, _AUDIO_CHUNK_LENGTHS_PATCHED, True)


def _patch_audio_encoder_cls(audio_cls: type[Any]) -> bool:
    orig_forward = audio_cls.forward
    if getattr(orig_forward, _AUDIO_CHUNK_LENGTHS_PATCHED, False):
        return False
    audio_cls._tpu_orig_forward = orig_forward
    audio_cls.forward = _padded_encoder_forward
    return True


def maybe_patch_qwen3_omni_thinker(model_config: ModelConfig | None = None) -> bool:
    """Apply TPU patches for Qwen3-Omni Thinker audio encoder static bucketing.

    Applies to ``Qwen3OmniMoeForConditionalGeneration`` only, matched
    literally in ``ModelConfig.architectures`` (for a vLLM-Omni stage that is
    the pipeline's ``model_arch``, which is this name for every Qwen3-Omni
    stage; upstream vLLM also registers this architecture directly for
    Thinker-only serving). Returns ``False`` without importing model modules
    for anything else, including the ``None`` that ``platform_activation``
    passes before a model is configured, so the registry does not record the
    patch as applied and retries at ``model_load``.

    Patches upstream vLLM's ``Qwen3OmniMoeAudioEncoder`` unconditionally and,
    when ``vllm_omni`` is installed, also patches vLLM-Omni's subclass so
    plain-vLLM Thinker-only startup without the ``omni`` extra neither fails
    with ``ImportError`` nor leaves vLLM's encoder unpatched.
    """
    if model_config is None or _QWEN3_OMNI_ARCH not in model_config.architectures:
        return False

    from vllm.model_executor.models import qwen3_omni_moe_thinker as vllm_thinker_mod

    patched = _patch_audio_encoder_cls(vllm_thinker_mod.Qwen3OmniMoeAudioEncoder)

    if importlib.util.find_spec("vllm_omni") is not None:
        from vllm_omni.model_executor.models.qwen3_omni import (
            qwen3_omni_moe_thinker as thinker_mod,
        )

        patched = (
            _patch_audio_encoder_cls(thinker_mod.Qwen3OmniMoeAudioEncoder) or patched
        )

    if patched:
        logger.info(
            "Applied TPU patch: Qwen3-Omni audio encoder static-shape bucketing."
        )
    return True
