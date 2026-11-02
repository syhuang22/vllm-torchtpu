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
"""Model geometry and 10-family layer registry from HuggingFace config.json."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

LAYER_FAMILIES: tuple[str, ...] = (
    "gqa",
    "mla",
    "dsa",
    "swa",
    "gdn",
    "kda",
    "dense_mlp",
    "moe",
    "mtp",
    "ple",
)

_VALID_LAYER_TOKENS: frozenset[str] = frozenset(
    {
        "full_attention",
        "attention",
        "gqa",
        "mla",
        "mla_attention",
        "dsa",
        "dsa_attention",
        "sparse_mla",
        "sliding_attention",
        "swa",
        "sliding_window_attention",
        "linear_attention",
        "gdn",
        "gated_deltanet",
        "mamba",
        "kda",
        "kimi_linear",
        "kda_attention",
        "dense_mlp",
        "mlp",
        "moe",
        "mtp",
        "ple",
        "ffn_only",
    }
)

_LINEAR_ATTENTION_LAYER_TOKENS: frozenset[str] = frozenset(
    {
        "linear_attention",
        "gdn",
        "gated_deltanet",
        "mamba",
        "kda",
        "kimi_linear",
        "kda_attention",
    }
)


@dataclass(frozen=True)
class ModelGeometry:
    """Architecture dimensions shared across attention, recurrence, and FFN families."""

    hidden: int
    num_layers: int
    layer_types: tuple[str, ...]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    attn_output_gate: bool
    linear_k_heads: int
    linear_v_heads: int
    linear_k_dim: int
    linear_v_dim: int
    conv_kernel: int
    num_experts: int
    top_k: int
    moe_inter: int
    shared_inter: int
    vocab: int
    mtp_layers: int
    dense_inter: int = 0
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    index_n_heads: int = 0
    index_head_dim: int = 0
    index_topk: int = 0
    sliding_window: int = 0
    num_shared_experts: int = 0
    first_k_dense_replace: int = 0
    ple_dim: int = 0
    compress_ratios: tuple[float, ...] = ()
    block_parallel: int = 1
    mtp_inter: int = 0
    n_moe_layers: int | None = None
    n_dense_layers: int | None = None
    global_head_dim: int = 0
    ffn_kinds: tuple[str, ...] = ()

    @staticmethod
    def from_hf_config(cfg: Mapping[str, Any]) -> ModelGeometry:
        """Build ModelGeometry from a raw or nested HF config mapping."""
        return from_hf_config(cfg)

    def ffn_layer_counts(self) -> tuple[int, int]:
        """Return (n_moe_ffn, n_dense_ffn) counted from ffn_kinds."""
        return (self.ffn_kinds.count("moe"), self.ffn_kinds.count("dense"))


class UnsupportedArchitectureError(ValueError):
    """Raised when a HuggingFace config describes an unmodeled architecture."""


_UNSUPPORTED_SSM_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "mamba",
        "mamba2",
        "rwkv",
        "rwkv6",
        "falcon_mamba",
        "xlstm",
        "retnet",
        "hyena",
        "stripedhyena",
    }
)

_UNSUPPORTED_NON_CAUSAL_TYPES: frozenset[str] = frozenset(
    {"bert", "roberta", "t5", "whisper", "vit", "clip", "dit", "flux", "sdxl"}
)

_MULTIMODAL_KEYS: tuple[str, ...] = (
    "vision_config",
    "audio_config",
    "video_config",
    "mm_projector_type",
    "image_token_index",
)


def _validate_text_config(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate and extract text_config mapping from top-level config."""
    if not cfg:
        msg = "Empty HuggingFace config mapping"
        raise UnsupportedArchitectureError(msg)
    has_text_key = "text_config" in cfg
    sub = cfg.get("text_config")
    if has_text_key and (sub is None or type(sub) is not dict):
        msg = "Multimodal config has non-mapping text_config"
        raise UnsupportedArchitectureError(msg)
    tc: Mapping[str, Any] = sub if sub is not None and type(sub) is dict else cfg
    is_multimodal = has_text_key or any(k in cfg for k in _MULTIMODAL_KEYS)
    if is_multimodal and not tc.get("hidden_size"):
        msg = "Multimodal config missing extractable text_config.hidden_size"
        raise UnsupportedArchitectureError(msg)
    return tc


def _has_gdn_kda_heads(tc: Mapping[str, Any]) -> bool:
    """Return whether text config declares linear or recurrent attention heads."""
    lin_raw = tc.get("linear_attn_config")
    lin_map: Mapping[str, Any] = (
        lin_raw if lin_raw is not None and type(lin_raw) is dict else {}
    )
    return bool(
        tc.get("linear_num_key_heads")
        or tc.get("linear_num_value_heads")
        or lin_map.get("num_key_heads")
        or lin_map.get("num_value_heads")
        or lin_map.get("num_heads")
        or tc.get("mamba_num_heads")
        or tc.get("attn_type_list")
        or tc.get("hybrid_override_pattern")
    )


def supports(cfg: Mapping[str, Any]) -> None:
    """Validate that cfg maps onto modeled attention, recurrence, and FFN families."""
    tc = _validate_text_config(cfg)
    mtype = str(tc.get("model_type") or cfg.get("model_type") or "").strip().lower()
    if mtype in _UNSUPPORTED_NON_CAUSAL_TYPES:
        msg = f"Unsupported non-causal-LM model_type {mtype!r}"
        raise UnsupportedArchitectureError(msg)
    has_gdn = _has_gdn_kda_heads(tc)
    if mtype in _UNSUPPORTED_SSM_MODEL_TYPES and not has_gdn:
        msg = (
            f"Unsupported state-space model_type {mtype!r} without GDN/KDA head keys"
        )
        raise UnsupportedArchitectureError(msg)
    if ("ssm_cfg" in tc or "state_size" in tc) and not (
        tc.get("num_attention_heads") or has_gdn
    ):
        msg = "Unmodeled state-space config missing attention and GDN/KDA head keys"
        raise UnsupportedArchitectureError(msg)
    raw_types = tc.get("layer_types")
    if raw_types is not None and type(raw_types) in (list, tuple):
        for token in raw_types:
            tok_s = str(token)
            if tok_s not in _VALID_LAYER_TOKENS:
                msg = (
                    f"Unknown layer_type {tok_s!r}; supported tokens: "
                    f"{sorted(_VALID_LAYER_TOKENS)}"
                )
                raise UnsupportedArchitectureError(msg)
            if tok_s == "mamba" and not has_gdn:
                msg = "Layer token 'mamba' requires GDN/KDA head dimensions"
                raise UnsupportedArchitectureError(msg)


def _resolve_layer_types(
    tc: Mapping[str, Any],
    cfg: Mapping[str, Any],
    num_layers: int,
    kv_lora_rank: int,
    index_topk: int,
    sliding_window: int,
) -> tuple[str, ...]:
    """Resolve sequence of layer type strings for each transformer layer."""
    raw_types = tc.get("layer_types")
    if raw_types is not None and type(raw_types) in (list, tuple):
        if len(raw_types) != num_layers:
            msg = (
                f"layer_types length {len(raw_types)} != num_hidden_layers {num_layers}"
            )
            raise ValueError(msg)
        layer_types = tuple(str(x) for x in raw_types)
        for token in layer_types:
            if token not in _VALID_LAYER_TOKENS:
                supported = sorted(_VALID_LAYER_TOKENS)
                msg = f"Unknown layer_type {token!r}; supported tokens: {supported}"
                raise ValueError(msg)
        return layer_types

    interval = int(tc.get("full_attention_interval", 0) or 0)
    attn_list = tc.get("attn_type_list")
    hyb_pat = tc.get("hybrid_override_pattern")
    sw_pat = int(
        tc.get("sliding_window_pattern") or tc.get("_sliding_window_pattern") or 0
    )
    if interval > 0:
        att_tok = (
            "dsa"
            if (kv_lora_rank > 0 and index_topk > 0)
            else ("mla" if kv_lora_rank > 0 else "full_attention")
        )
        archs = cfg.get("architectures") or tc.get("architectures") or ()
        is_kimi = str(tc.get("model_type", "")).lower().startswith("kimi") or any(
            "kimi" in str(a).lower() for a in archs
        )
        lin_tok = "kda" if is_kimi else "linear_attention"
        return tuple(
            att_tok if (i + 1) % interval == 0 else lin_tok for i in range(num_layers)
        )
    if (
        attn_list is not None
        and type(attn_list) in (list, tuple)
        and len(attn_list) == num_layers
    ):
        return tuple(
            "full_attention" if int(v) == 1 else "linear_attention"
            for v in attn_list
        )
    if type(hyb_pat) is str and len(hyb_pat.strip()) == num_layers:
        _hop = {
            "*": "full_attention",
            "M": "linear_attention",
            "-": "ffn_only",
            "E": "ffn_only",
        }
        pat_s = hyb_pat.strip()
        if any(ch not in _hop for ch in pat_s):
            msg = f"Unsupported hybrid_override_pattern tokens in {pat_s!r}"
            raise UnsupportedArchitectureError(msg)
        return tuple(_hop[ch] for ch in pat_s)
    if sw_pat > 1 and sliding_window > 0:
        return tuple(
            "full_attention" if (i + 1) % sw_pat == 0 else "sliding_attention"
            for i in range(num_layers)
        )
    if kv_lora_rank > 0:
        default_mla = "dsa" if index_topk > 0 else "mla"
        return (default_mla,) * num_layers

    return ("full_attention",) * num_layers


def _resolve_moe_dimensions(
    tc: Mapping[str, Any],
) -> tuple[int, int, int, int, int]:
    """Resolve (num_experts, top_k, moe_inter, num_shared_experts, shared_inter)."""
    num_experts = int(
        tc.get("num_experts")
        or tc.get("n_routed_experts")
        or tc.get("num_local_experts")
        or 0
    )
    dense_inter = int(
        tc.get("intermediate_size") or tc.get("dense_intermediate_size") or 0
    )
    moe_inter = int(tc.get("moe_intermediate_size") or tc.get("latent_moe_dim") or 0)
    if num_experts > 0 and moe_inter == 0 and dense_inter > 0:
        moe_inter = dense_inter

    num_shared = int(tc.get("n_shared_experts") or tc.get("num_shared_experts") or 0)
    shared_inter = int(tc.get("shared_expert_intermediate_size", 0) or 0)
    if shared_inter == 0 and num_shared > 0 and moe_inter > 0:
        shared_inter = num_shared * moe_inter

    top_k = int(tc.get("num_experts_per_tok") or tc.get("num_selected_experts") or 0)
    return num_experts, top_k, moe_inter, num_shared, shared_inter


def _resolve_ffn_kinds(
    tc: Mapping[str, Any],
    num_layers: int,
    num_experts: int,
) -> tuple[str, ...]:
    """Resolve (moe, dense, none) kind for each layer's feed-forward network."""
    hyb_s = str(tc.get("hybrid_override_pattern") or "").strip()
    if hyb_s and len(hyb_s) == num_layers:
        return tuple(
            "moe" if ch == "E" else ("dense" if ch == "-" else "none")
            for ch in hyb_s
        )

    raw_moe_layers = tc.get("moe_layers")
    if raw_moe_layers is not None and type(raw_moe_layers) in (list, tuple):
        moe_set = {int(x) for x in raw_moe_layers}
        return tuple("moe" if i in moe_set else "dense" for i in range(num_layers))

    interleave_step = int(tc.get("interleave_moe_layer_step", 0) or 0)
    if num_experts > 0 and interleave_step > 1:
        return tuple(
            "moe" if (i + 1) % interleave_step == 0 else "dense"
            for i in range(num_layers)
        )

    first_k_dense = int(tc.get("first_k_dense_replace", 0) or 0)
    if num_experts > 0:
        return tuple(
            "dense" if i < first_k_dense else "moe" for i in range(num_layers)
        )

    return ("dense",) * num_layers


def _resolve_linear_attention_dimensions(
    tc: Mapping[str, Any],
    layer_types: tuple[str, ...],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[int, int, int, int, int]:
    """Resolve (lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_kernel)."""
    lin_cfg_raw = tc.get("linear_attn_config")
    lin_cfg: Mapping[str, Any] = (
        lin_cfg_raw if lin_cfg_raw is not None and type(lin_cfg_raw) is dict else {}
    )
    has_attn_list = tc.get("attn_type_list") is not None
    lin_k_heads = int(
        tc.get("linear_num_key_heads")
        or lin_cfg.get("num_key_heads")
        or lin_cfg.get("num_heads")
        or tc.get("mamba_num_heads")
        or (num_kv_heads if has_attn_list else 0)
        or 0
    )
    lin_v_heads = int(
        tc.get("linear_num_value_heads")
        or lin_cfg.get("num_value_heads")
        or lin_cfg.get("num_heads")
        or tc.get("mamba_num_heads")
        or (num_heads if has_attn_list else 0)
        or 0
    )
    lin_k_dim = int(
        tc.get("linear_key_head_dim")
        or lin_cfg.get("key_head_dim")
        or lin_cfg.get("head_dim")
        or tc.get("mamba_head_dim")
        or (head_dim if has_attn_list else 0)
        or 0
    )
    lin_v_dim = int(
        tc.get("linear_value_head_dim")
        or lin_cfg.get("value_head_dim")
        or lin_cfg.get("head_dim")
        or tc.get("mamba_head_dim")
        or (head_dim if has_attn_list else 0)
        or 0
    )

    has_linear_layers = any(t in _LINEAR_ATTENTION_LAYER_TOKENS for t in layer_types)
    raw_conv = tc.get("linear_conv_kernel_dim")
    if raw_conv is None:
        raw_conv = (
            lin_cfg.get("short_conv_kernel_size")
            or lin_cfg.get("conv_kernel_size")
            or lin_cfg.get("linear_conv_kernel_dim")
            or tc.get("conv_kernel")
            or tc.get("d_conv")
        )
    if has_linear_layers:
        if raw_conv is None:
            if has_attn_list:
                conv_kernel = 0
            else:
                msg = (
                    "Model with linear-attention layers missing required field: "
                    "'linear_conv_kernel_dim'"
                )
                raise ValueError(msg)
        else:
            conv_kernel = int(raw_conv)
    else:
        conv_kernel = 0

    return lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_kernel


def _resolve_mtp_layers(tc: Mapping[str, Any], cfg: Mapping[str, Any]) -> int:
    """Resolve number of Multi-Token Prediction (MTP) layers."""
    mtp_raw = tc.get("mtp_num_hidden_layers")
    if mtp_raw is None:
        mtp_raw = cfg.get("mtp_num_hidden_layers")
    if mtp_raw is None:
        mtp_raw = tc.get(
            "num_nextn_predict_layers", cfg.get("num_nextn_predict_layers", 0)
        )
    return int(mtp_raw or 0)


def _resolve_compress_ratios(
    tc: Mapping[str, Any],
    cfg: Mapping[str, Any],
    num_layers: int,
) -> tuple[float, ...]:
    """Resolve per-layer compression ratios."""
    raw_cr = tc.get("compress_ratios", cfg.get("compress_ratios"))
    if raw_cr is not None and type(raw_cr) in (list, tuple) and len(raw_cr) > 0:
        if len(raw_cr) != num_layers:
            msg = (
                f"compress_ratios length {len(raw_cr)} != "
                f"num_hidden_layers {num_layers}"
            )
            raise ValueError(msg)
        compress_ratios = tuple(float(x) for x in raw_cr)
        if any(r <= 0.0 for r in compress_ratios):
            msg = f"compress_ratios entries must be > 0, got {compress_ratios}"
            raise ValueError(msg)
        return compress_ratios
    return ()


def _resolve_block_parallel(tc: Mapping[str, Any], cfg: Mapping[str, Any]) -> int:
    """Resolve block parallelism factor (1 or 2)."""
    raw_bp = tc.get("block_parallel", cfg.get("block_parallel"))
    if raw_bp is None:
        raw_bp = tc.get("parallel_attn_mlp", cfg.get("parallel_attn_mlp", 1))
    return (2 if raw_bp else 1) if type(raw_bp) is bool else max(1, int(raw_bp or 1))


def from_hf_config(cfg: Mapping[str, Any]) -> ModelGeometry:
    """Build ModelGeometry from a raw or nested HF config mapping."""
    supports(cfg)
    sub = cfg.get("text_config")
    tc: Mapping[str, Any] = sub if sub is not None and type(sub) is dict else cfg
    hidden = int(tc["hidden_size"])
    num_layers = int(tc["num_hidden_layers"])
    num_heads = int(tc["num_attention_heads"])
    num_kv_heads = int(tc.get("num_key_value_heads", num_heads))
    head_dim = int(tc.get("head_dim") or (hidden // num_heads))

    kv_lora_rank = int(tc.get("kv_lora_rank", 0) or 0)
    q_lora_rank = int(tc.get("q_lora_rank", 0) or 0)
    qk_nope_head_dim = int(tc.get("qk_nope_head_dim", 0) or 0)
    qk_rope_head_dim = int(tc.get("qk_rope_head_dim", 0) or 0)
    v_head_dim = int(tc.get("v_head_dim", 0) or 0)
    index_n_heads = int(tc.get("index_n_heads", 0) or 0)
    index_head_dim = int(tc.get("index_head_dim", 0) or 0)
    index_topk = int(tc.get("index_topk", 0) or 0)
    sliding_window = int(tc.get("sliding_window", 0) or 0)
    global_head_dim = int(tc.get("global_head_dim", 0) or 0)

    layer_types = _resolve_layer_types(
        tc, cfg, num_layers, kv_lora_rank, index_topk, sliding_window
    )
    dense_inter = int(
        tc.get("intermediate_size") or tc.get("dense_intermediate_size") or 0
    )
    num_experts, top_k, moe_inter, num_shared, shared_inter = _resolve_moe_dimensions(
        tc
    )
    ffn_kinds = _resolve_ffn_kinds(tc, num_layers, num_experts)
    n_moe_layers, n_dense_layers = (
        ffn_kinds.count("moe"),
        ffn_kinds.count("dense"),
    )
    lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_kernel = (
        _resolve_linear_attention_dimensions(
            tc, layer_types, num_heads, num_kv_heads, head_dim
        )
    )
    mtp_layers = _resolve_mtp_layers(tc, cfg)
    ple_dim = int(tc.get("per_layer_input_dim") or tc.get("ple_dim") or 0)
    compress_ratios = _resolve_compress_ratios(tc, cfg, num_layers)
    block_parallel = _resolve_block_parallel(tc, cfg)
    mtp_inter = int(
        tc.get("mtp_intermediate_size", cfg.get("mtp_intermediate_size", 0)) or 0
    )

    return ModelGeometry(
        hidden=hidden,
        num_layers=num_layers,
        layer_types=layer_types,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        attn_output_gate=bool(tc.get("attn_output_gate", False)),
        linear_k_heads=lin_k_heads,
        linear_v_heads=lin_v_heads,
        linear_k_dim=lin_k_dim,
        linear_v_dim=lin_v_dim,
        conv_kernel=conv_kernel,
        num_experts=num_experts,
        top_k=top_k,
        moe_inter=moe_inter,
        shared_inter=shared_inter,
        vocab=int(tc.get("vocab_size", cfg.get("vocab_size", 0)) or 0),
        mtp_layers=mtp_layers,
        dense_inter=dense_inter,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        sliding_window=sliding_window,
        num_shared_experts=num_shared,
        first_k_dense_replace=int(tc.get("first_k_dense_replace", 0) or 0),
        ple_dim=ple_dim,
        compress_ratios=compress_ratios,
        block_parallel=block_parallel,
        mtp_inter=mtp_inter,
        n_moe_layers=n_moe_layers,
        n_dense_layers=n_dense_layers,
        global_head_dim=global_head_dim,
        ffn_kinds=ffn_kinds,
    )
