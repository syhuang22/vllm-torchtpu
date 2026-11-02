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
"""HuggingFace config.json resolution and quantization dtype extraction."""

from __future__ import annotations

import http.client
import json
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import envs
from .modules import Dtypes

if TYPE_CHECKING:
    from collections.abc import Mapping
from dataclasses import dataclass

_FOUR_BITS: int = 4
_EIGHT_BITS: int = 8
_BLOCK_SIZE_LEN: int = 2
_HTTP_OK: int = 200
_MXFP4_BYTES: float = 0.5 + 1.0 / 32.0
_NVFP4_BYTES: float = 0.5 + 1.0 / 16.0

_BASE_DTYPE_BYTES: dict[str, float] = {
    "float32": 4.0,
    "fp32": 4.0,
    "bfloat16": 2.0,
    "bf16": 2.0,
    "float16": 2.0,
    "fp16": 2.0,
    "float8_e4m3fn": 1.0,
    "fp8": 1.0,
}


def _resolve_cached_or_local_config(raw_str: str) -> dict[str, Any] | str:
    """Resolve HF config from local filesystem or local HuggingFace cache."""
    p = Path(raw_str)
    if p.is_file():
        return dict(json.loads(p.read_text()))
    if p.is_dir() and (p / "config.json").is_file():
        return dict(json.loads((p / "config.json").read_text()))

    if "/" in raw_str and not raw_str.startswith(("http://", "https://")):
        org_repo = raw_str.strip("/").replace("/", "--")
        hf_home = Path(envs.HF_HOME or (Path.home() / ".cache" / "huggingface"))
        for root in (hf_home / "hub", hf_home):
            snap_dir = root / f"models--{org_repo}" / "snapshots"
            if snap_dir.is_dir():
                snaps = sorted(snap_dir.glob("*/config.json"))
                if snaps:
                    return dict(json.loads(snaps[-1].read_text()))
        return f"https://huggingface.co/{raw_str.strip('/')}/resolve/main/config.json"
    if raw_str.startswith(("http://", "https://")):
        return raw_str
    msg = f"Could not resolve HF config path: {raw_str}"
    raise FileNotFoundError(msg)


def _fetch_remote_hf_config(url: str, token: str) -> dict[str, Any]:
    """Fetch remote HF config via http.client to avoid urllib scheme auditing flags."""
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme not in ("https", "http") or not parsed_url.hostname:
        msg = f"Unsupported URL scheme for HF config: {url!r}"
        raise ValueError(msg)

    headers = {"Accept": "application/json"}
    if (
        token
        and parsed_url.scheme == "https"
        and parsed_url.hostname == "huggingface.co"
    ):
        headers["Authorization"] = f"Bearer {token}"

    is_https = parsed_url.scheme == "https"
    port = parsed_url.port or (443 if is_https else 80)
    conn_cls = http.client.HTTPSConnection if is_https else http.client.HTTPConnection
    conn = conn_cls(parsed_url.hostname, port=port, timeout=15)
    path = parsed_url.path or "/"
    if parsed_url.query:
        path = f"{path}?{parsed_url.query}"
    conn.request("GET", path, headers=headers)
    resp = conn.getresponse()
    if resp.status != _HTTP_OK:
        msg = f"HTTP {resp.status} fetching HF config from {url}"
        conn.close()
        raise FileNotFoundError(msg)
    data = resp.read()
    conn.close()
    return dict(json.loads(data.decode("utf-8")))


def load_hf_config(
    model_or_path: str | Path | Mapping[str, Any],
    *,
    offline: bool = False,
) -> dict[str, Any]:
    """Load HF config.json from an in-memory dict, local path, HF cache, or Hub URL."""
    if type(model_or_path) is dict:
        return dict(model_or_path)
    raw_str = str(model_or_path)
    resolved = _resolve_cached_or_local_config(raw_str)
    if isinstance(resolved, dict):
        return resolved

    if offline or envs.HF_HUB_OFFLINE or envs.TRANSFORMERS_OFFLINE:
        msg = f"Offline mode is enabled and {raw_str!r} was not found in local cache"
        raise FileNotFoundError(msg)

    token = envs.HF_TOKEN or envs.HUGGING_FACE_HUB_TOKEN
    return _fetch_remote_hf_config(resolved, token)


def _extract_base_and_ssm_dtypes(
    tc: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    has_qcfg: bool = False,
) -> tuple[float, float]:
    """Extract standard base model and SSM state byte sizes from config."""
    raw_dtype = (
        tc.get("dtype")
        or cfg.get("dtype")
        or tc.get("torch_dtype")
        or cfg.get("torch_dtype")
        or ("bfloat16" if has_qcfg else None)
    )
    if not raw_dtype:
        msg = "HF config missing required field: 'dtype' or 'torch_dtype'"
        raise ValueError(msg)
    base_name = str(raw_dtype).lower().removeprefix("torch.")
    if base_name not in _BASE_DTYPE_BYTES:
        supp = sorted(_BASE_DTYPE_BYTES)
        msg = f"Unknown base dtype: {base_name!r}; supported: {supp}"
        raise ValueError(msg)
    base_b = _BASE_DTYPE_BYTES[base_name]

    ssm_dtype_raw = (
        tc.get("mamba_ssm_dtype")
        or cfg.get("mamba_ssm_dtype")
        or tc.get("ssm_dtype")
        or cfg.get("ssm_dtype")
    )
    if ssm_dtype_raw:
        ssm_name = str(ssm_dtype_raw).lower().removeprefix("torch.")
        if ssm_name not in _BASE_DTYPE_BYTES:
            msg = f"Unknown mamba_ssm_dtype: {ssm_dtype_raw!r}"
            raise ValueError(msg)
        state_b = _BASE_DTYPE_BYTES[ssm_name]
    else:
        state_b = base_b
    return base_b, state_b


@dataclass(frozen=True)
class _QuantParams:
    """Extracted quantization parameters."""

    q_method: str
    q_algo: str
    bits: int
    group_size: int | None
    scale_bytes: float | None


def _extract_config_groups_params(
    qcfg: Mapping[str, Any],
) -> tuple[int, int | None, float | None]:
    """Extract bits, group_size, and scale_bytes from config_groups if present."""
    bits = 0
    group_size: int | None = None
    scale_bytes: float | None = None
    cfg_groups = qcfg.get("config_groups")
    if isinstance(cfg_groups, dict):
        for grp in cfg_groups.values():
            if isinstance(grp, dict) and isinstance(grp.get("weights"), dict):
                w = grp["weights"]
                nb = int(w.get("num_bits", 0) or 0)
                if nb > 0 and bits == 0:
                    bits = nb
                gs = w.get("group_size")
                if gs is not None and group_size is None:
                    group_size = int(gs)
                sb = w.get("scale_bytes")
                if sb is not None and scale_bytes is None:
                    scale_bytes = float(sb)
    return bits, group_size, scale_bytes


def _extract_scale_bytes(
    qcfg: Mapping[str, Any],
    fallback_sb: float | None,
) -> float | None:
    """Extract scale_bytes from scale_bytes or scale_dtype."""
    if qcfg.get("scale_bytes") is not None:
        return float(qcfg["scale_bytes"])
    if qcfg.get("scale_dtype") is not None:
        sd = str(qcfg["scale_dtype"]).lower().removeprefix("torch.")
        if sd not in _BASE_DTYPE_BYTES:
            msg = f"Unknown 4-bit scale_dtype: {sd!r}"
            raise ValueError(msg)
        return _BASE_DTYPE_BYTES[sd]
    return fallback_sb


def _extract_quant_params(
    qcfg: Mapping[str, Any],
) -> _QuantParams:
    """Extract quantization params from quantization_config."""
    q_method = str(qcfg.get("quant_method") or qcfg.get("format") or "").lower()
    q_algo = str(qcfg.get("quant_algo") or "").lower()
    bits = int(qcfg.get("bits", 0) or 0)
    group_size: int | None = None
    if "group_size" in qcfg and qcfg.get("group_size") is not None:
        group_size = int(qcfg["group_size"])

    g_bits, g_size, g_scale = _extract_config_groups_params(qcfg)
    if bits == 0 and g_bits > 0:
        bits = g_bits
    if group_size is None and g_size is not None:
        group_size = g_size

    scale_bytes = _extract_scale_bytes(qcfg, g_scale)
    return _QuantParams(q_method, q_algo, bits, group_size, scale_bytes)


def _extract_ignored_modules(
    qcfg: Mapping[str, Any],
) -> tuple[bool, bool, bool]:
    """Detect whether attention, lm_head, and aux layers skip quantization."""
    ign_raw = (
        qcfg.get("modules_to_not_convert")
        or qcfg.get("ignore")
        or qcfg.get("ignored_layers")
    )
    has_ign_list = ign_raw is not None and len(tuple(ign_raw)) > 0
    ignored = tuple(str(x).lower() for x in (ign_raw or ()))
    attn_ignored = any(
        "linear_attn" not in x
        and any(
            seg in ("self_attn", "attn", "q_proj", "k_proj", "v_proj", "o_proj")
            for seg in x.replace("/", ".").split(".")
        )
        for x in ignored
    )
    lm_ignored = any("lm_head" in x for x in ignored) or not bool(
        qcfg.get("quantize_lm_head", False)
    )
    aux_ignored = (
        any(k in x for x in ignored for k in ("mtp", "gate", "ba", "router"))
        if has_ign_list
        else not bool(qcfg.get("quantize_aux", False))
    )
    return attn_ignored, lm_ignored, aux_ignored


def _derive_nvfp4_dense(
    qcfg: Mapping[str, Any],
    fp8_b: float,
    base_b: float,
    *,
    attn_ignored: bool,
) -> float:
    """Derive dense weight bytes for NVFP4."""
    dense_override = str(qcfg.get("dense_dtype", "")).lower()
    if dense_override == "fp8":
        return fp8_b
    if attn_ignored:
        return base_b
    return _NVFP4_BYTES


def _derive_four_bit_bytes(
    group_size: int,
    scale_bytes: float | None,
) -> float:
    """Derive weight bytes for 4-bit grouped quantization."""
    if scale_bytes is None or scale_bytes <= 0.0:
        msg = (
            "4-bit quantization config missing required field: "
            "'scale_bytes' or 'scale_dtype'"
        )
        raise ValueError(msg)
    return 0.5 + scale_bytes / float(group_size)


def _derive_quantized_weight_bytes(
    qp: _QuantParams,
    fp8_b: float,
    base_b: float,
    qcfg: Mapping[str, Any],
    *,
    attn_ignored: bool,
) -> tuple[float, float]:
    """Derive (w_dense, w_expert) byte footprints across quantization schemes."""
    if qp.bits == _FOUR_BITS:
        if qp.group_size is None:
            msg = "4-bit quantization config missing required field: 'group_size'"
            raise ValueError(msg)
        if qp.group_size <= 0:
            msg = f"Invalid 4-bit group_size: {qp.group_size}"
            raise ValueError(msg)

    if "mxfp4" in qp.q_method or "mxfp4" in qp.q_algo:
        w_expert = _MXFP4_BYTES
        w_dense = (
            base_b
            if (attn_ignored or not qcfg.get("quantize_attn", False))
            else _MXFP4_BYTES
        )
    elif (
        "nvfp4" in qp.q_method
        or "nvfp4" in qp.q_algo
        or ("compressed" in qp.q_method and "nvfp4" in str(qcfg).lower())
        or ("modelopt" in qp.q_method and "nvfp4" in str(qcfg).lower())
    ):
        w_expert = _NVFP4_BYTES
        w_dense = _derive_nvfp4_dense(qcfg, fp8_b, base_b, attn_ignored=attn_ignored)
    elif qp.bits == _FOUR_BITS and qp.group_size is not None:
        w_expert = _derive_four_bit_bytes(qp.group_size, qp.scale_bytes)
        w_dense = base_b if attn_ignored else w_expert
    elif "fp8" in qp.q_method or qp.bits == _EIGHT_BITS:
        w_expert = fp8_b
        w_dense = base_b if attn_ignored else fp8_b
    else:
        w_expert = base_b
        w_dense = base_b
    return w_dense, w_expert


def dtypes_from_hf_config(cfg: Mapping[str, Any]) -> Dtypes:
    """Derive Dtypes from torch_dtype, quantization_config, and ignored module lists."""
    sub = cfg.get("text_config")
    tc: Mapping[str, Any] = sub if sub is not None and type(sub) is dict else cfg
    qraw_pre = tc.get("quantization_config") or cfg.get("quantization_config")
    has_qcfg = qraw_pre is not None and type(qraw_pre) is dict and len(qraw_pre) > 0

    base_b, state_b = _extract_base_and_ssm_dtypes(tc, cfg, has_qcfg=has_qcfg)
    qraw = tc.get("quantization_config") or cfg.get("quantization_config")
    qcfg: Mapping[str, Any] = qraw if qraw is not None and type(qraw) is dict else {}
    if not qcfg:
        act_mode = "a8" if base_b <= 1.0 else "a16"
        return Dtypes(
            w_dense=base_b,
            w_expert=base_b,
            w_aux=base_b,
            kv=base_b,
            state=state_b,
            conv=base_b,
            router=base_b,
            lm_head=base_b,
            act=act_mode,
        )

    qp = _extract_quant_params(qcfg)
    attn_ignored, lm_ignored, aux_ignored = _extract_ignored_modules(qcfg)

    block_sz = qcfg.get("weight_block_size")
    fp8_b = 1.0
    if (
        block_sz is not None
        and type(block_sz) in (list, tuple)
        and len(block_sz) == _BLOCK_SIZE_LEN
    ):
        b0, b1 = int(block_sz[0]), int(block_sz[1])
        if b0 > 0 and b1 > 0:
            fp8_b = 1.0 + 4.0 / float(b0 * b1)

    w_dense, w_expert = _derive_quantized_weight_bytes(
        qp,
        fp8_b,
        base_b,
        qcfg,
        attn_ignored=attn_ignored,
    )

    kv_scheme = str(
        qcfg.get("kv_cache_scheme") or qcfg.get("kv_cache_dtype") or ""
    ).lower()
    kv_b = 1.0 if "fp8" in kv_scheme or "8" in kv_scheme else base_b

    quant_conv = bool(qcfg.get("quantize_conv", False))
    conv_b = w_dense if quant_conv else base_b

    quant_router = bool(qcfg.get("quantize_router", False)) or (
        bool(qcfg.get("quantize_aux", False)) and not aux_ignored
    )
    router_b = w_dense if quant_router else base_b

    return Dtypes(
        w_dense=w_dense,
        w_expert=w_expert,
        w_aux=base_b if aux_ignored else w_dense,
        kv=kv_b,
        state=state_b,
        conv=conv_b,
        router=router_b,
        lm_head=base_b if lm_ignored else w_dense,
        act="a8",
    )
