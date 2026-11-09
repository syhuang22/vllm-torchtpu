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
"""Muse Spark Open 816B A42B Pallas decode megakernel fastpath for vllm serve."""

from __future__ import annotations

import dataclasses
import functools
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.muse_megakernel.config import (
    KV_GROUP,
    PAGE_TOKENS,
    SW_RING,
    TP_AXIS_NAME,
    MuseMegakernelConfig,
    residual_coefficients,
)
from vllm_torchtpu.kernels.muse_megakernel.load import (
    KERNEL_WEIGHT_NAMES,
    Checkpoint,
    lm_width,
    permute_gu_cols_sublane_jax,
    permute_k_sublane_jax,
    permute_sublane_axis_np,
    repack_nvfp4_k_sublane_jax,
    router_order_indices,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# vLLM's OpenAI server caps logprobs at 20; the step op always returns this many.
LOGPROB_K = 20
MAX_SLOTS = 32
KERNEL_BATCH = 16
SPEC_MAX_ROWS = 16
DEFAULT_BASE_KPAGES = 385
# DSpark spec decode. Draft depth per padded request batch: keeps nb * (kd + 1) <= 8
# for nb <= 4 (avoiding sparse MoE unique-expert fanout) and <= 16 for nb = 8.
SPEC_DRAFT_DEPTH = {1: 7, 2: 3, 4: 1, 8: 1, 16: 0, 32: 0}
# Tuned verify window length q per padded request batch on TPU v7x (TP=8).
SPEC_VERIFY_WINDOW_BY_BATCH = {1: 8, 2: 4, 4: 2, 8: 2, 16: 1}


def _bucket(n: int) -> int:
    return 1 << max(0, int(n - 1).bit_length())


_INSTANCE: MuseDecodeFastPath | None = None


def enabled() -> bool:
    """Return True when VLLM_TPU_MUSE_MEGAKERNEL=1 is enabled in vllm_torchtpu.envs."""
    return bool(envs.VLLM_TPU_MUSE_MEGAKERNEL)


def resolve_max_slots(max_num_seqs: int, *, spec: bool = False) -> int:
    """Derive megakernel decode slot table capacity from scheduler max_num_seqs."""
    del spec
    clamped = max(1, min(int(max_num_seqs), MAX_SLOTS))
    return min(MAX_SLOTS, _bucket(clamped))


def decode_buckets(n_slots: int) -> tuple[int, ...]:
    return tuple(b for b in (1, 2, 4, 8, 16) if b <= n_slots)


def instance() -> MuseDecodeFastPath | None:
    return _INSTANCE


def set_instance(fp: MuseDecodeFastPath | None) -> None:
    global _INSTANCE
    _INSTANCE = fp


class _StepTimer:
    """Host-side phase timer for megakernel decode steps (enabled at DEBUG level)."""

    def __init__(self) -> None:
        self._now = time.perf_counter
        self.sums: dict[str, dict[str, float]] = {"pure": {}, "mixed": {}}
        self.counts = {"pure": 0, "mixed": 0}
        self.walls: dict[str, list[float]] = {"pure": [], "mixed": []}
        self.cur: dict[str, float] = {}
        self.t: float | None = None
        self.t0: float | None = None

    def start(self) -> None:
        self.t = self.t0 = self._now()
        self.cur = {}

    def mark(self, name: str) -> None:
        if self.t is None:
            return
        now = self._now()
        self.cur[name] = self.cur.get(name, 0.0) + now - self.t
        self.t = now

    def end(self, kind: str) -> None:
        if self.t is None or self.t0 is None:
            return
        self.mark("assemble")
        for k, v in self.cur.items():
            self.sums[kind][k] = self.sums[kind].get(k, 0.0) + v
        self.walls[kind].append(self._now() - self.t0)
        self.counts[kind] += 1
        self.t = None
        n = self.counts["pure"] + self.counts["mixed"]
        if n % 256 == 0:
            for kd in ("pure", "mixed"):
                c = self.counts[kd]
                if not c:
                    continue
                w = np.asarray(self.walls[kd][-4096:]) * 1e3
                logger.info(
                    "Muse megakernel %s steps=%d mean phase ms %s "
                    "| wall ms mean %.2f p50 %.2f p90 %.2f max %.1f",
                    kd,
                    c,
                    {
                        k: round(1e3 * v / c, 3)
                        for k, v in sorted(self.sums[kd].items())
                    },
                    w.mean(),
                    np.percentile(w, 50),
                    np.percentile(w, 90),
                    w.max(),
                )


def _explicit_arity(body: Any, n_in: int, n_out: int) -> Any:
    args = ", ".join(f"a{i}: jax.Array" for i in range(n_in))
    ret = (
        "jax.Array" if n_out == 1 else "tuple[" + ", ".join(["jax.Array"] * n_out) + "]"
    )
    call_args = ", ".join(f"a{i}" for i in range(n_in))
    src = f"def f({args}) -> {ret}:\n    return body({call_args})\n"
    ns: dict[str, Any] = {"body": body, "jax": jax}
    exec(src, ns)  # noqa: S102
    return ns["f"]


def _to_torch(
    a: Any, dev: torch.device | str = "tpu", sync: bool = False
) -> torch.Tensor:
    a = np.ascontiguousarray(np.asarray(a))
    if not a.flags.writeable:
        a = a.copy()
    name = str(a.dtype)
    if name == "bfloat16":
        t = torch.from_numpy(a.view(np.uint16)).view(torch.bfloat16).to(dev)
    elif name == "float8_e4m3fn":
        t = torch.from_numpy(a.view(np.uint8)).view(torch.float8_e4m3fn).to(dev)
    else:
        t = torch.from_numpy(a).to(dev)
    if sync and not torch.compiler.is_compiling():
        from vllm_torchtpu.utils import synchronize_tensors

        synchronize_tensors([t])
    return t


def should_skip_expert_weight(name: str) -> bool:
    """Skip loading per-layer routed expert weights into vLLM when the megakernel shared
    store owns them.
    """
    return ".mlp.experts." in name and not name.endswith("post_expert_norm.weight")


def skipped_expert_param_names(model: Any) -> set[str]:
    """Return vLLM parameter names of the routed expert tensors skipped by
    `should_skip_expert_weight`.
    """
    names: set[str] = set()
    for pname, _ in model.named_parameters():
        if ".mlp.experts." in pname and not pname.endswith("post_expert_norm.weight"):
            names.add(pname)
    return names


def prepare_model_for_megakernel(model: Any, vllm_config: Any = None) -> None:
    """Free vLLM's duplicate EP routed expert parameter allocations before checkpoint
    loading.
    """
    del vllm_config
    text_model = getattr(model, "model", None)
    layers = getattr(text_model, "layers", None)
    if layers is None:
        return
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        routed = getattr(experts, "routed_experts", experts)
        if routed is None:
            continue
        for attr in (
            "w13_weight",
            "w13_weight_scale",
            "w13_weight_scale_inv",
            "w13_weight_scale_2",
            "w13_input_scale",
            "w2_weight",
            "w2_weight_scale",
            "w2_weight_scale_inv",
            "w2_weight_scale_2",
            "w2_input_scale",
        ):
            if hasattr(routed, attr) and getattr(routed, attr) is not None:
                t = getattr(routed, attr)
                if isinstance(t, torch.nn.Parameter):
                    setattr(
                        routed,
                        attr,
                        torch.nn.Parameter(
                            torch.empty(0, dtype=t.dtype, device=t.device),
                            requires_grad=False,
                        ),
                    )
                del t
        if hasattr(routed, "quant_method") and routed.quant_method is not None:
            routed.quant_method.process_weights_after_loading = lambda *a, **kw: None
    torch.accelerator.synchronize()


def build_for_model(model: Any) -> MuseDecodeFastPath:
    """Build and register the MuseDecodeFastPath instance after model weights load."""
    existing = getattr(model, "_muse_mk_fastpath", None) or instance()
    if existing is not None and existing.kw:
        existing._install_shared_moe(model)
        return existing

    from vllm.config import get_current_vllm_config_or_none

    vcfg = getattr(model, "vllm_config", None) or get_current_vllm_config_or_none()
    if vcfg is None:
        raise RuntimeError(
            "MuseDecodeFastPath requires vllm_config on model or active context"
        )
    if vcfg.scheduler_config.async_scheduling is not False:
        vcfg.scheduler_config.async_scheduling = False
    dummy_runner = dataclasses.make_dataclass(
        "_RunnerProxy",
        [("vllm_config", Any), ("cache_config", Any), ("model", Any)],
    )(vllm_config=vcfg, cache_config=vcfg.cache_config, model=model)
    fp = MuseDecodeFastPath(dummy_runner)
    fp.build(model)
    model._muse_mk_fastpath = fp
    set_instance(fp)
    return fp


class MuseSharedStoreMoE:
    """Callable installed on `MuseSparkMoE._fused_moe_args` to run prefill MoE on the
    shared TP8 expert store.
    """

    pre_projected = True

    def __init__(self, fp: MuseDecodeFastPath, layer_idx: int, moe: Any) -> None:
        self.fp = fp
        self.layer_idx = int(layer_idx)
        self.is_packed = self.layer_idx not in fp.cfg.bf16_expert_layers
        if self.is_packed:
            below = sum(1 for d in fp.cfg.bf16_expert_layers if self.layer_idx > d)
            store_idx = self.layer_idx - below
        else:
            store_idx = list(fp.cfg.bf16_expert_layers).index(self.layer_idx)
        dev = torch.device("tpu")
        self.idx_t = _to_torch(np.asarray([[store_idx]], dtype=np.int32), dev)
        self._head_tensors = (
            moe.router_w_kmajor.unsqueeze(0),
            moe.gate.e_score_correction_bias.unsqueeze(0),
            moe.pre_expert_norm.weight.unsqueeze(0),
            moe.post_expert_norm.weight.unsqueeze(0),
        )
        from vllm_torchtpu.utils import synchronize_tensors

        synchronize_tensors([*self._head_tensors, self.idx_t])

    def __call__(
        self, h: torch.Tensor, z_pre: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert z_pre is not None
        op = self.fp._get_shared_moe_op(self.is_packed)
        if self.is_packed:
            out = op(
                h.unsqueeze(0),
                z_pre.unsqueeze(0),
                *self._head_tensors,
                self.fp.kw["gu_pack"],
                self.fp.kw["dw_pack"],
                self.fp.kw["gu_global"],
                self.fp.kw["dw_global"],
                self.idx_t,
                self.fp.rank_t,
            )
        else:
            out = op(
                h.unsqueeze(0),
                z_pre.unsqueeze(0),
                *self._head_tensors,
                self.fp.kw["gu_dense"],
                self.fp.kw["dw_dense"],
                self.idx_t,
                self.fp.rank_t,
            )
        return out.squeeze(0)


@dataclasses.dataclass
class SlotState:
    slot: int
    kv_upto: int = 0
    draft_upto: int = 0
    owned: bool = False
    last_cached_tokens: int = 0
    last_next_tok: int = -1
    # DSpark: next verify window [bonus, d0, ...] at position kv_upto, and draft
    # confidences.
    next_verify: list[int] | None = None
    next_conf: list[float] | None = None
    # Global KV superpages this request may need over its lifetime (reserved at
    # takeover).
    kp_need: int = 0


class SlotTable:
    def __init__(self, n: int) -> None:
        self.free = list(range(n - 1, -1, -1))
        self.by_req: dict[str, SlotState] = {}

    def get(self, req_id: str) -> SlotState | None:
        return self.by_req.get(req_id)

    def acquire(self, req_id: str) -> SlotState | None:
        st = self.by_req.get(req_id)
        if st is not None:
            return st
        if not self.free:
            return None
        st = SlotState(slot=self.free.pop())
        self.by_req[req_id] = st
        return st

    def release(self, req_id: str) -> SlotState | None:
        st = self.by_req.pop(req_id, None)
        if st is not None:
            self.free.append(st.slot)
        return st


class MuseDecodeFastPath:
    """Single-dispatch Pallas decode megakernel fastpath for Muse Spark on TPU v7x
    (TP=8).
    """

    cache_kpages: int = 0

    def __init__(self, runner: Any) -> None:
        from vllm.distributed import parallel_state

        from vllm_torchtpu.distributed.mesh_utils import get_or_create_tp_mesh
        from vllm_torchtpu.models.vllm.muse_spark.det_collectives import (
            tp_collective_order,
        )

        self.runner = runner
        vcfg = runner.vllm_config
        self.vllm_config = vcfg
        self.model_dir = vcfg.model_config.model
        self.tp_group = parallel_state.get_tp_group()
        self.world = int(self.tp_group.world_size)
        if self.world != 8:
            raise NotImplementedError(
                "Muse Spark decode megakernel requires tensor_parallel_size=8 "
                f"(got {self.world}). Set VLLM_TPU_MUSE_MEGAKERNEL=0 otherwise."
            )
        self.rank = int(self.tp_group.rank_in_group)
        self.mesh = get_or_create_tp_mesh(TP_AXIS_NAME, self.tp_group)
        order = tp_collective_order()
        self.dev_table = [order.index(r) for r in range(self.world)]
        self.block_size = int(vcfg.cache_config.block_size)
        self.cfg = MuseMegakernelConfig.from_checkpoint(self.model_dir)
        self.max_model_len = int(vcfg.model_config.max_model_len)
        self.prefix_cache_enabled = bool(vcfg.cache_config.enable_prefix_caching)
        self.spec = False
        spec_cfg = vcfg.speculative_config
        if spec_cfg is not None and spec_cfg.method == "dspark":
            self.spec = True
            self.spec_k = int(spec_cfg.num_speculative_tokens)
            self.draft_dir = str(spec_cfg.model)
            self.spec_fixed_q = dict(SPEC_VERIFY_WINDOW_BY_BATCH)
            self.spec_steps = 0
            self.spec_req_steps = 0
            self.spec_accepted = 0
        self.timer = _StepTimer() if logger.isEnabledFor(logging.DEBUG) else None
        self.n_slots = resolve_max_slots(
            vcfg.scheduler_config.max_num_seqs, spec=self.spec
        )
        self.slots = SlotTable(self.n_slots)
        self.kw: dict[str, Any] = {}
        self._ops: dict[Any, Any] = {}
        self.ready = False

        self.max_kp = max(1, -(-self.max_model_len // PAGE_TOKENS))
        self.base_kpages = DEFAULT_BASE_KPAGES
        self.total_vllm_kv_blocks_per_rank = 0
        self.cache_kpages = 0
        self.num_kpages = self.base_kpages
        self._cache_subpage_base = self.base_kpages * KV_GROUP
        self._cache_subpage_cap = 0

        # Dynamic superpage allocator for base_kpages (1 .. base_kpages - 1; 0 is null
        # page)
        self._free_kpages: list[int] = list(range(self.base_kpages - 1, 0, -1))
        self._kp_reserved = 0
        self.kpt_host = np.zeros((self.n_slots, self.max_kp), dtype=np.int32)
        self._kpt_dirty = True

        # Lockstep cross-rank subpage ring state
        self._rank_cached_blocks: list[dict[int, int]] = [{} for _ in range(self.world)]
        self._rank_subpage_to_block: list[dict[int, int]] = [
            {} for _ in range(self.world)
        ]
        self._rank_cache_cursors: list[int] = [0] * self.world
        self._kv_group_id: int = 0
        self._chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        self._chunk_saves: list[list[tuple[int, int]]] = []
        self._hbm_reserve: torch.Tensor | None = None
        self.mk_decode_steps = 0
        self.mk_decoded_tokens = 0
        self.vllm_steps = 0
        self.req_mk_tokens: dict[str, int] = {}
        self.async_prefetch_enabled = bool(envs.VLLM_TPU_MUSE_ASYNC_PREFETCH)
        self._prefetched_step: dict[str, Any] | None = None
        # Chain decode steps on device (token from the previous step's targets).
        self.chain_enabled = True
        self.chained_steps = 0
        self._in_mixed_step: bool = False
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.total_sync_calls = 0
        self.last_step_sync_calls = 0
        self.last_step_n_calls = 0
        self.last_handover_ms = 0.0
        self._eos_token_ids: set[int] = set()
        for cfg_obj in (
            getattr(vcfg.model_config, "hf_config", None),
            getattr(getattr(vcfg.model_config, "hf_config", None), "text_config", None),
        ):
            eos_val = (
                getattr(cfg_obj, "eos_token_id", None) if cfg_obj is not None else None
            )
            if isinstance(eos_val, int):
                self._eos_token_ids.add(eos_val)
            elif isinstance(eos_val, (list, tuple, set)):
                self._eos_token_ids.update(int(x) for x in eos_val if x is not None)

    def _configure_cache_kpages(self, total_vllm_kv_blocks_per_rank: int) -> None:
        """Size cache_kpages to cover 100% of vLLM's KV block pool."""
        self.total_vllm_kv_blocks_per_rank = int(total_vllm_kv_blocks_per_rank)
        if self.prefix_cache_enabled:
            min_kpages = (
                -(-self.total_vllm_kv_blocks_per_rank // KV_GROUP)
                if self.total_vllm_kv_blocks_per_rank > 0
                else 0
            )
            self.cache_kpages = max(128, min_kpages)
        else:
            self.cache_kpages = 0
        self.num_kpages = self.base_kpages + self.cache_kpages
        self._cache_subpage_base = self.base_kpages * KV_GROUP
        self._cache_subpage_cap = self.cache_kpages * KV_GROUP
        assert (
            not self.prefix_cache_enabled
            or self.cache_kpages * KV_GROUP >= self.total_vllm_kv_blocks_per_rank
        ), (
            f"Prefix-cache capacity invariant violated: cache_kpages "
            f"({self.cache_kpages}) * subpages_per_kpage ({KV_GROUP}) < "
            f"total_vllm_kv_blocks_per_rank ({self.total_vllm_kv_blocks_per_rank})"
        )

    def _ensure_slot_kpage(self, slot: int, kp_idx: int) -> int:
        if not hasattr(self, "kpt_host"):
            return 1 + slot * self.max_kp + kp_idx
        kp = int(self.kpt_host[slot, kp_idx])
        if kp <= 0:
            if not self._free_kpages:
                raise RuntimeError(
                    "MuseDecodeFastPath exhausted "
                    f"{self.base_kpages - 1} active global KV superpages"
                )
            kp = self._free_kpages.pop()
            self.kpt_host[slot, kp_idx] = kp
            self._kpt_dirty = True
        return kp

    def _lifetime_kpages(self, runner: Any, req_id: str, seq_len: int) -> int:
        """Superpages a request can touch before it finishes: prompt plus remaining
        output,
        plus one for spec rows.
        """
        req = runner.requests.get(req_id)
        sp = getattr(req, "sampling_params", None)
        max_tokens = getattr(sp, "max_tokens", None)
        done = len(getattr(req, "output_token_ids", ()) or ())
        if max_tokens is None:
            last = self.max_model_len
        else:
            last = (
                seq_len
                + max(0, int(max_tokens) - done)
                + (self.spec_k if self.spec else 0)
            )
        last = min(last, self.max_model_len)
        return min(self.max_kp, last // PAGE_TOKENS + 1)

    def _free_slot_kpages(self, slot: int) -> None:
        if not hasattr(self, "kpt_host"):
            return
        for kp_idx in range(self.max_kp):
            kp = int(self.kpt_host[slot, kp_idx])
            if kp > 0:
                self._free_kpages.append(kp)
                self.kpt_host[slot, kp_idx] = 0
                self._kpt_dirty = True

    def _sync_kpt_device(self) -> None:
        if getattr(self, "_kpt_dirty", False) and hasattr(self, "_dev"):
            self.kpt = _to_torch(self.kpt_host, self._dev)
            self._kpt_dirty = False

    def _slot_subpage(self, slot: int, logical_block_idx: int) -> int:
        kp_idx = logical_block_idx // KV_GROUP
        sub_off = logical_block_idx % KV_GROUP
        kp = self._ensure_slot_kpage(slot, kp_idx)
        return kp * KV_GROUP + sub_off

    def _alloc_cache_subpage_for_rank(self, owner_r: int, block_id: int) -> int:
        """Advance the cross-rank subpage ring cursor in strict lockstep across all TP
        ranks.
        """
        cached = self._rank_cached_blocks[owner_r]
        existing = cached.get(block_id)
        if existing is not None:
            return existing
        if self._cache_subpage_cap <= 0:
            return -1
        cur = self._rank_cache_cursors[owner_r]
        subpage = self._cache_subpage_base + cur
        self._rank_cache_cursors[owner_r] = (cur + 1) % self._cache_subpage_cap
        old_block = self._rank_subpage_to_block[owner_r].get(subpage)
        if old_block is not None:
            cached.pop(old_block, None)
        cached[block_id] = subpage
        self._rank_subpage_to_block[owner_r][subpage] = block_id
        return subpage

    def _can_restore_prefix(self, block_row: Any, num_computed_tokens: int) -> bool:
        if not self.prefix_cache_enabled or num_computed_tokens <= 0:
            return num_computed_tokens == 0
        n_blocks = num_computed_tokens // self.block_size
        for lp in range(n_blocks):
            owner_r = lp % self.world
            blk_id = int(block_row[lp])
            if blk_id not in self._rank_cached_blocks[owner_r]:
                return False
        return True

    def notify_coordinator_cache_blocks(
        self, request: Any, num_computed_tokens: int
    ) -> None:
        """Invoke coordinator.cache_blocks(request, num_computed_tokens) on decode
        steps.
        """
        scheduler = getattr(self.runner, "scheduler", None)
        kv_mgr = getattr(scheduler, "kv_cache_manager", None) or getattr(
            self.runner, "kv_cache_manager", None
        )
        coordinator = getattr(kv_mgr, "coordinator", None)
        if coordinator is not None and hasattr(coordinator, "cache_blocks"):
            coordinator.cache_blocks(request, num_computed_tokens)

    def _get_shared_moe_op(self, is_packed: bool) -> Any:
        key = ("shared_moe", is_packed)
        op = self._ops.get(key)
        if op is not None:
            return op
        from vllm_torchtpu import envs
        from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
        from vllm_torchtpu.kernels.muse_megakernel.fastpath_ops import (
            tp8_shared_moe_local,
        )

        c = self.cfg
        batch_inv = bool(envs.VLLM_TPU_MUSE_BATCH_INVARIANT)
        s2_opt = batch_inv
        p = jax.sharding.PartitionSpec

        if is_packed:

            def body_p(
                h: jax.Array,
                z_pre: jax.Array,
                rw: jax.Array,
                rb: jax.Array,
                pn: jax.Array,
                pen: jax.Array,
                gu_w: jax.Array,
                dw_w: jax.Array,
                gu_g: jax.Array,
                dw_g: jax.Array,
                li: jax.Array,
                rk: jax.Array,
            ) -> jax.Array:
                return jax.shard_map(
                    functools.partial(
                        tp8_shared_moe_local,
                        is_packed=True,
                        topk=c.top_k,
                        output_multiplier=c.output_multiplier,
                        routed_scaling_factor=1.0,
                        rms_norm_eps=c.eps,
                        post_norm_eps=c.post_eps,
                        router_eps=c.router_eps,
                        batch_invariant=batch_inv,
                        s2_opt=s2_opt,
                    ),
                    mesh=self.mesh,
                    in_specs=(p(TP_AXIS_NAME),) * 12,
                    out_specs=p(TP_AXIS_NAME),
                    check_vma=False,
                )(h, z_pre, rw, rb, pn, pen, gu_w, dw_w, gu_g, dw_g, li, rk)

            op = sharded_jax_op(
                "muse_mk::shared_moe_packed",
                body_p,
                mesh=self.mesh,
                input_partition_specs=[p(TP_AXIS_NAME)] * 12,
                output_partition_specs=p(TP_AXIS_NAME),
            )
            op.register_fake(
                lambda h, z, *_rest: torch.empty(
                    (1, h.shape[1], z.shape[2]), dtype=torch.float32, device=h.device
                )
            )
        else:

            def _dense_local(h, z_pre, rw, rb, pn, pen, gu_w, dw_w, li, rk):
                return tp8_shared_moe_local(
                    h,
                    z_pre,
                    rw,
                    rb,
                    pn,
                    pen,
                    gu_w,
                    dw_w,
                    None,
                    None,
                    li,
                    rk,
                    is_packed=False,
                    topk=c.top_k,
                    output_multiplier=c.output_multiplier,
                    routed_scaling_factor=1.0,
                    rms_norm_eps=c.eps,
                    post_norm_eps=c.post_eps,
                    router_eps=c.router_eps,
                    batch_invariant=batch_inv,
                    s2_opt=s2_opt,
                )

            def body_d(
                h: jax.Array,
                z_pre: jax.Array,
                rw: jax.Array,
                rb: jax.Array,
                pn: jax.Array,
                pen: jax.Array,
                gu_w: jax.Array,
                dw_w: jax.Array,
                li: jax.Array,
                rk: jax.Array,
            ) -> jax.Array:
                return jax.shard_map(
                    _dense_local,
                    mesh=self.mesh,
                    in_specs=(p(TP_AXIS_NAME),) * 10,
                    out_specs=p(TP_AXIS_NAME),
                    check_vma=False,
                )(h, z_pre, rw, rb, pn, pen, gu_w, dw_w, li, rk)

            op = sharded_jax_op(
                "muse_mk::shared_moe_dense",
                body_d,
                mesh=self.mesh,
                input_partition_specs=[p(TP_AXIS_NAME)] * 10,
                output_partition_specs=p(TP_AXIS_NAME),
            )
            op.register_fake(
                lambda h, z, *_rest: torch.empty(
                    (1, h.shape[1], z.shape[2]), dtype=torch.float32, device=h.device
                )
            )
        self._ops[key] = op
        return op

    def build(self, model: Any, drafter_model: Any | None = None) -> None:
        """Free duplicate vLLM expert weights, load this rank's TP8 megakernel shard,
        and
        install shared prefill MoE.
        """
        del drafter_model
        from torch_tpu._internal import pallas

        t0 = time.perf_counter()
        prepare_model_for_megakernel(model)

        c = self.cfg
        tp = self.world
        r = self.rank
        dev = torch.device("tpu")
        self._dev = dev
        self.rank_t = _to_torch(np.asarray([[r]], dtype=np.int32), dev)
        self.rank_1d_t = _to_torch(np.asarray([r], dtype=np.int32), dev)
        self.dev_t = _to_torch(np.asarray(self.dev_table, dtype=np.int32), dev)

        ckpt = Checkpoint(self.model_dir)
        qh = c.heads // tp
        kvh = c.kv_heads // tp
        mh_r = c.moe_hidden // tp
        width = c.moe_intermediate // tp
        eshard = c.experts // tp
        h_r = c.dim // tp
        order = router_order_indices(c.experts)

        n_packed = max(1, c.layers - len(c.bf16_expert_layers))
        n_dense = max(1, len(c.bf16_expert_layers))

        def _put_layer(buf: jax.Array, val: jax.Array, idx: jax.Array) -> jax.Array:
            return jax.lax.dynamic_update_index_in_dim(buf, val, idx[0], axis=1)

        def _put_gu_u8(buf: jax.Array, val: jax.Array, idx: jax.Array) -> jax.Array:
            v_len = c.moe_hidden // 2
            gu_v = repack_nvfp4_k_sublane_jax(
                permute_gu_cols_sublane_jax(val[..., :v_len, :])
            )
            gu_s = permute_gu_cols_sublane_jax(val[..., v_len:, :])
            rep = jnp.concatenate([gu_v, gu_s], axis=-2)
            return jax.lax.dynamic_update_index_in_dim(buf, rep, idx[0], axis=1)

        def _put_dw_u8(buf: jax.Array, val: jax.Array, idx: jax.Array) -> jax.Array:
            v_len = width // 2
            dw_v = repack_nvfp4_k_sublane_jax(val[..., :v_len, :])
            dw_s = val[..., v_len:, :]
            rep = jnp.concatenate([dw_v, dw_s], axis=-2)
            return jax.lax.dynamic_update_index_in_dim(buf, rep, idx[0], axis=1)

        def _put_gu_bf16(buf: jax.Array, val: jax.Array, idx: jax.Array) -> jax.Array:
            rep = permute_k_sublane_jax(permute_gu_cols_sublane_jax(val))
            return jax.lax.dynamic_update_index_in_dim(buf, rep, idx[0], axis=1)

        def _put_dw_bf16(buf: jax.Array, val: jax.Array, idx: jax.Array) -> jax.Array:
            rep = permute_k_sublane_jax(val)
            return jax.lax.dynamic_update_index_in_dim(buf, rep, idx[0], axis=1)

        put_gu_u8 = pallas.jax_op("muse_mk::put_gu_u8", _put_gu_u8, donate_argnums=(0,))
        put_dw_u8 = pallas.jax_op("muse_mk::put_dw_u8", _put_dw_u8, donate_argnums=(0,))
        put_gu_bf16 = pallas.jax_op(
            "muse_mk::put_gu_bf16", _put_gu_bf16, donate_argnums=(0,)
        )
        put_dw_bf16 = pallas.jax_op(
            "muse_mk::put_dw_bf16", _put_dw_bf16, donate_argnums=(0,)
        )
        put_bf16 = pallas.jax_op(
            "muse_mk::put_layer_bf16", _put_layer, donate_argnums=(0,)
        )

        gu_pack = torch.zeros(
            (1, n_packed, c.experts, c.moe_hidden * 9 // 16, 2 * width),
            dtype=torch.uint8,
            device=dev,
        )
        dw_pack = torch.zeros(
            (1, n_packed, c.experts, width * 9 // 16, c.moe_hidden),
            dtype=torch.uint8,
            device=dev,
        )
        gu_dense = torch.zeros(
            (1, n_dense, c.experts, c.moe_hidden, 2 * width),
            dtype=torch.bfloat16,
            device=dev,
        )
        dw_dense = torch.zeros(
            (1, n_dense, c.experts, width, c.moe_hidden),
            dtype=torch.bfloat16,
            device=dev,
        )
        qwkv_t = torch.zeros(
            (1, c.layers, c.dim, (2 * qh + 2 * kvh) * c.head_dim),
            dtype=torch.bfloat16,
            device=dev,
        )
        ow_t = torch.zeros((1, c.layers, c.dim, h_r), dtype=torch.bfloat16, device=dev)
        pool_t = torch.zeros(
            (1, c.layers, c.dim, mh_r), dtype=torch.bfloat16, device=dev
        )
        unpool_t = torch.zeros(
            (1, c.layers, c.moe_hidden, h_r), dtype=torch.bfloat16, device=dev
        )
        from vllm_torchtpu.utils import synchronize_tensors

        synchronize_tensors(
            [gu_pack, dw_pack, gu_dense, dw_dense, qwkv_t, ow_t, pool_t, unpool_t]
        )

        scalars_l, router_l, rbias_l, xnorms_l, gu_glob_l, dw_glob_l = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        def read_rank_layer(i: int):
            p = f"model.language_model.layers.{i}."
            s = p + "self_attn."
            m = p + "mlp."
            q_w = ckpt.memmap(s + "q_proj.weight")
            k_w = ckpt.memmap(s + "k_proj.weight")
            v_w = ckpt.memmap(s + "v_proj.weight")
            g_w = ckpt.memmap(s + "gate_proj.weight")
            o_w = ckpt.memmap(s + "o_proj.weight")
            qs = r * qh * c.head_dim
            ks = r * kvh * c.head_dim
            qwkv_r = np.concatenate(
                (
                    np.asarray(q_w[qs : qs + qh * c.head_dim]).T,
                    np.asarray(g_w[qs : qs + qh * c.head_dim]).T,
                    np.asarray(k_w[ks : ks + kvh * c.head_dim]).T,
                    np.asarray(v_w[ks : ks + kvh * c.head_dim]).T,
                ),
                axis=1,
            )
            ow_r = np.asarray(o_w[r * h_r : (r + 1) * h_r, :]).T

            norm_in = (
                np.asarray(ckpt.get(p + "input_layernorm.weight"), np.float32) + 1.0
            )
            alpha_a, beta_a = residual_coefficients(
                ckpt.get(p + "post_attention_residual_gate.gate"), c.gate_temperature
            )
            norm_pre = (
                np.asarray(ckpt.get(p + "pre_feedforward_layernorm.weight"), np.float32)
                + 1.0
            )
            norm_post = (
                np.asarray(
                    ckpt.get(p + "post_feedforward_layernorm.weight"), np.float32
                )
                + 1.0
            )
            alpha_f, beta_f = residual_coefficients(
                ckpt.get(p + "post_feedforward_residual_gate.gate"), c.gate_temperature
            )
            row = np.zeros((8, c.dim), np.float32)
            row[0], row[1], row[2] = norm_in, norm_pre, norm_post
            row[3], row[4], row[5], row[6] = alpha_a, beta_a, alpha_f, beta_f

            rt = np.asarray(ckpt.get(m + "gate.weight"), np.float32)
            rb = np.asarray(ckpt.get(m + "gate.e_score_correction_bias"), np.float32)
            rt_ord = rt[order[r * eshard : (r + 1) * eshard], :]
            rb_ord = rb[order]

            pl_w = ckpt.memmap(m + "pre_expert_proj.weight")
            up_w = ckpt.memmap(m + "post_expert_proj.weight")
            p_sub_idx = (
                np.arange(c.moe_hidden, dtype=np.int32)
                .reshape(c.moe_hidden // 16, 16)
                .T.reshape(-1)
            )
            pool_r = np.asarray(pl_w[p_sub_idx[r * mh_r : (r + 1) * mh_r], :]).T
            unpool_r = np.asarray(up_w[r * h_r : (r + 1) * h_r, :]).T

            xn = np.stack(
                (
                    permute_sublane_axis_np(
                        np.asarray(ckpt.get(m + "pre_expert_norm.weight"), np.float32)
                        + 1.0,
                        0,
                    ),
                    np.asarray(
                        ckpt.get(m + "experts.post_expert_norm.weight"), np.float32
                    )
                    + 1.0,
                )
            )

            half = c.moe_intermediate
            if i in c.bf16_expert_layers:
                gu = ckpt.memmap(m + "experts.gate_up_proj")
                dw = ckpt.memmap(m + "experts.down_proj")
                gu_r = np.concatenate(
                    (
                        np.asarray(gu[:, r * width : (r + 1) * width, :]),
                        np.asarray(gu[:, half + r * width : half + (r + 1) * width, :]),
                    ),
                    axis=1,
                ).transpose(0, 2, 1)
                dw_r = np.asarray(dw[:, :, r * width : (r + 1) * width]).transpose(
                    0, 2, 1
                )
                ex: Any = (gu_r, dw_r)
            else:
                gu = ckpt.memmap(m + "experts.gate_up_proj")
                gus = ckpt.memmap(m + "experts.gate_up_proj_weight_scale").view(
                    np.uint8
                )
                dw = ckpt.memmap(m + "experts.down_proj")
                dws = ckpt.memmap(m + "experts.down_proj_weight_scale").view(np.uint8)
                gu_v = np.concatenate(
                    (
                        np.asarray(gu[:, r * width : (r + 1) * width, :]),
                        np.asarray(gu[:, half + r * width : half + (r + 1) * width, :]),
                    ),
                    axis=1,
                ).transpose(0, 2, 1)
                gu_s = np.concatenate(
                    (
                        np.asarray(gus[:, r * width : (r + 1) * width, :]),
                        np.asarray(
                            gus[:, half + r * width : half + (r + 1) * width, :]
                        ),
                    ),
                    axis=1,
                ).transpose(0, 2, 1)
                gu_r = np.concatenate((gu_v, gu_s), axis=1)
                dw_w = width // 2
                dw_sw = width // 16
                dw_v = np.asarray(dw[:, :, r * dw_w : (r + 1) * dw_w]).transpose(
                    0, 2, 1
                )
                dw_s = np.asarray(dws[:, :, r * dw_sw : (r + 1) * dw_sw]).transpose(
                    0, 2, 1
                )
                dw_r = np.concatenate((dw_v, dw_s), axis=1)
                g_glob = np.asarray(
                    ckpt.get(m + "experts.gate_up_proj_weight_scale_2"), np.float32
                )
                d_glob = np.asarray(
                    ckpt.get(m + "experts.down_proj_weight_scale_2"), np.float32
                )
                ex = (gu_r, dw_r, g_glob, d_glob)
            return row, rt_ord, rb_ord, pool_r, unpool_r, xn, qwkv_r, ow_r, ex

        bi = pi = 0
        with ThreadPoolExecutor(2) as pool_ex:
            futs = {0: pool_ex.submit(read_rank_layer, 0)}
            if c.layers > 1:
                futs[1] = pool_ex.submit(read_rank_layer, 1)
            for i in range(c.layers):
                if i + 2 < c.layers:
                    futs[i + 2] = pool_ex.submit(read_rank_layer, i + 2)
                row, rt_o, rb_o, pl_r, up_r, xn, qwkv_r, ow_r, ex = futs.pop(i).result()
                scalars_l.append(row)
                router_l.append(rt_o)
                rbias_l.append(rb_o)
                xnorms_l.append(xn)
                idx_i = _to_torch(np.asarray([i], dtype=np.int32), dev)
                qwkv_t = put_bf16(qwkv_t, _to_torch(qwkv_r[None, None], dev), idx_i)
                ow_t = put_bf16(ow_t, _to_torch(ow_r[None, None], dev), idx_i)
                pool_t = put_bf16(pool_t, _to_torch(pl_r[None, None], dev), idx_i)
                unpool_t = put_bf16(unpool_t, _to_torch(up_r[None, None], dev), idx_i)
                if i in c.bf16_expert_layers:
                    idx_b = _to_torch(np.asarray([bi], dtype=np.int32), dev)
                    gu_dense = put_gu_bf16(
                        gu_dense, _to_torch(ex[0][None, None], dev), idx_b
                    )
                    dw_dense = put_dw_bf16(
                        dw_dense, _to_torch(ex[1][None, None], dev), idx_b
                    )
                    bi += 1
                else:
                    idx_p = _to_torch(np.asarray([pi], dtype=np.int32), dev)
                    gu_pack = put_gu_u8(
                        gu_pack, _to_torch(ex[0][None, None], dev), idx_p
                    )
                    dw_pack = put_dw_u8(
                        dw_pack, _to_torch(ex[1][None, None], dev), idx_p
                    )
                    gu_glob_l.append(ex[2])
                    dw_glob_l.append(ex[3])
                    pi += 1
                del ex
                synchronize_tensors(
                    [
                        qwkv_t,
                        ow_t,
                        pool_t,
                        unpool_t,
                        gu_dense,
                        dw_dense,
                        gu_pack,
                        dw_pack,
                    ]
                )
                torch.accelerator.synchronize()

        if not gu_glob_l:
            gu_glob_l.append(np.zeros((c.experts, 2), np.float32))
            dw_glob_l.append(np.zeros((c.experts,), np.float32))

        vshard = c.vocab // tp
        emb_mm = ckpt.memmap("model.language_model.embed_tokens.weight")
        emb_r = np.asarray(emb_mm[r * vshard : (r + 1) * vshard])[None]
        lm_mm = ckpt.memmap("lm_head.weight")
        lmw_np = np.zeros((1, c.dim, lm_width(c, tp)), ml_dtypes.bfloat16)
        lmw_np[0, :, :vshard] = np.asarray(lm_mm[r * vshard : (r + 1) * vshard]).T
        fnorm_np = np.asarray(
            ckpt.get("model.language_model.norm.weight"), np.float32
        ).reshape(1, 1, c.dim)

        self.kw = {
            "qwkv": qwkv_t,
            "ow_group8": ow_t,
            "pool_local": pool_t,
            "unpool_group8": unpool_t,
            "router_ordered": _to_torch(np.stack(router_l, axis=0)[None], dev),
            "router_bias_ordered": _to_torch(np.stack(rbias_l, axis=0)[None], dev),
            "scalars": _to_torch(np.stack(scalars_l, axis=0)[None], dev),
            "expert_norms": _to_torch(np.stack(xnorms_l, axis=0)[None], dev),
            "gu_pack": gu_pack,
            "dw_pack": dw_pack,
            "gu_dense": gu_dense,
            "dw_dense": dw_dense,
            "gu_global": _to_torch(np.stack(gu_glob_l, axis=0)[None], dev),
            "dw_global": _to_torch(np.stack(dw_glob_l, axis=0)[None], dev),
            "lmw": _to_torch(lmw_np, dev),
            "fnorm": _to_torch(fnorm_np, dev),
            "emb": _to_torch(emb_r, dev),
        }
        self.weight_names = list(KERNEL_WEIGHT_NAMES)
        self._w_list = [self.kw[k] for k in self.weight_names]
        if self.spec:
            from vllm_torchtpu.kernels.muse_megakernel.dspark_step import (
                DSPARK_WEIGHT_NAMES,
                DSparkConfig,
                load_dspark_rank,
            )

            self.dcfg = DSparkConfig(self.draft_dir)
            d_np = load_dspark_rank(self.draft_dir, r, tp)
            self.dkw = {k: _to_torch(d_np[k], dev) for k in DSPARK_WEIGHT_NAMES}
            del d_np
            self._dw_list = [self.dkw[k] for k in DSPARK_WEIGHT_NAMES]
            self.kv_draft = torch.zeros(
                (1, self.dcfg.layers, self.n_slots, SW_RING, 128 * (c.kv_heads // tp)),
                dtype=torch.bfloat16,
                device=dev,
            )
            synchronize_tensors([*self._dw_list, self.kv_draft])

        # Eagerly register shared-store TP8 prefill MoE ops before torch.compile and
        # install on all layers
        self._get_shared_moe_op(True)
        self._get_shared_moe_op(False)
        self._install_shared_moe(model)

        # Pre-allocate kv_sw, kv_gl, kpt and warm up decode step buckets before
        # determine_available_memory
        self._configure_cache_kpages(0)
        kvwidth = 128 * (c.kv_heads // tp)
        self.kv_sw = torch.zeros(
            (1, c.num_sw_layers, self.n_slots, SW_RING, kvwidth),
            dtype=torch.bfloat16,
            device=dev,
        )
        self.kv_gl = torch.zeros(
            (1, c.num_gl_layers, self.num_kpages, PAGE_TOKENS, kvwidth),
            dtype=torch.bfloat16,
            device=dev,
        )
        self._kpt_dirty = True
        self._sync_kpt_device()
        synchronize_tensors([*self._w_list, self.kv_sw, self.kv_gl, self.kpt])
        torch.accelerator.synchronize()
        self._warmup_decode_steps()
        free_b, limit_b = torch.accelerator.get_memory_info(dev)
        used_b = limit_b - free_b
        logger.info(
            "Muse Spark megakernel weights + decode steps loaded in "
            "%.1fs | rank=%d num_kpages=%d HBM used=%.2f/%.2f GiB",
            time.perf_counter() - t0,
            r,
            self.num_kpages,
            used_b / 2**30,
            limit_b / 2**30,
        )

    def _install_shared_moe(self, model: Any) -> None:
        if (
            model is not None
            and hasattr(model, "model")
            and hasattr(model.model, "layers")
        ):
            for li, layer in enumerate(model.model.layers):
                mlp = getattr(layer, "mlp", None)
                if mlp is not None and not isinstance(
                    mlp.__dict__.get("_fused_moe_args", None), MuseSharedStoreMoE
                ):
                    mlp.__dict__["_fused_moe_args"] = MuseSharedStoreMoE(self, li, mlp)

    def _warmup_decode_steps(self) -> None:
        from vllm_torchtpu.utils import synchronize_tensors

        t0 = time.perf_counter()
        for need_lp in (False, True):
            b_list = (1, 2, 4, 8) if need_lp else (1, 2, 4, 8, 16)
            for b in b_list:
                if b > self.n_slots:
                    continue
                op = self._get_step_op(b, need_logprobs=need_lp)
                ctrl_tok_np = np.full((4, b), -1, dtype=np.int32)
                ctrl_tok_np[1] = np.arange(b, dtype=np.int32)
                ctrl_tok_np[2] = 0
                ctrl_tok_b = _to_torch(ctrl_tok_np, self._dev)
                outs = op(
                    *self._w_list,
                    self.kv_sw,
                    self.kv_gl,
                    ctrl_tok_b,
                    self.kpt,
                    self.rank_1d_t,
                    self.dev_t,
                    *(() if need_lp else (self._prev_dummy(b),)),
                )
                self.kv_sw, self.kv_gl = outs[1], outs[2]
                if need_lp:
                    synchronize_tensors(
                        [outs[0], self.kv_sw, self.kv_gl, outs[3], outs[4]]
                    )
                else:
                    synchronize_tensors([outs[0], self.kv_sw, self.kv_gl])
                _ = outs[0].cpu().numpy()
                torch.accelerator.synchronize()
        if self.prefix_cache_enabled:
            for b_cp in (16, 64, 256):
                cp_op = self._get_copy_subpages_op(b_cp)
                pairs = _to_torch(np.full((2, b_cp), -1, dtype=np.int32), self._dev)
                self.kv_gl = cp_op(self.kv_gl, pairs)
                synchronize_tensors([self.kv_gl])
                torch.accelerator.synchronize()
        if self.spec:
            self._warmup_spec_steps()
            torch.accelerator.synchronize()
        logger.info(
            "Muse Spark megakernel decode step warmup done in "
            "%.1fs | tpu_custom_call(B=1)=%d (B=8)=%d (B=16)=%d",
            time.perf_counter() - t0,
            self.count_custom_calls(1),
            self.count_custom_calls(8),
            self.count_custom_calls(min(16, self.n_slots)),
        )

    def spec_shapes(self) -> list[tuple[int, int, int]]:
        """Every (padded batch, verify rows, draft depth) the spec step can dispatch."""
        out = []
        for nb in (1, 2, 4, 8, 16, 32):
            if nb > self.n_slots:
                continue
            kd = min(SPEC_DRAFT_DEPTH.get(nb, 0), self.spec_k)
            if kd <= 0:
                if nb <= SPEC_MAX_ROWS:
                    out.append((nb, nb, 0))
                continue
            for rows in (1, 2, 4, 8, 16, 32):
                if nb <= rows <= min(SPEC_MAX_ROWS, nb * (kd + 1)):
                    out.append((nb, rows, kd))
        return out

    def _warmup_spec_steps(self) -> None:
        """Compile and run every spec bucket on idle rows (pos -1 writes no state)."""
        from vllm_torchtpu.utils import synchronize_tensors

        t0 = time.perf_counter()
        warmed_rows: set[int] = set()
        for nb, rows, kd in self.spec_shapes():
            ctrl = np.full((8, SPEC_MAX_ROWS), -1, np.int32)
            ctrl[1, :] = 0
            ctrl[2, :] = 0
            ctrl[3, :] = 0
            ctrl[6, :] = 0
            ctrl[7, :] = 0
            ctrl_t = _to_torch(ctrl, self._dev, sync=False)
            op = self._get_spec_step_op(nb, rows, kd, False)
            outs = op(
                *self._w_list,
                *self._dw_list,
                self.kv_sw,
                self.kv_gl,
                self.kv_draft,
                ctrl_t,
                self.kpt,
                self.rank_1d_t,
                self.dev_t,
            )
            self.kv_sw, self.kv_gl, self.kv_draft = outs[1], outs[2], outs[3]
            sw_o, dr_o = outs[-2], outs[-1]
            synchronize_tensors(
                [outs[0], self.kv_sw, self.kv_gl, self.kv_draft, sw_o, dr_o]
            )
            _ = outs[0].cpu().numpy()
            if rows not in warmed_rows:
                warmed_rows.add(rows)
                rb_op = self._get_spec_rollback_op(rows)
                self.kv_sw, self.kv_draft = rb_op(
                    self.kv_sw, self.kv_draft, sw_o, dr_o, ctrl_t
                )
                synchronize_tensors([self.kv_sw, self.kv_draft])
        torch.accelerator.synchronize()
        logger.info(
            "Muse Spark DSpark spec step warmup done in %.1fs | shapes=%s",
            time.perf_counter() - t0,
            self.spec_shapes(),
        )

    def bind_runner_state(self, runner: Any) -> None:
        """Bind vLLM's allocated KV cache pools after initialize_kv_cache and size
        cache_kpages.
        """
        from vllm_torchtpu.utils import synchronize_tensors

        self.runner = runner
        self.prefix_cache_enabled = bool(
            getattr(
                getattr(runner, "cache_config", None),
                "enable_prefix_caching",
                self.prefix_cache_enabled,
            )
        )

        layer_idx_to_gid: dict[int, int] = {}
        name_to_gid: dict[str, int] = {}
        drafter = getattr(runner, "drafter", None)
        draft_model = getattr(drafter, "draft_model", None)
        draft_names = (
            {dl.self_attn.attn.layer_name for dl in draft_model.model.layers}
            if draft_model is not None and hasattr(draft_model, "model")
            else set()
        )
        for group_list in getattr(runner, "attn_groups", []):
            for group in group_list:
                gid = int(group.kv_cache_group_id)
                for name in group.layer_names:
                    name_to_gid[name] = gid
                    m = re.search(r"layers\.(\d+)\.", name)
                    if m and name not in draft_names:
                        layer_idx_to_gid[int(m.group(1))] = gid

        self._kv_pools: list[torch.Tensor] = []
        ptrs: dict[tuple[int, int], int] = {}
        self._sw_pools: list[torch.Tensor] = []
        self._gl_pools: list[torch.Tensor] = []
        self._sw_pool_of: list[int] = []
        self._sw_gids: list[int] = []
        self._gl_pool_of: list[int] = []
        self._gl_gids: list[int] = []
        total_blocks = 0
        if runner.model is not None and hasattr(runner.model, "model"):
            for li, layer in enumerate(runner.model.model.layers):
                attn_mod = getattr(layer.self_attn, "attn", layer.self_attn)
                kv = getattr(attn_mod, "kv_cache", None)
                if kv is None:
                    kv = getattr(layer.self_attn, "kv_cache", None)
                if kv is not None and len(kv) >= 1:
                    t = kv[0] if isinstance(kv, (list, tuple)) else kv
                    total_blocks = max(total_blocks, int(t.shape[0]))
                    key_ptr = (
                        int(t.untyped_storage().data_ptr()),
                        int(t.storage_offset()),
                    )
                    p = ptrs.get(key_ptr)
                    if p is None:
                        p = len(self._kv_pools)
                        ptrs[key_ptr] = p
                        self._kv_pools.append(t)
                    gid = layer_idx_to_gid.get(li, 0)
                    if (li % 4) != 1:
                        self._sw_pools.append(t)
                        self._sw_pool_of.append(p)
                        self._sw_gids.append(gid)
                    else:
                        self._gl_pools.append(t)
                        self._gl_pool_of.append(p)
                        self._gl_gids.append(gid)

        self._pool_layers: list[list[tuple[int, int]]] = [
            [] for _ in range(len(self._kv_pools))
        ]
        self._draft_pools: list[torch.Tensor] = []
        self._draft_gids: list[int] = []
        if self.spec and draft_model is not None:
            for dl in draft_model.model.layers:
                attn_mod = dl.self_attn.attn
                kv = attn_mod.kv_cache
                t = kv[0] if isinstance(kv, (list, tuple)) else kv
                self._draft_pools.append(t)
                # Draft layer names reuse small layer indices, so look them up by full
                # name.
                self._draft_gids.append(name_to_gid[attn_mod.layer_name])
        self._unique_draft_gids = sorted(set(self._draft_gids)) or [0]
        self._draft_gid_idx_of = tuple(
            self._unique_draft_gids.index(g) for g in self._draft_gids
        )
        for gl_idx, p_idx in enumerate(self._gl_pool_of):
            self._pool_layers[p_idx].append((0, gl_idx))
        for sw_idx, p_idx in enumerate(self._sw_pool_of):
            self._pool_layers[p_idx].append((1, sw_idx))

        self._unique_sw_gids = sorted(set(self._sw_gids)) if self._sw_gids else [0]
        self._sw_gid_idx_of = tuple(
            self._unique_sw_gids.index(g) for g in self._sw_gids
        )
        self._gl_group_id = self._gl_gids[0] if self._gl_gids else 0
        self._kv_group_id = self._gl_group_id
        if self.rank == 0:
            logger.info(
                "Muse Spark megakernel KV groups | sw=%s "
                "gl=%s draft=%s names=%s pools sw=%s draft=%s",
                self._sw_gids,
                self._gl_gids,
                self._draft_gids,
                sorted(draft_names),
                tuple(self._sw_pools[0].shape) if self._sw_pools else None,
                tuple(self._draft_pools[0].shape) if self._draft_pools else None,
            )
        if len(set(self._gl_gids)) > 1:
            raise RuntimeError(
                "megakernel expects every global layer in "
                f"one KV cache group, got {self._gl_gids}"
            )

        per_rank_blocks = (
            max(1, -(-total_blocks // self.world)) if total_blocks > 0 else 0
        )
        self._configure_cache_kpages(per_rank_blocks)
        dev = getattr(self, "_dev", torch.device("tpu"))
        self._dev = dev
        kvwidth = 128 * (self.cfg.kv_heads // self.world)
        if not hasattr(self, "kv_sw") or not isinstance(self.kv_sw, torch.Tensor):
            self.kv_sw = torch.zeros(
                (1, self.cfg.num_sw_layers, self.n_slots, SW_RING, kvwidth),
                dtype=torch.bfloat16,
                device=dev,
            )
            synchronize_tensors([self.kv_sw])
        if (
            not hasattr(self, "kv_gl")
            or not isinstance(self.kv_gl, torch.Tensor)
            or int(self.kv_gl.shape[2]) != self.num_kpages
        ):
            self.kv_gl = torch.zeros(
                (1, self.cfg.num_gl_layers, self.num_kpages, PAGE_TOKENS, kvwidth),
                dtype=torch.bfloat16,
                device=dev,
            )
            synchronize_tensors([self.kv_gl])
            for b in (1, 2, 4, 8, 16, 32):
                self._ops.pop(("hybrid_step_op", b, False), None)
                self._ops.pop(("hybrid_step_op", b, True), None)
            for b_cp in (16, 64, 256):
                self._ops.pop(("copy_subpages", b_cp), None)
            for b_sync in (32, 128, 256):
                self._ops.pop(("sync_seq_kv_all_layers", b_sync), None)
            for k in [
                k
                for k in self._ops
                if isinstance(k, tuple)
                and k[0] in ("spec_step", "spec_pre", "spec_rollback")
            ]:
                self._ops.pop(k)
            self._warmup_decode_steps()
        self._kpt_dirty = True
        self._sync_kpt_device()
        self.warmup()
        self.ready = True
        free_b, limit_b = torch.accelerator.get_memory_info(dev)
        logger.info(
            "Muse Spark megakernel decode ready | slots=%d max_kp=%d base_kpages=%d "
            "cache_kpages=%d (covers %d >= %d vLLM "
            "blocks/rank) | HBM used=%.2f/%.2f GiB",
            self.n_slots,
            self.max_kp,
            self.base_kpages,
            self.cache_kpages,
            self.cache_kpages * KV_GROUP,
            self.total_vllm_kv_blocks_per_rank,
            (limit_b - free_b) / 2**30,
            limit_b / 2**30,
        )

    def warmup(self) -> None:
        """Pre-compile KV handoff and writeback ops so serving never compiles."""
        from vllm_torchtpu.utils import synchronize_tensors

        t0 = time.perf_counter()
        if self._kv_pools:
            synchronize_tensors(self._kv_pools)
            n_g = len(self._unique_sw_gids)
            for b_sync in (32, 128, 256):
                sync_op = self._get_sync_seq_kv_op(b_sync)
                sw_quads = _to_torch(
                    np.full((2 * n_g + 3, b_sync), -1, dtype=np.int32),
                    self._dev,
                )
                gl_pairs = _to_torch(
                    np.full((2, b_sync), -1, dtype=np.int32),
                    self._dev,
                )
                self.kv_sw, self.kv_gl = sync_op(
                    self.kv_sw,
                    self.kv_gl,
                    *self._kv_pools,
                    sw_quads,
                    gl_pairs,
                )
                synchronize_tensors([self.kv_sw, self.kv_gl])
                if self.spec and self._draft_pools:
                    n_dg = len(self._unique_draft_gids)
                    d_quads = _to_torch(
                        np.full((2 * n_dg + 3, b_sync), -1, dtype=np.int32), self._dev
                    )
                    self.kv_draft = self._get_sync_draft_op(b_sync)(
                        self.kv_draft, *self._draft_pools, d_quads
                    )
                    synchronize_tensors([self.kv_draft])
            for b_wb in (1, 16):
                gather_op = self._get_gather_kv_op(b_wb)
                scatter_op = self._get_scatter_pool_op(b_wb)
                gl_loc = _to_torch(
                    np.zeros((self.cfg.num_gl_layers, b_wb, 2), dtype=np.int32),
                    self._dev,
                )
                sw_loc = _to_torch(
                    np.zeros((self.cfg.num_sw_layers, b_wb, 2), dtype=np.int32),
                    self._dev,
                )
                gl_vals, sw_vals = gather_op(self.kv_sw, self.kv_gl, gl_loc, sw_loc)
                synchronize_tensors([gl_vals, sw_vals])
                meta_t = _to_torch(np.full((4, b_wb, 4), -1, dtype=np.int32), self._dev)
                nt = scatter_op(self._kv_pools[0], gl_vals, sw_vals, meta_t)
                self._kv_pools[0].copy_(nt)
                synchronize_tensors([self._kv_pools[0]])
                del nt
            torch.accelerator.synchronize()
        logger.info(
            "Muse Spark megakernel KV warmup done in %.1fs",
            time.perf_counter() - t0,
        )

    def count_custom_calls(self, batch: int = 1) -> int:
        """Return lowered HLO tpu_custom_call count for the hybrid decode step."""
        from vllm_torchtpu.kernels.muse_megakernel.kernel import make_hybrid_decode_step

        step: Any = make_hybrid_decode_step(
            self.mesh,
            self.cfg,
            max_slots=self.n_slots,
            num_kpages=self.num_kpages,
            max_kp=self.max_kp,
            batch_size=batch,
            force_rows8=True,
            explicit_rank_dev=False,
            return_logits=False,
        )
        tp = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec(TP_AXIS_NAME)
        )
        rep = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
        _dt = {
            torch.bfloat16: jnp.bfloat16,
            torch.float32: jnp.float32,
            torch.int32: jnp.int32,
            torch.uint8: jnp.uint8,
        }

        def shape_of(v: Any, s: Any) -> jax.ShapeDtypeStruct:
            if isinstance(v, torch.Tensor):
                sh = (self.world, *v.shape[1:]) if s == tp else tuple(v.shape)
                return jax.ShapeDtypeStruct(sh, _dt[v.dtype], sharding=s)
            return jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=s)

        w_shapes = {k: shape_of(v, tp) for k, v in self.kw.items()}
        sw_s = shape_of(self.kv_sw, tp)
        gl_s = shape_of(self.kv_gl, tp)
        tok_s = jax.ShapeDtypeStruct((batch,), jnp.int32, sharding=rep)
        ctrl_s = jax.ShapeDtypeStruct((2, batch), jnp.int32, sharding=rep)
        kpt_s = jax.ShapeDtypeStruct(
            (self.n_slots, self.max_kp), jnp.int32, sharding=rep
        )
        hlo = step.lower(w_shapes, sw_s, gl_s, tok_s, ctrl_s, kpt_s).as_text()
        return hlo.count("tpu_custom_call")

    def _megakernel_eligible_request(self, runner: Any, req_id: str) -> bool:
        if req_id.startswith("__") or req_id not in runner.input_batch.greedy_reqs:
            return False
        req = runner.requests.get(req_id)
        sp = getattr(req, "sampling_params", None)
        if sp is None:
            return False
        if (
            sp.max_tokens is not None
            and hasattr(req, "output_token_ids")
            and len(req.output_token_ids) >= sp.max_tokens
        ):
            return False
        lp_limit = 0 if self.spec else LOGPROB_K
        return not (
            getattr(sp, "logits_processors", None)
            or (sp.logprobs or 0) > lp_limit
            or sp.bad_words
            or sp.allowed_token_ids
            or sp.presence_penalty
            or sp.frequency_penalty
            or (sp.repetition_penalty not in (None, 1.0))
            or sp.logit_bias
            or (sp.min_tokens and not sp.ignore_eos)
            or getattr(sp, "structured_outputs", None)
        )

    def _flush_prefetched_step(self) -> None:
        pref = self._prefetched_step
        self._prefetched_step = None
        if pref is not None:
            if pref.get("is_spec", False):
                from vllm_torchtpu.utils import synchronize_tensors

                outs, sw_old, draft_old, ctrl_t = pref["spec_state"]
                _ = outs[0].cpu()
                rb_op = self._get_spec_rollback_op(pref["rows"])
                self.kv_sw, self.kv_draft = rb_op(
                    self.kv_sw, self.kv_draft, sw_old, draft_old, ctrl_t
                )
                synchronize_tensors([self.kv_sw, self.kv_draft])
            else:
                for outs, _ in pref["chunks"]:
                    _ = outs[0].cpu()
            self.prefetch_misses += 1

    def release(self, req_ids: Any) -> None:
        for r in req_ids:
            self.req_mk_tokens.pop(r, None)
            st = self.slots.release(r)
            if st is not None:
                self._kp_reserved -= st.kp_need
                st.kp_need = 0
                if (
                    self._prefetched_step is not None
                    and st.slot in self._prefetched_step["slot_map"]
                ):
                    self._flush_prefetched_step()
                self._free_slot_kpages(st.slot)

    def begin_step(self, num_chunks: int = 0) -> None:
        self._chunks = []
        self._chunk_saves = []
        self._in_mixed_step = num_chunks > 0
        if num_chunks > 0:
            self.vllm_steps += 1

    def _get_copy_subpages_op(self, n_pages: int) -> Any:
        key = ("copy_subpages", n_pages)
        op = self._ops.get(key)
        if op is not None:
            return op
        from torch_tpu._internal import pallas

        from vllm_torchtpu.kernels.muse_megakernel.fastpath_ops import (
            copy_subpages_local,
        )

        def body(kv_gl: jax.Array, pairs: jax.Array) -> jax.Array:
            return copy_subpages_local(kv_gl, pairs)

        op = pallas.jax_op(
            f"muse_mk::copy_subpages_{n_pages}",
            _explicit_arity(body, 2, 1),
            donate_argnums=(0,),
        )
        self._ops[key] = op
        return op

    def copy_subpages(self, pairs: list[tuple[int, int]]) -> None:
        if not pairs or not self.prefix_cache_enabled or not self.ready:
            return
        from vllm_torchtpu.utils import synchronize_tensors

        dedup = list(
            {dst: (src, dst) for src, dst in pairs if src >= 0 and dst >= 0}.values()
        )
        if not dedup:
            return
        buckets = (16, 64, 256)
        for off in range(0, len(dedup), buckets[-1]):
            chunk = dedup[off : off + buckets[-1]]
            b = next(x for x in buckets if x >= len(chunk))
            arr = np.full((2, b), -1, np.int32)
            for j, (s, d) in enumerate(chunk):
                arr[0, j], arr[1, j] = s, d
            op = self._get_copy_subpages_op(b)
            self.kv_gl = op(self.kv_gl, _to_torch(arr, self._dev))
            synchronize_tensors([self.kv_gl], wait=False)

    def _get_sync_seq_kv_op(self, bucket: int) -> Any:
        key = ("sync_seq_kv_all_layers", bucket)
        op = self._ops.get(key)
        if op is not None:
            return op
        from torch_tpu._internal import pallas

        from vllm_torchtpu.kernels.muse_megakernel.fastpath_ops import (
            sync_seq_kv_all_layers_local,
        )

        n_pools = len(self._kv_pools)
        sw_pool_of = tuple(self._sw_pool_of)
        sw_gid_idx_of = tuple(self._sw_gid_idx_of)
        gl_pool_of = tuple(self._gl_pool_of)
        n_sw_groups = len(self._unique_sw_gids)

        def body(*a: jax.Array) -> tuple[jax.Array, jax.Array]:
            kv_sw, kv_gl = a[0], a[1]
            pools = list(a[2 : 2 + n_pools])
            sw_quads, gl_pairs = a[2 + n_pools], a[3 + n_pools]
            return sync_seq_kv_all_layers_local(
                kv_sw,
                kv_gl,
                pools,
                sw_quads,
                gl_pairs,
                sw_pool_of=sw_pool_of,
                sw_gid_idx_of=sw_gid_idx_of,
                gl_pool_of=gl_pool_of,
                n_sw_groups=n_sw_groups,
            )

        op = pallas.jax_op(
            f"muse_mk::sync_seq_kv_all_layers_{bucket}",
            _explicit_arity(body, 2 + n_pools + 2, 2),
            donate_argnums=(0, 1),
        )
        self._ops[key] = op
        return op

    def _get_sync_draft_op(self, bucket: int) -> Any:
        key = ("sync_draft_ring", bucket)
        op = self._ops.get(key)
        if op is not None:
            return op
        from torch_tpu._internal import pallas

        from vllm_torchtpu.kernels.muse_megakernel.fastpath_ops import (
            sync_sw_rings_local,
        )

        n_pools = len(self._draft_pools)
        n_groups = len(self._unique_draft_gids)
        gid_idx_of = self._draft_gid_idx_of

        def body(*a: jax.Array) -> jax.Array:
            kv_draft = a[0]
            pools = list(a[1 : 1 + n_pools])
            return sync_sw_rings_local(
                kv_draft,
                pools,
                a[1 + n_pools],
                sw_pool_of=tuple(range(n_pools)),
                sw_gid_idx_of=gid_idx_of,
                n_sw_groups=n_groups,
            )

        op = pallas.jax_op(
            f"muse_mk::sync_draft_ring_{bucket}",
            _explicit_arity(body, 1 + n_pools + 1, 1),
            donate_argnums=(0,),
        )
        self._ops[key] = op
        return op

    def _get_handoff_sw_op(self) -> Any:
        key = "handoff_sw"
        op = self._ops.get(key)
        if op is not None:
            return op
        from torch_tpu._internal import pallas

        from vllm_torchtpu.kernels.muse_megakernel.fastpath_ops import (
            handoff_sw_ring_local,
        )

        n_pools = len(self._kv_pools)
        sw_pool_of = tuple(self._sw_pool_of)
        sw_gid_idx_of = tuple(self._sw_gid_idx_of)

        def body(*a: jax.Array) -> jax.Array:
            kv_sw = a[0]
            pools = list(a[1 : 1 + n_pools])
            blk_ids, blk_offs, slot_arr = a[1 + n_pools :]
            return handoff_sw_ring_local(
                kv_sw,
                pools,
                blk_ids,
                blk_offs,
                slot_arr,
                sw_pool_of=sw_pool_of,
                sw_gid_idx_of=sw_gid_idx_of,
            )

        op = pallas.jax_op(
            "muse_mk::handoff_sw_ring",
            _explicit_arity(body, 1 + n_pools + 3, 1),
            donate_argnums=(0,),
        )
        self._ops[key] = op
        return op

    def _get_handoff_gl_op(self, bucket: int) -> Any:
        key = ("handoff_gl", bucket)
        op = self._ops.get(key)
        if op is not None:
            return op
        from torch_tpu._internal import pallas

        from vllm_torchtpu.kernels.muse_megakernel.fastpath_ops import (
            handoff_gl_subpages_local,
        )

        n_pools = len(self._kv_pools)
        gl_pool_of = tuple(self._gl_pool_of)

        def body(*a: jax.Array) -> jax.Array:
            kv_gl = a[0]
            pools = list(a[1 : 1 + n_pools])
            pairs = a[1 + n_pools]
            return handoff_gl_subpages_local(kv_gl, pools, pairs, gl_pool_of=gl_pool_of)

        op = pallas.jax_op(
            f"muse_mk::handoff_gl_{bucket}",
            _explicit_arity(body, 1 + n_pools + 1, 1),
            donate_argnums=(0,),
        )
        self._ops[key] = op
        return op

    def _get_gather_kv_op(self, batch_b: int) -> Any:
        key = ("gather_kv", batch_b)
        op = self._ops.get(key)
        if op is not None:
            return op
        from torch_tpu._internal import pallas

        from vllm_torchtpu.kernels.muse_megakernel.fastpath_ops import (
            gather_step_kv_local,
        )

        def body(
            kv_sw: jax.Array, kv_gl: jax.Array, gl_loc: jax.Array, sw_loc: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            return gather_step_kv_local(kv_sw, kv_gl, gl_loc, sw_loc)

        op = pallas.jax_op(
            f"muse_mk::gather_kv_b{batch_b}",
            _explicit_arity(body, 4, 2),
        )
        self._ops[key] = op
        return op

    def _get_scatter_pool_op(self, batch_b: int) -> Any:
        key = ("scatter_pool_kv", batch_b)
        op = self._ops.get(key)
        if op is not None:
            return op
        from torch_tpu._internal import pallas

        from vllm_torchtpu.kernels.muse_megakernel.fastpath_ops import (
            scatter_pool_kv_local,
        )

        def body(
            pool: jax.Array, gl_vals: jax.Array, sw_vals: jax.Array, meta: jax.Array
        ) -> jax.Array:
            return scatter_pool_kv_local(pool, gl_vals, sw_vals, meta)

        op = pallas.jax_op(
            f"muse_mk::scatter_pool_kv_b{batch_b}",
            _explicit_arity(body, 4, 1),
            donate_argnums=(0,),
        )
        self._ops[key] = op
        return op

    def _sync_seq_kv_from_vllm(
        self, runner: Any, todo: list[tuple[str, SlotState, int, int]]
    ) -> float:
        """Vectorized single-op prefill-to-decode KV bridge across all 46 SW and 16 GL
        layers.
        """
        from vllm_torchtpu.utils import synchronize_tensors

        t0 = time.perf_counter()
        bt_gl = runner.input_batch.block_table[self._gl_group_id].get_cpu_tensor()
        bt_sw_by_g = [
            runner.input_batch.block_table[gid].get_cpu_tensor()
            for gid in self._unique_sw_gids
        ]
        n_g = len(self._unique_sw_gids)
        sw_subpages_per_slot = SW_RING // self.block_size
        sw_cols: list[list[int]] = []
        gl_pairs: list[tuple[int, int]] = []
        draft_bts = (
            [
                runner.input_batch.block_table[gid].get_cpu_tensor()
                for gid in self._unique_draft_gids
            ]
            if self.spec and self._draft_pools
            else []
        )
        draft_cols: list[list[int]] = []

        for _rid, st, i, seq_len in todo:
            for s in range(sw_subpages_per_slot):
                r0 = s * self.block_size
                if seq_len <= SW_RING:
                    lp_lo = s if r0 < seq_len else 0
                    lp_hi = lp_lo
                    split_off = self.block_size
                    valid_len = max(0, min(self.block_size, seq_len - r0))
                else:
                    start = seq_len - SW_RING
                    pos0 = start + ((r0 - start) % SW_RING)
                    pos_last = start + ((r0 + self.block_size - 1 - start) % SW_RING)
                    lp_lo = pos0 // self.block_size
                    lp_hi = pos_last // self.block_size
                    split_off = (
                        self.block_size if lp_lo == lp_hi else (start % self.block_size)
                    )
                    valid_len = self.block_size
                col = [int(bt_sw[i, lp_lo]) for bt_sw in bt_sw_by_g]
                col.extend(int(bt_sw[i, lp_hi]) for bt_sw in bt_sw_by_g)
                col.extend([split_off, valid_len, st.slot * sw_subpages_per_slot + s])
                sw_cols.append(col)
                if draft_bts:
                    dcol = [int(bt[i, lp_lo]) for bt in draft_bts]
                    dcol.extend(int(bt[i, lp_hi]) for bt in draft_bts)
                    dcol.extend(col[-3:])
                    draft_cols.append(dcol)

            n_blocks = -(-seq_len // self.block_size)
            max_chunk_needed = seq_len // PAGE_TOKENS
            for c_idx in range(max_chunk_needed + 1):
                if c_idx < self.max_kp:
                    self._ensure_slot_kpage(st.slot, c_idx)
            for lp in range(n_blocks):
                src_blk = int(bt_gl[i, lp])
                dst_sub = self._slot_subpage(st.slot, lp)
                gl_pairs.append((src_blk, dst_sub))
            if self.prefix_cache_enabled:
                full_blocks = seq_len // self.block_size
                for lp in range(full_blocks):
                    owner_r = lp % self.world
                    src_blk = int(bt_gl[i, lp])
                    c_sub = self._alloc_cache_subpage_for_rank(owner_r, src_blk)
                    if c_sub >= 0:
                        gl_pairs.append((src_blk, c_sub))
                st.last_cached_tokens = full_blocks * self.block_size

        dedup_gl = list(
            {dst: (src, dst) for src, dst in gl_pairs if src >= 0 and dst >= 0}.values()
        )
        total_items = max(len(sw_cols), len(dedup_gl), 1)
        buckets = (32, 128, 256)
        self._sync_kpt_device()
        for off in range(0, total_items, buckets[-1]):
            sw_chunk = sw_cols[off : off + buckets[-1]]
            gl_chunk = dedup_gl[off : off + buckets[-1]]
            need_len = max(len(sw_chunk), len(gl_chunk), 1)
            b_sync = next(x for x in buckets if x >= need_len)
            sw_arr = np.full((2 * n_g + 3, b_sync), -1, dtype=np.int32)
            for j, col in enumerate(sw_chunk):
                sw_arr[:, j] = col
            gl_arr = np.full((2, b_sync), -1, dtype=np.int32)
            for j, (s_blk, d_sub) in enumerate(gl_chunk):
                gl_arr[0, j], gl_arr[1, j] = s_blk, d_sub
            sync_op = self._get_sync_seq_kv_op(b_sync)
            self.kv_sw, self.kv_gl = sync_op(
                self.kv_sw,
                self.kv_gl,
                *self._kv_pools,
                _to_torch(sw_arr, self._dev),
                _to_torch(gl_arr, self._dev),
            )
        n_dg = len(self._unique_draft_gids)
        for off in range(0, len(draft_cols), buckets[-1]):
            chunk = draft_cols[off : off + buckets[-1]]
            b_sync = next(x for x in buckets if x >= len(chunk))
            d_arr = np.full((2 * n_dg + 3, b_sync), -1, dtype=np.int32)
            for j, col in enumerate(chunk):
                d_arr[:, j] = col
            self.kv_draft = self._get_sync_draft_op(b_sync)(
                self.kv_draft, *self._draft_pools, _to_torch(d_arr, self._dev)
            )
        synchronize_tensors(
            [self.kv_sw, self.kv_gl, self.kpt]
            + ([self.kv_draft] if draft_cols else []),
            wait=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.last_handover_ms = elapsed_ms
        return elapsed_ms

    def take_ownership(
        self,
        runner: Any,
        scheduler_output: Any,
        sampled: dict[str, Any],
        drafts: dict[str, list[int]] | None = None,
    ) -> None:
        del scheduler_output
        if not self.ready or not self._kv_pools:
            return

        todo = []
        nb_cur = _bucket(max(1, int(getattr(runner.input_batch, "num_reqs", 1))))
        kd_cur = min(SPEC_DRAFT_DEPTH.get(nb_cur, 0), self.spec_k) if self.spec else 0
        for rid, (_tok, seq_len) in sampled.items():
            if rid not in runner.input_batch.req_id_to_index:
                continue
            st = self.slots.get(rid)
            if st is not None and st.owned:
                continue
            if not self._megakernel_eligible_request(runner, rid):
                continue
            kp_need = self._lifetime_kpages(runner, rid, int(seq_len))
            if self._kp_reserved + kp_need > self.base_kpages - 1:
                # Not enough superpages for its whole lifetime; vLLM keeps decoding it.
                continue
            if st is None:
                st = self.slots.acquire(rid)
            if st is None:
                continue
            st.kp_need = kp_need
            self._kp_reserved += kp_need
            st.owned = True
            st.kv_upto = int(seq_len)
            st.draft_upto = int(seq_len)
            st.last_next_tok = int(_tok)
            if self.spec:
                # The vLLM drafter proposed from this prefill; its drafts seed the first
                # verify window.
                d = [int(x) for x in (drafts or {}).get(rid, [])] if kd_cur > 0 else []
                st.next_verify = [int(_tok)] + d
                st.next_conf = None
            todo.append(
                (rid, st, int(runner.input_batch.req_id_to_index[rid]), int(seq_len))
            )
        if not todo:
            return
        if (
            self.spec
            and kd_cur == 0
            and getattr(runner, "spec_decode_manager", None) is not None
        ):
            runner.spec_decode_manager._draft_token_ids = [
                [] for _ in range(int(runner.input_batch.num_reqs))
            ]

        if self._prefetched_step is not None:
            self._flush_prefetched_step()
        self._sync_seq_kv_from_vllm(runner, todo)
        if self.spec:
            self._maybe_prefetch_spec_from_runner(runner)
        else:
            self._maybe_prefetch_from_runner(runner, pending_out_inc=0)

    def move_owned_to_front(
        self, runner: Any, num_decode: int, num_windowed: int
    ) -> tuple[int, int, int]:
        n = runner.input_batch.num_reqs
        ids = runner.input_batch.req_ids

        def owned(i: int) -> bool:
            st = self.slots.get(ids[i])
            return st is not None and st.owned

        flags = [owned(i) for i in range(n)]
        k = sum(flags)
        if k == 0:
            return 0, num_decode, num_windowed
        order = [i for i in range(n) if flags[i]] + [
            i for i in range(n) if not flags[i]
        ]
        pos = list(range(n))
        where = list(range(n))
        for dst, orig in enumerate(order):
            src = where[orig]
            if src != dst:
                runner.input_batch.swap_states(dst, src)
                o_dst = pos[dst]
                pos[dst], pos[src] = orig, o_dst
                where[orig], where[o_dst] = dst, src
        n_dec_rest = sum(1 for i in range(num_decode) if not flags[i])
        n_win_rest = sum(1 for i in range(num_decode, num_windowed) if not flags[i])
        return k, k + n_dec_rest, k + n_dec_rest + n_win_rest

    def owned_indices(self, runner: Any) -> list[int]:
        out = []
        for i in range(runner.input_batch.num_reqs):
            st = self.slots.get(runner.input_batch.req_ids[i])
            if st is not None and st.owned:
                out.append(i)
        return out

    def _get_step_op(self, batch: int, need_logprobs: bool = False) -> Any:
        key = ("hybrid_step_op", batch, bool(need_logprobs))
        op = self._ops.get(key)
        if op is not None:
            return op
        from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
        from vllm_torchtpu.kernels.muse_megakernel.kernel import make_hybrid_decode_step

        step: Any = make_hybrid_decode_step(
            self.mesh,
            self.cfg,
            max_slots=self.n_slots,
            num_kpages=self.num_kpages,
            max_kp=self.max_kp,
            batch_size=batch,
            force_rows8=False,
            explicit_rank_dev=True,
            return_logits=bool(need_logprobs),
        )
        names = self.weight_names
        nw = len(names)
        p = jax.sharding.PartitionSpec
        in_specs = [p(TP_AXIS_NAME)] * nw + [
            p(TP_AXIS_NAME),
            p(TP_AXIS_NAME),
            p(),
            p(),
            p(TP_AXIS_NAME),
            p(),
        ]

        if not need_logprobs:

            def pick_tok(prev_l: jax.Array, ctrl_tok: jax.Array) -> jax.Array:
                src = ctrl_tok[3]
                prev_tok = prev_l[0, jnp.clip(src, 0, prev_l.shape[1] - 1)]
                return jnp.where(src >= 0, prev_tok, ctrl_tok[2])

            # Every rank holds identical targets; read the local copy so no collective
            # is needed.
            pick_sm = jax.shard_map(
                pick_tok,
                mesh=self.mesh,
                in_specs=(p(TP_AXIS_NAME), p()),
                out_specs=p(),
                check_vma=False,
            )

            def body_fast(*a: jax.Array) -> tuple[jax.Array, ...]:
                w = dict(zip(names, a[:nw]))
                kv_sw, kv_gl, ctrl_tok, kpt, rank_arr, dev_arr, prev = a[nw:]
                ctrl = ctrl_tok[:2]
                # Row 3 >= 0 takes the token from the previous step's targets on device,
                # so a chained step never waits on a host round trip.
                tok = pick_sm(prev, ctrl_tok)
                targets, kv_sw, kv_gl = step(
                    w, kv_sw, kv_gl, tok, ctrl, kpt, rank_arr, dev_arr
                )
                return targets, kv_sw, kv_gl

            out_specs_fast = (p(TP_AXIS_NAME), p(TP_AXIS_NAME), p(TP_AXIS_NAME))
            op = sharded_jax_op(
                f"muse_mk::hybrid_decode_b{batch}",
                _explicit_arity(body_fast, nw + 7, 3),
                mesh=self.mesh,
                input_partition_specs=in_specs + [p(TP_AXIS_NAME)],
                output_partition_specs=out_specs_fast,
                donate_argnums=(nw, nw + 1),
            )
            self._ops[key] = op
            return op

        topk_sm = self._topk_sm()

        def body_lp(*a: jax.Array) -> tuple[jax.Array, ...]:
            w = dict(zip(names, a[:nw]))
            kv_sw, kv_gl, ctrl_tok, kpt, rank_arr, dev_arr = a[nw:]
            ctrl = ctrl_tok[:2]
            tok = ctrl_tok[2]
            targets, kv_sw, kv_gl, logits = step(
                w, kv_sw, kv_gl, tok, ctrl, kpt, rank_arr, dev_arr
            )
            top_i, top_v, lse = topk_sm(logits, rank_arr)
            return targets, kv_sw, kv_gl, top_i, top_v, lse

        out_specs = (
            p(TP_AXIS_NAME),
            p(TP_AXIS_NAME),
            p(TP_AXIS_NAME),
            p(TP_AXIS_NAME),
            p(TP_AXIS_NAME),
            p(TP_AXIS_NAME),
        )
        op = sharded_jax_op(
            f"muse_mk::hybrid_decode_lp_b{batch}",
            _explicit_arity(body_lp, nw + 6, 6),
            mesh=self.mesh,
            input_partition_specs=in_specs,
            output_partition_specs=out_specs,
            donate_argnums=(nw, nw + 1),
        )
        self._ops[key] = op
        return op

    def _topk_sm(self) -> Any:
        """Per row top LOGPROB_K ids and logprobs from TP vocab sharded logits."""
        p = jax.sharding.PartitionSpec
        k_out = LOGPROB_K
        vshard = self.cfg.vocab // self.world
        mesh = self.mesh

        def topk_local(
            logits_l: jax.Array, rank_l: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            # Per rank: local top k plus log-sum-exp over this vocab shard, merged
            # across TP in fixed rank order.
            lg = logits_l[0].astype(jnp.float32)
            vals, idx = jax.lax.top_k(lg, k_out)
            idx = idx + rank_l[0] * vshard
            m = jnp.max(lg, axis=-1, keepdims=True)
            lse = m + jnp.log(jnp.sum(jnp.exp(lg - m), axis=-1, keepdims=True))
            all_v = jax.lax.all_gather(vals, TP_AXIS_NAME, axis=1, tiled=True)
            all_i = jax.lax.all_gather(idx, TP_AXIS_NAME, axis=1, tiled=True)
            all_l = jax.lax.all_gather(lse, TP_AXIS_NAME, axis=1, tiled=True)
            gm = jnp.max(all_l, axis=-1, keepdims=True)
            g_lse = gm + jnp.log(jnp.sum(jnp.exp(all_l - gm), axis=-1, keepdims=True))
            # Stable order: value descending, then token id ascending (matches the
            # kernel's argmax tie break).
            order = jnp.lexsort((all_i, -all_v), axis=-1)[:, :k_out]
            top_v = jnp.take_along_axis(all_v, order, axis=-1) - g_lse
            top_i = jnp.take_along_axis(all_i, order, axis=-1)
            return top_i[None], top_v[None], g_lse[None]

        return jax.shard_map(
            topk_local,
            mesh=mesh,
            in_specs=(p(TP_AXIS_NAME), p(TP_AXIS_NAME)),
            out_specs=(p(TP_AXIS_NAME), p(TP_AXIS_NAME), p(TP_AXIS_NAME)),
            check_vma=False,
        )

    def _get_spec_rollback_op(self, rows: int) -> Any:
        key = ("spec_rollback", rows)
        op = self._ops.get(key)
        if op is not None:
            return op
        from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
        from vllm_torchtpu.kernels.muse_megakernel.fastpath_ops import (
            ring_rows_write_local,
        )

        p = jax.sharding.PartitionSpec
        n_slots = self.n_slots

        def rb_local(kv_sw, kv_draft, sw_old, draft_old, ctrl):
            slots = jnp.clip(ctrl[1, :rows], 0, n_slots - 1)
            ridx = jnp.maximum(ctrl[0, :rows], 0) % SW_RING
            mask = (ctrl[0, :rows] >= 0).astype(jnp.int32)
            return (
                ring_rows_write_local(kv_sw, sw_old, slots, ridx, mask),
                ring_rows_write_local(kv_draft, draft_old, slots, ridx, mask),
            )

        sm = jax.shard_map(
            rb_local,
            mesh=self.mesh,
            in_specs=(
                p(TP_AXIS_NAME),
                p(TP_AXIS_NAME),
                p(TP_AXIS_NAME),
                p(TP_AXIS_NAME),
                p(),
            ),
            out_specs=(p(TP_AXIS_NAME), p(TP_AXIS_NAME)),
            check_vma=False,
        )
        op = sharded_jax_op(
            f"muse_mk::spec_rollback_r{rows}",
            _explicit_arity(sm, 5, 2),
            mesh=self.mesh,
            input_partition_specs=[
                p(TP_AXIS_NAME),
                p(TP_AXIS_NAME),
                p(TP_AXIS_NAME),
                p(TP_AXIS_NAME),
                p(),
            ],
            output_partition_specs=(p(TP_AXIS_NAME), p(TP_AXIS_NAME)),
            donate_argnums=(0, 1),
        )
        self._ops[key] = op
        return op

    def _get_spec_step_op(
        self, nb: int, rows: int, kd: int, need_lp: bool = False
    ) -> Any:
        """Fused DSpark step for `nb` request slots verifying `rows` packed tokens.

        Inputs: weights, draft weights, kv_sw, kv_gl, kv_draft, ctrl [8, 16], kpt, rank,
        dev.
        ctrl rows: 0 pos, 1 slot, 2 token (verify rows, -1 pos pads); 3 request of each
        row;
        4 accepted count per request (host computed after the previous step) unused
        here;
        5 bonus row index per request; 6 base pos per request; 7 slot per request.
        One megakernel call verifies the window and returns targets and aux hidden
        states;
        accepted rows (target matches the next draft) update the draft ring; the draft
        runs from
        each request's bonus = target after its last accepted row.
        """
        key = ("spec_step", nb, rows, kd, need_lp)
        op = self._ops.get(key)
        if op is not None:
            return op
        from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
        from vllm_torchtpu.kernels.muse_megakernel.dspark_step import (
            AUX_LAYERS,
            DSPARK_WEIGHT_NAMES,
            dspark_ctx_update_local,
            dspark_draft_local,
        )
        from vllm_torchtpu.kernels.muse_megakernel.fastpath_ops import (
            ring_rows_read_local,
            ring_rows_write_local,
        )
        from vllm_torchtpu.kernels.muse_megakernel.kernel import make_hybrid_decode_step

        step: Any = make_hybrid_decode_step(
            self.mesh,
            self.cfg,
            max_slots=self.n_slots,
            num_kpages=self.num_kpages,
            max_kp=self.max_kp,
            batch_size=rows,
            num_reqs=nb,
            force_rows8=False,
            explicit_rank_dev=True,
            return_logits=need_lp,
            aux_layers=AUX_LAYERS,
        )
        topk_sm = self._topk_sm() if need_lp else None
        names = self.weight_names
        dnames = DSPARK_WEIGHT_NAMES
        nw, nd = len(names), len(dnames)
        dc = self.dcfg
        p = jax.sharding.PartitionSpec
        mesh = self.mesh

        n_slots = self.n_slots
        no_rollback = rows == 1 or kd == 0
        save_draft_old = bool(self.async_prefetch_enabled) and not need_lp
        save_sw_old = save_draft_old or not no_rollback

        def ring_loc(ctrl):
            return (
                jnp.clip(ctrl[1, :rows], 0, n_slots - 1),
                jnp.maximum(ctrl[0, :rows], 0) % SW_RING,
            )

        def rows_pre(kv_sw, kv_draft, ctrl):
            slots, ridx = ring_loc(ctrl)
            sw_saved = (
                ring_rows_read_local(kv_sw, slots, ridx)
                if save_sw_old
                else jnp.zeros((1, kv_sw.shape[1], rows, kv_sw.shape[-1]), kv_sw.dtype)
            )
            dr_saved = (
                ring_rows_read_local(kv_draft, slots, ridx)
                if save_draft_old
                else jnp.zeros(
                    (1, kv_draft.shape[1], rows, kv_draft.shape[-1]), kv_draft.dtype
                )
            )
            return sw_saved, dr_saved

        pre_sm = jax.shard_map(
            rows_pre,
            mesh=mesh,
            in_specs=(p(TP_AXIS_NAME), p(TP_AXIS_NAME), p()),
            out_specs=(p(TP_AXIS_NAME), p(TP_AXIS_NAME)),
            check_vma=False,
        )

        def post(dw, emb, kv_sw, kv_draft, sw_old, targets, aux, rank_l, ctrl):
            """Acceptance, SW ring rollback, draft ring update and next draft (local
            arrays).
            """
            w1 = jax.tree.map(lambda a: a[0], dw)
            pos_r, slot_r, tok_r = ctrl[0, :rows], ctrl[1, :rows], ctrl[2, :rows]
            req_r = ctrl[3, :rows]
            base, rslot = ctrl[6, :nb], ctrl[7, :nb]
            tg = targets[0]
            if no_rollback:
                ok = pos_r >= 0
            else:
                nxt_tok = jnp.concatenate([tok_r[1:], jnp.full((1,), -1, jnp.int32)])
                nxt_req = jnp.concatenate([req_r[1:], jnp.full((1,), -1, jnp.int32)])
                match = (nxt_req == req_r) & (nxt_tok == tg)
                first = jnp.concatenate(
                    [jnp.ones((1,), jnp.bool_), req_r[1:] != req_r[:-1]]
                )
                run = jnp.bool_(True)
                oks = []
                for j in range(rows):
                    run = jnp.where(first[j], jnp.bool_(True), run)
                    oks.append(run & (pos_r[j] >= 0))
                    run = run & match[j]
                ok = jnp.stack(oks)
                rej = (~ok & (pos_r >= SW_RING)).astype(jnp.int32)
                kv_sw = ring_rows_write_local(kv_sw, sw_old, *ring_loc(ctrl), rej)
            req_c = jnp.clip(req_r, 0, nb - 1)
            n_acc = jnp.zeros((nb,), jnp.int32).at[req_c].add(ok.astype(jnp.int32))
            last_row = (
                jnp.zeros((nb,), jnp.int32)
                .at[req_c]
                .max(jnp.where(ok, jnp.arange(rows, dtype=jnp.int32), 0))
            )
            bonus = tg[last_row]
            ctx_pos = jnp.where(ok, pos_r, -1)
            kvd = dspark_ctx_update_local(
                dc, w1, kv_draft[0], aux[0], ctx_pos, slot_r, rank=rank_l[0]
            )
            if kd == 0:
                d = jnp.zeros((nb, 1), jnp.int32)
                cf = jnp.zeros((nb, 1), jnp.float32)
            else:
                d, cf = dspark_draft_local(
                    dc, w1, emb[0], kvd, bonus, base + n_acc, rslot, kd, rank=rank_l[0]
                )
            cf_i32 = jax.lax.bitcast_convert_type(cf, jnp.int32)
            meta_out = jnp.concatenate(
                [tg.reshape(-1), n_acc.reshape(-1), d.reshape(-1), cf_i32.reshape(-1)]
            )
            return meta_out, kv_sw, kvd[None]

        post_sm = jax.shard_map(
            post,
            mesh=mesh,
            in_specs=(p(TP_AXIS_NAME),) * 8 + (p(),),
            out_specs=(p(), p(TP_AXIS_NAME), p(TP_AXIS_NAME)),
            check_vma=False,
        )

        def body(*a: jax.Array) -> tuple[jax.Array, ...]:
            w = dict(zip(names, a[:nw]))
            dw = dict(zip(dnames, a[nw : nw + nd]))
            kv_sw, kv_gl, kv_draft, ctrl, kpt, rank_arr, dev_arr = a[nw + nd :]
            pos_r, slot_r, tok_r = ctrl[0, :rows], ctrl[1, :rows], ctrl[2, :rows]
            sw_old, draft_old = pre_sm(kv_sw, kv_draft, ctrl)
            # The kernel writes kv_sw in place through DMA and returns the input
            # array, so XLA sees no dependency between it and the ring gather/scatter.
            # The barriers pin the gather before the kernel and rollback after.
            kv_sw, kv_draft, sw_old, draft_old = jax.lax.optimization_barrier(
                (kv_sw, kv_draft, sw_old, draft_old)
            )
            outs = step(
                w,
                kv_sw,
                kv_gl,
                tok_r,
                jnp.stack([pos_r, slot_r]),
                kpt,
                rank_arr,
                dev_arr,
            )
            targets, kv_sw, kv_gl, aux = outs[0], outs[1], outs[2], outs[-1]
            lg_out = outs[3] if need_lp else None
            kv_sw, targets, aux = jax.lax.optimization_barrier((kv_sw, targets, aux))
            meta_out, kv_sw, kv_draft = post_sm(
                dw, w["emb"], kv_sw, kv_draft, sw_old, targets, aux, rank_arr, ctrl
            )
            if need_lp and topk_sm is not None and lg_out is not None:
                top_i, top_v, _ = topk_sm(lg_out, rank_arr)
                return meta_out, kv_sw, kv_gl, kv_draft, top_i, top_v, sw_old, draft_old
            return meta_out, kv_sw, kv_gl, kv_draft, sw_old, draft_old

        in_specs = [p(TP_AXIS_NAME)] * (nw + nd) + [
            p(TP_AXIS_NAME),
            p(TP_AXIS_NAME),
            p(TP_AXIS_NAME),
            p(),
            p(),
            p(TP_AXIS_NAME),
            p(),
        ]
        op = sharded_jax_op(
            f"muse_mk::spec_b{nb}_r{rows}_k{kd}_lp{int(need_lp)}",
            _explicit_arity(body, nw + nd + 7, 8 if need_lp else 6),
            mesh=self.mesh,
            input_partition_specs=in_specs,
            output_partition_specs=(p(),) + (p(TP_AXIS_NAME),) * (7 if need_lp else 5),
            donate_argnums=(nw + nd, nw + nd + 1, nw + nd + 2),
        )
        self._ops[key] = op
        return op

    def _prev_dummy(self, b: int) -> torch.Tensor:
        cache = self.__dict__.setdefault("_prev_dummies", {})
        t = cache.get(b)
        if t is None:
            t = _to_torch(np.zeros((1, b), np.int32), self._dev)
            cache[b] = t
        return t

    def _enqueue_step_chunks(
        self,
        tokens: np.ndarray,
        positions: np.ndarray,
        slots: np.ndarray,
        need_logprobs: bool = False,
        src_idx: np.ndarray | None = None,
        prev_chunks: list[tuple[Any, int]] | None = None,
    ) -> tuple[list[tuple[Any, int]], int]:
        """Dispatch decode chunks; rows with src_idx >= 0 read their token from
        prev_chunks
        on device.
        """
        n = len(slots)
        for j in range(n):
            c_idx = int(positions[j]) // PAGE_TOKENS
            if c_idx < self.max_kp:
                self._ensure_slot_kpage(int(slots[j]), c_idx)
        self._sync_kpt_device()

        chunks: list[tuple[Any, int]] = []
        kb = 8 if need_logprobs else KERNEL_BATCH
        for g in range(0, n, kb):
            tok_g = tokens[g : g + kb]
            pos_g = positions[g : g + kb]
            slot_g = slots[g : g + kb]
            m = len(slot_g)
            b = 1 << max(0, int(m - 1).bit_length())
            ctrl_tok = np.full((4, b), -1, np.int32)
            ctrl_tok[0, :m] = pos_g
            ctrl_tok[1, :] = int(slot_g[0])
            ctrl_tok[1, :m] = slot_g
            ctrl_tok[2, :] = max(int(tok_g[0]), 0)
            ctrl_tok[2, :m] = np.maximum(tok_g, 0)
            prev = None
            if not need_logprobs:
                prev = self._prev_dummy(b)
                if src_idx is not None and prev_chunks is not None:
                    pouts, _ = prev_chunks[g // kb]
                    if int(pouts[0].shape[1]) == b:
                        prev = pouts[0]
                        ctrl_tok[3, :m] = src_idx[g : g + kb]

            op = self._get_step_op(b, need_logprobs=need_logprobs)
            ctrl_tok_t = _to_torch(ctrl_tok, self._dev, sync=False)
            outs = op(
                *self._w_list,
                self.kv_sw,
                self.kv_gl,
                ctrl_tok_t,
                self.kpt,
                self.rank_1d_t,
                self.dev_t,
                *(() if prev is None else (prev,)),
            )
            self.kv_sw, self.kv_gl = outs[1], outs[2]
            if g + kb < n:
                from vllm_torchtpu.utils import synchronize_tensors

                c_sync = [outs[0], self.kv_sw, self.kv_gl]
                if need_logprobs:
                    c_sync.extend([outs[3], outs[4]])
                synchronize_tensors(c_sync, wait=False)
            chunks.append((outs, m))
        return chunks, len(chunks)

    def _maybe_prefetch_from_runner(
        self, runner: Any, pending_out_inc: int = 1
    ) -> None:
        num_reqs = int(getattr(runner.input_batch, "num_reqs", 0))
        if (
            not self.async_prefetch_enabled
            or (
                self.spec
                and min(SPEC_DRAFT_DEPTH.get(_bucket(max(1, num_reqs)), 0), self.spec_k)
                > 0
            )
            or self._in_mixed_step
            or getattr(runner.input_batch, "max_num_logprobs", None) is not None
            or bool(envs.VLLM_TPU_MUSE_MK_EAGER_KV_WRITEBACK)
        ):
            return
        if num_reqs <= 0 or num_reqs > self.n_slots:
            return
        req_ids = runner.input_batch.req_ids[:num_reqs]
        next_items: list[tuple[int, int, int]] = []
        for r in req_ids:
            st = self.slots.get(r)
            if st is None or not st.owned or st.last_next_tok < 0:
                return
            if st.kv_upto >= self.max_model_len:
                continue
            req_obj = runner.requests.get(r)
            sp = (
                getattr(req_obj, "sampling_params", None)
                if req_obj is not None
                else None
            )
            if sp is not None:
                out_len = (
                    len(getattr(req_obj, "output_token_ids", ()) or ())
                    + pending_out_inc
                )
                if sp.max_tokens is not None and out_len >= sp.max_tokens:
                    continue
                if not getattr(sp, "ignore_eos", False):
                    stop_ids = getattr(sp, "stop_token_ids", None) or ()
                    if (
                        st.last_next_tok in self._eos_token_ids
                        or st.last_next_tok in stop_ids
                    ):
                        continue
            next_items.append((int(st.slot), int(st.kv_upto), int(st.last_next_tok)))

        if (
            not next_items
            or len(next_items) > self.n_slots
            or self._prefetched_step is not None
        ):
            return
        from vllm_torchtpu.utils import synchronize_tensors

        # Runner row order, as on the synchronous path: a slot sorted prefetch diverged
        # in server (P1 gsm8k A/B).
        slots_arr = np.asarray([x[0] for x in next_items], dtype=np.int32)
        pos_arr = np.asarray([x[1] for x in next_items], dtype=np.int32)
        toks_arr = np.asarray([x[2] for x in next_items], dtype=np.int32)
        chunks, n_calls = self._enqueue_step_chunks(
            toks_arr, pos_arr, slots_arr, need_logprobs=False
        )
        sync_list = [self.kv_sw, self.kv_gl, *[outs[0] for outs, _ in chunks]]
        synchronize_tensors(sync_list, wait=False)
        self.total_sync_calls += 1
        self.last_step_sync_calls = 1
        self.last_step_n_calls = n_calls
        self._prefetched_step = {
            "slot_map": {
                int(s): (int(p), int(t), idx)
                for idx, (s, p, t) in enumerate(next_items)
            },
            "chunks": chunks,
            "n": len(next_items),
            "n_calls": n_calls,
            "items": next_items,
        }

    def _chain_next_step(self, pref: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch step N+1 for pref's rows before N is read; tokens come from N's
        targets on device.

        Rows that cannot continue (model length or max_tokens) are dropped; the token
        keys are
        filled in once N's targets reach the host. A row whose request stopped on EOS
        wastes one
        row and writes the KV of its true next token, which a later restore would
        recompute.
        """
        if (
            not self.chain_enabled
            or self._in_mixed_step
            or any(len(c[0]) != 3 for c in pref["chunks"])
        ):
            return None
        items = pref["items"]
        if any(p + 1 >= self.max_model_len for _, p, _ in items):
            return None
        runner = self.runner
        for s, _, _ in items:
            rid = next((r for r, st in self.slots.by_req.items() if st.slot == s), None)
            req_obj = runner.requests.get(rid) if rid is not None else None
            sp = getattr(req_obj, "sampling_params", None)
            if sp is None:
                return None
            out_len = len(getattr(req_obj, "output_token_ids", ()) or ()) + 2
            if sp.max_tokens is not None and out_len > sp.max_tokens:
                return None
        n = len(items)
        slots_arr = np.asarray([x[0] for x in items], np.int32)
        pos_arr = np.asarray([x[1] + 1 for x in items], np.int32)
        kb = KERNEL_BATCH
        src = np.asarray([j % kb for j in range(n)], np.int32)
        chunks, n_calls = self._enqueue_step_chunks(
            np.zeros((n,), np.int32),
            pos_arr,
            slots_arr,
            src_idx=src,
            prev_chunks=pref["chunks"],
        )
        from vllm_torchtpu.utils import synchronize_tensors

        synchronize_tensors(
            [self.kv_sw, self.kv_gl, *[o[0] for o, _ in chunks]], wait=False
        )
        self.chained_steps += 1
        return {
            "chunks": chunks,
            "n": n,
            "n_calls": n_calls,
            "pending": True,
            "items": [(int(s), int(p), -1) for s, p in zip(slots_arr, pos_arr)],
        }

    def decode_step(
        self,
        tokens: np.ndarray,
        positions: np.ndarray,
        slots: np.ndarray,
        need_logprobs: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray | None, np.ndarray | None]]:
        """Run one decode step for n requests (bucketed to 1, 2, 4, 8, 16; >16 uses
        N-call).

        Returns the greedy tokens [n] and (top_ids [n, LOGPROB_K], top_logprobs [n,
        LOGPROB_K]).
        """
        n = len(slots)
        if self._prefetched_step is not None:
            pref = self._prefetched_step
            self._prefetched_step = None
            slot_map = pref["slot_map"]
            if (
                not need_logprobs
                and n <= pref["n"]
                and all(
                    int(slots[j]) in slot_map
                    and slot_map[int(slots[j])][:2]
                    == (int(positions[j]), int(tokens[j]))
                    for j in range(n)
                )
            ):
                chain = self._chain_next_step(pref) if n == pref["n"] else None
                preds = [outs[0].cpu().numpy()[0, :m] for outs, m in pref["chunks"]]
                raw_targets = np.concatenate(preds) if len(preds) > 1 else preds[0]
                if chain is not None:
                    chain["items"] = [
                        (s, p, int(t))
                        for (s, p, _), t in zip(chain["items"], raw_targets)
                    ]
                    chain["slot_map"] = {
                        s: (p, t, idx) for idx, (s, p, t) in enumerate(chain["items"])
                    }
                    self._prefetched_step = chain
                order = [slot_map[int(s)][2] for s in slots]
                self.prefetch_hits += 1
                self.last_step_sync_calls = 1
                self.last_step_n_calls = pref["n_calls"]
                return raw_targets[order], (None, None)
            for outs, _ in pref["chunks"]:
                _ = outs[0].cpu()
            self.prefetch_misses += 1

        from vllm_torchtpu.utils import synchronize_tensors

        chunks, n_calls = self._enqueue_step_chunks(
            tokens, positions, slots, need_logprobs=need_logprobs
        )
        sync_list = [self.kv_sw, self.kv_gl]
        for outs, _ in chunks:
            sync_list.append(outs[0])
            if need_logprobs:
                sync_list.extend([outs[3], outs[4]])
        synchronize_tensors(sync_list)
        self.total_sync_calls += 1
        self.last_step_sync_calls = 1
        self.last_step_n_calls = n_calls

        preds, ids, lps = [], [], []
        for outs, m in chunks:
            preds.append(outs[0].cpu().numpy()[0, :m])
            if need_logprobs:
                ids.append(outs[3].cpu().numpy()[0, :m])
                lps.append(outs[4].cpu().numpy()[0, :m].astype(np.float32))
        targets = np.concatenate(preds) if len(preds) > 1 else preds[0]
        top_ids = (
            (np.concatenate(ids) if len(ids) > 1 else ids[0]) if need_logprobs else None
        )
        top_lps = (
            (np.concatenate(lps) if len(lps) > 1 else lps[0]) if need_logprobs else None
        )
        return targets, (top_ids, top_lps)

    def _writeback_decoded_kv(
        self,
        runner: Any,
        indices: list[int],
        slots: np.ndarray,
        pos: np.ndarray,
    ) -> None:
        if not self._kv_pools:
            return
        from vllm_torchtpu.utils import synchronize_tensors

        n = len(indices)
        bt_gl = runner.input_batch.block_table[self._gl_group_id].get_cpu_tensor()
        bt_sw_by_gid = {
            gid: runner.input_batch.block_table[gid].get_cpu_tensor()
            for gid in self._unique_sw_gids
        }
        for g in range(0, n, KERNEL_BATCH):
            m = min(KERNEL_BATCH, n - g)
            b_wb = 1 if m == 1 else KERNEL_BATCH
            gl_loc = np.zeros((self.cfg.num_gl_layers, b_wb, 2), dtype=np.int32)
            sw_loc = np.zeros((self.cfg.num_sw_layers, b_wb, 2), dtype=np.int32)
            req_blk_gl = np.full((b_wb,), -1, dtype=np.int32)
            req_boff = np.zeros((b_wb,), dtype=np.int32)
            req_blk_sw = np.full((self.cfg.num_sw_layers, b_wb), -1, dtype=np.int32)
            for j in range(m):
                req_i = indices[g + j]
                s = int(slots[g + j])
                p = int(pos[g + j])
                lp = p // self.block_size
                boff = p % self.block_size
                kp = int(self.kpt_host[s, p // PAGE_TOKENS])
                poff = p % PAGE_TOKENS
                ridx = p % SW_RING
                gl_loc[:, j, 0] = kp
                gl_loc[:, j, 1] = poff
                sw_loc[:, j, 0] = s
                sw_loc[:, j, 1] = ridx
                req_blk_gl[j] = int(bt_gl[req_i, lp])
                req_boff[j] = boff
                for sw_idx, gid in enumerate(self._sw_gids):
                    req_blk_sw[sw_idx, j] = int(bt_sw_by_gid[gid][req_i, lp])

            gather_op = self._get_gather_kv_op(b_wb)
            scatter_op = self._get_scatter_pool_op(b_wb)
            gl_vals, sw_vals = gather_op(
                self.kv_sw,
                self.kv_gl,
                _to_torch(gl_loc, self._dev),
                _to_torch(sw_loc, self._dev),
            )
            synchronize_tensors([gl_vals, sw_vals])
            for idx_p, layer_specs in enumerate(self._pool_layers):
                meta_p = np.full((4, b_wb, 4), -1, dtype=np.int32)
                for k_slot, (is_sw, lidx) in enumerate(layer_specs[:4]):
                    meta_p[k_slot, :m, 0] = is_sw
                    meta_p[k_slot, :m, 1] = lidx
                    meta_p[k_slot, :m, 2] = (
                        req_blk_sw[lidx, :m] if is_sw else req_blk_gl[:m]
                    )
                    meta_p[k_slot, :m, 3] = req_boff[:m]
                nt = scatter_op(
                    self._kv_pools[idx_p],
                    gl_vals,
                    sw_vals,
                    _to_torch(meta_p, self._dev),
                )
                self._kv_pools[idx_p].copy_(nt)
                synchronize_tensors([self._kv_pools[idx_p]])
                del nt

    def _plan_drafts(self, nb: int, drafts: np.ndarray) -> int:
        """Verify window length q for the next step of a padded nb-request batch."""
        kd = drafts.shape[1] if drafts.ndim == 2 else 0
        cands = [q for q in (1, 2, 4, 8) if q - 1 <= kd and nb * q <= SPEC_MAX_ROWS]
        if not cands:
            return 1
        q = self.spec_fixed_q.get(nb, cands[-1])
        return max((c for c in cands if c <= q), default=1)

    def _build_spec_ctrl(
        self,
        req_ids: list[str],
        pos0: np.ndarray,
        windows: list[list[int]],
    ) -> tuple[int, int, np.ndarray, list[int]]:
        n = len(req_ids)
        nb = _bucket(n)
        max_w = max(len(wd) for wd in windows)
        q_stride = _bucket(max_w)
        rows = max(nb, nb * q_stride)
        ctrl = np.full((8, SPEC_MAX_ROWS), -1, np.int32)
        st0 = self.slots.get(req_ids[0])
        assert st0 is not None
        ctrl[1, :] = st0.slot
        ctrl[2, :] = windows[0][0]
        ctrl[3, :] = 0
        for req_b in range(nb):
            st_b = self.slots.get(req_ids[req_b]) if req_b < n else st0
            assert st_b is not None
            r0 = req_b * q_stride
            ctrl[1, r0 : r0 + q_stride] = st_b.slot
            ctrl[3, r0 : r0 + q_stride] = min(req_b, n - 1)
        row_of: list[int] = []
        for b, (r, wd) in enumerate(zip(req_ids, windows)):
            st_b = self.slots.get(r)
            assert st_b is not None
            slot = st_b.slot
            r0 = b * q_stride
            row_of.append(r0)
            for k, tok in enumerate(wd):
                j = r0 + k
                ctrl[0, j], ctrl[1, j], ctrl[2, j], ctrl[3, j] = (
                    pos0[b] + k,
                    slot,
                    tok,
                    b,
                )
                c_idx = (int(pos0[b]) + k) // PAGE_TOKENS
                if c_idx < self.max_kp:
                    self._ensure_slot_kpage(slot, c_idx)
            ctrl[6, b], ctrl[7, b] = pos0[b], slot
        ctrl[7, n:nb] = ctrl[7, 0]
        ctrl[6, n:nb] = 0
        self._sync_kpt_device()
        return nb, rows, ctrl, row_of

    def _maybe_prefetch_spec_from_runner(
        self,
        runner: Any,
        pending_out: dict[str, int] | None = None,
        emitted: dict[str, list[int]] | None = None,
    ) -> None:
        num_reqs = int(getattr(runner.input_batch, "num_reqs", 0))
        if (
            not self.async_prefetch_enabled
            or not self.spec
            or self._in_mixed_step
            or self._prefetched_step is not None
            or getattr(runner.input_batch, "max_num_logprobs", None) is not None
            or bool(envs.VLLM_TPU_MUSE_MK_EAGER_KV_WRITEBACK)
            or num_reqs <= 0
            or num_reqs > self.n_slots
        ):
            return
        nb = _bucket(num_reqs)
        kd = min(SPEC_DRAFT_DEPTH.get(nb, 0), self.spec_k)
        req_ids = list(runner.input_batch.req_ids[:num_reqs])
        pos0_list: list[int] = []
        windows: list[list[int]] = []
        slot_map: dict[int, int] = {}
        max_q_req = SPEC_MAX_ROWS // nb
        for b, r in enumerate(req_ids):
            st = self.slots.get(r)
            if (
                st is None
                or not st.owned
                or not st.next_verify
                or st.draft_upto != st.kv_upto
            ):
                return
            slot_map[int(st.slot)] = b
            p0 = int(st.kv_upto)
            w = list(st.next_verify)[: (max_q_req if kd > 0 else 1)]
            extra = min(len(w) - 1, PAGE_TOKENS - 1 - (p0 % PAGE_TOKENS))
            w = w[: 1 + max(0, extra)]
            if p0 + len(w) >= self.max_model_len:
                return
            req_obj = runner.requests.get(r)
            sp = (
                getattr(req_obj, "sampling_params", None)
                if req_obj is not None
                else None
            )
            if sp is not None:
                pend_inc = pending_out.get(r, 0) if pending_out is not None else 0
                out_len = len(getattr(req_obj, "output_token_ids", ()) or ()) + pend_inc
                if sp.max_tokens is not None and out_len + len(w) >= sp.max_tokens:
                    return
                if not getattr(sp, "ignore_eos", False):
                    if getattr(sp, "stop", None):
                        return
                    stop_ids = getattr(sp, "stop_token_ids", None) or ()
                    pend_toks = emitted.get(r, ()) if emitted is not None else ()
                    if any(
                        t in self._eos_token_ids or t in stop_ids
                        for t in (*pend_toks, *w[:1])
                    ):
                        return
            pos0_list.append(p0)
            windows.append(w)

        from vllm_torchtpu.utils import synchronize_tensors

        pos0 = np.asarray(pos0_list, dtype=np.int32)
        nb, rows, ctrl, row_of = self._build_spec_ctrl(req_ids, pos0, windows)
        ctrl_t = _to_torch(ctrl, self._dev, sync=False)
        op = self._get_spec_step_op(nb, rows, kd, False)
        outs = op(
            *self._w_list,
            *self._dw_list,
            self.kv_sw,
            self.kv_gl,
            self.kv_draft,
            ctrl_t,
            self.kpt,
            self.rank_1d_t,
            self.dev_t,
        )
        self.kv_sw, self.kv_gl, self.kv_draft = outs[1], outs[2], outs[3]
        sw_old, draft_old = outs[-2], outs[-1]
        synchronize_tensors(
            [outs[0], self.kv_sw, self.kv_gl, self.kv_draft, sw_old, draft_old],
            wait=False,
        )
        self.total_sync_calls += 1
        self.last_step_sync_calls = 1
        self.last_step_n_calls = 1
        self._prefetched_step = {
            "is_spec": True,
            "req_ids": req_ids,
            "pos0": pos0,
            "windows": [tuple(w) for w in windows],
            "nb": nb,
            "rows": rows,
            "kd": kd,
            "row_of": row_of,
            "slot_map": slot_map,
            "spec_state": (outs, sw_old, draft_old, ctrl_t),
        }

    def _run_decode_spec(
        self, runner: Any, indices: list[int], scheduler_output: Any
    ) -> dict[str, Any]:
        """One fused DSpark step: verify each request window, then draft the next."""
        from vllm_torchtpu.utils import synchronize_tensors

        req_ids = [runner.input_batch.req_ids[i] for i in indices]
        n = len(req_ids)
        nb = _bucket(n)
        kd = min(SPEC_DRAFT_DEPTH.get(nb, 0), self.spec_k)
        sched = (
            scheduler_output.scheduled_spec_decode_tokens
            if scheduler_output is not None
            else {}
        )
        pos0 = np.asarray(
            [runner.input_batch.num_computed_tokens_cpu[i] for i in indices], np.int32
        )
        windows: list[list[int]] = []
        max_q_req = SPEC_MAX_ROWS // nb
        for b, (i, r) in enumerate(zip(indices, req_ids)):
            st = self.slots.get(r)
            assert st is not None
            bonus = int(runner.input_batch.token_ids_cpu[i, pos0[b]])
            ours = (st.next_verify or [bonus])[1:]
            s = [int(x) for x in sched.get(r, ())]
            # Verify only the scheduled prefix of our drafts; vLLM may trim at length
            # limits.
            m = 0
            while m < len(s) and m < len(ours) and s[m] == ours[m]:
                m += 1
            extra = min(m, max_q_req - 1) if kd > 0 else 0
            # Rows on a new global superpage stream the previous page from HBM while
            # earlier
            # rows still write into it, so a window never crosses a 2048 token page.
            extra = min(extra, PAGE_TOKENS - 1 - int(pos0[b]) % PAGE_TOKENS)
            windows.append([bonus] + ours[:extra])
        rows_n = sum(len(wd) for wd in windows)
        need_lp = runner.input_batch.max_num_logprobs is not None
        tm = self.timer
        if tm is not None:
            tm.mark("mk_host_prep")

        outs = None
        row_of: list[int] = []
        rows = 0
        if self._prefetched_step is not None:
            pref = self._prefetched_step
            if (
                pref.get("is_spec", False)
                and not need_lp
                and pref["req_ids"] == req_ids
                and np.array_equal(pref["pos0"], pos0)
                and pref["windows"] == [tuple(w) for w in windows]
            ):
                self._prefetched_step = None
                outs = pref["spec_state"][0]
                rows = pref["rows"]
                row_of = pref["row_of"]
                self.prefetch_hits += 1
                self.last_step_sync_calls = 1
                self.last_step_n_calls = 1
            else:
                self._flush_prefetched_step()

        if outs is None:
            nb, rows, ctrl, row_of = self._build_spec_ctrl(req_ids, pos0, windows)
            op = self._get_spec_step_op(nb, rows, kd, need_lp)
            outs = op(
                *self._w_list,
                *self._dw_list,
                self.kv_sw,
                self.kv_gl,
                self.kv_draft,
                _to_torch(ctrl, self._dev, sync=False),
                self.kpt,
                self.rank_1d_t,
                self.dev_t,
            )
            self.kv_sw, self.kv_gl, self.kv_draft = outs[1], outs[2], outs[3]
            sync = [
                outs[0],
                self.kv_sw,
                self.kv_gl,
                self.kv_draft,
                outs[-2],
                outs[-1],
            ]
            if need_lp:
                sync += [outs[4], outs[5]]
            synchronize_tensors(sync)
            self.total_sync_calls += 1
            self.last_step_sync_calls = 1
            self.last_step_n_calls = 1

        meta_np = outs[0].cpu().numpy()
        if tm is not None:
            tm.mark("mk_spec_step")
        kd_out = max(1, kd)
        tg = meta_np[:rows]
        n_acc = meta_np[rows : rows + nb]
        drafts = meta_np[rows + nb : rows + nb + nb * kd_out].reshape(nb, kd_out)
        top_i = outs[4].cpu().numpy()[0] if need_lp else None
        top_v = outs[5].cpu().numpy()[0].astype(np.float32) if need_lp else None
        q_next = self._plan_drafts(nb, drafts) if kd else 1

        bt = (
            runner.input_batch.block_table[self._gl_group_id].get_cpu_tensor()
            if self.prefix_cache_enabled
            else None
        )
        save_pairs: list[tuple[int, int]] = []
        wb_indices: list[int] = []
        wb_slots: list[int] = []
        wb_pos: list[int] = []
        res: dict[str, Any] = {}
        for b, (i, r) in enumerate(zip(indices, req_ids)):
            st = self.slots.get(r)
            assert st is not None
            a = int(n_acc[b])
            r0 = row_of[b]
            toks = [int(t) for t in tg[r0 : r0 + a]]
            new_computed = int(pos0[b]) + a
            st.kv_upto = new_computed
            st.draft_upto = new_computed
            st.last_next_tok = toks[-1]
            nxt = [int(x) for x in drafts[b, : q_next - 1]] if kd else []
            st.next_verify = [toks[-1]] + nxt
            if bool(envs.VLLM_TPU_MUSE_MK_EAGER_KV_WRITEBACK):
                for k_off in range(a):
                    wb_indices.append(i)
                    wb_slots.append(int(st.slot))
                    wb_pos.append(int(pos0[b]) + k_off)
            if self.prefix_cache_enabled and bt is not None:
                prev_blocks = st.last_cached_tokens // self.block_size
                new_blocks = new_computed // self.block_size
                for lp in range(prev_blocks, new_blocks):
                    c_sub = self._alloc_cache_subpage_for_rank(
                        lp % self.world, int(bt[i, lp])
                    )
                    if c_sub >= 0:
                        save_pairs.append((self._slot_subpage(st.slot, lp), c_sub))
                st.last_cached_tokens = new_blocks * self.block_size
            req_obj = runner.requests.get(r)
            if req_obj is not None:
                self.notify_coordinator_cache_blocks(req_obj, new_computed)
            if need_lp and top_i is not None and top_v is not None:
                tid = np.asarray(top_i[r0 : r0 + a], np.int32)
                tlp = np.asarray(top_v[r0 : r0 + a], np.float32)
                for k, t in enumerate(toks):
                    if not np.any(tid[k] == t):
                        raise RuntimeError(
                            f"megakernel greedy token {t} missing "
                            f"from its top {tid.shape[1]} for {r}"
                        )
            else:
                tid = tlp = None
            res[r] = (toks, tid, tlp, nxt)
            self.req_mk_tokens[r] = self.req_mk_tokens.get(r, 0) + a
            self.spec_accepted += a
        if wb_indices:
            self._writeback_decoded_kv(
                runner,
                wb_indices,
                np.asarray(wb_slots, dtype=np.int32),
                np.asarray(wb_pos, dtype=np.int32),
            )
        self.spec_steps += 1
        self.spec_req_steps += n
        if tm is not None:
            tm.mark("mk_bookkeeping")
        if save_pairs:
            self.copy_subpages(save_pairs)
        if not need_lp and not self._in_mixed_step:
            self._maybe_prefetch_spec_from_runner(
                runner,
                pending_out={r: len(res[r][0]) for r in req_ids},
                emitted={r: res[r][0] for r in req_ids},
            )
        if tm is not None:
            tm.mark("mk_prefetch_dispatch")
        self.mk_decode_steps += 1
        self.mk_decoded_tokens += int(n_acc[:n].sum())
        if self.rank == 0 and (
            self.spec_steps in (1, 16, 64) or self.spec_steps % 256 == 0
        ):
            logger.info(
                "MUSE_MEGAKERNEL spec stats | steps=%d accepted=%d "
                "mean_accept=%.3f batch=%d rows=%d q_next=%d "
                "prefetch_hits=%d misses=%d",
                self.spec_steps,
                self.spec_accepted,
                self.spec_accepted / max(1, self.spec_req_steps),
                n,
                rows_n,
                q_next,
                self.prefetch_hits,
                self.prefetch_misses,
            )
        return res

    def run_decode(
        self, runner: Any, indices: list[int], scheduler_output: Any = None
    ) -> dict[str, Any]:
        if self.spec:
            nb = _bucket(max(1, len(indices)))
            draft_fresh = nb <= SPEC_MAX_ROWS and all(
                (st := self.slots.get(runner.input_batch.req_ids[i])) is not None
                and st.draft_upto == st.kv_upto
                for i in indices
            )
            if draft_fresh:
                if self._prefetched_step is not None and not self._prefetched_step.get(
                    "is_spec", False
                ):
                    self._flush_prefetched_step()
                return self._run_decode_spec(runner, indices, scheduler_output)
        if self._prefetched_step is not None and self._prefetched_step.get(
            "is_spec", False
        ):
            self._flush_prefetched_step()
        del scheduler_output
        req_ids = [runner.input_batch.req_ids[i] for i in indices]
        slot_list: list[int] = []
        for r in req_ids:
            st_r = self.slots.get(r)
            assert st_r is not None
            slot_list.append(st_r.slot)
        slots = np.asarray(slot_list, np.int32)
        pos = np.asarray(
            [runner.input_batch.num_computed_tokens_cpu[i] for i in indices], np.int32
        )
        toks = np.asarray(
            [
                int(runner.input_batch.token_ids_cpu[i, pos[b]])
                for b, i in enumerate(indices)
            ],
            np.int32,
        )
        need_lp = runner.input_batch.max_num_logprobs is not None
        tm = self.timer
        if tm is not None:
            tm.mark("mk_host_prep")
        nxt, (top_ids_all, top_lps_all) = self.decode_step(
            toks, pos, slots, need_logprobs=need_lp
        )
        if tm is not None:
            tm.mark("mk_decode_step")
        if bool(envs.VLLM_TPU_MUSE_MK_EAGER_KV_WRITEBACK):
            self._writeback_decoded_kv(runner, indices, slots, pos)

        # Advance subpage ring in lockstep on newly completed blocks and notify
        # coordinator.cache_blocks(request, num_computed_tokens) on every decode step.
        bt = (
            runner.input_batch.block_table[self._gl_group_id].get_cpu_tensor()
            if self.prefix_cache_enabled
            else None
        )
        save_pairs: list[tuple[int, int]] = []
        res: dict[str, Any] = {}
        for b, (i, r) in enumerate(zip(indices, req_ids)):
            st = self.slots.get(r)
            new_computed = int(pos[b]) + 1
            tok_out = int(nxt[b])
            if st is not None:
                st.kv_upto = new_computed
                st.last_next_tok = tok_out
                if self.spec:
                    st.next_verify = [tok_out]
                if self.prefix_cache_enabled and bt is not None:
                    prev_blocks = st.last_cached_tokens // self.block_size
                    new_blocks = new_computed // self.block_size
                    for lp in range(prev_blocks, new_blocks):
                        owner_r = lp % self.world
                        c_sub = self._alloc_cache_subpage_for_rank(
                            owner_r, int(bt[i, lp])
                        )
                        if c_sub >= 0:
                            save_pairs.append((self._slot_subpage(st.slot, lp), c_sub))
                    st.last_cached_tokens = new_blocks * self.block_size
            req_obj = runner.requests.get(r)
            if req_obj is not None:
                self.notify_coordinator_cache_blocks(req_obj, new_computed)
            if top_ids_all is not None and top_lps_all is not None:
                top_id = np.asarray(top_ids_all[b], np.int32)[None, :]
                top_lp = np.asarray(top_lps_all[b], np.float32)[None, :]
                if not np.any(top_id[0] == tok_out):
                    raise RuntimeError(
                        f"megakernel greedy token {tok_out} missing "
                        f"from its top {top_id.shape[1]} for {r}"
                    )
            else:
                top_id = None
                top_lp = None
            res[r] = ([tok_out], top_id, top_lp, [] if self.spec else None)
            self.req_mk_tokens[r] = self.req_mk_tokens.get(r, 0) + 1
        if tm is not None:
            tm.mark("mk_bookkeeping")
        if save_pairs:
            self.copy_subpages(save_pairs)
        self._maybe_prefetch_from_runner(runner, pending_out_inc=1)
        if tm is not None:
            tm.mark("mk_prefetch_dispatch")

        self.mk_decode_steps += 1
        self.mk_decoded_tokens += len(indices)
        if self.rank == 0 and (
            self.mk_decode_steps in (1, 16, 64) or self.mk_decode_steps % 64 == 0
        ):
            logger.info(
                "MUSE_MEGAKERNEL stats | mk_decode_steps=%d mk_decoded_tokens=%d "
                "vllm_forward_steps=%d batch=%d active_slots=%d "
                "prefetch_hits=%d misses=%d last_handover_ms=%.2f",
                self.mk_decode_steps,
                self.mk_decoded_tokens,
                self.vllm_steps,
                len(indices),
                len(self.slots.by_req),
                self.prefetch_hits,
                self.prefetch_misses,
                self.last_handover_ms,
            )
        return res
