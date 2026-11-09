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
"""Tests for the Muse Spark Open 816B A42B single-dispatch Pallas decode megakernel."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.muse_megakernel import (
    KV_GROUP,
    TP_AXIS_NAME,
    MuseMegakernelConfig,
    canonical_test_weights,
    empty_hybrid_kv_cache,
    logical_mesh,
    make_hybrid_decode_step,
)
from vllm_torchtpu.models.vllm.muse_spark.megakernel_fastpath import MuseDecodeFastPath


def test_prefix_cache_kpages_and_lockstep_cursor():
    """Verify 100% cache_kpages coverage, lockstep subpage ring, and
    coordinator.cache_blocks notification.
    """
    fp = object.__new__(MuseDecodeFastPath)
    fp.world = 8
    fp.rank = 0
    fp.block_size = 128
    fp.n_slots = 16
    fp.max_kp = 64
    fp.base_kpages = fp.n_slots * fp.max_kp + 1
    fp.cache_kpages = 0
    fp.prefix_cache_enabled = True
    fp._rank_cached_blocks = [{} for _ in range(fp.world)]
    fp._rank_subpage_to_block = [{} for _ in range(fp.world)]
    fp._rank_cache_cursors = [0] * fp.world

    # 1. Verify cache_kpages * subpages_per_kpage >= total_vllm_kv_blocks_per_rank
    for total_blocks in (500, 2048, 4097):
        fp._configure_cache_kpages(total_blocks)
        assert fp.cache_kpages * KV_GROUP >= total_blocks

    # 2. Verify cross-rank subpage ring cursor advances in strict lockstep across all
    # ranks
    fp2 = object.__new__(MuseDecodeFastPath)
    fp2.world = 8
    fp2.rank = 3
    fp2.block_size = 128
    fp2.n_slots = 16
    fp2.max_kp = 64
    fp2.base_kpages = fp2.n_slots * fp2.max_kp + 1
    fp2.cache_kpages = 0
    fp2.prefix_cache_enabled = True
    fp2._rank_cached_blocks = [{} for _ in range(fp2.world)]
    fp2._rank_subpage_to_block = [{} for _ in range(fp2.world)]
    fp2._rank_cache_cursors = [0] * fp2.world
    fp2._configure_cache_kpages(1024)
    fp._configure_cache_kpages(1024)

    for step_idx in range(32):
        owner_r = step_idx % 8
        blk_id = 100 + step_idx
        sub0 = fp._alloc_cache_subpage_for_rank(owner_r, blk_id)
        sub3 = fp2._alloc_cache_subpage_for_rank(owner_r, blk_id)
        assert sub0 == sub3
        assert fp._rank_cache_cursors == fp2._rank_cache_cursors
        assert fp._rank_cached_blocks == fp2._rank_cached_blocks

    # 3. Verify coordinator.cache_blocks(request, num_computed_tokens) is called on
    # decode steps
    calls = []
    coordinator = SimpleNamespace(
        cache_blocks=lambda req, n_tok: calls.append((req, n_tok))
    )
    fp.runner = SimpleNamespace(
        scheduler=SimpleNamespace(
            kv_cache_manager=SimpleNamespace(coordinator=coordinator)
        )
    )
    dummy_req = SimpleNamespace(req_id="req_0")
    fp.notify_coordinator_cache_blocks(dummy_req, 256)
    assert calls == [(dummy_req, 256)]


@pytest.mark.multichip
def test_hybrid_decode_step_single_custom_call_and_128k_streaming():
    """Verify single tpu_custom_call lowering and hybrid KV execution at 1K and 120K
    positions.
    """
    if jax.default_backend() != "tpu" or len(jax.devices()) != 8:
        pytest.skip("Requires 8 TPU TensorCores")
    c = MuseMegakernelConfig(
        dim=8192,
        layers=4,
        heads=128,
        kv_heads=16,
        head_dim=64,
        vocab=202048,
        moe_hidden=4096,
        moe_intermediate=4096,
        experts=256,
        top_k=8,
        full_attention=(1,),
        bf16_expert_layers=(0,),
    )
    mesh = logical_mesh(8)
    tp = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(TP_AXIS_NAME))
    rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    max_slots = 16
    max_kp = 64  # 64 * 2048 = 131,072 tokens (128K)
    num_kpages = max_slots * max_kp + 2

    w = canonical_test_weights(mesh, c, seed=42)
    kv = empty_hybrid_kv_cache(
        mesh, c, max_slots=max_slots, num_kpages=num_kpages, max_kp=max_kp
    )

    for b in (1, 2, 4, 8, 16):
        step = make_hybrid_decode_step(
            mesh,
            c,
            max_slots=max_slots,
            num_kpages=num_kpages,
            max_kp=max_kp,
            batch_size=b,
        )
        w_shapes = {
            k: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=tp) for k, v in w.items()
        }
        sw_s = jax.ShapeDtypeStruct(kv["kv_sw"].shape, kv["kv_sw"].dtype, sharding=tp)
        gl_s = jax.ShapeDtypeStruct(kv["kv_gl"].shape, kv["kv_gl"].dtype, sharding=tp)
        tok_s = jax.ShapeDtypeStruct((b,), jnp.int32, sharding=rep)
        ctrl_s = jax.ShapeDtypeStruct((2, b), jnp.int32, sharding=rep)
        kpt_s = jax.ShapeDtypeStruct(kv["kpt"].shape, jnp.int32, sharding=rep)
        hlo = step.lower(w_shapes, sw_s, gl_s, tok_s, ctrl_s, kpt_s).as_text()
        assert hlo.count("tpu_custom_call") == 1, (
            f"Expected 1 tpu_custom_call for batch={b}"
        )

    step1 = make_hybrid_decode_step(
        mesh,
        c,
        max_slots=max_slots,
        num_kpages=num_kpages,
        max_kp=max_kp,
        batch_size=1,
    )
    for pos in (0, 1024, 4096, 32768, 122880):
        tok = jax.device_put(np.asarray([100], np.int32), rep)
        ctrl = jax.device_put(np.asarray([[pos], [0]], np.int32), rep)
        targets, aux, kv["kv_sw"], kv["kv_gl"] = step1(
            w, kv["kv_sw"], kv["kv_gl"], tok, ctrl, kv["kpt"]
        )
        assert targets.shape == (1,)
        assert np.all(np.isfinite(np.asarray(aux, dtype=np.float32)))


@pytest.mark.multichip
def test_hybrid_decode_head_covers_full_vocab_and_returns_logits():
    """The last vocab column of the last shard is reachable; return_logits agrees with
    the
    argmax.
    """
    if jax.default_backend() != "tpu" or len(jax.devices()) != 8:
        pytest.skip("Requires 8 TPU TensorCores")
    c = MuseMegakernelConfig(layers=2, full_attention=(1,), bf16_expert_layers=(0,))
    mesh = logical_mesh(8)
    tp = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(TP_AXIS_NAME))
    rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    vshard = c.vocab // 8
    w = canonical_test_weights(mesh, c, seed=7)
    kv = empty_hybrid_kv_cache(mesh, c, max_slots=1, num_kpages=3, max_kp=2)
    step = make_hybrid_decode_step(
        mesh, c, max_slots=1, num_kpages=3, max_kp=2, batch_size=1, return_logits=True
    )
    shapes = [
        {k: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=tp) for k, v in w.items()},
        jax.ShapeDtypeStruct(kv["kv_sw"].shape, kv["kv_sw"].dtype, sharding=tp),
        jax.ShapeDtypeStruct(kv["kv_gl"].shape, kv["kv_gl"].dtype, sharding=tp),
        jax.ShapeDtypeStruct((1,), jnp.int32, sharding=rep),
        jax.ShapeDtypeStruct((2, 1), jnp.int32, sharding=rep),
        jax.ShapeDtypeStruct(kv["kpt"].shape, jnp.int32, sharding=rep),
    ]
    assert step.lower(*shapes).as_text().count("tpu_custom_call") == 1

    tok = jax.device_put(np.asarray([100], np.int32), rep)
    ctrl = jax.device_put(np.asarray([[0], [0]], np.int32), rep)
    targets, aux, kv["kv_sw"], kv["kv_gl"], logits = step(
        w, kv["kv_sw"], kv["kv_gl"], tok, ctrl, kv["kpt"]
    )
    logits = np.asarray(logits)
    assert logits.shape == (1, c.vocab)
    assert np.all(np.abs(logits) <= c.soft_cap)
    assert int(np.asarray(targets)[0]) == int(np.argmax(logits[0]))

    # Align the head column of token vocab - 1 (rank 7, last real column) with the final
    # hidden state.
    hidden = np.asarray(aux, np.float32).reshape(-1)[: c.dim]
    column = jnp.asarray(hidden / np.linalg.norm(hidden) * 4.0, jnp.bfloat16)
    w["lmw"] = jax.jit(lambda a: a.at[7, :, vshard - 1].set(column), out_shardings=tp)(
        w["lmw"]
    )
    targets, _, kv["kv_sw"], kv["kv_gl"], logits = step(
        w, kv["kv_sw"], kv["kv_gl"], tok, ctrl, kv["kpt"]
    )
    assert int(np.asarray(targets)[0]) == c.vocab - 1
    assert int(np.argmax(np.asarray(logits)[0])) == c.vocab - 1


@pytest.mark.multichip
def test_hybrid_decode_b1_b8_b16_bit_exact():
    """Verify batch_size=1, 8, 16 (with force_rows8=False) produce bit-identical
    hidden states and logits.
    """
    if jax.default_backend() != "tpu" or len(jax.devices()) != 8:
        pytest.skip("Requires 8 TPU TensorCores")
    c = MuseMegakernelConfig(layers=2, full_attention=(1,), bf16_expert_layers=(0,))
    mesh = logical_mesh(8)
    rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    w = canonical_test_weights(mesh, c, seed=11)
    max_slots = 16
    max_kp = 4
    num_kpages = max_slots * max_kp + 2

    outs_by_b = {}
    for b in (1, 8, 16):
        kv = empty_hybrid_kv_cache(
            mesh, c, max_slots=max_slots, num_kpages=num_kpages, max_kp=max_kp
        )
        step = make_hybrid_decode_step(
            mesh,
            c,
            max_slots=max_slots,
            num_kpages=num_kpages,
            max_kp=max_kp,
            batch_size=b,
            force_rows8=False,
            return_logits=True,
        )
        tok = jax.device_put(np.full((b,), 321, np.int32), rep)
        ctrl = np.full((2, b), -1, np.int32)
        ctrl[0, 0] = 0
        ctrl[1, :] = np.arange(b, dtype=np.int32) % max_slots
        ctrl_d = jax.device_put(ctrl, rep)
        targets, aux, _, _, logits = step(
            w, kv["kv_sw"], kv["kv_gl"], tok, ctrl_d, kv["kpt"]
        )
        outs_by_b[b] = (
            int(np.asarray(targets)[0]),
            np.asarray(aux, np.float32)[0],
            np.asarray(logits, np.float32)[0],
        )

    assert outs_by_b[1][0] == outs_by_b[8][0] == outs_by_b[16][0]
    np.testing.assert_array_equal(outs_by_b[1][1], outs_by_b[8][1])
    np.testing.assert_array_equal(outs_by_b[1][1], outs_by_b[16][1])
    np.testing.assert_array_equal(outs_by_b[1][2], outs_by_b[8][2])
    np.testing.assert_array_equal(outs_by_b[1][2], outs_by_b[16][2])

    # Also verify spec grouped num_reqs=1, batch_size=8 matches sequential b=1
    kv_seq = empty_hybrid_kv_cache(
        mesh, c, max_slots=max_slots, num_kpages=num_kpages, max_kp=max_kp
    )
    step1 = make_hybrid_decode_step(
        mesh,
        c,
        max_slots=max_slots,
        num_kpages=num_kpages,
        max_kp=max_kp,
        batch_size=1,
        return_logits=True,
    )
    seq_toks = [101, 202, 303, 404]
    seq_logits = []
    for k_idx, tk in enumerate(seq_toks):
        tok_1 = jax.device_put(np.asarray([tk], np.int32), rep)
        ctrl_1 = jax.device_put(np.asarray([[4096 + k_idx], [0]], np.int32), rep)
        _, _, kv_seq["kv_sw"], kv_seq["kv_gl"], lg_1 = step1(
            w, kv_seq["kv_sw"], kv_seq["kv_gl"], tok_1, ctrl_1, kv_seq["kpt"]
        )
        seq_logits.append(np.asarray(lg_1, np.float32)[0])

    kv_grp = empty_hybrid_kv_cache(
        mesh, c, max_slots=max_slots, num_kpages=num_kpages, max_kp=max_kp
    )
    step_grp = make_hybrid_decode_step(
        mesh,
        c,
        max_slots=max_slots,
        num_kpages=num_kpages,
        max_kp=max_kp,
        batch_size=8,
        num_reqs=1,
        return_logits=True,
    )
    tok_8 = np.full((8,), seq_toks[0], np.int32)
    tok_8[:4] = seq_toks
    ctrl_8 = np.full((2, 8), -1, np.int32)
    ctrl_8[0, :4] = [4096 + k_idx for k_idx in range(4)]
    ctrl_8[1, :] = 0
    _, _, _, _, lg_8 = step_grp(
        w,
        kv_grp["kv_sw"],
        kv_grp["kv_gl"],
        jax.device_put(tok_8, rep),
        jax.device_put(ctrl_8, rep),
        kv_grp["kpt"],
    )
    lg_8_np = np.asarray(lg_8, np.float32)
    for k_idx in range(4):
        np.testing.assert_array_equal(seq_logits[k_idx], lg_8_np[k_idx])


@pytest.mark.tpu_test
def test_ring_rows_coalesced_read_write_bit_exact():
    """Verify coalesced 16-row block DMAs in ring_rows_read/write_local."""
    if jax.default_backend() != "tpu" or len(jax.devices()) < 1:
        pytest.skip("Requires TPU")
    from vllm_torchtpu.kernels.muse_megakernel.fastpath_ops import (
        ring_rows_read_local,
        ring_rows_write_local,
    )

    dev0 = jax.devices()[0]
    rng = np.random.default_rng(123)
    kv_np = rng.standard_normal((1, 5, 4, 64, 256), dtype=np.float32)
    kv_j = jax.device_put(jnp.asarray(kv_np, dtype=jnp.bfloat16), dev0)
    kv_ref = np.asarray(kv_j, dtype=np.float32).copy()

    # Cross 16-row boundaries, switch slots, include duplicates and masked-out rows.
    slots = np.asarray([0, 0, 0, 0, 1, 1, 0, 2], dtype=np.int32)
    ridx = np.asarray([14, 15, 16, 17, 3, 3, 15, 31], dtype=np.int32)
    mask = np.asarray([1, 1, 0, 1, 1, 1, 1, 0], dtype=np.int32)

    got_rd = np.asarray(
        jax.jit(ring_rows_read_local)(
            kv_j, jax.device_put(slots, dev0), jax.device_put(ridx, dev0)
        ),
        dtype=np.float32,
    )
    exp_rd = np.stack(
        [kv_ref[0, :, slots[j], ridx[j], :] for j in range(len(slots))], axis=1
    )[None]
    np.testing.assert_array_equal(got_rd, exp_rd)

    vals_np = np.asarray(
        jnp.asarray(
            rng.standard_normal((1, 5, len(slots), 256), dtype=np.float32),
            dtype=jnp.bfloat16,
        ),
        dtype=np.float32,
    )
    kv_out = jax.jit(ring_rows_write_local)(
        kv_j,
        jax.device_put(vals_np, dev0),
        jax.device_put(slots, dev0),
        jax.device_put(ridx, dev0),
        jax.device_put(mask, dev0),
    )
    exp_kv = kv_ref.copy()
    for j in range(len(slots)):
        if mask[j] != 0:
            exp_kv[0, :, slots[j], ridx[j], :] = vals_np[0, :, j, :]
    np.testing.assert_array_equal(np.asarray(kv_out, dtype=np.float32), exp_kv)

    # All-zero mask must leave KV untouched.
    kv_zero = jax.jit(ring_rows_write_local)(
        kv_out,
        jax.device_put(vals_np, dev0),
        jax.device_put(slots, dev0),
        jax.device_put(ridx, dev0),
        jax.device_put(np.zeros_like(mask), dev0),
    )
    np.testing.assert_array_equal(np.asarray(kv_zero, dtype=np.float32), exp_kv)


def test_megakernel_config_disables_async_scheduling_and_validates_tp8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_torchtpu.models.vllm.muse_spark import megakernel_fastpath as mk_fp
    from vllm_torchtpu.models.vllm.muse_spark import text_model as tm

    monkeypatch.setenv("VLLM_TPU_MUSE_MEGAKERNEL", "1")
    for initial_async in (None, True):
        vllm_cfg = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace()),
            cache_config=SimpleNamespace(
                block_size=128, user_specified_block_size=True
            ),
            parallel_config=SimpleNamespace(tensor_parallel_size=8),
            scheduler_config=SimpleNamespace(async_scheduling=initial_async),
            speculative_config=None,
        )
        tm.TPUMuseSparkConfig.verify_and_update_config(vllm_cfg)
        assert vllm_cfg.scheduler_config.async_scheduling is False

    bad_tp_cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        cache_config=SimpleNamespace(block_size=128, user_specified_block_size=True),
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
        scheduler_config=SimpleNamespace(async_scheduling=None),
        speculative_config=None,
    )
    with pytest.raises(NotImplementedError, match="tensor_parallel_size=8"):
        tm.TPUMuseSparkConfig.verify_and_update_config(bad_tp_cfg)

    assert [mk_fp.resolve_max_slots(n) for n in (1, 3, 8, 12, 32, 256)] == [
        1,
        4,
        8,
        16,
        32,
        32,
    ]
    assert mk_fp.resolve_max_slots(64, spec=True) == 32
