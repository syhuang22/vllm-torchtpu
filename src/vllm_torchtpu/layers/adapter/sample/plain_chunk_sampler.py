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
"""Shared plain-chunk logits-masking and sampling pipeline.

Consolidates raw-logprob capture, grammar / ``min_tokens`` / future
``LogitsProcessor`` masking, ``thinking_token_budget`` forcing, and greedy /
stochastic sampling for both plain decode steps and draft-less chunks inside
speculative steps.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from vllm_torchtpu.layers.adapter.sample.thinking_budget_tracker import (
        ThinkingBudgetStepState,
    )

type LogitsProcessorFn = Callable[[torch.Tensor], torch.Tensor]
type SamplingParamsBuilderFn = Callable[
    [int, int, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]


class SampleFromLogitsFn(Protocol):
    def __call__(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        u: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        all_greedy: bool = ...,
        topk_candidates: int = ...,
        apply_topk_topp: bool = ...,
    ) -> torch.Tensor: ...


def sample_plain_chunk(
    logits: torch.Tensor,
    *,
    batch_req_ids: Sequence[str | None],
    cur_start_idx: int,
    cur_end_idx: int,
    top_tokens: bool,
    all_greedy: bool,
    sampling_generator: torch.Generator | None,
    sample_stage: AbstractContextManager[object],
    greedy_sample_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    sample_from_logits_fn: SampleFromLogitsFn | None = None,
    build_sampling_params_fn: SamplingParamsBuilderFn | None = None,
    topk_candidates: int = 0,
    apply_topk_topp: bool = True,
    grammar_mask_fn: LogitsProcessorFn | None = None,
    logits_processors: Sequence[LogitsProcessorFn] = (),
    tb_step: ThinkingBudgetStepState | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask and sample a non-verify chunk ``[cur_start_idx, cur_end_idx)``.

    Returns ``(selected_token_ids, logprob_logits)``, preserving the unmasked
    ``logprob_logits`` tensor for ``gather_logprobs`` (matching upstream vLLM's
    default ``raw_logprobs`` behavior).
    """
    logprob_logits = logits
    if grammar_mask_fn is not None:
        logits = grammar_mask_fn(logits)
    if not top_tokens:
        for proc in logits_processors:
            logits = proc(logits)

    with sample_stage:
        if top_tokens:
            selected = logits
            if tb_step is not None:
                selected = tb_step.override_and_advance_plain(
                    logits.device,
                    batch_req_ids,
                    cur_start_idx,
                    cur_end_idx,
                    selected,
                )
        else:
            tb_slots = (
                tb_step.chunk_slot_ids(
                    logits.device,
                    batch_req_ids,
                    cur_start_idx,
                    cur_end_idx,
                )
                if tb_step is not None
                else None
            )
            if tb_slots is not None:
                assert tb_step is not None
                logits = tb_step.mask_plain_logits(logits, tb_slots)
            if all_greedy:
                assert greedy_sample_fn is not None
                selected = greedy_sample_fn(logits)
            else:
                assert (
                    build_sampling_params_fn is not None
                    and sample_from_logits_fn is not None
                )
                temperatures_tpu, top_k_tpu, top_p_tpu = build_sampling_params_fn(
                    cur_start_idx, cur_end_idx, logits
                )
                u = torch.rand_like(
                    logits,
                    dtype=torch.float32,
                    generator=sampling_generator,
                )
                selected = sample_from_logits_func_call(
                    sample_from_logits_fn,
                    logits,
                    temperatures_tpu,
                    u,
                    top_k_tpu,
                    top_p_tpu,
                    topk_candidates=topk_candidates,
                    apply_topk_topp=apply_topk_topp,
                )
            if tb_slots is not None:
                assert tb_step is not None
                tb_step.advance_plain(tb_slots, selected)
    return selected, logprob_logits


def sample_from_logits_func_call(
    sample_from_logits_fn: SampleFromLogitsFn,
    logits: torch.Tensor,
    temperatures_tpu: torch.Tensor,
    u: torch.Tensor,
    top_k_tpu: torch.Tensor,
    top_p_tpu: torch.Tensor,
    *,
    topk_candidates: int,
    apply_topk_topp: bool,
) -> torch.Tensor:
    return sample_from_logits_fn(
        logits,
        temperatures_tpu,
        u,
        top_k_tpu,
        top_p_tpu,
        all_greedy=False,
        topk_candidates=topk_candidates,
        apply_topk_topp=apply_topk_topp,
    )
