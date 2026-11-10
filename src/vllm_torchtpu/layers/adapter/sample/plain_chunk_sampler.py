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

Consolidates grammar masking, post-grammar ``raw_logprobs`` capture, stateful
and stateless ``PlainChunkProcessor`` hooks (``min_tokens``,
``thinking_token_budget``, and future ``LogitsProcessor``s), and greedy /
stochastic sampling for both plain decode steps and draft-less chunks inside
speculative steps.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

import torch

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


class PlainChunkProcessor(Protocol):
    """Per-chunk pre-sample mask and post-sample state update hook."""

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor: ...

    def on_sampled(self, selected_token_ids: torch.Tensor) -> None: ...

    def override_top_tokens(self, selected_token_ids: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class StatelessPlainChunkProcessor:
    """Adapter wrapping a stateless ``logits -> logits`` mask function."""

    mask_fn: LogitsProcessorFn

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return self.mask_fn(logits)

    def on_sampled(self, selected_token_ids: torch.Tensor) -> None:
        return None

    def override_top_tokens(self, selected_token_ids: torch.Tensor) -> torch.Tensor:
        return selected_token_ids


def sample_plain_chunk(
    logits: torch.Tensor,
    *,
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
    processors: Sequence[PlainChunkProcessor] = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask and sample a non-verify chunk ``[cur_start_idx, cur_end_idx)``.

    Returns ``(selected_token_ids, logprob_logits)``. Matching upstream vLLM
    (``apply_grammar_bitmask`` in-place followed by ``Sampler.raw_logprobs``
    before ``logits_processors``), ``logprob_logits`` is captured immediately
    after ``grammar_mask_fn`` and before ``processors`` run.
    """
    if grammar_mask_fn is not None:
        logits = grammar_mask_fn(logits)
    # Upstream vLLM applies grammar bitmasks in-place before Sampler computes
    # raw_logprobs, and runs logits processors (min_tokens, thinking_budget,
    # penalties) afterward.
    logprob_logits = logits

    with sample_stage:
        if top_tokens:
            selected = logits
            for proc in processors:
                selected = proc.override_top_tokens(selected)
        else:
            for proc in processors:
                logits = proc.mask_logits(logits)
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
                selected = sample_from_logits_fn(
                    logits,
                    temperatures_tpu,
                    u,
                    top_k_tpu,
                    top_p_tpu,
                    all_greedy=False,
                    topk_candidates=topk_candidates,
                    apply_topk_topp=apply_topk_topp,
                )
            for proc in processors:
                proc.on_sampled(selected)
    return selected, logprob_logits
