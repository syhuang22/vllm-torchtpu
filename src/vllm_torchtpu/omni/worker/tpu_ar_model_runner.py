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
"""TPU model runner for autoregressive vLLM-Omni stages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.multimodal.utils import (
    copy_mm_embedding_modality,
    set_mm_embedding_modality,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import ModelRunnerOutput
from vllm_omni.data_entry_keys import flatten_payload
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVTransferManager,
)
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
    needs_omni_connector,
)

from vllm_torchtpu.omni.worker.tpu_model_runner import (
    OmniTPUModelRunner,
    _unwrap_model,
)
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUModelRunnerOutput
from vllm_torchtpu.utils import synchronize_tensors

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput


def _decode_additional_info(info: Any) -> dict[str, Any]:
    if not info:
        return {}
    if isinstance(info, dict):
        return info
    decoded = deserialize_additional_information(info)
    return decoded if isinstance(decoded, dict) else {}


def _slice_inter_stage_payload(
    raw_mm: dict[str, Any],
    req_idx: int,
    tok_start: int | None,
    tok_end: int | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, val in raw_mm.items():
        if key == "hidden_states" and isinstance(val, dict) and "layers" in val:
            layers = {}
            for lk, lt in val["layers"].items():
                if isinstance(lt, torch.Tensor):
                    sliced = (
                        lt[tok_start:tok_end]
                        if tok_start is not None and tok_end is not None
                        else lt
                    )
                    layers[lk] = sliced.detach().cpu()
                else:
                    layers[lk] = lt
            payload["hidden_states"] = {"layers": layers}
        elif key == "codes" and isinstance(val, dict):
            codes = {}
            for ck, cv in val.items():
                if isinstance(cv, torch.Tensor):
                    sliced = (
                        cv[tok_start:tok_end]
                        if tok_start is not None
                        and tok_end is not None
                        and cv.shape[0] >= tok_end
                        else cv
                    )
                    codes[ck] = sliced.detach().cpu()
                else:
                    codes[ck] = cv
            payload["codes"] = codes
        elif key == "embed" and isinstance(val, dict):
            embed = {}
            for ek, ev in val.items():
                item = ev[req_idx] if isinstance(ev, list) and req_idx < len(ev) else ev
                embed[ek] = (
                    item.detach().cpu() if isinstance(item, torch.Tensor) else item
                )
            payload["embed"] = embed
        elif isinstance(val, torch.Tensor):
            payload[key] = val.detach().cpu()
        else:
            payload[key] = val
    return payload


class TPUARModelRunner(OmniTPUModelRunner, OmniConnectorModelRunnerMixin):
    """TPU runner for autoregressive stages; counterpart of ``GPUARModelRunner``.

    Serves the Thinker and Talker stages on TPU, wiring inter-stage Omni
    connectors (with ``OmniKVTransferManager``), multimodal mixed-dimension
    chunk slicing, and the Talker's ``preprocess``, ``talker_mtp``,
    ``postprocess``, and ``make_omni_output`` hooks.
    """

    SUPPORTED_STAGES = frozenset({"thinker", "talker"})
    _omni_connector_initialized: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._omni_mm_embeds_list: list[torch.Tensor] = []
        self._omni_chunk_req_slice: tuple[int, int] | None = None
        self._omni_last_hidden_states: torch.Tensor | IntermediateTensors | None = None
        self._omni_last_aux_hidden_states: list[torch.Tensor] | None = None
        self.kv_transfer_manager: OmniKVTransferManager | None = None
        if (
            self.model_stage is not None
            and self.model_stage not in self.SUPPORTED_STAGES
        ):
            raise NotImplementedError(
                f"Stage {self.stage_id} ({self.model_stage!r}) is not supported on "
                f"TPU: TPUARModelRunner serves {sorted(self.SUPPORTED_STAGES)} only."
            )
        if needs_omni_connector(self.model_config):
            self.kv_transfer_manager = OmniKVTransferManager.from_vllm_config(
                self.vllm_config, self.model_config
            )
            self.init_omni_connectors(
                model_config=self.model_config,
                kv_transfer_manager=self.kv_transfer_manager,
            )

    def _update_states(self, scheduler_output: SchedulerOutput) -> bool:
        """Release connector and buffer state for finished requests."""
        for req_id in scheduler_output.finished_req_ids:
            self.model_intermediate_buffer.pop(req_id, None)
            if self._omni_connector_initialized:
                self.cleanup_finished_request(req_id)
        res = super()._update_states(scheduler_output)
        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            existing = self.model_intermediate_buffer.setdefault(req_id, {})
            self._update_intermediate_buffer(
                req_id,
                _decode_additional_info(
                    getattr(new_req, "additional_information", None)
                ),
            )
            self._update_intermediate_buffer(
                req_id,
                _decode_additional_info(
                    getattr(new_req, "model_intermediate_buffer", None)
                ),
            )
            reqs_map = getattr(self, "requests", None)
            req_state = reqs_map.get(req_id) if isinstance(reqs_map, dict) else None
            if req_state is not None:
                req_state.additional_information_cpu = existing
        cached_reqs = scheduler_output.scheduled_cached_reqs
        cached_infos = getattr(cached_reqs, "additional_information", None)
        if isinstance(cached_infos, dict):
            for req_id in cached_reqs.req_ids:
                self._update_intermediate_buffer(
                    req_id, _decode_additional_info(cached_infos.get(req_id))
                )
        return res

    def _prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        start_index: int,
        num_decode_reqs: int,
        num_windowed_reqs: int | None = None,
    ) -> Any:
        res = super()._prepare_inputs(
            scheduler_output, start_index, num_decode_reqs, num_windowed_reqs
        )
        self._omni_chunk_req_slice = (start_index, res[4])
        return res

    def _chunk_req_spans(self) -> list[tuple[str, int, int, int]]:
        input_batch = self.input_batch
        query_start_loc = self.query_start_loc_np
        num_reqs = input_batch.num_reqs
        start_idx, end_idx = self._omni_chunk_req_slice or (0, num_reqs)
        start_idx = min(start_idx, num_reqs)
        end_idx = min(end_idx, num_reqs)
        spans: list[tuple[str, int, int, int]] = []
        for r_i, req_id in enumerate(input_batch.req_ids[start_idx:end_idx]):
            if req_id is None or r_i + 1 >= len(query_start_loc):
                continue
            spans.append(
                (
                    req_id,
                    start_idx + r_i,
                    int(query_start_loc[r_i]),
                    int(query_start_loc[r_i + 1]),
                )
            )
        return spans

    def _gather_mm_embeddings(
        self,
        scheduler_output: SchedulerOutput,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        # Audio embeddings (width 2048) and DeepStack vision embeddings (width
        # 8192) cannot be concatenated along dim 0 by TPUModelRunner.execute_model.
        # Cache the heterogeneous list and return a 1D row-index range into
        # the cached embeddings so execute_model slices row indices per
        # prefill chunk.
        mm_embeds, is_multimodal = super()._gather_mm_embeddings(
            scheduler_output, *args, **kwargs
        )
        self._omni_mm_embeds_list = list(mm_embeds)
        if mm_embeds:
            total = sum(embed.shape[0] for embed in mm_embeds)
            return [torch.arange(total, dtype=torch.int64)], is_multimodal
        return mm_embeds, is_multimodal

    def capture_model(self) -> None:
        super().capture_model()
        if (
            self.enforce_eager
            or not self.supports_mm_inputs
            or not self._pp_is_first
            or self.model_stage not in (None, "thinker")
        ):
            return
        self._precompile_mm_inputs()

    @torch.no_grad()
    def _precompile_mm_inputs(self) -> None:
        if self.mm_budget is None or not self.mm_budget.mm_max_toks_per_item:
            return
        # Sort modalities so every TP worker runs the same encoder (and its TP
        # collectives) in lockstep; MultiModalBudget builds its dict from a set
        # whose iteration order varies across processes with PYTHONHASHSEED.
        modalities = sorted(self.mm_budget.mm_max_toks_per_item)
        if "image" in modalities and "video" in modalities:
            # Image and video share the vision tower; warming "image" compiles
            # the max patch bucket in one pass instead of synthesizing and
            # chunking a full max_model_len dummy video.
            modalities.remove("video")
        with self._precompile_timed("multimodal encoders and merge"):
            warmed_embeds: list[tuple[str, torch.Tensor]] = []
            for modality in modalities:
                batched_dummy_mm_inputs = self._get_mm_dummy_batch(modality, 1)
                dummy_encoder_outputs = self.model.embed_multimodal(
                    **batched_dummy_mm_inputs
                )
                if not dummy_encoder_outputs:
                    continue
                outputs_list = list(dummy_encoder_outputs)
                synchronize_tensors(outputs_list)
                warmed_embeds.append((modality, outputs_list[0]))
            self.mm_budget.reset_cache()
            if not warmed_embeds:
                return
            token_buckets = [n for n in self.num_tokens_paddings if n > 0]
            for modality, sample_embed in warmed_embeds:
                dummy_mm = set_mm_embedding_modality(
                    torch.zeros_like(sample_embed[:1]), modality
                )
                for num_tokens in token_buckets:
                    dummy_input_ids = torch.zeros(
                        (num_tokens,), dtype=torch.int32, device=self.device
                    )
                    is_multimodal = torch.zeros((num_tokens,), dtype=torch.bool)
                    is_multimodal[0] = True
                    _, inputs_embeds = super()._get_model_inputs(
                        dummy_input_ids, ([dummy_mm], is_multimodal)
                    )
                    if inputs_embeds is not None:
                        synchronize_tensors(inputs_embeds)
            # Clear any deepstack_input_embeds state left populated by the
            # vision merge warmup above before serving starts.
            dummy_input_ids = torch.zeros(
                (token_buckets[0],), dtype=torch.int32, device=self.device
            )
            _, inputs_embeds = super()._get_model_inputs(dummy_input_ids, None)
            if inputs_embeds is not None:
                synchronize_tensors(inputs_embeds)

    def _get_model_inputs(
        self,
        input_ids: torch.Tensor,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if mm_embed_inputs is not None:
            chunk_embeds, is_multimodal = mm_embed_inputs
            mm_embeds = self._omni_mm_embeds_list
            if chunk_embeds and mm_embeds and chunk_embeds[0].ndim == 1:
                chunk_start = int(chunk_embeds[0][0])
                chunk_end = chunk_start + chunk_embeds[0].shape[0]
                sliced_embeds: list[torch.Tensor] = []
                offset = 0
                for embed in mm_embeds:
                    rows = embed.shape[0]
                    start = max(0, chunk_start - offset)
                    end = min(rows, chunk_end - offset)
                    if start < end:
                        sliced = embed[start:end]
                        copy_mm_embedding_modality(embed, sliced)
                        sliced_embeds.append(sliced)
                    offset += rows
                if chunk_end >= offset:
                    self._omni_mm_embeds_list.clear()
                mm_embed_inputs = (sliced_embeds, is_multimodal)
        model_input_ids, inputs_embeds = super()._get_model_inputs(
            input_ids, mm_embed_inputs
        )
        orig_model = _unwrap_model(self.model)
        if getattr(orig_model, "has_preprocess", False):
            if self._omni_connector_initialized and not getattr(
                self.model_config, "async_chunk", False
            ):
                self._sync_local_stage_payloads()
            preprocess_ids = (
                model_input_ids if model_input_ids is not None else input_ids
            )
            if inputs_embeds is None and preprocess_ids is not None:
                embed_fn = getattr(
                    orig_model,
                    "embed_input_ids",
                    getattr(self.model, "embed_input_ids", None),
                )
                if callable(embed_fn):
                    inputs_embeds = embed_fn(preprocess_ids)
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds.clone()
            num_computed_arr = self.input_batch.num_computed_tokens_cpu
            mtp_fn = getattr(orig_model, "talker_mtp", None)
            reqs_map = getattr(self, "requests", None)
            for req_id, req_idx, s, e in self._chunk_req_spans():
                if s >= e:
                    continue
                req_info = self.model_intermediate_buffer.setdefault(req_id, {})
                num_computed = (
                    int(num_computed_arr[req_idx])
                    if req_idx < len(num_computed_arr)
                    else 0
                )
                req_state = reqs_map.get(req_id) if isinstance(reqs_map, dict) else None
                prompt_token_ids = (
                    getattr(req_state, "prompt_token_ids", None)
                    if req_state is not None
                    else None
                )
                prompt_len = len(prompt_token_ids) if prompt_token_ids else 0
                is_prefill = (
                    (num_computed < prompt_len)
                    if prompt_len > 0
                    else (num_computed == 0)
                )
                req_info["_omni_prompt_len"] = prompt_len
                req_info["_omni_num_computed_tokens"] = num_computed
                req_info["_omni_is_prefill"] = is_prefill
                r_ids = preprocess_ids[s:e] if preprocess_ids is not None else None
                r_embeds = inputs_embeds[s:e] if inputs_embeds is not None else None
                p_ids, p_embeds, update_dict = orig_model.preprocess(
                    input_ids=r_ids, input_embeds=r_embeds, **req_info
                )
                self._update_intermediate_buffer(req_id, update_dict)
                if isinstance(p_embeds, torch.Tensor):
                    if inputs_embeds is None:
                        total_tokens = (
                            preprocess_ids.shape[0] if preprocess_ids is not None else e
                        )
                        inputs_embeds = torch.zeros(
                            (total_tokens, p_embeds.shape[-1]),
                            dtype=p_embeds.dtype,
                            device=p_embeds.device,
                        )
                    p_len = min(e - s, p_embeds.shape[0])
                    inputs_embeds[s : s + p_len] = p_embeds[:p_len].to(
                        device=inputs_embeds.device, dtype=inputs_embeds.dtype
                    )
                mtp_inputs = req_info.pop("mtp_inputs", None)
                if (
                    not is_prefill
                    and callable(mtp_fn)
                    and mtp_inputs is not None
                    and inputs_embeds is not None
                ):
                    last_talker_hidden, text_step = mtp_inputs
                    mtp_embeds, code_predictor_codes = mtp_fn(
                        p_ids,
                        inputs_embeds[s:e],
                        last_talker_hidden.to(device=inputs_embeds.device),
                        text_step.to(device=inputs_embeds.device),
                    )
                    inputs_embeds[s:e] = mtp_embeds.reshape(-1, inputs_embeds.shape[-1])
                    self._update_intermediate_buffer(
                        req_id, {"codes": {"audio": code_predictor_codes}}
                    )
            if inputs_embeds is not None:
                return None, inputs_embeds
        return model_input_ids, inputs_embeds

    def forward_model(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[torch.Tensor | IntermediateTensors, list[torch.Tensor] | None]:
        # OmniOutput is a 4-field NamedTuple (text_hidden_states,
        # multimodal_outputs, intermediate_tensors, next_token_id), so
        # TPUModelRunner.forward_model's 2-tuple draft-model unpack does not
        # split it before we extract text_hidden_states here.
        out, aux = super().forward_model(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )
        if isinstance(out, OmniOutput) and out.text_hidden_states is not None:
            out = out.text_hidden_states
        orig_model = _unwrap_model(self.model)
        if getattr(orig_model, "has_postprocess", False) and isinstance(
            out, torch.Tensor
        ):
            for req_id, _req_idx, s, e in self._chunk_req_spans():
                req_info = self.model_intermediate_buffer.get(req_id, {})
                post_kwargs = {
                    k: v for k, v in req_info.items() if k != "hidden_states"
                }
                update_dict = orig_model.postprocess(out[s:e], **post_kwargs)
                self._update_intermediate_buffer(req_id, update_dict)
        self._omni_last_hidden_states = out
        self._omni_last_aux_hidden_states = aux
        return out, aux

    def _populate_inter_stage_outputs(self, real_output: ModelRunnerOutput) -> None:
        orig_model = _unwrap_model(self.model)
        make_fn = getattr(orig_model, "make_omni_output", None)
        raw_out = self._omni_last_hidden_states
        aux_out = self._omni_last_aux_hidden_states
        self._omni_last_hidden_states = None
        self._omni_last_aux_hidden_states = None
        req_ids = real_output.req_ids
        if (
            not callable(make_fn)
            or not req_ids
            or (raw_out is None and aux_out is None)
        ):
            return
        model_out = (raw_out, aux_out) if aux_out is not None else raw_out
        req_infos = [self.model_intermediate_buffer.get(rid, {}) for rid in req_ids]
        omni_out = make_fn(model_out, model_intermediate_buffer=req_infos)
        raw_mm = (
            omni_out.multimodal_outputs
            if isinstance(omni_out, OmniOutput)
            else (omni_out if isinstance(omni_out, dict) else None)
        )
        if not isinstance(raw_mm, dict):
            return
        query_start_loc = self.query_start_loc_np
        per_req_payloads = []
        for idx in range(len(req_ids)):
            tok_start = (
                int(query_start_loc[idx]) if idx + 1 < len(query_start_loc) else None
            )
            tok_end = (
                int(query_start_loc[idx + 1])
                if idx + 1 < len(query_start_loc)
                else None
            )
            per_req_payloads.append(
                _slice_inter_stage_payload(raw_mm, idx, tok_start, tok_end)
            )
        real_output.inter_stage_outputs = per_req_payloads  # type: ignore[attr-defined]
        reqs_map = getattr(self, "requests", None)
        if (
            self._omni_connector_initialized
            and self._should_accumulate_full_payload_output()
            and isinstance(reqs_map, dict)
        ):
            for idx, rid in enumerate(req_ids):
                req_state = reqs_map.get(rid)
                if req_state is not None and per_req_payloads[idx]:
                    self.accumulate_full_payload_output(
                        rid, flatten_payload(per_req_payloads[idx]), req_state
                    )

    def _with_omni_connector_output(
        self,
        output: ModelRunnerOutput | AsyncTPUModelRunnerOutput | None,
    ) -> ModelRunnerOutput | AsyncTPUModelRunnerOutput | None:
        """Attach inter-stage payloads and connector signals before ``get_output()``."""
        if output is None:
            return None
        if isinstance(output, AsyncTPUModelRunnerOutput):
            self._populate_inter_stage_outputs(output._model_runner_output)
            if self._omni_connector_initialized:
                output._model_runner_output = self.attach_omni_connector_output(
                    output._model_runner_output
                )
            return output
        self._populate_inter_stage_outputs(output)
        if self._omni_connector_initialized:
            return self.attach_omni_connector_output(output)
        return output

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | AsyncTPUModelRunnerOutput | None:
        if self._omni_connector_initialized:
            for request in getattr(scheduler_output, "pending_input_registrations", []):
                self.register_chunk_recv(request)
            self.recv_full_payload_inputs(scheduler_output)
            if getattr(self, "_pending_full_payload_send", None):
                reqs_map = getattr(self, "requests", {})
                flush_ids = set(getattr(scheduler_output, "finished_req_ids", set()))
                flush_ids.update(
                    {
                        rid
                        for rid in self._pending_full_payload_send
                        if rid not in reqs_map
                    }
                )
                if flush_ids:
                    self.flush_full_payload_outputs(flush_ids)
        try:
            return self._with_omni_connector_output(
                super().execute_model(scheduler_output, intermediate_tensors)
            )
        finally:
            self._omni_chunk_req_slice = None

    def sample_tokens(
        self, grammar_output: GrammarOutput | None = None
    ) -> ModelRunnerOutput | AsyncTPUModelRunnerOutput:
        return self._with_omni_connector_output(  # type: ignore[return-value]
            super().sample_tokens(grammar_output)
        )
