# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD model runner subclass (cloud side).

All LWD runner-side logic lives here (moved out of
``model_runner_v1.py``): the per-step cloud-side collection of the
DOWN payload (hidden-only tensor) plus the step metadata (global ranks
/ num_accepted / req_ids) that rides back to the scheduler on
``ModelRunnerOutput.lwd_c2e_meta``.

Selected via ``LwdCloudWorker`` (worker_cls), which swaps the model
runner class at init_device when ``lwd_config`` is set.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.lwd_logger import lwd_logger
from vllm.v1.outputs import ModelRunnerOutput

from vllm_ascend.worker.lwd_base_model.lwd_base_model_runner import LwdBaseModelRunner
from vllm_ascend.worker.lwd_cloud_model.lwd_mtp_proposer import LwdMTPProposer
from vllm_ascend.worker.lwd_hash.lwd_hash_routing import (
    LwdHashRoutingState,
    hash_layer_count,
    hash_payload_numel,
    unpack_hash_payload,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager
    from typing import Any

    from vllm.config import VllmConfig
    from vllm.sequence import IntermediateTensors
    from vllm.v1.core.sched.output import (
        GrammarOutput,
        NewRequestData,
        SchedulerOutput,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import ECConnectorOutput, SamplerOutput
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

    from vllm_ascend.worker.lwd_cloud_model.lwd_cloud_worker import LwdCloudWorker
    from vllm_ascend.worker.lwd_cloud_model.lwd_host_stage import LwdHostStager
from vllm_ascend.worker.lwd_cloud_model.lwd_draft_cache import LwdDraftCache


class LwdDownSelection(NamedTuple):
    """Row/rank selection product of one DOWN collect step."""

    hidden_packet: torch.Tensor
    ranks: list[torch.Tensor]
    counts_dev: torch.Tensor | None
    seg_lens_host: list[int] | None


class LwdCloudModelRunner(LwdBaseModelRunner):
    """NPUModelRunner + LWD cloud-side runner logic."""

    # Config-derived constants, computed once at first use; lwd_config and
    # the HF config are immutable at runtime. Class-level defaults keep
    # instances built via __new__ (tests) from tripping on missing attrs.
    _lwd_hash_dims_cached: tuple[int, int] | None = None

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        worker: LwdCloudWorker | None = None,
    ) -> None:
        super().__init__(vllm_config, device, worker)
        self._lwd_pending_down_payload = None
        # Row-slot residue cleanup: flush reused rows as soon as a request
        # is admitted (2026-10-21 MTE root-cause closure)
        self._lwd_reset_reused_rows_on_admission()
        # Stage per-request prompt lengths for device-side mask refresh.
        if self.use_async_spec_decode and self.enable_prompt_embeds:
            self._lwd_prompt_lens_gpu = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=self.device)
        else:
            self._lwd_prompt_lens_gpu = None
        self.lwd_hash_state = None
        # None marks "no real batch staged yet"; distinguishes dummy forwards.
        self._lwd_hash_step_tokens = None
        # Spec-decode draft snapshots for requests temporarily absent
        # from the batch (async scheduling + MTP only): the graph input
        # buffers are reused every step, so a departing request's draft
        # rows must survive in owned storage until it returns.
        self._lwd_draft_cache = (
            LwdDraftCache()
            if self.use_async_scheduling
            and self.speculative_config is not None
            and self.speculative_config.method == "mtp"
            # This lineage's runner has no pcp_size attribute: the PCP
            # width comes from the parallel config (as the DSV4 check
            # above reads it); dcp_size is a real base-runner attribute.
            and vllm_config.parallel_config.prefill_context_parallel_size == 1
            and self.dcp_size == 1
            else None
        )
        config = self.model_config.hf_config
        num_hash_layers = hash_layer_count(config)
        if num_hash_layers and vllm_config.lwd_config is not None:
            # UP currently broadcasts one edge's batch to one cloud TP group.
            # CP/DP need additional request-to-rank layouts before supporting V4.
            parallel = vllm_config.parallel_config
            if (
                parallel.data_parallel_size != 1
                or parallel.prefill_context_parallel_size != 1
                or parallel.decode_context_parallel_size != 1
            ):
                raise ValueError("LWD V4 Hash routing currently requires DP=PCP=DCP=1 (TP/EP are supported)")
            self.lwd_hash_state = LwdHashRoutingState(
                num_hash_layers,
                config.num_experts_per_tok,
                config.n_routed_experts,
            )
            # Persistent buffers are populated before model execution, so graph
            # capture and replay see the same addresses for prompt and decode.
            self._lwd_hash_experts = torch.zeros(
                self.max_num_tokens,
                num_hash_layers,
                config.num_experts_per_tok,
                dtype=torch.int32,
                device=device,
            )
            self._lwd_hash_prompt_mask = torch.zeros(self.max_num_tokens, dtype=torch.bool, device=device)
            self._lwd_hash_token_ids = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=device)

    def _lwd_reset_reused_row(
        self,
        request: CachedRequestState,
        req_index: int,
    ) -> None:
        """Clear tokens left in a reused batch row by the previous request.

        LWD cloud requests carry no token ids (the wire ships embeddings
        only), so the native add_request skips writing the row's prompt
        region for them. Stale cells from the row's previous occupant -
        including -1 placeholders that async scheduling leaves behind and
        nothing repairs for penalty-free requests - survive row reuse and
        ride the prompt-token tensor into the penalty bincount kernel,
        where negative ids become negative-offset writes (the aivec
        fault). Native requests carry real ids and overwrite the region
        on every admission, which is exactly the behavior missing here.

        Fill [0, num_prompt) with vocab_size - the "no token" padding
        value every downstream consumer already ignores - once, at
        admission, before any reader can observe the row.
        """
        if request.prompt_token_ids is not None:
            return  # rows with real ids are already overwritten natively
        batch = self.input_batch
        num_prompt = int(batch.num_prompt_tokens[req_index])
        batch.token_ids_cpu[req_index, :num_prompt] = batch.vocab_size

    def _lwd_reset_reused_rows_on_admission(self) -> None:
        """Hook the row reset onto request admission (instance-level).

        The admission call site lives inside the shared update_states
        with no LWD-owned seam; overriding it would mean copying shared
        logic that breaks on every upstream change. Wrap add_request on
        this process's InputBatch instance instead: native bookkeeping
        first, then _lwd_reset_reused_row. Migrate to an upstream hook
        if one is ever provided.

        Idempotent per InputBatch instance: initialize_kv_cache
        REBUILDS the InputBatch when the final KV geometry differs from
        the pre-load guess (always true for the hybrid groups here,
        model_runner_v1 re-init), so the hook must be re-anchored on
        the live instance - see initialize_kv_cache below. The first
        acceptance run failed exactly this way: the wrapper stayed on
        the discarded instance and the residue poison survived.
        """
        batch = self.input_batch
        if getattr(batch, "_lwd_row_reset_hooked", False):
            return
        batch._lwd_row_reset_hooked = True
        orig_add = batch.add_request

        def add_request_then_reset_row(request):
            req_index = orig_add(request)
            self._lwd_reset_reused_row(request, req_index)
            return req_index

        batch.add_request = add_request_then_reset_row

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_allocation_context: AbstractContextManager | None = None,
    ) -> None:
        """Re-anchor the row-reset hook after the InputBatch rebuild.

        The base may replace self.input_batch here (hybrid KV geometry);
        the per-instance hook flag on the fresh instance is absent, so
        the installer below (re)hooks the live one."""
        super().initialize_kv_cache(kv_cache_config, kv_cache_allocation_context)
        self._lwd_reset_reused_rows_on_admission()

    def _get_drafter(self) -> SpecDecodeBaseProposer | None:
        """Use the LWD MTP embedding adapter only in LWD mode."""
        if (
            self.vllm_config.lwd_config is not None
            and self.speculative_config.method == "mtp"
            and not self.speculative_config.use_step3p5_mtp()
        ):
            return LwdMTPProposer(self.vllm_config, self.device, self)
        return super()._get_drafter()

    # ------------------------------------------------------------------ #
    # Remote embeds injection (cloud input has NO token ids — the prompt   #
    # embeddings arrive over the UP channel and must drive forward)        #
    # ------------------------------------------------------------------ #

    def _prepare_input_ids(
        self,
        scheduler_output: SchedulerOutput,
        num_reqs: int,
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
    ) -> None:
        super()._prepare_input_ids(
            scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens,
        )
        self._lwd_restore_drafts(scheduler_output, num_reqs, cu_num_tokens)

    def _lwd_restore_drafts(
        self,
        scheduler_output: SchedulerOutput,
        num_reqs: int,
        cu_num_tokens: np.ndarray,
    ) -> None:
        """Reinstate a returning request's draft rows into input_ids.

        A request absent for a few steps (paused/queued under async
        scheduling) returns with its scheduled spec-decode segment; the
        native fill wrote prompt/decode ids, the draft rows at the
        segment tail come back from the preserved snapshot (device
        scatter, same stream as the native input assembly - no D2H, no
        host wait).
        """
        cache = self._lwd_draft_cache
        if cache is None or not cache.entries:
            return
        rows, indices = [], []
        for idx, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            if req_id not in cache.entries:
                continue
            tokens = scheduler_output.scheduled_spec_decode_tokens.get(req_id, ())
            row = cache.take(req_id, self.requests[req_id], len(tokens))
            # Consume the generation even when the scheduler dropped all
            # drafts (the snapshot must not leak into a later return).
            if not tokens or self.prev_positions.np[idx] >= 0:
                continue
            if row is None:
                raise RuntimeError(f"LWD draft snapshot is invalid for {req_id}")
            end = int(cu_num_tokens[idx])
            begin = int(cu_num_tokens[idx - 1]) if idx else 0
            if end - begin < len(tokens) + 1:
                raise RuntimeError(f"LWD draft segment is too short for {req_id}")
            rows.append(row)
            indices.extend(range(end - len(tokens), end))
        if rows:
            dst = torch.tensor(
                indices, dtype=torch.int64, pin_memory=self.pin_memory
            ).to(self.device, non_blocking=True)
            self.input_ids.gpu.scatter_(
                0,
                dst,
                torch.cat(rows).to(dtype=self.input_ids.gpu.dtype),
            )

    def _prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[torch.Tensor, SpecDecodeMetadata | None]:
        """Stage the step's prefill windows before super(); inject after.

        Arrival-driven host staging (cpu_offload design v8): the comm
        thread lands each recv's rows in the per-request pinned buffers
        as the recvs complete. This step's injectable rows are staged
        H2D into a shared window buffer on a dedicated copy stream
        (endpoint rank only) and broadcast once across the TP group, so
        every rank holds the same rows (W4: every rank derives an
        identical plan from the same scheduler output and input batch).
        ``_update_states`` runs before ``_prepare_inputs`` is entered
        (base execute_model), so the plan already includes this step's
        newly admitted requests.

        The inject stays after super(): the base fill loop and
        copy_to_gpu only see zero buffers, then the window rows are
        written straight into the inputs_embeds.gpu scheduling window
        of the current step (last write wins).
        """
        # Wire v3: new requests' full-prompt mrope positions ride their
        # NewRequestData (relayed from the RequestNotify payload by the
        # scheduler); prime req_state once at the request's first
        # scheduled step - before super(), so the current step's
        # _calc_mrope_positions already reads the primed rows.
        for d in scheduler_output.scheduled_new_reqs:
            payload = getattr(d, "lwd_mrope_positions", b"") or b""
            if payload:
                self._lwd_prime_mrope_positions(d.req_id, payload)
        # H2D staging + TP broadcast before super(): the side-stream
        # copies overlap with super()'s host-side work; the compute
        # stream only waits on the ready event recorded after the last
        # copy.
        dev, win_idx = self._lwd_harvest_staged_window(num_scheduled_tokens)
        # Skip the base fill loop's per-row copy for LWD placeholder rows
        # (gpu_model_runner.py:952-996): it host-memcpys 84MB from the CPU
        # assembly buffer (all zeros at this point) plus an H2D into
        # inputs_embeds, which the inject then overwrites on the same
        # window (last write wins) — pure waste, measured ~17-20ms per
        # chunk in the super() segment. Clearing the map skips it
        # wholesale; the inject/release logic restores the reference
        # after super() returns. Every cloud-side LWD request is a
        # placeholder row, so no real prompt-embeds consumer is hurt.
        embeds_map = self.input_batch.req_prompt_embeds
        self.input_batch.req_prompt_embeds = {}
        try:
            out = super()._prepare_inputs(scheduler_output, num_scheduled_tokens)
        finally:
            self.input_batch.req_prompt_embeds = embeds_map
        self._lwd_rebuild_is_token_ids_mask(num_scheduled_tokens)
        self._lwd_release_consumed_prompt_embeds()
        self._lwd_inject_from_stager(dev, win_idx, num_scheduled_tokens)
        if self.lwd_hash_state is not None:
            self._lwd_prepare_hash_batch(num_scheduled_tokens)
        return out

    def _lwd_prepare_hash_batch(self, num_scheduled_tokens: np.ndarray) -> None:
        """Skip remote expert staging for decode, preserving graph buffers."""
        num_reqs = self.input_batch.num_reqs
        counts = num_scheduled_tokens[:num_reqs]
        computed = self.input_batch.num_computed_tokens_cpu[:num_reqs]
        prompt_lengths = self.input_batch.num_prompt_tokens[:num_reqs]
        self._lwd_hash_step_tokens = sum(int(n) for n in counts)
        has_prompt = any(
            int(count) > 0 and int(start) < int(prompt_len)
            for count, start, prompt_len in zip(counts, computed, prompt_lengths)
        )
        # Clear the full mask, including graph padding and previous larger
        # batches. Keep its address and the captured merge graph unchanged.
        self._lwd_hash_prompt_mask.zero_()
        if not has_prompt:
            # The merge selects local lookup for every row. Stale expert IDs
            # are only unselected values, never indices into an expert table.
            return
        batch, mask = self.lwd_hash_state.build_batch(
            self.input_batch.req_ids,
            counts,
            computed,
            prompt_lengths,
        )
        self._lwd_hash_experts.zero_()
        self._lwd_hash_experts[: batch.shape[0]].copy_(batch)
        self._lwd_hash_prompt_mask[: mask.shape[0]].copy_(mask)

    def _lwd_rebuild_is_token_ids_mask(self, num_scheduled_tokens: np.ndarray) -> None:
        """Rebuild the flattened is_token_ids mask for the prompt-embeds branch.

        Root cause (async spec decode optimism): the runner's
        num_computed_tokens_cpu assumes every previous draft was accepted
        (r tokens past the real commit position), while the native
        flattening (token_indices = positions + idx*M in
        model_runner_v1.py) gathers from the 2D mask with those
        optimistic positions — decode rows shift right by r and the last
        r rows of each segment fall past the True-marked region, reading
        False/stale values. The prompt-embeds branch then treats them as
        embeds rows, skipping local embedding and leaving stale garbage
        in inputs_embeds.gpu; those rows are exactly the sampled rows
        (logits_indices takes the last 1+k rows of each segment), so
        garbage logits pollute committed tokens directly. Native/phase/
        single-request shapes are unaffected (the mask is only consumed
        on the prompt-embeds branch; those shapes take the MM branch).

        Under LWD semantics only injected rows (prompt segments) may be
        embeds rows (False); decode/draft rows are always token ids
        (True). This rebuild is deterministic from scheduler state and
        needs no optimistic-value correction: a decode request's
        optimistic computed only over-estimates, so prefill_rows is
        always 0 — inherently safe. The CPU-view repair is kept for the
        parent path (.np and .cpu share storage); the async path
        regenerates the GPU mask in _preprocess from corrected device
        positions and consumes the device embedding directly.
        """
        if not self.input_batch.req_prompt_embeds:
            # No embeds buffers in the batch (MM/text branch; nobody reads the mask)
            return
        mask = self.is_token_ids.np
        computed = self.input_batch.num_computed_tokens_cpu
        num_prompt = self.input_batch.num_prompt_tokens
        n_reqs = len(num_scheduled_tokens)
        total = int(np.sum(num_scheduled_tokens[:n_reqs]))
        mask[:total] = True
        off = 0
        for i in range(n_reqs):
            n = int(num_scheduled_tokens[i])
            prefill_rows = min(n, max(int(num_prompt[i]) - int(computed[i]), 0))
            if prefill_rows > 0:
                mask[off : off + prefill_rows] = False
            off += n

    def _preprocess(
        self,
        scheduler_output: SchedulerOutput,
        num_input_tokens: int,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        IntermediateTensors | None,
        dict[str, Any],
        ECConnectorOutput | None,
    ]:
        """Refresh the decode mask after the async speculative-position fix, then prepare model inputs."""
        if (
            self.use_async_spec_decode
            and self.enable_prompt_embeds
            and intermediate_tensors is None
            and (self.vllm_config.parallel_config.prefill_context_parallel_size == 1)
        ):
            self._lwd_refresh_decode_embedding_mask(scheduler_output)
            if (
                get_pp_group().is_first_rank
                and not self.model_config.is_encoder_decoder
                and (not self.supports_mm_inputs or self.input_batch.req_prompt_embeds)
            ):
                return self._lwd_preprocess_prompt_embeds(scheduler_output, num_input_tokens)
            # Other model paths keep the parent's preprocessing, which may
            # still consume the CPU mask (for example encoder-decoder models).
            count = scheduler_output.total_num_scheduled_tokens
            self.is_token_ids.np[:count] = self.is_token_ids.gpu[:count].cpu().numpy()
        return super()._preprocess(scheduler_output, num_input_tokens, intermediate_tensors)

    # ------------------------------------------------------------------ #
    # Zero-base-weave seams: all LWD behavior surfaces via subclass    #
    # overrides; the base classes carry no seams.                       #
    # ------------------------------------------------------------------ #

    def _update_states(self, scheduler_output: SchedulerOutput) -> Callable[[], None] | None:
        self._lwd_rewind_staged_reads(scheduler_output)
        if self._lwd_draft_cache is not None:
            # Snapshot departing requests' draft rows before the batch
            # mapping changes (same stream as the input prep).
            self._lwd_draft_cache.preserve(
                scheduler_output,
                self.requests,
                self.input_batch.prev_req_id_to_index,
                self._draft_token_ids,
            )
        """LWD: materialize meta placeholder embeds in place before they
        reach the base (zero base seams).

        The engine attaches prompt embeds as meta tensors (shape only,
        empty storage), so the placeholder rides the MQ inline pickle
        path instead of the overflow slow channel; the base
        CachedRequestState construction passes new_req_data.prompt_embeds
        straight through — swap meta tensors for local all-zeros
        allocations here, native path behavior unchanged.
        NOTE: a meta tensor still reports full nbytes (numel * element_size);
        do not infer transport cost from nbytes."""
        for new_req in scheduler_output.scheduled_new_reqs:
            embeds = new_req.prompt_embeds
            if embeds is not None and embeds.is_meta:
                new_req.prompt_embeds = torch.zeros(embeds.shape, dtype=embeds.dtype)
        return super()._update_states(scheduler_output)

    def _update_streaming_request(self, req_id: str, new_req_data: NewRequestData) -> CachedRequestState:
        """LWD: meta placeholder materialization for streaming updates
        (same as _update_states, zero base seams)."""
        embeds = new_req_data.prompt_embeds
        if embeds is not None and embeds.is_meta:
            new_req_data.prompt_embeds = torch.zeros(embeds.shape, dtype=embeds.dtype)
        return super()._update_streaming_request(req_id, new_req_data)

    def _init_mrope_positions(self, req_state: CachedRequestState) -> None:
        """LWD prompt-embeds-only: text-only M-RoPE positions do not depend
        on the ids content (the scan only honors vision markers), so build
        them directly from the prompt length; with ids present, fall back
        to the native scan (zero base seams)."""
        if req_state.prompt_token_ids is None:
            n = req_state.num_prompt_tokens
            req_state.mrope_positions = torch.arange(n).unsqueeze(0).expand(3, n).clone()
            req_state.mrope_position_delta = 0
            return
        super()._init_mrope_positions(req_state)

    @property
    def supports_mm_inputs(self) -> bool:
        """LWD: present injected batches as "no MM this step" so the native
        MM gate skips the encoder naturally (recomputed embeddings would
        wholesale-overwrite the remotely injected values); decode and
        text-only steps return the model's true value. Zero base seams."""
        input_batch = getattr(self, "input_batch", None)
        if input_batch is not None and input_batch.req_prompt_embeds:
            return False
        return getattr(self, "_model_supports_mm_inputs", False)

    @supports_mm_inputs.setter
    def supports_mm_inputs(self, value: bool) -> None:
        # The base __init__ assigns directly; keep the backing for the dynamic getter.
        self._model_supports_mm_inputs = value

    def _lwd_preprocess_prompt_embeds(
        self, scheduler_output: SchedulerOutput, num_input_tokens: int
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        IntermediateTensors | None,
        dict[str, Any],
        ECConnectorOutput | None,
    ]:
        """Consume the corrected device mask on the LWD cloud decoder path."""
        count = scheduler_output.total_num_scheduled_tokens
        is_token_ids = self.is_token_ids.gpu[:count]
        # Remote prompt rows do not carry meaningful token IDs.
        token_ids = self.input_ids.gpu[:count].masked_fill(~is_token_ids, 0)
        tokens_to_embeds = self.model.embed_input_ids(input_ids=token_ids)
        embeds = self.inputs_embeds.gpu[:count]
        embeds.copy_(torch.where(is_token_ids.unsqueeze(-1), tokens_to_embeds, embeds))

        # Match the parent's position layout and clear padding for graph replay.
        # The retired base exposed an xdrope-dimension attribute here; this
        # base splits positions two-way (mrope/plain) exactly like its own
        # dummy run, so xdrope models follow whichever layout the base uses.
        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]
            if num_input_tokens > count:
                self.positions[count:num_input_tokens].zero_()

        # DSV4-only: the upstream model-layer fail-fast requires a non-None ids
        # tensor on hash MoE layers, but LWD keeps real token IDs off the cloud
        # entirely. Nothing true exists here, so the placeholder is
        # information-free by construction (all zeros): the gate below and the
        # remote expert-table publication in _model_forward key off the same
        # lwd_hash_state, so the router always swaps the tensor for row
        # indices before any read -- the values are dead on arrival. Other
        # models keep the native None unchanged.
        model_input_ids: torch.Tensor | None = None
        if self.lwd_hash_state is not None:
            model_input_ids = torch.zeros_like(self.input_ids.gpu[:num_input_tokens])
        return (
            model_input_ids,
            self.inputs_embeds.gpu[:num_input_tokens],
            positions,
            None,
            self._init_model_kwargs(),
            None,
        )

    def _lwd_refresh_decode_embedding_mask(self, scheduler_output: SchedulerOutput) -> None:
        """Rebuild the mask from actual device positions so decode rows never
        reuse stale embeddings.

        Upload per-request prompt lengths asynchronously and expand them
        to token rows via the on-device request indices. The mask is
        written straight into the device buffer and consumed on-device
        by the LWD prompt-embeds path; positions are not round-tripped
        and the CPU mask is not refreshed.
        """
        num_reqs = len(self.input_batch.req_ids)
        count = scheduler_output.total_num_scheduled_tokens
        prompt_lens_gpu = self._lwd_prompt_lens_gpu
        assert prompt_lens_gpu is not None
        prompt_lens_gpu[:num_reqs].copy_(
            self.input_batch.num_prompt_tokens_cpu_tensor[:num_reqs],
            non_blocking=True,
        )
        prompt_lens_rows = prompt_lens_gpu[self.req_indices.gpu[:count]]
        self.is_token_ids.gpu[:count] = self.positions[:count] >= prompt_lens_rows

    # Clear the real-batch marker before cloud-side dummy forwards, so the
    # previous step's routing is never reused.
    def _dummy_run(self, *args, **kwargs) -> None:
        """Clear the real-batch Hash routing marker, then run the dummy forward."""
        # None marks "no real batch staged yet"; distinguishes dummy forwards.
        self._lwd_hash_step_tokens = None
        return super()._dummy_run(*args, **kwargs)

    def _model_forward(
        self,
        num_tokens_padded: int,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """Shard embeddings per the SP layout and publish the full Hash routing data into the forward context."""
        context = get_forward_context()
        # flash_comm_v1_enabled is only set by the SP fast path; without SP
        # (the LWD 1+8 TP layout) it is absent and the sharding is skipped.
        if (
            inputs_embeds is not None
            and intermediate_tensors is None
            and getattr(context, "flash_comm_v1_enabled", False)
            and not self.supports_mm_inputs
        ):
            # Passing embeddings directly bypasses the native embedding layer; re-apply the sequence split here.
            tp_group = get_tp_group()
            if tp_group.world_size > 1:
                if inputs_embeds.shape[0] != num_tokens_padded:
                    raise ValueError("LWD SP expects full-token inputs_embeds before slicing")
                if context.pad_size:
                    inputs_embeds = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, context.pad_size))
                inputs_embeds = inputs_embeds.chunk(tp_group.world_size, dim=0)[tp_group.rank_in_group]
            # Positions, attention metadata and the Hash routing tables keep the full-token layout.
        if self.lwd_hash_state is not None:
            # Initialize the routing index cache first so padded rows use the legal zero index.
            self._lwd_hash_token_ids.zero_()
            if self._lwd_hash_step_tokens is None:
                # Synthetic forwards have no remote prompt; route dummy token 0.
                self._lwd_hash_prompt_mask.zero_()
            else:
                count = self._lwd_hash_step_tokens
                if count > num_tokens_padded:
                    raise ValueError("LWD Hash routing rows exceed model input rows")
                # Native input preparation retains real decode IDs even when
                # model input_ids=None because the model consumes embeddings.
                self._lwd_hash_token_ids[:count].copy_(self.input_ids.gpu[:count])
            context.input_ids = self._lwd_hash_token_ids[:num_tokens_padded]
            context.lwd_hash_expert_ids = self._lwd_hash_experts[:num_tokens_padded]
            context.lwd_hash_prompt_mask = self._lwd_hash_prompt_mask[:num_tokens_padded]
            if input_ids is None:
                # Covers the dummy/profile run: with enable_prompt_embeds it
                # also takes the inputs_embeds branch (input_ids=None), and
                # the model-layer hash fail-fast rejects None before the
                # router's dummy-token-0 path can run. Same contract as the
                # request-side placeholder: all zeros, never read (the
                # router swaps in row indices).
                input_ids = torch.zeros_like(self.input_ids.gpu[:num_tokens_padded])
        return super()._model_forward(
            num_tokens_padded,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **model_kwargs,
        )

    def _lwd_hash_dims(self) -> tuple[int, int]:
        """(num_hash_layers, experts_per_tok), cached at first use."""
        dims = self._lwd_hash_dims_cached
        if dims is None:
            config = self.model_config.hf_config
            dims = (
                hash_layer_count(config),
                getattr(config, "num_experts_per_tok", 0),
            )
            self._lwd_hash_dims_cached = dims
        return dims

    def _lwd_release_consumed_prompt_embeds(self) -> None:
        """Release assembly buffers whose prompt was fully consumed.

        A buffer becomes releasable once the request's computed tokens
        cover the whole prompt: the last chunk's rows were consumed by
        the previous step's fill loop, and the draft proposer read its
        window during that step's sampling (which precedes this call).
        Decode steps never read prompt embeds, so dropping the entry
        here cannot corrupt anything.  Entries follow native index
        compaction (condense/swap), so keying by idx stays correct.

        Also drops orphan entries whose slot no longer holds a live
        request (preemption removes the request without finishing it, so
        the flush hook never fires): a stale buffer at a recycled index
        would otherwise be mistaken for the new occupant's embeds.
        """
        if self.lwd_hash_state is not None:
            for req_id in list(self.lwd_hash_state.requests):
                request = self.requests.get(req_id)
                if request is None or request.num_computed_tokens >= request.num_prompt_tokens:
                    self.lwd_hash_state.discard(req_id)
        embeds_map = self.input_batch.req_prompt_embeds
        if not embeds_map:
            return
        num_prompt = self.input_batch.num_prompt_tokens
        computed = self.input_batch.num_computed_tokens_cpu
        slot_req_ids = self.input_batch.req_ids
        for idx in list(embeds_map.keys()):
            if idx >= len(computed):
                embeds_map.pop(idx)
                continue
            if idx >= len(slot_req_ids) or slot_req_ids[idx] is None:
                embeds_map.pop(idx)
                continue
            if computed[idx] >= num_prompt[idx]:
                embeds_map.pop(idx)

    def lwd_release_finished(self, finished_req_ids: list[str]) -> None:
        """Release collection bookkeeping for finished requests (sole owner of
        the release path): the Hash chunk table plus prompt-embeds assembly
        buffers. Called by the worker on the engine's finished_req_ids signal
        (backstop for prefill aborted mid-way; normal prefill completion is
        already released in _lwd_release_consumed_prompt_embeds)."""
        embeds_map = self.input_batch.req_prompt_embeds
        req_id_to_index = self.input_batch.req_id_to_index
        for req_id in finished_req_ids:
            if self.lwd_hash_state is not None:
                self.lwd_hash_state.discard(req_id)
            idx = req_id_to_index.get(req_id)
            if idx is not None:
                embeds_map.pop(idx, None)

    def _lwd_rewind_staged_reads(self, scheduler_output) -> None:
        """Cloud-side KV preemption recompute: the scheduler watermark
        regressed below the staged read pointer -> roll it back (the
        pinned host payload lives until request end, so a local re-prefill
        re-reads earlier rows; zero edge traffic). Also retires finished
        and aborted staging entries (batch is a transport container)."""
        stager = getattr(self.worker, "_lwd_stager", None)
        if stager is None:
            return
        cached = scheduler_output.scheduled_cached_reqs
        for req_id, new_computed in zip(cached.req_ids, cached.num_computed_tokens):
            emb = stager.take_req(req_id)
            if emb is None:
                continue
            if emb.consumed > new_computed:
                stager.rewind_consumed(req_id, new_computed)
        finished = getattr(scheduler_output, "finished_req_ids", None) or []
        for req_id in finished:
            if stager.take_req(req_id) is not None:
                stager.release_request(req_id)

    def _lwd_prefill_windows(
        self, num_scheduled_tokens: np.ndarray
    ) -> list[tuple[int, str, int, int]]:
        """This step's injectable rows: ``(input_batch index, req_id,
        num_computed, window rows)`` per request with prefill rows left.

        Single derivation shared by the staging and inject passes - both
        must agree on the row set and order: that is the TP broadcast
        layout contract (every rank computes the same plan from the same
        scheduler output and input batch, so shapes match on all ends).
        Only rows below the prompt end qualify (``ncomp < nprog``);
        decode/draft rows ride the token-ids path and never touch the
        stager - without the guard, a request's first decode step would
        read past ``recv_rows`` (the W2 invariant) and crash.
        """
        num_prompt = self.input_batch.num_prompt_tokens
        num_computed = self.input_batch.num_computed_tokens_cpu
        plan: list[tuple[int, str, int, int]] = []
        for i, rid in enumerate(self.input_batch.req_ids):
            if i >= len(num_scheduled_tokens):
                break
            n = int(num_scheduled_tokens[i])
            ncomp = int(num_computed[i]) if i < len(num_computed) else 0
            nprog = int(num_prompt[i]) if i < len(num_prompt) else 0
            if n <= 0 or ncomp >= nprog:
                continue
            w = min(n, nprog - ncomp)
            if w > 0:
                plan.append((i, rid, ncomp, w))
        return plan

    def _lwd_h2d_stream(self) -> torch.npu.Stream:
        """Dedicated H2D copy stream: the window copies overlap with
        super()'s host-side work; the compute stream only waits on the
        ready event recorded after the last copy."""
        stream = getattr(self, "_lwd_h2d_copy_stream", None)
        if stream is None:
            stream = torch.npu.Stream()
            self._lwd_h2d_copy_stream = stream
        return stream

    def _lwd_window_device(self, numel: int) -> tuple[torch.Tensor, int]:
        """Shared device window holding one step's staged rows,
        double-buffered (alternating steps) so the side-stream copy of
        step N+1 can start while step N's forward still reads its buffer
        (WAR via the used event recorded at inject time). Grows once to
        the scheduler's token budget, then reuses."""
        bufs = getattr(self, "_lwd_win_dev", None)
        if bufs is None or bufs[0].numel() < numel:
            hidden = self.model_config.get_hidden_size()
            num_hash_layers, top_k = self._lwd_hash_dims()
            per_row = hidden + 4 * num_hash_layers * top_k
            max_tokens = int(
                getattr(
                    getattr(self.vllm_config, "scheduler_config", None),
                    "max_num_batched_tokens",
                    0,
                )
                or 0
            )
            cap = max(numel, per_row * max_tokens)
            bufs = [
                torch.empty(max(cap, numel), dtype=torch.bfloat16, device=self.device)
                for _ in range(2)
            ]
            self._lwd_win_dev = bufs
            self._lwd_win_flip = 0
            self._lwd_win_used_ev: list[torch.npu.Event | None] = [None, None]
            lwd_logger.info(
                "[Lwd][host-stage] win_dev allocated numel=%d x2 "
                "(double-buffered)",
                bufs[0].numel(),
            )
        win_idx = self._lwd_win_flip
        self._lwd_win_flip = 1 - win_idx
        return bufs[win_idx], win_idx

    def _lwd_harvest_staged_window(
        self, num_scheduled_tokens: np.ndarray
    ) -> tuple[torch.Tensor | None, int | None]:
        """Stage this step's prefill rows H2D and broadcast them to the TP
        group: the endpoint copies each request's ``[consumed,
        consumed+w)`` host rows into a shared window, every other rank
        receives the same window (it holds no host stager - the broadcast
        is its only source of the rows).

        Returns ``(window, win_idx)``; ``(None, None)`` when this step
        injects nothing - the empty-plan decision is derived identically
        on every rank, so the broadcast call cannot diverge across the
        group.
        """
        plan = self._lwd_prefill_windows(num_scheduled_tokens)
        if not plan:
            return None, None
        worker = self.worker
        is_endpoint = bool(worker is not None and worker._lwd_early_recv_endpoint)
        hidden_size = self.model_config.get_hidden_size()
        # DSV4 expert-ID tail (4 * hash_layers * top_k bf16 elements per
        # row): rides the SAME window after the embeds region and
        # broadcasts with it - a non-endpoint rank holds no host stager,
        # so the broadcast is its only source of expert IDs.
        num_hash_layers, top_k = self._lwd_hash_dims()
        per_row_tail = 4 * num_hash_layers * top_k
        # Per-request tail-slice base: the read pointer BEFORE this
        # step's advance (slicing with the post-advance `consumed` would
        # read the NEXT window's tail rows).
        self._lwd_step_tail_base: dict[str, int] = {}
        total_rows = sum(w for _i, _rid, _ncomp, w in plan)
        embed_numel = total_rows * hidden_size
        window_numel = embed_numel + total_rows * per_row_tail
        if is_endpoint:
            stager = getattr(worker, "_lwd_stager", None)
            if stager is None:
                # The stager is built in init_device before any recv can
                # be posted; missing here is a wiring error, and running
                # on would broadcast zeros to every rank.
                raise RuntimeError(
                    "[Lwd][cloud-runner] endpoint rank without host stager "
                    "(host staging is mandatory in this deployment)"
                )
            dev, win_idx = self._lwd_window_device(window_numel)
            stream = self._lwd_h2d_stream()
            used_ev = self._lwd_win_used_ev[win_idx]
            with torch.npu.stream(stream):
                if used_ev is not None:
                    # WAR gate: wait for the previous consumer (inject +
                    # broadcast) of this buffer before overwriting it.
                    stream.wait_event(used_ev)
                offset = 0
                tail_offset = 0
                for _i, rid, ncomp, w in plan:
                    self._lwd_stage_one_request(
                        stager, rid, ncomp, w, dev, offset
                    )
                    offset += w * hidden_size
                    if per_row_tail:
                        req = stager.take_req(rid)
                        assert req is not None
                        t_start, _t_end = req.tail_range_at(
                            self._lwd_step_tail_base[rid], w
                        )
                        n = w * per_row_tail
                        dev[
                            embed_numel + tail_offset:
                            embed_numel + tail_offset + n
                        ].copy_(
                            req.host[t_start : t_start + n],
                            non_blocking=True,
                        )
                        tail_offset += n
                ready_ev = torch.npu.Event()
                ready_ev.record(stream)
            # Order the compute stream after the copies before the
            # broadcast reads the window (stream-order marker only - the
            # CPU does not block here).
            torch.npu.current_stream().wait_event(ready_ev)
            for _i, rid, _ncomp, w in plan:
                stager.consume(rid, w)
            lwd_logger.info(
                "[Lwd][cloud-runner] CONSUME rows=%d reqs=%d host_MB=%.1f",
                total_rows,
                len(plan),
                stager.host_bytes / 2**20,
            )
        else:
            # Receive side of the broadcast: same row total as the
            # endpoint (W4), plain allocation - no host stager locally.
            dev = torch.empty(window_numel, dtype=torch.bfloat16, device=self.device)
            win_idx = None
        tp = get_tp_group()
        if tp.world_size > 1:
            # Framework TP group on the compute stream: launch + wait in
            # the same stream context is the verified reliable shape
            # (broadcasts launched out-of-stream were measured racing
            # ahead of the producing copy and delivering stale rows).
            work = dist.broadcast(
                dev, src=tp.ranks[0], group=tp.device_group, async_op=True
            )
            work.wait()
        return dev, win_idx

    def _lwd_stage_one_request(
        self,
        stager: LwdHostStager,
        rid: str,
        ncomp: int,
        w: int,
        buf: torch.Tensor,
        elem_offset: int,
    ) -> None:
        """Copy one request's ``[consumed, consumed+w)`` host rows into its
        window slice (endpoint rank, side copy stream).

        Identity check: the host read pointer MUST equal the scheduler's
        computed watermark. Rows are read from ``[consumed, ...)`` and
        injected at prompt positions starting at ``num_computed``; if the
        two diverge (a batch staged twice, a window start mis-derived,
        preemption without a rewind), the model gets shifted embeddings
        - silent truncation/blank output, no error, no garbage. Fail
        loudly instead. W2 (rows <= arrived) is enforced authoritatively
        inside ``segment_for``'s read_range.
        """
        consumed = stager.consumed_rows(rid)
        if consumed != ncomp:
            raise RuntimeError(
                f"[Lwd][cloud-runner] STAGE POSITION MISMATCH req={rid}: "
                f"host read pointer consumed={consumed} != computed="
                f"{ncomp} (rows={w}); the embeddings would be injected "
                f"at a position that does not match the data read"
            )
        # Tail-slice base for this step: the PRE-advance read pointer
        # (the harvest slices the expert-ID tail after the embeds but
        # must not observe the pointer this call's consumer advances).
        self._lwd_step_tail_base[rid] = consumed
        host_start, numel, _tail = stager.segment_for(rid, w)
        buf[elem_offset : elem_offset + numel].copy_(
            stager.take_req(rid).host[host_start : host_start + numel],
            non_blocking=True,
        )

    @staticmethod
    def _lwd_decode_expert_ids(
        seg: torch.Tensor, w: int, num_layers: int, top_k: int
    ) -> torch.Tensor:
        """Decode the window's expert-ID byte region into ``[w, L, K]``
        int32.

        Same VALUE conversion as ``unpack_hash_payload``: the wire order
        stores one byte value (0-255, exactly representable in bf16) per
        bf16 element, so cast ``.to(torch.uint8)`` first and re-view as
        int32. NEVER ``.view(torch.uint8)`` - that reinterprets the bf16
        bit pattern as two bytes (wrong values, doubled element count)
        and the downstream reshape would explode.
        """
        return (
            seg.contiguous()
            .to(torch.uint8)
            .view(torch.int32)
            .view(w, num_layers, top_k)
        )

    def _lwd_prime_mrope_positions(self, req_id: str, payload: bytes) -> None:
        """Prime req_state's full-prompt mrope positions (wire v3).

        The positions arrive once with the RequestNotify ([3, prompt_len]
        int64 raw bytes, computed by the edge at admission); frombuffer
        writes them with zero transposition and the delta is derived in
        place (= positions.max()+1-prompt_len, value-identical to the
        edge side) for the native _calc_mrope_positions to compute the
        completion-segment positions during decode.

        req_state.mrope_positions starts as the ids=None arange
        passthrough (text-only positions); multimodal requests overwrite
        it here, text-only requests never carry a payload.
        """
        req_state = self.requests.get(req_id)
        if req_state is None:
            raise RuntimeError(
                f"[Lwd][cloud-runner] mrope prime req={req_id}: req_state "
                f"not found (admission ordering broken)"
            )
        plen = int(req_state.num_prompt_tokens)
        positions = torch.frombuffer(bytearray(payload), dtype=torch.int64)
        if positions.numel() != 3 * plen:
            raise RuntimeError(
                f"[Lwd][cloud-runner] mrope prime req={req_id}: payload "
                f"{positions.numel()} elems != 3 * prompt_len {plen} "
                f"(edge/cloud version mismatch; deploy matching versions)"
            )
        positions = positions.view(3, plen)
        req_state.mrope_positions[:, :plen] = positions
        req_state.mrope_position_delta = int(positions.max().item()) + 1 - plen
        lwd_logger.info(
            "[Lwd][cloud-runner] mrope primed req=%s prompt=%d delta=%d",
            req_id,
            plen,
            req_state.mrope_position_delta,
        )

    def _lwd_inject_from_stager(
        self,
        dev: torch.Tensor | None,
        win_idx: int | None,
        num_scheduled_tokens: np.ndarray,
    ) -> None:
        """Write the staged window rows into each request's flattened
        ``inputs_embeds`` slot (post-super(), device-side).

        Layout: the window packs this step's injectable rows in
        input_batch order (the same plan the endpoint staged); each
        request's prefill rows are the FIRST ``w`` rows of its flattened
        segment ``[ncomp, ncomp+n)`` - the flat offset advances by the
        full ``n`` so the trailing decode rows keep their positions.
        """
        if dev is None:
            return
        hidden_size = self.model_config.get_hidden_size()
        num_hash_layers, top_k = self._lwd_hash_dims()
        per_row_tail = 4 * num_hash_layers * top_k
        gpu_embeds = self.inputs_embeds.gpu
        num_prompt = self.input_batch.num_prompt_tokens
        num_computed = self.input_batch.num_computed_tokens_cpu
        planned = {
            i: (rid, w)
            for i, rid, _ncomp, w in self._lwd_prefill_windows(num_scheduled_tokens)
        }
        # The window's first total_rows*H elements are the embeds; the
        # expert-ID tail region follows immediately in the same row
        # order as the plan (the broadcast layout is the authority).
        n_reqs = min(len(self.input_batch.req_ids), len(num_scheduled_tokens))
        total_rows = 0
        for i in range(n_reqs):
            entry = planned.get(i)
            if entry is None:
                continue
            ncomp = int(num_computed[i])
            nprog = int(num_prompt[i])
            if ncomp < nprog:
                total_rows += min(entry[1], nprog - ncomp)
        embed_numel = total_rows * hidden_size
        embeds = dev[:embed_numel].view(-1, hidden_size)
        row = 0
        flat = 0
        for i in range(n_reqs):
            n = int(num_scheduled_tokens[i])
            entry = planned.get(i)
            w = entry[1] if entry is not None else 0
            if w > 0:
                assert entry is not None
                rid = entry[0]
                gpu_embeds[flat : flat + w].copy_(
                    embeds[row : row + w], non_blocking=True
                )
                if self.lwd_hash_state is not None and per_row_tail:
                    ncomp = int(num_computed[i])
                    nprog = int(num_prompt[i])
                    seg = dev[
                        embed_numel + row * per_row_tail:
                        embed_numel + (row + w) * per_row_tail
                    ]
                    ids = self._lwd_decode_expert_ids(
                        seg, w, num_hash_layers, top_k
                    )
                    self.lwd_hash_state.add_chunk(rid, nprog, ncomp, ids)
                row += w
            flat += n
        # Window lifecycle: the double-buffered slot's used event gates
        # the next side-stream overwrite (WAR); the ad-hoc broadcast
        # receive buffer only needs record_stream so the allocator
        # cannot recycle the block while the inject copy is queued.
        if win_idx is not None:
            ev = torch.npu.Event()
            ev.record()
            self._lwd_win_used_ev[win_idx] = ev
        else:
            dev.record_stream(torch.npu.current_stream())

    # ------------------------------------------------------------------ #
    # Collect probe (measurement only; output discarded, protocol        #
    # still returns token_ids directly)                                   #
    # ------------------------------------------------------------------ #

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> SamplerOutput:
        """Capture the sampler output for the collect probe
        (the base sample_tokens clears execute_model_state on return)."""
        sampler_output = super()._sample(logits, spec_decode_metadata)
        self._lwd_captured_sampler_output = sampler_output
        return sampler_output

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: GrammarOutput | None) -> ModelRunnerOutput:
        from vllm.distributed.parallel_state import get_tp_group

        # Only the cloud TP-group first rank (= the DOWN channel endpoint
        # rank) collects/sends; the other ranks have no channel peer, so
        # their collect-and-discard is skipped too (8-card redundancy).
        wire_endpoint = get_tp_group().is_first_rank
        captured = None
        if wire_endpoint and self.execute_model_state is not None:
            # ExecuteModelState layout (see model_runner_v1.sample_tokens):
            # (scheduler_output, logits, spec_decode_metadata,
            #  spec_decode_common_attn_metadata, hidden_states,
            #  sample_hidden_states, ...)
            state = self.execute_model_state
            # Snapshot the batch order/mask as well: with overlapping
            # batch queues, the next step's _prepare_inputs may reorder
            # input_batch before the collect; reading live state would
            # slice batch-A tensors with batch-B order (the root cause of
            # high-concurrency misalignment)
            captured = (
                state[5],
                state[1],
                state[2],
                state[0],
                list(self.input_batch.req_ids),
                self.discard_request_mask.np.copy(),
            )
        self._lwd_captured_sampler_output = None
        output = super().sample_tokens(grammar_output)
        if captured is not None and self._lwd_captured_sampler_output is not None:
            self._lwd_pending_down_payload = self._lwd_collect_down_payload(
                captured[0],
                captured[1],
                captured[2],
                captured[3],
                captured[4],
                captured[5],
                self._lwd_captured_sampler_output,
            )
        return output

    def take_lwd_pending_down_payload(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, None, list[str], None] | None:
        """The worker layer takes away this step's DOWN payload (single-slot overwrite, consumed every step)."""
        payload = getattr(self, "_lwd_pending_down_payload", None)
        self._lwd_pending_down_payload = None
        return payload

    @torch.inference_mode()
    def _lwd_collect_down_payload(
        self,
        sample_hidden_states: torch.Tensor | None,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        scheduler_output: SchedulerOutput,
        batch_req_ids: list[str],
        discard_mask_np: np.ndarray,
        sampler_output: SamplerOutput,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int] | None, torch.npu.Event, list[str]] | None:
        """Production DOWN collection (rank-replay): hidden packaging + global
        ranks + num_accepted.

        Orchestration: row selection / rank computation branches on spec
        vs non-spec (_lwd_pack_plain_down_rows / _lwd_pack_spec_down_rows).
        The step meta stays on device (no pinned D2H here — the side-stream
        copy was not fenceable on this platform); a per-step ready event is
        recorded on the main stream, and the fenced harvest (dedicated
        stream + wait_event + same-stream done event) runs in the worker's
        sample_tokens, off this collect. Zero added synchronization on the
        compute critical path; batch order/mask come from the capture-time
        snapshot (guards against batch-queue overlap reordering).
        Returns:
          (hidden_packet, meta_dev, seg_lens_host, ready_event, req_ids)
        or None (no in-flight requests this step).
        """
        _t0 = time.monotonic()
        sampled = sampler_output.sampled_token_ids
        if sampled is None or sampled.dim() != 2 or logits is None:
            return None
        # batch_req_ids/discard_mask come from the capture-time snapshot
        # (same batch order as logits/hidden); the live input_batch is
        # never read here
        valid = ~discard_mask_np[: len(batch_req_ids)]
        if spec_decode_metadata is None:
            packed = self._lwd_pack_plain_down_rows(
                sample_hidden_states,
                logits,
                sampled,
                valid,
                batch_req_ids,
            )
        else:
            packed = self._lwd_pack_spec_down_rows(
                sample_hidden_states,
                logits,
                sampled,
                spec_decode_metadata,
                valid,
                batch_req_ids,
            )
        if packed is None:
            return None
        hidden_packet, ranks_list, counts_dev, seg_lens_host = packed
        if seg_lens_host is not None:
            meta_dev = torch.cat(ranks_list + [counts_dev])
            # seg_lens stay off-device (host writes the pinned tail
            # segment): disjoint from the D2H prefix region, and the
            # engine reads the pinned buffer only after the ready_event
            # sync (later still) — no race.
        else:
            n_req = sum(1 for i in range(len(batch_req_ids)) if valid[i])
            counts_dev = torch.ones(n_req, dtype=torch.int32, device=logits.device)
            meta_dev = torch.cat(ranks_list + [counts_dev, counts_dev])
        selection_done = time.monotonic()
        # Per-step ready event: recorded on the main stream before the
        # payload leaves the collect. The fenced harvest (dedicated stream
        # + wait_event + same-stream done event) lives in the worker's
        # sample_tokens, NOT here — keeping the compute critical path free
        # of any synchronization. The device tensor travels with the
        # payload; the pinned D2H + shared-event scheme is retired (it
        # could not fence the side-stream copy on this platform).
        ready_event = torch.npu.Event()
        ready_event.record()
        lwd_logger.info(
            "[Lwd][perf] collect rows=%d rank=%.2f total=%.2fms",
            hidden_packet.shape[0],
            (selection_done - _t0) * 1000,
            (time.monotonic() - _t0) * 1000,
        )
        return (
            hidden_packet,
            meta_dev,
            seg_lens_host,
            ready_event,
            [r for i, r in enumerate(batch_req_ids) if valid[i]],
        )

    def _lwd_pack_plain_down_rows(
        self,
        sample_hidden_states: torch.Tensor | None,
        logits: torch.Tensor,
        sampled: torch.Tensor,
        valid: np.ndarray,
        batch_req_ids: list[str],
    ) -> LwdDownSelection | None:
        """Non-spec row selection and rank computation (correspondence:
        equivalent to the not-is_spec branch of the original
        _lwd_collect_down_payload).

        Cheap computation: direct bf16 comparison (logits are natively
        bf16, bit-identical to a cast) and full-batch rows computed
        wholesale when fully covered (no fancy-index gathers)."""
        idx = [i for i in range(len(batch_req_ids)) if valid[i]]
        if not idx:
            return None
        full_cover = all(valid[: len(batch_req_ids)])
        lg_sel = logits if full_cover else logits[idx]
        sm_sel = sampled[:, 0] if full_cover else sampled[idx][:, 0]
        thresh = lg_sel.gather(1, sm_sel.long().unsqueeze(1))
        ranks = [(lg_sel > thresh).sum(dim=1).to(torch.int32)]
        hidden_packet = torch.cat([sample_hidden_states[i : i + 1] for i in idx])
        return LwdDownSelection(
            hidden_packet=hidden_packet,
            ranks=ranks,
            counts_dev=None,
            seg_lens_host=None,
        )

    def _lwd_pack_spec_down_rows(
        self,
        sample_hidden_states: torch.Tensor,
        logits: torch.Tensor,
        sampled: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata,
        valid: np.ndarray,
        batch_req_ids: list[str],
    ) -> LwdDownSelection | None:
        """All-zero synchronous spec row selection and rank computation
        (vectorized; correspondence: equivalent to the is_spec branch of
        the original _lwd_collect_down_payload).

        Segment lengths derive from the host-side
        scheduled_spec_decode_tokens (1+draft_len); rows are packed as
        whole segments (rejected rows included; the edge takes the valid
        prefix by num_accepted). Lengths come from
        spec_decode_metadata.num_draft_tokens (host list, the real
        runtime layout, matching the sample_hidden_states segment
        structure), NOT from
        scheduler_output.scheduled_spec_decode_tokens (scheduler input
        that may disagree with what actually ran). counts derive from
        the same sampled tensor on-device (same source as ranks/rows,
        naturally batch-position aligned); seg_lens/counts only include
        valid requests so [ranks|counts|seg_lens] stay length-aligned.
        One host pass builds the indices, replacing the per-request
        device loop (the original issued ~6 small kernels per request
        and the rank segment grew past 20ms with concurrency; after
        vectorization the launch count is independent of N). Index
        semantics are bitwise-equivalent to the original loop:
        vidx/vlen = valid request batch position / actual row count
        (capped by total segment length, = the original rows_i);
        req_idx/pos_idx = per-row request position / in-segment offset
        (the original sampled[i, :rows_i]); row_sel = global row number
        of each valid row."""
        seg_lens = [d + 1 for d in spec_decode_metadata.num_draft_tokens]
        counts_all = (sampled != -1).sum(dim=1)
        seg_lens_np = np.asarray(seg_lens, dtype=np.int32)
        seg_starts = np.cumsum(seg_lens_np) - seg_lens_np
        total_rows = sample_hidden_states.shape[0]
        rows_per_req = np.minimum(seg_lens_np, np.maximum(total_rows - seg_starts, 0))
        vidx_np = np.nonzero(valid[: len(batch_req_ids)] & (rows_per_req >= 1))[0].astype(np.int64)
        if vidx_np.size == 0:
            return None
        vlen_np = rows_per_req[vidx_np].astype(np.int64)
        starts_v = seg_starts[vidx_np].astype(np.int64)
        n_rows = int(vlen_np.sum())
        req_idx_np = np.repeat(vidx_np, vlen_np)
        pos_idx_np = np.arange(n_rows, dtype=np.int64) - np.repeat(np.cumsum(vlen_np) - vlen_np, vlen_np)
        row_sel_np = np.repeat(starts_v, vlen_np) + pos_idx_np
        # All indices go through one pinned async H2D (a pageable
        # synchronous H2D would stall the main stream waiting for the
        # device to drain — the same lesson as seg_lens)
        buf = self._lwd_h2d_stage_i64(np.concatenate([req_idx_np, pos_idx_np, row_sel_np, vidx_np, vlen_np]))
        n_valid = vidx_np.size
        req_d, pos_d = buf[:n_rows], buf[n_rows : 2 * n_rows]
        sel_d = buf[2 * n_rows : 3 * n_rows]
        vidx_d = buf[3 * n_rows : 3 * n_rows + n_valid]
        vlen_d = buf[3 * n_rows + n_valid :]
        sampled_rows = sampled[req_d, pos_d]
        if n_rows == total_rows:
            # All valid with no capping: rows are a contiguous prefix; skip the gather
            lg_rows = logits[:n_rows]
            hidden_packet = sample_hidden_states[:n_rows]
        else:
            lg_rows = logits[sel_d]
            hidden_packet = sample_hidden_states[sel_d]
        thresh = lg_rows.gather(1, sampled_rows.long().unsqueeze(1))
        ranks = [(lg_rows > thresh).sum(dim=1).to(torch.int32)]
        # Stale sampled slots (requests the spec step skipped may retain
        # the previous spec step's tokens): accepted/ranks/seg_lens are
        # all capped by the actual hidden row count — the row count is
        # the ground truth of real sampling positions
        counts_dev = torch.minimum(counts_all[vidx_d], vlen_d).to(torch.int32)
        return LwdDownSelection(
            hidden_packet=hidden_packet,
            ranks=ranks,
            counts_dev=counts_dev,
            seg_lens_host=vlen_np.tolist(),
        )

    def _lwd_h2d_stage_i64(self, arr: np.ndarray) -> torch.Tensor:
        """Async H2D staging for int64 host index arrays (pinned 4-slot
        rotation + non_blocking): replaces torch.tensor(arr, device=...)
        pageable synchronous copies — those stall the main stream waiting
        for the device to drain (under MTP, the draft-propose tail).
        Ring depth rationale mirrors the meta ring: cover batch_queue
        depth + engine lag so a later host overwrite cannot land while
        the previous copy is still queued.
        Correspondence: LwdPinnedStaging.stage_to_device ≍ the original
        hand-rolled pinned+device 4-slot pair rotation (capacity floor
        1024 unchanged)."""
        return self._lwd_staging(
            "down_index",
            dtype=torch.int64,
            min_capacity=1024,
        ).stage_to_device(arr, arr.size)
