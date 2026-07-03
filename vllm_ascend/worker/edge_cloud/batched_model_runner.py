# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-model edge head_tail batched :class:`ModelRunner`.

This module is the edge-cloud batched-compute counterpart of
``NPUModelRunner``. It subclasses
``vllm_ascend.worker.model_runner_v1.NPUModelRunner`` and overrides
``initialize_kv_cache`` plus a handful of new methods
(``execute_model_pre`` / ``execute_model_batched_head`` /
``execute_model_batched_tail`` / ``execute_model_post_batched`` /
``initialize_kv_cache_post``) that drive the shared-model edge
``head_tail`` batched forward (a single ``_model_forward`` +
``compute_logits`` call replaces ``dp_size`` per-dp_rank calls).

The original ``NPUModelRunner`` (``model_runner_v1.py``) is **not
modified**. All batched-compute extensions live in this module so the
base runner keeps its single-DP semantics.
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    ModelRunnerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.utils import (
    vllm_version_is,
)
from vllm_ascend.worker.edge_cloud.execute_model_bundle import (
    _ExecuteModelBundle,
    _MergedAttnContext,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner, get_tp_context
from vllm_ascend.eplb.utils import model_register
from vllm_ascend.patch.worker.patch_draft_quarot import patch_load_weights
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)


@dataclass
class _StubCommonAttnMetadata:
    """Minimal stub carrying the fields the GDN sub-patchers
    (``_patched_build_spec`` / ``_patched_build_prefill`` /
    ``_patched_build_decode``) read off
    ``common_attn_metadata``.

    Used by ``BatchedModelRunner._merge_gdn_layer_attn`` to feed the
    patched builders when running the batched GDN path. Only
    ``query_start_loc`` (for ``.device``) and
    ``query_start_loc_cpu`` (the merged full cu_seqlen) plus
    ``num_decode_draft_tokens_cpu`` (the merged per-req spec flag)
    are needed; the rest of
    :class:`vllm.v1.attention.backend.CommonAttentionMetadata` is not
    read.
    """

    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    num_decode_draft_tokens_cpu: torch.Tensor | None = None


class BatchedModelRunner(NPUModelRunner):
    """ModelRunner for the shared-model edge ``head_tail`` batched
    compute path.

    Subclasses :class:`NPUModelRunner` and:

    - Overrides :meth:`initialize_kv_cache` with an edge-cloud
      head_tail coordination branch that registers the current
      worker's ``KVCacheConfig`` and, on the LAST caller, allocates
      ONE global KV buffer spanning all dp_ranks' blocks (the per-
      dp_rank offset / num_blocks bookkeeping lives in
      ``self._per_dp_offsets`` / ``self._per_dp_num_blocks`` /
      ``self._global_num_blocks``).
    - Adds :meth:`initialize_kv_cache_post` — runs the downstream
      hooks (drafter, KV transfer, routed-experts capturer) that
      ``initialize_kv_cache`` would normally call but that the
      head_tail branch defers (the global buffer is built before
      them so they can mirror the bound state).
    - Adds the batched-compute entry points
      :meth:`execute_model_pre` (per-dp_rank preprocess),
      :meth:`execute_model_batched_head` / ``_tail`` (single-shot
      merged forward + ``compute_logits``),
      :meth:`execute_model_post_batched` (per-dp_rank state write).
    - Adds the merged-attn helpers :meth:`_get_or_build_merged_attn_ctx`,
      :meth:`_merge_layer_attn` (per-layer ``AscendMetadata``),
      :meth:`_merge_gdn_layer_attn` (per-layer
      ``GDNAttentionMetadata``) plus the per-layer routing helpers
      :meth:`_get_bpb_for_layer` / :meth:`_is_c8_attn_layer` and the
      global-buffer allocator :meth:`_allocate_global_kv_cache_tensors`.

    The original :class:`NPUModelRunner` is unchanged. Use this class
    in place of :class:`NPUModelRunner` when constructing the shared-
    model edge ``head_tail`` worker.
    """

    # ------------------------------------------------------------------
    # Class-level sync state for shared-model-edge KV cache global remap
    # ------------------------------------------------------------------
    # When the shared-model edge runs in ``head_tail`` mode, every
    # dp_rank worker's ``BatchedModelRunner.initialize_kv_cache`` is
    # called independently and produces a per-dp_rank
    # ``KVCacheConfig``. To allow a single batched forward to address
    # the union of all dp_rank KV blocks as one global buffer, the
    # runners coordinate here:
    #
    # - ``_KV_CACHE_CONFIGS_PER_DP_RANK``: each runner registers its
    #   config under its own ``data_parallel_rank``. Last writer wins
    #   the construction.
    # - ``_KV_CACHE_CONSTRUCTED``: set by the LAST caller after the
    #   global buffer is built and bound; subsequent entries (e.g.
    #   unit tests calling ``initialize_kv_cache`` twice) short-
    #   circuit.
    #
    # These are class-level (not module-level) because the state
    # conceptually belongs to the runner class — the worker module
    # doesn't need to import ``KVCacheConfig`` or know about the
    # construction protocol. The original ``NPUModelRunner`` does not
    # have these attributes; the subclass adds them.
    _KV_CACHE_CONFIGS_PER_DP_RANK: "dict[int, KVCacheConfig]" = {}
    _KV_CACHE_CONSTRUCTED: bool = False

    # ------------------------------------------------------------------
    # KV cache init (overridden): edge-cloud head_tail coordination
    # ------------------------------------------------------------------
    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Initialize KV cache based on ``kv_cache_config``.

        Edge-cloud non-embedding_only extension (single-NPU multi-DP
        only)
        ------------------------------------------------------------
        When the edge runs in ``head_tail`` mode on a SINGLE NPU that
        hosts ``data_parallel_size`` virtual DP workers, every
        dp_rank worker receives a per-dp_rank ``KVCacheConfig`` whose
        ``num_blocks`` reflects only that dp_rank's share. To allow a
        single batched forward to address the union of all dp_rank KV
        blocks as one global buffer, this method:

        1. Registers the current worker's ``KVCacheConfig`` in the
           class-level
           :attr:`BatchedModelRunner._KV_CACHE_CONFIGS_PER_DP_RANK`
           (keyed by ``data_parallel_rank``).
        2. Checks — BEFORE calling ``_allocate_kv_cache_tensors`` —
           whether every dp_rank has registered. If not, returns
           immediately; per-dp_rank allocation is **deferred** in
           favour of the global buffer the last caller will build.
        3. If this worker is the last caller, flips
           :attr:`BatchedModelRunner._KV_CACHE_CONSTRUCTED = True`,
           computes per-dp_rank block offsets and
           ``global_num_blocks = sum(...)``, and delegates to
           :meth:`_allocate_global_kv_cache_tensors` on THIS runner
           to build the shared buffer.
        4. Propagates the shared ``self.kv_caches`` reference to
           every registered worker so all dp_ranks point at the same
           tensor.

        The "last caller" can be ANY dp_rank worker — not restricted
        to the leader. The leader's model_runner is used as the
        executor (it owns the device), but detection happens here
        without any leader-specific branch.

        This branch is only taken when ALL of the following hold:
        - ``edge_cloud_enabled``
        - ``role == "edge"``
        - ``mode != "embedding_only"`` (i.e. ``head_tail``)
        - ``is_shared_model_edge`` (single-NPU multi-DP edge layout)
        """
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self._mamba_bufs = None
        self._mamba_copy_bufs = None

        # For embedding_only edge, skip KV cache tensor allocation and
        # attention backend initialization. The edge does not execute
        # any attention layers; keeping a full kv_cache_config is only
        # for the scheduler to correctly schedule requests.
        if (
            self._edge_cloud_enabled
            and self.edge_cloud_cfg.mode == "embedding_only"
            and self.edge_cloud_cfg.role == "edge"
        ):
            self.attn_groups = [
                [] for _ in range(len(kv_cache_config.kv_cache_groups))
            ]
            self.use_hybrid_blocks = False
            self.need_accepted_tokens = False
            self.may_reinitialize_input_batch(kv_cache_config)
            self.kv_cache = {}
            logger.info(
                "[EdgeCloud] embedding_only edge skipped KV cache "
                "tensor allocation and attention backend initialization."
            )
            return

        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(
            kv_cache_config)
        # NOTE(cmq): initialize_attn_backend must before using
        # self.attn_groups
        self.initialize_attn_backend(kv_cache_config)
        self.use_hybrid_blocks = len(self.attn_groups) > 1
        # NOTE: Currently, we determine whether we need
        # ``num_accepted_tokens`` through ``MambaSpec``.
        from vllm.v1.kv_cache_interface import MambaSpec as _MambaSpec
        self.need_accepted_tokens = any(
            [isinstance(attn_group[0].kv_cache_spec, _MambaSpec)
             for attn_group in self.attn_groups])

        self.may_reinitialize_input_batch(kv_cache_config)
        # === Edge-cloud single-NPU multi-DP, head_tail mode ===
        # Only the shared-model edge topology (single NPU, multiple
        # DP ranks) needs the global KV buffer; other edge layouts
        # (multi-NPU, single-DP) fall through to the standard
        # per-dp_rank allocation below.
        if (
            self._edge_cloud_enabled
            and self.edge_cloud_cfg.role == "edge"
            and self.edge_cloud_cfg.mode != "embedding_only"
            and self.parallel_config.is_shared_model_edge
        ):
            # ``_KV_CACHE_CONFIGS_PER_DP_RANK`` is pre-registered by
            # :meth:`SharedModelEdgeWorker.initialize_from_config`
            # (which knows ``self.local_rank`` — the unique virtual
            # worker identifier in the shared-model edge topology —
            # whereas ``self.parallel_config.data_parallel_rank`` is
            # the same for every virtual worker on the edge since
            # they all share the same NPU). The model_runner only
            # READS the registry here.
            dp_size = self.parallel_config.data_parallel_size
            if (len(
                    BatchedModelRunner._KV_CACHE_CONFIGS_PER_DP_RANK)
                    < dp_size):
                logger.info(
                    "[EdgeCloud] head_tail shared-model edge: "
                    "deferred allocation (%d/%d dp_ranks registered); "
                    "waiting for the last caller.",
                    len(BatchedModelRunner
                        ._KV_CACHE_CONFIGS_PER_DP_RANK),
                    dp_size,
                )
                return
            if BatchedModelRunner._KV_CACHE_CONSTRUCTED:
                return

            # Claim the construction slot.
            BatchedModelRunner._KV_CACHE_CONSTRUCTED = True

            # Per-dp_rank block layout: dp_k's local physical block
            # N maps to global physical block ``N * dp_size + k``
            # (interleaved by dp_rank).
            sorted_configs = sorted(
                BatchedModelRunner._KV_CACHE_CONFIGS_PER_DP_RANK.items(),
                key=lambda kv: kv[0])
            per_dp_offsets: dict[int, int] = {}
            per_dp_num_blocks: dict[int, int] = {}
            for rank, cfg in sorted_configs:
                per_dp_offsets[rank] = rank
                per_dp_num_blocks[rank] = cfg.num_blocks
            template_num_blocks = min(
                cfg.num_blocks for cfg in
                BatchedModelRunner._KV_CACHE_CONFIGS_PER_DP_RANK.values())
            global_num_blocks = template_num_blocks * dp_size

            # Persist offsets on this runner.
            self._per_dp_offsets = per_dp_offsets
            self._per_dp_num_blocks = per_dp_num_blocks
            self._global_num_blocks = global_num_blocks

            # Allocate ONE global KV buffer on THIS runner (the last
            # caller). ``bind_kv_cache`` mutates
            # ``compilation_config.static_forward_context`` which is
            # shared across all dp_rank runners.
            kv_caches = self._allocate_global_kv_cache_tensors(
                BatchedModelRunner._KV_CACHE_CONFIGS_PER_DP_RANK)
            return
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)
        # TODO: refactor the logic of attention
        # Initialize drafter attention group initialization
        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
        ):
            assert isinstance(
                self.drafter,
                AscendEagleProposer | AscendDflashProposer
                | AscendDraftModelProposer)
            block_size = (
                self.kernel_block_sizes[0]
                if isinstance(self.kernel_block_sizes, list)
                else self.kernel_block_sizes)
            self.drafter.initialize_attn_backend(
                self.kv_cache_config, block_size)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)

        if self.model_config.enable_return_routed_experts:
            self.init_routed_experts_capturer()

    def initialize_kv_cache_post(self):
        """Run the post-allocation KV-cache hooks.

        Does NOT call ``initialize_kv_cache_tensors`` (already done
        by :meth:`_allocate_global_kv_cache_tensors` for the
        head_tail branch, or by :meth:`initialize_kv_cache` for the
        standard path). Only runs the downstream hooks (drafter,
        routed-experts capturer) — the KV transfer registration is
        handled by the head_tail branch's
        :meth:`_allocate_global_kv_cache_tensors` directly.
        """
        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
        ):
            assert isinstance(
                self.drafter,
                AscendEagleProposer | AscendDflashProposer
                | AscendDraftModelProposer)
            block_size = (
                self.kernel_block_sizes[0]
                if isinstance(self.kernel_block_sizes, list)
                else self.kernel_block_sizes)
            self.drafter.initialize_attn_backend(
                self.kv_cache_config, block_size)

        if self.model_config.enable_return_routed_experts:
            self.init_routed_experts_capturer()

    # ------------------------------------------------------------------
    # Global KV buffer allocator (head_tail last caller)
    # ------------------------------------------------------------------
    def _allocate_global_kv_cache_tensors(
        self,
        kv_cache_config_per_dp: dict[int, KVCacheConfig],
    ) -> dict[str, torch.Tensor]:
        """Allocate ONE global KV buffer spanning every dp_rank's
        blocks.

        Called once on the LAST caller's model_runner by
        :meth:`initialize_kv_cache` after every dp_rank has registered
        its ``KVCacheConfig``.

        Implementation
        --------------
        1. Pick the smallest ``num_blocks`` across dp_ranks as the
           "template" config — it has the smallest per-tensor
           ``size`` for each layer (since ``size`` is linear in
           ``num_blocks``).
        2. Build a synthetic ``KVCacheConfig`` that mirrors the
           template but with ``num_blocks = global_num_blocks`` and
           each ``kv_cache_tensor.size`` scaled by
           ``global_num_blocks / template.num_blocks``.
        3. Delegate to the existing
           :meth:`initialize_kv_cache_tensors` to allocate + bind
           the global buffer. The result populates the leader's
           ``self.kv_caches`` and binds the forward context exactly
           once.

        Returns the global ``kv_caches`` dict (``{layer_name: tensor}``)
        for callers that want to introspect it; ``self.kv_caches`` is
        also populated as a side effect.
        """
        if not kv_cache_config_per_dp:
            raise RuntimeError(
                "_allocate_global_kv_cache_tensors called with empty "
                "kv_cache_config_per_dp")

        # 1. Pick the template config (smallest num_blocks). All
        #    configs share the same layer spec / block_size /
        #    dtype / head layout, only ``num_blocks`` and the
        #    resulting ``size`` differ.
        template_dp_rank = min(
            kv_cache_config_per_dp,
            key=lambda k: kv_cache_config_per_dp[k].num_blocks)
        template_cfg = kv_cache_config_per_dp[template_dp_rank]
        global_num_blocks = sum(
            cfg.num_blocks for cfg in kv_cache_config_per_dp.values())
        if global_num_blocks <= template_cfg.num_blocks:
            raise RuntimeError(
                "Global num_blocks (%d) must exceed template "
                "num_blocks (%d, dp_rank=%d). This indicates a "
                "dp_rank did not register a positive num_blocks."
                % (global_num_blocks, template_cfg.num_blocks,
                   template_dp_rank))

        # 2. Build the synthetic config with scaled sizes.
        scale = global_num_blocks / template_cfg.num_blocks
        scaled_tensors: list[KVCacheTensor] = []
        for tensor in template_cfg.kv_cache_tensors:
            scaled_tensors.append(
                KVCacheTensor(
                    size=int(tensor.size * global_num_blocks
                             // template_cfg.num_blocks),
                    shared_by=list(tensor.shared_by),
                ))
        synthetic = replace(
            deepcopy(template_cfg),
            num_blocks=global_num_blocks,
            kv_cache_tensors=scaled_tensors,
        )
        # Run the SAME init chain as the standard
        # ``initialize_kv_cache`` path.
        kv_caches = self.initialize_kv_cache_tensors(synthetic)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)

        logger.info(
            "[EdgeCloud] head_tail shared-model edge: allocated "
            "global KV buffer of %d blocks (template dp_rank=%d, "
            "%d blocks; scale=%.2fx across %d dp_ranks).",
            global_num_blocks, template_dp_rank,
            template_cfg.num_blocks, scale,
            len(kv_cache_config_per_dp))
        return kv_caches

    def bind_to_shared_model(self, model: nn.Module) -> None:
        """Bind this runner to a model object loaded by another runner.

        Used by ``SharedModelEdgeWorker`` follower workers to share a single
        ``nn.Module`` instance across multiple model runners in the same
        process. Replaces ``self.model`` with ``model`` and mirrors the
        post-model-creation side effects of :meth:`load_model`:

        - edge-cloud: re-derives ``num_layers`` and re-creates the segment
          callables from the shared (already sharded) model.
        - dynamic EPLB: registers the shared model.
        - drafter: runs the drafter ``load_model`` hook against the shared
          model and sets the eagle3 aux hidden state layers.
        - ACLGraphWrapper: wraps the shared model if cudagraph mode is on.
        - profiler: starts the data dump if cudagraph mode is on.

        The caller is responsible for:

        - ensuring ``model`` has already been fully loaded by another
          runner in the same process (i.e. the leader
          ``SharedModelEdgeWorker``);
        - assigning ``self.model_memory_usage`` after binding, because only
          the leader's profile run actually measures it.
        """
        self.model = model

        # Edge-cloud specific state, derived from the (already sharded) model.
        if self._edge_cloud_enabled:
            # Locate the transformer layers — the model may be wrapped in a
            # multimodal ConditionalGeneration (language_model.model.layers)
            # or be a plain CausalLM (model.layers).
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                transformer_layers = self.model.model.layers
            else:
                transformer_layers = self.model.language_model.model.layers
            self.num_layers = len(transformer_layers)
            if hasattr(self.model, "set_moe_parameters"):
                self.model.set_moe_parameters()
            if self.edge_cloud_cfg.role == "edge":
                self.segment_a = self._create_segment_callable(
                    self.model,
                    0,
                    self.head_k,
                    is_first_segment=True,
                    is_last_segment=False,
                )
                self.segment_e = self._create_segment_callable(
                    self.model,
                    self.num_layers - self.tail_k,
                    self.num_layers,
                    is_first_segment=False,
                    is_last_segment=True,
                )
                self.segment_a_wrapper = self._wrap_segment_if_needed(
                    self.segment_a)
                self.segment_e_wrapper = self._wrap_segment_if_needed(
                    self.segment_e)
            else:
                self.segment_c = self._create_segment_callable(
                    self.model,
                    self.head_k,
                    self.num_layers - self.tail_k,
                    is_first_segment=False,
                    is_last_segment=False,
                )
                self.segment_c_wrapper = self._wrap_segment_if_needed(
                    self.segment_c)

        # Standard post-model-creation side effects.
        if self.dynamic_eplb:
            model_register(self.model)
        if self.drafter:
            logger.info("Loading drafter model for shared model binding...")
            if self.vllm_config.quant_config is not None:
                patch_load_weights(self.vllm_config)
            with get_tp_context(self.drafter):
                self.drafter.load_model(self.model)
            if self.use_aux_hidden_state_outputs:
                from vllm.model_executor.models.interfaces import supports_eagle3
                if not supports_eagle3(self.model):
                    raise RuntimeError(
                        "Model does not support EAGLE3 interface but "
                        "aux_hidden_state_outputs was requested"
                    )
                aux_layers = self._get_eagle3_aux_layers_from_config()
                if not aux_layers:
                    aux_layers = self.model.get_eagle3_default_aux_hidden_state_layers()
                self.model.set_aux_hidden_state_layers(aux_layers)

        # wrap the model with full graph wrapper if needed.
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            if not isinstance(self.model, ACLGraphWrapper):
                self.update_stream: torch.npu.Stream = torch.npu.Stream()
                self.model = ACLGraphWrapper(
                    self.model,
                    self.vllm_config,
                    runtime_mode=CUDAGraphMode.FULL,
                    use_eagle=self.use_eagle,
                    enable_enpu=self.enable_enpu,
                )

        if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            self._start_dump_data()

    # ------------------------------------------------------------------
    # Batched compute entry points
    # ------------------------------------------------------------------
    def execute_model_pre(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> "_ExecuteModelBundle | ModelRunnerOutput | None":
        """Per-dp_rank preprocess for the batched compute path.

        Mirrors the prep portion of ``NPUModelRunner.execute_model``
        (everything BEFORE ``set_ascend_forward_context``). Updates
        ``self.input_batch``. Always runs the standard path
        (segment_e / cloud fast paths are reserved for the
        second call into ``execute_model`` when the cloud has
        already produced ``intermediate_tensors``; here
        ``intermediate_tensors`` is always ``None``).

        For empty / no-work cases, returns the same early
        ``ModelRunnerOutput`` / ``None`` values as the original
        ``execute_model``; the busy_loop detects these via
        ``handle_output`` and skips the batched forward. Otherwise
        returns a :class:`_ExecuteModelBundle` consumed by
        ``execute_model_batched_head`` / ``_tail`` / ``_post_batched``.

        Does NOT call ``set_ascend_forward_context``, ``_model_forward``
        or ``compute_logits``. Does NOT write
        ``self.execute_model_state`` (that is the job of
        ``execute_model_post_batched``). Does NOT populate
        ``self._edge_prepare_cache`` (the head-tail busy_loop
        doesn't reuse the segment_e fast path — every round runs
        a fresh ``execute_model_pre``).
        """
        if self.vllm_config.model_config.enable_return_routed_experts:
            if vllm_version_is("0.20.2"):
                capturer = RoutedExpertsCapturer.get_instance()
                if capturer is not None:
                    capturer.clear_buffer()
            elif self.routed_experts_initialized:
                self.routed_experts_capturer.clear_buffer()

        if self.ascend_config.profiling_chunk_config.need_timing:
            if getattr(scheduler_output, "disable_profiling_timing",
                       False):
                self.ascend_config.profiling_chunk_config.need_timing = (
                    False)
            else:
                self._sync_device()
                self._execution_start_time = time.perf_counter()
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called after "
                "execute_model() returns None.")

        if (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        ):
            num_scheduled_tokens_copy = (
                scheduler_output.num_scheduled_tokens.copy())
            spec_decode_tokens_copy = (
                scheduler_output.scheduled_spec_decode_tokens.copy())
            scheduler_output = replace(
                scheduler_output,
                num_scheduled_tokens=num_scheduled_tokens_copy,
                scheduled_spec_decode_tokens=spec_decode_tokens_copy,
            )

        self._start_dump_data()
        if ((
            self.use_async_scheduling and self.num_spec_tokens
                and self._draft_token_ids is None
        ) or (
            self.pcp_size > 1 and self.supports_mm_inputs
            and get_pp_group().is_first_rank
            and not self.model_config.is_encoder_decoder
        )):
            scheduler_output = deepcopy(scheduler_output)
        num_scheduled_tokens = (
            scheduler_output.total_num_scheduled_tokens)
        with record_function_or_nullcontext("prepare input"):
            with self.synchronize_input_prep():
                if (
                    self.use_async_scheduling
                    and self.num_spec_tokens
                    and self.input_batch.prev_req_id_to_index
                    is not None
                ):
                    for req_id in (
                            scheduler_output.scheduled_cached_reqs
                            .req_ids):
                        if (
                            req_id
                            not in self.input_batch.prev_req_id_to_index
                            and (req_state := self.requests.get(req_id))
                                is not None
                            and req_state.prev_num_draft_len
                        ):
                            req_state.prev_num_draft_len = 0

                deferred_state_corrections_fn = self._update_states(
                    scheduler_output)

                if has_ec_transfer() and get_ec_transfer().is_producer:
                    with self.maybe_get_ec_connector_output(
                        scheduler_output,
                        encoder_cache=self.encoder_cache,
                    ) as ec_connector_output:
                        self._execute_mm_encoder(scheduler_output)
                        self._finalize_dump_data()
                    return make_empty_encoder_model_runner_output(
                        scheduler_output)

                num_reqs = self.input_batch.num_reqs
                req_ids = self.input_batch.req_ids
                tokens = [
                    scheduler_output.num_scheduled_tokens[i]
                    for i in req_ids]
                num_scheduled_tokens_np = np.array(
                    tokens, dtype=np.int32)

                if not num_scheduled_tokens:
                    if (
                        self.parallel_config.distributed_executor_backend
                        == "external_launcher"
                        and self.parallel_config.data_parallel_size > 1
                    ):
                        self._dummy_run(1)
                    if not has_kv_transfer_group():
                        return EMPTY_MODEL_RUNNER_OUTPUT
                    return self.kv_connector_no_forward(
                        scheduler_output, self.vllm_config)
                if self.cache_config.kv_sharing_fast_prefill:
                    assert not self.num_prompt_logprobs, (
                        "--kv-sharing-fast-prefill produces incorrect "
                        "logprobs for prompt tokens, tokens, please "
                        "disable it when the requests need prompt "
                        "logprobs")

                # Apply deferred state corrections, then mamba
                # preprocess (must run after _update_states, before
                # input preparation).
                if deferred_state_corrections_fn:
                    deferred_state_corrections_fn()
                    deferred_state_corrections_fn = None
                if self.cache_config.mamba_cache_mode == "align":
                    if vllm_version_is("0.20.2"):
                        mamba_bufs = self._get_mamba_copy_bufs()
                        preprocess_bufs = mamba_bufs
                    else:
                        mamba_bufs = self._get_mamba_bufs()
                        preprocess_bufs = mamba_bufs.preprocess
                    mamba_utils.preprocess_mamba(
                        scheduler_output,
                        self.kv_cache_config,
                        self.cache_config,
                        self.mamba_state_idx,
                        self.input_batch,
                        self.requests,
                        self.compilation_config.static_forward_context,
                        self.model.get_mamba_state_copy_func(),
                        preprocess_bufs,
                    )
                    self.num_accepted_tokens.np[:num_reqs] = (
                        self.input_batch.num_accepted_tokens_cpu[
                            :num_reqs])
                    self.num_accepted_tokens.copy_to_gpu(num_reqs)
                    if (not vllm_version_is("0.20.2")
                            and mamba_bufs.postprocess_align is not None):
                        mamba_utils.stage_postprocess_inputs_to_gpu(
                            mamba_bufs.postprocess_align,
                            scheduler_output,
                            self.input_batch.req_ids,
                            num_reqs,
                            self.requests,
                            self.mamba_state_idx,
                        )
                if self.use_compress:
                    if deferred_state_corrections_fn:
                        deferred_state_corrections_fn()
                        deferred_state_corrections_fn = None
                    num_reqs = self.input_batch.num_reqs
                    req_indices = np.repeat(
                        self.arange_np[:num_reqs],
                        num_scheduled_tokens_np)
                    dsa_positions_np = (
                        self._dsa_positions_np_buf[
                            :total_num_scheduled_tokens])
                    np.add(
                        self.input_batch.num_computed_tokens_cpu[
                            req_indices],
                        self.query_pos.np[:total_num_scheduled_tokens],
                        out=dsa_positions_np,
                    )

                # Run core input preparation.
                cache = self._run_input_preparation(scheduler_output)
                total_num_scheduled_tokens = (
                    cache["total_num_scheduled_tokens"])
                num_tokens_padded = cache["num_tokens_padded"]
                num_tokens_across_dp = cache["num_tokens_across_dp"]
                attn_metadata = cache["attn_metadata"]
                logits_indices = cache["logits_indices"]
                spec_decode_metadata = cache["spec_decode_metadata"]
                spec_decode_common_attn_metadata = (
                    cache["spec_decode_common_attn_metadata"])
                cudagraph_mode = cache["cudagraph_mode"]
                batch_desc = cache["batch_desc"]
                cudagraph_stats = cache["cudagraph_stats"]

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output,
                num_tokens_padded
                if not (self.use_cp
                        and self.pcp_manager.pcp_use_hybrid_attn)
                else total_num_scheduled_tokens,
                None,
            )

            if not self.edge_cloud_cfg.role == "edge":
                # update global cos, sin
                update_cos_sin(positions)

        if self.dynamic_eplb:
            with record_function_or_nullcontext("EPLB weight D2D"):
                self.eplb_updator.forward_before()

        # Set cudagraph mode to none if calc_kv_scales is true.
        if self.calculate_kv_scales:
            cudagraph_mode = CUDAGraphMode.NONE
            self.calculate_kv_scales = False
        if self.ascend_config.enable_async_exponential:
            self.sampler.do_async_exponential(
                b_s=logits_indices.shape[0],
                head_dim=self.model_config.get_vocab_size(),
                generators=self.input_batch.sampling_metadata.generators,
            )

        return _ExecuteModelBundle(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=None,
            hidden_states=None,
            logits_indices=logits_indices,
            spec_decode_metadata=spec_decode_metadata,
            spec_decode_common_attn_metadata=(
                spec_decode_common_attn_metadata),
            scheduler_output=scheduler_output,
            num_tokens_padded=num_tokens_padded,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_mode=cudagraph_mode,
            batch_desc=batch_desc,
            attn_metadata=attn_metadata,
            ec_connector_output=ec_connector_output,
            cudagraph_stats=cudagraph_stats,
            deferred_state_corrections_fn=(
                deferred_state_corrections_fn),
        )

    def execute_model_batched_head(
        self,
        bundles: list[_ExecuteModelBundle],
        batched_dp_ranks: list[int] | None = None,
    ) -> list[IntermediateTensors]:
        """1 batched head model_forward = 1 ``model.embed_tokens``.

        Called ONCE per round from
        :class:`SharedModelWorkerProc.worker_busy_loop`. The forward
        context is set up for the merged batch; ``_model_forward``
        runs once on the shared ``nn.Module``; the resulting
        ``hidden_states`` is sliced back to per-dp_rank.

        Returns per-dp_rank ``IntermediateTensors`` slices, padded to
        ``self.max_num_tokens`` in the ``embedding_only`` mode so
        that the cloud's pre-allocated buffer is large enough.
        """
        # Per-bundle actual (non-padded) token counts.
        n_actuals = [
            (b.attn_metadata[0][
                next(iter(b.attn_metadata[0]))].num_actual_tokens
             if isinstance(b.attn_metadata, list) and b.attn_metadata
             else next(iter(b.attn_metadata.values()))
             .num_actual_tokens)
            for b in bundles
        ]

        num_tokens_merged = sum(n_actuals)
        num_tokens_across_dp_merged = None

        any_bundle = bundles[0]
        is_non_embedding_only_edge = (
            self._edge_cloud_enabled
            and self.edge_cloud_cfg.role == "edge"
            and self.edge_cloud_cfg.mode != "embedding_only")
        if is_non_embedding_only_edge:
            assert batched_dp_ranks is not None, (
                "execute_model_batched_head: batched_dp_ranks "
                "required for non-embedding_only edge mode.")
            ctx = self._get_or_build_merged_attn_ctx(
                bundles, batched_dp_ranks)
            head_attn_metadata: Any = ctx.merged_attn_metadata
            batch_descriptor = ctx.merged_batch_descriptor
            num_tokens_padded_merged = ctx.num_tokens_padded_merged
            merged_input_ids = ctx.merged_input_ids
            merged_positions = ctx.merged_positions
            merged_inputs_embeds = ctx.merged_inputs_embeds
        else:
            head_attn_metadata = any_bundle.attn_metadata
            batch_descriptor = any_bundle.batch_desc
            num_tokens_padded_merged = sum(
                b.num_tokens_padded for b in bundles)
            if all(b.input_ids is not None for b in bundles):
                merged_input_ids = torch.cat(
                    [b.input_ids[:n]
                     for b, n in zip(bundles, n_actuals)])
            else:
                merged_input_ids = None
            if all(b.positions is not None for b in bundles):
                cat_dim = bundles[0].positions.dim() - 1
                merged_positions = torch.cat(
                    [b.positions[..., :n]
                     for b, n in zip(bundles, n_actuals)],
                    dim=cat_dim)
            else:
                merged_positions = None
            if all(b.inputs_embeds is not None for b in bundles):
                merged_inputs_embeds = torch.cat(
                    [b.inputs_embeds[:n]
                     for b, n in zip(bundles, n_actuals)])
            else:
                merged_inputs_embeds = None
        with (
            record_function_or_nullcontext("forward"),
            set_ascend_forward_context(
                attn_metadata=head_attn_metadata,
                vllm_config=self.vllm_config,
                num_tokens=num_tokens_padded_merged,
                num_tokens_across_dp=num_tokens_across_dp_merged,
                aclgraph_runtime_mode=any_bundle.cudagraph_mode,
                batch_descriptor=batch_descriptor,
                num_actual_tokens=num_tokens_merged,
                model_instance=self.model,
                max_tokens_across_pcp=(
                    0 if self.pcp_size == 1
                    else self.pcp_manager
                    .max_num_tokens_across_pcp),
                skip_compiled=False,
            ),
        ):
            hidden_states = self._model_forward(
                num_tokens_padded_merged,
                merged_input_ids,
                merged_positions,
                None,
                merged_inputs_embeds,
            )

        token_offsets = [0]
        for n in n_actuals:
            token_offsets.append(token_offsets[-1] + n)
        results: list[IntermediateTensors] = []
        head_hidden = hidden_states["hidden_states"]
        head_residual = hidden_states["residual"]
        for i, b in enumerate(bundles):
            slice_hs = head_hidden[
                token_offsets[i]:token_offsets[i + 1]]
            slice_res = head_residual[
                token_offsets[i]:token_offsets[i + 1]]
            if (self.edge_cloud_cfg.mode == "embedding_only"
                    and slice_hs.shape[0] < self.max_num_tokens):
                pad = torch.zeros(
                    self.max_num_tokens - slice_hs.shape[0],
                    *slice_hs.shape[1:],
                    dtype=slice_hs.dtype,
                    device=slice_hs.device,
                )
                slice_hs = torch.cat([slice_hs, pad], dim=0)
                slice_res = torch.cat([slice_res, pad], dim=0)
            results.append(
                IntermediateTensors({
                    "hidden_states": slice_hs,
                    "residual": slice_res,
                }))
        return results

    def execute_model_batched_tail(
        self,
        bundles: list[_ExecuteModelBundle],
        intermediates: list[IntermediateTensors],
        batched_dp_ranks: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """1 batched tail model_forward + 1 batched ``compute_logits``.

        Called ONCE per round on the leader runner
        (``self.worker[0].model_runner``) to avoid duplicated forward
        across dp_ranks. Returns a 4-tuple of:

        * ``merged_hidden_states`` — the full [merged_num_tokens]
          hidden_states from the batched tail ``_model_forward``.
        * ``merged_sample_hidden_states`` — ``hidden_states`` indexed
          by ``logits_indices`` (per-sample rows).
        * ``merged_logits`` — the per-sample logits from
          ``compute_logits``.
        * ``kv_connector_output`` — the captured kv-connector output.
        """
        # Per-bundle actual token counts (defensive trim of the
        # cloud-returned intermediates).
        n_actuals_tail = [
            (b.attn_metadata[0][
                next(iter(b.attn_metadata[0]))].num_actual_tokens
             if isinstance(b.attn_metadata, list) and b.attn_metadata
             else next(iter(b.attn_metadata.values()))
             .num_actual_tokens)
            for b in bundles
        ]
        if all(it["hidden_states"] is not None
               for it in intermediates):
            merged_hidden = torch.cat(
                [it["hidden_states"][:n]
                 for it, n in zip(intermediates, n_actuals_tail)])
        else:
            merged_hidden = None
        if all(it["residual"] is not None for it in intermediates):
            merged_residual = torch.cat(
                [it["residual"][:n]
                 for it, n in zip(intermediates, n_actuals_tail)])
        else:
            merged_residual = None
        merged_intermediate = IntermediateTensors({
            "hidden_states": merged_hidden,
            "residual": merged_residual,
        })

        token_offsets = [0]
        for n in n_actuals_tail:
            token_offsets.append(token_offsets[-1] + n)
        merged_logits_indices = torch.cat(
            [bundles[i].logits_indices + token_offsets[i]
             for i in range(len(bundles))])

        any_bundle = bundles[0]
        is_non_embedding_only_edge = (
            self._edge_cloud_enabled
            and self.edge_cloud_cfg.role == "edge"
            and self.edge_cloud_cfg.mode != "embedding_only")
        if is_non_embedding_only_edge:
            assert batched_dp_ranks is not None, (
                "execute_model_batched_tail: batched_dp_ranks "
                "required for non-embedding_only edge mode.")
            ctx = self._get_or_build_merged_attn_ctx(
                bundles, batched_dp_ranks)
            tail_attn_metadata: Any = ctx.merged_attn_metadata
            batch_descriptor = ctx.merged_batch_descriptor
            num_tokens_padded_merged = ctx.num_tokens_padded_merged
        else:
            tail_attn_metadata = any_bundle.attn_metadata
            batch_descriptor = any_bundle.batch_desc
            num_tokens_padded_merged = sum(
                b.num_tokens_padded for b in bundles)
        kv_connector_output = None
        num_tokens_merged = sum(n_actuals_tail)
        num_tokens_across_dp_merged = None
        with (
            record_function_or_nullcontext("forward"),
            set_ascend_forward_context(
                attn_metadata=tail_attn_metadata,
                vllm_config=self.vllm_config,
                num_tokens=num_tokens_padded_merged,
                num_tokens_across_dp=num_tokens_across_dp_merged,
                aclgraph_runtime_mode=any_bundle.cudagraph_mode,
                batch_descriptor=batch_descriptor,
                num_actual_tokens=num_tokens_merged,
                model_instance=self.model,
                max_tokens_across_pcp=(
                    0 if self.pcp_size == 1
                    else self.pcp_manager
                    .max_num_tokens_across_pcp),
                skip_compiled=False,
            ),
            self.maybe_get_kv_connector_output(
                any_bundle.scheduler_output,
                **{"defer_finalize": True},
            ) as kv_connector_output,
        ):
            hidden_states = self._model_forward(
                num_tokens_padded_merged,
                (ctx.merged_input_ids
                 if (is_non_embedding_only_edge
                     and ctx.merged_input_ids is not None)
                 else any_bundle.input_ids),
                (ctx.merged_positions
                 if (is_non_embedding_only_edge
                     and ctx.merged_positions is not None)
                 else any_bundle.positions),
                merged_intermediate,
                (ctx.merged_inputs_embeds
                 if (is_non_embedding_only_edge
                     and ctx.merged_inputs_embeds is not None)
                 else any_bundle.inputs_embeds),
            )

        merged_sample_hidden_states = (
            hidden_states[merged_logits_indices])
        merged_logits = self.model.compute_logits(
            merged_sample_hidden_states)

        return (hidden_states, merged_sample_hidden_states,
                merged_logits, kv_connector_output)

    def execute_model_post_batched(
        self,
        bundle: _ExecuteModelBundle,
        sample_hidden_states_k: torch.Tensor,
        logits_k: torch.Tensor,
        hidden_states_k: torch.Tensor,
        kv_connector_output: Any,
    ) -> None:
        """Per-dp_rank state writing after batched ``compute_logits``.

        Mirrors the tail portion of ``NPUModelRunner.execute_model``:
        writes ``self.execute_model_state`` and
        ``self.kv_connector_output`` and triggers the deferred state
        corrections from ``_update_states``.

        Strictly does NOT include any ``sample_tokens`` logic:
        no ``apply_grammar_bitmask``, no ``_sample``, no
        ``_bookkeeping_sync``, no ``propose_draft_token_ids``, no
        ``ModelRunnerOutput`` return. It also does NOT call
        ``set_ascend_forward_context`` / ``_model_forward`` /
        ``compute_logits`` (those live in
        ``execute_model_batched_tail``).
        """
        # ``ExecuteModelState`` is a NamedTuple defined in the base
        # model_runner_v1 module — use the same constructor.
        from vllm_ascend.worker.model_runner_v1 import (
            ExecuteModelState,
        )
        self.execute_model_state = ExecuteModelState(
            scheduler_output=bundle.scheduler_output,
            logits=logits_k,
            spec_decode_metadata=bundle.spec_decode_metadata,
            spec_decode_common_attn_metadata=(
                bundle.spec_decode_common_attn_metadata),
            hidden_states=hidden_states_k,
            sample_hidden_states=sample_hidden_states_k,
            aux_hidden_states=None,
            attn_metadata=bundle.attn_metadata,
            positions=bundle.positions,
            ec_connector_output=bundle.ec_connector_output,
            cudagraph_stats=bundle.cudagraph_stats,
            batch_desc=bundle.batch_desc,
        )
        self.kv_connector_output = kv_connector_output
        if bundle.deferred_state_corrections_fn:
            bundle.deferred_state_corrections_fn()

    # ------------------------------------------------------------------
    # Merged attn ctx (head_tail shared-model edge batched forward)
    # ------------------------------------------------------------------
    def _get_bpb_for_layer(self, layer_name: str) -> int:
        """Look up ``blocks_per_phys_block`` for the kv_cache_group
        this layer belongs to.

        Each kv_cache_group has its own ``BlockTable`` with its own
        ``blocks_per_phys_block`` (see
        vllm_ascend/worker/block_table.py:47,67,72). Attention
        groups in hybrid mode have ``bpb > 1``; MambaSpec groups
        always have ``bpb == 1`` (``kernel_sizes=[0]``).
        """
        for group_idx, group in enumerate(
                self.kv_cache_config.kv_cache_groups):
            if layer_name in group.layer_names:
                return (
                    self.input_batch.block_table
                    .block_tables[group_idx].blocks_per_phys_block
                )
        raise KeyError(
            f"Layer {layer_name!r} not found in any kv_cache_group")

    def _is_c8_attn_layer(self, layer_name: str) -> bool:
        """Return whether ``layer_name`` uses the C8 (INT8 KV cache)
        backend.

        The C8 backend
        (``vllm_ascend.attention.attention_v1.AscendC8AttentionBackendImpl``)
        explicitly splits the merged batch into decode / prefill
        sub-batches in Python
        (``_forward_c8_chunked_prefill``). That split assumes a
        contiguous ``[decodes…][prefills…]`` layout, which the
        simple ``block_tables`` / ``slot_mapping`` /
        ``actual_seq_lengths_q`` concat in ``_merge_layer_attn`` does
        NOT guarantee when different dp_ranks have different
        per-dp_rank ``attn_state``. We therefore refuse C8 layers
        in the batched path; per-dp_rank execution handles them.
        """
        # The edge cloud shared model never runs through a
        # C8-quantized attention layer today, so returning False is
        # correct in practice.
        del layer_name
        return False

    def _get_or_build_merged_attn_ctx(
        self,
        bundles: list["_ExecuteModelBundle"],
        batched_dp_ranks: list[int],
    ) -> "_MergedAttnContext":
        """Build (or reuse cached) merged per-layer attn state.

        ``bundle.attn_metadata`` is a per-layer dict
        (``AttnMetadataDict = dict[layer_name, AscendMetadata]``);
        the attention kernel reads it as
        ``forward_context.attn_metadata[layer]`` per layer. We merge
        **per layer** across dp_ranks so that the result is itself a
        per-layer dict that the kernel can consume unchanged.

        Called once per round on the LEADER runner by
        ``_BatchedExecuteMarker.run_batched_head``;
        ``drain_batched_round`` reuses the cache for the tail
        segment.
        """
        cached = getattr(self, "_merged_attn_ctx_cache", None)
        if cached is not None:
            return cached

        per_dp_offsets = getattr(self, "_per_dp_offsets", None)
        if per_dp_offsets is None:
            raise RuntimeError(
                "BatchedModelRunner._get_or_build_merged_attn_ctx "
                "called but self._per_dp_offsets is unset. Did "
                "initialize_kv_cache run for all dp_ranks?")

        def _layer_dict(b: "_ExecuteModelBundle") -> dict:
            md = b.attn_metadata
            return md[0] if isinstance(md, list) else md

        template_layers = list(_layer_dict(bundles[0]).keys())
        for b in bundles[1:]:
            assert _layer_dict(b).keys() == set(template_layers), (
                "dp_rank attn_metadata layer sets must agree across "
                "dp_ranks for batched forward")

        merged_attn_metadata: dict = {}
        from vllm_ascend.attention.attention_v1 import AscendMetadata
        from vllm.v1.attention.backends.gdn_attn import (
            GDNAttentionMetadata,
        )

        def _merge_layer(layer_name: str):
            md_template = _layer_dict(bundles[0])[layer_name]
            bpb = self._get_bpb_for_layer(layer_name)
            if isinstance(md_template, AscendMetadata):
                return self._merge_layer_attn(
                    layer_name, bundles, batched_dp_ranks,
                    per_dp_offsets, bpb)
            elif isinstance(md_template, GDNAttentionMetadata):
                return self._merge_gdn_layer_attn(
                    layer_name, bundles, batched_dp_ranks,
                    per_dp_offsets, bpb)
            else:
                raise NotImplementedError(
                    f"Unsupported attention metadata type for "
                    f"layer {layer_name!r}: "
                    f"{type(md_template).__name__}; batched forward "
                    f"currently supports AscendMetadata and "
                    f"GDNAttentionMetadata only.")

        for layer_name in template_layers:
            merged_attn_metadata[layer_name] = _merge_layer(
                layer_name)

        first_merged = next(iter(merged_attn_metadata.values()))
        num_tokens_merged = first_merged.num_actual_tokens
        any_batch_desc = bundles[0].batch_desc
        if any_batch_desc is None:
            merged_batch_descriptor = None
            num_tokens_padded_merged = num_tokens_merged
        else:
            has_lora_any = any(
                b.batch_desc is not None and b.batch_desc.has_lora
                for b in bundles)
            uniform_decode_merged = all(
                b.batch_desc is not None and b.batch_desc.uniform
                for b in bundles)
            num_active_loras_merged = sum(
                (b.batch_desc.num_active_loras
                 if b.batch_desc is not None else 0)
                for b in bundles)
            _, merged_batch_descriptor = (
                self.cudagraph_dispatcher.dispatch(
                    num_tokens=num_tokens_merged,
                    has_lora=has_lora_any,
                    uniform_decode=uniform_decode_merged,
                    valid_modes=None,
                    invalid_modes={CUDAGraphMode.FULL},
                    num_active_loras=num_active_loras_merged,
                ))
            num_tokens_padded_merged = (
                merged_batch_descriptor.num_tokens)

        def _actual_tokens(b: "_ExecuteModelBundle") -> int:
            md = b.attn_metadata
            if isinstance(md, list):
                md = md[0][next(iter(md[0]))]
            else:
                md = next(iter(md.values()))
            return md.num_actual_tokens

        n_actuals = [_actual_tokens(b) for b in bundles]
        if all(b.input_ids is not None for b in bundles):
            merged_input_ids_ctx = torch.cat(
                [b.input_ids[:n]
                 for b, n in zip(bundles, n_actuals)])
        else:
            merged_input_ids_ctx = None
        if all(b.positions is not None for b in bundles):
            cat_dim_ctx = bundles[0].positions.dim() - 1
            merged_positions_ctx = torch.cat(
                [b.positions[..., :n]
                 for b, n in zip(bundles, n_actuals)],
                dim=cat_dim_ctx)
        else:
            merged_positions_ctx = None
        if all(b.inputs_embeds is not None for b in bundles):
            merged_inputs_embeds_ctx = torch.cat(
                [b.inputs_embeds[:n]
                 for b, n in zip(bundles, n_actuals)])
        else:
            merged_inputs_embeds_ctx = None

        ctx = _MergedAttnContext(
            merged_attn_metadata=merged_attn_metadata,
            merged_batch_descriptor=merged_batch_descriptor,
            num_tokens_padded_merged=num_tokens_padded_merged,
            merged_input_ids=merged_input_ids_ctx,
            merged_positions=merged_positions_ctx,
            merged_inputs_embeds=merged_inputs_embeds_ctx,
        )
        self._merged_attn_ctx_cache = ctx
        return ctx

    def _merge_layer_attn(
        self,
        layer_name: str,
        bundles: list["_ExecuteModelBundle"],
        batched_dp_ranks: list[int],
        per_dp_offsets: dict[int, int],
        bpb: int,
    ) -> Any:
        """Merge one layer's ``AscendMetadata`` across dp_ranks.

        Field-by-field strategy (see design doc §4.5):
        - cumsum: ``query_start_loc``
        - concat: ``seq_lens`` / ``seq_lens_cpu``
        - concat + per-dp_rank remap: ``block_tables`` /
          ``slot_mapping``
        - sum / max: ``num_actual_tokens`` / ``num_decode_tokens`` /
          ``num_prefills`` / ``num_decodes`` / ``max_query_len``

        Per-dp_rank-identical fields (``attn_state`` / ``causal`` /
        ``model_runner_type``) fall through from the first bundle.
        """
        from vllm_ascend.attention.attention_v1 import AscendMetadata

        def _layer_md(b: "_ExecuteModelBundle") -> Any:
            d = b.attn_metadata
            return d[0][layer_name] if isinstance(d, list) else (
                d[layer_name])

        any_md = _layer_md(bundles[0])
        per_layer_mds = [_layer_md(b) for b in bundles]

        # Unify ``attn_state`` across dp_ranks.
        states = [md.attn_state for md in per_layer_mds]
        unique_states = set(states)
        if len(unique_states) > 1:
            if AscendAttentionState.SpecDecoding in unique_states:
                raise NotImplementedError(
                    f"Layer {layer_name}: cannot merge SpecDecoding "
                    f"with other states {unique_states}; SpecDecoding "
                    f"uses a separate tree path that the batched "
                    f"forward does not yet support")
            merged_attn_state = AscendAttentionState.ChunkedPrefill
        else:
            merged_attn_state = next(iter(unique_states))

        if self._is_c8_attn_layer(layer_name):
            raise NotImplementedError(
                f"Layer {layer_name}: C8 (INT8 KV cache) backend "
                "does not yet support batched forward; fall back "
                "to per-dp_rank execution for this layer.")

        for b in bundles[1:]:
            md = _layer_md(b)
            for f in ("causal", "model_runner_type", "attn_mask"):
                lhs = getattr(md, f)
                rhs = getattr(any_md, f)
                if (isinstance(lhs, torch.Tensor)
                        or isinstance(rhs, torch.Tensor)):
                    if not torch.equal(lhs, rhs):
                        raise RuntimeError(
                            f"Layer {layer_name}: dp_rank "
                            f"b.attn_metadata[{f!r}] disagrees across "
                            f"dp_ranks; cannot merge for batched "
                            f"forward")
                elif lhs != rhs:
                    raise RuntimeError(
                        f"Layer {layer_name}: dp_rank "
                        f"b.attn_metadata[{f!r}] disagrees across "
                        f"dp_ranks; cannot merge for batched forward")
        for f in ("prefill", "decode_meta", "reshape_cache_event",
                  "kvcomp_metadata"):
            if any(getattr(_layer_md(b), f) is not None
                   for b in bundles):
                raise NotImplementedError(
                    f"Layer {layer_name}: {f!r} is set on at least "
                    f"one dp_rank; batched forward does not yet "
                    f"support merging this field.")

        # 1. query_start_loc (cumsum token offsets).
        merged_query_start_loc = [0]
        for b in bundles:
            md = _layer_md(b)
            n_reqs = md.seq_lens.shape[0]
            if n_reqs > 0:
                local = md.query_start_loc[1:n_reqs + 1]
                cumulative = local + merged_query_start_loc[-1]
                merged_query_start_loc.extend(cumulative.tolist())
            else:
                merged_query_start_loc.append(
                    merged_query_start_loc[-1])
        merged_query_start_loc = torch.tensor(
            merged_query_start_loc, dtype=torch.int32,
            device=self.device)

        # 2. seq_lens (concat).
        merged_seq_lens = torch.cat(
            [_layer_md(b).seq_lens for b in bundles])
        merged_seq_lens_cpu = (
            torch.cat([_layer_md(b).seq_lens_cpu for b in bundles])
            if any_md.seq_lens_cpu is not None else None)

        # 3. block_tables (concat + remap).
        dp_size = self.parallel_config.data_parallel_size
        merged_block_tables = torch.cat([
            torch.where(
                _layer_md(b).block_tables == 0,
                _layer_md(b).block_tables,
                ((_layer_md(b).block_tables // bpb) * dp_size
                 + per_dp_offsets[k]) * bpb
                + (_layer_md(b).block_tables % bpb),
            )
            for b, k in zip(bundles, batched_dp_ranks)
        ])

        # 4. slot_mapping (concat + remap).
        bs = self.block_size
        merged_slot_mapping = torch.cat([
            torch.where(
                _layer_md(b).slot_mapping == PAD_SLOT_ID,
                _layer_md(b).slot_mapping,
                ((_layer_md(b).slot_mapping // bs) * dp_size
                 + per_dp_offsets[k]) * bs
                + (_layer_md(b).slot_mapping % bs),
            )
            for b, k in zip(bundles, batched_dp_ranks)
        ])

        # 5. actual_seq_lengths_q (cu_seqlen of merged batch).
        merged_actual_seq_lengths_q = (
            merged_query_start_loc[1:].tolist())

        # 6. Scalars.
        merged_num_actual_tokens_pcp_padded = sum(
            _layer_md(b).num_actual_tokens_pcp_padded
            for b in bundles)
        merged_num_actual_tokens = sum(
            _layer_md(b).num_actual_tokens for b in bundles)
        merged_num_decode_tokens = sum(
            _layer_md(b).num_decode_tokens for b in bundles)
        merged_num_prefills = sum(
            _layer_md(b).num_prefills for b in bundles)
        merged_num_decodes = sum(
            _layer_md(b).num_decodes for b in bundles)
        merged_num_decodes_flatten = sum(
            _layer_md(b).num_decodes_flatten for b in bundles)
        merged_max_query_len = max(
            _layer_md(b).max_query_len or 0 for b in bundles)

        return AscendMetadata(
            num_actual_tokens_pcp_padded=(
                merged_num_actual_tokens_pcp_padded),
            num_actual_tokens=merged_num_actual_tokens,
            num_decode_tokens=merged_num_decode_tokens,
            num_prefills=merged_num_prefills,
            num_decodes=merged_num_decodes,
            num_decodes_flatten=merged_num_decodes_flatten,
            seq_lens=merged_seq_lens,
            seq_lens_cpu=merged_seq_lens_cpu,
            seq_lens_list=merged_seq_lens.cpu().tolist(),
            actual_seq_lengths_q=merged_actual_seq_lengths_q,
            query_start_loc=merged_query_start_loc,
            max_query_len=merged_max_query_len,
            block_tables=merged_block_tables,
            slot_mapping=merged_slot_mapping,
            attn_mask=any_md.attn_mask,
            attn_state=merged_attn_state,
            causal=any_md.causal,
            model_runner_type=any_md.model_runner_type,
        )

    def _merge_gdn_layer_attn(
        self,
        layer_name: str,
        bundles: list["_ExecuteModelBundle"],
        batched_dp_ranks: list[int],
        per_dp_offsets: dict[int, int],
        bpb: int,
    ) -> Any:
        """Merge one layer's ``GDNAttentionMetadata`` across dp_ranks.

        See ``non_embedding_only_support.md`` §3 for the 21-field
        strategy (scalar sums, per-request cat, cu_seqlen cumsum
        with offset, token-indices shift, mamba state-indices
        remap, derived-from-cu_seqlen fallback).
        """
        from vllm.v1.attention.backends.gdn_attn import (
            GDNAttentionMetadata,
        )

        def _layer_md(b: "_ExecuteModelBundle") -> Any:
            d = b.attn_metadata
            return d[0][layer_name] if isinstance(d, list) else (
                d[layer_name])

        any_md = _layer_md(bundles[0])
        per_layer_mds = [_layer_md(b) for b in bundles]

        # A. scalar sums.
        merged_num_prefills = sum(
            _layer_md(b).num_prefills for b in bundles)
        merged_num_prefill_tokens = sum(
            _layer_md(b).num_prefill_tokens for b in bundles)
        merged_num_decodes = sum(
            _layer_md(b).num_decodes for b in bundles)
        merged_num_decode_tokens = sum(
            _layer_md(b).num_decode_tokens for b in bundles)
        merged_num_spec_decodes = sum(
            _layer_md(b).num_spec_decodes for b in bundles)
        merged_num_spec_decode_tokens = sum(
            _layer_md(b).num_spec_decode_tokens for b in bundles)
        merged_num_actual_tokens = sum(
            _layer_md(b).num_actual_tokens for b in bundles)

        # B. per-request tensors (concat dim=0).
        def _cat_or_none(getter):
            values = [getter(_layer_md(b)) for b in bundles]
            if all(v is None for v in values):
                return None
            if any(v is None for v in values):
                raise NotImplementedError(
                    f"Layer {layer_name}: field is None on some "
                    f"dp_ranks and non-None on others; batched "
                    f"forward requires uniform None-ness.")
            return torch.cat(values)

        device = self.device
        parts: list[torch.Tensor] = []
        for md in per_layer_mds:
            n_non_spec = md.num_decodes + md.num_prefills
            if md.has_initial_state is not None:
                parts.append(md.has_initial_state)
            elif n_non_spec > 0:
                parts.append(
                    torch.ones(n_non_spec, dtype=torch.bool,
                               device=device))
        merged_has_initial_state = (
            torch.cat(parts) if parts else None)
        merged_spec_sequence_masks = _cat_or_none(
            lambda md: md.spec_sequence_masks)
        merged_num_accepted_tokens = _cat_or_none(
            lambda md: md.num_accepted_tokens)

        # E. mamba state indices (cat, with block_id remap).
        dp_size = self.parallel_config.data_parallel_size

        def _remap_state_indices(getter):
            values = [getter(_layer_md(b)) for b in bundles]
            if all(v is None for v in values):
                return None
            if any(v is None for v in values):
                raise NotImplementedError(
                    f"Layer {layer_name}: state indices are None on "
                    f"some dp_ranks and non-None on others; batched "
                    f"forward requires uniform None-ness.")
            return torch.cat([
                ((v // bpb) * dp_size + per_dp_offsets[k]) * bpb
                + (v % bpb)
                for v, k in zip(values, batched_dp_ranks)
            ])

        merged_spec_state_indices_tensor = _remap_state_indices(
            lambda md: md.spec_state_indices_tensor)
        merged_non_spec_state_indices_tensor = _remap_state_indices(
            lambda md: md.non_spec_state_indices_tensor)

        # D. token indices (cat + prior_token_count offset).
        prior_token_count = 0
        spec_token_indx_parts = []
        non_spec_token_indx_parts = []
        for b in bundles:
            md = _layer_md(b)
            if md.spec_token_indx is not None:
                spec_token_indx_parts.append(
                    md.spec_token_indx + prior_token_count)
            if md.non_spec_token_indx is not None:
                non_spec_token_indx_parts.append(
                    md.non_spec_token_indx + prior_token_count)
            prior_token_count += md.num_actual_tokens
        merged_spec_token_indx = (
            torch.cat(spec_token_indx_parts)
            if spec_token_indx_parts else None)
        merged_non_spec_token_indx = (
            torch.cat(non_spec_token_indx_parts)
            if non_spec_token_indx_parts else None)

        # C. cu_seqlen-style (cumsum concat with prior offset).
        prior_spec_count = 0
        spec_qsl_parts = []
        for b in bundles:
            md = _layer_md(b)
            if md.spec_query_start_loc is None:
                continue
            n = md.num_spec_decodes
            if n > 0:
                local = md.spec_query_start_loc[1:n + 1]
                spec_qsl_parts.append(local + prior_spec_count)
                prior_spec_count += int(
                    md.spec_query_start_loc[n].item())
        if spec_qsl_parts or any(
                _layer_md(b).spec_query_start_loc is None
                for b in bundles):
            merged_spec_query_start_loc = (
                torch.cat(
                    [torch.zeros(1, dtype=torch.int32,
                                 device=self.device)]
                    + spec_qsl_parts)
                if spec_qsl_parts else None)
        else:
            merged_spec_query_start_loc = None

        prior_non_spec_count = 0
        non_spec_qsl_parts = []
        for b in bundles:
            md = _layer_md(b)
            if md.non_spec_query_start_loc is None:
                continue
            n = md.num_decodes + md.num_prefills
            if n > 0:
                local = md.non_spec_query_start_loc[1:n + 1]
                non_spec_qsl_parts.append(
                    local + prior_non_spec_count)
                prior_non_spec_count += int(
                    md.non_spec_query_start_loc[n].item())
        merged_non_spec_query_start_loc = (
            torch.cat(
                [torch.zeros(1, dtype=torch.int32,
                             device=self.device)]
                + non_spec_qsl_parts)
            if non_spec_qsl_parts else None)

        # F-class: leave as None (the patched Ascend forward path
        # reads the ``*_fallback_meta`` set by the sub-patchers below,
        # not these dataclass fields).
        merged_chunk_indices = None
        merged_chunk_offsets = None
        merged_nums_dict = None
        merged_batch_ptr = None
        merged_token_chunk_offset_ptr = None

        # Run the three ``_patched_build_*`` sub-patchers (mirrors
        # ``_patched_build`` patch_gdn_attn.py:709-750).
        if (merged_num_prefills > 0 or merged_num_decodes > 0
                or merged_num_spec_decodes > 0):
            from vllm_ascend.patch.worker.patch_gdn_attn import (
                _patched_build_decode,
                _patched_build_prefill,
                _patched_build_spec,
            )

            builder = None
            for attn_group_list in self.attn_groups:
                for attn_group in attn_group_list:
                    if layer_name in attn_group.layer_names:
                        builder = attn_group.get_metadata_builder(0)
                        break
                if builder is not None:
                    break
            assert builder is not None, (
                f"GDN builder not found for layer {layer_name!r}")

            # Reconstruct merged full cu_seqlen (non-spec first,
            # then spec — matches ``argsort(spec_token_masks,
            # stable=True)`` ordering in ``_ORIGINAL_BUILD``
            # gdn_attn.py:271).
            n_non_spec = merged_num_decodes + merged_num_prefills
            non_spec_cumsum_cpu = (
                merged_non_spec_query_start_loc.cpu())
            if merged_spec_query_start_loc is not None:
                spec_cumsum_cpu = (
                    merged_spec_query_start_loc.cpu())
                merged_full_cumsum_cpu = torch.cat([
                    non_spec_cumsum_cpu,
                    spec_cumsum_cpu[1:] + non_spec_cumsum_cpu[-1],
                ])
            else:
                merged_full_cumsum_cpu = non_spec_cumsum_cpu
            # Per-req spec flag: ``-1`` non-spec, ``0`` spec.
            merged_nddt_cpu = torch.cat([
                torch.full((n_non_spec,), -1, dtype=torch.int32),
                torch.zeros(merged_num_spec_decodes,
                            dtype=torch.int32),
            ])

            common_stub = _StubCommonAttnMetadata(
                query_start_loc=torch.empty(0, device=self.device),
                query_start_loc_cpu=merged_full_cumsum_cpu,
                num_decode_draft_tokens_cpu=merged_nddt_cpu,
            )

        attn_metadata = GDNAttentionMetadata(
            num_prefills=merged_num_prefills,
            num_prefill_tokens=merged_num_prefill_tokens,
            num_decodes=merged_num_decodes,
            num_decode_tokens=merged_num_decode_tokens,
            num_spec_decodes=merged_num_spec_decodes,
            num_spec_decode_tokens=merged_num_spec_decode_tokens,
            num_actual_tokens=merged_num_actual_tokens,
            has_initial_state=merged_has_initial_state,
            spec_query_start_loc=merged_spec_query_start_loc,
            non_spec_query_start_loc=merged_non_spec_query_start_loc,
            spec_state_indices_tensor=merged_spec_state_indices_tensor,
            non_spec_state_indices_tensor=(
                merged_non_spec_state_indices_tensor),
            spec_sequence_masks=merged_spec_sequence_masks,
            spec_token_indx=merged_spec_token_indx,
            non_spec_token_indx=merged_non_spec_token_indx,
            num_accepted_tokens=merged_num_accepted_tokens,
            chunk_indices=merged_chunk_indices,
            chunk_offsets=merged_chunk_offsets,
            nums_dict=merged_nums_dict,
            batch_ptr=merged_batch_ptr,
            token_chunk_offset_ptr=merged_token_chunk_offset_ptr,
        )
        if merged_num_spec_decodes > 0:
            _patched_build_spec(builder, attn_metadata, common_stub)
        if merged_num_prefills > 0:
            _patched_build_prefill(builder, attn_metadata, common_stub)
        if merged_num_decodes > 0:
            _patched_build_decode(builder, attn_metadata, common_stub)
        attn_metadata.skip_graph_params_update = True
        return attn_metadata
