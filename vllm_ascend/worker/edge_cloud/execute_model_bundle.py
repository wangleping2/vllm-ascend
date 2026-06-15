# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-dp_rank state bundle for the batched compute path.

The shared-model edge worker batches head / tail model_forward across
dp_ranks: a single ``_model_forward`` + ``compute_logits`` call replaces
``dp_size`` per-dp_rank calls. The state that has to flow through the
batched forward is captured in :class:`_ExecuteModelBundle`, produced by
``NPUModelRunner.execute_model_pre`` (per-dp_rank) and consumed by
``NPUModelRunner.execute_model_batched_head`` / ``_tail`` /
``_post_batched``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


@dataclass
class _ExecuteModelBundle:
    """Per-dp_rank state produced by ``execute_model_pre`` and consumed by
    the batched head / tail / per-dp_rank post.

    Field mapping against ``ExecuteModelState`` (set by
    ``execute_model_post_batched``) is documented in the design doc; see
    ``worker-dp-rank-worker-dp-rank-worker-mo-dapper-truffle.md``.
    """

    # Tensors to be concat'd for the batched forward. The first
    # two are ``| None`` because ``NPUModelRunner._preprocess`` may
    # legitimately return ``(None, None, ...)`` (e.g. encoder-only
    # / mm-input paths that feed the model via ``inputs_embeds``
    # only). The batched head / tail cat guards against None
    # explicitly and falls back to a merged ``None`` rather than
    # crashing on ``torch.cat``. ``inputs_embeds`` is mutually
    # exclusive with ``input_ids`` (mm path → ``inputs_embeds``
    # only, text path → ``input_ids`` only); the model runner
    # picks one of them inside ``forward_edge_cloud_segment``.
    input_ids: torch.Tensor | None
    positions: torch.Tensor | None
    inputs_embeds: torch.Tensor | None
    intermediate_tensors: IntermediateTensors | None
    hidden_states: torch.Tensor | None

    # Per-dp_rank state for the post-batched / sample path
    logits_indices: torch.Tensor
    spec_decode_metadata: "SpecDecodeMetadata | None"
    spec_decode_common_attn_metadata: Any | None
    scheduler_output: "SchedulerOutput"

    # Forward-context fields
    num_tokens_padded: int
    num_tokens_across_dp: torch.Tensor | None
    cudagraph_mode: CUDAGraphMode
    batch_desc: BatchDescriptor | None
    attn_metadata: Any
    ec_connector_output: Any | None
    cudagraph_stats: Any | None

    # Pre-stage side effects (from ``_update_states``)
    deferred_state_corrections_fn: Callable[[], None] | None
