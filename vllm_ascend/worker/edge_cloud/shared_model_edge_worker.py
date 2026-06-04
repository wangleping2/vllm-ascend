# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-model edge worker for the edge-cloud collaboration feature.

This module defines :class:`SharedModelEdgeWorker`, a subclass of
:class:`NPUWorker` that lets multiple DP-rank edge workers live in a
single process and share a single ``nn.Module`` replica in NPU memory.

Key invariants:

- All virtual workers in the process share one ``self.rank`` (one
  distributed process group per process).
- ``self.local_rank`` uniquely identifies a virtual worker inside the
  process and is set to be equal to ``self.parallel_config.data_parallel_rank``.
- The total number of virtual workers is
  ``self.parallel_config.data_parallel_size``.
- Only the virtual worker with ``local_rank == 0`` (the *leader*) runs
  the heavy one-shot initialisation (device init, distributed init,
  workspace, model load, NPU graph capture). Followers (the rest) reuse
  the leader's process-level state and bind to the leader's model
  through :meth:`NPUModelRunner.bind_to_shared_model`.
- All virtual workers participate in PP communication at runtime.
  ``local_rank == k`` routes its PP messages to the cloud first-worker
  for DP-rank ``k``.

PP group layout (with ``data_parallel_size == N``):

    in-group rank 0:        edge_R (the single edge distributed rank)
    in-group rank 1..N:     cloud_dp_0_first, ..., cloud_dp_{N-1}_first

So:

- The edge (one rank) is at in-group rank 0.
- The cloud's first workers are at in-group ranks 1..N. The cloud
  first-worker for DP instance ``k`` is at in-group rank ``k + 1``.

Each virtual edge worker ``k`` communicates with the cloud peer at
in-group rank ``k + 1``. From the cloud's view, the edge is at
in-group rank 0.

This class only implements the *edge* side. The cloud side keeps the
existing edge-cloud layout (one process per cloud rank, TP within each
DP instance, only the first cloud rank per DP instance participating in
the shared PP group).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import logger
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    ModelRunnerOutput,
)



from vllm_ascend.distributed.parallel_state import (
    edge_cloud_broadcast_recv,
    edge_cloud_isend_tensor_dict,
    get_edge_cloud_tensor_meta,
    init_ascend_model_parallel,
    init_edge_cloud_tensor_meta,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.worker import NPUWorker, _detect_has_residual
from vllm_ascend.utils import enable_sp

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

# Process-wide list of SharedModelEdgeWorker instances, in local_rank
# order. We use a list rather than a rank-keyed dict because all
# virtual workers in the process share the same distributed ``rank``;
# a dict would silently overwrite earlier entries.
_SHARED_MODEL_REGISTRY: list["SharedModelEdgeWorker"] = []


def get_leader_worker() -> "SharedModelEdgeWorker | None":
    """Return the leader worker (local_rank == 0) in this process."""
    for w in _SHARED_MODEL_REGISTRY:
        if w._is_leader:
            return w
    return None

class DeferredExecutePostprocess(AsyncModelRunnerOutput):
    """Marker returned by
    :meth:`SharedModelEdgeWorker.execute_model` indicating that
    the tail recv + tail forward is deferred to the end of the
    current round.

    The class is both an :class:`AsyncModelRunnerOutput` *and*
    a :class:`collections.abc.Callable` — the
    :class:`vllm.v1.executor.shared_model_multiproc_executor.SharedModelWorkerProc`
    detects markers via the conjunction
    ``isinstance(output, AsyncModelRunnerOutput) and callable(output)``
    and accumulates them in ``self._pending_deferred``,
    invoking ``get_output()`` at the round boundary (which
    runs the postprocess and writes the final result to
    ``response_mqs[dp_rank]``).

    Two call surfaces are supported:

    * :meth:`get_output` — the path the WorkerProc takes via
      :meth:`vllm.v1.executor.shared_model_multiproc_executor.SharedModelWorkerProc.enqueue_output`.
      After running the postprocess, it does **one more type
      check** on the result: if the postprocess happens to
      return another :class:`AsyncModelRunnerOutput` (e.g. a
      future from a nested deferred step), it is unwrapped via
      ``result.get_output()`` recursively. The result the
      WorkerProc sees is therefore guaranteed to be a
      ``ModelRunnerOutput`` / ``None`` / ``IntermediateTensors``
      / ``Exception``, never a still-deferred object.

    * :meth:`__call__` — the direct-call path. The postprocess
      is run and the raw result is returned **without any
      type check**. This is what other code (e.g. unit tests
      or hand-rolled drivers) gets when it bypasses the
      WorkerProc's enqueue_output unwrap and simply calls the
      marker.
    """

    __slots__ = ("postprocess",)

    def __init__(self, postprocess) -> None:
        # ``postprocess`` is a zero-argument callable produced
        # by ``execute_model``. When invoked it runs the tail
        # recv + tail forward and returns the raw result with
        # signature ``ModelRunnerOutput | AsyncModelRunnerOutput
        # | IntermediateTensors | None``.
        self.postprocess = postprocess

    def __call__(self):
        # Direct call: run the postprocess and hand the raw
        # result back to the caller with no further
        # processing. This is the "shortcut" path: it is the
        # caller's responsibility to deal with the result
        # type.
        return self.postprocess()

    def get_output(self):
        # WorkerProc path. Run the postprocess, then do one
        # more type check on the result: if the postprocess
        # itself returned a nested
        # :class:`AsyncModelRunnerOutput` (e.g. a still-deferred
        # future), unwrap it via its own ``get_output()``.
        # ``enqueue_output`` downstream treats this return
        # value as a "ready" output and writes it straight to
        # the response MQ.
        result = self.postprocess()
        if isinstance(result, AsyncModelRunnerOutput):
            result = result.get_output()
        return result


class SharedModelEdgeWorker(NPUWorker):
    """Edge worker that shares one ``nn.Module`` across virtual DP workers.

    See module-level docstring for the design. The class follows the
    same RPC contract as :class:`NPUWorker` so that the existing
    :class:`MultiprocExecutor` can drive it through ``collective_rpc``
    without changes.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
            **kwargs,
        )
        # ``SharedModelEdgeWorker`` is only valid in the
        # shared-model edge-cloud topology: the worker must be on
        # the edge side and the edge must have exactly one NPU
        # (i.e. ``is_shared_model_edge`` is True). Using this
        # worker on the cloud side, or on a multi-NPU edge, would
        # silently produce incorrect PP routing.
        if not vllm_config.parallel_config.is_shared_model_edge:
            raise RuntimeError(
                "SharedModelEdgeWorker can only be used in the "
                "shared-model edge-cloud topology "
                "(edge_npu_count == 1 across the whole world). "
                "The current parallel_config has "
                f"is_shared_model_edge=False "
                f"(edge_npu_count="
                f"{vllm_config.parallel_config.edge_npu_count}, "
                f"cloud_npu_count="
                f"{vllm_config.parallel_config.cloud_npu_count}, "
                f"data_parallel_size="
                f"{vllm_config.parallel_config.data_parallel_size}); "
                "use a regular NPUWorker instead.")
        if not vllm_config.parallel_config.is_edge_node:
            raise RuntimeError(
                "SharedModelEdgeWorker is for the edge side of an "
                "edge-cloud configuration; the current process has "
                "is_edge_node=False. Use a regular NPUWorker on the "
                "cloud side.")
        # local_rank doubles as the worker's dp_rank in this design.
        self._is_leader: bool = (self.local_rank == 0)
        # Published by the leader in load_model; read by followers.
        self._shared_model: nn.Module | None = None
        # Published by the leader in determine_available_memory; read
        # by followers so they can return the same per-worker share
        # without redoing the memory profiling.
        self._per_worker_kv_cache_memory: int | None = None
        _SHARED_MODEL_REGISTRY.append(self)

    # --------------------------------------------------------- init_device
    def init_device(self) -> None:
        """Set up the NPU device and (for all virtual workers) the
        :class:`NPUModelRunner`.

        The leader runs the full one-shot initialisation (device set,
        memory snapshot, distributed init, workspace). Followers reuse
        the leader's ``self.device`` and skip those steps because they
        are process-wide; they only construct their own
        :class:`NPUModelRunner`. The shared model is bound later in
        :meth:`load_model`.
        """
        if self._is_leader:
            self.device = self._init_device()
            from vllm.v1.worker.workspace import init_workspace_manager
            init_workspace_manager(self.device, num_ubatches=1)
        else:
            # Reuse the leader's device: the process has one NPU
            # card and the leader has already called set_device.
            leader = get_leader_worker()
            assert leader is not None, (
                "SharedModelEdgeWorker follower constructed before "
                "the leader; ensure local_rank=0 is constructed first.")
            self.device = leader.device
            # ``_init_device`` is leader-only, so followers do not
            # have ``init_snapshot`` / ``requested_memory``;
            # inherit them from the leader for
            # ``determine_available_memory``.
            self.init_snapshot = leader.init_snapshot
            self.requested_memory = leader.requested_memory

        # All virtual workers construct their own model_runner; the
        # shared model is bound later in load_model. NPUModelRunner's
        # constructor does not depend on the model object.
        if self.use_v2_model_runner:
            from vllm_ascend.worker.v2.model_runner import (
                NPUModelRunner as NPUModelRunnerV2,
            )
            self.model_runner = NPUModelRunnerV2(self.vllm_config, self.device)
        else:
            self.model_runner = NPUModelRunner(self.vllm_config, self.device)
        
        if self._is_leader:
            # Initialize edge-cloud tensor metadata for optimized communication
            # (skips inter-node metadata sync in irecv_tensor_dict/isend_tensor_dict)
            if getattr(self.model_runner, '_edge_cloud_enabled', False):
                hidden_size = self.model_config.hf_text_config.hidden_size
                # Derive dtype directly from model config (same as MindIE's
                # self.config.torch_dtype from config.json), instead of
                # requiring a separate user-configured hidden_dtype.
                # model_config.dtype is a torch.dtype resolved from the
                # model's config.json torch_dtype field by _get_and_verify_dtype().
                hidden_dtype = self.model_config.dtype
                has_residual = _detect_has_residual(self.model_config)
                # DeepSeek V4 uses hc_mult > 1 (HC mechanism produces 3D
                # intermediate tensors).  Standard models (Qwen3.5, Llama,
                # etc.) do not have hc_mult, defaulting to 1 (2D tensors).
                hc_mult = getattr(self.model_config.hf_text_config, 'hc_mult', 1)
                init_edge_cloud_tensor_meta(
                    hidden_size=hidden_size,
                    hidden_dtype=hidden_dtype,
                    has_residual=has_residual,
                    hc_mult=hc_mult,
                    mode=self.model_runner.edge_cloud_cfg.mode,
                )

    # ----------------------------------------------- distributed env (leader)
    def _init_worker_distributed_environment(self) -> None:
        """Run the HCCL-backend distributed init once per process.

        Only the leader invokes the upstream machinery; followers inherit
        the process-level distributed state.
        """
        if not self._is_leader:
            return
        super()._init_worker_distributed_environment()

    # --------------------------------------------------------- model load
    def load_model(self) -> None:
        """Load the model (leader) or bind to the leader's model (followers).

        In the same process, the leader's ``__init__`` → ``init_device``
        → ``load_model`` runs strictly before any follower's, so the
        leader has already assigned :attr:`_shared_model` by the time
        followers need it. There is no polling — followers read
        ``_shared_model`` directly.
        """
        if self._is_leader:
            super().load_model()
            self._shared_model = self.model_runner.model
        else:
            leader = get_leader_worker()
            if leader is None or leader._shared_model is None:
                raise RuntimeError(
                    "Shared model not published by the leader worker. "
                    "Ensure SharedModelEdgeWorker instances are constructed "
                    "in local_rank order so the leader's load_model runs "
                    "before any follower's."
                )
            self.model_runner.bind_to_shared_model(leader._shared_model)
            self._shared_model = leader._shared_model
            # Inherit the leader's measured model memory usage so that
            # determine_available_memory can correctly subtract the
            # shared weight footprint.
            self.model_runner.model_memory_usage = (
                leader.model_runner.model_memory_usage)

    # ------------------------------------------------- execute_model / PP
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """Run one scheduler step and route PP communication to the
        cloud first-worker for this virtual worker's dp_rank.

        Mirrors :meth:`NPUWorker.execute_model` exactly. The only
        divergence is the two ``edge_cloud_broadcast_recv`` calls in
        the original: they are called with an explicit ``src=``
        argument (the shared-model edge worker cannot rely on the
        implicit "previous PP rank" routing — each virtual worker
        has its own cloud peer based on ``local_rank``).

        The head forward + PP send is performed synchronously;
        the tail recv + tail forward is wrapped into a zero-
        argument callable (a closure over ``self`` and
        ``scheduler_output``) and returned in lieu of the final
        result. The
        :class:`vllm.v1.executor.shared_model_multiproc_executor.SharedModelWorkerProc`
        accumulates these callables across dp_ranks and invokes
        them in batch when the round barrier is reached (just
        before the result is enqueued onto the response MQ), so
        per-dp_rank tail processing stays in lockstep. The
        ``method == "execute_model"`` filter on the dispatch
        side keeps the callable-detection unambiguous: no other
        return value of any ``SharedModelEdgeWorker`` method is
        a plain function.
        """
        from types import NoneType
        from vllm.sequence import IntermediateTensors
        from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors
        from vllm_ascend import envs as envs_ascend

        # enable msMonitor to monitor the performance of vllm-ascend
        if envs_ascend.MSMONITOR_USE_DAEMON:
            from vllm_ascend.profiler.torch_npu_profiler import (
                dynamic_profile as dp,
            )
            dp.step()

        if self._pp_send_work:
            for handle in self._pp_send_work:
                handle.wait()
            self._pp_send_work = []

        # SharedModelEdgeWorker always sits at PP rank 0 (the edge is
        # the first stage of the shared PP group), so there is no
        # upstream PP receive before the first forward.

        if self.profiler is not None:
            self.profiler.step()

        output = self.model_runner.execute_model(scheduler_output, None)
        if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput,
                               NoneType)):
            return output

        assert isinstance(output, IntermediateTensors)

        # Edge-cloud with heterogeneous SP: aggregate SP shards to full
        # sequence before cross-PP send so cloud can re-chunk by its SP.
        if enable_sp() and (self.model_runner.edge_cloud_cfg.mode != "embedding_only"
            or not self.model_runner.supports_mm_inputs):
            _gathered = self._all_gather_tensor_dict(output.tensors)
        else:
            _gathered = output.tensors
        # Send the head-layer output to the cloud first-worker of
        # ``local_rank``'s dp_rank (in-group rank
        # ``self.local_rank + 1``). The explicit ``dst=`` is required
        # because the edge sits at in-group rank 0 — without it every
        # virtual worker would send to in-group rank 1, which is only
        # correct for the first virtual worker.
        self._pp_send_work = edge_cloud_isend_tensor_dict(
            _gathered,
            dst=self.local_rank + 1,
            num_tokens=scheduler_output.total_num_scheduled_tokens,
        )

        edge_sp = enable_sp()
        edge_merge = get_edge_cloud_tensor_meta().merge_payload
        # Defer the tail recv + tail forward to the end of the
        # current round. The WorkerProc accumulates these
        # callables in ``_pending_deferred`` and invokes them
        # in batch (one ``postprocess()`` per dp_rank) when the
        # round barrier is reached, just before the result is
        # enqueued onto ``response_mqs[dp_rank]``. This keeps
        # the per-dp_rank streams in lockstep — the cloud's
        # middle forward can run concurrently with the edge's
        # head forwards for subsequent dp_ranks, and the edge's
        # tail recvs happen in lockstep with the round
        # boundary.
        def _tail_postprocess():
            # Receive the cloud's middle-layer result and run
            # the second forward (tail layers). The cloud peer
            # is at in-group rank ``self.local_rank + 1``.
            tensor_dict, comm_handles, comm_postprocess = (
                edge_cloud_broadcast_recv(
                    num_tokens=scheduler_output.total_num_scheduled_tokens,
                    sp_chunk=edge_sp and edge_merge,
                    src=self.local_rank + 1))
            intermediate_tensors = AsyncIntermediateTensors(
                tensor_dict,
                comm_handles=comm_handles,
                comm_postprocess=comm_postprocess,
            )
            tail_output = self.model_runner.execute_model(
                scheduler_output, intermediate_tensors)
            if isinstance(tail_output,
                          (ModelRunnerOutput, AsyncModelRunnerOutput,
                           NoneType)):
                return tail_output
            # Edge path in the original NPUWorker.execute_model
            # always returns after the second forward — the
            # trailing KV-connector passthrough is for non-edge/
            # non-cloud middle PP stages, which never run for
            # SharedModelEdgeWorker.
            assert isinstance(tail_output, IntermediateTensors)
            return tail_output

        return DeferredExecutePostprocess(postprocess=_tail_postprocess)

    # ------------------------------------------- memory / compile / warmup
    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Per-virtual-worker share of the available KV-cache memory.

        Whichever virtual worker is called first performs the
        actual memory profiling (via ``profile_run``); all
        subsequent callers (regardless of ``local_rank``) return
        the per-worker share already cached on the registry.
        This avoids requiring any specific ``local_rank`` to be
        called first while still ensuring the expensive profile
        run executes exactly once.
        """
        # Fast path: already computed by some virtual worker in
        # this process. Each virtual worker still needs to run
        # ``profile_run`` on its own model_runner to initialise
        # its compiled artifacts.
        for w in _SHARED_MODEL_REGISTRY:
            if w._per_worker_kv_cache_memory is not None:
                self.model_runner.profile_run()
                return w._per_worker_kv_cache_memory

        # Slow path: we are the first caller. Do the actual
        # profiling and divide by dp_size.
        if self.cache_config.kv_cache_memory_bytes:
            self._per_worker_kv_cache_memory = int(
                self.cache_config.kv_cache_memory_bytes
                // self.parallel_config.data_parallel_size)
            return self._per_worker_kv_cache_memory

        from vllm.utils.mem_utils import memory_profiling

        weights_memory = int(self.model_runner.model_memory_usage)
        with memory_profiling(self.init_snapshot,
                              weights_memory=weights_memory) as result:
            self.model_runner.profile_run()
            profile_torch_peak = torch.npu.memory_stats(
                self.device).get("allocated_bytes.all.peak", 0)

        result.torch_peak_increase = (
            profile_torch_peak - result.before_profile.torch_peak)
        result.non_kv_cache_memory = (
            result.non_torch_increase + result.torch_peak_increase
            + result.weights_memory)

        free_gpu_memory = result.after_profile.free_memory
        if self.init_snapshot.free_memory <= free_gpu_memory:
            raise RuntimeError(
                "Error in memory profiling: free memory increased.")
        available = int(self.requested_memory - result.non_kv_cache_memory)
        self._per_worker_kv_cache_memory = (
            available // self.parallel_config.data_parallel_size)
        # For embedding_only edge, the edge device does not actually store KV
        # cache tensors. Return a very large virtual value so that
        # get_kv_cache_configs() does not clamp num_blocks to the edge's
        # (small) available memory. The real num_blocks is determined by cloud.
        if (
            self.model_runner.edge_cloud_cfg.enabled
            and self.model_runner.edge_cloud_cfg.mode == "embedding_only"
            and self.model_runner.edge_cloud_cfg.role == "edge"
        ):
            self._per_worker_kv_cache_memory = 1 << 40  # 1 TiB virtual
        self.available_kv_cache_memory_bytes = self._per_worker_kv_cache_memory
        logger.info(
            "SharedModelEdgeWorker[local_rank=%d] per-worker KV cache "
            "memory: %.2f GiB", self.local_rank,
            self._per_worker_kv_cache_memory / (1024 ** 3))
        return self._per_worker_kv_cache_memory

    def execute_dummy_batch(self) -> None:
        """No-op for all virtual workers.

        NPU graphs are captured in
        :meth:`compile_or_warm_up_model`; in a single-process design
        there is no need to re-run a decode-only dummy per DP rank.
        """
        return None

    # --------------------------------------------------------- sleep/wake
    # TODO(shared-model): these overrides assume a single dp rank on
    # the edge. With multiple dp ranks sharing one process, the
    # leader's sleep/wake_up only acts on the leader's view of the
    # shared ``nn.Module``; followers' sleep/wake_up calls are
    # silently dropped. Verify that this is consistent with the
    # downstream / upstream CaMemAllocator state for all dp ranks,
    # and add a barrier (e.g. a flag in ``_SHARED_MODEL_REGISTRY``)
    # if a coordinated sleep/wake_up across virtual workers is
    # required by the upper layer.
    def sleep(self, level: int = 1) -> None:
        """Sleep: leader runs the standard offload; followers no-op.

        The model is shared across virtual workers, so weights are
        offloaded exactly once by the leader. Followers inherit the
        offloaded state through the shared ``nn.Module``.
        """
        if not self._is_leader:
            return
        super().sleep(level=level)

    def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake: leader runs the standard restore; followers no-op.

        See :meth:`sleep` for the rationale on the leader-only
        behaviour.
        """
        if not self._is_leader:
            return
        super().wake_up(tags=tags)

    # --------------------------------------------- static kernel cleanup
    def uninstall_static_kernel(self) -> None:
        """Uninstall the static ATB kernel at shutdown (leader only)."""
        if not self._is_leader:
            return
        super().uninstall_static_kernel()
