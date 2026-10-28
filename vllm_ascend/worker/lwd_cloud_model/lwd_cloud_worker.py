# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD (edge-cloud) worker subclass.

All LWD worker-side logic lives here (moved out of ``worker.py``):
init bring-up of the duplex channels + recv managers, the EDGE_EMBED
dispatch hook, the cloud finished-flush, and the
DOWN send (draining the runner's per-step hidden payload + attaching
``lwd_c2e_meta`` to the ModelRunnerOutput).

Selected via ``parallel_config.worker_cls`` (see platform.py:
``vllm_ascend.worker.lwd_cloud_worker.LwdCloudWorker``) when
``lwd_config`` is set (LWD deployment).
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from vllm.lwd_logger import lwd_logger

from vllm_ascend import envs
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

# ops must initialize before the model_runner chain, otherwise device_op
# and the ops package circularly import (mirrors worker.py's import order)
import vllm_ascend.ops  # noqa: F401
from vllm_ascend.distributed.lwd_data_plane.lwd_data_plane import lwd_get_data_plane
from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_future import LwdTransferFuture
from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType, LwdTransferRequest
from vllm_ascend.worker.lwd_base_model.lwd_worker_base import LwdWorkerBase
from vllm_ascend.worker.lwd_cloud_model.lwd_cloud_model_runner import (
    LwdCloudModelRunner,
)
from vllm_ascend.worker.lwd_hash.lwd_hash_routing import hash_layer_count, hash_payload_numel
from vllm_ascend.worker.lwd_cloud_model.lwd_host_stage import (
    LwdHostStager,
    LwdHostStageError,
    LwdHostStageGeometryError,
    LwdHostStageSlotBusy,
)
from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
    LwdRecvRowTable,
    hint_fields,
    hint_row_table,
)

if TYPE_CHECKING:
    from typing import Any

    from vllm.distributed.device_communicators.shm_broadcast import Handle
    from vllm.v1.core.sched.output import (
        GrammarOutput,
        SchedulerOutput,
    )

# Early-recv thread loop tuning: MQ dequeue timeout, backoff sleep after a
# dequeue error, and the per-iteration sleep that caps CPU while the MQ
# reader window spins.
_LWD_EARLY_RECV_DEQUEUE_TIMEOUT_S = 0.0005
_LWD_EARLY_RECV_ERROR_SLEEP_S = 0.01
_LWD_EARLY_RECV_LOOP_SLEEP_S = 0.002

# Posted-but-not-published recvs before hints queue (see
# VLLM_ASCEND_LWD_MAX_INFLIGHT_RECV in envs.py). Must stay >= the edge's
# pending-chunks gate so the cloud can always post whatever the edge may
# still have in flight; larger values only hold more device-slot memory.
_LWD_EARLY_RECV_MAX_INFLIGHT = envs.VLLM_ASCEND_LWD_MAX_INFLIGHT_RECV


class LwdCloudWorker(LwdWorkerBase):
    """LwdWorkerBase + LWD data-plane wiring (cloud side only).

    Selected by ``platform.py`` only when LWD is enabled on a cloud
    process; the data-plane wiring below runs unconditionally."""

    def init_device(self) -> None:
        super().init_device()
        # channel-global DOWN seqno counter (worker layer, send-time alloc)
        self._lwd_down_next_seqno = 0
        # [Lwd][pre-recv] Early receive: only the endpoint rank (= the only
        # cloud rank holding a cross-node P2P channel, same convention as
        # the data-plane channels) pre-posts the real irecv and reports completion
        # watermarks; every other rank gets its data via the in-group TP
        # broadcast at inject time — no recv posted, nothing reported.
        self._init_lwd_early_recv_state()
        if self.use_v2_model_runner:
            # The LWD hooks live on the V1 model runner; the V2 runner
            # (separate class) lacks them entirely — fail fast at bring-up
            # instead of AttributeError at first request.
            raise RuntimeError("LWD data plane requires the V1 model runner; use_v2_model_runner is not supported")
        # Swap in the LWD model runner BEFORE any model load/usage.
        # The cloud feeds on edge-computed prompt embeddings (no token
        # ids on its input), so the native prompt-embeds path must be
        # enabled for the runner to allocate inputs_embeds buffers.
        self.model_config.enable_prompt_embeds = True
        self.model_runner = LwdCloudModelRunner(self.vllm_config, self.device, worker=self)
        self._init_lwd_data_plane()
        # Only the endpoint rank (the cloud leader — first rank of the
        # first cloud DP in the topology) has a cross-node channel: start
        # its early-recv comm thread. The topology ranks are GLOBAL, while
        # self.rank is the executor's per-node numbering — the edge card
        # count offsets them (seen live: local rank 1 of the cloud was
        # picked as "endpoint", its comm thread then died on "rank 2 holds
        # no lwd_up channel"). Compare the distributed global rank.
        import torch.distributed as dist

        global_rank = dist.get_rank()
        self._lwd_early_recv_endpoint = global_rank == self.vllm_config.lwd_config.topology.clouds[0].dp[0].ranks[0]
        self._lwd_up_hidden_size = self.model_config.get_hidden_size()
        # One log line per cloud rank: which ranks own the comm thread is obvious at a glance.
        lwd_logger.info(
            "[Lwd][pre-recv] worker local_rank=%d global_rank=%d endpoint=%s hidden=%d",
            self.rank,
            global_rank,
            self._lwd_early_recv_endpoint,
            self._lwd_up_hidden_size,
        )
        if self._lwd_early_recv_endpoint:
            # Host stager BEFORE any recv can be posted (the comm thread
            # starts right after): the arrival-ledger driver requires it.
            self._lwd_init_host_stager(self._lwd_up_hidden_size)
            self._start_lwd_early_recv_thread()
        self._register_lwd_prompt_embeds_provider()

    def _lwd_init_host_stager(self, hidden_size: int) -> None:
        """Build the host staging pool (endpoint rank only).

        Arrival-driven scheduling DEPENDS on the per-request arrival rows
        this pool publishes: without the stager the ledger stays at 0,
        the visible-set predicate rejects every prefill request and the
        whole system deadlocks silently - scheduling and reporting are
        two halves of the same feature and must both be present. The
        engine fail-fasts at startup when the switch is off; this gate
        exists for the same-script A/B comparison.

        Slot sizing uses the hash payload formula (embeds + optional
        expert-ID tail): the UP recv numel is hash_payload_numel
        (_lwd_build_up_recv_request); sizing by hidden alone makes the
        allocate geometry check reject every DSV4 batch.
        """
        if not envs.VLLM_ASCEND_LWD_HOST_STAGE:
            lwd_logger.warning(
                "[Lwd][host-stage] disabled by env on the endpoint rank; "
                "the arrival ledger will stay empty (the engine's startup "
                "check rejects this combination)"
            )
            return

        def _pinned(n: int) -> torch.Tensor:
            return torch.empty(n, dtype=torch.bfloat16, device="cpu").pin_memory()

        _cfg = self.model_config.hf_config
        _max_tok = (
            getattr(self.vllm_config.scheduler_config, "max_num_batched_tokens", 0) or 0
        )
        _max_numel = hash_payload_numel(
            _max_tok,
            hidden_size,
            hash_layer_count(_cfg),
            int(getattr(_cfg, "num_experts_per_tok", 0) or 0),
        )
        slots = (
            envs.VLLM_ASCEND_LWD_HOST_STAGE_SLOTS
            or self._early_recv_max_inflight
        )
        self._lwd_stager = LwdHostStager(
            device="cpu",
            slots=slots,
            empty_factory=_pinned,
            max_numel=_max_numel or None,
        )
        lwd_logger.info(
            "[Lwd][host-stage] stager ready slots=%d max_numel=%d "
            "(max_tokens=%d hidden=%d, no host cap)",
            slots,
            _max_numel,
            _max_tok,
            hidden_size,
        )

    def load_model(self) -> None:
        """Log the real layer split after loading: layer count / first-last
        layer names / pp slicing args, so a half-model suspicion is decided
        at startup (full coverage from layer 0 = normal; halved or nonzero
        start = wrong pp slicing)."""
        super().load_model()
        # [layer-trace] Parity instrumentation (env-gated, zero overhead
        # when off): the cloud install point; edge and centralized share
        # the same code (see lwd_debug.lwd_layer_trace).
        from vllm_ascend.worker.lwd_debug.lwd_layer_trace import (
            install_lwd_layer_trace,
        )

        install_lwd_layer_trace(self.model_runner.get_model())
        model = self.model_runner.get_model()
        backbone = getattr(model, "model", model)
        layers = getattr(backbone, "layers", None) or getattr(backbone, "decoder_layers", None)
        pc = self.vllm_config.parallel_config
        layer_names = [
            n for n, _ in model.named_modules() if n.count("layers.") == 1 and n.endswith(tuple("0123456789"))
        ]
        lwd_logger.info(
            "[Lwd][cloud-model] loaded layers=%s first=%s last=%s "
            "(config pp=%d tp=%d, my rank=%d) — full load should cover every layer from 0",
            len(layers) if layers is not None else "?",
            layer_names[0] if layer_names else "?",
            layer_names[-1] if layer_names else "?",
            pc.pipeline_parallel_size,
            pc.tensor_parallel_size,
            self.rank,
        )

    # ------------------------------------------------------------------ #
    # Engine step wiring                                                  #
    # ------------------------------------------------------------------ #

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        # [Lwd][perf] Worker start time (subtract the engine's rpc-enqueue
        # ts = dispatch-to-start latency; spread across ranks = collective
        # alignment cost)
        lwd_logger.info(
            "[Lwd][perf] worker-exec-start rank=%d ts=%.3f",
            self.rank,
            time.monotonic(),
        )
        # No SO-driven UP recv posting under host staging: the comm thread
        # pre-posts every recv from the engine's hints, and the runner
        # harvests by input_batch (arrival-driven, no SO consumption
        # point).

        # cloud: finished requests trigger local cleanup (collector
        # bookkeeping + UP chunk table); finished_req_ids is the
        # engine's own liveness signal.
        if scheduler_output.finished_req_ids:
            self._lwd_cloud_flush_finished(scheduler_output.finished_req_ids)

        lwd_get_data_plane().poll_completions()  # lazy keepalive reap
        _t0 = time.monotonic()
        output = super().execute_model(scheduler_output)
        # [Lwd][perf] Per-step exec_model timing on the cloud: total = the
        # whole forward pass (sample/collect are itemized separately in
        # sample_tokens)
        lwd_logger.info(
            "[Lwd][perf] cloud-exec total=%.2fms",
            (time.monotonic() - _t0) * 1000,
        )
        return output

    # ------------------------------------------------------------------ #
    # Early receive: comm thread (hint -> post recv -> report watermark) #
    # ------------------------------------------------------------------ #

    def _init_lwd_early_recv_state(self) -> None:
        """The early-recv ledger (same keys as the engine-issued hints and
        the reported watermarks).

        Key = (channel, seqno). Entry lifecycle under host staging:
        a hint submits the recv and registers the entry; once the recv
        completes AND the segmented D2H lands in the per-request pinned
        buffers (``publish_received``), the entry is popped
        (**publish-then-pop**: after the publish the future bookkeeping
        has no readers - the runner harvests by input_batch, there is no
        SO consumption point) and the next queued hint is posted.

        ``_early_recv_last_delivered`` is the highest seqno already
        delivered (received + staged + reported). Channel completion
        order = seqno order, so the delivered set is always a prefix and
        one watermark represents it exactly. ``start_early_irecv``
        rejects any hint at or below it: the edge sends one frame per
        seqno, and posting a second recv for an already-delivered seqno
        misaligns the whole channel by one frame WITHOUT any error.

        **In-flight cap** (env ``VLLM_ASCEND_LWD_MAX_INFLIGHT_RECV``,
        default 8): each posted recv holds a device ping-pong slot until
        its segmented D2H lands, and the cap must stay >= the edge-side
        pending gate (``LwdEdgeScheduler._LWD_MAX_PENDING_CHUNKS``) so the
        cloud can always post whatever the edge may still have in
        flight. Hints beyond the cap queue first
        (``_early_recv_pending``, arrival-ordered, fields only — no
        device memory) and post as soon as an earlier batch publishes.
        """
        self._early_recv_max_inflight = _LWD_EARLY_RECV_MAX_INFLIGHT
        self._early_recv_futures: dict[tuple[LwdChannelType, int], LwdTransferFuture] = {}
        self._early_recv_lock = threading.Lock()
        # Highest seqno already delivered (published + reported); the
        # prefix-watermark guard against duplicate/late hints
        self._early_recv_last_delivered: int = -1
        # Hints queued while the in-flight cap is full (key -> hint, insertion order = arrival order)
        self._early_recv_pending: dict[tuple[LwdChannelType, int], dict[str, Any]] = {}
        # req_id -> last REPORTED cumulative arrived rows (skips
        # no-progress requests in the done-MQ piggyback; entries whose
        # request left the stager are pruned on the same pass)
        self._lwd_last_total: dict[str, int] = {}
        # Rate-limit timestamp for the stuck-queue diagnostic
        self._early_recv_stuck_log_at = 0.0
        # Downstream hint MQ (reader) / upstream completion MQ (writer); the
        # engine attaches the done reader through the handle returned by
        # lwd_early_recv_done_handle (fetched over collective_rpc).
        self._irecv_hint_mq = None
        self._irecv_done_mq = None
        self._lwd_early_recv_thread: threading.Thread | None = None

    def _lwd_build_up_recv_request(
        self,
        seqno: int,
        num_tokens: int,
        aux_rows: int,
        row_table: LwdRecvRowTable | None,
    ) -> LwdTransferRequest:
        """The single construction point for UP recv requests.

        Host staging on: at SUBMIT time this also builds the stager plan
        (per-request pinned buffers sized once at prompt length + the
        segmented D2H layout) and borrows a device ping-pong slot as the
        receive buffer - the channel layer then lands the payload
        segment-by-segment into the pinned buffers on its copy stream as
        soon as the recv completes, and the comm thread only polls
        ``staged_done()`` and publishes (it never blocks on a copy).

        HCCL P2P requires both ends to match numel exactly: the main frame
        = the hash payload formula (embeds + optional expert-ID bytes).
        The aux frame mechanism is kept for deepstack and other future
        payloads (mrope positions moved to the control plane in
        wire_version 2; callers always pass aux_rows=0). The size formula
        is computed here exactly once so the two paths cannot drift.

        Raises:
            LwdHostStageGeometryError: the row table is missing or
                inconsistent with the payload size (protocol-level, the
                caller drops the hint).
            LwdHostStageSlotBusy: every device slot is in use (the caller
                re-queues the hint; it is never dropped).
        """
        num_elements = hash_payload_numel(
            num_tokens,
            self._lwd_up_hidden_size,
            hash_layer_count(self.model_config.hf_config),
            getattr(self.model_config.hf_config, "num_experts_per_tok", 0),
        )
        request = LwdTransferRequest(
            channel=LwdChannelType.UP,
            op="recv",
            num_elements=num_elements,
            seqno=int(seqno),
            # The aux second payload shares the main frame's seqno; always
            # 0 since mrope retired (zero aux traffic); future payloads
            # (deepstack etc.) just pass their row count
            aux_num_elements=int(aux_rows) * 3,
        )
        stager = self._lwd_stager
        if stager is None:
            return request
        if row_table is None:
            raise LwdHostStageGeometryError(
                f"seqno={seqno}: staged recv needs the per-request row "
                f"table (req_ids/rows/prompt_offsets/prompt_lens); the "
                f"hint carried none (both sides must deploy the "
                f"row-table hint protocol)"
            )
        req_ids, rows, prompt_offsets, prompt_lens = row_table
        cfg = self.model_config.hf_config
        segments = stager.plan_recv_segments(
            seqno=int(seqno),
            numel=num_elements,
            hidden_size=self._lwd_up_hidden_size,
            req_ids=req_ids,
            num_tokens_per_req=rows,
            prompt_lens=prompt_lens,
            row_base=list(prompt_offsets) if prompt_offsets else None,
            num_hash_layers=hash_layer_count(cfg),
            num_experts_per_tok=int(getattr(cfg, "num_experts_per_tok", 0) or 0),
        )
        slot_view, slot = stager.allocate(int(seqno), num_elements)
        return replace(
            request, recv_buffer=slot_view, stage_segments=segments, recv_slot=slot
        )

    def _start_lwd_early_recv_thread(self) -> None:
        """Rebuild the hint MQ reader, create the done MQ writer, and start
        the early-recv comm thread.

        One log line before and after the thread start: "starting" before
        (proof this code ran) and "up" from inside the thread body after
        setting the device (proof the thread is alive). Only the first
        line = the thread died during device setup (traceback on stderr);
        both lines but no SUBMIT = no hints arrived (hint MQ not attached;
        the log will say attached/none).
        """
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
            attach_hint_mq,
            create_report_mq,
        )

        self._irecv_hint_mq = attach_hint_mq()
        self._irecv_done_mq = create_report_mq()
        # The engine fetches this MQ's handle through
        # lwd_early_recv_done_handle over collective_rpc once the executor
        # is up, so the vllm executor stays free of LWD plumbing.
        lwd_logger.info(
            "[Lwd][pre-recv] comm thread starting rank=%d hint MQ=%s",
            self.rank,
            "attached" if self._irecv_hint_mq is not None else "none",
        )
        self._lwd_early_recv_thread = threading.Thread(
            target=self._lwd_early_recv_loop,
            name="lwd-early-recv",
            daemon=True,
        )
        self._lwd_early_recv_thread.start()

    def lwd_early_recv_done_handle(self) -> Handle | None:
        """Handle of the early-recv done MQ, for the engine to attach its
        reader; None on ranks that do not report completions (the
        non-endpoint ranks hold no cross-node channel)."""
        if self._irecv_done_mq is None:
            return None
        return self._irecv_done_mq.export_handle()

    def _lwd_early_recv_loop(self) -> None:
        """Comm thread body: take hints -> pre-post recvs; poll futures ->
        report completions.

        It only does four featherweight things: ``dequeue`` /
        ``submit_recv`` / ``future.done()`` / ``enqueue`` — it never
        ``wait()``s any future and never touches the model, so it cannot
        conflict with the busy loop's HCCL usage (an irecv on one side of
        a channel plus a cross-thread wait on the other is disallowed by
        HCCL).
        """
        # The current device is thread-local state; the comm thread must
        # set it once itself (or the irecv lands on card 0)
        torch.npu.set_device(self.device)
        lwd_logger.info(
            "[Lwd][pre-recv] comm thread up rank=%d device=%s hint MQ=%s",
            self.rank,
            self.device,
            "attached" if self._irecv_hint_mq is not None else "none",
        )
        hint_mq = self._irecv_hint_mq
        while True:
            if hint_mq is not None:
                try:
                    hint = hint_mq.dequeue(timeout=_LWD_EARLY_RECV_DEQUEUE_TIMEOUT_S)
                except TimeoutError:
                    pass
                except Exception:
                    # Ring cancelled at shutdown etc.: log once and sleep a
                    # little; never hot-spin (daemon threads exit with the
                    # process)
                    lwd_logger.exception("[Lwd][pre-recv] hint MQ dequeue failed")
                    time.sleep(_LWD_EARLY_RECV_ERROR_SLEEP_S)
                else:
                    self.start_early_irecv(hint)
            self._report_irecv_completions()
            self._warn_if_early_recv_stuck()
            # The ring's read side sched_yield-spins within the
            # just-read window (busy_loop_s=1s), so a 0.5ms timeout alone
            # does not cap the CPU — this sleep does; the cost is ~2ms of
            # added visibility latency on hints and watermarks each.
            time.sleep(_LWD_EARLY_RECV_LOOP_SLEEP_S)

    _LWD_EARLY_RECV_STUCK_LOG_INTERVAL_S = 5.0

    def _warn_if_early_recv_stuck(self) -> None:
        """Warn every 5s while the queue cannot drain: hints are queued but
        every in-flight slot is held.

        Held slots = posted recvs neither consumed nor released (so the
        watermark stops advancing and the gate stops releasing prefills).
        The log names exactly which seqnos hold the slots — no more
        "prefill stalled and nobody knows why".
        """
        with self._early_recv_lock:
            if not self._early_recv_pending or len(self._early_recv_futures) < self._early_recv_max_inflight:
                return
            now = time.monotonic()
            if now - self._early_recv_stuck_log_at < self._LWD_EARLY_RECV_STUCK_LOG_INTERVAL_S:
                return
            self._early_recv_stuck_log_at = now
            oldest = next(iter(self._early_recv_pending))[1]
            holders = sorted(key[1] for key in self._early_recv_futures)
        lwd_logger.warning(
            "[Lwd][pre-recv] queue stuck: oldest queued seqno=%d, %d slot(s) "
            "held by seqno=%s (slots held = those batches posted recv but "
            "were never consumed; downstream dispatch is the pacer)",
            oldest,
            len(holders),
            holders,
        )

    def start_early_irecv(self, hint: dict[str, Any]) -> None:
        """Take one engine hint; post the recv immediately while slots
        remain, otherwise queue it.

        Idempotent: duplicate hints for the same (channel, seqno), or
        hints for an already-delivered seqno (the
        ``_early_recv_last_delivered`` prefix watermark), are ignored —
        the edge sends one frame per seqno, and a second irecv for a
        delivered seqno would misalign the whole channel by one frame
        without any error anywhere.
        """
        fields = hint_fields(hint)
        if fields is None:
            lwd_logger.warning("[Lwd][pre-recv] malformed hint %s, skipped", hint)
            return
        table = hint_row_table(hint)
        if table is None:
            # Without a consistent row table the stager cannot split the
            # payload per request; a wrong table writes embeddings into
            # another request's buffer (silent corruption). Dropping the
            # hint leaves that batch's requests unscheduled - loud enough
            # to debug, unlike a mis-staged batch.
            lwd_logger.warning(
                "[Lwd][pre-recv] hint seqno=%d carries no consistent row "
                "table, dropped (both sides must deploy the row-table "
                "hint protocol)",
                fields[0],
            )
            return
        seqno = fields[0]
        key = (hint["channel"], seqno)
        with self._early_recv_lock:
            if (
                seqno <= self._early_recv_last_delivered
                or key in self._early_recv_futures
                or key in self._early_recv_pending
            ):
                return
            # The row table rides the hint itself; the pump builds the
            # stager plan from it at submit time.
            self._early_recv_pending[key] = hint
        self._early_recv_pump()

    def _early_recv_pump(self) -> None:
        """Post queued hints as recvs in arrival order until the in-flight
        cap is full.

        In-flight = posted but not yet published (each entry holds a
        whole-batch channel receive buffer until its D2H lands), so this
        cap is a memory cap; ``_report_irecv_completions`` calls this
        after each publish, so the hint queue never drops anything.
        """
        while True:
            with self._early_recv_lock:
                if not self._early_recv_pending or len(self._early_recv_futures) >= self._early_recv_max_inflight:
                    return
                key, hint = next(iter(self._early_recv_pending.items()))
                fields = hint_fields(hint)
                if fields is None:
                    # Belt-and-braces: malformed hints were rejected at
                    # start_early_irecv; never let one wedge the queue.
                    del self._early_recv_pending[key]
                    continue
                seqno, num_tokens, aux_rows = fields
                try:
                    # Submit time: build the stager plan (per-request
                    # pinned buffers, segmented D2H layout) and borrow a
                    # device slot as the receive buffer - the channel
                    # layer then lands the payload itself.
                    request = self._lwd_build_up_recv_request(
                        seqno, num_tokens, aux_rows, hint_row_table(hint)
                    )
                except LwdHostStageSlotBusy:
                    # No free device slot: the hint stays queued in place
                    # (same discipline as the in-flight cap - never
                    # dropped); the next publish frees a slot and pumps.
                    return
                except LwdHostStageError:
                    # Geometry/row-table errors are protocol-level and
                    # not retryable: drop the hint so later batches keep
                    # flowing (this batch's requests stall loudly).
                    del self._early_recv_pending[key]
                    lwd_logger.exception(
                        "[Lwd][pre-recv] seqno=%s: build recv request "
                        "failed (geometry/protocol), hint dropped",
                        seqno,
                    )
                    continue
                del self._early_recv_pending[key]
                future = lwd_get_data_plane().submit_recv(request)
                self._early_recv_futures[key] = future
                inflight = len(self._early_recv_futures)
                queued = len(self._early_recv_pending)
            lwd_logger.info(
                "[Lwd][pre-recv] SUBMIT channel=%s seqno=%d num_tokens=%d aux_rows=%d inflight=%d queued=%d ts=%.3f",
                key[0].value,
                seqno,
                num_tokens,
                aux_rows,
                inflight,
                queued,
                time.monotonic(),
            )

    def _lwd_commit_received(
        self, key: tuple[LwdChannelType, int], future: LwdTransferFuture
    ) -> list[tuple[str, int]]:
        """Arrival publish after the segmented D2H (endpoint comm thread).

        The copies themselves ran on the channel's copy stream as soon as
        the recv completed (submit-time plan; see
        ``_lwd_build_up_recv_request``) - this only advances the arrival
        high-water marks, returns the device slot to the pool and drops
        the future's device-frame reference so the slot memory can be
        recycled by the next receive.

        Raises on a failed transfer (``result()`` raises in ERROR state):
        the caller logs it, still pops the entry and reports the bare
        watermark key - that batch's rows never publish, its requests
        stall loudly instead of the ledger advancing on data that never
        landed.

        Returns the per-request CUMULATIVE arrived rows ``[(req_id,
        rows), ...]`` for the done-MQ piggyback. Cumulative, never a
        per-chunk delta: the engine merges with max(), the merge
        designed for cumulative counts — a delta report once left the
        ledger stuck at the last chunk's rows (1168+2342 delivered, the
        ledger said 2342) and the W2 bound then rejected legitimate
        windows. Requests whose total did not advance are skipped (the
        max() merge makes re-reports idempotent but pointless).
        """
        stager = self._lwd_stager
        if stager is None:
            return []
        seqno = key[1]
        future.result()  # raises in ERROR state (skipped/aborted seqno)
        # Returns this batch's plan rows [(req_id, n, offset)] while
        # advancing the high-water marks and returning the device slot.
        rows = stager.publish_received(seqno)
        # The slot's device memory is pool-owned and recycled by the next
        # receive; the completed future's view would only pin it.
        future.release_device_tensor()
        arrived: list[tuple[str, int]] = []
        last_total = self._lwd_last_total
        for rid, _n, _offset in rows:
            total = stager.recv_rows(rid)
            if total <= last_total.get(rid, 0):
                continue
            last_total[rid] = total
            arrived.append((rid, total))
        if arrived:
            lwd_logger.info(
                "[Lwd][host-stage] ARRIVED seqno=%d cumulative=%s host_MB=%.1f",
                seqno,
                arrived,
                stager.host_bytes / 2**20,
            )
        return arrived

    def _report_irecv_completions(self) -> None:
        """Report completed recvs to the done MQ (comm thread), three
        phases per entry (design v7 §5.2):

          ① ``future.done()``       — HCCL recv completed (device slot)
          ② ``future.staged_done()``— segmented D2H into the per-request
             pinned buffers completed (channel copy stream; pure CPU
             query, never a wait)
          ③ publish + report — ``_lwd_commit_received`` advances the
             per-request high-water marks, returns the device slot and
             drops the future's device-frame reference; ONE message then
             carries both the watermark key and the per-request
             cumulative rows: ``((channel, seqno), [(req_id, rows), ...])``
             — the engine's drain advances the watermark AND the arrival
             ledger from the same message, and the arrival rows are what
             the visible-set scheduler reads to size the next window.

        Publish-then-pop: once ② holds, the future (and its slot) has no
        more readers — the runner harvests by input_batch, there is no SO
        consumption point — so the entry is popped, the delivered
        watermark advances (blocking late/duplicate hints from re-posting
        a delivered seqno), and the next queued hint posts immediately.
        """
        done_mq = self._irecv_done_mq
        if done_mq is None:
            return
        with self._early_recv_lock:
            pending = list(self._early_recv_futures.items())
        for key, future in pending:
            if not future.done():
                continue
            # ① Received: this recv's data (main frame + aux frame) has
            # fully landed in the device slot.
            lwd_logger.info(
                "[Lwd][pre-recv] DONE-RECV channel=%s seqno=%d ts=%.3f",
                key[0].value,
                key[1],
                time.monotonic(),
            )
            # ② Staged: the channel copy stream landed the segments in the
            # per-request pinned buffers. Keep polling until it flips -
            # never wait, never block this thread.
            if not future.staged_done():
                continue
            # ③ Publish the arrival high-water marks + report.
            arrived: list[tuple[str, int]] = []
            try:
                arrived = self._lwd_commit_received(key, future)
            except Exception:
                # Transfer/publish failure = this batch's rows never
                # publish; its requests stall loudly (the ledger stays
                # behind). Still pop the entry below: a held in-flight
                # slot would wedge every later batch behind it instead.
                lwd_logger.exception(
                    "[Lwd] comm-thread staged publish failed key=%s "
                    "(arrival stalls for its requests)",
                    key,
                )
            done_mq.enqueue((key, arrived) if arrived else key)
            lwd_logger.info(
                "[Lwd][pre-recv] REPORT channel=%s seqno=%d arrived=%s ts=%.3f",
                key[0].value,
                key[1],
                arrived or "-",
                time.monotonic(),
            )
            with self._early_recv_lock:
                # Publish-then-pop (even on publish failure: the recv WAS
                # delivered, re-posting it would misalign the channel)
                self._early_recv_futures.pop(key, None)
                if key[1] > self._early_recv_last_delivered:
                    self._early_recv_last_delivered = key[1]
            self._early_recv_pump()

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: GrammarOutput) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        _t0 = time.monotonic()
        output = self.model_runner.sample_tokens(grammar_output)
        _t_sample = time.monotonic()
        # rank-replay DOWN: send the hidden packet (channel-async,
        # in-stream ordered); the step meta stays on device until the
        # fenced harvest below brings it to the host with a device-side
        # stream dependency (dedicated stream wait_event on the ready
        # event, completion event recorded on the copy stream itself).
        payload = self.model_runner.take_lwd_pending_down_payload()
        if payload is not None and output is not None:
            (hidden, meta_dev, seg_lens_host, ready_event, req_ids) = payload
            seqno = self._lwd_next_down_seqno()
            lwd_logger.info(
                "[Lwd][cloud-worker] DOWN send seqno=%d numel=%d",
                seqno,
                hidden.numel(),
            )
            lwd_get_data_plane().submit_send(
                LwdTransferRequest(
                    channel=LwdChannelType.DOWN,
                    op="send",
                    num_elements=hidden.numel(),
                    tensor=hidden,
                    seqno=seqno,
                )
            )
            # Fenced harvest moved OFF the compute thread: attach the
            # unharvested 5-tuple + the per-step ready event to the output
            # object; the executor's output thread (enqueue_output, off
            # the forward path) does the device-side-fenced copy and
            # rewrites the carrier to the finished 4-tuple. Events do not
            # pickle - the output thread removes the attribute before the
            # response leaves the process.
            target = getattr(output, "_model_runner_output", output)
            target.lwd_down_carrier = (
                meta_dev,
                seg_lens_host,
                req_ids,
                hidden.numel(),
                seqno,
            )
            target.lwd_meta_ready_event = ready_event
            # [Lwd][perf] Per-step cloud timing: sample = sampling +
            # collection (rank computation included); send = the DOWN
            # submit (snapshot clone + isend + bridge wait); total = the
            # whole step. Harvest timing lives on the output thread's
            # meta-harvest log.
            lwd_logger.info(
                "[Lwd][perf] cloud-step seqno=%d sample=%.2f send=%.2f total=%.2fms rows=%d",
                seqno,
                (_t_sample - _t0) * 1000,
                (time.monotonic() - _t_sample) * 1000,
                (time.monotonic() - _t0) * 1000,
                hidden.shape[0] if hidden is not None else 0,
            )
        return output

    # ------------------------------------------------------------------ #
    # LWD housekeeping (control-plane glue)                               #
    # ------------------------------------------------------------------ #

    def _lwd_cloud_flush_finished(self, finished_req_ids: list[str]) -> None:
        """Cloud side (streaming): a finished request's DOWN stream simply
        STOPS (its last step packet already went out with that step).
        No FIN packet -- request termination is signaled by the control
        plane (v2.6).  Bookkeeping release lives on the runner
        (``lwd_release_finished``, the single owner of the release path);
        finished_req_ids is the engine's own liveness signal, so this
        cleanup does not depend on the control plane."""
        lwd_logger.info("[Lwd][cloud-worker] flush finished reqs=%s", list(finished_req_ids))
        for req_id in finished_req_ids:
            self._lwd_last_total.pop(req_id, None)
        self.model_runner.lwd_release_finished(finished_req_ids)

    def _register_lwd_prompt_embeds_provider(self) -> None:
        """Wire the draft proposer's first-pass prompt-embeds provider.

        The provider resolves the request's prompt embeds from the
        runner's ``input_batch.req_prompt_embeds`` (already injected by
        ``_lwd_inject_from_stager`` during prefill); a missing entry
        (not scheduled / not LWD) returns None and the proposer falls
        back to the token-id path.
        """
        from vllm_ascend.spec_decode.llm_base_proposer import (
            AscendSpecDecodeBaseProposer,
        )

        runner = self.model_runner

        def _lwd_prompt_embeds_provider(req_id: str) -> torch.Tensor | None:
            idx = runner.input_batch.req_id_to_index.get(req_id)
            if idx is None:
                return None
            return runner.input_batch.req_prompt_embeds.get(idx)

        AscendSpecDecodeBaseProposer.set_lwd_prompt_embeds_provider(_lwd_prompt_embeds_provider)

    def _lwd_next_down_seqno(self) -> int:
        seqno = self._lwd_down_next_seqno
        self._lwd_down_next_seqno += 1
        return seqno
