# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LwdTransferFuture: completion handle for one data plane transfer.

Ported (simplified) from the demo branch's ``lwd_comm/future.py``:
pure-CPU ``done()`` via ``event.query()``, blocking ``wait()`` for the
debug/timeout path, in-place ``_bind`` for deferred (out-of-order
sequenced) futures, and exactly-once ``_finalize`` releasing the send
keepalive.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import torch

from vllm_ascend.distributed.lwd_data_plane.lwd_types import (
    LwdTransferRequest,
    LwdTransferResult,
    LwdTransferStatus,
)


class LwdTransferFuture:
    """Tracks one posted (or held) channel op.

    Completion means the cross-node P2P op has finished on the device,
    witnessed by ``done_event`` recorded on the channel stream right
    behind the bridged HCCL op.
    """

    def __init__(
        self,
        request: LwdTransferRequest,
        handles: list[Any],
        done_event: torch.npu.Event | None,
        tensor: torch.Tensor | None = None,
        keepalive: Any = None,
        aux_tensor: torch.Tensor | None = None,
        staged_event: torch.npu.Event | None = None,
    ) -> None:
        self._request = request
        self._handles = handles
        self._done_event = done_event
        self._tensor = tensor
        self._aux_tensor = aux_tensor
        self._keepalive = keepalive
        self._staged_event = staged_event
        self._status = LwdTransferStatus.PENDING
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._done_cond = threading.Condition(self._lock)
        self._finalized = False

    # ------------------------------------------------------------------ #
    # Deferred (out-of-order sequenced) support                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def deferred(cls, request: LwdTransferRequest) -> LwdTransferFuture:
        """A future for a held request; bound in place when its seqno
        turn arrives."""
        return cls(request, [], None)

    def _bind(
        self,
        *,
        handles: list[Any],
        done_event: torch.npu.Event | None,
        tensor: torch.Tensor | None = None,
        keepalive: Any = None,
        aux_tensor: torch.Tensor | None = None,
        staged_event: torch.npu.Event | None = None,
    ) -> None:
        with self._done_cond:
            self._handles = handles
            self._done_event = done_event
            self._tensor = tensor
            self._aux_tensor = aux_tensor
            self._keepalive = keepalive
            self._staged_event = staged_event
            self._done_cond.notify_all()

    # ------------------------------------------------------------------ #
    # Status                                                              #
    # ------------------------------------------------------------------ #

    @property
    def request(self) -> LwdTransferRequest:
        return self._request

    def done(self) -> bool:
        """Pure CPU query; never blocks, never touches the device."""
        with self._lock:
            if self._status is not LwdTransferStatus.PENDING:
                return True
            event = self._done_event
        if event is None:
            return False
        try:
            if event.query():
                with self._lock:
                    if self._status is LwdTransferStatus.PENDING:
                        self._status = LwdTransferStatus.OK
                        self._done_cond.notify_all()
                return True
        except Exception as exc:  # device error surfaces as ERROR
            with self._lock:
                if self._status is LwdTransferStatus.PENDING:
                    self._status = LwdTransferStatus.ERROR
                    self._error = exc
                    self._done_cond.notify_all()
            return True
        return False

    def wait(self, timeout: float | None = None) -> LwdTransferResult:
        """Block until completion (readiness-gate path)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._done_cond:
            while self._status is LwdTransferStatus.PENDING:
                if self._done_event is None:
                    # Deferred future not yet bound; wait for _bind.
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError(
                            f"lwd-comm wait timed out (unbound) on "
                            f"{self._request.channel.value} seqno="
                            f"{self._request.seqno}"
                        )
                    self._done_cond.wait(timeout=remaining or 0.001)
                    continue
                if self._done_event.query():
                    self._status = LwdTransferStatus.OK
                    break
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(
                        f"lwd-comm wait timed out on {self._request.channel.value} seqno={self._request.seqno}"
                    )
                self._done_cond.wait(timeout=min(0.001, remaining or 0.001))
            if self._status is LwdTransferStatus.ERROR:
                raise RuntimeError(
                    f"lwd-comm op failed on {self._request.channel.value} seqno={self._request.seqno}"
                ) from self._error
        return LwdTransferResult(
            status=self._status,
            tensor=self._tensor,
            aux_tensor=self._aux_tensor,
        )

    def result(self) -> LwdTransferResult:
        # TODO(lwd): confirm the behavior for a held future that is not bound
        # yet: it returns tensor=None, and wait_on_current_stream() is a no-op
        # before _bind.
        with self._lock:
            status, tensor, aux_tensor, error = (
                self._status,
                self._tensor,
                self._aux_tensor,
                self._error,
            )
        if status is LwdTransferStatus.ERROR:
            raise RuntimeError(
                f"lwd-comm op failed on {self._request.channel.value} seqno={self._request.seqno}"
            ) from error
        return LwdTransferResult(status=status, tensor=tensor, aux_tensor=aux_tensor)

    # ------------------------------------------------------------------ #
    # Device-side ordering for consumers                                  #
    # ------------------------------------------------------------------ #

    def wait_on_current_stream(self) -> None:
        """Order the *current* stream after the channel-stream completion
        event (pure device-side; CPU returns immediately)."""
        event = self._done_event
        if event is not None:
            torch.npu.current_stream().wait_event(event)

    # ------------------------------------------------------------------ #
    # Host staging (segmented D2H)                                       #
    # ------------------------------------------------------------------ #

    def staged_done(self) -> bool:
        """Pure-CPU query of the segmented-D2H completion event.

        ``True`` implies the pinned host copies are complete (the event is
        recorded on the channel's copy stream behind the last segment).
        Always ``True`` when staging was not requested, so callers can use
        it as a plain readiness gate after ``done()``.
        """
        event = self._staged_event
        if event is None:
            return True
        try:
            return bool(event.query())
        except Exception as exc:  # device error surfaces as ERROR
            with self._lock:
                if self._status is LwdTransferStatus.PENDING:
                    self._status = LwdTransferStatus.ERROR
                    self._error = exc
                    self._done_cond.notify_all()
            return True

    def release_device_tensor(self) -> None:
        """Drop the device-frame reference after its D2H copy completed.

        With host staging the receive buffer belongs to the caller's
        ping-pong pool and is recycled by the next receive once the pool
        frees the slot; a completed future holding the view would only
        keep the slot's device memory pinned. No-op when staging was not
        requested (then the device frame IS the delivered payload and
        ``result()`` must keep serving it)."""
        if self._staged_event is None:
            return
        with self._done_cond:
            self._tensor = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def _finalize(self, error: BaseException | None = None) -> None:
        """Exactly-once terminal bookkeeping: release the send keepalive.
        With ``error`` the future completes in ERROR state immediately
        (abort/skip path) so waiters fail promptly instead of timing out."""
        with self._done_cond:
            if self._finalized:
                return
            self._finalized = True
            self._keepalive = None
            self._handles = []
            if error is not None:
                self._status = LwdTransferStatus.ERROR
                self._error = error
                self._done_cond.notify_all()
                return
            if self._status is LwdTransferStatus.PENDING:
                self._status = (
                    LwdTransferStatus.OK
                    if (self._done_event is not None and self._done_event.query())
                    else self._status
                )
