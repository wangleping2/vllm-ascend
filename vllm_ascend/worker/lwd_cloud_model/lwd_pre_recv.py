# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD early receive: engine⇄worker bypass link + hint/watermark protocol.

There is exactly one data path (same as the existing early receive; no new bypass)::

    cloud engine IO thread                cloud endpoint comm thread
    ───────────────                       ──────────────────────
    RangeNotify arrives (edge announcement)
      └─ post_irecv_hint ──hint MQ (downstream)──►  start_early_irecv
                                              └─ submit_recv(UP, seqno, numel, aux)
                                                    │  irecv already pre-posted
                                              future.done() polling
                                                    │
    record_irecv_completions ◄──done MQ (upstream)── _report_irecv_completions
      └─ watermark → is_irecv_complete(UP, seqno)
           └─ PD_mix scheduler gate: release the prefill batch only once received

Both MQs are bypasses; neither may borrow ``rpc_broadcast_mq``: that queue
is consumed single-threaded by ``worker_busy_loop``, which blocks for
hundreds of milliseconds inside ``execute_model`` for a prefill batch — a
hint queued there would never get its recv posted "early".

* hint MQ: the engine creates the writer (handle exported to env before
  worker creation); the cloud endpoint rank attaches the reader by handle;
* done MQ: the cloud endpoint rank creates the writer; after the executor
  is up, the engine fetches its handle over ``collective_rpc``
  (``LwdCloudWorker.lwd_early_recv_done_handle``) and attaches the reader.

Watermark semantics: ``is_irecv_complete(channel, seqno)`` ==
``seqno <= watermark``. Per-channel completion order = seqno order
(``channel.py`` FIFO: if the head has not completed, nothing behind it
has), so the "completed set" is always a prefix and one max watermark
represents it exactly. The watermark is only a **readiness predicate**;
it plays no role in task priority.

Hint fields on this branch (chunked prefill + mm/hash): ``num_tokens`` =
the batch's (this chunk batch's) flat token count, ``aux_rows`` = the sum
of token rows of **mrope-carrying** requests in the batch — the UP main
frame numel is computed **worker-side** from the hash payload formula
using local config (see
``lwd_cloud_worker._lwd_build_up_recv_request``); the hint carries only
"facts" (token count, mrope row count), never the size formula, so the
two sides cannot compute it differently; aux frame numel = aux_rows*3
(int64, [n,3], same seqno as the main frame).

Non-endpoint ranks (the cloud's remaining TP ranks) hold no cross-node
channel and no host stager: they post no recv and report nothing; the
runner's shared-window TP broadcast is their only source of the staged
rows. This module's hint/report line runs on the endpoint rank only.
"""

from __future__ import annotations

import base64
import json
import os
import threading
from collections.abc import Callable, Iterable
from dataclasses import asdict
from typing import TYPE_CHECKING

from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue

#: Env var carrying the downstream hint MQ handle. The executor exports it
#: before creating the worker; the worker rebuilds the reader in
#: ``init_device`` — both sides honor this single constant name.
LWD_RECV_HINT_MQ_ENV = "VLLM_ASCEND_LWD_RECV_HINT_HANDLE"

#: Per-request row table carried by a recv hint (see ``hint_row_table``):
#: (req_ids, chunk rows, prompt_offsets, prompt_lens). Chunk rows are
#: batch-layout coordinates; prompt offsets are prompt coordinates - the
#: two must never be merged (a request can arrive across several chunks).
LwdRecvRowTable = tuple[list[str], list[int], list[int], list[int]]


# Bypass MQ spec: 1KB per slot, 64 slots. One hint/report per batch; the
# edge pending gate (``LwdEdgeScheduler._LWD_MAX_PENDING_CHUNKS = 3``)
# keeps unconsumed chunks in the single digits, and hints are a
# correctness dependency (losing one = that recv is never pre-posted, the
# watermark stalls, and the gated batch is never dispatched) — so plenty
# of headroom, no flow control.
_SIDEBAND_MAX_CHUNK_BYTES = 1024
_SIDEBAND_MAX_CHUNKS = 64


# ---------------------------------------------------------------------- #
# Hints: produced by the scheduler, consumed by the worker comm thread   #
# ---------------------------------------------------------------------- #
def make_recv_hint(
    seqno: int,
    num_tokens: int,
    aux_rows: int = 0,
    req_ids: list | None = None,
    rows: list | None = None,
    prompt_offsets: list | None = None,
    prompt_lens: list | None = None,
) -> dict:
    """Build one UP early-recv hint carrying the host-staging row table.

    * ``num_tokens``: the batch's flat token count (the main-frame size
      is derived worker-side from the hash payload formula)
    * ``aux_rows``: the batch's mrope-carrying token rows (0 = text-only
      batch, zero aux traffic)
    * ``req_ids``/``rows``: per-request ids and THIS CHUNK's row counts
      (same order as the notify items; the batch-internal payload layout
      derives from the rows by accumulation)
    * ``prompt_offsets``: each chunk's start within its OWN prompt (the
      stager's row_base)
    * ``prompt_lens``: per-request full prompt rows (host buffer sizing
      authority, one-shot allocation at first chunk)

    The two coordinate systems (chunk rows vs prompt offsets) must both
    ride the hint and must never be merged: a request can be delivered
    across several chunks, and where the second segment lands in the
    request's buffer (prompt coordinate) is independent of where it sits
    in this batch's payload (chunk-accumulated coordinate). Merging them
    writes the second chunk over the first one's rows - silently.
    """
    return {
        "channel": LwdChannelType.UP,
        "seqno": int(seqno),
        "req_ids": [str(r) for r in (req_ids or ())],
        "rows": [int(r) for r in (rows or ())],
        "prompt_offsets": [int(o) for o in (prompt_offsets or ())],
        "prompt_lens": [int(n) for n in (prompt_lens or ())],
        "num_tokens": int(num_tokens),
        "aux_rows": int(aux_rows),
    }


def hint_fields(hint: dict) -> tuple[int, int, int] | None:
    """Hint self-check: returns (seqno, num_tokens, aux_rows); None for a malformed hint."""
    try:
        seqno = int(hint["seqno"])
        num_tokens = int(hint["num_tokens"])
        aux_rows = int(hint.get("aux_rows", 0))
    except (KeyError, TypeError, ValueError):
        return None
    if seqno < 0 or num_tokens <= 0 or aux_rows < 0 or aux_rows > num_tokens:
        return None
    return seqno, num_tokens, aux_rows


def hint_row_table(hint: dict) -> LwdRecvRowTable | None:
    """Row-table self-check: ``(req_ids, rows, prompt_offsets,
    prompt_lens)``; None when the table is absent or inconsistent.

    A mismatched table (lengths disagree with the request count, rows
    not summing to the flat token count, non-positive lengths) would
    make the stager slice windows by wrong rows and write embeddings
    into another request's position - silent corruption - so a bad table
    dies at the entrance and the hint is dropped by the caller.
    """
    try:
        req_ids = [str(r) for r in hint["req_ids"]]
        rows = [int(r) for r in hint["rows"]]
        prompt_offsets = [int(o) for o in hint["prompt_offsets"]]
        prompt_lens = [int(n) for n in hint["prompt_lens"]]
        num_tokens = int(hint["num_tokens"])
    except (KeyError, TypeError, ValueError):
        return None
    if not req_ids:
        return None
    if (
        len(rows) != len(req_ids)
        or len(prompt_offsets) != len(req_ids)
        or len(prompt_lens) != len(req_ids)
    ):
        return None
    if any(r <= 0 for r in rows) or sum(rows) != num_tokens:
        return None
    if any(o < 0 for o in prompt_offsets):
        return None
    if any(n <= 0 for n in prompt_lens):
        return None
    return req_ids, rows, prompt_offsets, prompt_lens


# ---------------------------------------------------------------------- #
# Bypass MQ lifecycle                                                    #
# ---------------------------------------------------------------------- #
def _new_sideband_mq() -> MessageQueue:
    """A 1-reader 1-writer bypass MQ (hint and report share the spec; only the endpoint rank participates)."""
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    return MessageQueue(
        1,
        1,
        max_chunk_bytes=_SIDEBAND_MAX_CHUNK_BYTES,
        max_chunks=_SIDEBAND_MAX_CHUNKS,
    )


def _encode_handle(handle: Handle) -> str:
    """Encode a MessageQueue handle into an env-safe ASCII string.

    Every Handle field is a JSON-native type, so the sideband handoff
    never needs pickle (a forbidden import in this repo).
    """
    return base64.b64encode(json.dumps(asdict(handle)).encode("ascii")).decode("ascii")


def _decode_handle(raw: str) -> Handle:
    """Rebuild a Handle from its JSON encoding.

    JSON arrays come back as lists; restore the tuple the MessageQueue
    expects in buffer_handle.
    """
    fields = json.loads(base64.b64decode(raw))
    if fields.get("buffer_handle") is not None:
        fields["buffer_handle"] = tuple(fields["buffer_handle"])
    from vllm.distributed.device_communicators.shm_broadcast import Handle

    return Handle(**fields)


def create_hint_mq() -> MessageQueue:
    """Engine side: create the hint MQ (writer) and export the handle to
    the environment.

    Must be called before workers are created (they rebuild the reader
    from the env at startup); both fork and spawn inherit via env vars.
    """
    mq = _new_sideband_mq()
    os.environ[LWD_RECV_HINT_MQ_ENV] = _encode_handle(mq.export_handle())
    return mq


def attach_hint_mq() -> MessageQueue | None:
    """Worker side: rebuild the hint MQ reader from the handle in the
    environment.

    Returns None when no handle exists (this role takes no hints): the
    comm thread still runs reporting, so a missing hint only delays the
    early recv, never stalls the watermark.
    """
    raw = os.environ.get(LWD_RECV_HINT_MQ_ENV)
    if raw is None:
        return None
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    return MessageQueue.create_from_handle(_decode_handle(raw), 0)


def create_report_mq() -> MessageQueue:
    """Worker side: create the done MQ (writer); the engine fetches its
    handle over ``collective_rpc`` once the executor is up."""
    return _new_sideband_mq()


# ---------------------------------------------------------------------- #
# Scheduler process side: hint dispatch + completion watermark           #
# ---------------------------------------------------------------------- #
_hint_sender: Callable[[dict], None] | None = None
_watermarks: dict[LwdChannelType, int] = {}
_lock = threading.Lock()


def register_hint_sender(sender: Callable[[dict], None]) -> None:
    """Install the transport used to dispatch hints (once per engine
    process).

    The closure passed in must **deliver reliably** (enqueue the hint
    into the hint MQ): with the gate on, hints are a correctness
    dependency — lose one and that recv is never pre-posted, the
    watermark stalls, and the gated batch is never dispatched.
    """
    global _hint_sender
    _hint_sender = sender


def post_irecv_hint(hint: dict) -> None:
    """Send one recv hint to the local worker's comm thread.

    The healthy path logs nothing (the caller ``lwd_cloud_engine`` logs
    ``HINT ...``): this module keeps only should-never-happen diagnostics,
    and they must be visible in the logs.
    """
    if _hint_sender is None:
        # The engine registers the sender during startup, before it accepts
        # any edge message, so a missing one is a wiring error; dropping the
        # hint would stall the pre-recv gate forever (nobody would pre-post
        # that recv, the watermark would never advance).
        raise RuntimeError(f"[LWD] early-recv hint for seqno {hint.get('seqno')} has no registered sender")
    _hint_sender(hint)


def record_irecv_completions(
    items: Iterable[tuple],
) -> list[tuple[LwdChannelType, int]]:
    """Merge worker-reported completions into the per-channel watermark
    (called after the engine drains the done MQ each step).

    Returns the entries that **actually advanced** the watermark this
    call, for the caller (``lwd_cloud_engine``) to log: whether the
    watermark is moving is the first-hand evidence for locating which
    segment is stuck, so the log lives in a module proven to emit.
    """
    advanced: list[tuple[LwdChannelType, int]] = []
    with _lock:
        for item in items:
            channel, seqno = item[0], item[1]
            if seqno > _watermarks.get(channel, -1):
                _watermarks[channel] = seqno
                advanced.append((channel, seqno))
    return advanced


def watermark(channel: LwdChannelType) -> int:
    """Current watermark: the max confirmed-complete seqno on the channel (-1 = none yet)."""
    with _lock:
        return _watermarks.get(channel, -1)


def is_irecv_complete(channel: LwdChannelType, seqno: int) -> bool:
    """Whether the seqno recv has completed on the channel — a pure
    readiness predicate, no logging.

    "Who is being waited on" comes from the gate's own log line
    (``lwd_cloud_mixed_scheduler`` logs ``gate seqno=.. watermark=..
    ready=..`` every step), so this stays a pure predicate.
    """
    return seqno <= watermark(channel)
