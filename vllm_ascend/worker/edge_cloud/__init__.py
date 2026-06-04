# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Edge-cloud collaboration subpackage.

Contains workers and helpers for the edge-cloud collaboration feature, including
the :class:`SharedModelEdgeWorker` which allows multiple DP-rank edge workers
to live in a single process and share one model replica in NPU memory.
"""
