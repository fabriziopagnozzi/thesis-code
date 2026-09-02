"""Bounded concurrency for report reads from high-latency artifact stores."""

from __future__ import annotations

import os

_REPORT_IO_WORKERS_ENV = 'REPORT_IO_WORKERS'
_DEFAULT_REPORT_IO_WORKERS = 1


def report_io_workers(item_count: int) -> int:
    """Return a deterministic, bounded worker count for independent artifact reads."""
    if item_count <= 0:
        return 1
    raw = os.environ.get(_REPORT_IO_WORKERS_ENV, str(_DEFAULT_REPORT_IO_WORKERS))
    try:
        configured = int(raw)
    except ValueError as exc:
        raise ValueError(
            f'{_REPORT_IO_WORKERS_ENV} must be a positive integer, got {raw!r}'
        ) from exc
    if configured < 1:
        raise ValueError(f'{_REPORT_IO_WORKERS_ENV} must be positive, got {configured}')
    return min(configured, item_count)
