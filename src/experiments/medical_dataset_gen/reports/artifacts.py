from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    df = pl.DataFrame([dict(row) for row in rows], infer_schema_length=None)
    df.write_csv(path)
