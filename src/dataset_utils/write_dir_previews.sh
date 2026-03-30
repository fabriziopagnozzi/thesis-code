#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dir> [rows] [outdir]" >&2
    exit 1
fi

DIR="$(realpath "$1")"
ROWS="${2:-100}"
OUT_DIR="${3:-"$(dirname "$0")/data/$(basename "$DIR")"}"

mkdir -p "$OUT_DIR"

for file in "$DIR"/*.csv "$DIR"/*.json "$DIR"/*.jsonl; do
    [[ -e "$file" ]] || continue
    ext="${file##*.}"
    name="$(basename "$file" ".$ext")"
    out_ext="$([[ "$ext" == "csv" ]] && echo "csv" || echo "json")"

    echo "Reading $name.$ext..."

    uv run "$(dirname "$0")/file_head_reader.py" --path "$file" --rows "$ROWS" > "$OUT_DIR/${name}.${out_ext}"
done

echo "Done. Output in $OUT_DIR/"
