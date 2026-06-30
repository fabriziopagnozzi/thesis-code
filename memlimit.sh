#!/usr/bin/env bash
set -euo pipefail

unit="${SYSTEMD_UNIT_NAME:-thesis-$(date +%Y%m%d-%H%M%S)-$$}"

export EVALUATION_WORKERS=20
export QUERY_GEOMETRY_WORKERS=10

exec systemd-run --user --scope \
  --unit="$unit" \
  -p MemoryHigh="${MEMORY_HIGH:-40G}" \
  -p MemoryMax="${MEMORY_MAX:-44G}" \
  -p MemorySwapMax="${MEMORY_SWAP_MAX:-1G}" \
  "$@"
  # -p TasksMax="${TASKS_MAX:-64}" \
