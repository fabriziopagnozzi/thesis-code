#!/usr/bin/env bash
set -euo pipefail

unit="${SYSTEMD_UNIT_NAME:-thesis-$(date +%Y%m%d-%H%M%S)-$$}"
scope="${unit%.scope}.scope"

export EVALUATION_WORKERS="${EVALUATION_WORKERS:-20}"
export QUERY_GEOMETRY_WORKERS="${QUERY_GEOMETRY_WORKERS:-5}"

cleanup() {
  rc=$?
  systemctl --user stop "$scope" >/dev/null 2>&1 || true
  systemctl --user kill --kill-whom=all --signal=SIGKILL "$scope" >/dev/null 2>&1 || true
  exit "$rc"
}

trap cleanup EXIT INT TERM HUP

systemd-run --user --scope \
  --unit="$unit" \
  --collect \
  -p MemoryHigh="${MEMORY_HIGH:-30G}" \
  -p MemoryMax="${MEMORY_MAX:-35G}" \
  -p MemorySwapMax="${MEMORY_SWAP_MAX:-1G}" \
  -p MemoryOOMGroup=yes \
  -p KillMode=control-group \
  -p SendSIGKILL=yes \
  -p TasksMax="${TASKS_MAX:-128}" \
  "$@"