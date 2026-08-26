#!/usr/bin/env bash
set -euo pipefail

UNIT_BASE="${SYSTEMD_UNIT_NAME:-thesis-$(date +%m%d-%H%M%S)-$$}"
UNIT="${UNIT_BASE%.service}.service"

cleanup() {
  rc=$?
  trap - EXIT INT TERM HUP
  systemctl --user stop "$UNIT" >/dev/null 2>&1 || true
  systemctl --user kill "$UNIT" \
    --kill-whom=all \
    --signal=SIGKILL >/dev/null 2>&1 || true
  exit "$rc"
}

trap cleanup EXIT INT TERM HUP

systemd-run --user \
  --unit="$UNIT" \
  --collect \
  --wait \
  --pipe \
  --same-dir \
  --setenv="EVALUATION_WORKERS=${EVALUATION_WORKERS:-9}" \
  --setenv="QUERY_GEOMETRY_WORKERS=${QUERY_GEOMETRY_WORKERS:-8}" \
  --setenv=HF_TOKEN \
  --property=Type=exec \
  -p MemoryHigh="${MEMORY_HIGH:-59G}" \
  -p MemoryMax="${MEMORY_MAX:-60G}" \
  -p MemorySwapMax="${MEMORY_SWAP_MAX:-0G}" \
  -p OOMPolicy=kill \
  -p KillMode=control-group \
  -p SendSIGKILL=yes \
  "$@"
