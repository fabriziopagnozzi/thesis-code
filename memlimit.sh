#!/usr/bin/env bash
set -euo pipefail

unit_base="${SYSTEMD_UNIT_NAME:-thesis-$(date +%Y%m%d-%H%M%S)-$$}"
unit="${unit_base%.service}.service"

cleanup() {
  rc=$?
  trap - EXIT INT TERM HUP
  systemctl --user stop "$unit" >/dev/null 2>&1 || true
  systemctl --user kill "$unit" \
    --kill-whom=all \
    --signal=SIGKILL >/dev/null 2>&1 || true
  exit "$rc"
}

trap cleanup EXIT INT TERM HUP

systemd-run --user \
  --unit="$unit" \
  --collect \
  --wait \
  --pipe \
  --same-dir \
  --setenv="EVALUATION_WORKERS=${EVALUATION_WORKERS:-18}" \
  --setenv="QUERY_GEOMETRY_WORKERS=${QUERY_GEOMETRY_WORKERS:-10}" \
  --setenv=HF_TOKEN \
  --property=Type=exec \
  -p MemoryHigh="${MEMORY_HIGH:-47G}" \
  -p MemoryMax="${MEMORY_MAX:-48G}" \
  -p MemorySwapMax="${MEMORY_SWAP_MAX:-0G}" \
  -p OOMPolicy=kill \
  -p KillMode=control-group \
  -p SendSIGKILL=yes \
  "$@"
