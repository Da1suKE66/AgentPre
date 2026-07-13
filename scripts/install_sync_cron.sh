#!/usr/bin/env bash
set -euo pipefail

ROOT="${AGENTPRE_ROOT:-/workspace/liluchen/AgentPre}"
CACHE_ROOT="${AGENTPRE_CACHE_ROOT:-/cache/liluchen/agentpre}"
INTERVAL_MINUTES="${AGENTPRE_SYNC_INTERVAL_MINUTES:-30}"
MARKER="# AgentPre periodic GitHub sync"

if ! [[ "${INTERVAL_MINUTES}" =~ ^[0-9]+$ ]] \
  || (( INTERVAL_MINUTES < 1 || INTERVAL_MINUTES > 59 )); then
  echo "AGENTPRE_SYNC_INTERVAL_MINUTES must be an integer in [1, 59]." >&2
  exit 2
fi

mkdir -p "${CACHE_ROOT}/logs"
existing="$(crontab -l 2>/dev/null || true)"
filtered="$(printf '%s\n' "${existing}" | awk -v marker="${MARKER}" '
  skip == 1 { skip = 0; next }
  $0 == marker { skip = 1; next }
  { print }
')"
{
  printf '%s\n' "${filtered}"
  printf '%s\n' "${MARKER}"
  printf '*/%s * * * * AGENTPRE_ROOT=%q AGENTPRE_CACHE_ROOT=%q %q >> %q 2>&1\n' \
    "${INTERVAL_MINUTES}" "${ROOT}" "${CACHE_ROOT}" \
    "${ROOT}/scripts/sync_to_github.sh" "${CACHE_ROOT}/logs/github-sync.log"
} | crontab -

echo "Installed ${INTERVAL_MINUTES}-minute AgentPre GitHub sync."

