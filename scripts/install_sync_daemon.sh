#!/usr/bin/env bash
set -euo pipefail

ROOT="${AGENTPRE_ROOT:-/workspace/liluchen/AgentPre}"
LEGACY_CRON_MARKER="# AgentPre periodic GitHub sync"

remove_legacy_cron_schedule() {
  local existing
  local filtered

  # Some compute images do not install crontab at all.  The user daemon is
  # independent of cron, so absence of this cleanup tool must not block start.
  if ! command -v crontab >/dev/null 2>&1; then
    echo "crontab is unavailable; no legacy AgentPre schedule was removed."
    return 0
  fi

  # `crontab -l` exits non-zero when the user has no crontab.  That is also a
  # clean state and should not block daemon installation.
  if ! existing="$(crontab -l 2>/dev/null)"; then
    return 0
  fi

  # Remove an exact legacy marker only when the immediately following line is
  # the matching AgentPre sync command for this repository.  A stray marker
  # must never consume an unrelated cron entry.
  filtered="$(printf '%s\n' "${existing}" | awk \
    -v marker="${LEGACY_CRON_MARKER}" \
    -v sync_script="${ROOT}/scripts/sync_to_github.sh" '
    pending_marker == 1 {
      if (index($0, sync_script) > 0) {
        pending_marker = 0
        next
      }
      print marker
      pending_marker = 0
    }
    $0 == marker { pending_marker = 1; next }
    { print }
    END { if (pending_marker == 1) print marker }
  ')"
  if [[ "${filtered}" != "${existing}" ]]; then
    printf '%s\n' "${filtered}" | crontab -
    echo "Removed legacy AgentPre cron schedule; unrelated crontab entries were preserved."
  fi
}

if [[ ! -x "${ROOT}/scripts/sync_daemon.sh" ]]; then
  echo "AgentPre sync daemon script is missing or not executable: ${ROOT}/scripts/sync_daemon.sh" >&2
  exit 2
fi

remove_legacy_cron_schedule
exec "${ROOT}/scripts/sync_daemon.sh" start
