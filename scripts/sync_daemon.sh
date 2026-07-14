#!/usr/bin/env bash
set -euo pipefail

# User-owned periodic runner for sync_to_github.sh.  This deliberately does not
# depend on cron/systemd, which makes it suitable for compute hosts where the
# system cron daemon is unavailable.  It performs no simulation or GPU work.

ROOT="${AGENTPRE_ROOT:-/workspace/liluchen/AgentPre}"
CACHE_ROOT="${AGENTPRE_CACHE_ROOT:-/cache/liluchen/agentpre}"
INTERVAL_MINUTES="${AGENTPRE_SYNC_INTERVAL_MINUTES:-30}"
INTERVAL_SECONDS_OVERRIDE="${AGENTPRE_SYNC_INTERVAL_SECONDS:-}"
HEARTBEAT_SECONDS="${AGENTPRE_SYNC_HEARTBEAT_SECONDS:-60}"

SCRIPT_PATH="${BASH_SOURCE[0]}"
if [[ "${SCRIPT_PATH}" != /* ]]; then
  SCRIPT_PATH="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)/$(basename "${SCRIPT_PATH}")"
fi
SYNC_SCRIPT="${ROOT}/scripts/sync_to_github.sh"

STATE_DIR="${CACHE_ROOT}/run"
LOCK_DIR="${CACHE_ROOT}/locks"
LOG_DIR="${CACHE_ROOT}/logs"
PID_FILE="${STATE_DIR}/github-sync-daemon.pid"
HEARTBEAT_FILE="${STATE_DIR}/github-sync-daemon.heartbeat"
DAEMON_LOCK="${LOCK_DIR}/github-sync-daemon.lock"
CONTROL_LOCK="${LOCK_DIR}/github-sync-daemon-control.lock"
LOG_FILE="${LOG_DIR}/github-sync-daemon.log"

usage() {
  cat <<'EOF'
Usage: sync_daemon.sh {start|run|status|stop}

  start   Launch a detached user-owned periodic sync daemon.
  run     Run the daemon in the foreground (used internally and for debugging).
  status  Report the PID and most recent heartbeat; exit 0 only while running.
  stop    Gracefully stop the daemon without signalling an unrelated process.

Environment:
  AGENTPRE_ROOT                       repository root
  AGENTPRE_CACHE_ROOT                 cache/state root
  AGENTPRE_SYNC_INTERVAL_MINUTES      sync interval, default 30
  AGENTPRE_SYNC_INTERVAL_SECONDS      positive-integer override for tests
  AGENTPRE_SYNC_HEARTBEAT_SECONDS     heartbeat interval, default 60
EOF
}

die_config() {
  echo "$*" >&2
  exit 2
}

if ! [[ "${INTERVAL_MINUTES}" =~ ^[0-9]+$ ]] || (( INTERVAL_MINUTES < 1 )); then
  die_config "AGENTPRE_SYNC_INTERVAL_MINUTES must be a positive integer."
fi
if [[ -n "${INTERVAL_SECONDS_OVERRIDE}" ]]; then
  if ! [[ "${INTERVAL_SECONDS_OVERRIDE}" =~ ^[0-9]+$ ]] \
    || (( INTERVAL_SECONDS_OVERRIDE < 1 )); then
    die_config "AGENTPRE_SYNC_INTERVAL_SECONDS must be a positive integer when set."
  fi
  INTERVAL_SECONDS="${INTERVAL_SECONDS_OVERRIDE}"
else
  INTERVAL_SECONDS="$((INTERVAL_MINUTES * 60))"
fi
if ! [[ "${HEARTBEAT_SECONDS}" =~ ^[0-9]+$ ]] || (( HEARTBEAT_SECONDS < 1 )); then
  die_config "AGENTPRE_SYNC_HEARTBEAT_SECONDS must be a positive integer."
fi

# Git-only service: make accidental CUDA/ROCm access impossible and keep CPU
# thread counts bounded even if a future helper imports a numerical library.
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES="void"
export HIP_VISIBLE_DEVICES=""
export ROCR_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ensure_state_directories() {
  mkdir -p "${STATE_DIR}" "${LOCK_DIR}" "${LOG_DIR}"
}

read_pid() {
  local pid=""
  if [[ -r "${PID_FILE}" ]]; then
    IFS= read -r pid < "${PID_FILE}" || true
  fi
  if [[ "${pid}" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "${pid}"
  fi
}

process_is_daemon() {
  local pid="$1"
  local command_line=""

  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1

  if [[ -r "/proc/${pid}/cmdline" ]]; then
    command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
  else
    command_line="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
  fi
  [[ "${command_line}" == *"sync_daemon.sh"* && "${command_line}" == *" run"* ]]
}

atomic_write() {
  local destination="$1"
  local contents="$2"
  local temporary="${destination}.tmp.$$"
  printf '%s\n' "${contents}" > "${temporary}"
  mv -f "${temporary}" "${destination}"
}

write_heartbeat() {
  local state="$1"
  local last_exit_code="$2"
  local next_sync_epoch="$3"
  local now_epoch
  local now_utc
  now_epoch="$(date +%s)"
  now_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  atomic_write "${HEARTBEAT_FILE}" \
    "pid=$$ state=${state} timestamp_utc=${now_utc} timestamp_epoch=${now_epoch} last_sync_exit_code=${last_exit_code} next_sync_epoch=${next_sync_epoch} interval_seconds=${INTERVAL_SECONDS}"
}

daemon_status() {
  local pid
  pid="$(read_pid)"
  if [[ -n "${pid}" ]] && process_is_daemon "${pid}"; then
    echo "AgentPre GitHub sync daemon is running (pid=${pid})."
    if [[ -r "${HEARTBEAT_FILE}" ]]; then
      echo "heartbeat: $(<"${HEARTBEAT_FILE}")"
    else
      echo "heartbeat: not written yet"
    fi
    return 0
  fi

  echo "AgentPre GitHub sync daemon is stopped."
  if [[ -r "${HEARTBEAT_FILE}" ]]; then
    echo "last heartbeat: $(<"${HEARTBEAT_FILE}")"
  fi
  return 3
}

run_daemon() {
  local current_pid
  local sync_pid=""
  local wait_pid=""
  local stop_requested=0
  local last_exit_code="not_run"
  local next_sync_epoch=0

  command -v flock >/dev/null 2>&1 \
    || die_config "flock is required for single-instance daemon operation."
  [[ -x "${SYNC_SCRIPT}" ]] \
    || die_config "GitHub sync script is missing or not executable: ${SYNC_SCRIPT}"

  ensure_state_directories

  exec 9>"${DAEMON_LOCK}"
  if ! flock -n 9; then
    echo "AgentPre GitHub sync daemon is already running; skip duplicate run."
    return 0
  fi

  current_pid="$(read_pid)"
  if [[ -n "${current_pid}" ]] && process_is_daemon "${current_pid}" \
    && [[ "${current_pid}" != "$$" ]]; then
    echo "AgentPre GitHub sync daemon is already running (pid=${current_pid})."
    return 0
  fi
  atomic_write "${PID_FILE}" "$$"

  request_stop() {
    stop_requested=1
    if [[ -n "${sync_pid}" ]]; then
      kill -TERM "${sync_pid}" 2>/dev/null || true
    fi
    if [[ -n "${wait_pid}" ]]; then
      kill -TERM "${wait_pid}" 2>/dev/null || true
    fi
  }

  cleanup() {
    local recorded_pid
    recorded_pid="$(read_pid)"
    write_heartbeat "stopped" "${last_exit_code}" 0 || true
    if [[ "${recorded_pid}" == "$$" ]]; then
      rm -f "${PID_FILE}"
    fi
  }

  trap request_stop TERM INT HUP
  trap cleanup EXIT
  write_heartbeat "starting" "${last_exit_code}" 0

  while (( stop_requested == 0 )); do
    write_heartbeat "syncing" "${last_exit_code}" 0
    echo "sync_attempt_started timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) daemon_pid=$$"
    "${SYNC_SCRIPT}" &
    sync_pid=$!
    if wait "${sync_pid}"; then
      last_exit_code=0
    else
      last_exit_code=$?
    fi
    sync_pid=""
    echo "sync_attempt_finished timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) exit_code=${last_exit_code} daemon_pid=$$"
    (( stop_requested == 0 )) || break

    next_sync_epoch="$(( $(date +%s) + INTERVAL_SECONDS ))"
    local remaining="${INTERVAL_SECONDS}"
    while (( remaining > 0 && stop_requested == 0 )); do
      local sleep_seconds="${HEARTBEAT_SECONDS}"
      if (( sleep_seconds > remaining )); then
        sleep_seconds="${remaining}"
      fi
      write_heartbeat "sleeping" "${last_exit_code}" "${next_sync_epoch}"
      sleep "${sleep_seconds}" &
      wait_pid=$!
      wait "${wait_pid}" 2>/dev/null || true
      wait_pid=""
      remaining="$((remaining - sleep_seconds))"
    done
  done
}

start_daemon() {
  local pid
  local launcher_pid
  local attempt

  command -v flock >/dev/null 2>&1 \
    || die_config "flock is required for single-instance daemon operation."
  [[ -x "${SYNC_SCRIPT}" ]] \
    || die_config "GitHub sync script is missing or not executable: ${SYNC_SCRIPT}"

  ensure_state_directories

  exec 8>"${CONTROL_LOCK}"
  flock 8
  pid="$(read_pid)"
  if [[ -n "${pid}" ]] && process_is_daemon "${pid}"; then
    echo "AgentPre GitHub sync daemon is already running (pid=${pid})."
    return 0
  fi
  rm -f "${PID_FILE}"

  AGENTPRE_ROOT="${ROOT}" \
  AGENTPRE_CACHE_ROOT="${CACHE_ROOT}" \
  AGENTPRE_SYNC_INTERVAL_MINUTES="${INTERVAL_MINUTES}" \
  AGENTPRE_SYNC_INTERVAL_SECONDS="${INTERVAL_SECONDS_OVERRIDE}" \
  AGENTPRE_SYNC_HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS}" \
    nohup "${SCRIPT_PATH}" run 8>&- </dev/null >>"${LOG_FILE}" 2>&1 &
  launcher_pid=$!

  for attempt in $(seq 1 50); do
    pid="$(read_pid)"
    if [[ -n "${pid}" ]] && process_is_daemon "${pid}"; then
      echo "Started AgentPre GitHub sync daemon (pid=${pid}, interval=${INTERVAL_SECONDS}s)."
      echo "Log: ${LOG_FILE}"
      return 0
    fi
    if ! kill -0 "${launcher_pid}" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done

  echo "Failed to start AgentPre GitHub sync daemon; inspect ${LOG_FILE}." >&2
  return 1
}

stop_daemon() {
  local pid
  local attempt

  command -v flock >/dev/null 2>&1 \
    || die_config "flock is required for single-instance daemon operation."
  ensure_state_directories
  exec 8>"${CONTROL_LOCK}"
  flock 8
  pid="$(read_pid)"
  if [[ -z "${pid}" ]]; then
    echo "AgentPre GitHub sync daemon is already stopped."
    return 0
  fi
  if ! process_is_daemon "${pid}"; then
    echo "Removing stale daemon PID file without signalling pid=${pid}."
    rm -f "${PID_FILE}"
    return 0
  fi

  kill -TERM "${pid}"
  for attempt in $(seq 1 100); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      rm -f "${PID_FILE}"
      echo "Stopped AgentPre GitHub sync daemon (pid=${pid})."
      return 0
    fi
    sleep 0.1
  done

  echo "Daemon pid=${pid} did not stop within 10 seconds; it was not force-killed." >&2
  return 1
}

case "${1:-}" in
  start)
    start_daemon
    ;;
  run)
    run_daemon
    ;;
  status)
    daemon_status
    ;;
  stop)
    stop_daemon
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 64
    ;;
esac
