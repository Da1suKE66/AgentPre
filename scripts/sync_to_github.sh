#!/usr/bin/env bash
set -euo pipefail

ROOT="${AGENTPRE_ROOT:-/workspace/liluchen/AgentPre}"
CACHE_ROOT="${AGENTPRE_CACHE_ROOT:-/cache/liluchen/agentpre}"
REMOTE_URL="${AGENTPRE_GIT_REMOTE:-ssh://git@ssh.github.com:443/Da1suKE66/AgentPre.git}"
GITHUB_KEY="${AGENTPRE_GITHUB_KEY:-${HOME}/.ssh/id_ed25519_github}"
LOCK_DIR="${CACHE_ROOT}/locks/github-sync.lock"

mkdir -p "${CACHE_ROOT}/locks" "${CACHE_ROOT}/logs"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Another AgentPre GitHub sync is already running; skip."
  exit 0
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

cd "${ROOT}"
if [[ ! -d .git ]]; then
  git init -b main
fi

git config user.name "AgentPre Sync"
git config user.email "agentpre-sync@users.noreply.github.com"
git config core.sshCommand "ssh -i ${GITHUB_KEY} -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new"

if git remote get-url origin >/dev/null 2>&1; then
  current_remote="$(git remote get-url origin)"
  if [[ "${current_remote}" != "${REMOTE_URL}" ]]; then
    echo "Refusing to replace unexpected origin: ${current_remote}" >&2
    exit 2
  fi
else
  git remote add origin "${REMOTE_URL}"
fi

# Runtime outputs, environments, and cache data are intentionally excluded.
tracked_paths=()
for path in .gitignore README.md pyproject.toml requirements.lock assets configs scripts src tests reports; do
  [[ -e "${path}" ]] && tracked_paths+=("${path}")
done
if (( ${#tracked_paths[@]} == 0 )); then
  echo "No source, config, test, or report paths exist; skip."
  exit 0
fi
git add -- "${tracked_paths[@]}"

staged_patch="$(git diff --cached --no-ext-diff --binary)"
if grep -Eiq '(BEGIN [A-Z ]*PRIVATE KEY|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{20,})' <<<"${staged_patch}"; then
  echo "Refusing to commit: staged content resembles a credential." >&2
  exit 3
fi

if ! git diff --cached --quiet; then
  git commit -m "sync: AgentPre $(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

if git rev-parse --verify HEAD >/dev/null 2>&1; then
  git push -u origin main
fi
