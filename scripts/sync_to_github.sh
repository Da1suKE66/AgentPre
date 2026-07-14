#!/usr/bin/env bash
set -euo pipefail

ROOT="${AGENTPRE_ROOT:-/workspace/liluchen/AgentPre}"
CACHE_ROOT="${AGENTPRE_CACHE_ROOT:-/cache/liluchen/agentpre}"
REMOTE_URL="${AGENTPRE_GIT_REMOTE:-ssh://git@ssh.github.com:443/Da1suKE66/AgentPre.git}"
GITHUB_KEY="${AGENTPRE_GITHUB_KEY:-${HOME}/.ssh/id_ed25519_github}"
LOCK_FILE="${CACHE_ROOT}/locks/github-sync.lock"

mkdir -p "${CACHE_ROOT}/locks" "${CACHE_ROOT}/logs" "${CACHE_ROOT}/tmp"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another AgentPre GitHub sync is already running; skip."
  exit 0
fi

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

if ! git diff --cached --quiet; then
  echo "Refusing periodic sync because the working index already has staged changes." >&2
  exit 5
fi

# Remember the ordinary index bytes before doing any alternate-index work.  A
# user may stage a file while this script is building its isolated commit.  In
# that case refreshing the ordinary index to the new HEAD would silently erase
# their staging, so the refresh below is allowed only when this fingerprint is
# unchanged.  A repository without an index is represented explicitly rather
# than confused with an empty file.
ORDINARY_INDEX_PATH="$(git rev-parse --git-path index)"
ordinary_index_fingerprint() {
  if [[ -f "${ORDINARY_INDEX_PATH}" ]]; then
    git hash-object --no-filters -- "${ORDINARY_INDEX_PATH}"
  elif [[ -e "${ORDINARY_INDEX_PATH}" ]]; then
    echo "Refusing periodic sync because the ordinary Git index is not a regular file: ${ORDINARY_INDEX_PATH}" >&2
    return 1
  else
    printf '%s\n' "INDEX_ABSENT"
  fi
}
ORDINARY_INDEX_FINGERPRINT_BEFORE="$(ordinary_index_fingerprint)"
# Close the small gap between the initial staged-work check and the snapshot:
# staging that completed in that interval must stop the sync before the
# alternate index can move HEAD.
if ! git diff --cached --quiet; then
  echo "Refusing periodic sync because the working index gained staged changes before isolated-index work began." >&2
  exit 5
fi

# Build the commit from an isolated index.  This prevents an unrelated file
# staged by an interactive user from being swept into the periodic commit.
INDEX_FILE="$(mktemp "${CACHE_ROOT}/tmp/github-sync-index.XXXXXX")"
rm -f "${INDEX_FILE}"
trap 'rm -f "${INDEX_FILE}"' EXIT
if git rev-parse --verify HEAD >/dev/null 2>&1; then
  GIT_INDEX_FILE="${INDEX_FILE}" git read-tree HEAD
fi

# Runtime outputs, environments, and cache data are intentionally excluded.
allowed_roots=(.gitignore README.md pyproject.toml requirements.lock assets configs scripts src tests outputs reports)
tracked_paths=()
for path in "${allowed_roots[@]}"; do
  if [[ -e "${path}" ]] || git ls-files --error-unmatch -- "${path}" >/dev/null 2>&1; then
    tracked_paths+=("${path}")
  fi
done
if (( ${#tracked_paths[@]} == 0 )); then
  echo "No source, config, test, or report paths exist; skip."
  exit 0
fi
GIT_INDEX_FILE="${INDEX_FILE}" git add -A -- "${tracked_paths[@]}"

while IFS= read -r -d '' staged_path; do
  case "${staged_path}" in
    .gitignore|README.md|pyproject.toml|requirements.lock|assets/*|configs/*|scripts/*|src/*|tests/*|outputs/*|reports/*)
      ;;
    *)
      echo "Refusing to commit non-allowlisted staged path: ${staged_path}" >&2
      exit 4
      ;;
  esac
done < <(GIT_INDEX_FILE="${INDEX_FILE}" git diff --cached --name-only -z)

staged_patch="$(GIT_INDEX_FILE="${INDEX_FILE}" git diff --cached --no-ext-diff --binary)"
if grep -Eiq '(BEGIN [A-Z ]*PRIVATE KEY|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{20,})' <<<"${staged_patch}"; then
  echo "Refusing to commit: staged content resembles a credential." >&2
  exit 3
fi

if ! GIT_INDEX_FILE="${INDEX_FILE}" git diff --cached --quiet; then
  GIT_INDEX_FILE="${INDEX_FILE}" git commit -m "sync: AgentPre $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  # The alternate index created the new HEAD; refresh the normal clean index
  # to that commit without changing any working-tree file.  Fail closed if an
  # interactive process staged anything after our initial check.
  ORDINARY_INDEX_FINGERPRINT_AFTER="$(ordinary_index_fingerprint)"
  if [[ "${ORDINARY_INDEX_FINGERPRINT_AFTER}" != "${ORDINARY_INDEX_FINGERPRINT_BEFORE}" ]]; then
    echo "Refusing to refresh the ordinary Git index because it changed during periodic sync; user staging was preserved." >&2
    exit 6
  fi
  git read-tree HEAD
fi

if git rev-parse --verify HEAD >/dev/null 2>&1; then
  git push -u origin main
fi
