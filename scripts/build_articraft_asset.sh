#!/usr/bin/env bash
set -euo pipefail

# Recompile the pinned official Articraft record without network access, apply
# the checked-in all-link inertial specification, run the project's stricter
# name/inertial/mesh inspection, then copy only accepted runtime URDF/assets
# into the AgentPre cache.  Environment installation is a separate
# bootstrap step and must already be complete.

ROOT="${AGENTPRE_ROOT:-/workspace/liluchen/AgentPre}"
AGENTPRE_CACHE_ROOT="${AGENTPRE_CACHE_ROOT:-/cache/liluchen/agentpre}"
ARTICRAFT_ROOT="${ARTICRAFT_ROOT:-/cache/liluchen/articraft}"
ARTICRAFT_DATA_ROOT="${ARTICRAFT_DATA_ROOT:-/cache/liluchen/articraft-data}"
ARTICRAFT_ENV="${ARTICRAFT_ENV:-/cache/liluchen/articraft-env}"
ARTICRAFT_UV="${ARTICRAFT_UV:-/cache/liluchen/articraft-uv-bootstrap/bin/uv}"
ARTICRAFT_UV_CACHE="${ARTICRAFT_UV_CACHE:-/cache/liluchen/articraft-uv-cache}"
AGENTPRE_PYTHON="${AGENTPRE_PYTHON:-${AGENTPRE_CACHE_ROOT}/envs/agentpre-conda/bin/python}"

RECORD_ID="rec_microwave_oven_5e86f3429e954dcd9ab6c9d3a94db707"
ARTICRAFT_COMMIT="59eb5e0ed72a734111012b43f881423b15d4931d"
DATA_COMMIT="0cdcaa49f5571e9b4df04476c7f09587ee3ab7bd"
MATERIALIZATION_ROOT="${ARTICRAFT_DATA_ROOT}/cache/record_materialization/${RECORD_ID}"
INERTIAL_SPEC="${ROOT}/assets/articraft/${RECORD_ID}/inertials.json"
INERTIAL_SIDECAR="${MATERIALIZATION_ROOT}/agentpre_inertial_completion.json"

fail() {
  echo "$*" >&2
  exit 2
}

[[ -x "${ARTICRAFT_UV}" ]] || fail "Pinned uv bootstrap is missing: ${ARTICRAFT_UV}"
[[ -x "${AGENTPRE_PYTHON}" ]] || fail "AgentPre Python is missing: ${AGENTPRE_PYTHON}"
[[ -f "${INERTIAL_SPEC}" ]] || fail "Inertial specification is missing: ${INERTIAL_SPEC}"
[[ -x "${ARTICRAFT_ENV}/bin/articraft" ]] \
  || fail "Articraft environment is incomplete: ${ARTICRAFT_ENV}"
[[ -d "${ARTICRAFT_ROOT}/.git" ]] || fail "Articraft checkout is missing: ${ARTICRAFT_ROOT}"
[[ -d "${ARTICRAFT_DATA_ROOT}/.git" ]] \
  || fail "Articraft data checkout is missing: ${ARTICRAFT_DATA_ROOT}"

actual_articraft_commit="$(git -C "${ARTICRAFT_ROOT}" rev-parse HEAD)"
actual_data_commit="$(git -C "${ARTICRAFT_DATA_ROOT}" rev-parse HEAD)"
[[ "${actual_articraft_commit}" == "${ARTICRAFT_COMMIT}" ]] \
  || fail "Unexpected Articraft commit: ${actual_articraft_commit}"
[[ "${actual_data_commit}" == "${DATA_COMMIT}" ]] \
  || fail "Unexpected Articraft data commit: ${actual_data_commit}"
git -C "${ARTICRAFT_ROOT}" diff --quiet \
  || fail "Articraft checkout has unstaged tracked changes."
git -C "${ARTICRAFT_ROOT}" diff --cached --quiet \
  || fail "Articraft checkout has staged changes."
git -C "${ARTICRAFT_DATA_ROOT}" diff --quiet \
  || fail "Articraft data checkout has unstaged tracked changes."
git -C "${ARTICRAFT_DATA_ROOT}" diff --cached --quiet \
  || fail "Articraft data checkout has staged changes."
[[ -z "$(git -C "${ARTICRAFT_ROOT}" status --porcelain --untracked-files=normal)" ]] \
  || fail "Articraft checkout is not clean."
[[ -z "$(git -C "${ARTICRAFT_DATA_ROOT}" status --porcelain --untracked-files=normal)" ]] \
  || fail "Articraft data checkout is not clean."

mkdir -p "${AGENTPRE_CACHE_ROOT}/tmp" "${ARTICRAFT_UV_CACHE}"
export AGENTPRE_ROOT AGENTPRE_CACHE_ROOT
export UV_PROJECT_ENVIRONMENT="${ARTICRAFT_ENV}"
export UV_CACHE_DIR="${ARTICRAFT_UV_CACHE}"
export UV_PYTHON="${AGENTPRE_PYTHON}"
export UV_PYTHON_DOWNLOADS=never
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=void
export HIP_VISIBLE_DEVICES=""
export ROCR_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export TMPDIR="${AGENTPRE_CACHE_ROOT}/tmp"

"${ARTICRAFT_UV}" run \
  --frozen \
  --offline \
  --no-sync \
  --directory "${ARTICRAFT_ROOT}" \
  articraft compile \
  --repo-root "${ARTICRAFT_ROOT}" \
  --data-dir "${ARTICRAFT_DATA_ROOT}" \
  --target full \
  --validate \
  --strict-geom-qc \
  "${RECORD_ID}"

cd "${ROOT}"
"${AGENTPRE_PYTHON}" scripts/apply_articraft_inertials.py \
  --urdf "${MATERIALIZATION_ROOT}/model.urdf" \
  --spec "${INERTIAL_SPEC}" \
  --sidecar "${INERTIAL_SIDECAR}"

"${AGENTPRE_PYTHON}" -m src.asset_inspector \
  "${MATERIALIZATION_ROOT}/model.urdf" \
  --door-joint door_hinge \
  --door-link door \
  --handle-link door

"${AGENTPRE_PYTHON}" scripts/materialize_articraft_asset.py \
  --source-root "${MATERIALIZATION_ROOT}" \
  --cache-root "${AGENTPRE_CACHE_ROOT}" \
  --articraft-commit "${ARTICRAFT_COMMIT}" \
  --data-commit "${DATA_COMMIT}" \
  --inertial-spec "${INERTIAL_SPEC}" \
  --inertial-sidecar "${INERTIAL_SIDECAR}"
