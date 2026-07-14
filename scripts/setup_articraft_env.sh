#!/usr/bin/env bash
set -euo pipefail

# Install the pinned Articraft harness environment under /cache/liluchen.
# The two external Git checkouts must already exist at the exact commits below.

AGENTPRE_CACHE_ROOT="${AGENTPRE_CACHE_ROOT:-/cache/liluchen/agentpre}"
ARTICRAFT_ROOT="${ARTICRAFT_ROOT:-/cache/liluchen/articraft}"
ARTICRAFT_DATA_ROOT="${ARTICRAFT_DATA_ROOT:-/cache/liluchen/articraft-data}"
ARTICRAFT_ENV="${ARTICRAFT_ENV:-/cache/liluchen/articraft-env}"
ARTICRAFT_UV_BOOTSTRAP="${ARTICRAFT_UV_BOOTSTRAP:-/cache/liluchen/articraft-uv-bootstrap}"
ARTICRAFT_UV_CACHE="${ARTICRAFT_UV_CACHE:-/cache/liluchen/articraft-uv-cache}"
AGENTPRE_PYTHON="${AGENTPRE_PYTHON:-${AGENTPRE_CACHE_ROOT}/envs/agentpre-conda/bin/python}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/cache/liluchen/pip-cache}"

ARTICRAFT_COMMIT="59eb5e0ed72a734111012b43f881423b15d4931d"
DATA_COMMIT="0cdcaa49f5571e9b4df04476c7f09587ee3ab7bd"
UV_VERSION="0.9.17"

fail() {
  echo "$*" >&2
  exit 2
}

[[ -x "${AGENTPRE_PYTHON}" ]] || fail "AgentPre Python is missing: ${AGENTPRE_PYTHON}"
[[ -d "${ARTICRAFT_ROOT}/.git" ]] || fail "Articraft checkout is missing: ${ARTICRAFT_ROOT}"
[[ -d "${ARTICRAFT_DATA_ROOT}/.git" ]] \
  || fail "Articraft data checkout is missing: ${ARTICRAFT_DATA_ROOT}"
[[ "$(git -C "${ARTICRAFT_ROOT}" rev-parse HEAD)" == "${ARTICRAFT_COMMIT}" ]] \
  || fail "Articraft checkout is not at ${ARTICRAFT_COMMIT}."
[[ "$(git -C "${ARTICRAFT_DATA_ROOT}" rev-parse HEAD)" == "${DATA_COMMIT}" ]] \
  || fail "Articraft data checkout is not at ${DATA_COMMIT}."

mkdir -p "${ARTICRAFT_UV_CACHE}" "${PIP_CACHE_DIR}"
export PIP_CACHE_DIR
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=void
export HIP_VISIBLE_DEVICES=""
export ROCR_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export UV_CACHE_DIR="${ARTICRAFT_UV_CACHE}"
export UV_PROJECT_ENVIRONMENT="${ARTICRAFT_ENV}"
export UV_PYTHON="${AGENTPRE_PYTHON}"
export UV_PYTHON_DOWNLOADS=never
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-1}"
export UV_CONCURRENT_BUILDS="${UV_CONCURRENT_BUILDS:-1}"
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-1}"

if [[ ! -x "${ARTICRAFT_UV_BOOTSTRAP}/bin/uv" ]]; then
  "${AGENTPRE_PYTHON}" -m venv "${ARTICRAFT_UV_BOOTSTRAP}"
  "${ARTICRAFT_UV_BOOTSTRAP}/bin/python" -m pip install "uv==${UV_VERSION}"
fi
actual_uv_version="$("${ARTICRAFT_UV_BOOTSTRAP}/bin/uv" --version)"
[[ "${actual_uv_version}" == "uv ${UV_VERSION}" ]] \
  || fail "Unexpected uv version: ${actual_uv_version}"

"${ARTICRAFT_UV_BOOTSTRAP}/bin/uv" sync \
  --frozen \
  --no-dev \
  --directory "${ARTICRAFT_ROOT}" \
  --python "${AGENTPRE_PYTHON}"

"${ARTICRAFT_ENV}/bin/python" -c \
  'import cadquery, manifold3d, trimesh; print("Articraft compile imports: OK")'
"${ARTICRAFT_ENV}/bin/articraft" --help >/dev/null
echo "Articraft environment ready: ${ARTICRAFT_ENV}"
