#!/usr/bin/env bash
set -euo pipefail

ROOT="${AGENTPRE_ROOT:-/workspace/liluchen/AgentPre}"
CACHE_ROOT="${AGENTPRE_CACHE_ROOT:-/cache/liluchen/agentpre}"
ENV_PATH="${AGENTPRE_ENV:-${CACHE_ROOT}/envs/agentpre-conda}"
CONDA_BIN="${CONDA_BIN:-/home/ma-user/miniconda3/bin/conda}"

mkdir -p \
  "${CACHE_ROOT}/envs" \
  "${CACHE_ROOT}/assets" \
  "${CACHE_ROOT}/outputs" \
  "${CACHE_ROOT}/logs" \
  "${CACHE_ROOT}/pip-cache" \
  "${CACHE_ROOT}/conda-pkgs" \
  "${CACHE_ROOT}/xdg-cache" \
  "${CACHE_ROOT}/warp-cache" \
  "${CACHE_ROOT}/newton-cache" \
  "${CACHE_ROOT}/tmp"

export CONDA_PKGS_DIRS="${CACHE_ROOT}/conda-pkgs"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip-cache"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg-cache"
export TMPDIR="${CACHE_ROOT}/tmp"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
if [[ -n "${AGENTPRE_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${AGENTPRE_CUDA_VISIBLE_DEVICES}"
elif [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  unset CUDA_VISIBLE_DEVICES
fi
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
unset PYTHONHOME PYTHONPATH

if [[ ! -x "${ENV_PATH}/bin/python" ]]; then
  "${CONDA_BIN}" create -y -p "${ENV_PATH}" python=3.11 pip
fi

"${ENV_PATH}/bin/python" -m pip install --only-binary=:all: -r "${ROOT}/requirements.lock"
AGENTPRE_ROOT="${ROOT}" AGENTPRE_CACHE_ROOT="${CACHE_ROOT}" \
  "${ENV_PATH}/bin/python" "${ROOT}/scripts/fetch_assets.py"

echo "Environment ready: ${ENV_PATH}"
echo "Cache root: ${CACHE_ROOT}"
