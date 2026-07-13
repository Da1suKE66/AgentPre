#!/usr/bin/env bash
set -euo pipefail

AGENTPRE_ROOT="${AGENTPRE_ROOT:-/workspace/liluchen/AgentPre}"
AGENTPRE_CACHE_ROOT="${AGENTPRE_CACHE_ROOT:-/cache/liluchen/agentpre}"
AGENTPRE_ENV="${AGENTPRE_ENV:-${AGENTPRE_CACHE_ROOT}/envs/agentpre-conda}"

export AGENTPRE_ROOT AGENTPRE_CACHE_ROOT AGENTPRE_ENV
export PIP_CACHE_DIR="${AGENTPRE_CACHE_ROOT}/pip-cache"
export XDG_CACHE_HOME="${AGENTPRE_CACHE_ROOT}/xdg-cache"
export WARP_CACHE_PATH="${AGENTPRE_CACHE_ROOT}/warp-cache"
export NEWTON_CACHE_PATH="${AGENTPRE_CACHE_ROOT}/newton-cache"
export TMPDIR="${AGENTPRE_CACHE_ROOT}/tmp"

# The host GPU is occupied and its driver is older than Newton 1.3's minimum.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
unset PYTHONHOME

if [[ ! -x "${AGENTPRE_ENV}/bin/python" ]]; then
  echo "AgentPre environment is missing: ${AGENTPRE_ENV}" >&2
  echo "Run: bash scripts/setup_env.sh" >&2
  return 2 2>/dev/null || exit 2
fi

export PATH="${AGENTPRE_ENV}/bin:${PATH}"
export PYTHONPATH="${AGENTPRE_ROOT}"
