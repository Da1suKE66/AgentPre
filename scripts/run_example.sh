#!/usr/bin/env bash
set -euo pipefail

ROOT="${AGENTPRE_ROOT:-/workspace/liluchen/AgentPre}"
source "${ROOT}/scripts/env.sh"
cd "${ROOT}"

python -m src.run \
  --config "${2:-configs/microwave_franka.json}" \
  --mode "${1:-kinematic}"
