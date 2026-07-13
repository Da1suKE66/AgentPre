"""AgentPre deterministic articulated-manipulation pipeline.

The target host has a deliberately small ``/workspace`` filesystem.  Set
cache defaults before importing Newton/Warp so the exact documented
``python -m src.run`` command keeps generated kernels and temporary data on
the large cache volume.  Callers can still override every path explicitly.
"""

import os


_CACHE_ROOT = os.environ.get("AGENTPRE_CACHE_ROOT", "/cache/liluchen/agentpre")
os.environ.setdefault("AGENTPRE_CACHE_ROOT", _CACHE_ROOT)
os.environ["XDG_CACHE_HOME"] = f"{_CACHE_ROOT}/xdg-cache"
os.environ["WARP_CACHE_PATH"] = f"{_CACHE_ROOT}/warp-cache"
os.environ["NEWTON_CACHE_PATH"] = f"{_CACHE_ROOT}/newton-cache"
os.environ["TMPDIR"] = f"{_CACHE_ROOT}/tmp"
os.environ["PIP_CACHE_DIR"] = f"{_CACHE_ROOT}/pip-cache"
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

__version__ = "0.1.0"
