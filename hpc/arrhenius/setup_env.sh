#!/usr/bin/env bash
# Build the Python environment for TrajPlan-ScheMPC on Arrhenius. Run once per
# clone, on the login node:
#
#   bash -l hpc/arrhenius/setup_env.sh
#
# The venv is created with --system-site-packages on top of the SciPy bundle
# module, so numpy, scipy and sortedcontainers come from the site's optimised
# build and only the rest is installed from PyPI. It lives at .venv in the
# project directory, which the sync excludes, so it is never overwritten from
# the laptop.
#
# This does not build the PANOC/OpEn solver (needs a Rust toolchain -- see
# build_panoc.sh) or apply AOC-CBS's two local patches (see below). Both are
# required before ExperimentsRunner.py's default configuration (mpc_backend
# "panoc", scheduler_backend "aoccbs") will run; this script only gets the
# Python side importable.

set -euo pipefail

PYTHON_MODULE=${PYTHON_MODULE:-Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb}
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

if ! command -v module >/dev/null 2>&1; then
  echo "The module command is not available. Run this from a login shell:" >&2
  echo "  bash -l hpc/arrhenius/setup_env.sh" >&2
  exit 1
fi

module load "$PYTHON_MODULE"
echo "Python: $(python3 -V), $(command -v python3)"

if [ -d "$ROOT/.venv" ]; then
  echo "Reusing the existing venv at $ROOT/.venv."
else
  python3 -m venv --system-site-packages "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

python -m pip install --upgrade --quiet pip
python -m pip install --quiet -r "$ROOT/requirements.txt"

# TrajPlan-ScheMPC has no pyproject.toml/setup.py of its own -- it is run as
# "python src/<script>.py" from the project root, which puts src/ on
# sys.path (see CLAUDE.md's "Path conventions"). Only AOC-CBS is an editable
# install.
python -m pip install --quiet -e "$ROOT/external/AOC-CBS"
python -m pip install --quiet sortedcontainers  # aoccbs's pyproject.toml omits it

# Two local patches to the vendored AOC-CBS checkout, required every time
# external/ is re-cloned or re-checked-out (e.g. after "sync.sh submodules"):
# one fixes a genuine networkx 3.4.2 incompatibility, the other adds the
# conflict_time_margin knob. Both are idempotent to re-apply-when-already-applied
# only in the sense that git apply fails loudly (not silently) if so -- this
# script does not try to re-apply an already-applied patch.
apply_patch() {
  local patch=$1
  if git -C "$ROOT/external/AOC-CBS" apply --check "$ROOT/$patch" 2>/dev/null; then
    git -C "$ROOT/external/AOC-CBS" apply "$ROOT/$patch"
    echo "applied $patch"
  elif git -C "$ROOT/external/AOC-CBS" apply --reverse --check "$ROOT/$patch" 2>/dev/null; then
    echo "$patch already applied"
  else
    echo "$patch does not apply cleanly (and is not already applied) -- check by hand" >&2
    exit 1
  fi
}
apply_patch src/pkg_sche/aoccbs/aoccbs_node_link_data.patch
apply_patch src/pkg_sche/aoccbs/aoccbs_conflict_time_margin.patch

python - <<'PY'
import importlib
for name in ("numpy", "scipy", "networkx", "matplotlib", "casadi", "opengen",
             "z3", "sortedcontainers", "yaml", "aoccbs"):
    module = importlib.import_module(name)
    print(f"  {name:16s} {getattr(module, '__version__', '-'):10s} {module.__file__}")
PY

echo
echo "Python side ready. Still needed once per clone:"
echo "  hpc/arrhenius/build_panoc.sh   (compiles the Rust/PANOC solver)"
echo "In every later shell and every job script:"
echo "  source hpc/arrhenius/env.sh"
