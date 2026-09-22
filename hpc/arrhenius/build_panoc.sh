#!/usr/bin/env bash
# Compile the PANOC/OpEn solver ExperimentsRunner.py's default configuration
# needs (mpc_backend="panoc" -- see CLAUDE.md's "Three NMPC backends"). Run
# once per clone, on the login node, after setup_env.sh:
#
#   hpc/arrhenius/build_panoc.sh
#
# opengen (a pip dependency, already installed by setup_env.sh) compiles a
# generated Rust crate via cargo the first time a solver is built, so a Rust
# toolchain has to be on PATH first. This project has not previously built on
# Arrhenius, so unlike build_env.sh's CBSH2-RTC module pins (measured against
# a real build there), the module name below is a guess -- check what the
# site actually provides before trusting it:
#
#   module spider Rust
#
# and set RUST_MODULE to override if the name differs. If no module ships
# Rust at all, install it per-user instead (this only needs to happen once):
#
#   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
#   source "$HOME/.cargo/env"
#
# then re-run this script with RUST_MODULE=none.
#
# The build itself needs no GPU and little memory; it runs on the login node
# like setup_env.sh, not as a Slurm job.

set -euo pipefail

RUST_MODULE=${RUST_MODULE:-Rust}
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

if [ "$RUST_MODULE" != "none" ]; then
  if ! command -v module >/dev/null 2>&1; then
    echo "The module command is not available. Run this from a login shell." >&2
    exit 1
  fi
  module load "$RUST_MODULE"
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo is not on PATH. Either set RUST_MODULE to the site's actual Rust" >&2
  echo "module (see 'module spider Rust'), or install Rust per-user with rustup" >&2
  echo "and source \$HOME/.cargo/env before running this script with RUST_MODULE=none." >&2
  exit 1
fi
echo "cargo: $(cargo --version)"

# shellcheck disable=SC1091
source "$ROOT/hpc/arrhenius/env.sh"

cd "$ROOT"
python src/build_solver.py

echo
echo "Built. mpc_backend=\"panoc\" (ExperimentsRunner.py's default) is now usable."
