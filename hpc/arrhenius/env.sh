# Put the project's Python environment on the path. Source it, do not run it:
#
#   source hpc/arrhenius/env.sh
#
# Every interactive session and every job script needs this. It loads the
# module that provides numpy and scipy, then activates the venv that
# setup_env.sh built on top of it.

PYTHON_MODULE=${PYTHON_MODULE:-Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb}

_env_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

module load "$PYTHON_MODULE"
# shellcheck disable=SC1091
source "$_env_root/.venv/bin/activate"

# AOC-CBS writes its preprocessing products and run directories under these
# roots. Pointing them into the project directory means every job shares one
# copy of the preprocessing, and nothing large lands in the 30 GiB home.
export AOCCBS_CACHE_DIR="$_env_root/external/AOC-CBS/cache"
export AOCCBS_DATA_DIR="$_env_root/external/AOC-CBS/data"
export AOCCBS_SCRATCH_DIR="$_env_root/external/AOC-CBS/scratch"

# Python block-buffers stdout when it is a file, which a job's output always is.
# Without this a job's log can sit an hour behind the run, which looks exactly
# like a job stuck on its first problem.
export PYTHONUNBUFFERED=1

# AOC-CBS sizes its preprocessing pools from os.cpu_count() when this is unset,
# and that reports every logical core on the machine -- 256 on a compute node,
# whatever Slurm actually allocated, and the same on the login node. The job
# scripts set this to the job's core count before sourcing this file; the
# default below is only to keep an interactive session on the login node from
# starting 256 processes.
export AOCCBS_PP_WORKERS="${AOCCBS_PP_WORKERS:-4}"

unset _env_root
