#!/usr/bin/env bash
#SBATCH --account=naiss2026-4-1495-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=trajplan-experiments
#SBATCH --output=logs/experiments-%j.out
#
# Run one shard of ExperimentsRunner.ExpRunner (scheduler + NMPC simulation over a slice of
# instances) via hpc/arrhenius/run_experiments.py. Submit from the project root, normally
# through hpc/arrhenius/submit_experiments.sh rather than by hand:
#
#   sbatch hpc/arrhenius/run_experiments.sbatch SCHEDULERS MAPS SCENARIOS N_AGENTS SEEDS \
#       [METHOD] [CONNECTEDNESS] [AGENT_RADIUS] [CONFLICT_TIME_MARGIN]
#
#   SCHEDULERS  comma-separated, e.g. aoccbs
#   MAPS        comma-separated MovingAI map names, e.g. maze-32-32-2
#   SCENARIOS   comma-separated scenario numbers, e.g. 1
#   N_AGENTS    comma-separated agent counts, e.g. 21,22,23
#   SEEDS       comma-separated seeds, e.g. 5,6,7,8,9
#   METHOD      "grid" or "sampled", default grid
#   CONNECTEDNESS  roadmap connectedness for method=grid, default 4
#   AGENT_RADIUS   metres; omit/blank keeps the backend's own 0.35 m default
#   CONFLICT_TIME_MARGIN  seconds; omit/blank keeps the backend's own 0.0 s default
#
# Every shard writes to the same data/results/experiments_results.csv -- ExpRunner already
# treats it as append-only/resumable (see ExperimentsRunner._failed_instances's docstring), and
# each row write is one small csv.DictWriter.writerow() call, so concurrent shards appending
# is the same pattern local reruns already rely on. The one thing that is NOT safe concurrently
# is a *header migration* (ExperimentsRunner._migrate_results_header rewrites the whole file if
# its header is stale) -- harmless in the steady state (a fresh or current-format file never
# triggers it), but do not start two shards at the very first moment results_csv_path is an
# old-format file.
#
# cpus-per-task=8 and time=12:00:00 are starting points, not measurements -- unlike
# solve.sbatch's figures (a different pipeline, timed on this cluster), nobody has yet timed
# ExperimentsRunner's scheduler+NMPC pipeline on Arrhenius. AOC-CBS's own solve pool is capped
# at min(6, cpus) regardless (DEFAULT_SEARCH_PORTFOLIO has 6 policies -- see
# pkg_sche/aoccbs/runner.py), so cores beyond ~6 only help AOC-CBS's preprocessing (state graph
# distances / intersection intervals, capped by AOCCBS_PP_WORKERS below) go faster on a radius
# the cache hasn't seen yet. Time a single shard's first few instances, then retune both.
#
# Preprocessing note: a new agent_radius means a new AOC-CBS agent-model id and so a cold
# intersection_intervals cache for it (see CLAUDE.md). If a submit_experiments.sh sweep drives
# several radii, the first shard to touch a given radius pays that cost; later shards at the
# same radius reuse the cache AOCCBS_CACHE_DIR points at (set in env.sh), which is one shared
# location for every job, so concurrent shards at the *same new* radius would duplicate that
# one-time build rather than race-corrupt it (AOC-CBS writes the finished file, not a partial
# one, under its own cache key).

set -euo pipefail

SCHEDULERS=${1:?usage: sbatch run_experiments.sbatch SCHEDULERS MAPS SCENARIOS N_AGENTS SEEDS [METHOD] [CONNECTEDNESS] [AGENT_RADIUS] [CONFLICT_TIME_MARGIN]}
MAPS=${2:?missing MAPS}
SCENARIOS=${3:?missing SCENARIOS}
N_AGENTS=${4:?missing N_AGENTS}
SEEDS=${5:?missing SEEDS}
METHOD=${6:-grid}
CONNECTEDNESS=${7:-4}
AGENT_RADIUS=${8:-}
CONFLICT_TIME_MARGIN=${9:-}

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export AOCCBS_PP_WORKERS="${SLURM_CPUS_PER_TASK:-1}"
# shellcheck disable=SC1091
source hpc/arrhenius/env.sh

echo "schedulers=$SCHEDULERS maps=$MAPS scenarios=$SCENARIOS n_agents=$N_AGENTS seeds=$SEEDS"
echo "method=$METHOD connectedness=$CONNECTEDNESS agent_radius=${AGENT_RADIUS:-<default>} conflict_time_margin=${CONFLICT_TIME_MARGIN:-<default>}"
echo "cores=${SLURM_CPUS_PER_TASK:-1} AOCCBS_PP_WORKERS=$AOCCBS_PP_WORKERS"

# ExperimentsRunner's default general_funct call uses mpc_backend="panoc" (see
# ExperimentsRunner.py), which needs the Rust-compiled solver from build_solver.py. Fail here,
# before spending any scheduler time, rather than once per instance deep in run_mpc.
SOLVER_DIR="mpc_solver/navi_default"  # config/mpc_default.yaml: build_directory / optimizer_name
if [ ! -d "$SOLVER_DIR" ]; then
    echo "no compiled PANOC solver at '$SOLVER_DIR'. Run hpc/arrhenius/build_panoc.sh once first." >&2
    exit 1
fi

EXTRA=()
[ -n "$AGENT_RADIUS" ] && EXTRA+=(--agent-radius "$AGENT_RADIUS")
[ -n "$CONFLICT_TIME_MARGIN" ] && EXTRA+=(--conflict-time-margin "$CONFLICT_TIME_MARGIN")

python hpc/arrhenius/run_experiments.py \
    --schedulers "$SCHEDULERS" \
    --maps "$MAPS" \
    --scenarios "$SCENARIOS" \
    --n-agents "$N_AGENTS" \
    --seeds "$SEEDS" \
    --method "$METHOD" \
    --connectedness "$CONNECTEDNESS" \
    ${EXTRA[@]+"${EXTRA[@]}"}
