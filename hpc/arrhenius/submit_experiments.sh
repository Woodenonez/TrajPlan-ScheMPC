#!/usr/bin/env bash
# Fan a sweep out into several run_experiments.sbatch jobs, one per value of a chosen axis, so
# they run in parallel on Arrhenius instead of one process working through the whole sweep
# serially. Every job writes to the same data/results/experiments_results.csv -- see
# run_experiments.sbatch's header for why that is safe in the steady state.
#
#   hpc/arrhenius/submit_experiments.sh --shard-by n-agents|seeds \
#       SCHEDULERS MAPS SCENARIOS N_AGENTS SEEDS [METHOD] [CONNECTEDNESS] \
#       [AGENT_RADIUS] [CONFLICT_TIME_MARGIN]
#
# All arguments after --shard-by are exactly run_experiments.sbatch's own positional arguments
# (see its header) -- comma-separated lists. One job is submitted per value in whichever list
# --shard-by names (default n-agents); every other argument is passed to every job unchanged.
#
# Example: one job per agent count, 21..25, all five seeds in each job:
#
#   hpc/arrhenius/submit_experiments.sh --shard-by n-agents \
#       aoccbs maze-32-32-2 1 21,22,23,24,25 5,6,7,8,9 grid 4
#
# This does not reproduce ExperimentsRunner.py's own __main__ retry-across-agent_radius chain
# (each radius's retries depend on the previous radius's results, so that step is inherently
# sequential) -- submit one sweep per radius, in order, waiting for each to finish before
# submitting the next, if that behaviour is wanted on the cluster.

set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)

SHARD_BY=n-agents
if [ "${1:-}" = "--shard-by" ]; then
  SHARD_BY=$2
  shift 2
fi
case "$SHARD_BY" in
  n-agents|seeds) ;;
  *) echo "--shard-by must be n-agents or seeds, got '$SHARD_BY'" >&2; exit 1 ;;
esac

SCHEDULERS=${1:?usage: submit_experiments.sh [--shard-by n-agents|seeds] SCHEDULERS MAPS SCENARIOS N_AGENTS SEEDS [METHOD] [CONNECTEDNESS] [AGENT_RADIUS] [CONFLICT_TIME_MARGIN]}
MAPS=${2:?missing MAPS}
SCENARIOS=${3:?missing SCENARIOS}
N_AGENTS=${4:?missing N_AGENTS}
SEEDS=${5:?missing SEEDS}
METHOD=${6:-grid}
CONNECTEDNESS=${7:-4}
AGENT_RADIUS=${8:-}
CONFLICT_TIME_MARGIN=${9:-}

mkdir -p "$ROOT/logs"

if [ "$SHARD_BY" = "n-agents" ]; then
  SHARD_VALUES=$N_AGENTS
else
  SHARD_VALUES=$SEEDS
fi

njobs=0
for value in ${SHARD_VALUES//,/ }; do
  if [ "$SHARD_BY" = "n-agents" ]; then
    job=$(sbatch --parsable --job-name="trajplan-exp-n$value" \
          "$HERE/run_experiments.sbatch" \
          "$SCHEDULERS" "$MAPS" "$SCENARIOS" "$value" "$SEEDS" \
          "$METHOD" "$CONNECTEDNESS" "$AGENT_RADIUS" "$CONFLICT_TIME_MARGIN")
    echo "n_agents=$value: job $job"
  else
    job=$(sbatch --parsable --job-name="trajplan-exp-s$value" \
          "$HERE/run_experiments.sbatch" \
          "$SCHEDULERS" "$MAPS" "$SCENARIOS" "$N_AGENTS" "$value" \
          "$METHOD" "$CONNECTEDNESS" "$AGENT_RADIUS" "$CONFLICT_TIME_MARGIN")
    echo "seed=$value: job $job"
  fi
  njobs=$((njobs + 1))
done

echo
echo "$njobs job(s) queued, sharded by $SHARD_BY."
echo "Results accumulate in data/results/experiments_results.csv as jobs finish;"
echo "pull with: hpc/arrhenius/sync.sh pull"
