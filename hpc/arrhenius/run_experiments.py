#!/usr/bin/env python
"""CLI shard runner for ExperimentsRunner.ExpRunner, for parallel Slurm submission.

ExperimentsRunner.py's own __main__ block is a single hardcoded sweep (fixed schedulers,
maps, scenarios, n_agents, seeds, methods) with a sequential agent_radius retry chain baked
in -- meant to be run as one process, once. This script does not modify or replace that; it
imports the same ExpRunner(...) function ExperimentsRunner.py's __main__ calls and exposes its
arguments on the command line, so hpc/arrhenius/submit_experiments.sh can launch several Slurm
jobs, each covering one slice (e.g. one seed, or one agent-count) of a larger sweep, all
writing to the same data/results/experiments_results.csv -- exactly the file ExpRunner already
treats as append-only/resumable (see ExperimentsRunner._failed_instances's docstring).

Only ExpRunner's non-retry, single-value knobs are exposed (agent_radius, conflict_time_margin
are each one run, not ExperimentsRunner.py's own retry-across-radii loop -- reproduce that by
submitting one job per radius, in order, or by scripting repeat calls to this file).

Usage (from the project root, with hpc/arrhenius/env.sh sourced):

    python hpc/arrhenius/run_experiments.py \\
        --schedulers aoccbs --maps maze-32-32-2 --scenarios 1 \\
        --n-agents 21,22,23 --seeds 5,6,7,8,9 --method grid --connectedness 4 \\
        --agent-radius 0.4375 --conflict-time-margin 0
"""
import argparse
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ExperimentsRunner import ExpRunner  # noqa: E402


def _csv(cast):
    return lambda raw: [cast(v) for v in raw.split(",") if v != ""]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schedulers", type=_csv(str), required=True,
                         help="comma-separated, e.g. aoccbs or ComSat,aoccbs")
    parser.add_argument("--maps", type=_csv(str), required=True)
    parser.add_argument("--scenarios", type=_csv(str), required=True)
    parser.add_argument("--n-agents", type=_csv(int), required=True)
    parser.add_argument("--seeds", type=_csv(int), required=True)
    parser.add_argument("--method", choices=["grid", "sampled"], default="grid")
    parser.add_argument("--connectedness", type=int, default=4,
                         help="roadmap connectedness for method=grid; ignored for sampled")
    parser.add_argument("--agent-radius", type=float, default=None,
                         help="metres; omit to keep the aoccbs/pp_sipp backends' own 0.35 m default")
    parser.add_argument("--conflict-time-margin", type=float, default=None,
                         help="seconds; omit to keep the backends' own 0.0 s default")
    args = parser.parse_args()

    print(f"schedulers={args.schedulers} maps={args.maps} scenarios={args.scenarios} "
          f"n_agents={args.n_agents} seeds={args.seeds} method={args.method} "
          f"connectedness={args.connectedness} agent_radius={args.agent_radius} "
          f"conflict_time_margin={args.conflict_time_margin}", flush=True)

    results_csv_path = ExpRunner(
        args.schedulers, args.maps, args.scenarios, args.n_agents, args.seeds,
        method=args.method, connectedness=args.connectedness,
        agent_radius=args.agent_radius, conflict_time_margin=args.conflict_time_margin,
    )
    print(f"results: {results_csv_path}", flush=True)


if __name__ == "__main__":
    main()
