# Running this project on a cluster

Arrhenius (at NSC in Linköping) is the only cluster set up so far, and
everything below is about it. Its scripts live in `arrhenius/`.

This runs `ExperimentsRunner.py`'s `ExpRunner(...)` -- the scheduler (default
`aoccbs`) plus the headless NMPC simulation (`mpc_backend="panoc"`), swept
over maps/scenarios/agent counts/seeds -- not a separate benchmarking
pipeline. Sharding a sweep into several parallel Slurm jobs is
`submit_experiments.sh`'s job; see "Submitting jobs" below.

## The connection

Arrhenius asks for a password and a one-time verification code on every SSH
connection, so `arrhenius/sync.sh` uses OpenSSH connection multiplexing: one
authenticated connection is held open and every later transfer reuses it. The
`arrhenius` entry in `~/.ssh/config` on this machine is what holds it, with
`ControlMaster auto` and `ControlPersist 8h`. That file is outside the
repository; recreate it on a new machine before anything below works.

A PyCharm deployment cannot use that connection -- its SFTP client is not
OpenSSH -- so it would ask for a verification code several times an hour.
File transfer on this project is a terminal operation.

## A working session

    ./hpc/arrhenius/sync.sh connect    # password + code, once, good for 8 hours
    ./hpc/arrhenius/sync.sh push       # this machine -> Arrhenius
    ./hpc/arrhenius/sync.sh pull       # Arrhenius -> this machine, data/results/ only

`connect` is optional. Every other command opens the connection if it is not
already up, so skipping it just moves the prompt to the first transfer.

    ./hpc/arrhenius/sync.sh status     # is the connection still alive?
    ./hpc/arrhenius/sync.sh stop       # close it
    ./hpc/arrhenius/sync.sh shell      # log in, already in the project directory
    ./hpc/arrhenius/sync.sh submodules # put external/ at the commits checked out here

The connection dies on sleep or a network change; the next command reopens it.

## A campaign, end to end

1. Edit here. Never on the cluster -- see the rules below.
2. `sync.sh push`, then `sync.sh submodules` (needed after any commit that
   moves `external/AOC-CBS`, and on a first push).
3. On the cluster, once per clone:

       bash -l hpc/arrhenius/setup_env.sh   # Python venv, AOC-CBS editable install, the two patches
       hpc/arrhenius/build_panoc.sh         # Rust/PANOC solver -- see its own header for the caveats

   Neither has been run on Arrhenius yet as of this writing, so treat the
   module names in both scripts (the Python bundle in `setup_env.sh`, the
   guessed Rust module in `build_panoc.sh`) as a starting point, not a
   verified fact -- check `module spider` if either `module load` fails.
4. On the cluster: submit a sweep with `submit_experiments.sh` (see
   "Submitting jobs" below).
5. `sync.sh pull` here, once the jobs are done or you want a look at partial
   results -- `data/results/experiments_results.csv` is append-only, so a
   pull mid-sweep is safe, just incomplete.

Delete/rerun the way `ExperimentsRunner._failed_instances` already expects:
a combo's most recent row in `experiments_results.csv` is what counts, so
rerunning a shard is always safe, never something that needs manual cleanup
first.

## The Python environment

Built once per clone, on the login node:

    bash -l hpc/arrhenius/setup_env.sh

It loads `Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb`, which
supplies numpy, scipy and sortedcontainers from the site's own build, makes a
venv at `.venv` with `--system-site-packages` on top of it, installs the rest
of `requirements.txt` from PyPI, installs `external/AOC-CBS` editable, and
applies its two local patches (`src/pkg_sche/aoccbs/aoccbs_node_link_data.patch`
and `aoccbs_conflict_time_margin.patch` -- see CLAUDE.md's "Third scheduler
backend" for what each fixes). TrajPlan-ScheMPC itself has no
pyproject.toml/setup.py; it runs as `python src/<script>.py` from the project
root, which is what puts `src/` on `sys.path` (see CLAUDE.md's "Path
conventions") -- there is nothing of this project's own to `pip install -e`.

Then in every later shell, and in every job script:

    source hpc/arrhenius/env.sh

That loads the same module, activates the venv, and points `AOCCBS_CACHE_DIR`,
`AOCCBS_DATA_DIR` and `AOCCBS_SCRATCH_DIR` into `external/AOC-CBS/` so every
job shares one copy of the preprocessing and nothing large lands in the 30 GiB
home. It also sets `AOCCBS_PP_WORKERS` (default 4 interactively; job scripts
override it from their own core count before sourcing this file) -- AOC-CBS's
preprocessing pools default to `os.cpu_count()` otherwise, which is every
logical core on the *node*, not what Slurm actually allocated to the job.

To check the environment:

    source hpc/arrhenius/env.sh && python -c "from aoccbs.solver import aoccbs"

The venv lives inside the project directory but is excluded from the sync, so
the two machines keep their own and a push never overwrites either.

## The PANOC/OpEn solver

Also once per clone, on the login node, after `setup_env.sh`:

    hpc/arrhenius/build_panoc.sh

`ExperimentsRunner.py`'s own `general_funct` call uses `mpc_backend="panoc"`,
which needs the Rust crate `src/build_solver.py` compiles via `opengen`/cargo
-- see CLAUDE.md's "Three NMPC backends". `build_panoc.sh`'s header explains
the module-name uncertainty and the rustup fallback if no module provides
Rust at all. The build output lands in the gitignored `mpc_solver/`, excluded
from the sync like the venv, so each machine keeps its own.

## Submitting jobs

Submit from the project root, so `logs/` and the relative paths resolve.
Normally through the submit wrapper, which fans a sweep out across several
parallel jobs:

    hpc/arrhenius/submit_experiments.sh [--shard-by n-agents|seeds] \
        SCHEDULERS MAPS SCENARIOS N_AGENTS SEEDS [METHOD] [CONNECTEDNESS] \
        [AGENT_RADIUS] [CONFLICT_TIME_MARGIN]

One job per value of the `--shard-by` axis (default `n-agents`); every other
argument is passed to every job unchanged. Every job writes to the same
`data/results/experiments_results.csv` -- see `run_experiments.sbatch`'s
header for why concurrent appends there are fine but a header migration is
not (irrelevant in the steady state: it only fires against a stale-format
file). Example, one job per agent count:

    hpc/arrhenius/submit_experiments.sh --shard-by n-agents \
        aoccbs maze-32-32-2 1 21,22,23,24,25 5,6,7,8,9 grid 4

A single shard can also be submitted directly with `sbatch
hpc/arrhenius/run_experiments.sbatch ...` (same arguments) -- what
`submit_experiments.sh` does in a loop.

`ExperimentsRunner.py`'s own `__main__` block additionally retries failed
instances across a sequence of `agent_radius` values, each retry depending on
the previous radius's results -- that chain is not something a shard can
parallelise (see `run_experiments.py`'s docstring); reproduce it on the
cluster, if wanted, by submitting one sweep per radius in order, each after
the previous finishes.

`--cpus-per-task`/`--time` in `run_experiments.sbatch` are unmeasured
starting points, not the kind of timed figures the (removed) AOC-CBS-sweep
scripts this file used to describe had -- this pipeline has not been timed on
Arrhenius yet. Time a small shard first and retune both; see the sbatch
file's header for what bounds the core count usefully (AOC-CBS's own solve
pool is capped at 6 regardless -- `DEFAULT_SEARCH_PORTFOLIO`'s size -- so
more cores only help preprocessing on a radius the cache hasn't seen yet).

## What syncs

Everything in the repository except `external/`, `.venv/`, `mpc_solver/`,
`.idea/` and the usual caches. The exact list is `arrhenius/rsync-exclude.txt`.

`.git` is synced, including `.git/modules/`, the two submodules' object stores.
That is what lets `sync.sh submodules` check `external/` out on the cluster
with no network and no GitHub credentials, which matters because AOC-CBS is a
private repository that the login node cannot clone.

`external/` itself is never synced. Keeping it out means a push cannot touch
AOC-CBS's own preprocessing cache/scratch directories, which exist only on
the cluster (pointed there by `AOCCBS_CACHE_DIR`/`AOCCBS_SCRATCH_DIR` in
`env.sh`).

`sync.sh submodules` mirrors the commit each submodule has checked out
**here**, not the commit the superproject records -- see CLAUDE.md's "Third
scheduler backend" for a concrete case where those differed.

## Rules

- **Edit on this machine, not on the cluster.** A push overwrites the cluster's
  copy of every synced file. Job scripts under `hpc/arrhenius/` are the easy
  mistake: write them here and push them.
- **Neither direction deletes.** `data/results/` accumulates records on both
  machines and a mirror would destroy work. `push --delete` is available and
  is safe for the AOC-CBS run/cache directories, since those sit under the
  excluded `external/`.
- **Never run git on the solvers on the cluster.** `external/AOC-CBS` there
  is a mirror of this machine, not a place to pull, check out or commit.
  Update the solver here, then `sync.sh push` and `sync.sh submodules`, in
  that order: the push carries the objects, the second command forces the
  cluster's checkout onto the commit you have here. It overwrites tracked
  files under `external/` and leaves untracked ones, so AOC-CBS's own
  cache/scratch directories survive it -- re-apply the two patches
  (`setup_env.sh` does this automatically) after any such re-checkout.
- **Only commits travel** through `sync.sh submodules` -- `push`/`pull`
  themselves are plain rsync of the working tree and carry uncommitted
  changes too, `external/` (excluded) aside. `sync.sh submodules` prints a
  warning when the local submodule checkout itself has uncommitted changes,
  since those never reach the cluster either way.
- **Do not commit on both machines.** `.git` travels whole and overwrites; it
  does not merge.

## Settings

The host alias and the remote directory are the defaults in `sync.sh` and can
be overridden per invocation:

    ARRHENIUS_HOST=arrhenius3 ./hpc/arrhenius/sync.sh push
    ARRHENIUS_DIR=/nobackup/proj/disk/<project>/personal/<user>/TrajPlan-ScheMPC \
        ./hpc/arrhenius/sync.sh push

The project lives under the allocation's shared project storage, which
`sync.sh` names as its default remote. That storage is at least 250 GiB and
**is not backed up**.
