import os
import csv
import shutil
import pathlib
import multiprocessing as mp
import concurrent.futures as cf

import pandas as pd  # type: ignore

from main import general_funct
from roadmap_to_testcase import convert_movingai

project_root = pathlib.Path(__file__).resolve().parents[1]
data_path = os.path.join(project_root, "data")
schedule_dir = os.path.join(data_path, "schedule_demo2_data")
results_dir = os.path.join(data_path, "results")
results_csv_path = os.path.join(results_dir, "experiments_results.csv")

# run_mpc reports its NMPC failures as status "late" / "collision" / "stuck" / "timeout";
# spell the first one out to match how the failure is actually configured (late_threshold_s).
MPC_REASON_LABELS = {"late": "late_threshold"}

RESULT_FIELDS = [
    "scheduler", "map", "scenario", "n_agents", "seed", "method",
    "agent_radius", "conflict_time_margin",
    "scheduler_success", "sum_of_costs", "actual_sum_of_cost",
    "n_robots_finished", "mpc_failure_reason",
    "n_nodes_compared", "n_nodes_missing",
    "mean_eta_diff_s", "max_abs_eta_diff_s",
    "mean_schedule_adherence_m",
    "error",
]

# python src/roadmap_to_testcase.py movingai --map den312d --scenario random-1 --n-agents 7 --cell-size 2 --seed 7 --clearance 0.7 --out test_9


def _merged_schedule_df(instance_name):
    """Join the planned schedule_<instance_name>.csv against the realised
    Actual_<instance_name>.csv, both in the robot_id,node_id,ETA format, on
    (robot_id, node_id). Returns None if either file is missing (scheduler failed, so
    schedule_<instance_name>.csv wasn't written, or the controller never ran).

    Both files are named after `instance_name` because `ExpRunner` passes it to
    `general_funct` as `run_tag` (see its call below): several instances' scheduler+controller
    runs can be in flight against this one shared project checkout at once -- e.g. parallel
    Slurm shards on a cluster (hpc/arrhenius/) -- and a fixed "schedule.csv"/"Actual_*.csv"
    name would let two such runs clobber each other's files."""
    schedule_path = os.path.join(schedule_dir, f"schedule_{instance_name}.csv")
    actual_path = os.path.join(schedule_dir, f"Actual_{instance_name}.csv")
    if not (os.path.exists(schedule_path) and os.path.exists(actual_path)):
        return None

    planned = pd.read_csv(schedule_path)
    actual = pd.read_csv(actual_path)
    merged = planned.merge(actual, on=["robot_id", "node_id"], how="left",
                            suffixes=("_planned", "_actual"))
    merged["ETA_diff"] = merged["ETA_actual"] - merged["ETA_planned"]
    return merged


def _schedule_diff_stats(merged):
    """Summary stats (CSV-ready, blank where no comparison is possible) over a
    `_merged_schedule_df` result's per-node ETA deviations (actual - planned)."""
    if merged is None:
        return {"n_nodes_compared": 0, "n_nodes_missing": "",
                "mean_eta_diff_s": "", "max_abs_eta_diff_s": ""}
    compared = merged["ETA_diff"].dropna()
    return {
        "n_nodes_compared": int(compared.shape[0]),
        "n_nodes_missing": int(merged["ETA_diff"].isna().sum()),
        "mean_eta_diff_s": round(float(compared.mean()), 3) if not compared.empty else "",
        "max_abs_eta_diff_s": round(float(compared.abs().max()), 3) if not compared.empty else "",
    }


def _actual_sum_of_cost(merged):
    """The measured counterpart to main.py's compute_sum_of_costs: sum over robots of each
    robot's ACTUAL arrival time at its last scheduled node, rather than the planned ETA.
    `merged` preserves schedule_<instance_name>.csv's per-robot chronological row order (see
    `_merged_schedule_df`), so a robot's last row is its route's goal node.

    Left blank ("") -- rather than a partial sum over only the robots that finished -- unless
    every robot in the run actually reached its own last scheduled node; a robot that never
    arrives (collision/timeout/late-threshold abort) has no real goal-arrival time to sum."""
    if merged is None or merged.empty:
        return ""
    last_per_robot = merged.groupby("robot_id", sort=False).tail(1)
    if last_per_robot["ETA_actual"].isna().any():
        return ""
    return float(last_per_robot["ETA_actual"].sum())


def _migrate_results_header():
    """Rewrite experiments_results.csv in place if it was written with an older RESULT_FIELDS.

    Rows are appended to that file across runs, so a column added to RESULT_FIELDS would
    otherwise be appended past the stored header and silently misalign every new row against
    the old ones. Existing rows keep their values and get an empty cell for each new column."""
    if not os.path.exists(results_csv_path):
        return
    with open(results_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == RESULT_FIELDS:
            return
        old_rows = list(reader)
    with open(results_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for old_row in old_rows:
            writer.writerow({field: old_row.get(field, "") or "" for field in RESULT_FIELDS})


def _write_result_row(row):
    _migrate_results_header()
    file_exists = os.path.exists(results_csv_path)
    with open(results_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _mean_schedule_adherence(src_path):
    """Average positional deviation (metres) between a robot's actual and schedule-expected
    (x, y), over every row -- every robot, every second -- of run_mpc's per-second
    SchedAdherence_<instance_name>.csv (see `ScheduleAdherenceLogger`). Blank if the controller
    never ran or never wrote that file."""
    if not src_path or not os.path.exists(src_path):
        return ""
    df = pd.read_csv(src_path)
    if df.empty:
        return ""
    deviation = ((df["actual_x"] - df["expected_x"])**2 + (df["actual_y"] - df["expected_y"])**2)**0.5
    return round(float(deviation.mean()), 6)


def _copy_sched_adherence_csv(src_path, out_name):
    """Copy run_mpc's per-second SchedAdherence_<instance_name>.csv (per agent, per second of
    simulated time: actual (x, y) vs where the schedule expects the robot to be, assuming
    constant speed across each scheduled ETA gap -- see `ScheduleAdherenceLogger`) into
    data/results under `out_name`, alongside the other per-instance CSVs. Returns None (and
    copies nothing) if the controller never ran or never wrote that file."""
    if not src_path or not os.path.exists(src_path):
        return None
    out_path = os.path.join(results_dir, f"{out_name}.csv")
    shutil.copyfile(src_path, out_path)
    return out_path


def _write_instance_csv(instance_name, merged):
    """Per-instance node log: just the planned-vs-actual ETA breakdown, one line per scheduled
    node -- i.e. schedule_<instance_name>.csv and Actual_<instance_name>.csv joined on
    (robot_id, node_id).
    The run's summary fields (RESULT_FIELDS) live in experiments_results.csv only; this file
    carries no per-run identification of its own."""
    node_fields = ["robot_id", "node_id", "ETA_planned", "ETA_actual", "ETA_diff"]
    out_path = os.path.join(results_dir, f"{instance_name}.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=node_fields)
        writer.writeheader()
        if merged is not None and not merged.empty:
            for _, node_row in merged.iterrows():
                writer.writerow({
                    "robot_id": node_row["robot_id"], "node_id": node_row["node_id"],
                    "ETA_planned": node_row["ETA_planned"], "ETA_actual": node_row["ETA_actual"],
                    "ETA_diff": node_row["ETA_diff"],
                })
    return out_path


def _method_label(method, connectedness):
    """The value stored in experiments_results.csv's "method" column: "grid" is parameterised
    by its roadmap connectedness (e.g. "grid(4)"), since that knob changes the instance graph;
    "sampled" carries no such suffix, since its own knobs (density/clearance) live elsewhere."""
    return f"{method}({connectedness})" if method == "grid" else method


def _failed_instances(prev_value, schedulers, maps, scenarios, n_agents, seeds, method,
                       value_column="conflict_time_margin", fail_reasons=None):
    """(scheduler, map, scenario, n_agent, seed) combos -- drawn from the given lists -- whose
    row in experiments_results.csv at `value_column` == prev_value (same method) counts as
    failed. Used to re-run, at the next value in a sweep, only the instances the previous value
    failed on; a row that succeeded is never retried. Returns [] if experiments_results.csv
    doesn't exist yet, or no prior row matches.

    `method` must be the exact label stored in the "method" column (see `_method_label`), e.g.
    "grid(4)" rather than bare "grid".

    What counts as failed is controlled by `fail_reasons`:
    - None (the ctm sweep's original behaviour): finished fewer robots than n_agents, or has no
      usable n_robots_finished at all (blank, e.g. an exception or a "no_schedule" run).
    - a collection of `mpc_failure_reason` values (e.g. the agent_radius sweep's
      {"late_threshold", "collision"}): only those specific failure modes count, since e.g. a
      scheduler failure or a stuck/timeout abort would not be fixed by the swept parameter."""
    if not os.path.exists(results_csv_path):
        return []

    df = pd.read_csv(results_csv_path, dtype=str, keep_default_na=False)
    prev_value_str = "" if prev_value is None else str(prev_value)
    scenario_strs = {str(s) for s in scenarios}
    n_agent_strs = {str(n) for n in n_agents}
    seed_strs = {str(s) for s in seeds}

    mask = (
        (df["method"] == method)
        & (df[value_column] == prev_value_str)
        & (df["scheduler"].isin(schedulers))
        & (df["map"].isin(maps))
        & (df["scenario"].isin(scenario_strs))
        & (df["n_agents"].isin(n_agent_strs))
        & (df["seed"].isin(seed_strs))
    )
    # experiments_results.csv is append-only, so the same combo can have been (re-)run more
    # than once at this value_column setting across separate sweeps; judge failure on each
    # combo's most recent row only, not on every historical attempt.
    candidates = df[mask].drop_duplicates(
        subset=["scheduler", "map", "scenario", "n_agents", "seed"], keep="last"
    )

    if fail_reasons is not None:
        failed = candidates[candidates["mpc_failure_reason"].isin(fail_reasons)]
    else:
        finished = pd.to_numeric(candidates["n_robots_finished"], errors="coerce")
        n_agents_num = pd.to_numeric(candidates["n_agents"], errors="coerce")
        failed = candidates[finished.isna() | (finished < n_agents_num)]

    return [
        (row["scheduler"], row["map"], row["scenario"], int(row["n_agents"]), int(row["seed"]))
        for _, row in failed.iterrows()
    ]


def _available_cores():
    """The CPU core ids this process may actually be scheduled on -- respects a taskset/cgroup
    restriction on Linux -- or, on a platform with no affinity API at all (notably macOS), every
    logical core by count. Used to pick default core assignments in `_dispatch_jobs`."""
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def _pin_worker(counter, lock, cores):
    """`ProcessPoolExecutor` initializer: claim this worker's index off a shared counter (so
    concurrently-starting workers never claim the same one) and pin the process to
    `cores[index % len(cores)]` for the rest of its life.

    CPU affinity is inherited by everything the process goes on to do -- its own threads
    (numpy/BLAS, Gurobi) and any subprocesses it forks (AOC-CBS's own internal search-pool
    processes) -- so this is what actually keeps one instance's work off a core another worker
    owns, regardless of how many threads/processes that instance's scheduler backend tries to
    spin up internally. Anything that lands on the same pinned core just time-shares it.

    No-op, with a one-time printed warning, on a platform with no `os.sched_setaffinity` --
    there is no such call on macOS at all, so `n_workers` still caps how many instances run at
    once there, just not to specific cores."""
    with lock:
        index = counter.value
        counter.value += 1
    if not hasattr(os, "sched_setaffinity"):
        print(f"[ExperimentsRunner] worker pid={os.getpid()}: CPU pinning unavailable on this "
              f"OS (no os.sched_setaffinity, e.g. macOS) -- limited to n_workers concurrency, "
              f"not pinned to a dedicated core.", flush=True)
        return
    core_id = cores[index % len(cores)]
    os.sched_setaffinity(0, {core_id})
    print(f"[ExperimentsRunner] worker pid={os.getpid()} pinned to core {core_id}", flush=True)


def _run_one_combo(job):
    """Run a single (scheduler, map, scenario, n_agents, seed) combo end to end -- generate the
    instance, run the scheduler+controller, and write this combo's own node-log and
    schedule-adherence CSVs -- and return the experiments_results.csv row for it.

    `job` is a plain dict (picklable, for `_dispatch_jobs`'s process pool) with keys: scheduler,
    map, scenario, n_agent, seed, method, connectedness, method_label, agent_radius,
    conflict_time_margin, save_instance_files.

    Writing the row itself to experiments_results.csv is left to the caller, since that file is
    shared across every combo in a sweep and `_dispatch_jobs` is what serialises those writes
    when combos run in separate processes."""
    scheduler = job["scheduler"]
    map_name = job["map"]
    scenario = job["scenario"]
    n_agent = job["n_agent"]
    seed = job["seed"]
    method = job["method"]
    connectedness = job["connectedness"]
    method_label = job["method_label"]
    agent_radius = job["agent_radius"]
    conflict_time_margin = job["conflict_time_margin"]
    save_instance_files = job.get("save_instance_files", True)

    instance_name = f'{map_name}_scenario-{scenario}_{n_agent}_{seed}'
    row = {
        "scheduler": scheduler, "map": map_name, "scenario": scenario,
        "n_agents": n_agent, "seed": seed, "method": method_label,
        "agent_radius": agent_radius if agent_radius is not None else "",
        "conflict_time_margin": conflict_time_margin if conflict_time_margin is not None else "",
        "scheduler_success": 0, "sum_of_costs": "", "actual_sum_of_cost": "",
        "n_robots_finished": "", "mpc_failure_reason": "",
        "n_nodes_compared": 0, "n_nodes_missing": "",
        "mean_eta_diff_s": "", "max_abs_eta_diff_s": "",
        "mean_schedule_adherence_m": "", "error": "",
    }

    sched_adherence_src = None
    merged = None
    try:
        # create instance
        if method == "grid":
            convert_movingai(
                map_name=map_name,
                n_agents=n_agent,
                scenario=f'random-{scenario}',
                seed=seed,
                method=method,
                connectedness=connectedness, # only for "grid"
                simplify=True, # only for "grid
                cell_size=2,
                out_name=instance_name,
            )
        elif method == "sample":
            convert_movingai(
                map_name=map_name,
                n_agents=n_agent,
                scenario=f'random-{scenario}',
                seed=seed,
                method=method,
                clearance=0.7,  # only for "sampled"
                density=0.1,  # only for "sampled"
                cell_size=2,
                out_name=instance_name,
            )

        result = general_funct(
            instance_name,
            scheduler=True,
            controller=True,
            naive_tracker=False,  # True = proportional baseline, False = NMPC (see mpc_backend)
            ignore_speed_ref=False,
            recording=False,
            scheduler_backend=scheduler,  # "ComSat", "occbs", "aoccbs", or "pp_sipp"
            scheduler_timeout_s=60,
            assign_via_routing=False,
            first_solution_only=False,
            mpc_backend="panoc",
            headless=True,
            late_threshold_s=30.0,
            stuck_timeout_s=False,
            collision_check=True,
            collision_margin=0.0,
            agent_radius=agent_radius,
            conflict_time_margin=conflict_time_margin,
            # Keys schedule.csv/robot_start.json/Actual_*.csv/SchedAdherence_*.csv by
            # instance_name instead of the fixed default names, so this instance's run can't
            # collide with a different instance's run in flight at the same time (see
            # `_dispatch_jobs`).
            run_tag=instance_name,
        )

        # general_funct returns {"status": "no_schedule", ...} without ever
        # touching the controller when the scheduler can't find a solution
        # (see Compo_slim's empty solution on unsat/unknown); any other status
        # comes from run_mpc, i.e. the scheduler succeeded.
        scheduler_success = result.get("status") != "no_schedule"
        row["scheduler_success"] = int(scheduler_success)

        if scheduler_success:
            row["sum_of_costs"] = result.get("sum_of_costs", "")

            mpc_status = result.get("status")
            row["n_robots_finished"] = result.get("n_robots_finished", "")
            if mpc_status != "success":
                row["mpc_failure_reason"] = MPC_REASON_LABELS.get(mpc_status, mpc_status)

            merged = _merged_schedule_df(instance_name)
            row.update(_schedule_diff_stats(merged))
            row["actual_sum_of_cost"] = _actual_sum_of_cost(merged)
            sched_adherence_src = result.get("sched_adherence_path")
            row["mean_schedule_adherence_m"] = _mean_schedule_adherence(sched_adherence_src)

    except Exception as exc:
        merged = None
        row["error"] = f"{type(exc).__name__}: {exc}"

    # agent_radius=None falls back to the aoccbs/pp_sipp backends' own
    # DEFAULT_AGENT_RADIUS (0.35 m); "mpc" is resolved internally by
    # general_funct but not passed back here, so it's named literally.
    rd_str = "0.35" if agent_radius is None else str(agent_radius)
    # conflict_time_margin=None falls back to the aoccbs/pp_sipp backends' own
    # 0.0 s default (see general_funct), so name the file after what actually ran.
    ctm_str = "0.0" if conflict_time_margin is None else str(conflict_time_margin)
    if save_instance_files:
        node_log_name = f'{instance_name}_{scheduler}_{method_label}_rd{rd_str}_ctm{ctm_str}_nodeLog'
        _write_instance_csv(node_log_name, merged)
        sched_adher_name = f'{instance_name}_{scheduler}_{method_label}_rd{rd_str}_ctm{ctm_str}_SchedAdher'
        _copy_sched_adherence_csv(sched_adherence_src, sched_adher_name)

    return row


def _dispatch_jobs(jobs, n_workers=None, cpu_ids=None):
    """Run a flat list of `_run_one_combo` job specs and append each one's result row to
    experiments_results.csv as soon as it's ready.

    n_workers=None (or <=1, the default) runs the jobs one at a time, in order, in this process
    -- unchanged from ExpRunner's original behaviour.

    n_workers>1 runs up to that many instances at once, each in its own worker process pinned
    to its own CPU core for its whole life (see `_pin_worker`). `cpu_ids` picks which cores --
    default is every core this process can currently run on (`_available_cores()`); pass a
    shorter explicit list to reserve some cores for other work on a shared machine. If
    n_workers exceeds the number of cores given/available, cores are reused round-robin and a
    warning is printed -- parallelism still works, it just stops being one-instance-per-core.

    Two jobs that share an instance_name (same map/scenario/n_agents/seed -- e.g. the same
    generated instance compared across two entries in `schedulers`) are never allowed in flight
    together, no matter how many workers are free: `convert_movingai`'s `_write` unconditionally
    deletes that instance's AOC-CBS state-graph/preprocessing cache on every regeneration
    (`_clear_aoccbs_cache` in roadmap_to_testcase.py), so two concurrent writers for the same
    instance would race on both the test-case JSON and that cache. Jobs with the same
    instance_name are therefore queued and always run strictly one after another; only jobs for
    genuinely different instances actually overlap. `run_tag` (see `_run_one_combo`) is what
    keeps their schedule.csv/robot_start.json/Actual_*.csv files from colliding once they do.

    A "ComSat" job (or any aoccbs/pp_sipp job run with assign_via_routing=True) and AOC-CBS's
    own internal search-pool processes may still try to use more than one thread/process for a
    single instance; the CPU affinity pin is what stops that from spreading onto another
    worker's core (anything sharing the pinned core just contends for it), not a per-library
    thread-count setting. OMP_NUM_THREADS and friends are additionally capped to 1 below purely
    to cut down on wasted contention on that one core -- the dedicated-core guarantee itself
    comes from the affinity pin."""
    if not jobs:
        return
    os.makedirs(results_dir, exist_ok=True)
    if n_workers is None or n_workers <= 1:
        for job in jobs:
            _write_result_row(_run_one_combo(job))
        return

    cores = list(cpu_ids) if cpu_ids else _available_cores()
    if n_workers > len(cores):
        print(f"[ExperimentsRunner] n_workers={n_workers} exceeds {len(cores)} available "
              f"core(s) -- some workers will share a core.", flush=True)

    # Cuts down on wasted thread contention on each worker's one pinned core -- set before the
    # pool is created so every worker inherits it as part of its own OS environment, ahead of
    # any of its own imports (a BLAS library typically sizes its thread pool once, on first use).
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")

    def _instance_name(job):
        return f'{job["map"]}_scenario-{job["scenario"]}_{job["n_agent"]}_{job["seed"]}'

    # Deliberately "spawn", not the platform default -- macOS has no other option, and on Linux
    # a plain fork() of a process that already holds Gurobi/BLAS locks or background threads is
    # a well-known way to deadlock a child before it ever runs anything.
    mp_ctx = mp.get_context("spawn")
    counter = mp_ctx.Value('i', 0)
    lock = mp_ctx.Lock()

    pending = list(jobs)
    in_flight = {}  # future -> instance_name
    active_instances = set()
    with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx,
                                 initializer=_pin_worker, initargs=(counter, lock, cores)) as executor:
        while pending or in_flight:
            submitted = True
            while submitted and len(in_flight) < n_workers:
                submitted = False
                for idx, job in enumerate(pending):
                    inst = _instance_name(job)
                    if inst not in active_instances:
                        future = executor.submit(_run_one_combo, job)
                        in_flight[future] = inst
                        active_instances.add(inst)
                        pending.pop(idx)
                        submitted = True
                        break

            done, _ = cf.wait(in_flight.keys(), return_when=cf.FIRST_COMPLETED)
            for future in done:
                inst = in_flight.pop(future)
                active_instances.discard(inst)
                _write_result_row(future.result())


def ExpRunner(schedulers, maps, scenarios, n_agents, seeds, method="grid",
              connectedness=4, agent_radius=None, conflict_time_margin=None,
              n_workers=None, cpu_ids=None, save_instance_files=True):
    """Run every combo in the cross product of schedulers x maps x scenarios x n_agents x
    seeds, at the given method/connectedness/agent_radius/conflict_time_margin, and append
    each combo's row to experiments_results.csv.

    n_workers / cpu_ids: see `_dispatch_jobs` -- None (default) runs every combo sequentially,
    exactly as this function always has; an int > 1 runs up to that many instances at once,
    each pinned to its own CPU core (combos that share an instance_name, e.g. the same instance
    compared across two entries in `schedulers`, still run one after another).

    save_instance_files: True (default, original behaviour) writes each combo's own
    <instance>..._nodeLog.csv (planned-vs-actual per-node ETA breakdown) and
    ..._SchedAdher.csv (per-second schedule-adherence copy) into data/results, alongside the
    always-written summary row in experiments_results.csv. False skips both files for every
    combo in this call -- useful for a large sweep (many maps/agents/seeds x several sweep
    values, optionally n_workers-parallel) where experiments_results.csv's summary row is all
    that's actually consulted afterwards and the per-instance CSVs would just be disk churn."""
    print('Agent radius:', agent_radius)

    method_label = _method_label(method, connectedness)  # e.g. "grid(4)"; only "grid" uses it

    os.makedirs(results_dir, exist_ok=True)

    jobs = [
        {
            "scheduler": scheduler, "map": map_name, "scenario": scenario,
            "n_agent": n_agent, "seed": seed, "method": method,
            "connectedness": connectedness, "method_label": method_label,
            "agent_radius": agent_radius, "conflict_time_margin": conflict_time_margin,
            "save_instance_files": save_instance_files,
        }
        for scheduler in schedulers
        for map_name in maps
        for scenario in scenarios
        for n_agent in n_agents
        for seed in seeds
    ]
    _dispatch_jobs(jobs, n_workers=n_workers, cpu_ids=cpu_ids)

    return results_csv_path

if __name__ == "__main__":

    schedulers = ['aoccbs'] # ComSat, occbs, aoccbs, or pp_sipp

    maps = [
            # 'den312d',
            # 'maze-32-32-2',
            'empty-16-16'
            # 'room-32-32-4',
            ]

    scenarios = ['1']

    n_agents = [
        # 4
        21,22,23,24,25,26,27,28,29,30,
        31,32,33,34,35,36,37,38,39,40,
    ]

    seeds = [
        5,6,7,8,9
    ]

    # How many instances to run at once, each pinned to its own CPU core (see
    # `_dispatch_jobs`/`_pin_worker`). None or 1 = sequential, one instance at a time, exactly
    # like this script always ran. cpu_ids=None uses every core this process can currently run
    # on; pass an explicit list (e.g. [2, 3, 4, 5]) to reserve the rest for other work.
    n_workers = 20
    cpu_ids = None

    # True (default) writes each combo's own nodeLog/SchedAdher CSVs into data/results, on top
    # of the always-written summary row in experiments_results.csv (see `ExpRunner`'s
    # save_instance_files docstring). Set False for a big sweep where only the summary rows get
    # consulted afterwards, to avoid writing one pair of per-instance files per combo.
    save_instance_files = True

    methods = ["grid","sampled"]  # "grid" or "sampled" -- how convert_movingai builds the instance graph

    # "grid" sweeps roadmap connectedness itself (it changes the instance graph); any other
    # method ignores the knob and runs once, so it isn't listed here.
    connectedness_by_method = {"grid": [4, 8]}

    # Which knob this run sweeps -- "agent_radius" or "conflict_time_margin". The other one is
    # held at its default (None) for every instance in the sweep.
    sweep_param = "conflict_time_margin"

    agent_radii = [0.35 * m for m in [1,1.25,1.5,1.75,2,2.25,2.5]]
    ctms = [0,10,15,20,25,30]

    if sweep_param == "agent_radius":
        sweep_values = agent_radii
        # only retry, at the next radius, the instances that failed specifically on
        # late_threshold or collision at the previous radius -- a run that already
        # succeeded is left alone, and a failure mode a bigger radius can't fix
        # (scheduler failure, stuck, timeout) is not retried either.
        fail_reasons = ("late_threshold", "collision")
    elif sweep_param == "conflict_time_margin":
        sweep_values = ctms
        # a bigger margin can't fix a scheduler failure, stuck robot, or timeout either, but it
        # also can't be judged by late/collision alone -- any instance that didn't finish every
        # robot counts as failed (see _failed_instances' fail_reasons=None branch).
        fail_reasons = None
    else:
        raise ValueError(f"sweep_param must be 'agent_radius' or 'conflict_time_margin', got {sweep_param!r}")

    for method in methods:
        for connectedness in connectedness_by_method.get(method, [None]):
            method_label = _method_label(method, connectedness)
            # sweep_values[0] is already fully run (see experiments_results.csv) -- start the
            # retry chain from it instead of re-running the whole grid at that value.
            prev_value = sweep_values[0]
            for value in sweep_values[1:]:
                retry = _failed_instances(prev_value, schedulers, maps, scenarios, n_agents, seeds, method_label,
                                           value_column=sweep_param,
                                           fail_reasons=fail_reasons)
                sweep_kwargs = {"agent_radius": None, "conflict_time_margin": None, sweep_param: value}
                jobs = [
                    {
                        "scheduler": scheduler, "map": map_name, "scenario": scenario,
                        "n_agent": n_agent, "seed": seed, "method": method,
                        "connectedness": connectedness, "method_label": method_label,
                        "agent_radius": sweep_kwargs["agent_radius"],
                        "conflict_time_margin": sweep_kwargs["conflict_time_margin"],
                        "save_instance_files": save_instance_files,
                    }
                    for scheduler, map_name, scenario, n_agent, seed in retry
                ]
                _dispatch_jobs(jobs, n_workers=n_workers, cpu_ids=cpu_ids)
                prev_value = value
