import os
import csv
import shutil
import pathlib

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
    "error",
]

# python src/roadmap_to_testcase.py movingai --map den312d --scenario random-1 --n-agents 7 --cell-size 2 --seed 7 --clearance 0.7 --out test_9


def _merged_schedule_df(instance_name):
    """Join the planned schedule.csv against the realised Actual_<instance_name>.csv, both
    in the robot_id,node_id,ETA format, on (robot_id, node_id). Returns None if either file
    is missing (scheduler failed, so schedule.csv wasn't refreshed for this instance, or the
    controller never ran)."""
    schedule_path = os.path.join(schedule_dir, "schedule.csv")
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
    `merged` preserves schedule.csv's per-robot chronological row order (see
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
    node -- i.e. schedule.csv and Actual_<instance_name>.csv joined on (robot_id, node_id).
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


def ExpRunner(schedulers, maps, scenarios, n_agents, seeds, method="grid",
              agent_radius=None, conflict_time_margin=None):

    os.makedirs(results_dir, exist_ok=True)

    for scheduler in schedulers:
        for map in maps:
            for scenario in scenarios:
                for n_agent in n_agents:
                    for seed in seeds:

                        instance_name = f'{map}_scenario-{scenario}_{n_agent}_{seed}'
                        row = {
                            "scheduler": scheduler, "map": map, "scenario": scenario,
                            "n_agents": n_agent, "seed": seed, "method": method,
                            "agent_radius": agent_radius if agent_radius is not None else "",
                            "conflict_time_margin": conflict_time_margin if conflict_time_margin is not None else "",
                            "scheduler_success": 0, "sum_of_costs": "", "actual_sum_of_cost": "",
                            "n_robots_finished": "", "mpc_failure_reason": "",
                            "n_nodes_compared": 0, "n_nodes_missing": "",
                            "mean_eta_diff_s": "", "max_abs_eta_diff_s": "", "error": "",
                        }

                        sched_adherence_src = None
                        try:
                            # create instance
                            if method == "grid":
                                convert_movingai(
                                    map_name=map,
                                    n_agents=n_agent,
                                    scenario=f'random-{scenario}',
                                    seed=seed,
                                    method=method,
                                    connectedness=4, # only for "grid"
                                    simplify=True, # only for "grid
                                    cell_size=2,
                                    out_name=instance_name,
                                )
                            elif method == "sample":
                                convert_movingai(
                                    map_name=map,
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
                            )

                            # general_funct returns {"status": "no_schedule", ...} without ever
                            # touching the controller when the scheduler can't find a solution
                            # (see Compo_slim's empty solution on unsat/unknown); any other status
                            # comes from run_mpc, i.e. the scheduler succeeded.
                            scheduler_success = result.get("status") != "no_schedule"
                            row["scheduler_success"] = int(scheduler_success)

                            merged = None
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

                        except Exception as exc:
                            merged = None
                            row["error"] = f"{type(exc).__name__}: {exc}"

                        _write_result_row(row)
                        # conflict_time_margin=None falls back to the aoccbs/pp_sipp backends' own
                        # 0.0 s default (see general_funct), so name the file after what actually ran.
                        ctm_str = "0.0" if conflict_time_margin is None else str(conflict_time_margin)
                        node_log_name = f'{instance_name}_{scheduler}_{method}_ctm{ctm_str}_nodeLog'
                        _write_instance_csv(node_log_name, merged)
                        sched_adher_name = f'{instance_name}_{scheduler}_{method}_ctm{ctm_str}_SchedAdher'
                        _copy_sched_adherence_csv(sched_adherence_src, sched_adher_name)

    return results_csv_path

if __name__ == "__main__":

    schedulers = ['aoccbs','pp_sipp'] # ComSat, occbs, aoccbs, or pp_sipp

    maps = [
            # 'den312d',
            'maze-32-32-2',
            # 'room-32-32-4',
            ]

    scenarios = ['1']

    n_agents = [
        # 4,5,6,7,8,9,10,
        # 11,12,13,14,15,16,17,18,19,20,
        21,22,23,24,25,26,27,28,29,30,
        31,32,33,34,35,36,37,38,39,40,
    ]

    seeds = [
        7
    ]

    method = "grid"  # "grid" or "sampled" -- how convert_movingai builds the instance graph

    agent_radius = None  # None, a metres float, or "mpc" -- see general_funct's docstring
    conflict_time_margin = None  # seconds, aoccbs/pp_sipp only -- see general_funct's docstring

    ExpRunner(schedulers, maps, scenarios, n_agents, seeds, method=method,
              agent_radius=agent_radius, conflict_time_margin=conflict_time_margin)
