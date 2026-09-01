import os
import csv
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
    "instance", "map", "scenario", "n_agents", "seed",
    "scheduler_success",
    "mpc_success", "mpc_failure_reason",
    "n_nodes_compared", "n_nodes_missing",
    "mean_eta_diff_s", "max_abs_eta_diff_s",
    "error",
]

maps = ['den312d',
        # 'den520d',
        # 'emtpy-16-16',
        # 'maze-128-128-2',
        # 'maze-32-32-2',
        # 'random-64-64-8',
        # 'room-32-32-4',
        # 'room-64-64-8',
        # 'warehhouse-10-20-10-2-2'
        ]

scenarios = ['1']

n_agents = [
    '4',
    # '5','6','7','8','9','10',
    # '11','12','13','14','15','16','17','18','19','20'
]

seeds = [
    '3',
    # '4','5','6','7',
    # '8','9','10','11','12'
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


def _write_result_row(row):
    file_exists = os.path.exists(results_csv_path)
    with open(results_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _write_instance_csv(instance_name, summary_row, merged):
    """Per-instance CSV. One header line naming all 18 columns, then the summary line --
    the same fields written to experiments_results.csv, filling columns 1-13 only -- then
    one line per scheduled node in columns 14-18: the full planned-vs-actual breakdown,
    i.e. schedule.csv and Actual_<instance_name>.csv joined on (robot_id, node_id).
    The summary is written once rather than repeated on every node line."""
    node_fields = ["robot_id", "node_id", "ETA_planned", "ETA_actual", "ETA_diff"]
    fieldnames = RESULT_FIELDS + node_fields
    blank_summary = {f: "" for f in RESULT_FIELDS}
    blank_nodes = {f: "" for f in node_fields}
    out_path = os.path.join(results_dir, f"{instance_name}.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({**summary_row, **blank_nodes})
        if merged is not None and not merged.empty:
            for _, node_row in merged.iterrows():
                writer.writerow({
                    **blank_summary,
                    "robot_id": node_row["robot_id"], "node_id": node_row["node_id"],
                    "ETA_planned": node_row["ETA_planned"], "ETA_actual": node_row["ETA_actual"],
                    "ETA_diff": node_row["ETA_diff"],
                })
    return out_path


def ExpRunner(maps, scenarios, n_agents, seeds):

    os.makedirs(results_dir, exist_ok=True)

    for map in maps:
        for scenario in scenarios:
            for n_agent in n_agents:
                for seed in seeds:

                    instance_name = f'{map}_scenario-{scenario}_{n_agent}_{seed}'
                    row = {
                        "instance": instance_name, "map": map, "scenario": scenario,
                        "n_agents": n_agent, "seed": seed,
                        "scheduler_success": 0, "mpc_success": "", "mpc_failure_reason": "",
                        "n_nodes_compared": 0, "n_nodes_missing": "",
                        "mean_eta_diff_s": "", "max_abs_eta_diff_s": "", "error": "",
                    }

                    try:
                        # create instance
                        convert_movingai(
                            map_name=map,
                            n_agents=n_agent,
                            scenario=f'{map}-random-{scenario}',
                            seed=seed,
                            method="grid",
                            connectedness=8,
                            simplify=True,
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
                            scheduler_backend="sp_comsat",  # "sp_comsat", "occbs", or "aoccbs"
                            assign_via_routing=False,
                            first_solution_only=False,
                            mpc_backend="panoc",
                            headless=True,
                            late_threshold_s=30.0,
                            stuck_timeout_s=30.0,
                            collision_check=True,
                            collision_margin=0.0,
                        )

                        # general_funct returns {"status": "no_schedule", ...} without ever
                        # touching the controller when the scheduler can't find a solution
                        # (see Compo_slim's empty solution on unsat/unknown); any other status
                        # comes from run_mpc, i.e. the scheduler succeeded.
                        scheduler_success = result.get("status") != "no_schedule"
                        row["scheduler_success"] = int(scheduler_success)

                        merged = None
                        if scheduler_success:
                            mpc_status = result.get("status")
                            mpc_success = mpc_status == "success"
                            row["mpc_success"] = int(mpc_success)
                            if not mpc_success:
                                row["mpc_failure_reason"] = MPC_REASON_LABELS.get(mpc_status, mpc_status)

                            merged = _merged_schedule_df(instance_name)
                            row.update(_schedule_diff_stats(merged))

                    except Exception as exc:
                        merged = None
                        row["error"] = f"{type(exc).__name__}: {exc}"

                    _write_result_row(row)
                    _write_instance_csv(instance_name, row, merged)

    return results_csv_path
