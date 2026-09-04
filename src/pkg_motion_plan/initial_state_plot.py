"""Static preview of a test case's initial state, before the scheduler runs.

Shows the map, the roadmap graph overlaid on it, and each robot's start node --
plus its final node, when that can be determined without running the scheduler
(see `_robot_goals` below). Everything here is read straight from
`data/test_cases/<problem>.json`; nothing depends on `schedule.csv` or
`robot_start.json`, so this can pop up before the scheduler computes anything.
"""

import json
import os
import pathlib

import matplotlib.pyplot as plt # type: ignore

from basic_map.map_geometric import GeometricMap
from basic_map.graph import NetGraph

_COLOR_LIST = [
    "#0072B2", "#D55E00", "#009E73", "#F0E442", "#56B4E9",
    "#E69F00", "#CC79A7",
]


def _robot_goals(data: dict) -> dict:
    """Each robot's final node, when the instance makes that unambiguous.

    Reuses `pkg_sche.occbs.runner.robot_task_chains`, which walks precedence
    among jobs to recover each robot's ordered node sequence -- but only when
    every job is already pinned to exactly one robot and precedence forms a
    single chain per robot. That is a pure read of the test case (no Gurobi/Z3,
    no solver run), but it is also the same degenerate case the occbs/aoccbs
    backends require: an instance needing real task assignment (a job listing
    several candidate ATRs) raises instead, since the final node then depends
    on a decision the scheduler itself has not made yet.
    """
    from pkg_sche.occbs.runner import robot_task_chains
    try:
        chains = robot_task_chains(data)
    except ValueError as e:
        print(f"[initial_state_plot] Can't determine final nodes before scheduling "
              f"({e}); showing start nodes only.")
        return {}
    return {rid: chain[-1] for rid, chain in chains.items() if chain}


def plot_initial_state(problem: str, block: bool = True) -> None:
    """Pop up a plot of `problem`'s initial state and (if `block`) wait for it to close."""
    root_dir = pathlib.Path(__file__).resolve().parents[2]
    test_case_path = os.path.join(root_dir, "data", "test_cases", f"{problem}.json")
    with open(test_case_path, "r") as f:
        data = json.load(f)

    env_folder = data["test_data"]["Environment"]
    map_path = os.path.join(root_dir, "data", "schedule_demo2_data", env_folder, "map.json")

    geometric_map = GeometricMap.from_json(map_path)
    graph = NetGraph.from_test_case_json(test_case_path)
    nodes = data["test_data"]["nodes"]
    robot_starts = data["ATRs"]
    robot_goals = _robot_goals(data)

    fig, ax = plt.subplots(figsize=(10, 10))
    geometric_map.plot(ax)
    graph.plot(ax, alpha=0.3)
    ax.set_xlabel("X [m]", fontsize=15)
    ax.set_ylabel("Y [m]", fontsize=15)
    ax.axis("equal")
    ax.set_title(f"Initial state -- {problem}")

    for i, (rid, start_node) in enumerate(sorted(robot_starts.items())):
        color = _COLOR_LIST[i % len(_COLOR_LIST)]
        sx, sy = nodes[start_node]["x"], nodes[start_node]["y"]
        ax.plot(sx, sy, marker="*", color=color, markersize=18, label=f"{rid} start")
        if rid in robot_goals:
            gx, gy = nodes[robot_goals[rid]]["x"], nodes[robot_goals[rid]]["y"]
            ax.plot(gx, gy, marker="X", color=color, markersize=15, label=f"{rid} goal")

    ax.legend(loc="best", fontsize=8)
    plt.show(block=block)
