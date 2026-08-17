"""Drive the OC-CBS solver over a test case and emit this project's schedule.

This is the OC-CBS counterpart to `pkg_sche.sp_comsat.Compo_slim.Compo_slim`:
it takes a test-case name and returns {robot_id: [(node_id, eta), ...]}, which
`main.py` writes to schedule.csv for the MPC layer to track.

The two schedulers do not solve the same problem. sp_comsat assigns tasks to
vehicles and honours time windows, precedence, service times and battery
autonomy; OC-CBS solves continuous-time MAPF, where each agent has one start
and one goal. Everything sp_comsat models beyond geometry and conflicts is
therefore outside what this backend can express -- see `plan_legs` for how the
per-robot task chains are fed through a single-goal solver.
"""

import json
import os
import pathlib
import subprocess
import xml.etree.ElementTree as ET

from .roadmap import Roadmap
from . import solution as solution_mod

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_BINARY = PROJECT_ROOT / "external" / "Optimal-Continuous-CBS" / "CCBS"

DEFAULT_CONFIG = {
    'use_cardinal': 'true',
    'use_disjoint_splitting': 'false',
    'hlh_type': '0',
    'connectedness': '2',
    'focal_weight': '1.0',
    'agent_size': '0.35',
    'timelimit': '30',
    'precision': '1e-6',
    'branching_gamma': '0.5',
}


def write_config(path: str, overrides: dict = None) -> None:
    values = dict(DEFAULT_CONFIG)
    values.update(overrides or {})
    root = ET.Element('root')
    algorithm = ET.SubElement(root, 'algorithm')
    for key, value in values.items():
        ET.SubElement(algorithm, key).text = str(value)
    tree = ET.ElementTree(root)
    ET.indent(tree, space='  ')
    tree.write(path, encoding='UTF-8', xml_declaration=True)


def write_task(path: str, roadmap: Roadmap, legs: list) -> list:
    """Write a task file for one batch of (robot_id, start, goal, release) rows.

    `release` is the earliest time the robot may leave `start`; it becomes the
    `start_time` attribute this project adds to the solver (see
    external/occbs_release_times.patch). Returns the robot ids in the order they
    were written, which is the order the solver reports them back in.
    """
    root = ET.Element('root')
    agent_names = []
    for robot_id, start, goal, release in legs:
        ET.SubElement(root, 'agent', {
            'start_id': str(roadmap.name_to_id[start]),
            'goal_id': str(roadmap.name_to_id[goal]),
            'start_time': repr(float(release)),
        })
        agent_names.append(robot_id)
    tree = ET.ElementTree(root)
    ET.indent(tree, space='  ')
    tree.write(path, encoding='UTF-8', xml_declaration=True)
    return agent_names


def robot_task_chains(problem_data: dict) -> dict:
    """Recover each robot's ordered node sequence from a test case.

    Only the degenerate case is handled: every job pinned to exactly one robot,
    with precedence forming a single chain per robot. That is what makes task
    assignment trivial and lets a MAPF solver stand in for the scheduler. Any
    instance needing real assignment raises instead of quietly guessing.
    """
    jobs = problem_data['jobs']
    chains = {robot_id: [] for robot_id in problem_data['ATRs']}

    owned = {}
    for job_id, job in jobs.items():
        if job_id.split('_')[0] in ('start', 'recharge'):
            continue
        if len(job['ATR']) != 1:
            raise ValueError(
                f"job {job_id} may be run by {job['ATR']}; OC-CBS cannot assign "
                "tasks to vehicles, so this instance needs sp_comsat")
        owned.setdefault(job['ATR'][0], []).append(job_id)

    for robot_id, job_ids in owned.items():
        # Order the robot's jobs by following the precedence chain.
        predecessors = {j: [p for p in jobs[j]['precedence'] if p in job_ids] for j in job_ids}
        ordered, remaining = [], list(job_ids)
        while remaining:
            ready = [j for j in remaining if all(p in ordered for p in predecessors[j])]
            if not ready:
                raise ValueError(f"precedence among {remaining} for {robot_id} is cyclic")
            if len(ready) > 1:
                raise ValueError(
                    f"jobs {ready} for {robot_id} are unordered; OC-CBS has no "
                    "notion of task ordering, so this instance needs sp_comsat")
            ordered.append(ready[0])
            remaining.remove(ready[0])
        chains[robot_id] = [jobs[j]['location'] for j in ordered]

    return chains


def plan_legs(problem: str, work_dir: str, binary: str = None,
              config_overrides: dict = None, verbose: bool = True) -> tuple:
    """Solve a test case as a sequence of single-goal MAPF problems.

    OC-CBS gives each agent one goal, but a robot here has a chain of task
    locations. The chain is therefore cut into legs: leg k routes every robot
    from its k-th location to its (k+1)-th, and all robots' k-th legs are solved
    together as one MAPF instance.

    Robots do not finish a leg at the same moment, so each agent enters leg k
    with a release time equal to its own arrival time at the end of leg k-1, and
    is treated as parked at that node until then. Leg times are therefore already
    on a common absolute clock and are concatenated without shifting.

    This is still a decomposition rather than an equivalent reformulation: each
    leg is solved optimally in isolation, so the concatenation is not jointly
    optimal over the whole mission. What release times buy is soundness of the
    conflict resolution -- a robot waiting at a node is visible to the robots
    still moving, which is what the simultaneous-departure version got wrong.
    """
    with open(f"{PROJECT_ROOT}/data/test_cases/{problem}.json") as f:
        data = json.load(f)

    binary = str(binary or DEFAULT_BINARY)
    if not os.path.exists(binary):
        raise FileNotFoundError(
            f"OC-CBS binary not found at {binary}. Build it with:\n"
            "  g++ -std=c++11 -O2 -I. -I/opt/homebrew/include main.cpp config.cpp "
            "tinyxml2.cpp xml_logger.cpp map.cpp heuristic.cpp sipp.cpp task.cpp "
            "cbs.cpp simplex/*.cpp -o CCBS\n"
            f"from {PROJECT_ROOT}/external/Optimal-Continuous-CBS")

    os.makedirs(work_dir, exist_ok=True)
    roadmap = Roadmap(data['test_data']['nodes'])
    map_path = os.path.join(work_dir, 'roadmap.xml')
    config_path = os.path.join(work_dir, 'config.xml')
    roadmap.write_graphml(map_path)
    write_config(config_path, config_overrides)

    chains = robot_task_chains(data)
    positions = dict(data['ATRs'])
    release = {rid: 0.0 for rid in chains}
    n_legs = max((len(c) for c in chains.values()), default=0)

    legs_solved = []
    stats = []
    for k in range(n_legs):
        batch = [(rid, positions[rid], chains[rid][k], release[rid])
                 for rid in sorted(chains) if k < len(chains[rid])]
        batch = [row for row in batch if row[1] != row[2]]
        if not batch:
            continue

        task_path = os.path.join(work_dir, f'task_leg{k}.xml')
        agent_names = write_task(task_path, roadmap, batch)

        result = subprocess.run([binary, map_path, task_path, config_path],
                                capture_output=True, text=True)
        if verbose:
            print(f"--- OC-CBS leg {k} ({len(batch)} agents) ---")
            print(result.stdout.strip())
        if 'Solution found: true' not in result.stdout:
            raise solution_mod.NoSolution(
                f"OC-CBS found no solution for leg {k}:\n{result.stdout}{result.stderr}")

        log_path = task_path.replace('.xml', '_log.xml')
        leg = solution_mod.parse_log(log_path, roadmap, agent_names)
        legs_solved.append(leg)
        stats.append(solution_mod.parse_summary(log_path))

        for rid, _, goal, _ in batch:
            positions[rid] = goal
            # Absolute arrival at this leg's goal releases the robot for the next.
            release[rid] = leg[rid][-1][1]

    return solution_mod.stitch(legs_solved), stats


def OCCBS(problem: str, work_dir: str = None, **kwargs) -> tuple:
    """Entry point mirroring `Compo_slim`: returns (solution, stats)."""
    work_dir = work_dir or os.path.join(PROJECT_ROOT, 'data', 'occbs_work', problem)
    return plan_legs(problem, work_dir, **kwargs)
