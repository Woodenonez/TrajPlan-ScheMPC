"""Tier 3 -- re-plan one robot with SIPP while every other robot's plan stays fixed.

This is the escalation of last resort, and it reuses the `pp_sipp` backend's own machinery
rather than calling a scheduler again: `SIPP_with_collision_manager` plans a single agent
against whatever plans a `ConflictManager` already holds, which is exactly the semantics
wanted here -- replan the robot that has to give way, treat the rest of the fleet as
immovable. It is a heuristic, and a fast one; the point of escalating is to get the robot
moving again, not to recover optimality.

Two details drive the shape of this module:

* SIPP always starts its search at time zero -- there is no start-time concept anywhere in
  the agent model -- so a mid-run replan has to be done in a shifted frame whose origin is
  the moment the robot reaches its current target node, and the answer shifted back.
* A `ConflictManager` expands a move's swept region from its *start* time and the edge's
  nominal duration, so a synthesised plan for another robot must use exactly that duration
  for each move and express any waiting as separate wait actions.
"""

import contextlib
import io
import json
import math
from typing import Any, Optional

import numpy as np

from aoccbs.config.schema import RunConfig, SolverConfig
from aoccbs.core_types import MoveAction, Plan, WaitAction
from aoccbs.solver.conflict_manager import ConflictManager, load_ii_lookups
from aoccbs.solver.lookups import StateGraphDistanceTable
from aoccbs.solver.sipp import SIPP
from aoccbs.models.model1.circular_agent_model import create_circular_agent
from aoccbs.models.state_graph import load_state_graph
from aoccbs.models import state_graph_distances, intersection_intervals

from pkg_sche.aoccbs.runner import (
    _build_state_graph, _robot_task_specs, _robot_task_specs_via_routing, _build_problem_config,
)
from pkg_sche.pp_sipp.runner import _build_agents


class SippReplanner:
    """The SIPP solver and its caches, built once and reused for every escalation.

    Construction is the expensive part -- state graph, all-pairs distances, pairwise
    intersection intervals -- and all of it is disk-cached, so the cost is paid at most once
    per problem and agent radius. A replan afterwards is one A* search plus the rebuilding of
    the other robots' fixed plans.
    """

    def __init__(self, problem: str, test_case_path: str, gpc, agent_radius: float,
                 assign_via_routing: bool = False, verbose: bool = False) -> None:
        self.gpc = gpc
        self.verbose = verbose

        with open(test_case_path) as f:
            data = json.load(f)

        stdout_ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
        with stdout_ctx:
            self.sg_id = _build_state_graph(problem, data['test_data']['nodes'], verbose=verbose)
            self.am_id = create_circular_agent(agent_radius)
            state_graph_distances.ensure_state_graph_distances(self.sg_id)
            intersection_intervals.ensure_intersection_intervals(
                self.sg_id, self.am_id, self.sg_id, self.am_id)

            chains = (_robot_task_specs_via_routing(problem) if assign_via_routing
                      else _robot_task_specs(data))
            problem_config = _build_problem_config(chains, data['ATRs'], self.am_id, self.sg_id)

            self.solver_config = SolverConfig(verbosity='summary' if verbose else 'silent')
            self.sg = load_state_graph(self.sg_id)
            self.agents = _build_agents(problem_config, {self.sg_id: self.sg})

            # Every vertex, not just the task vertices `pp_sipp` needs: a mid-run replan can be
            # rooted anywhere on the graph, and the distance table raises for a target it was
            # not built with.
            lookups = {self.sg_id: StateGraphDistanceTable(self.sg_id).build_lookup(set(self.sg.vertices))}
            self.ii_lookups = load_ii_lookups({agent.type for agent in self.agents.values()})
            self.cm = ConflictManager(RunConfig(solver_config=self.solver_config), self.agents,
                                      ii_lookups=self.ii_lookups)
            self.sipp = SIPP(state_graphs={self.sg_id: self.sg}, state_graph_lookups=lookups,
                             agents=self.agents, config=self.solver_config)

        self.edge_of = {(self.sg.edge_source(e), self.sg.edge_target(e)): e for e in self.sg.edges}

    # ----------------------------------------------------------------- plans

    def synthesise_plan(self, robot_id, node_ids, times, epoch: float) -> Optional[Plan]:
        """A fixed `Plan` for a robot that is simply carrying on with its own schedule.

        Times are shifted so that `epoch` becomes zero; an action already under way keeps a
        negative start time, which correctly blocks the beginning of the search. Each move
        takes exactly the edge's nominal duration, with any slack expressed as a wait, because
        that is the only form the conflict manager reads correctly.
        """
        if len(node_ids) < 1:
            return None
        actions: list = []
        cursor = times[0] - epoch
        for i in range(len(node_ids) - 1):
            edge = self.edge_of.get((node_ids[i], node_ids[i+1]))
            if edge is None:
                return None
            duration = self.sg.edge_duration(edge)
            t_end = times[i+1] - epoch
            t_start = t_end - duration
            if t_start - cursor > 1e-9:
                actions.append(WaitAction(robot_id, vertex=node_ids[i],
                                          t_start=cursor, t_end=t_start))
            elif t_start < cursor - 1e-9:
                # The recorded ETAs imply a faster traversal than the edge allows (a robot
                # running ahead of a stale schedule). Anchor on the start instead, so the
                # swept region stays a full, correctly-sized traversal.
                t_start = cursor
            actions.append(MoveAction(robot_id, edge=edge, t_start=t_start,
                                      t_end=t_start + duration))
            cursor = max(t_end, t_start + duration)
        # A robot stays wherever it finishes, for good -- which is what makes it an obstacle
        # for anyone routed through that node later.
        actions.append(WaitAction(robot_id, vertex=node_ids[-1], t_start=cursor, t_end=math.inf))
        return Plan(robot_id, tuple(actions), duration=cursor)

    def replan(self, robot_id, start_vertex, epoch: float, remaining_tasks,
               others: dict) -> Optional[list[tuple[Any, float]]]:
        """Plan `robot_id` from `start_vertex` at `epoch`, avoiding every plan in `others`.

        `others` maps robot id -> (node_ids, times) for the rest of the fleet, in absolute
        time. Returns the new `[(node_id, eta), ...]` timetable in absolute time, or None.
        """
        import dataclasses

        if not remaining_tasks:
            return None

        agent = self.agents.get(robot_id)
        if agent is None:
            return None

        for other_id, (node_ids, times) in others.items():
            if other_id == robot_id:
                continue
            plan = self.synthesise_plan(other_id, node_ids, times, epoch)
            if plan is None:
                return None
            self.cm.replace_plan(other_id, plan)

        replacement = dataclasses.replace(agent, start_state=start_vertex,
                                          tasks=tuple(remaining_tasks))
        self.agents[robot_id] = replacement
        self.sipp.agents[robot_id] = replacement
        self.cm.agents[robot_id] = replacement
        self.cm.replace_plan(robot_id, Plan(robot_id, tuple(), duration=0.0))

        stdout_ctx = contextlib.nullcontext() if self.verbose else contextlib.redirect_stdout(io.StringIO())
        with stdout_ctx:
            plan = self.sipp.SIPP_with_collision_manager(agent_id=robot_id, cm=self.cm)
        if plan is None:
            return None

        timetable = [(start_vertex, epoch)]
        for action in plan.actions:
            if action.is_move:
                timetable.append((self.sg.edge_target(action.edge), action.t_end + epoch))
        return timetable


def apply_replan(coordinator, active, now: float) -> tuple[bool, str]:
    """Re-plan the yielding robot and install the result. Returns (succeeded, detail)."""
    rid = active.yielder
    tt = coordinator.timetable(rid)
    if tt.finished:
        return False, f"{rid} has no route left to replan"

    planner = coordinator.rm.get_planner(rid)
    robot = coordinator.rm.get_robot(rid)

    # Start from the node the robot is already committed to driving into: it cannot back out
    # of the edge it is on, and any other choice would mean either reversing or inventing a
    # graph vertex the cached distance tables do not know about.
    start_vertex = tt.node_ids[tt.index]
    pos = np.asarray(robot.state[:2], dtype=float)
    dist = float(np.linalg.norm(pos - np.asarray(tt.coords[tt.index], dtype=float)))
    # A robot that is being held has a measured speed of zero, and estimating its arrival
    # from that would put the replan's origin tens of seconds into the future -- leaving the
    # robot crawling towards deadlines it has already been given. The replan is what releases
    # the hold, so estimate from the speed it is about to travel at instead.
    speed = (coordinator.v_max if coordinator.held(rid)
             else max(coordinator._measured_speed(rid),
                      coordinator.cfg.min_speed_fraction*coordinator.v_max))
    epoch = now + dist/speed

    try:
        replanner = coordinator._ensure_replanner()
    except Exception as exc:  # the AOC-CBS install or its caches are the usual culprit
        return False, f"could not build the SIPP replanner: {exc}"

    remaining_tasks = _remaining_tasks(replanner, rid, tt)
    if not remaining_tasks:
        return False, f"{rid} has no remaining task vertices to plan towards"

    others = {}
    for other_id in coordinator.robot_ids:
        if other_id == rid:
            continue
        other_tt = coordinator.timetable(other_id)
        idx = min(other_tt.index, len(other_tt) - 1)
        nodes = other_tt.node_ids[idx:]
        times = [other_tt.projected_arrival(i) for i in range(idx, len(other_tt))]
        if nodes:
            others[other_id] = (nodes, times)

    try:
        timetable = replanner.replan(rid, start_vertex, epoch, remaining_tasks, others)
    except Exception as exc:
        return False, f"SIPP replan raised {type(exc).__name__}: {exc}"
    if timetable is None:
        return False, f"SIPP found no path for {rid} from {start_vertex} at t={epoch:.2f}"

    coords = [coordinator.gpc.current_graph.get_node_coord(n) for n, _ in timetable]
    node_ids = [n for n, _ in timetable]
    etas = [t for _, t in timetable]

    path_coords = [(float(pos[0]), float(pos[1]))] + [(float(c[0]), float(c[1])) for c in coords]
    path_times = [now] + etas
    if _has_duplicate(path_coords):
        path_coords, path_times = path_coords[1:], path_times[1:]
    if len(path_coords) < 2:
        return False, f"replan for {rid} produced a degenerate path"

    from run_mpc import relax_final_eta
    from .geometry import feasible_times

    path_times = feasible_times(path_coords, path_times, coordinator.v_max, now)
    path_times = relax_final_eta(path_coords, path_times, coordinator.v_max)
    planner.load_path(path_coords, path_times, nomial_speed=coordinator.v_max, method='linear')
    coordinator.replace_timetable(rid, node_ids, coords, etas)

    lateness = etas[-1] - tt.etas[-1]
    return True, (f"{rid} replanned from {start_vertex} at t={epoch:.2f}: "
                  f"{len(node_ids)} nodes, finishes {lateness:+.1f}s vs its previous plan")


def _remaining_tasks(replanner: SippReplanner, robot_id, tt) -> tuple:
    """The robot's original task list, trimmed to the tasks it has not reached yet.

    Falls back to a task at the route's final node, so a robot whose tasks are all behind it
    still has somewhere to be planned to -- SIPP dereferences the last task unconditionally.
    """
    from aoccbs.core_types import VertexTask

    agent = replanner.agents.get(robot_id)
    visited = set(tt.node_ids[:tt.index])
    if agent is not None:
        remaining = tuple(task for task in agent.tasks
                          if getattr(task, 'vertex', None) not in visited)
        if remaining:
            return remaining
    return (VertexTask(vertex=tt.node_ids[-1], duration=0.0),)


def _has_duplicate(coords, tol: float = 1e-6) -> bool:
    return any(math.dist(coords[i], coords[i+1]) < tol for i in range(len(coords)-1))
