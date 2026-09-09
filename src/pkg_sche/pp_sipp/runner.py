"""Drive prioritized planning with SIPP over a test case and emit this project's schedule.

This is a fourth scheduler backend, alongside `sp_comsat`, `occbs`, and `aoccbs`: a much simpler
baseline meant for comparison against `aoccbs`'s full anytime CBS search rather than as a
replacement for it. Prioritized planning fixes an order over the robots up front, then plans them
one at a time with SIPP: each robot's plan must avoid every higher-priority robot's plan (already
fixed), but nothing enforces the reverse, so a low-priority robot's plan can wait arbitrarily long
on a high-priority robot's path. It is fast (one shortest-path search per robot, no search tree
over joint plans) but incomplete -- a bad priority order can make a solvable instance fail here
even though `sp_comsat`/`occbs`/`aoccbs` would still find a solution, which is exactly what makes it
a useful floor to compare the other backends against.

This module deliberately does not duplicate `pkg_sche.aoccbs.runner`'s problem-setup pipeline: it
reuses that module's state-graph construction/caching, task-chain recovery (both the "every job
pre-pinned to one ATR" path and the `assign_via_routing` Gurobi-assignment path), problem-config
assembly, and schedule extraction directly. What is genuinely different is the solve step -- this
runs a single greedy priority sweep with `aoccbs.solver.sipp.SIPP.SIPP_with_collision_manager`
instead of `aoccbs.solver.aoccbs.AOCCBS`'s conflict-based search, so the AOC-CBS worker pool and
constraint-tree machinery are never constructed. `AOCCBS.evaluate_joint_plan` is reused as-is to
score the result, since scoring a joint plan does not depend on how it was produced.

Requires the same `external/AOC-CBS` install and node-link-data patch as `aoccbs` -- see that
module's docstring and the project's CLAUDE.md ("Third scheduler backend").
"""

import time

from aoccbs.config.schema import RunConfig, SolverConfig, ProblemConfig
from aoccbs.core_types import Agent, AgentType, VertexTask, EdgeTask, JointPlan
from aoccbs.solver.aoccbs import AOCCBS as AOCCBSSolver
from aoccbs.solver.conflict_manager import ConflictManager, load_ii_lookups
from aoccbs.solver.lookups import StateGraphDistanceTable
from aoccbs.solver.sipp import SIPP
from aoccbs.models.model1.circular_agent_model import create_circular_agent
from aoccbs.models.state_graph import load_state_graph
from aoccbs.models import state_graph_distances, intersection_intervals

from pkg_sche.occbs.solution import NoSolution
from pkg_sche.aoccbs.runner import (
    PROJECT_ROOT, DEFAULT_AGENT_RADIUS,
    _build_state_graph, _robot_task_specs, _robot_task_specs_via_routing,
    _build_problem_config, _extract_schedule,
)

import contextlib
import io
import json
import os


def _build_agents(problem_config: ProblemConfig, state_graphs: dict) -> dict:
    """Concrete Agent objects (start state + tasks) for a problem config.

    Mirrors `aoccbs.solver.aoccbs.AOCCBS._build_agents`, duplicated (not called) because that
    method is small but reads `self.state_graphs` off a live solver instance, and constructing one
    just to reach this method would pull in the worker pool and CT-node search machinery that
    prioritized planning has no use for.
    """
    agents: dict = {}
    for agent_id, agent_config in problem_config.agent_map.items():
        tasks = []
        for task in agent_config.tasks:
            if isinstance(task, str):
                tasks.append(EdgeTask(edge=task, state_graph=state_graphs[agent_config.state_graph]))
            elif isinstance(task, tuple) and len(task) == 2:
                tasks.append(VertexTask(vertex=task[0], duration=task[1]))
            else:
                raise ValueError(
                    f"Invalid task format for agent {agent_id}: {task!r}. Tasks must be either "
                    "vertex tasks (tuple of vertex label and duration) or edge tasks (edge label).")
        agents[agent_id] = Agent(
            id=agent_id,
            type=AgentType(agent_model_id=agent_config.agent_model, state_graph_id=agent_config.state_graph),
            start_state=agent_config.start_state,
            tasks=tuple(tasks),
        )
    return agents


def _targets_per_sg(agents: dict) -> dict:
    """The set of task (goal) vertices per state graph -- the only vertices SIPP queries distances
    to. Same role as `AOCCBS._targets_per_sg`, recomputed here for the same reason as `_build_agents`."""
    targets: dict = {}
    for agent in agents.values():
        targets.setdefault(agent.type.state_graph_id, set()).update(task.source for task in agent.tasks)
    return targets


def PP_SIPP(problem: str, agent_radius: float = DEFAULT_AGENT_RADIUS,
           priority_order: list = None, solver_overrides: dict = None, workers: int = None,
           verbose: bool = True, assign_via_routing: bool = False, timeout: float = None) -> tuple:
    """Entry point mirroring `AOCCBS`/`OCCBS`/`Compo_slim`: returns (solution, stats).

    `priority_order` fixes the robot planning order; a robot's SIPP plan avoids every
    already-planned (i.e. earlier in this list) robot's fixed plan, but is never revisited once
    planned. Defaults to `sorted(robot_id)` -- deterministic, but arbitrary with respect to the
    instance, which is the point: this backend's failures (`NoSolution` on an instance the other
    backends solve) are exactly the cost of not searching over priority orders or replanning.

    `assign_via_routing` has the same meaning as on `AOCCBS`: it lifts the usual "every job
    already pinned to one robot" restriction by running sp_comsat's Gurobi routing sub-solver
    first to decide the assignment (see `pkg_sche.aoccbs.runner._robot_task_specs_via_routing`).

    `agent_radius` has the same meaning as on `AOCCBS`: SIPP's safe intervals come from the same
    disc-overlap test, so the plan keeps robot centres at least `2*agent_radius` apart and that
    radius is the only clearance knob (see `pkg_sche.aoccbs.runner.mpc_matched_agent_radius`).

    `timeout` is a wall-clock budget (seconds) for the whole priority sweep, checked between
    robots. Unlike `AOCCBS`'s `timelimit`, there is nothing here to hand it to: each robot's SIPP
    call is a single bounded shortest-path search, not an anytime loop, so it cannot be cut off
    mid-search -- `timeout` only stops the sweep from starting another robot once the budget is
    already spent, raising `NoSolution` the same way a robot with no feasible SIPP path does.
    """
    with open(f"{PROJECT_ROOT}/data/test_cases/{problem}.json") as f:
        data = json.load(f)

    sg_id = _build_state_graph(problem, data['test_data']['nodes'], verbose=verbose)
    am_id = create_circular_agent(agent_radius)

    workers = workers or os.cpu_count() or 1
    stdout_ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with stdout_ctx:
        state_graph_distances.ensure_state_graph_distances(sg_id, workers=workers)
        intersection_intervals.ensure_intersection_intervals(sg_id, am_id, sg_id, am_id, workers=workers)

    chains = _robot_task_specs_via_routing(problem) if assign_via_routing else _robot_task_specs(data)
    problem_config = _build_problem_config(chains, data['ATRs'], am_id, sg_id)

    solver_config = SolverConfig(**{
        'verbosity': 'summary' if verbose else 'silent',
        **(solver_overrides or {}),
    })

    state_graphs = {sg_id: load_state_graph(sg_id)}
    agents = _build_agents(problem_config, state_graphs)

    order = sorted(agents) if priority_order is None else list(priority_order)
    if set(order) != set(agents):
        raise ValueError(
            f"priority_order {order} does not match this problem's robots {sorted(agents)}")

    state_graph_lookups = {
        graph_id: StateGraphDistanceTable(graph_id).build_lookup(targets)
        for graph_id, targets in _targets_per_sg(agents).items()
    }
    ii_lookups = load_ii_lookups({agent.type for agent in agents.values()})

    cm = ConflictManager(RunConfig(solver_config=solver_config), agents, ii_lookups=ii_lookups)
    sipp = SIPP(state_graphs=state_graphs, state_graph_lookups=state_graph_lookups,
               agents=agents, config=solver_config)

    t0 = time.time()
    plans = {}
    for i, agent_id in enumerate(order):
        if timeout is not None and time.time() - t0 > timeout:
            raise NoSolution(
                f"Prioritized planning exceeded its timeout ({timeout}s) after planning "
                f"{i} of {len(order)} robots; priority order was {order}")
        plan = sipp.SIPP_with_collision_manager(agent_id=agent_id, cm=cm)
        if plan is None:
            raise NoSolution(
                f"Prioritized planning found no SIPP path for robot {agent_id!r} given the fixed "
                f"plans of higher-priority robots {order[:i]}; priority order was {order}")
        plans[agent_id] = plan
        cm.replace_plan(agent_id, plan)
    runtime = time.time() - t0

    joint_plan = AOCCBSSolver.evaluate_joint_plan(JointPlan(plans=plans), cm, solver_config.objective_function)
    if not joint_plan.is_solution:
        # Should not happen: each robot's SIPP call already avoids every earlier robot's fixed
        # plan, so the joint plan built one robot at a time is collision-free by construction.
        raise RuntimeError(
            f"prioritized planning produced {joint_plan.nr_collisions} residual collisions with "
            f"priority order {order}; this violates the algorithm's own invariant")

    if verbose:
        print(f"PP+SIPP: {runtime:.2f}s, objective {joint_plan.objective_value:.2f}, "
             f"priority order {order}")

    schedule = _extract_schedule(joint_plan, problem_config.agent_map, state_graphs[sg_id])
    stats = {
        'runtime': runtime,
        'priority_order': order,
        'objective_value': joint_plan.objective_value,
    }
    return schedule, stats
