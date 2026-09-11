"""Drive the AOC-CBS solver over a test case and emit this project's schedule.

This is the AOC-CBS counterpart to `pkg_sche.occbs.runner.OCCBS` (and, through it, to
`pkg_sche.sp_comsat.Compo_slim.Compo_slim`): it takes a test-case name and returns
{robot_id: [(node_id, eta), ...]}, which `main.py` writes to schedule.csv for the MPC layer to
track.

Unlike OC-CBS -- vendored, unbuilt C++ that only solves single-goal MAPF, so a robot's task chain
has to be cut into legs and solved leg-by-leg with hand-rolled release times (see
`pkg_sche.occbs.runner.plan_legs`) -- AOC-CBS (github.com/Adcombrink/AOC-CBS) is a pure-Python,
pip-installable library whose agents natively carry an ordered sequence of tasks, each an optional
service time at a vertex. A robot's whole job chain is therefore handed to the solver as one
agent, and the search branches over the true joint problem instead of a sequence of
independently-optimal legs -- which is what OC-CBS's leg decomposition cannot do (see the
"Multi-waypoint instances currently do not work" note in the project's CLAUDE.md).

The same assignment restriction as OC-CBS still applies, and for the same reason: this backend has
no notion of assigning tasks to vehicles, so it only accepts instances where assignment is already
decided -- every job pinned to exactly one robot, with precedence forming a single chain per robot
(see `_robot_task_specs`, which is `pkg_sche.occbs.runner.robot_task_chains` with each task's
service time kept instead of discarded). Time windows are not modelled either, matching OC-CBS.

One thing this runner does not take from AOC-CBS is its default search portfolio: the shipped one
declares no repair heuristics, which leaves the constraint-tree search as the only source of a
feasible plan and so returns nothing at all on instances with many agents. `DEFAULT_SEARCH_PORTFOLIO`
below is the same portfolio with TPR switched on; see the comment there.

AOC-CBS is vendored unbuilt into the gitignored `external/AOC-CBS` (its own convention: `pip
install -e external/AOC-CBS` after cloning; it also needs `sortedcontainers`, which its
pyproject.toml does not declare). Unlike OC-CBS it needs no separate compile step. It builds its
own model library (state graphs, agent models) and preprocessing cache (distance matrices,
intersection intervals) under `external/AOC-CBS/scratch/` and `external/AOC-CBS/cache/` by
default -- both already inside the gitignored `external/` tree, so nothing generated here needs
its own gitignore entry. `_build_state_graph` names the state graph `TrajPlan_<problem>` and skips
rebuilding it if that id is already in AOC-CBS's model library; the cached distance matrix and
intersection intervals are keyed by the same id, so if a test case's node graph changes, delete
`external/AOC-CBS/scratch/models/StateGraph_TrajPlan_<problem>.json` (and the matching files under
`external/AOC-CBS/cache/`) or the new graph will silently be solved against the old one's
preprocessing.
"""

import contextlib
import io
import json
import os
import pathlib
import time

import networkx as nx

from aoccbs import paths as aoccbs_paths
from aoccbs.config.schema import (
    RunConfig, SolverConfig, ProblemConfig, NamedAgent, AgentConfig, Policy,
    extract_plant_spec, extract_problem_spec,
)
from aoccbs.solver.aoccbs import AOCCBS as AOCCBSSolver
from aoccbs.models.core.loaders import Model1StateGraph, assign_edge_ids, save_model1_state_graph
from aoccbs.models.model1.circular_agent_model import create_circular_agent
from aoccbs.models import state_graph_distances, intersection_intervals
from aoccbs.models.state_graph import load_state_graph

from pkg_sche.occbs.solution import NoSolution

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]

# Matches OC-CBS's default `agent_size` (see occbs.runner.DEFAULT_CONFIG), so the two backends
# model the same physical footprint.
DEFAULT_AGENT_RADIUS = 0.35


def mpc_matched_agent_radius(config_file: str = "robot_spec.yaml") -> float:
    """The agent radius that makes AOC-CBS plan the clearance the NMPC actually wants.

    The radius handed to `create_circular_agent` is the *only* clearance knob these backends
    have: AOC-CBS declares two agents in conflict exactly when their discs overlap, so a plan
    is guaranteed to keep robot centres at least `2*agent_radius` apart and nothing else --
    there is no separate temporal padding to add, and padding the emitted ETAs after the fact
    would not change where two robots pass each other.

    `DEFAULT_AGENT_RADIUS` (0.35) is about the shipped robot's own body radius
    (`vehicle_width` = 0.35355), so a schedule built with it may route two robots past each
    other at ~0.7 m centre-to-centre -- inside `casadi_impl`'s fleet safe distance
    (`2*(vehicle_width + vehicle_margin)` = 1.107 m, where the NMPC's heavy `qfleet_pred` term
    switches on) and barely outside its critical distance (0.907 m). The tracker then has to
    buy that clearance back by deviating from the schedule, which shows up as lateness.
    Planning with `vehicle_width + vehicle_margin` instead makes the planned clearance
    `2*(vehicle_width + vehicle_margin)`, i.e. exactly the fleet safe distance, so the schedule
    is one the NMPC can track without fighting it.

    This is a geometric widening of the agents, not a change to the roadmap: every test case in
    `data/test_cases/` spaces its nodes at least 2 m apart, so it stays well clear of the point
    where two occupied vertices would overlap and the instance would go infeasible.

    Note that a different radius means a different agent-model id, hence a different (and
    initially cold) `intersection_intervals` cache under `external/AOC-CBS/cache/` -- the first
    run at a new radius pays for that preprocessing once.
    """
    from configs import CircularRobotSpecification
    spec = CircularRobotSpecification.from_yaml(f"{PROJECT_ROOT}/config/{config_file}")
    return spec.vehicle_width + spec.vehicle_margin

# AOC-CBS's shipped `SolverConfig.search_portfolio` declares `heuristics=()` on all six policies,
# so its TPR (Tier-Prioritized Repair) heuristic never runs and the *only* way a run acquires a
# feasible joint plan is for the best-first constraint-tree search to reach a conflict-free leaf.
# That is fine on a handful of agents and hopeless beyond it: on `test_2` (21 agents, 261 vertices)
# the shipped portfolio expanded 16k CT nodes in 60s, lifted the lower bound by 0.7 of the ~2138 it
# started at, and never produced a solution at all, so `_solve` raised NoSolution. TPR builds a
# feasible plan directly -- it clears the conflicting agents, pins them at their starts and replans
# them one at a time with SIPP -- so the same instance gets its first solution in ~0.2s and is
# inside a 0.4% gap within seconds.
#
# Every policy carries TPR rather than only some: each runs the repair from its own CT node, and
# the diversity is what tightens the upper bound. Measured on `test_2` at a 25s budget, TPR on 1 /
# 2 / 6 policies closed to a 2.5% / 2.3% / 0.4% gap.
#
# Selections are AOC-CBS's own; only `heuristics` differs.
DEFAULT_SEARCH_PORTFOLIO = (
    Policy(selection="min objective", heuristics=("TPR",)),
    Policy(selection="min nr collisions", heuristics=("TPR",)),
    Policy(selection="min collision time", heuristics=("TPR",)),
    Policy(selection="min objective", heuristics=("TPR",)),
    Policy(selection="min objective", heuristics=("TPR",)),
    Policy(selection="min objective", heuristics=("TPR",)),
)

# Distance units per second on every edge. sp_comsat and OC-CBS both treat an edge's raw
# Euclidean length as its travel time (see support_functions.json_parser and occbs.roadmap's
# docstring); matching that here keeps travel times comparable across all three backends.
DEFAULT_EDGE_SPEED = 1.0


def _build_state_graph(problem: str, nodes: dict, verbose: bool = True) -> str:
    """Build, save and return the id of the Model1 state graph AOC-CBS solves the problem on.

    Reused by id across runs -- see the module docstring for the cache-invalidation caveat.
    """
    sg_id = f"TrajPlan_{problem}"
    try:
        aoccbs_paths.state_graph_model_file(sg_id)
        if verbose:
            print(f"AOC-CBS state graph cached: {sg_id}")
        return sg_id
    except FileNotFoundError:
        pass

    graph = nx.DiGraph()
    for name, node in nodes.items():
        graph.add_node(name, pos=(float(node['x']), float(node['y'])))
    for name, node in nodes.items():
        for other in node['next']:
            other = str(other)
            if other not in nodes:
                raise KeyError(f"node {name!r} lists unknown neighbour {other!r}")
            graph.add_edge(name, other, speed=DEFAULT_EDGE_SPEED)

    state_graph = Model1StateGraph(assign_edge_ids(graph))
    state_graph.id = sg_id
    save_model1_state_graph(state_graph)
    if verbose:
        print(f"built AOC-CBS state graph {sg_id} "
             f"({graph.number_of_nodes()} vertices, {graph.number_of_edges()} edges)")
    return sg_id


def _robot_task_specs(problem_data: dict) -> dict:
    """Recover each robot's ordered (location, service_time) sequence from a test case.

    Same degenerate case OC-CBS requires (see `pkg_sche.occbs.runner.robot_task_chains`): every
    job pinned to exactly one robot, with precedence forming a single chain per robot. Kept
    separate rather than reusing that function directly, because it discards each job's service
    time and AOC-CBS's vertex tasks need it.
    """
    jobs = problem_data['jobs']
    chains = {robot_id: [] for robot_id in problem_data['ATRs']}

    owned = {}
    for job_id, job in jobs.items():
        if job_id.split('_')[0] in ('start', 'recharge'):
            continue
        if len(job['ATR']) != 1:
            raise ValueError(
                f"job {job_id} may be run by {job['ATR']}; AOC-CBS cannot assign "
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
                    f"jobs {ready} for {robot_id} are unordered; AOC-CBS has no "
                    "notion of task ordering, so this instance needs sp_comsat")
            ordered.append(ready[0])
            remaining.remove(ready[0])
        chains[robot_id] = [(jobs[j]['location'], float(jobs[j]['Service'])) for j in ordered]

    return chains


def _robot_task_specs_via_routing(problem: str) -> dict:
    """Recover each robot's ordered (location, service_time) sequence by solving the
    task-to-vehicle assignment with sp_comsat's Gurobi routing sub-solver.

    Unlike `_robot_task_specs`, this does not require every job to already be pinned to a
    single robot -- a job's `ATR` list may name several candidate vehicles, and
    `E_Routing_Gurobi.routing` (the same MILP `sp_comsat.Compo_slim` uses for its own
    scheduler) decides both the assignment and each robot's visit order, subject to
    precedence, time windows and autonomy. AOC-CBS then only has to solve the trajectories.
    Needs Gurobi; imported lazily so the rest of this module -- and the plain
    `_robot_task_specs` path -- stays usable without it.
    """
    from z3 import sat
    from pkg_sche.sp_comsat.Compo_slim import build_instance
    from pkg_sche.sp_comsat.E_Routing_Gurobi import routing as gurobi_routing

    the_instance, atrs = build_instance(problem)
    feasibility, routes, _ = gurobi_routing(the_instance)
    if feasibility != sat:
        raise NoSolution(
            f"E_Routing_Gurobi found no feasible task-to-vehicle assignment for {problem!r}")

    # Keep only the real job visits -- 'start'/'end' are depot bookends E_Routing_Gurobi adds
    # to close each route, and AOC-CBS has no notion of a battery so 'recharge' stops (whose
    # Service field is always 0 -- the actual charging duration lives in the route's separate
    # charging_time output, which nothing here consumes) would just be inert waypoints.
    chains = {robot_id: [] for robot_id in atrs}
    for route in routes:
        chains[route.vehicle.id] = [
            (task.location, float(task.Service))
            for task in route.tasks
            if task.task_type not in ('start', 'end', 'recharge')
        ]
    return chains


def _build_problem_config(chains: dict, atrs: dict, agent_model: str, state_graph: str) -> ProblemConfig:
    agents = tuple(
        NamedAgent(
            name=robot_id,
            config=AgentConfig(
                agent_model=agent_model,
                state_graph=state_graph,
                start_state=atrs[robot_id],
                tasks=tuple(chains[robot_id]),
            ),
        )
        for robot_id in sorted(chains)
    )
    return ProblemConfig(agents=agents)


def _solve(problem_config: ProblemConfig, solver_config: SolverConfig, verbose: bool = True,
           first_solution_only: bool = False) -> tuple:
    """`first_solution_only` stops as soon as `solver.best_solution` is set (any feasible joint
    plan, however far from optimal), rather than running AOC-CBS's normal anytime loop out to its
    `optimality_gap`/`timelimit` stopping point. There is no such mode in the vendored library
    itself -- `SolverConfig.optimality_gap` is a *relative* gap against the current lower bound,
    not reachable in general right after the first solution, so this is implemented as an extra
    break condition here rather than by tuning that config field.
    """
    run_config = RunConfig(solver_config=solver_config, problem_config=problem_config)
    solver = AOCCBSSolver(extract_plant_spec(run_config))
    try:
        solver.reset_to_problem(extract_problem_spec(run_config))

        t0 = time.time()
        gap_target = solver_config.optimality_gap
        timelimit = solver_config.timelimit
        while not solver.found_optimal:
            if solver.optimality_gap <= gap_target:
                break
            if timelimit is not None and time.time() - t0 > timelimit:
                break
            solver.iterate()
            if first_solution_only and solver.best_solution is not None:
                break

        if verbose:
            if solver.found_optimal:
                status = 'optimal'
            elif first_solution_only and solver.best_solution is not None:
                status = f'first solution, gap {solver.optimality_gap:.3f}'
            else:
                status = f'gap {solver.optimality_gap:.3f}'
            print(f"AOC-CBS: {solver.iterations} iterations, {time.time() - t0:.2f}s, {status}")

        if solver.best_solution is None:
            raise NoSolution(
                f"AOC-CBS found no solution within the time limit ({solver_config.timelimit}s)")

        return solver.best_solution, solver.stats
    finally:
        solver.close()


def _extract_schedule(joint_plan, agent_map: dict, state_graph) -> dict:
    """Turn a solved JointPlan into {robot_id: [(node_id, eta), ...]}.

    A WaitAction (parking, or a task's service time) never changes the robot's vertex, so only
    MoveAction endpoints are recorded -- the same convention `occbs.solution.parse_log` uses.
    """
    schedule = {}
    for robot_id, plan in joint_plan.plans.items():
        timetable = [(agent_map[robot_id].start_state, 0.0)]
        for action in plan.actions:
            if action.is_move:
                timetable.append((state_graph.edge_target(action.edge), action.t_end))
        schedule[robot_id] = timetable
    return schedule


def AOCCBS(problem: str, agent_radius: float = DEFAULT_AGENT_RADIUS,
          solver_overrides: dict = None, workers: int = None, verbose: bool = True,
          assign_via_routing: bool = False, first_solution_only: bool = False,
          timeout: float = None, optimality_gap: float = 0.0) -> tuple:
    """Entry point mirroring `OCCBS`/`Compo_slim`: returns (solution, stats).

    `assign_via_routing` lifts the usual "every job already pinned to one robot" restriction
    by running sp_comsat's Gurobi routing sub-solver first to decide the assignment -- see
    `_robot_task_specs_via_routing`.

    `first_solution_only` returns as soon as AOC-CBS finds any feasible joint plan instead of
    running its normal anytime search out to `solver_overrides['optimality_gap']`/`'timelimit'`
    (defaulted below to 0.0/60s) -- see `_solve`.

    `timeout` sets that default `'timelimit'` (seconds); it is overridden by an explicit
    `solver_overrides['timelimit']` if both are given. `_solve` raises `NoSolution` if AOC-CBS
    has found nothing by the time it hits this limit.

    `optimality_gap` is the *relative* gap against the current lower bound at which the anytime
    search stops early. It defaults to 0.0, i.e. stop only on a proven optimum, so a run that
    cannot close the gap always spends its whole `timeout` -- on crowded instances the last
    percent of the bound is far more expensive than the plan quality it buys, so raise it (0.01
    is a reasonable start) when a good schedule sooner is worth more than a proven one later.

    `agent_radius` is how much room the plan leaves between robots: AOC-CBS keeps their discs
    from overlapping, so robot centres stay at least `2*agent_radius` apart. Raise it to get a
    schedule the NMPC does not have to deviate from to stay safe -- see
    `mpc_matched_agent_radius`, which returns the value matching the tracker's own fleet safe
    distance.

    The search runs `DEFAULT_SEARCH_PORTFOLIO`, which is AOC-CBS's own portfolio with its TPR
    repair heuristic switched on; without it the library finds no solution at all on instances
    with many agents. Pass `solver_overrides['search_portfolio']` to override.
    """
    with open(f"{PROJECT_ROOT}/data/test_cases/{problem}.json") as f:
        data = json.load(f)

    sg_id = _build_state_graph(problem, data['test_data']['nodes'], verbose=verbose)
    am_id = create_circular_agent(agent_radius)

    workers = workers or os.cpu_count() or 1
    # AOC-CBS's own cache/progress prints (state_graph_distances.py, intersection_intervals.py)
    # are not gated by any flag it exposes, so silence them here when quiet rather than
    # patching the vendored library.
    stdout_ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with stdout_ctx:
        state_graph_distances.ensure_state_graph_distances(sg_id, workers=workers)
        intersection_intervals.ensure_intersection_intervals(sg_id, am_id, sg_id, am_id, workers=workers)

    chains = _robot_task_specs_via_routing(problem) if assign_via_routing else _robot_task_specs(data)
    problem_config = _build_problem_config(chains, data['ATRs'], am_id, sg_id)

    solver_config = SolverConfig(**{
        'timelimit': timeout if timeout is not None else 60.0,
        'optimality_gap': optimality_gap,
        'verbosity': 'summary' if verbose else 'silent',
        'search_portfolio': DEFAULT_SEARCH_PORTFOLIO,
        **(solver_overrides or {}),
    })

    best_solution, stats = _solve(problem_config, solver_config, verbose=verbose,
                                   first_solution_only=first_solution_only)

    state_graph = load_state_graph(sg_id)
    solution = _extract_schedule(best_solution, problem_config.agent_map, state_graph)
    return solution, stats
