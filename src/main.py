import os
import pathlib
import json
import csv
import math
import sys
import time

from status_log import status


project_root = pathlib.Path(__file__).resolve().parents[1]
src_path = os.path.join(project_root, "src")
data_path = os.path.join(project_root, "data")

# Backends whose solution's node_id is confirmed to be a raw key into a test case's
# test_data['nodes'] dict (see Compo_slim.py, pkg_sche.aoccbs.runner, pkg_sche.pp_sipp.runner) --
# total travel distance is only computed/reported for these.
DISTANCE_SCHEDULER_BACKENDS = ("ComSat", "aoccbs", "pp_sipp")


def compute_total_travel_distance(solution, nodes):
    """Sum of Euclidean segment lengths between consecutive scheduled nodes, over all robots.

    `solution` is {robot_id: [(node_id, ETA), ...]}, as returned by a scheduler backend.
    `nodes` is a test case's `test_data['nodes']` dict (node_id -> {'x':, 'y':, ...}).
    """
    total = 0.0
    for timetable in solution.values():
        for (node_a, _), (node_b, _) in zip(timetable, timetable[1:]):
            xa, ya = nodes[node_a]['x'], nodes[node_a]['y']
            xb, yb = nodes[node_b]['x'], nodes[node_b]['y']
            total += math.hypot(xb - xa, yb - ya)
    return total


def compute_makespan(solution):
    """Latest scheduled arrival across every robot -- the time needed to execute all routes.

    `solution` is {robot_id: [(node_id, ETA), ...]}. Backend-agnostic: every backend's ETA is
    a real, finite arrival time -- ComSat's exported `visit_node` is never forced to the
    scheduling model's `Big_number` sentinel, only its separate (unexported) `leave_node` is.
    """
    return max(eta for timetable in solution.values() for _, eta in timetable)


def compute_sum_of_costs(solution):
    """Sum-of-costs: the sum over robots of each robot's arrival time at its goal.

    `solution` is {robot_id: [(node_id, ETA), ...]}, time-ordered, so a robot's goal
    arrival is the ETA of its last scheduled node. Any waiting a robot does en route is
    therefore already included -- waiting pushes the goal arrival later, which is the
    standard MAPF sum-of-costs convention. Backend-agnostic for the same reason
    `compute_makespan` is: every backend's exported ETA is a real, finite arrival time.
    """
    return sum(timetable[-1][1] for timetable in solution.values() if timetable)


def general_funct(problem, scheduler=True, controller=True, naive_tracker=False, ignore_speed_ref=False, recording=False,
                  scheduler_backend="ComSat", mpc_backend=None, assign_via_routing=False,
                  first_solution_only=False, headless=False, late_threshold_s=30.0, stuck_timeout_s=30.0,
                  collision_check=True, collision_margin=0.0, verbose=False, show_initial_state=False,
                  scheduler_timeout_s=None, scheduler_optimality_gap=0.0, agent_radius=None,
                  coordinator=False, coordinator_overrides=None):
    """
    verbose: If False (default), the scheduler and MPC loop only print a handful of
        timestamped status lines (scheduler executing/done/UNSAT, MPC executing/done).
        If True, both layers additionally print their normal per-iteration/per-tick
        diagnostics (CEGAR loop status, AOC-CBS cache/build lines, the MPC's per-tick
        reference/cost prints, work-mode transitions, ...).
    show_initial_state: If True, pop up a plot of the map, the roadmap graph overlaid on it,
        and each robot's start node -- plus its final node, when that's determinable without
        running the scheduler -- as soon as this function is called, before the scheduler (or
        anything else) starts computing. Blocks until the plot window is closed.
    scheduler_timeout_s: If given, an overall wall-clock budget (seconds) for whichever
        scheduler_backend runs -- "ComSat", "aoccbs", or "pp_sipp" (not "occbs", which exposes
        none). For "ComSat" this bounds the whole CEGAR loop: each Gurobi/Z3 sub-solver call is
        capped at whatever's left of the budget when it starts, not at the budget itself, so the
        loop's several route-set attempts can't multiply the total past `scheduler_timeout_s`
        (see `Compo_slim`'s docstring). For "aoccbs" it is AOC-CBS's own anytime-search
        `timelimit`. For "pp_sipp" it is checked between robots in the priority sweep. `None`
        (default) leaves each backend at its own default (uncapped for "ComSat", 60s for
        "aoccbs", uncapped for "pp_sipp").
    scheduler_optimality_gap: "aoccbs" only -- the relative gap at which its anytime search
        stops early. 0.0 (default) means stop only on a proven optimum, so a run that cannot
        close the gap always spends its whole `scheduler_timeout_s`. On crowded instances the
        upper bound is usually within a fraction of a percent within seconds and then barely
        moves, so a small value (0.01) buys back nearly the whole budget for almost no plan
        quality. Ignored by every other backend.
    agent_radius: "aoccbs"/"pp_sipp" only -- how much room the schedule leaves between robots.
        Both backends model a robot as a disc and forbid overlap, so the plan keeps robot centres
        at least 2*agent_radius apart; this is the only clearance knob, and padding the emitted
        ETAs instead would not move where two robots pass each other. Pass a radius in metres,
        the string "mpc" to use the one matching the NMPC's own fleet safe distance
        (vehicle_width + vehicle_margin = 0.554 m, so 1.107 m between centres -- see
        pkg_sche.aoccbs.runner.mpc_matched_agent_radius), or None (default) to keep the backends'
        own 0.35 m, which is roughly the robot's bare body radius and so plans passes the NMPC
        then has to widen by deviating from the schedule. Changing it makes the first run pay
        once for a fresh AOC-CBS intersection-intervals cache.
    coordinator: If True, run the coordination layer between the schedule and the MPC
        trackers. It compares each robot's measured progress against its scheduled arrival
        times and, when drift is about to put two robots at the same node at once, holds the
        robot the schedule put second, sidesteps it, or re-plans it with SIPP -- leaving
        every other robot's schedule untouched. Off by default and fully inert when off.
    coordinator_overrides: Optional dict of `pkg_coordinator.CoordinatorConfig` field
        overrides, e.g. {"enable_crossing": False, "hold_timeout_s": 4.0}.
    """
    if show_initial_state:
        from pkg_motion_plan.initial_state_plot import plot_initial_state
        plot_initial_state(problem)

    total_travel_distance = None
    makespan = None
    sum_of_costs = None

    if scheduler:
        status(f"Scheduler executing ({scheduler_backend}, problem={problem!r})")
        # Resolved here rather than at the top of the function so that neither the "mpc" lookup
        # nor the AOC-CBS import it needs happens on a run that never reaches those backends.
        if agent_radius == "mpc" and scheduler_backend in ("aoccbs", "pp_sipp"):
            from pkg_sche.aoccbs.runner import mpc_matched_agent_radius
            agent_radius = mpc_matched_agent_radius()
            status(f"agent_radius resolved to {agent_radius:.5f} m "
                   f"({2*agent_radius:.3f} m planned clearance between robot centres)")
        radius_kwargs = {} if agent_radius is None else {'agent_radius': agent_radius}
        if scheduler_backend == "ComSat":
            from pkg_sche.sp_comsat.Compo_slim import Compo_slim
            instance, optimum, running_time, len_previous_routes, paths_changed, solution = Compo_slim(
                problem, verbose=verbose, timeout=scheduler_timeout_s)
        elif scheduler_backend == "occbs":
            from pkg_sche.occbs.runner import OCCBS
            solution, _ = OCCBS(problem, verbose=verbose)
        elif scheduler_backend == "aoccbs":
            from pkg_sche.aoccbs.runner import AOCCBS
            solution, _ = AOCCBS(problem, assign_via_routing=assign_via_routing,
                                  first_solution_only=first_solution_only, verbose=verbose,
                                  timeout=scheduler_timeout_s,
                                  optimality_gap=scheduler_optimality_gap, **radius_kwargs)
        elif scheduler_backend == "pp_sipp":
            from pkg_sche.pp_sipp.runner import PP_SIPP
            solution, _ = PP_SIPP(problem, assign_via_routing=assign_via_routing, verbose=verbose,
                                   timeout=scheduler_timeout_s, **radius_kwargs)
        else:
            raise ValueError(f"unknown scheduler_backend {scheduler_backend!r}")

        status(f"Scheduler done: {'SAT' if solution else 'UNSAT'}")
        if not solution:
            return {"status": "no_schedule", "problem": problem}

        # save the schedule (I don't actually need this step, but it is more readable than the csv)
        with open(f"{src_path}/pkg_sche/MPC_input.json",'w') as logfile:
            json.dump(solution, logfile, indent=4)

        with open(f"{data_path}/schedule_demo2_data/schedule.csv", mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["robot_id", "node_id", "ETA"])
            for robot_id, nodes in solution.items():
                for node_id, eta in nodes:
                    csv_writer.writerow([robot_id, node_id, eta])
        with open(f"{data_path}/test_cases/{problem}.json",'r') as read_file:
            data = json.load(read_file)
            ATRs = data['ATRs']
            node_coords = data['test_data']['nodes']
        robot_starts = {
            key:[
                node_coords[value]['x'],
                node_coords[value]['y'],
                -1.57
            ]
            for key,value in ATRs.items()
        }
        with open(f"{data_path}/schedule_demo2_data/robot_start.json", 'w') as write_file:
            json.dump(robot_starts, write_file, indent=4)

        if scheduler_backend in DISTANCE_SCHEDULER_BACKENDS:
            total_travel_distance = compute_total_travel_distance(solution, node_coords)
            status(f"Total travel distance ({scheduler_backend}): {total_travel_distance:.2f}")

        makespan = compute_makespan(solution)
        status(f"Makespan ({scheduler_backend}): {makespan:.2f}")

        sum_of_costs = compute_sum_of_costs(solution)
        status(f"Sum-of-costs ({scheduler_backend}): {sum_of_costs:.2f}")

    if controller:
        from run_mpc import run_mpc
        with open(f"{data_path}/test_cases/{problem}.json",'r') as read_file:
            data = json.load(read_file)
            EnvFolder = data['test_data']['Environment']
        sim_start = time.perf_counter()
        result = run_mpc(EnvFolder, problem, naive_tracker=naive_tracker, ignore_speed_ref=ignore_speed_ref,
                recording=recording, mpc_backend=mpc_backend, headless=headless,
                late_threshold_s=late_threshold_s, stuck_timeout_s=stuck_timeout_s,
                collision_check=collision_check, collision_margin=collision_margin, verbose=verbose,
                coordinator=coordinator, coordinator_overrides=coordinator_overrides,
                scheduler_backend=scheduler_backend, assign_via_routing=assign_via_routing)
        simulation_runtime_s = time.perf_counter() - sim_start
        status(f"MPC simulation wall-clock runtime: {simulation_runtime_s:.2f}s")
        result["simulation_runtime_s"] = simulation_runtime_s
        if total_travel_distance is not None:
            result["total_travel_distance"] = total_travel_distance
        if makespan is not None:
            result["makespan"] = makespan
        if sum_of_costs is not None:
            result["sum_of_costs"] = sum_of_costs
        return result
    return None

if __name__ == "__main__":
    # problem = '4Small' # SAFETY COEFF 20
    # problem = '4SmallNu' # 4Small's graph, one destination per robot (single-goal MAPF)
    # problem = "10Large"
    # problem = 'ccbs_sparse_1_4'
    # problem = 'movingai_empty16_1_8'
    # problem = 'test_4' # why do agents go to a THIRD location?
    result = general_funct(
        sys.argv[1],
        scheduler = True,
        controller= False,
        naive_tracker= False, # True = proportional baseline, False = NMPC (see mpc_backend)
        ignore_speed_ref= False,
        recording= False,
        scheduler_backend= "aoccbs", # "ComSat", "occbs", "aoccbs", or "pp_sipp"
        scheduler_optimality_gap= 0.0, # aoccbs only: relative gap at which the anytime search
                              # stops early. 0.0 always spends the whole scheduler_timeout_s;
                              # 0.01 typically returns in seconds at near-identical quality
        scheduler_timeout_s= None, # timeout (seconds) for "ComSat"/"aoccbs"/"pp_sipp" (not
                              # "occbs", which has none) -- see general_funct's docstring for
                              # what it means on each backend. None = each backend's own default.
        assign_via_routing= False, # aoccbs only: use ComSat's Gurobi routing sub-solver to
                              # assign jobs to robots first, instead of requiring every job
                              # pre-pinned to one ATR (see pkg_sche.aoccbs.runner)
        first_solution_only= False, # aoccbs only: stop at the first feasible joint plan instead
                              # of running the normal anytime search out to optimality/timelimit
        agent_radius= None,   # aoccbs/pp_sipp only: robot disc radius the scheduler plans with,
                              # so the plan keeps robot centres 2*agent_radius apart. None =
                              # the backends' 0.35 m (about the bare body radius); "mpc" = the
                              # radius matching the NMPC's fleet safe distance (0.554 m, i.e.
                              # 1.107 m between centres), which is what to use if robots pass
                              # each other too closely for the tracker to follow the schedule.
        mpc_backend= "panoc", # "casadi" (IPOPT, no build step); "panoc" or "panoc_light" (both
                              # need build_solver.py, with panoc_builder set to match -- see
                              # build_solver.py); None falls back to solver_type in config/mpc_fast.yaml
        headless= False, # True = no matplotlib window, no blocking prompt at the end; run
                              # non-interactively and just return/print a status dict --
                              # see late_threshold_s/stuck_timeout_s below for failure detection
        late_threshold_s= False, # fail the run once a robot is still short of the node it is
                              # targeting more than this many seconds past that node's
                              # scheduled ETA. None disables the check.
        stuck_timeout_s= False, # fail the run once a robot has not translated more than a couple
                              # centimetres for this many consecutive seconds while active
                              # (excluding in-place `aligning` rotation). None disables the check.
        collision_check= False, # fail the run as soon as two robot bodies overlap, or a robot body
                              # overlaps a static obstacle or leaves the map boundary. Robots are
                              # discs of radius `vehicle_width` (config/robot_spec.yaml) and the
                              # test uses the un-inflated map, so this is physical contact, not a
                              # breach of the planner's safety margin. False disables the check.
        collision_margin= False, # extra clearance (metres) required on top of the body radius before
                              # the collision check trips: 0.0 = bodies must actually touch,
                              # positive values also fail on near-misses (0.1 = closer than 10 cm).
        verbose= True,        # True = restore the scheduler's/MPC's full per-iteration/per-tick
                              # console output; False = just the timestamped status lines
                              # (scheduler executing/done/UNSAT, MPC executing/done) -- handy
                              # when running several instances back to back.
        show_initial_state= False, # True = pop up a plot of the map, graph, and each robot's
                              # start/goal markers as soon as this runs, before the scheduler
                              # starts computing. Blocks until the plot window is closed.
        coordinator= False,   # True = run the coordinator between the schedule and the MPC:
                              # it watches each robot's measured progress against its
                              # scheduled arrival times and, when drift is about to put two
                              # robots on the same node at once, holds the robot the schedule
                              # put second -- escalating to a lateral sidestep and then to a
                              # SIPP replan of that one robot if holding does not clear it.
                              # Writes data/schedule_demo2_data/Coordinator_<problem>.csv.
        coordinator_overrides= None, # dict of CoordinatorConfig fields to override, e.g.
                              # {"enable_crossing": False, "enable_replan": False} to run the
                              # hold tier alone, or {"enable_hold": False, "enable_crossing":
                              # False, "enable_replan": False} to detect and log only.
    )
    if result is not None and result["status"] != "success":
        raise SystemExit(f"[main] run failed: {result}")
