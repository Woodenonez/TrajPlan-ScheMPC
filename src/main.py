import os
import pathlib
import json
import csv
import sys

from status_log import status


project_root = pathlib.Path(__file__).resolve().parents[1]
src_path = os.path.join(project_root, "src")
data_path = os.path.join(project_root, "data")


def general_funct(problem, scheduler=True, controller=True, naive_tracker=False, ignore_speed_ref=False, recording=False,
                  scheduler_backend="ComSat", mpc_backend=None, assign_via_routing=False,
                  first_solution_only=False, headless=False, late_threshold_s=30.0, stuck_timeout_s=30.0,
                  collision_check=True, collision_margin=0.0, verbose=False, show_initial_state=False):
    """
    verbose: If False (default), the scheduler and MPC loop only print a handful of
        timestamped status lines (scheduler executing/done/UNSAT, MPC executing/done).
        If True, both layers additionally print their normal per-iteration/per-tick
        diagnostics (CEGAR loop status, AOC-CBS cache/build lines, the MPC's per-tick
        reference/cost prints, work-mode transitions, ...).
    show_initial_state: If True (and controller and not headless), pause right after the map,
        roadmap graph, and each robot's start/goal markers are drawn, before the simulation
        starts, so the initial state can be inspected in the plot window. Requires the
        controller to run, since the plot lives inside run_mpc.
    """
    if scheduler:
        status(f"Scheduler executing ({scheduler_backend}, problem={problem!r})")
        if scheduler_backend == "ComSat":
            from pkg_sche.sp_comsat.Compo_slim import Compo_slim
            instance, optimum, running_time, len_previous_routes, paths_changed, solution = Compo_slim(problem, verbose=verbose)
        elif scheduler_backend == "occbs":
            from pkg_sche.occbs.runner import OCCBS
            solution, _ = OCCBS(problem, verbose=verbose)
        elif scheduler_backend == "aoccbs":
            from pkg_sche.aoccbs.runner import AOCCBS
            solution, _ = AOCCBS(problem, assign_via_routing=assign_via_routing,
                                  first_solution_only=first_solution_only, verbose=verbose)
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
        robot_starts = {
            key:[
                data['test_data']['nodes'][value]['x'],
                data['test_data']['nodes'][value]['y'],
                -1.57
            ]
            for key,value in ATRs.items()
        }
        with open(f"{data_path}/schedule_demo2_data/robot_start.json", 'w') as write_file:
            json.dump(robot_starts, write_file, indent=4)

    if controller:
        from run_mpc import run_mpc
        with open(f"{data_path}/test_cases/{problem}.json",'r') as read_file:
            data = json.load(read_file)
            EnvFolder = data['test_data']['Environment']
        return run_mpc(EnvFolder, problem, naive_tracker=naive_tracker, ignore_speed_ref=ignore_speed_ref,
                recording=recording, mpc_backend=mpc_backend, headless=headless,
                late_threshold_s=late_threshold_s, stuck_timeout_s=stuck_timeout_s,
                collision_check=collision_check, collision_margin=collision_margin, verbose=verbose,
                show_initial_state=show_initial_state)
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
        scheduler = False,
        controller= True,
        naive_tracker= False, # True = proportional baseline, False = NMPC (see mpc_backend)
        ignore_speed_ref= False,
        recording= False,
        scheduler_backend= "aoccbs", # "ComSat", "occbs", or "aoccbs"
        assign_via_routing= False, # aoccbs only: use ComSat's Gurobi routing sub-solver to
                              # assign jobs to robots first, instead of requiring every job
                              # pre-pinned to one ATR (see pkg_sche.aoccbs.runner)
        first_solution_only= False, # aoccbs only: stop at the first feasible joint plan instead
                              # of running the normal anytime search out to optimality/timelimit
        mpc_backend= "panoc", # "casadi" (IPOPT, no build step); "panoc" or "panoc_light" (both
                              # need build_solver.py, with panoc_builder set to match -- see
                              # build_solver.py); None falls back to solver_type in config/mpc_fast.yaml
        headless= True, # True = no matplotlib window, no blocking prompt at the end; run
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
        verbose= False, # True = restore the scheduler's/MPC's full per-iteration/per-tick
                              # console output; False = just the timestamped status lines
                              # (scheduler executing/done/UNSAT, MPC executing/done) -- handy
                              # when running several instances back to back.
        show_initial_state= False, # True = pause in the plot window right after the map, graph,
                              # and each robot's start/goal markers are drawn, before the
                              # simulation starts -- lets you inspect the initial state before
                              # it runs. Needs controller=True and headless=False.
    )
    if result is not None and result["status"] != "success":
        raise SystemExit(f"[main] run failed: {result}")


