import os
import pathlib
import json
import csv
import sys


project_root = pathlib.Path(__file__).resolve().parents[1]
src_path = os.path.join(project_root, "src")
data_path = os.path.join(project_root, "data")


def general_funct(problem, scheduler=True, controller=True, naive_tracker=False, ignore_speed_ref=False, recording=False,
                  scheduler_backend="sp_comsat", mpc_backend=None, assign_via_routing=False,
                  first_solution_only=False, headless=False, late_threshold_s=30.0, stuck_timeout_s=30.0,
                  collision_check=True, collision_margin=0.0):
    if scheduler:
        if scheduler_backend == "sp_comsat":
            from pkg_sche.sp_comsat.Compo_slim import Compo_slim
            instance, optimum, running_time, len_previous_routes, paths_changed, solution = Compo_slim(problem)
        elif scheduler_backend == "occbs":
            from pkg_sche.occbs.runner import OCCBS
            solution, _ = OCCBS(problem)
        elif scheduler_backend == "aoccbs":
            from pkg_sche.aoccbs.runner import AOCCBS
            solution, _ = AOCCBS(problem, assign_via_routing=assign_via_routing,
                                  first_solution_only=first_solution_only)
        else:
            raise ValueError(f"unknown scheduler_backend {scheduler_backend!r}")
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
                collision_check=collision_check, collision_margin=collision_margin)
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
        scheduler_backend= "sp_comsat", # "sp_comsat", "occbs", or "aoccbs"
        assign_via_routing= False, # aoccbs only: use sp_comsat's Gurobi routing sub-solver to
                              # assign jobs to robots first, instead of requiring every job
                              # pre-pinned to one ATR (see pkg_sche.aoccbs.runner)
        first_solution_only= False, # aoccbs only: stop at the first feasible joint plan instead
                              # of running the normal anytime search out to optimality/timelimit
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
    )
    if result is not None and result["status"] != "success":
        raise SystemExit(f"[main] run failed: {result}")


