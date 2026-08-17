import os
import pathlib
import json
import csv


project_root = pathlib.Path(__file__).resolve().parents[1]
src_path = os.path.join(project_root, "src")
data_path = os.path.join(project_root, "data")


def general_funct(problem, scheduler=True, controller=True, naive_tracker=False, ignore_speed_ref=False, recording=False,
                  scheduler_backend="sp_comsat", mpc_backend=None, assign_via_routing=False):
    if scheduler:
        if scheduler_backend == "sp_comsat":
            from pkg_sche.sp_comsat.Compo_slim import Compo_slim
            instance, optimum, running_time, len_previous_routes, paths_changed, solution = Compo_slim(problem)
        elif scheduler_backend == "occbs":
            from pkg_sche.occbs.runner import OCCBS
            solution, _ = OCCBS(problem)
        elif scheduler_backend == "aoccbs":
            from pkg_sche.aoccbs.runner import AOCCBS
            solution, _ = AOCCBS(problem, assign_via_routing=assign_via_routing)
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
        run_mpc(EnvFolder, problem, naive_tracker=naive_tracker, ignore_speed_ref=ignore_speed_ref,
                recording=recording, mpc_backend=mpc_backend)

if __name__ == "__main__":
    problem = '4Small' # SAFETY COEFF 20
    # problem = '4SmallNu' # 4Small's graph, one destination per robot (single-goal MAPF)
    # problem = "10Large"

    general_funct(
        problem,
        scheduler = False,
        controller= False,
        naive_tracker= False, # True = proportional baseline, False = NMPC (see mpc_backend)
        ignore_speed_ref= False,
        recording= False,
        scheduler_backend= "sp_comsat", # "sp_comsat", "occbs", or "aoccbs"
        assign_via_routing= False, # aoccbs only: use sp_comsat's Gurobi routing sub-solver to
                              # assign jobs to robots first, instead of requiring every job
                              # pre-pinned to one ATR (see pkg_sche.aoccbs.runner)
        mpc_backend= "panoc_light" # "casadi" (IPOPT, no build step); "panoc" or "panoc_light" (both
                              # need build_solver.py, with panoc_builder set to match -- see
                              # build_solver.py); None falls back to solver_type in config/mpc_fast.yaml
    )


