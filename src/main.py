import os
import pathlib
import json
import csv


project_root = pathlib.Path(__file__).resolve().parents[1]
src_path = os.path.join(project_root, "src")
data_path = os.path.join(project_root, "data")


def general_funct(problem, scheduler=True, controller=True, naive_tracker=False, ignore_speed_ref=False, recording=False):
    if scheduler:
        from pkg_sche.sp_comsat.Compo_slim import Compo_slim
        instance, optimum, running_time, len_previous_routes, paths_changed, solution = Compo_slim(problem)
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
        run_mpc(EnvFolder, naive_tracker=naive_tracker, ignore_speed_ref=ignore_speed_ref, recording=recording)

if __name__ == "__main__":
    problem = '4Small' # SAFETY COEFF 20
    # problem = "10Large"

    general_funct(
        problem,
        scheduler = True,
        controller= True,
        naive_tracker= False,
        ignore_speed_ref= False,
        recording=False
    )


