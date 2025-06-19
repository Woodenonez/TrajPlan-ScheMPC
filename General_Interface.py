from Scheduler_MPC.sp_comsat.Compo_slim import Compo_slim
from src.main1 import run_mpc
import json
import csv


def general_funct(problem,scheduler = True,MPC = True):
    if scheduler:
        # run the scheduler from the
        instance,optimum,running_time,len_previous_routes,paths_changed, solution = Compo_slim(problem)
        # save the schedule (I don't actually need this step, but it is more readable than the csv)
        with open(f"Scheduler_MPC/MPC_input.json",'w') as logfile:
            json.dump(solution,logfile,indent=4)

        # convert the schedule to csv
        # Open CSV file for writing
        with open('data/schedule_demo2_data/schedule.csv', mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write header
            csv_writer.writerow(["robot_id", "node_id", "ETA"])

            # Process JSON data and write to CSV
            for robot_id, nodes in solution.items():
                for node_id, eta in nodes:
                    csv_writer.writerow([robot_id, node_id, eta])
        # load vehicles starting nodes from json
        with open(f'data/test_cases/{problem}.json','r') as read_file:
            data = json.load(read_file)
            ATRs = data['ATRs']
        # create a dict with vehicles starting coordinates
        robot_starts = {
            key:[
                data['test_data']['nodes'][value]['x'],
                data['test_data']['nodes'][value]['y'],
                -1.57
            ]
            for key,value in ATRs.items()
        }
        # upload json file for the MPC
        with open('data/schedule_demo2_data/robot_start.json','w') as write_file:
            json.dump(robot_starts,write_file,indent=4)
    if MPC:
        with open(f'data/test_cases/{problem}.json','r') as read_file:
            data = json.load(read_file)
            EnvFolder = data['test_data']['Environment']
        # run the MPC
        run_mpc(EnvFolder)

if __name__ == "__main__":
    problem = '4Small' # SAFETY COEFF 20

    general_funct(
        problem,
        scheduler = True,
        MPC= True
    )


