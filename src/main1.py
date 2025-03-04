import os
import json
import pathlib

import numpy as np

from basic_motion_model.motion_model import UnicycleModel

from pkg_motion_plan.global_path_coordinate import GlobalPathCoordinator
from pkg_motion_plan.local_traj_plan import LocalTrajPlanner
from pkg_mpc_tracker.trajectory_tracker import TrajectoryTracker
from pkg_robot.robot import RobotManager

from configs import MpcConfiguration
from configs import CircularRobotSpecification

from visualizer.object import CircularVehicleVisualizer
from visualizer.mpc_plot import MpcPlotInLoop # type: ignore

DATA_NAME = "schedule_demo2_data" # "schedule_demo_data"
CFG_FNAME = "mpc_fast.yaml" # "mpc_default.yaml" or "mpc_fast.yaml"
MAP_ONLY = True
AUTORUN = False # if false, press key (in the plot window) to continue
MONITOR_COST = False # if true, monitor the cost (this will slow down the simulation)
VERBOSE = False
TIMEOUT = 1000

root_dir = pathlib.Path(__file__).resolve().parents[1]
data_dir = os.path.join(root_dir, "data", DATA_NAME)
cnfg_dir = os.path.join(root_dir, "config")

robot_ids = None # if none, read from schedule

### Configurations
config_mpc_path = os.path.join(cnfg_dir, CFG_FNAME)
config_robot_path = os.path.join(cnfg_dir, "robot_spec.yaml")

config_mpc = MpcConfiguration.from_yaml(config_mpc_path)
config_robot = CircularRobotSpecification.from_yaml(config_robot_path)

### Map, graph, and schedule paths
map_path = os.path.join(data_dir, "map.json")
graph_path = os.path.join(data_dir, "graph.json")
schedule_path = os.path.join(data_dir, "schedule.csv")
start_path = os.path.join(data_dir, "robot_start.json")
with open(start_path, "r") as f:
    robot_starts = json.load(f)

### Set up the global path/schedule coordinator
gpc = GlobalPathCoordinator.from_csv(schedule_path)
gpc.load_graph_from_json(graph_path)
gpc.load_map_from_json(map_path, inflation_margin=config_robot.vehicle_width+config_robot.vehicle_margin)
robot_ids = gpc.robot_ids if robot_ids is None else robot_ids
static_obstacles = gpc.inflated_map.obstacle_coords_list

### Set up robots
robot_manager = RobotManager()
for rid in robot_ids:
    robot = robot_manager.create_robot(config_robot, UnicycleModel(sampling_time=config_robot.ts), rid)
    robot.set_state(np.asarray(robot_starts[str(rid)]))
    planner = LocalTrajPlanner(config_mpc.ts, config_mpc.N_hor, config_robot.lin_vel_max, verbose=VERBOSE)
    planner.load_map(gpc.inflated_map.boundary_coords, gpc.inflated_map.obstacle_coords_list)
    controller = TrajectoryTracker(config_mpc, config_robot, robot_id=rid, verbose=VERBOSE)
    controller.load_motion_model(UnicycleModel(sampling_time=config_mpc.ts))
    controller.set_monitor(monitor_on=MONITOR_COST)
    visualizer = CircularVehicleVisualizer(config_robot.vehicle_width, indicate_angle=True)
    robot_manager.add_robot(robot, controller, planner, visualizer)

    path_coords, path_times = gpc.get_robot_schedule(rid)
    robot_manager.add_schedule(rid, np.asarray(robot_starts[str(rid)]), path_coords, path_times)

### Run
main_plotter = MpcPlotInLoop(config_robot, map_only=MAP_ONLY)
# main_plotter.plot_in_loop_pre(gpc.current_map, gpc.inflated_map, gpc.current_graph)
main_plotter.plot_in_loop_pre(gpc.current_map, graph_manager=gpc.current_graph)
color_list = [
    "#0072B2", "#D55E00", "#009E73", "#F0E442", "#56B4E9", 
    "#E69F00", "#CC79A7", "#0072B2", "#D55E00", "#009E73", 
    "#F0E442", "#56B4E9", "#E69F00", "#CC79A7"
]
for i, rid in enumerate(robot_ids):
    planner = robot_manager.get_planner(rid)
    controller = robot_manager.get_controller(rid)
    visualizer = robot_manager.get_visualizer(rid)
    main_plotter.add_object_to_pre(rid,
                                   None, #planner.ref_traj,
                                   controller.state,
                                   controller.final_goal,
                                   color=color_list[i])
    visualizer.plot(main_plotter.map_ax, *robot.state)

for kt in range(TIMEOUT):
    robot_states = []
    incomplete = False
    for i, rid in enumerate(robot_ids):
        robot = robot_manager.get_robot(rid)
        planner = robot_manager.get_planner(rid)
        controller = robot_manager.get_controller(rid)
        visualizer = robot_manager.get_visualizer(rid)
        other_robot_states = robot_manager.get_other_robot_states(rid, config_mpc)

        ref_states, ref_speed, *_ = planner.get_local_ref(kt*config_mpc.ts, (float(robot.state[0]), float(robot.state[1])) )
        print(f"Robot {rid} ref speed: {round(ref_speed, 4)}") # XXX
        controller.set_current_state(robot.state)
        controller.set_ref_states(ref_states, ref_speed=ref_speed)
        (actions, pred_states, current_refs, debug_info) = controller.run_step(static_obstacles=static_obstacles,
                                                       full_dyn_obstacle_list=None,
                                                       other_robot_states=other_robot_states,
                                                       map_updated=True, report_cost=False)
        controller.report_cost(debug_info['cost'], 
                               debug_info['step_runtime'], 
                               debug_info['monitored_cost'],
                               object_id=f"Robot {rid}")

        ### Real run
        robot.step(actions[-1])
        robot_manager.set_pred_states(rid, np.asarray(pred_states))

        main_plotter.update_plot(rid, kt, actions[-1], None, debug_info['cost'], np.asarray(pred_states), current_refs)
        visualizer.update(*robot.state)

        if not controller.check_termination_condition(external_check=planner.idle):
            incomplete = True

        robot_states.append(robot.state)
    
    main_plotter.plot_in_loop(time=kt*config_mpc.ts, autorun=AUTORUN, zoom_in=None)
    if not incomplete:
        break
    
    
main_plotter.show()
input('Press anything to finish!')
main_plotter.close()

if MONITOR_COST: # XXX
    import matplotlib.pyplot as plt # type: ignore
    fig, ax = plt.subplots(1, 1)
    solve_time = controller.solver_time_timelist
    ax.plot(solve_time, label="Solve time")
    ax.set_title(f"Solve time for Robot {rid}")
    ax.legend()
    plt.show()
