import os
import json
import pathlib
import datetime

import numpy as np
from shapely.geometry import Point, Polygon # type: ignore
from shapely.ops import unary_union # type: ignore

from basic_motion_model.motion_model import UnicycleModel

from pkg_motion_plan import GlobalPathCoordinator
from pkg_motion_plan import LocalTrajPlanner
from pkg_motion_plan import logger_from_schedule
from pkg_mpc_tracker import TrajectoryTracker
from pkg_robot.robot import RobotManager

from configs import MpcConfiguration
from configs import CircularRobotSpecification
from configs import panoc_light_optimizer_name

from visualizer.object import CircularVehicleVisualizer
from visualizer.mpc_plot import MpcPlotInLoop # type: ignore

### NMPC backends selectable at run time. Keys are what callers pass as `mpc_backend`;
### values are the exact strings `TrajectoryTracker` switches on.
MPC_BACKENDS = {'casadi': 'Casadi', 'panoc': 'PANOC', 'panoc_light': 'PANOC_LIGHT'}

# solver_type values that need a compiled OpEn/PANOC solver imported at run time -- as
# opposed to 'Casadi', whose NLP is built in-process. 'PANOC' and 'PANOC_LIGHT' differ only
# in which builder produced the compiled solver (see build_solver.py); the runtime call
# contract (parameter-vector layout, `solver.run(p=...)`) is identical for both.
PANOC_SOLVER_TYPES = {'PANOC', 'PANOC_LIGHT'}


def resolve_mpc_backend(config_mpc: MpcConfiguration, mpc_backend=None, root_dir=None) -> str:
    """Apply the runtime backend choice on top of the config file, and sanity-check it.

    Args:
        config_mpc: Loaded MPC configuration. Its `solver_type` (and, for 'panoc_light',
            `optimizer_name`) are overwritten in place.
        mpc_backend: 'casadi', 'panoc', 'panoc_light', or None to keep whatever the YAML says.
        root_dir: Project root, used to check that the PANOC solver has been built.

    Returns:
        The resolved solver type, i.e. the new value of `config_mpc.solver_type`.

    Raises:
        ValueError: The backend name is not recognised.
        FileNotFoundError: A PANOC variant was selected but its compiled solver is missing.
    """
    if mpc_backend is not None:
        try:
            config_mpc.solver_type = MPC_BACKENDS[str(mpc_backend).strip().lower()]
        except KeyError:
            raise ValueError(f"unknown mpc_backend {mpc_backend!r}, "
                             f"expected one of {sorted(MPC_BACKENDS)}") from None

    # 'panoc_light' is built from the same config file as 'panoc' (see build_solver.py), so
    # the only thing distinguishing the two compiled solvers is this derived name.
    if config_mpc.solver_type == 'PANOC_LIGHT':
        config_mpc.optimizer_name = panoc_light_optimizer_name(config_mpc.optimizer_name)

    # PANOC (either variant) needs the Rust crate compiled by build_solver.py; failing here
    # is far clearer than the ImportError raised deep inside the tracker when the .so is missing.
    if config_mpc.solver_type in PANOC_SOLVER_TYPES and root_dir is not None:
        solver_path = os.path.join(root_dir, config_mpc.build_directory, config_mpc.optimizer_name)
        if not os.path.isdir(solver_path):
            raise FileNotFoundError(
                f"The {config_mpc.solver_type} backend needs a compiled solver at '{solver_path}', "
                f"which does not exist. Run 'python src/build_solver.py' from the project root (with "
                f"panoc_builder set to match, and the same config file as run_mpc), or switch to "
                f"mpc_backend='casadi', which needs no build step."
            )
    return config_mpc.solver_type


class StaticCollisionChecker:
    """Distance from a robot's centre to the nearest static obstacle surface or wall.

    Built from the *un-inflated* map, so what it returns is physical geometry rather
    than the planning-time inflation the MPC sees: a circular robot of radius `r`
    centred at `(x, y)` overlaps something static exactly when `min_distance(x, y) < r`.
    The value is negative when the centre is already inside an obstacle or outside the
    map boundary, so a deep overlap reads as a large negative number.
    """

    def __init__(self, boundary_coords, obstacle_coords_list):
        polygons = []
        for coords in (obstacle_coords_list or []):
            if coords is None or len(coords) < 3:
                continue
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0) # self-intersecting rings are common in hand-drawn maps
            if not poly.is_empty:
                polygons.append(poly)
        self._obstacles = unary_union(polygons) if polygons else None

        self._boundary = None
        if boundary_coords is not None and len(boundary_coords) >= 3:
            boundary = Polygon(boundary_coords)
            if not boundary.is_valid:
                boundary = boundary.buffer(0)
            if not boundary.is_empty:
                self._boundary = boundary

    @property
    def active(self) -> bool:
        """False if the map carried neither obstacles nor a usable boundary."""
        return self._obstacles is not None or self._boundary is not None

    def min_distance(self, x: float, y: float) -> float:
        """Signed distance from `(x, y)` to the nearest obstacle surface or wall."""
        point = Point(float(x), float(y))
        distance = float('inf')
        if self._obstacles is not None:
            d_obs = point.distance(self._obstacles.boundary)
            if self._obstacles.contains(point):
                d_obs = -d_obs
            distance = min(distance, d_obs)
        if self._boundary is not None:
            d_bnd = point.distance(self._boundary.exterior)
            if not self._boundary.contains(point):
                d_bnd = -d_bnd
            distance = min(distance, d_bnd)
        return distance


def relax_final_eta(path_coords, path_times, lin_vel_max):
    """Replace a sentinel "no deadline" ETA on the last node with a reachable one.

    `sp_comsat` marks the final depot visit with the instance's `Big_number` (50000 in
    `test_7`'s schedule.csv, against a makespan of ~340), meaning "there is no deadline for
    going home". `LocalTrajPlanner` cannot read it that way: its reference speed is
    `distance_to_next_node / (ETA - t)`, so an ETA 50000 s away yields ~1e-4 m/s, and
    `downsample_ref_states` -- which rescales the horizon by ref_speed/nominal_speed --
    then squeezes the whole 4.2 m horizon down to a fraction of a millimetre. Every
    reference point collapses onto the robot's current docking point, `run_mpc`'s
    "within 0.3 m of the last reference" guard below suppresses `robot.step` permanently,
    and because the docking index only advances with the robot's position the reference can
    never move again. Measured on `test_7`/PANOC: A2 and A3 arrested mid-corridor at ticks
    1056 and 947 -- 5.7 m short of their final node -- and stood still until TIMEOUT. That is
    the reported "it simply freezes".

    Nothing is scheduled after the final node, so arriving there as early as the robot can is
    always admissible; `min` guarantees this only ever brings the last ETA forward, never
    delays it, so a schedule with a genuine (finite) final deadline is left untouched.
    """
    if not path_times or len(path_times) < 2 or len(path_coords) != len(path_times):
        return path_times
    leg = float(np.hypot(path_coords[-1][0]-path_coords[-2][0], path_coords[-1][1]-path_coords[-2][1]))
    reachable = float(path_times[-2]) + leg/float(lin_vel_max)
    path_times = list(path_times)
    path_times[-1] = min(float(path_times[-1]), reachable)
    return path_times


def run_mpc(EnvFolder, problem, naive_tracker=False, ignore_speed_ref=False, recording=False, mpc_backend=None,
            headless=False, late_threshold_s=30.0, stuck_timeout_s=30.0, stuck_eps=0.02,
            stuck_arrival_tol=0.3, collision_check=True, collision_margin=0.0):
    """Run the MPC simulation loop.

    Args:
        headless: If True, skip the matplotlib live plotter (and its blocking "press
            anything to finish" prompt) entirely, so the whole pipeline can run
            non-interactively -- e.g. from a script or CI -- and simply return a result.
        late_threshold_s: A robot is declared failed ("late") once it is still short of
            the node it is currently targeting more than this many seconds past that
            node's scheduled ETA. Pass None or False to disable the check.
        stuck_timeout_s: A robot is declared failed ("stuck") once it has not moved
            (translated) more than `stuck_eps` for this many consecutive seconds while
            not idle, not in the `aligning` work mode (which legitimately rotates in
            place), and more than `stuck_arrival_tol` away from the node it is currently
            targeting -- a robot parked at/near a node it already reached (e.g. waiting
            out a recharge or a time-window/precedence gap) is not stuck. Pass None or
            False to disable the check.
        stuck_eps: Minimum position change (metres) between ticks to count as "moved".
        stuck_arrival_tol: Distance (metres) to the current target node within which the
            robot is considered "arrived" and therefore exempt from the stuck check.
        collision_check: If True, the run fails ("collision") as soon as two robot bodies
            overlap, or a robot body overlaps a static obstacle or leaves the map
            boundary. Robots are treated as discs of radius `vehicle_width` and the test
            uses the *un-inflated* map, so this reports physical contact, not a breach of
            the planner's safety margin. Pass False (or None) to disable the check.
        collision_margin: Extra clearance (metres) a robot must keep on top of its own
            body radius before the check trips. 0.0 (the default) means bodies must
            actually touch; a positive value fails the run on near-misses, e.g. 0.1 flags
            any pass closer than 10 cm. Ignored when `collision_check` is off.

    Returns:
        A dict: {"status": "success"|"late"|"stuck"|"collision"|"timeout", "failure": dict
        or None, "ticks": int, "time": float, "actual_schedule_path": str}. "timeout" means
        the simulation ran out its full tick budget (`TIMEOUT`) without every robot
        finishing and without tripping a late/stuck/collision failure.
    """

    DATA_NAME = "schedule_demo2_data" # "schedule_demo_data"
    CFG_FNAME = "mpc_default.yaml" # "mpc_default.yaml" or "mpc_fast.yaml"
    MAP_ONLY = True
    AUTORUN = True # if false, press key (in the plot window) to continue
    MONITOR_COST = False # if true, monitor the cost (this will slow down the simulation)
    VERBOSE = False
    TIMEOUT = 10000

    # `False` is accepted as a synonym for `None` ("disable this check") since callers
    # naturally reach for it as the off-switch for a threshold.
    late_threshold_s = None if late_threshold_s in (None, False) else late_threshold_s
    stuck_timeout_s = None if stuck_timeout_s in (None, False) else stuck_timeout_s
    collision_check = bool(collision_check) # None/False both mean "off"
    collision_margin = 0.0 if collision_margin is None else float(collision_margin)

    if recording:
        save_video_path = f'./Demo/{DATA_NAME}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4'
    else:
        save_video_path = None

    root_dir = pathlib.Path(__file__).resolve().parents[1]
    data_dir = os.path.join(root_dir, "data", DATA_NAME)
    cnfg_dir = os.path.join(root_dir, "config")

    robot_ids = None # if none, read from schedule

    ### Configurations
    config_mpc_path = os.path.join(cnfg_dir, CFG_FNAME)
    config_robot_path = os.path.join(cnfg_dir, "robot_spec.yaml")

    config_mpc = MpcConfiguration.from_yaml(config_mpc_path)
    config_robot = CircularRobotSpecification.from_yaml(config_robot_path)

    solver_type = resolve_mpc_backend(config_mpc, mpc_backend, root_dir)
    if not naive_tracker:
        print(f"[run_mpc] NMPC backend: {solver_type}")

    ### Map, graph, and schedule paths
    map_path = os.path.join(data_dir, f"{EnvFolder}/map.json")
    test_case_path = os.path.join(root_dir, "data", "test_cases", f"{problem}.json")
    schedule_path = os.path.join(data_dir, "schedule.csv")
    start_path = os.path.join(data_dir, "robot_start.json")
    with open(start_path, "r") as f:
        robot_starts = json.load(f)

    ### Set up the global path/schedule coordinator
    gpc = GlobalPathCoordinator.from_csv(schedule_path)
    gpc.load_graph_from_test_case(test_case_path)
    gpc.load_map_from_json(map_path, inflation_margin=config_robot.vehicle_width+config_robot.vehicle_margin)
    robot_ids = gpc.robot_ids if robot_ids is None else robot_ids
    boundary_coords = gpc.current_map.boundary_coords
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
        path_times = relax_final_eta(path_coords, path_times, config_robot.lin_vel_max)
        robot_manager.add_schedule(rid, np.asarray(robot_starts[str(rid)]), path_coords, path_times)

    ### Run
    main_plotter = None
    if not headless:
        map_width  = max(np.asarray(boundary_coords)[:, 0]) - min(np.asarray(boundary_coords)[:, 0])
        map_height = max(np.asarray(boundary_coords)[:, 1]) - min(np.asarray(boundary_coords)[:, 1])
        save_params = {'skip_frame': 0, 'frame_size': (1280, int(map_height/map_width * 1280)), 'dpi': 300}
        main_plotter = MpcPlotInLoop(config_robot, map_only=MAP_ONLY, fig_ratio=(map_width/map_height), save_to_path=save_video_path, save_params=save_params)
        # main_plotter.plot_in_loop_pre(gpc.current_map, gpc.inflated_map, gpc.current_graph)
        main_plotter.plot_in_loop_pre(gpc.current_map, graph_manager=gpc.current_graph)
        color_list = [
            "#0072B2", "#D55E00", "#009E73", "#F0E442", "#56B4E9",
            "#E69F00", "#CC79A7",
        ]
        for i, rid in enumerate(robot_ids):
            planner = robot_manager.get_planner(rid)
            controller = robot_manager.get_controller(rid)
            visualizer = robot_manager.get_visualizer(rid)
            main_plotter.add_object_to_pre(rid,
                                           None, #planner.ref_traj,
                                           controller.state,
                                           controller.final_goal,
                                           color=color_list[i % len(color_list)])
            visualizer.plot(main_plotter.map_ax, *robot.state)

    ### Records when each robot actually reaches each node of its schedule, and writes that
    ### out below in `schedule.csv`'s own `robot_id,node_id,ETA` format, so planned and
    ### actual can be joined on (robot_id, node_id) and differenced.
    arrival_logger = logger_from_schedule(gpc, robot_ids)
    last_pos = {rid: None for rid in robot_ids}
    stuck_ticks = {rid: 0 for rid in robot_ids}
    failure = None

    ### Collision detection works on the raw map, not `gpc.inflated_map`/`static_obstacles`:
    ### the latter is already grown by vehicle_width+vehicle_margin for the planner, so a
    ### robot touching it is merely inside its safety margin, not in contact with anything.
    collision_checker = None
    if collision_check:
        collision_checker = StaticCollisionChecker(gpc.current_map.boundary_coords,
                                                   gpc.current_map.obstacle_coords_list)
        print(f"[run_mpc] Collision detection on (robot radius {config_robot.vehicle_width:.3f} m, "
              f"margin {collision_margin:.3f} m)")

    for kt in range(TIMEOUT):
        robot_states = []
        incomplete = False
        for i, rid in enumerate(robot_ids):
            # if rid != 'A1':
            #     continue
            robot = robot_manager.get_robot(rid)
            planner = robot_manager.get_planner(rid)
            controller = robot_manager.get_controller(rid)
            visualizer = robot_manager.get_visualizer(rid)
            other_robot_states = robot_manager.get_other_robot_states(rid, config_mpc)

            # Logged before the idle check, and before this tick's step, so that the time
            # stamp is the time of the position being reported -- and so that a robot which
            # has already parked on its final node still gets that arrival recorded.
            arrival_logger.update(rid, kt*config_mpc.ts, robot.state[:2])

            if controller.idle:
                if not headless:
                    main_plotter.update_plot(rid, kt, 0, None, 0, None, None)
                continue
            
            # `idx_check_range` is how many base-trajectory samples ahead of the current
            # docking point the planner may look for the sample nearest the robot, and the
            # docking index never regresses -- so it is also the furthest the reference can
            # advance in one tick. Samples are `lin_vel_max*ts` apart, so the old value of 5
            # gave 1.2 m of path lookahead. That is less than the path a robot skips when it
            # cuts a sharp corner: at `test_2`'s v15 (a 149-degree turn) A1 rounded the
            # corner 0.07 m from the outgoing edge but 3.1 m further along the path than the
            # docking point, so every sample in the window was further away than the docking
            # point itself, the reference stayed pinned behind the robot, and the MPC settled
            # on standing still -- moving forward only increased the reference-path-deviation
            # cost. One MPC horizon of lookahead covers the skip, and matches the length of
            # reference the controller is tracking anyway; `LocalTrajPlanner` truncates the
            # window at a path reversal so the wider range cannot jump a dead-end detour.
            ref_states, ref_speed, *_ = planner.get_local_ref(
                kt*config_mpc.ts, 
                (float(robot.state[0]), float(robot.state[1])), 
                idx_check_range=config_mpc.N_hor,
                ignore_speed_ref=ignore_speed_ref,
                # The reversal guard in `LocalTrajPlanner` needs the heading to tell an
                # outbound leg from the inbound one that retraces it; without it the docking
                # index cannot cross a dead-end turnaround at all.
                current_heading=float(robot.state[2])
            )
            print(f"(K:{kt}) Robot {rid}, ref speed: {round(ref_speed if ref_speed else -1, 4)}, next goal:{planner._current_target_node}") # XXX
            controller.set_current_state(robot.state)
            controller.set_ref_states(ref_states, ref_speed=ref_speed)
            if naive_tracker:
                (actions, pred_states, current_refs, debug_info) = controller.run_naive_step()
            else:
                (actions, pred_states, current_refs, debug_info) = controller.run_step(static_obstacles=static_obstacles,
                                                           full_dyn_obstacle_list=None,
                                                           other_robot_states=other_robot_states,
                                                           map_updated=True, report_cost=False, ignore_speed_ref=ignore_speed_ref)
            
            controller.report_cost(debug_info['cost'],
                                   debug_info['step_runtime'],
                                   debug_info['monitored_cost'],
                                   object_id=f"Robot {rid}")

            ### Real run
            # controller.run_step re-solves every tick (this loop calls it once per kt,
            # never batching config_mpc.action_steps real ticks per solve), so exactly one
            # physical step happens per solve. `actions` holds action_steps planned control
            # pairs from that one solve; actions[-1] is the *last* of them -- e.g. the 3rd of
            # 3 when action_steps=3 -- which is whatever the solver settled the trajectory
            # into several planning steps out, not what it computed for right now. With
            # action_steps=1 (configs.py's documented "normal" value) actions[-1] and
            # actions[0] are the same element, which is why this went unnoticed; at
            # action_steps=3 the two consistently differ, and actions[-1] is frequently near
            # zero even when actions[0] calls for a large correction (e.g. mid-turn), stalling
            # the robot. Apply actions[0], the control the solve computed for this tick.
            if (np.linalg.norm(robot.state[:2] - current_refs[-1][:2]) > 0.3) or controller._mode == 'aligning':
                if controller._mode != 'safe' or (np.linalg.norm(robot.state[:2] - current_refs[-1][:2]) > 0.8) or planner.idle:
                    robot.step(actions[0])
            robot_manager.set_pred_states(rid, np.asarray(pred_states))

            if not headless:
                main_plotter.update_plot(rid, kt, actions[0], None, debug_info['cost'], np.asarray(pred_states), current_refs)
                visualizer.update(*robot.state)

            if not controller.check_termination_condition(external_check=planner.idle):
                incomplete = True

            robot_states.append(robot.state)

            ### Failure detection: a robot is "stuck" if it has not moved for
            ### stuck_timeout_s while active, not merely rotating in place during
            ### `aligning`, and still more than stuck_arrival_tol from the node it is
            ### targeting -- a robot parked at/near a node it already reached (e.g.
            ### waiting out a recharge or a time-window/precedence gap) is not stuck.
            ### "late" fires if a robot is still short of its current target node more
            ### than late_threshold_s past that node's scheduled ETA.
            pos = np.asarray(robot.state[:2], dtype=float)
            target_node = planner.current_target_node
            dist_to_target = float(np.hypot(pos[0]-target_node[0], pos[1]-target_node[1]))
            if (stuck_timeout_s is not None and last_pos[rid] is not None
                    and controller._mode != 'aligning'
                    and dist_to_target > stuck_arrival_tol
                    and np.linalg.norm(pos - last_pos[rid]) < stuck_eps):
                stuck_ticks[rid] += 1
            else:
                stuck_ticks[rid] = 0
            last_pos[rid] = pos
            if stuck_timeout_s is not None and stuck_ticks[rid]*config_mpc.ts > stuck_timeout_s:
                failure = {"type": "stuck", "robot_id": rid, "time": kt*config_mpc.ts,
                           "stuck_for_s": stuck_ticks[rid]*config_mpc.ts}

            eta = planner.current_target_eta
            if late_threshold_s is not None and eta is not None and (kt*config_mpc.ts - eta) > late_threshold_s:
                failure = {"type": "late", "robot_id": rid, "time": kt*config_mpc.ts,
                           "scheduled_eta": eta, "lateness_s": kt*config_mpc.ts - eta}

            ### "collision" (part 1 of 2): this robot's body against the static world --
            ### obstacles and the map boundary. `clearance` is the gap between the robot's
            ### circumference and the nearest wall surface; negative means overlap.
            if collision_checker is not None and collision_checker.active:
                clearance = collision_checker.min_distance(pos[0], pos[1]) - config_robot.vehicle_width
                if clearance < collision_margin:
                    failure = {"type": "collision", "with": "obstacle", "robot_id": rid,
                               "time": kt*config_mpc.ts, "position": [float(pos[0]), float(pos[1])],
                               "clearance_m": clearance, "required_clearance_m": collision_margin}

            if failure is not None:
                break

        ### "collision" (part 2 of 2): robot against robot. This runs outside the per-robot
        ### loop because it needs every robot's post-step position, including robots that
        ### are idle (they `continue` above, but a robot parked on the path of another is
        ### still something to hit).
        if failure is None and collision_check:
            min_separation = 2*config_robot.vehicle_width + collision_margin
            positions = {rid: np.asarray(robot_manager.get_robot(rid).state[:2], dtype=float)
                         for rid in robot_ids}
            for i, rid_a in enumerate(robot_ids):
                for rid_b in robot_ids[i+1:]:
                    separation = float(np.linalg.norm(positions[rid_a] - positions[rid_b]))
                    if separation < min_separation:
                        failure = {"type": "collision", "with": "robot",
                                   "robot_ids": [rid_a, rid_b], "time": kt*config_mpc.ts,
                                   "distance_m": separation, "min_distance_m": min_separation,
                                   "overlap_m": min_separation - separation}
                        break
                if failure is not None:
                    break

        if not headless:
            main_plotter.plot_in_loop(time=kt*config_mpc.ts, autorun=AUTORUN, zoom_in=None)
        if failure is not None:
            print(f"[run_mpc] FAILURE detected: {failure}")
            break
        if not incomplete:
            break


    if not headless:
        main_plotter.show()
        input('Press anything to finish!')
        main_plotter.close()

    # The realised timetable, in the planned schedule's own format.
    arrival_logger.finalize()
    actual_schedule_path = os.path.join(data_dir, f"Actual_{problem}.csv")
    arrival_logger.to_csv(actual_schedule_path)
    print(f"Actual schedule saved to: {actual_schedule_path}")
    unreached = arrival_logger.unreached()
    if unreached:
        print(f"[run_mpc] Nodes never reached: {unreached}")

    # Non-convergence is per-solve and easy to miss tick by tick; the totals are not. A run
    # can finish "successfully" while a robot was steered by hundreds of non-optimal iterates.
    bad_exits = {rid: robot_manager.get_controller(rid).bad_exit_count for rid in robot_ids}
    if any(bad_exits.values()):
        print(f"[run_mpc] Solves with a bad exit status (per robot): {bad_exits}")

    if MONITOR_COST and not headless: # XXX
        import matplotlib.pyplot as plt # type: ignore
        fig, ax = plt.subplots(1, 1)
        solve_time = controller.solver_time_timelist
        ax.plot(solve_time, label="Solve time")
        ax.set_title(f"Solve time for Robot {rid}")
        ax.legend()
        plt.show()

    if failure is not None:
        status = failure["type"]
    elif incomplete:
        status = "timeout"
    else:
        status = "success"

    result = {
        "status": status,
        "failure": failure,
        "ticks": kt,
        "time": kt*config_mpc.ts,
        "actual_schedule_path": actual_schedule_path,
    }
    print(f"[run_mpc] Result: status={status}, ticks={kt}, time={kt*config_mpc.ts:.2f}s")
    return result