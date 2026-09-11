"""Tier 2 -- sidestep a contested node -- plus the coordinator's diagnostic distance measure.

The detour is deliberately built out of machinery that already exists: the offset point is
handed to `LocalTrajPlanner.get_new_path`, which routes through the *inflated* map and
raises if a waypoint is unreachable. That raise is the feasibility test, so there is no
separate geometry check to keep in step with the map.

Whether a sidestep fits at all is a property of the map, not of the robot: `SmallEnv`'s 2 m
corridors leave under half a metre of lateral room once the map is inflated, so on those
instances every candidate offset is rejected and the coordinator escalates instead. That is
expected behaviour, not a failure.
"""

import math
from typing import Optional

import numpy as np


def min_horizon_separation(pos_a: np.ndarray, pred_a, pos_b: np.ndarray, pred_b) -> float:
    """Closest the two robots come, over as much of their predicted horizons as both have.

    Recorded for diagnosis only -- nothing in the coordinator triggers on it. The horizons
    can be ragged (the naive tracker publishes references, which are resampled to a variable
    length), so the comparison runs over whatever length both share.
    """
    track_a = _horizon(pos_a, pred_a)
    track_b = _horizon(pos_b, pred_b)
    k = min(len(track_a), len(track_b))
    if k == 0:
        return float(np.linalg.norm(pos_a - pos_b))
    return float(np.min(np.linalg.norm(track_a[:k] - track_b[:k], axis=1)))


def _horizon(pos: np.ndarray, pred) -> np.ndarray:
    start = np.asarray(pos, dtype=float).reshape(1, 2)
    if pred is None or len(pred) == 0:
        return start
    return np.vstack([start, np.asarray(pred, dtype=float)[:, :2]])


def feasible_times(waypoints, wanted, v_max: float, t_start: float) -> list[float]:
    """Times for a waypoint chain that are strictly increasing and physically reachable.

    The schedule's own ETAs are kept wherever they are still achievable, so a robot that is
    on time keeps its deadlines and one that is behind is simply asked to drive flat out --
    which is what the local planner's "distance over time remaining" reference speed does
    with a deadline it cannot meet. Times that have already passed would otherwise produce a
    non-increasing chain, which the path builder cannot scale.
    """
    times = [t_start]
    for i in range(1, len(waypoints)):
        earliest = times[-1] + math.dist(waypoints[i-1], waypoints[i])/max(v_max, 1e-6)
        times.append(max(float(wanted[i]), earliest))
    return times


def plan_crossing(coordinator, active, now: float) -> Optional[dict]:
    """Install a lateral sidestep around the contested node, or return None if none fits.

    Candidates are tried widest-first and on the side away from the other robot. Each one is
    accepted only if the local planner can actually route through it and the resulting path
    is not dramatically longer than the straight-line waypoint chain -- a much longer route
    means the visibility planner went around an obstacle rather than stepping around it.
    """
    cfg = coordinator.cfg
    rid = active.yielder
    planner = coordinator.rm.get_planner(rid)
    robot = coordinator.rm.get_robot(rid)
    tt = coordinator.timetable(rid)

    node_index = coordinator._node_index(rid, active.node)
    if node_index is None or node_index + 1 > len(tt) - 1:
        return None

    pos = np.asarray(robot.state[:2], dtype=float)
    node = np.asarray(tt.coords[node_index], dtype=float)
    approach = node - (np.asarray(tt.coords[node_index-1], dtype=float) if node_index > 0 else pos)
    if float(np.linalg.norm(approach)) < 1e-9:
        return None
    approach = approach/np.linalg.norm(approach)
    normal = np.array([-approach[1], approach[0]])

    other_pos = np.asarray(coordinator.rm.get_robot(active.priority).state[:2], dtype=float)
    preferred = -np.sign(float(np.dot(other_pos - node, normal))) or 1.0

    body = 2*coordinator.cfg_robot.vehicle_width
    floor = coordinator.cfg_robot.vehicle_width
    want = cfg.crossing_offset_factor*body

    remaining_coords = [(float(c[0]), float(c[1])) for c in tt.coords[node_index:]]
    remaining_times = [float(x) for x in tt.etas[node_index:]]

    for scale in (1.0, 0.75, 0.5):
        offset = want*scale
        if offset < floor:
            break   # anything narrower than the robot's own radius is not a manoeuvre
        for side in (preferred, -preferred):
            point = node + side*offset*normal
            waypoints = [tuple(pos), (float(point[0]), float(point[1]))] + remaining_coords
            if _has_duplicate(waypoints):
                continue
            # The inserted point has no ETA of its own; ask for it as early as the robot can
            # get there, and let the following nodes keep their scheduled deadlines.
            wanted = [now, now] + remaining_times
            times = feasible_times(waypoints, wanted, coordinator.v_max, now)
            try:
                new_path, new_times = planner.get_new_path(waypoints, times)
            except (ValueError, ZeroDivisionError):
                continue
            if _too_long(new_path, waypoints, cfg.detour_detour_ratio):
                continue
            _install(coordinator, rid, new_path, new_times, remaining_coords,
                     remaining_times, node_index)
            return {'offset_m': offset, 'side': side}
    return None


def _has_duplicate(waypoints, tol: float = 1e-6) -> bool:
    """Consecutive identical waypoints make the path builder divide by a zero-length leg."""
    return any(math.dist(waypoints[i], waypoints[i+1]) < tol for i in range(len(waypoints)-1))


def _too_long(path, waypoints, ratio: float) -> bool:
    routed = sum(math.dist(path[i], path[i+1]) for i in range(len(path)-1))
    direct = sum(math.dist(waypoints[i], waypoints[i+1]) for i in range(len(waypoints)-1))
    return direct > 0 and routed > ratio*direct


def _install(coordinator, robot_id, new_path, new_times, remaining_coords,
             remaining_times, node_index) -> None:
    """Swap the detour in and tell the coordinator's own timetable about it.

    The robot's node sequence is unchanged -- only an extra, non-node waypoint is inserted --
    so the timetable keeps its original nodes and ETAs. Keeping the ETAs is deliberate: the
    reference speed is distance over time-remaining, so an unchanged deadline asks the robot
    to make the detour up rather than slowing it down further.
    """
    from run_mpc import relax_final_eta

    planner = coordinator.rm.get_planner(robot_id)
    new_times = relax_final_eta(new_path, new_times, coordinator.v_max)
    planner.load_path(new_path, new_times, nomial_speed=coordinator.v_max, method='linear')

    tt = coordinator.timetable(robot_id)
    coordinator.replace_timetable(
        robot_id,
        tt.node_ids[node_index:],
        remaining_coords,
        remaining_times)
