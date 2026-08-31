import math
from typing import Optional, Any

import numpy as np
from scipy import interpolate # type: ignore
from matplotlib.axes import Axes # type: ignore

from ._ref_traj_generation import TrajectoryGeneration
from .path_plan_cspace import visibility # optional if don't need to use the local replanner


PathNode = tuple[float, float]
TrajNode = tuple[float, float, float]

class LocalTrajPlanner:
    """The local planner for each individual robot takes path nodes and ETAs as inputs, and outputs local reference.
    
    Attributes:
        current_target_node: The current target path node.
        ref_traj: The reference trajectory.
        ref_speed: The reference speed.
        docking_point: The docking point on the reference trajectory.

    Notes:
        The path must be loaded for any methods to work.
        To use the local replanner, call `load_map` first.
    """
    def __init__(self, sampling_time: float, horizon: int, max_speed: float, verbose:bool=False) -> None:
        """The local planner takes path nodes and ETAs as inputs, and outputs local reference.

        Args:
            sampling_time: The sampling time of the local planner.
            horizon: The horizon of the local planner.
            verbose: If True, the local planner will print out the debug information.
        """
        self.ts = sampling_time
        self.N_hor = horizon
        self.v_max = max_speed
        self.vb = verbose

        self.path_planner:Optional[Any] = None

        self._ref_path:Optional[list[PathNode]] = None
        self._ref_path_time:Optional[list[float]] = None

        self.traj_gen = TrajectoryGeneration()
        self.traj_gen.set_sample_time(self.ts)

        self._ref_speed:Optional[float] = None
        self._base_traj:Optional[list[TrajNode]] = None
        self._base_traj_time:Optional[list[float]] = None

        self._current_target_node:Optional[PathNode] = None
        self._current_target_node_idx:Optional[int] = None
        self._base_traj_target_node:Optional[list[PathNode]] = None
        self._base_traj_docking_idx:Optional[int] = None # the index of the docking point on the base trajectory

        self._idle = True

    @property
    def idle(self) -> bool:
        return self._idle

    @property
    def current_target_node(self) -> tuple:
        assert self._current_target_node is not None
        return self._current_target_node
    
    @property
    def ref_traj(self) -> np.ndarray:
        return np.asarray(self._base_traj)
    
    @property
    def ref_speed(self) -> float:
        assert self._ref_speed is not None
        return self._ref_speed
    
    @property
    def current_target_eta(self) -> Optional[float]:
        """Scheduled arrival time for the node currently being targeted, or None if the loaded path carries no ETAs."""
        if self._ref_path_time is None:
            return None
        assert self._current_target_node_idx is not None
        return self._ref_path_time[self._current_target_node_idx]

    @property
    def docking_point(self) -> tuple:
        assert self._base_traj is not None
        assert self._base_traj_docking_idx is not None
        return self._base_traj[self._base_traj_docking_idx]

    @staticmethod
    def downsample_ref_states(original_states: np.ndarray, original_speed: float, new_speed: float):
        """Downsample the reference states to the given speed. 

        Args:
            original_states: The original reference states, each row is a state.
            original_speed: The original speed used to generate the original reference states.
            new_speed: The new speed to downsample the reference states.

        Returns:
            New reference states with the given speed, each row is a state.

        Notes:
            The new speed should be smaller than the original speed. Otherwise, the will be few states in the new reference.
        """
        n_states = original_states.shape[0]
        distances = np.cumsum(np.sqrt(np.sum(np.diff(original_states[:, :2], axis=0)**2, axis=1))) # distance traveled along the path at each point
        if n_states < 2 or distances[-1] <= 0.0:
            # Zero-length reference: at the very end of the base trajectory the horizon is
            # padded with copies of the last point, so every arc-length is 0 and the
            # normalisation below would divide by zero -- producing an all-NaN reference that
            # the solver silently turns into garbage actions. There is nothing to resample.
            return original_states
        distances = np.insert(distances, 0, 0)/distances[-1] # normalize distances to [0, 1]
        fx = interpolate.interp1d(distances, original_states[:, 0], kind='linear')
        fy = interpolate.interp1d(distances, original_states[:, 1], kind='linear')

        num_points = int(original_speed/new_speed*n_states)  
        new_distances = np.linspace(0, 1, num_points)
        new_x = fx(new_distances)
        new_y = fy(new_distances)
        new_heading = np.arctan2(np.diff(new_y), np.diff(new_x))
        new_heading = np.append(new_heading, new_heading[-1])
        new_states = np.column_stack([new_x, new_y, new_heading])[:n_states, :]
        return new_states

    def load_map(self, boundary_coords: list[PathNode], obstacle_list: list[list[PathNode]]):
        """Load the map for the local path planner."""
        self.path_planner = visibility.VisibilityPathFinder(boundary_coords, obstacle_list, verbose=self.vb)

    def load_path(self, path_coords: list[PathNode], path_times: Optional[list[float]], nomial_speed:Optional[float]=None, method:str='linear'):
        """The reference speed is used to generate the base trajectory.
        
        Notes:
            Linear sampling: The base trajectory is sampled with a constant distance (step-size).
            Time sampling: The base trajectory is sampled with a constant time interval.
        """
        self._ref_path = path_coords
        self._ref_path_time = path_times
        self._current_target_node = self._ref_path[0]
        self._current_target_node_idx = 0

        self.traj_gen.set_reference(self._ref_path, self._ref_path_time)
        if nomial_speed is not None:
            self.traj_gen.set_nominal_speed(nomial_speed)
        self._base_traj, self._base_traj_time, self._base_traj_target_node = self.traj_gen.generate_trajectory(method=method)
        self._base_traj_docking_idx = 0
        self._sampling_method = method

        self._idle = False

    def get_local_ref(self, current_time: float, current_pos: PathNode, idx_check_range:int=10,
                      external_ref_speed:Optional[float]=None, ignore_speed_ref:bool=False,
                      current_heading:Optional[float]=None):
        """Get the local reference from the current time and position.

        Args:
            idx_check_range: For linear sampling, the range of the index to check the docking point.
            external_ref_speed: The external reference speed used for resampling the reference states.
            current_heading: The robot's current heading in radians. Optional, but without it a
                path reversal cannot be crossed -- see `_reversal_bound`.

        Raises:
            ValueError: Sampling method not supported.

        Returns:
            ref_states: The local state reference.
            ref_speed: The reference speed.
            done: If the current node is the last node in the reference path, return True. Otherwise, return False.
        """
        if self._sampling_method == 'time':
            ref_states, ref_speed, done = self.get_local_ref_from_time_sampling(current_time)
        elif self._sampling_method == 'linear':
            ref_states, ref_speed, done = self.get_local_ref_from_linear_sampling(current_time, current_pos, idx_check_range, current_heading)
        else:
            raise ValueError('Sampling method not supported.')
        if ignore_speed_ref:
            ref_speed = None
        if (ref_speed is not None) and (ref_speed > 1e-6):
            ref_states = self.downsample_ref_states(ref_states, self.traj_gen.speed, ref_speed)
        if external_ref_speed is not None:
            ref_states = self.downsample_ref_states(ref_states, self.traj_gen.speed, external_ref_speed)
        if done:
            self._idle = True
        return ref_states, ref_speed, done

    def get_local_ref_from_linear_sampling(self, current_time: float, current_pos: PathNode,
                                           idx_check_range:int=10, current_heading:Optional[float]=None):
        """The local planner takes the current position as input, and outputs the local reference.

        Args:
            current_pos: The current position of the agent (robot).
            idx_check_range: The range of the index to check the docking point.
            current_heading: The robot's current heading in radians, used to decide whether a path
                reversal ahead has already been executed. See `_reversal_bound`.

        Returns:
            ref_states: The local state reference.
            ref_speed: The reference speed. If not reference time, this is None.
            done: If the current node is the last node in the reference path, return True. Otherwise, return False.
        """
        assert self._ref_path is not None
        assert self._base_traj is not None
        assert self._base_traj_target_node is not None
        assert self._base_traj_docking_idx is not None
        assert self._current_target_node is not None
        assert self._current_target_node_idx is not None

        # At a path reversal (the robot turns ~180 degrees in place, e.g. backtracking
        # the same edge) the outbound and inbound samples run back over the same (x,y)
        # positions with opposite headings -- nearest-position search can't tell them
        # apart anywhere along the corridor, not just at the pivot. Letting the docking
        # index regress (the old `-1` allowance) lets it walk back across a reversal it
        # already committed past, and even without that, near-ties right at the fold
        # flip on sub-mm position noise. Both make the reference heading -- and the
        # tracker's aligning/work mode -- chatter forever instead of completing the
        # turn. Fix both: never regress the docking index, and among forward
        # candidates within a small tolerance of the true nearest, keep the one
        # closest to the previous index instead of whichever wins the raw argmin.
        lb_idx = self._base_traj_docking_idx
        # `max(lb_idx+1, ...)`: once the docking index reaches the last sample the old cap of
        # `len-1` made the slice below empty and `min()` raised ValueError. Keep at least the
        # current sample in the window.
        ub_idx = max(lb_idx+1, min(self._base_traj_docking_idx+idx_check_range, len(self._base_traj)-1))
        # Never let the window reach across a path reversal the robot has not performed yet.
        # Widening `idx_check_range` enough to round a sharp corner (see `run_mpc`, which now
        # passes the MPC horizon) would otherwise let the docking index jump straight from the
        # outbound leg of a dead-end detour to the inbound one -- the two legs retrace the same
        # (x,y) samples, so a robot part-way out is genuinely nearest to an inbound sample, and
        # the node at the dead end would be skipped without ever being visited. Which leg the
        # robot is on is not decidable from position, only from heading, which is why
        # `_reversal_bound` needs `current_heading`.
        ub_idx = min(ub_idx, self._reversal_bound(lb_idx, current_heading))

        distances = [math.hypot(current_pos[0]-x[0], current_pos[1]-x[1]) for x in self._base_traj[lb_idx:ub_idx]]
        min_dist = min(distances)
        tie_tol = 1e-3
        tied = [i for i, d in enumerate(distances) if d <= min_dist + tie_tol]
        best = min(tied, key=lambda i: abs((lb_idx + i) - self._base_traj_docking_idx))
        self._base_traj_docking_idx = lb_idx + best

        if self._ref_path_time is not None:
            distance_to_current_node = math.hypot(current_pos[0]-self._current_target_node[0], current_pos[1]-self._current_target_node[1])
            timediff_to_current_node = max(self._ref_path_time[self._current_target_node_idx] - current_time, 0) + 1e-6
            ref_speed = min(distance_to_current_node/timediff_to_current_node, self.v_max)
            # print(distance_to_current_node, '/', timediff_to_current_node, '->', self._ref_path_time[self._current_target_node_idx]) # XXX
        else:
            ref_speed = None

        if (self._base_traj_docking_idx+self.N_hor >= len(self._base_traj)): # if horizon exceeds the base trajectory
            ref_states = np.array(self._base_traj[self._base_traj_docking_idx:] + \
                                  [self._base_traj[-1]]*(self.N_hor-(len(self._base_traj)-self._base_traj_docking_idx)))
        else:
            ref_states = np.array(self._base_traj[self._base_traj_docking_idx:self._base_traj_docking_idx+self.N_hor])

        self._current_target_node = self._base_traj_target_node[self._base_traj_docking_idx]
        target_node_idc = np.where([x==self._current_target_node for x in self._ref_path])[0]
        try:
            self._current_target_node_idx = target_node_idc[target_node_idc >= self._current_target_node_idx][0] 
        except IndexError:
            self._current_target_node_idx = self._current_target_node_idx

        if self._current_target_node_idx == len(self._ref_path)-1:
            done = True
        else:
            done = False

        return ref_states, ref_speed, done

    def _reversal_bound(self, lb_idx: int, current_heading: Optional[float]=None) -> int:
        """Exclusive upper index for a docking search starting at `lb_idx`, stopping at a reversal.

        A path reversal -- the robot drives to a dead end and comes straight back -- makes the
        outbound and inbound samples share the same (x,y) positions with opposite headings, so
        the nearest-sample search cannot tell them apart anywhere along the corridor. Bounding
        the search at the reversal keeps it on the leg the robot is actually on.

        Which leg that is depends on where the robot is *pointing*, not where it is. So when
        `current_heading` is given, a reversal whose post-fold heading the robot is already
        closer to counts as executed: the robot has turned around, the samples beyond the fold
        are the ones ahead of it, and the search is allowed through (up to the next reversal it
        has *not* performed). Without that release the docking index sticks one sample short of
        the fold forever, because the trajectory generator places the first inbound sample
        slightly *past* the last outbound one (`4Small`, A4: outbound ends at x=-19.8, the fold
        sample sits at x=-19.9, and the inbound samples the robot actually drives over start at
        x=-19.6, outside a window clamped to the fold). The robot then drove 2.6 m back east
        with its reference pinned behind it and stalled.

        With no `current_heading` the bound still admits one sample past the fold, which lets
        the docking index at least reach the turning point; that is the most it can do blind.
        """
        assert self._base_traj is not None
        theta_ref = self._base_traj[lb_idx][2]
        scan_from = lb_idx
        while True:
            fold = None
            for idx in range(scan_from+1, len(self._base_traj)):
                if self._angle_gap(self._base_traj[idx][2], theta_ref) > math.radians(170):
                    fold = idx
                    break
            if fold is None:
                return len(self._base_traj)
            if current_heading is None:
                return fold + 1
            theta_fold = self._base_traj[fold][2]
            if self._angle_gap(current_heading, theta_fold) >= self._angle_gap(current_heading, theta_ref):
                # Still pointing along the pre-fold leg: the reversal is ahead of the robot.
                return fold + 1
            theta_ref = theta_fold
            scan_from = fold

    @staticmethod
    def _angle_gap(a: float, b: float) -> float:
        """Absolute difference between two angles, wrapped into [0, pi]."""
        return abs((a - b + math.pi) % (2*math.pi) - math.pi)

    def get_local_ref_from_time_sampling(self, current_time: float):
        """The local planner takes the current time as input, and outputs the local reference.

        Returns:
            ref_states: The local state reference.
            ref_speed: The reference speed.
            done: If the current node is the last node in the reference path, return True. Otherwise, return False.

        Notes:
            The time sampling will keep forwarding the docking point until the current time is larger than the reference time.
        """
        assert self._ref_path is not None
        assert self._base_traj is not None
        assert self._base_traj_time is not None
        assert self._base_traj_target_node is not None
        assert self._base_traj_docking_idx is not None
        assert self._current_target_node is not None
        assert self._current_target_node_idx is not None

        try:
            self._base_traj_docking_idx = int(np.where(current_time>np.asarray(self._base_traj_time))[0][-1] + 1)
        except IndexError:
            self._base_traj_docking_idx = 0

        done = False
        if self._base_traj_docking_idx >= len(self._base_traj):
            done = True
            self._base_traj_docking_idx = len(self._base_traj) - 1

        if (self._base_traj_docking_idx+self.N_hor >= len(self._base_traj)):
            ref_states = np.array(self._base_traj[self._base_traj_docking_idx:] + [self._base_traj[-1]]*(self.N_hor-(len(self._base_traj)-self._base_traj_docking_idx)))
        else:
            ref_states = np.array(self._base_traj[self._base_traj_docking_idx:self._base_traj_docking_idx+self.N_hor])

        ref_speed = math.hypot(ref_states[0, 0]-ref_states[1,0], ref_states[0, 1]-ref_states[1,1]) / self.ts
        ref_speed = min(ref_speed, self.v_max)

        self._current_idx = self._ref_path.index(self._base_traj_target_node[self._base_traj_docking_idx])
        self._current_node = self._ref_path[self._current_idx]
        if self._current_idx == len(self._ref_path)-1:
            done = True
        else:
            done = False

        return ref_states, ref_speed, done

    def get_new_path(self, waypoints: list[PathNode], time_list: list[float]) -> tuple[list[PathNode], list[float]]:
        """Get a new path from the given waypoints and time list to avoid static obstacles.

        Args:
            waypoints: List of waypoints, each waypoint is a tuple (x, y).
            time_list: List of estimated arrival time at each waypoint.

        Raises:
            ValueError: Waypoints must have at least two points.

        Returns:
            new_path: The new path.
            new_time: The new time.
        """
        if len(waypoints) < 2:
            raise ValueError("Waypoints must have at least two points")
        assert isinstance(self.path_planner, visibility.VisibilityPathFinder)
        new_path = [waypoints[0]]
        new_path_segment_length = []
        new_time = [time_list[0]]
        for i in range(len(waypoints)-1):
            start, end = waypoints[i], waypoints[i+1]
            section_time = time_list[i+1] - time_list[i]

            new_segment, segment_length = self.path_planner.get_ref_path(start, end)
            new_segment = new_segment[1:]
            new_path.extend(new_segment)
            new_path_segment_length.extend(segment_length)
            new_section_time: list[float] = list(np.cumsum(segment_length) / sum(segment_length) * section_time)
            new_section_time = [x + time_list[i] for x in new_section_time]
            new_time.extend(new_section_time)
        return new_path, new_time
    

    def plot_schedule(self, ax: Axes, plot_args:dict={'c':'r'}):
        ax.plot(self.ref_traj[:,0], self.ref_traj[:,1], 'o', markerfacecolor='none', **plot_args)

