"""Record when each robot actually reaches each node of its planned route.

The simulation's counterpart to the scheduler's `schedule.csv`: the scheduler says
*when a robot should be* at a node, this says *when it was*, in the very same
`robot_id,node_id,ETA` layout so the two can be joined on `(robot_id, node_id)` and
differenced (see `src/schedule_visualization.py`).

Arrival is decided geometrically -- the robot's closest approach to the node, during the
pass in which it came within a small tolerance of it -- rather than from the local
planner's internal target index. The planner advances its target with a horizon of
lookahead, so it switches away from a node while the robot is still short of it, and it
leaves the start node targeted only briefly, which is why arrival at the start node used
to be missing from the output entirely.
"""

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd # type: ignore


Coord = tuple[float, float]

DEFAULT_ARRIVAL_TOL = 0.5 # metres


class ArrivalLogger:
    """Turn a stream of per-tick robot positions into a table of node arrival times.

    Usage mirrors the simulation loop: register every robot's planned route once, call
    `update` once per robot per tick, then `finalize` and `to_csv` when the run ends.

    Attributes:
        arrival_tol: Radius (metres) of the ball around a node that counts as "at" it.
        lookahead: How many nodes past the next expected one are watched, so that a
            robot which cuts a corner and never enters a node's ball does not stall the
            rest of its route behind that node.
    """

    def __init__(self, arrival_tol: float = DEFAULT_ARRIVAL_TOL, lookahead: int = 3) -> None:
        self.arrival_tol = float(arrival_tol)
        self.lookahead = int(lookahead)
        self._routes: dict[Any, dict] = {}

    # ------------------------------------------------------------------ setup

    def register_robot(self, robot_id: Any, node_ids: Sequence[Any], node_coords: Sequence[Coord]) -> None:
        """Declare the ordered route a robot is expected to follow.

        Args:
            robot_id: The robot's id, reproduced verbatim in the `robot_id` column.
            node_ids: Node ids in visiting order, as they appear in `schedule.csv`. A
                node visited twice must appear twice, in the right places.
            node_coords: The matching (x, y) for each entry in `node_ids`.

        Notes:
            The per-node tolerance is capped at half the distance to the nearer of a
            node's two route neighbours, so on a dense roadmap -- MovingAI and CCBS
            graphs can space nodes well under a metre apart -- one node's ball can never
            swallow the next node and silently drop a row.
        """
        if len(node_ids) != len(node_coords):
            raise ValueError(f"robot {robot_id}: {len(node_ids)} node ids but {len(node_coords)} coordinates")
        n = len(node_ids)
        coords = np.asarray(node_coords, dtype=float).reshape(n, 2)

        tolerances = np.full(n, self.arrival_tol, dtype=float)
        if n > 1:
            gaps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            # A zero gap is a repeated coordinate, not a real neighbour; taking it into
            # account would drive the tolerance to zero and make the node unreachable.
            gaps = np.where(gaps > 0.0, gaps, np.inf)
            for j in range(n):
                before = gaps[j-1] if j > 0 else np.inf
                after = gaps[j] if j < n-1 else np.inf
                tolerances[j] = min(self.arrival_tol, 0.5*min(before, after))

        self._routes[robot_id] = {
            'node_ids': list(node_ids),
            'coords': coords,
            'tol': tolerances,
            'next_idx': 0,
            'first_in_t': [None]*n,  # first time inside the ball; None means never arrived
            'closest_d': [np.inf]*n, # closest approach -- and its time, which is the arrival
            'closest_t': [None]*n,
            'arrivals': [],          # (route index, node id, time, exact) in commit order
        }

    # -------------------------------------------------------------------- run

    def update(self, robot_id: Any, time: float, position: Sequence[float]) -> None:
        """Feed one robot's position at one instant. Cheap enough to call every tick.

        Call it for idle robots too: a robot that reaches its last node and stops still
        needs that arrival recorded.
        """
        route = self._routes.get(robot_id)
        if route is None:
            return
        n = len(route['node_ids'])
        pos = np.asarray(position, dtype=float)[:2]

        ### Only the node currently being awaited is watched for arrival. Watching the
        ### lookahead window too would break routes that revisit a node: a robot standing on
        ### its start node is also standing on the later occurrence of that same node, and
        ### the later visit would be timestamped at t=0. Route order is the only thing that
        ### disambiguates them, so a node cannot register until its predecessor is settled.
        while route['next_idx'] < n:
            j = route['next_idx']
            d_j = float(np.linalg.norm(pos - route['coords'][j]))
            if d_j < route['closest_d'][j]:
                route['closest_d'][j] = d_j
                route['closest_t'][j] = float(time)
            if d_j <= route['tol'][j] and route['first_in_t'][j] is None:
                route['first_in_t'][j] = float(time)

            ### A node is settled either when the robot has been inside its ball and has now
            ### left it, or when the robot has clearly gone past without ever entering -- it
            ### is receding from its closest approach by more than a tolerance, and some
            ### later node on the route is now nearer than this one. That second case is a
            ### cut corner, and the closest approach is the honest estimate of when it
            ### happened. Both tests use the *current* geometry, never a coordinate match
            ### recorded earlier, so a repeated node cannot settle the wrong visit.
            if route['first_in_t'][j] is not None:
                if d_j > route['tol'][j]:
                    self._commit(route, j, exact=True)
                    continue
            elif d_j > route['closest_d'][j] + route['tol'][j]:
                later = [float(np.linalg.norm(pos - route['coords'][k]))
                         for k in range(j+1, min(n, j+1+self.lookahead))]
                if later and min(later) < d_j:
                    self._commit(route, j, exact=False)
                    continue
            break

    def finalize(self) -> None:
        """Close out every route once the run is over.

        The last node of a route is never committed by `update`, because the robot parks
        there and so never leaves its ball; likewise a run cut short by a timeout or a
        collision leaves a tail of nodes pending. Everything the robot was ever seen
        approaching is recorded here; nodes it never got near at all are left out of the
        output, exactly as an unvisited node should be -- `unreached` names them.
        """
        for route in self._routes.values():
            while route['next_idx'] < len(route['node_ids']):
                j = route['next_idx']
                if route['first_in_t'][j] is not None:
                    self._commit(route, j, exact=True)
                elif route['closest_t'][j] is not None and np.isfinite(route['closest_d'][j]):
                    self._commit(route, j, exact=False)
                else:
                    route['next_idx'] += 1 # never observed at all -- drop it

    @staticmethod
    def _commit(route: dict, j: int, exact: bool) -> None:
        # The arrival is the moment of closest approach, not the moment the tolerance ball
        # was first entered: the latter is early by roughly tol/speed (0.5 s at 1 m/s with
        # the default half-metre ball), a bias that would show up as the whole fleet running
        # systematically ahead of schedule. `exact` records only whether the robot got
        # within tolerance at all, i.e. whether this is an arrival or a fly-past.
        route['arrivals'].append((j, route['node_ids'][j], float(route['closest_t'][j]), exact))
        route['next_idx'] = j + 1

    # ----------------------------------------------------------------- output

    def to_dataframe(self, include_flags: bool = False) -> pd.DataFrame:
        """The recorded arrivals as `robot_id,node_id,ETA`, sorted like `schedule.csv`.

        Args:
            include_flags: Also emit `exact` (False where the arrival is a closest
                approach rather than a real entry into the node's ball) and
                `closest_distance_m`. Diagnostics only -- left out of the CSV by default
                so its columns stay identical to the planned schedule's.
        """
        rows = []
        for robot_id, route in self._routes.items():
            for j, node_id, time, exact in route['arrivals']:
                # Times are integer multiples of the sampling time; rounding keeps
                # accumulated float noise (7.0000000000000036) out of the CSV.
                row = {'robot_id': robot_id, 'node_id': node_id, 'ETA': round(time, 6)}
                if include_flags:
                    row['exact'] = exact
                    row['closest_distance_m'] = float(route['closest_d'][j])
                rows.append(row)
        columns = ['robot_id', 'node_id', 'ETA'] + (['exact', 'closest_distance_m'] if include_flags else [])
        df = pd.DataFrame(rows, columns=columns)
        return df.sort_values(['robot_id', 'ETA']).reset_index(drop=True)

    def to_csv(self, path: str, include_flags: bool = False) -> str:
        """Write the arrivals to `path` in `schedule.csv`'s format and return the path."""
        self.to_dataframe(include_flags=include_flags).to_csv(path, index=False)
        return path

    def unreached(self) -> dict[Any, list]:
        """Per robot, the scheduled node ids that produced no arrival at all.

        Non-empty after a timeout or a run that failed part-way; empty on a clean run.
        Only meaningful once `finalize` has been called.
        """
        missing = {}
        for robot_id, route in self._routes.items():
            committed = {j for j, _, _, _ in route['arrivals']}
            pending = [node_id for j, node_id in enumerate(route['node_ids']) if j not in committed]
            if pending:
                missing[robot_id] = pending
        return missing


def logger_from_schedule(gpc, robot_ids: Sequence[Any],
                         arrival_tol: float = DEFAULT_ARRIVAL_TOL) -> ArrivalLogger:
    """Build an `ArrivalLogger` pre-registered with every robot's planned route.

    Args:
        gpc: A `GlobalPathCoordinator` with both the schedule and the node graph loaded,
            so route coordinates can be named back into the node ids the CSV needs.
        robot_ids: The robots to track.
        arrival_tol: Passed through to `ArrivalLogger`.
    """
    logger = ArrivalLogger(arrival_tol=arrival_tol)
    for rid in robot_ids:
        path_coords, _ = gpc.get_robot_schedule(rid)
        coords = [(float(c[0]), float(c[1])) for c in path_coords]
        logger.register_robot(rid, [gpc.get_node_id(c) for c in coords], coords)
    return logger
