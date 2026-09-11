"""Projected timetables and the schedule-overlap tests that trigger the coordinator.

The scheduler already separated the robots in time: it gave each one an arrival time at
every node of its route, spaced so that no two robots occupy the same node at once. What
the coordinator watches for is that separation being *undone by execution drift* -- a robot
running late arrives into a window another robot was promised.

So detection here is arithmetic on schedules, not geometry on positions. Each robot carries
its remaining `(node, scheduled ETA)` list plus a single scalar `delay`, its timetable is
projected forward as `scheduled ETA + delay`, and two robots conflict when their projected
occupancy intervals at a shared node overlap (or an edge is traversed head-on).

Nothing in this module imports from the simulation; it is plain data and arithmetic, so the
overlap logic can be exercised without running anything.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


Coord = tuple[float, float]

# Two coordinates are the same node if they agree to this many metres. Node coordinates
# come from the same graph on both sides of every comparison, so this only absorbs float
# round-tripping through the schedule CSV, not genuine position error.
COORD_TOL = 1e-6


@dataclass
class RobotTimetable:
    """One robot's remaining route as the coordinator understands it.

    `index` is the entry the robot is currently travelling towards -- everything before it
    has been passed. `delay` is how far behind (positive) or ahead (negative) of schedule
    the robot is projected to be; the coordinator recomputes it every tick.
    """

    robot_id: Any
    node_ids: list
    coords: list[Coord]
    etas: list[float]
    nominal_speed: float = 1.0
    index: int = 0
    delay: float = 0.0
    dwell: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.node_ids)
        if not (len(self.coords) == len(self.etas) == n):
            raise ValueError(
                f"robot {self.robot_id}: {n} nodes, {len(self.coords)} coordinates, "
                f"{len(self.etas)} ETAs -- all three must match")
        if not self.dwell:
            self.dwell = self._infer_dwell()

    def _infer_dwell(self) -> list[float]:
        """How long the robot stands at each node, read out of its own schedule.

        The schedule gives arrival times, not departures, so a stop has to be inferred:
        whatever time between two consecutive arrivals is not accounted for by driving the
        leg at nominal speed is time spent standing at the earlier node. That is what
        distinguishes a robot passing straight through a junction from one parked on it --
        and the latter blocks the node for everyone else for as long as it sits there.
        """
        dwell = []
        speed = max(self.nominal_speed, 1e-6)
        for i in range(len(self.node_ids) - 1):
            leg = math.dist(self.coords[i], self.coords[i + 1])
            dwell.append(max(0.0, (self.etas[i + 1] - self.etas[i]) - leg/speed))
        # A robot never leaves its final node: it parks there for the rest of the run.
        dwell.append(math.inf)
        return dwell

    def __len__(self) -> int:
        return len(self.node_ids)

    @property
    def finished(self) -> bool:
        return self.index >= len(self.node_ids)

    def projected_arrival(self, i: int) -> float:
        """When the robot is expected to reach entry `i`, given its current delay."""
        return self.etas[i] + self.delay

    def occupancy(self, i: int) -> tuple[float, float]:
        """The interval during which entry `i`'s node is the robot's: arrival plus dwell.

        Deliberately *not* "until it reaches the next node" -- that would charge the whole
        outgoing edge to the node and make every schedule look self-conflicting. A robot
        passing through a junction occupies it only momentarily; one that stops there
        occupies it for as long as it stands, which `dwell` captures. The required gap
        between two robots' intervals is the caller's `clearance_s`.
        """
        start = self.projected_arrival(i)
        return start, start + self.dwell[i]

    def upcoming(self, now: float, lookahead_s: float):
        """Yield `(i, node_id)` for entries whose projected arrival is within the window.

        The node in flight is always included even if its projected arrival has already
        slipped past `now` -- that is the one conflict that is happening right now.
        """
        horizon = now + lookahead_s
        for i in range(self.index, len(self.node_ids)):
            if i > self.index and self.projected_arrival(i) > horizon:
                return
            yield i, self.node_ids[i]

    def advance_to_target(self, target_coord: Optional[Coord], search_ahead: int = 4) -> None:
        """Move `index` up to the entry matching the node the robot is now driving to.

        Matching is by coordinate rather than by counting arrivals, so this stays correct
        after a replan has given the robot a different route, and a waypoint that is not a
        graph node at all (the offset point a crossing detour inserts) simply matches
        nothing and leaves the index where it was.
        """
        if target_coord is None:
            return
        for i in range(self.index, min(len(self.node_ids), self.index + search_ahead + 1)):
            if (abs(self.coords[i][0] - target_coord[0]) <= COORD_TOL
                    and abs(self.coords[i][1] - target_coord[1]) <= COORD_TOL):
                self.index = i
                return


@dataclass
class Conflict:
    """A predicted loss of the separation the schedule had arranged."""

    kind: str                 # 'node' or 'edge'
    robots: tuple             # (robot id, robot id), ordered as `priority` then `yielder`
    node: Any                 # the contested node ('edge': the node being swapped into)
    priority: Any             # the robot the schedule says goes first -- it keeps its slot
    yielder: Any              # the other one
    overlap_s: float          # how much the intervals overlap; negative means a gap
    intervals: dict = field(default_factory=dict)   # robot id -> (start, end)
    scheduled: dict = field(default_factory=dict)   # robot id -> scheduled ETA at `node`
    projected: dict = field(default_factory=dict)   # robot id -> projected arrival at `node`

    @property
    def key(self) -> tuple:
        return (frozenset(self.robots), self.node, self.kind)


def _separation(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Seconds between two intervals: positive is a gap, negative is an overlap."""
    return max(b[0] - a[1], a[0] - b[1])


def _order(tt_a: RobotTimetable, i_a: int, tt_b: RobotTimetable, i_b: int) -> tuple[Any, Any]:
    """Which robot the *schedule* intended to go first, and which therefore yields.

    Decided on the scheduled ETAs, never the projected ones: the coordinator's job is to
    restore the order the scheduler chose, not to bless whichever robot drifted ahead. Ties
    break on slack (more slack yields, since delaying it costs least) and then on robot id,
    so the choice is deterministic.
    """
    eta_a, eta_b = tt_a.etas[i_a], tt_b.etas[i_b]
    if eta_a != eta_b:
        return (tt_a.robot_id, tt_b.robot_id) if eta_a < eta_b else (tt_b.robot_id, tt_a.robot_id)
    slack_a = eta_a - tt_a.projected_arrival(i_a)
    slack_b = eta_b - tt_b.projected_arrival(i_b)
    if slack_a != slack_b:
        return (tt_a.robot_id, tt_b.robot_id) if slack_a < slack_b else (tt_b.robot_id, tt_a.robot_id)
    ids = sorted([tt_a.robot_id, tt_b.robot_id], key=str)
    return ids[0], ids[1]


def find_conflicts(tt_a: RobotTimetable, tt_b: RobotTimetable, now: float,
                   lookahead_s: float, clearance_s: float) -> list[Conflict]:
    """Every predicted conflict between two robots inside the lookahead window."""
    if tt_a.finished or tt_b.finished:
        return []

    upcoming_b = list(tt_b.upcoming(now, lookahead_s))
    if not upcoming_b:
        return []
    index_b: dict = {}
    for i, node_id in upcoming_b:
        index_b.setdefault(node_id, []).append(i)

    conflicts = []
    for i_a, node_id in tt_a.upcoming(now, lookahead_s):
        for i_b in index_b.get(node_id, ()):
            occ_a, occ_b = tt_a.occupancy(i_a), tt_b.occupancy(i_b)
            gap = _separation(occ_a, occ_b)
            if gap >= clearance_s:
                continue
            priority, yielder = _order(tt_a, i_a, tt_b, i_b)
            conflicts.append(Conflict(
                kind='node',
                robots=(priority, yielder),
                node=node_id,
                priority=priority,
                yielder=yielder,
                overlap_s=-gap,
                intervals={tt_a.robot_id: occ_a, tt_b.robot_id: occ_b},
                scheduled={tt_a.robot_id: tt_a.etas[i_a], tt_b.robot_id: tt_b.etas[i_b]},
                projected={tt_a.robot_id: tt_a.projected_arrival(i_a),
                           tt_b.robot_id: tt_b.projected_arrival(i_b)},
            ))

    conflicts.extend(_find_edge_conflicts(tt_a, tt_b, now, lookahead_s, clearance_s))
    return conflicts


def _find_edge_conflicts(tt_a: RobotTimetable, tt_b: RobotTimetable, now: float,
                         lookahead_s: float, clearance_s: float) -> list[Conflict]:
    """Head-on swaps: the two robots traverse the same edge in opposite directions.

    Waiting does not reliably fix this -- whoever waits is still in the other's way on a
    single-lane edge -- which is why it is reported separately and allowed to escalate
    past the sidestep tier.
    """
    legs_b = {}
    for i, node_id in tt_b.upcoming(now, lookahead_s):
        if i + 1 < len(tt_b.node_ids):
            legs_b.setdefault((node_id, tt_b.node_ids[i + 1]), []).append(i)

    conflicts = []
    for i_a, node_id in tt_a.upcoming(now, lookahead_s):
        if i_a + 1 >= len(tt_a.node_ids):
            continue
        next_a = tt_a.node_ids[i_a + 1]
        for i_b in legs_b.get((next_a, node_id), ()):
            travel_a = (tt_a.projected_arrival(i_a), tt_a.projected_arrival(i_a + 1))
            travel_b = (tt_b.projected_arrival(i_b), tt_b.projected_arrival(i_b + 1))
            gap = _separation(travel_a, travel_b)
            if gap >= clearance_s:
                continue
            priority, yielder = _order(tt_a, i_a, tt_b, i_b)
            conflicts.append(Conflict(
                kind='edge',
                robots=(priority, yielder),
                node=next_a,
                priority=priority,
                yielder=yielder,
                overlap_s=-gap,
                intervals={tt_a.robot_id: travel_a, tt_b.robot_id: travel_b},
                scheduled={tt_a.robot_id: tt_a.etas[i_a], tt_b.robot_id: tt_b.etas[i_b]},
                projected={tt_a.robot_id: tt_a.projected_arrival(i_a),
                           tt_b.robot_id: tt_b.projected_arrival(i_b)},
            ))
    return conflicts


def timetable_from_schedule(robot_id: Any, node_ids: Sequence, coords: Sequence[Coord],
                            etas: Sequence[float], nominal_speed: float = 1.0) -> RobotTimetable:
    return RobotTimetable(robot_id=robot_id,
                          node_ids=list(node_ids),
                          coords=[(float(c[0]), float(c[1])) for c in coords],
                          etas=[float(t) for t in etas],
                          nominal_speed=float(nominal_speed))
