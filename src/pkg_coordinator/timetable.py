"""Projected timetables and the geometric proximity test that triggers the coordinator.

The scheduler already separated the robots in time: it gave each one an arrival time at
every node of its route, spaced so that no two robots occupy the same node at once. What
the coordinator watches for is that separation being *undone by execution drift* -- a robot
running late arrives into a window another robot was promised.

Detection works by turning each robot's remaining route into a continuous position-over-time
curve (`RobotTimetable.position_at`) -- the schedule's own ETAs, shifted by the robot's
current projected delay, interpolated between nodes and held during any inferred dwell --
and sampling both curves across a lookahead window. A conflict opens wherever the two curves
come within a clearance distance of each other at a *shared* future time, regardless of
whether either robot's route ever names that location by the same node id. This is
deliberately not a test on node identity: two robots can be scheduled to visit different
nodes and still pass close enough to violate the fleet's safe distance (a diagonal crossing,
a corner cut, one robot's detour brushing another's straight line), and the old node/edge
matching test missed all of those. It can also flag conflicts that the old test caught
already, since two robots occupying the same node fail a proximity test at that node too.

Nothing in this module imports from the simulation; it is plain data and arithmetic, so the
detection logic can be exercised without running anything.
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
        occupies it for as long as it stands, which `dwell` captures.
        """
        start = self.projected_arrival(i)
        return start, start + self.dwell[i]

    def position_at(self, t: float, now: float, now_pos: Coord) -> Coord:
        """Predicted (x, y) at time `t >= now`.

        Walks the remaining route from the robot's actual position `now_pos`, driving each
        leg at the pace the projected (delay-shifted) arrivals imply and holding position
        for any inferred dwell, so this is a continuous parameterisation of "where the
        schedule says this robot will be" rather than just a list of node visit times. This
        is what proximity-based conflict detection samples.
        """
        if self.finished or t <= now:
            return now_pos
        prev_t, prev_pos = now, now_pos
        for i in range(self.index, len(self.node_ids)):
            arrival, depart = self.occupancy(i)
            if t <= arrival:
                span = arrival - prev_t
                frac = 0.0 if span <= 1e-9 else (t - prev_t)/span
                return _lerp(prev_pos, self.coords[i], frac)
            if t <= depart:   # `depart` is +inf at the final node -- it parks there forever
                return self.coords[i]
            prev_t, prev_pos = depart, self.coords[i]
        return self.coords[-1]

    def direction_at(self, t: float, now: float, now_pos: Coord,
                     eps: float = 0.5) -> Optional[Coord]:
        """Unit heading at time `t`, by finite difference of `position_at`. `None` if the
        robot is (projected to be) stationary at that instant, e.g. dwelling at a node."""
        p0 = self.position_at(max(now, t - eps), now, now_pos)
        p1 = self.position_at(t + eps, now, now_pos)
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return None
        return dx/norm, dy/norm

    def node_index_at(self, t: float, now: float) -> int:
        """The route entry that is "current" for the robot at time `t`: the node it is
        still heading towards, or the one it is dwelling at."""
        if self.finished:
            return len(self.node_ids) - 1
        for i in range(self.index, len(self.node_ids)):
            _, depart = self.occupancy(i)
            if t <= depart:
                return i
        return len(self.node_ids) - 1

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


def _lerp(a: Coord, b: Coord, frac: float) -> Coord:
    frac = min(max(frac, 0.0), 1.0)
    return a[0] + (b[0] - a[0])*frac, a[1] + (b[1] - a[1])*frac


@dataclass
class Conflict:
    """A projected close pass: two robots' schedules put them within clearance of each
    other at some shared future time, whether or not either schedule ever names that
    location by the same node id."""

    kind: str                 # 'crossing' or 'head_on' (from closing headings at approach)
    robots: tuple             # (priority, yielder)
    priority: Any             # the robot the schedule says goes first -- it keeps its slot
    yielder: Any              # the other one
    approach_t: float         # projected time of closest approach
    separation_m: float       # distance between the two robots' projected positions there
    clearance_m: float        # the threshold that was breached
    approach_point: dict = field(default_factory=dict)   # robot id -> its position there
    nodes: dict = field(default_factory=dict)            # robot id -> its own nearest node
    node_indices: dict = field(default_factory=dict)     # robot id -> that node's route index
    scheduled: dict = field(default_factory=dict)        # robot id -> scheduled ETA at it
    projected: dict = field(default_factory=dict)        # robot id -> schedule-projected ETA

    @property
    def deficit_m(self) -> float:
        """How far into the violation this is: positive once the two robots are projected
        to be closer together than `clearance_m` allows."""
        return self.clearance_m - self.separation_m

    @property
    def key(self) -> tuple:
        # Node is deliberately not part of the key: which route node is nearest to either
        # robot at the moment of closest approach can drift, tick to tick, as both robots
        # move -- keying on it would make the same ongoing conflict read as repeatedly
        # closing and reopening.
        return (frozenset(self.robots), self.kind)


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


def find_conflicts(tt_a: RobotTimetable, pos_a: Coord, tt_b: RobotTimetable, pos_b: Coord,
                   now: float, lookahead_s: float, clearance_m: float,
                   sample_dt_s: float = 0.5, head_on_cosine: float = -0.5) -> list[Conflict]:
    """Every predicted close pass between two robots' projected positions.

    Each robot's remaining route becomes a position-over-time curve anchored at its actual
    current position (`pos_a`/`pos_b`), sampled every `sample_dt_s` out to `lookahead_s`.
    Any contiguous stretch where the two curves come within `clearance_m` of each other
    opens one `Conflict`, reported at the point of closest approach within that stretch.
    """
    if tt_a.finished or tt_b.finished:
        return []

    n_steps = max(1, int(round(lookahead_s/sample_dt_s)))
    samples = []
    for k in range(n_steps + 1):
        t = now + k*sample_dt_s
        pa = tt_a.position_at(t, now, pos_a)
        pb = tt_b.position_at(t, now, pos_b)
        samples.append((t, math.dist(pa, pb)))

    conflicts = []
    i = 0
    while i < len(samples):
        if samples[i][1] >= clearance_m:
            i += 1
            continue
        j = i
        while j + 1 < len(samples) and samples[j + 1][1] < clearance_m:
            j += 1
        t_star, d_star = min(samples[i:j + 1], key=lambda s: s[1])
        conflicts.append(_build_conflict(tt_a, pos_a, tt_b, pos_b, now, t_star, d_star,
                                         clearance_m, head_on_cosine))
        i = j + 1
    return conflicts


def _build_conflict(tt_a: RobotTimetable, pos_a: Coord, tt_b: RobotTimetable, pos_b: Coord,
                    now: float, t_star: float, d_star: float, clearance_m: float,
                    head_on_cosine: float) -> Conflict:
    i_a = tt_a.node_index_at(t_star, now)
    i_b = tt_b.node_index_at(t_star, now)
    priority, yielder = _order(tt_a, i_a, tt_b, i_b)

    dir_a = tt_a.direction_at(t_star, now, pos_a)
    dir_b = tt_b.direction_at(t_star, now, pos_b)
    kind = 'crossing'
    if dir_a is not None and dir_b is not None:
        cosine = dir_a[0]*dir_b[0] + dir_a[1]*dir_b[1]
        if cosine < head_on_cosine:
            kind = 'head_on'

    return Conflict(
        kind=kind,
        robots=(priority, yielder),
        priority=priority,
        yielder=yielder,
        approach_t=t_star,
        separation_m=d_star,
        clearance_m=clearance_m,
        approach_point={tt_a.robot_id: tt_a.position_at(t_star, now, pos_a),
                       tt_b.robot_id: tt_b.position_at(t_star, now, pos_b)},
        nodes={tt_a.robot_id: tt_a.node_ids[i_a], tt_b.robot_id: tt_b.node_ids[i_b]},
        node_indices={tt_a.robot_id: i_a, tt_b.robot_id: i_b},
        scheduled={tt_a.robot_id: tt_a.etas[i_a], tt_b.robot_id: tt_b.etas[i_b]},
        projected={tt_a.robot_id: tt_a.projected_arrival(i_a),
                  tt_b.robot_id: tt_b.projected_arrival(i_b)},
    )


def timetable_from_schedule(robot_id: Any, node_ids: Sequence, coords: Sequence[Coord],
                            etas: Sequence[float], nominal_speed: float = 1.0) -> RobotTimetable:
    return RobotTimetable(robot_id=robot_id,
                          node_ids=list(node_ids),
                          coords=[(float(c[0]), float(c[1])) for c in coords],
                          etas=[float(t) for t in etas],
                          nominal_speed=float(nominal_speed))
