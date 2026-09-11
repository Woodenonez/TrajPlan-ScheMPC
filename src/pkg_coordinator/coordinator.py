"""The coordinator: watch the fleet's schedules drift, and intervene before they collide.

The scheduler hands down a timetable whose node separations are, on paper, conflict-free.
Execution then drifts -- a robot rounds a corner wide, waits out a tracking error, or is
slowed by the NMPC avoiding someone -- and the separation the scheduler arranged stops
holding. This layer watches for that: it projects every robot's remaining route forward as
a continuous position-over-time curve and looks for two robots' curves coming within a
clearance distance of each other at a shared future time -- a *geometric* proximity test,
not a test on whether the two schedules happen to name the same node. When it finds one it
intervenes on one of the two robots, escalating only as far as it has to:

    hold  ->  sidestep  ->  replan

The robot that keeps its slot is the one the *schedule* put first, so the coordinator is
restoring the scheduler's intended ordering rather than inventing a new one.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd # type: ignore

from configs import resolve_fleet_distances
from status_log import status

from .config import CoordinatorConfig
from .timetable import Conflict, RobotTimetable, find_conflicts, timetable_from_schedule


TIER_NAMES = {0: 'observed', 1: 'hold', 2: 'crossing', 3: 'replan', 4: 'exhausted'}


@dataclass
class ActiveConflict:
    """A conflict the coordinator has opened and is now managing."""

    key: tuple
    kind: str
    priority: Any
    yielder: Any
    opened_t: float
    priority_node_index: int
    """Where the contested node sits in the priority robot's timetable. The conflict is
    over once that robot has actually progressed past it."""

    snapshot: Conflict = None       # type: ignore[assignment]
    tier: int = 0
    tier_since_t: float = 0.0
    breach_ticks: int = 0
    clear_ticks: int = 0
    max_deficit_m: float = 0.0
    replan_attempts: int = 0
    retry_after_t: Optional[float] = None
    opened_logged: bool = False

    @property
    def node(self) -> Any:
        """The priority robot's own nearest route node at the point of closest approach."""
        return self.snapshot.nodes.get(self.priority) if self.snapshot else None

    @property
    def yielder_node(self) -> Any:
        """The yielder's own nearest route node at the point of closest approach -- kept
        apart from `node` because a geometric conflict need not sit on a node shared by
        both robots' routes."""
        return self.snapshot.nodes.get(self.yielder) if self.snapshot else None


class Coordinator:
    """Per-tick conflict monitoring and intervention. One instance per simulation run."""

    def __init__(self, robot_manager, robot_ids, gpc, config_mpc, config_robot,
                 arrival_logger, problem: str, test_case_path: str,
                 config: Optional[CoordinatorConfig] = None,
                 scheduler_backend: Optional[str] = None,
                 assign_via_routing: bool = False,
                 naive_tracker: bool = False,
                 ignore_speed_ref: bool = False,
                 verbose: bool = False) -> None:
        self.rm = robot_manager
        self.robot_ids = list(robot_ids)
        self.gpc = gpc
        self.cfg_mpc = config_mpc
        self.cfg_robot = config_robot
        self.arrival_logger = arrival_logger
        self.problem = problem
        self.test_case_path = test_case_path
        self.cfg = config or CoordinatorConfig()
        self.scheduler_backend = scheduler_backend
        self.assign_via_routing = assign_via_routing
        self.verbose = verbose

        self.safe_distance, self.critical_distance = resolve_fleet_distances(config_mpc, config_robot)
        self.v_max = float(config_robot.lin_vel_max)
        self.proximity_clearance_m = (self.cfg.proximity_clearance_m
                                      if self.cfg.proximity_clearance_m is not None
                                      else self.safe_distance)

        self._timetables: dict[Any, RobotTimetable] = {}
        self._conflicts: dict[tuple, ActiveConflict] = {}
        # A robot can be the yielder in more than one conflict at once, so a hold is a set of
        # reasons rather than one: it stays still until every one of them is resolved.
        self._held: dict[Any, set] = {}
        self._pos_history: dict[Any, deque] = {rid: deque(maxlen=max(2, self.cfg.speed_window_ticks))
                                               for rid in self.robot_ids}
        self._interventions: dict[Any, int] = {rid: 0 for rid in self.robot_ids}
        self._events: list[dict] = []
        self._replanner = None
        self._replan_disabled_reason: Optional[str] = None
        self._frozen_delay: dict[Any, bool] = {rid: False for rid in self.robot_ids}

        # A crossing detour rewrites the reference geometry; without a speed reference the
        # schedule's ETAs no longer drive the robot at all, so a scheduled wait cannot be
        # recognised from the reference speed.
        self.honour_waits = self.cfg.honour_waits and not ignore_speed_ref
        self.naive_tracker = naive_tracker

        self.enabled = self._build_timetables()
        if not self.enabled:
            print("[coordinator] disabled: the schedule carries no ETAs, so schedule-based "
                  "conflict detection has nothing to compare against")
            return

        self._check_replan_availability()
        tiers = ([name for name, on in (('hold', self.cfg.enable_hold),
                                        ('crossing', self.cfg.enable_crossing),
                                        ('replan', self._replan_enabled)) if on]
                 or ['detect only'])
        status(f"Coordinator on (tiers: {', '.join(tiers)}; "
               f"proximity_clearance={self.proximity_clearance_m:.3f}m, "
               f"lookahead={self.cfg.lookahead_s:.0f}s, safe={self.safe_distance:.3f}m)")
        if self._replan_disabled_reason:
            print(f"[coordinator] replan tier disabled: {self._replan_disabled_reason}")

    # ------------------------------------------------------------------ setup

    @property
    def _replan_enabled(self) -> bool:
        return self.cfg.enable_replan and self._replan_disabled_reason is None

    def _build_timetables(self) -> bool:
        for rid in self.robot_ids:
            node_ids, etas, _ = self.gpc.get_schedule_with_node_index(rid)
            if etas is None:
                return False
            coords = [self.gpc.current_graph.get_node_coord(n) for n in node_ids]
            self._timetables[rid] = timetable_from_schedule(rid, node_ids, coords, etas, self.v_max)
        return True

    def _check_replan_availability(self) -> None:
        """Decide once whether the SIPP replan tier can run on this problem at all."""
        if not self.cfg.enable_replan:
            return
        # The replan builds its state graph from the test case's raw node keys, so the
        # schedule's node ids have to *be* those keys.
        known = ("ComSat", "aoccbs", "pp_sipp")
        if self.scheduler_backend is not None and self.scheduler_backend not in known:
            self._replan_disabled_reason = (
                f"scheduler_backend {self.scheduler_backend!r} does not emit test-case node ids")
            return
        if self.scheduler_backend == "ComSat":
            print("[coordinator] note: a SIPP replan models geometry and the other robots' fixed "
                  "plans only -- it does not honour ComSat's time windows, cross-robot precedence "
                  "or battery autonomy, so an escalation to tier 3 relaxes those constraints")

    def _ensure_replanner(self):
        if self._replanner is None:
            from .replan import SippReplanner
            radius = (self.cfg.agent_radius if self.cfg.agent_radius is not None
                      else self.cfg_robot.vehicle_width + self.cfg_robot.vehicle_margin)
            self._replanner = SippReplanner(
                problem=self.problem, test_case_path=self.test_case_path, gpc=self.gpc,
                agent_radius=radius, assign_via_routing=self.assign_via_routing,
                verbose=self.verbose)
        return self._replanner

    # ------------------------------------------------- interface used by run_mpc

    def held(self, robot_id) -> bool:
        """True if this robot is being deliberately held still this tick."""
        return bool(self._held.get(robot_id))

    def _hold(self, robot_id, reason) -> None:
        self._held.setdefault(robot_id, set()).add(reason)

    def _unhold(self, robot_id, reason) -> None:
        reasons = self._held.get(robot_id)
        if reasons is None:
            return
        reasons.discard(reason)
        if not reasons:
            del self._held[robot_id]

    def step(self, kt: int, t: float) -> None:
        """One tick of monitoring. Call after every robot has moved and been logged."""
        if not self.enabled:
            return
        self._observe(t)
        self._apply_scheduled_waits(t)
        self._update_registry(t)
        self._act(kt, t)

    # ------------------------------------------------------------- observation

    def _observe(self, t: float) -> None:
        """Refresh every robot's progress and its projected delay."""
        for rid in self.robot_ids:
            tt = self._timetables[rid]
            robot = self.rm.get_robot(rid)
            planner = self.rm.get_planner(rid)
            controller = self.rm.get_controller(rid)
            pos = np.asarray(robot.state[:2], dtype=float)
            self._pos_history[rid].append((t, pos))

            if controller.idle:
                # A finished robot sits on its final node. Freeze its delay at whatever it
                # was when it arrived, so its occupancy interval keeps reporting the time it
                # actually got there rather than sliding forward with the clock.
                tt.index = len(tt) - 1
                self._frozen_delay[rid] = True
                continue

            try:
                target = planner.current_target_node
            except AssertionError:
                target = None
            tt.advance_to_target(tuple(target) if target is not None else None)
            if tt.finished:
                continue

            dist = float(np.linalg.norm(pos - np.asarray(tt.coords[tt.index], dtype=float)))
            speed = max(self._measured_speed(rid), self.cfg.min_speed_fraction*self.v_max)
            tt.delay = (t + dist/speed) - tt.etas[tt.index]

    def _measured_speed(self, robot_id) -> float:
        history = self._pos_history[robot_id]
        if len(history) < 2:
            return 0.0
        (t0, p0), (t1, p1) = history[0], history[-1]
        dt = t1 - t0
        if dt <= 0.0:
            return 0.0
        return float(np.linalg.norm(p1 - p0)/dt)

    def _measured_delay(self, robot_id) -> Optional[float]:
        """Delay at the last node the robot actually reached, from the arrival log.

        This is measurement rather than projection, and it is what gets recorded; the live
        signal the registry runs on is the projected delay, which also accounts for how far
        the robot has got along the leg it is on right now.
        """
        last = self.arrival_logger.last_arrival(robot_id)
        if last is None:
            return None
        node_id, actual_t, _ = last
        tt = self._timetables[robot_id]
        for i in range(len(tt) - 1, -1, -1):
            if tt.node_ids[i] == node_id:
                return actual_t - tt.etas[i]
        return None

    # ------------------------------------------------------------ scheduled waits

    def _apply_scheduled_waits(self, t: float) -> None:
        """Hold robots whose own schedule tells them to stand still.

        A scheduled wait reaches the tracker as a reference speed close to zero, which the
        local planner turns into a reference collapsed onto the robot -- the degenerate case
        `relax_final_eta` exists to avoid. Holding the robot outright expresses the same
        intent without the degeneracy, and releases itself: the robot does not move, so the
        distance to its target stays put while the deadline approaches.
        """
        for rid in list(self._held):
            self._unhold(rid, 'scheduled_wait')
        if not self.honour_waits:
            return
        floor = self.cfg.wait_speed_floor_fraction*self.v_max
        for rid in self.robot_ids:
            if self.held(rid) or self._timetables[rid].finished:
                continue
            controller = self.rm.get_controller(rid)
            if controller.idle:
                continue
            tt = self._timetables[rid]
            pos = np.asarray(self.rm.get_robot(rid).state[:2], dtype=float)
            dist = float(np.linalg.norm(pos - np.asarray(tt.coords[tt.index], dtype=float)))
            if dist <= self.safe_distance/2:
                continue  # already there; the tracker's own arrival handling applies
            remaining = tt.etas[tt.index] - t
            if remaining > 0 and dist/remaining < floor:
                self._hold(rid, 'scheduled_wait')

    # --------------------------------------------------------------- registry

    def _current_conflicts(self, t: float) -> dict[tuple, Conflict]:
        found: dict[tuple, Conflict] = {}
        for i, rid_a in enumerate(self.robot_ids):
            pos_a = self._pos_history[rid_a][-1][1]
            for rid_b in self.robot_ids[i+1:]:
                pos_b = self._pos_history[rid_b][-1][1]
                for conflict in find_conflicts(
                        self._timetables[rid_a], pos_a, self._timetables[rid_b], pos_b,
                        t, self.cfg.lookahead_s, self.proximity_clearance_m,
                        self.cfg.proximity_sample_dt_s, self.cfg.head_on_cosine):
                    # Two breach windows for the same pair (of the same kind) collapse to
                    # one registry entry (see `Conflict.key`); keep whichever is worse.
                    existing = found.get(conflict.key)
                    if existing is None or conflict.separation_m < existing.separation_m:
                        found[conflict.key] = conflict
        return found

    def _update_registry(self, t: float) -> None:
        current = self._current_conflicts(t)

        for key, conflict in current.items():
            active = self._conflicts.get(key)
            if active is None:
                idx = conflict.node_indices.get(conflict.priority)
                if idx is None:
                    continue
                active = ActiveConflict(
                    key=key, kind=conflict.kind,
                    priority=conflict.priority, yielder=conflict.yielder,
                    opened_t=t, priority_node_index=idx, snapshot=conflict,
                    tier_since_t=t)
                self._conflicts[key] = active
            active.snapshot = conflict
            active.breach_ticks += 1
            active.clear_ticks = 0
            active.max_deficit_m = max(active.max_deficit_m, conflict.deficit_m)

        for key, active in list(self._conflicts.items()):
            # A conflict that has been acted on is only over once the robot that had the slot
            # has *measurably* driven past the contested node. It cannot be released on the
            # projection, because the intervention is what changed the projection: holding a
            # robot pushes its own arrival later, the predicted overlap disappears, and
            # releasing on that would let it straight back into the conflict it was held out
            # of -- the two robots then creep forward together, a metre at a time. Escalation
            # timeouts are what stop an un-clearable conflict from holding a robot forever.
            if self._timetables[active.priority].index > active.priority_node_index:
                self._release(active, t, 'priority robot cleared the node')
                continue
            if key in current:
                continue
            active.clear_ticks += 1
            if active.tier == 0 and active.clear_ticks >= self.cfg.release_ticks:
                self._release(active, t, 'projection clear before any intervention')

    def _release(self, active: ActiveConflict, t: float, reason: str) -> None:
        self._conflicts.pop(active.key, None)
        self._unhold(active.yielder, active.key)
        if active.opened_logged:
            measured = self.arrival_logger.last_arrival(active.priority)
            detail = reason
            if measured is not None:
                detail += f"; {active.priority} last reached {measured[0]} at t={measured[1]:.2f}"
            self._log(t, None, 'released', active, detail=detail,
                      duration_s=t - active.opened_t)

    # ---------------------------------------------------------------- acting

    def _act(self, kt: int, t: float) -> None:
        handled_pairs = set()
        # Head-on conflicts first: they are the ones a sidestep cannot fix, so if a pair has
        # both a head-on and a crossing conflict open the head-on one should drive the decision.
        ordered = sorted(self._conflicts.values(), key=lambda a: (a.kind != 'head_on', a.opened_t))
        for active in ordered:
            pair = frozenset((active.priority, active.yielder))
            if pair in handled_pairs:
                continue
            if active.breach_ticks < self.cfg.open_ticks:
                continue
            handled_pairs.add(pair)
            self._advance(kt, t, active)

    def _tier_ladder(self, active: ActiveConflict) -> list[int]:
        """The tiers available for this conflict, in escalation order.

        A head-on conflict skips the sidestep: stepping aside on a single-lane stretch
        leaves the robot just as much in the way, which is why that case goes straight
        from waiting to re-planning.
        """
        ladder = []
        if self.cfg.enable_hold:
            ladder.append(1)
        if self.cfg.enable_crossing and active.kind != 'head_on':
            ladder.append(2)
        if self._replan_enabled:
            ladder.append(3)
        return ladder

    def _advance(self, kt: int, t: float, active: ActiveConflict) -> None:
        if not active.opened_logged:
            yielder = self._choose_yielder(active)
            if yielder is None:
                active.opened_logged = True
                self._log(t, kt, 'unresolvable', active,
                          detail='neither robot can give way (idle, finishing, or already yielding)')
                active.tier = 4
                active.tier_since_t = t
                return
            active.yielder = yielder
            active.priority = next(r for r in active.snapshot.robots if r != yielder)
            idx = active.snapshot.node_indices.get(active.priority)
            if idx is not None:
                active.priority_node_index = idx
            active.opened_logged = True
            self._log(t, kt, 'opened', active,
                      detail=f"{active.priority} is scheduled through {active.node} first; "
                             f"{active.yielder} would pass within "
                             f"{active.snapshot.separation_m:.2f}m of it near {active.yielder_node} "
                             f"(clearance {active.snapshot.clearance_m:.2f}m)")
            self._escalate(kt, t, active)
            return

        elapsed = t - active.tier_since_t
        if active.tier == 1:
            if active.retry_after_t is not None:
                if t < active.retry_after_t:
                    return
                active.retry_after_t = None
                self._enter_replan(kt, t, active)
            elif elapsed >= self.cfg.hold_timeout_s:
                self._escalate(kt, t, active)
        elif active.tier == 2 and elapsed >= self.cfg.detour_timeout_s:
            self._escalate(kt, t, active)

    def _escalate(self, kt: int, t: float, active: ActiveConflict) -> None:
        """Move to the next enabled tier, or stop trying."""
        ladder = self._tier_ladder(active)
        if not ladder:
            return   # detection only: watch and record, never intervene
        nxt = next((tier for tier in ladder if tier > active.tier), None)
        if nxt is None:
            active.tier = 4
            active.tier_since_t = t
            self._unhold(active.yielder, active.key)
            self._log(t, kt, 'exhausted', active,
                      detail='every available tier has been tried')
            return
        if nxt == 1:
            self._enter_hold(kt, t, active)
        elif nxt == 2:
            self._enter_crossing(kt, t, active)
        else:
            self._enter_replan(kt, t, active)

    def _choose_yielder(self, active: ActiveConflict) -> Optional[Any]:
        """The schedule's loser yields, unless it cannot -- then the other one does."""
        for candidate in (active.yielder, active.priority):
            if self._can_yield(candidate):
                return candidate
        return None

    def _can_yield(self, robot_id) -> bool:
        controller = self.rm.get_controller(robot_id)
        planner = self.rm.get_planner(robot_id)
        if controller.idle or planner.idle:
            return False
        if self._interventions[robot_id] >= self.cfg.max_interventions_per_robot:
            return False
        pos = np.asarray(self.rm.get_robot(robot_id).state[:2], dtype=float)
        goal = np.asarray(controller.final_goal[:2], dtype=float)
        # Holding a robot inside its own arrival tolerance would fight the tracker's
        # termination check rather than resolve anything.
        return float(np.linalg.norm(pos - goal)) > self.cfg.goal_veto_radius_m

    def _enter_hold(self, kt: int, t: float, active: ActiveConflict) -> None:
        active.tier = 1
        active.tier_since_t = t
        self._hold(active.yielder, active.key)
        self._interventions[active.yielder] += 1
        self._log(t, kt, 'hold', active,
                  detail=f"{active.yielder} yields near {active.yielder_node}; "
                         f"{active.priority} keeps its scheduled slot at {active.node}")

    def _enter_crossing(self, kt: int, t: float, active: ActiveConflict) -> None:
        from .geometry import plan_crossing
        result = plan_crossing(self, active, t)
        if result is None:
            self._log(t, kt, 'crossing_infeasible', active,
                      detail='no lateral offset fits the inflated map here')
            self._enter_replan(kt, t, active)
            return
        active.tier = 2
        active.tier_since_t = t
        self._unhold(active.yielder, active.key)
        self._interventions[active.yielder] += 1
        self._log(t, kt, 'crossing', active,
                  detail=f"{active.yielder} sidesteps {result['offset_m']:.2f} m around "
                         f"{active.yielder_node}")

    def _enter_replan(self, kt: int, t: float, active: ActiveConflict) -> None:
        if not self._replan_enabled:
            active.tier = 4
            active.tier_since_t = t
            self._log(t, kt, 'exhausted', active,
                      detail=f"replan tier unavailable ({self._replan_disabled_reason or 'disabled'})")
            self._unhold(active.yielder, active.key)
            return

        active.replan_attempts += 1
        from .replan import apply_replan
        ok, detail = apply_replan(self, active, t)
        if ok:
            active.tier = 3
            active.tier_since_t = t
            self._unhold(active.yielder, active.key)
            self._interventions[active.yielder] += 1
            self._log(t, kt, 'replan', active, detail=detail)
            return

        self._log(t, kt, 'replan_failed', active, detail=detail)
        if active.replan_attempts >= 2:
            active.tier = 4
            active.tier_since_t = t
            self._unhold(active.yielder, active.key)
            self._log(t, kt, 'exhausted', active,
                      detail='two replan attempts failed; leaving the pair to the tracker')
            return
        # Hold a while longer and try once more: the other robot's own progress often
        # clears the contested node in the meantime.
        active.tier = 1
        active.tier_since_t = t
        active.retry_after_t = t + self.cfg.replan_fail_hold_s
        if self.cfg.enable_hold and self._can_yield(active.yielder):
            self._hold(active.yielder, active.key)

    # ------------------------------------------------------------- bookkeeping

    def replace_timetable(self, robot_id, node_ids, coords, etas) -> None:
        """Adopt a new route for a robot after a detour or a replan."""
        tt = timetable_from_schedule(robot_id, node_ids, coords, etas, self.v_max)
        self._timetables[robot_id] = tt

    def timetable(self, robot_id) -> RobotTimetable:
        return self._timetables[robot_id]

    def _log(self, t: float, kt: Optional[int], event: str, active: ActiveConflict,
             detail: str = '', duration_s: Optional[float] = None) -> None:
        snapshot = active.snapshot
        row = {
            'time': round(t, 3),
            'tick': kt,
            'event': event,
            'tier': TIER_NAMES.get(active.tier, active.tier),
            'pair': f"{active.priority}/{active.yielder}",
            'node': active.node,
            'yielder_node': active.yielder_node,
            'kind': active.kind,
            'yielder': active.yielder,
            'other': active.priority,
            'sched_eta_yielder': _get(snapshot.scheduled, active.yielder),
            'sched_eta_other': _get(snapshot.scheduled, active.priority),
            'proj_arrival_yielder': _get(snapshot.projected, active.yielder),
            'proj_arrival_other': _get(snapshot.projected, active.priority),
            'measured_delay_yielder': _round(self._measured_delay(active.yielder)),
            'measured_delay_other': _round(self._measured_delay(active.priority)),
            'approach_separation_m': round(snapshot.separation_m, 3),
            'clearance_m': round(snapshot.clearance_m, 3),
            'min_separation_m': self._diagnostic_separation(active),
            'detail': detail,
            'duration_s': None if duration_s is None else round(duration_s, 3),
        }
        self._events.append(row)
        print(f"[coordinator] t={t:7.2f} {event.upper():<20s} {active.priority}/{active.yielder} "
              f"at {active.node} ({active.kind}) -- {detail}")

    def _diagnostic_separation(self, active: ActiveConflict) -> Optional[float]:
        """How close the two robots actually are right now, in metres.

        Recorded only. Detection runs on schedules, so this is the independent measurement
        that says whether a schedule-predicted conflict corresponded to a real near-miss.
        """
        from .geometry import min_horizon_separation
        try:
            a, b = active.priority, active.yielder
            return _round(min_horizon_separation(
                np.asarray(self.rm.get_robot(a).state[:2], dtype=float),
                self.rm.get_pred_states(a),
                np.asarray(self.rm.get_robot(b).state[:2], dtype=float),
                self.rm.get_pred_states(b)))
        except Exception:
            return None

    def finalize(self, path: Optional[str] = None) -> Optional[str]:
        if not self.enabled:
            return None
        open_conflicts = len(self._conflicts)
        counts = self.summary()
        status(f"Coordinator done: {counts['holds']} holds, {counts['crossings']} crossings, "
               f"{counts['replans']} replans, {counts['unresolved']} unresolved")
        if path is None or not self._events:
            return None
        pd.DataFrame(self._events).to_csv(path, index=False)
        if self.verbose and open_conflicts:
            print(f"[coordinator] {open_conflicts} conflict(s) still open when the run ended")
        return path

    def summary(self) -> dict:
        def count(event):
            return sum(1 for row in self._events if row['event'] == event)
        return {
            'holds': count('hold'),
            'crossings': count('crossing'),
            'crossings_infeasible': count('crossing_infeasible'),
            'replans': count('replan'),
            'replans_failed': count('replan_failed'),
            'unresolved': count('exhausted') + count('unresolvable'),
            'conflicts_opened': len({row['pair'] + str(row['node']) for row in self._events}),
        }


def _get(mapping: dict, key) -> Optional[float]:
    value = mapping.get(key)
    return None if value is None else round(float(value), 3)


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 3)
