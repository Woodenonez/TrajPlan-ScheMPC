"""Tunables for the coordinator layer."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CoordinatorConfig:
    """Every threshold the coordinator uses. Nothing here is read from a YAML file.

    The distances the coordinator compares against come from `robot_spec.yaml` via
    `configs.resolve_fleet_distances`, not from this class -- what lives here is the
    coordinator's own policy: how long to look ahead, how much schedule separation to
    insist on, and how quickly to escalate.
    """

    # --- detection -----------------------------------------------------------------
    lookahead_s: float = 20.0
    """How far into the projected timetable to look for a close approach. Past this the
    'delay stays constant' projection is too crude to act on."""

    proximity_clearance_m: Optional[float] = None
    """Two robots' projected positions must stay at least this far apart at every shared
    future time within the lookahead window; closer is a conflict. `None` derives the
    NMPC's own fleet safe distance (`configs.resolve_fleet_distances`), so the coordinator
    intervenes exactly when the tracker's own predictive avoidance term would already be
    under strain -- before it, ideally, since the coordinator sees the whole schedule while
    the NMPC only sees its own horizon."""

    proximity_sample_dt_s: float = 0.5
    """Time step used to sample each robot's projected position curve when checking for a
    close approach. Finer catches brief crossings a coarser grid could step over; coarser
    is cheaper. It has nothing to do with the simulation's own tick rate."""

    head_on_cosine: float = -0.5
    """Two robots' projected headings at the point of closest approach are classified as a
    head-on conflict when the cosine between them falls below this (roughly 120 degrees of
    difference or more), rather than a crossing. The tier ladder skips the sidestep tier for
    a head-on conflict: stepping aside on a single-lane stretch leaves the yielder just as
    much in the way, so escalation goes straight from hold to replan."""

    open_ticks: int = 3
    """Consecutive ticks a predicted overlap must persist before a conflict is opened.
    Projections are noisy just after a robot starts a new edge."""

    release_ticks: int = 5
    """Consecutive ticks with no predicted overlap before a conflict is closed, for the
    case where the robots' own drift resolved it. A measured clearance of the contested
    node closes it immediately, regardless of this."""

    min_speed_fraction: float = 0.2
    """Floor on the speed estimate used to project an arrival, as a fraction of
    `lin_vel_max`. Keeps a momentarily stationary robot from projecting an infinite ETA."""

    speed_window_ticks: int = 5
    """How many ticks of position history the measured-speed estimate averages over."""

    goal_veto_radius_m: float = 0.5
    """A robot this close to its final goal is never chosen as the yielder: holding it
    there would interfere with the tracker's own termination check."""

    # --- tiers ---------------------------------------------------------------------
    enable_hold: bool = True
    enable_crossing: bool = True
    enable_replan: bool = True

    hold_timeout_s: float = 6.0
    """Held this long without the overlap clearing -> escalate to a crossing detour.
    Roughly one full edge of travel on `4Small` plus slack."""

    detour_timeout_s: float = 8.0
    """Detour installed this long without the overlap clearing -> escalate to a replan."""

    replan_fail_hold_s: float = 10.0
    """After a failed replan, hold this long before retrying once."""

    max_interventions_per_robot: int = 5
    """Bound on how many times one robot may be intervened on, so that the
    hold -> more delay -> new conflict feedback loop cannot thrash."""

    # --- tier 2 geometry -----------------------------------------------------------
    crossing_offset_factor: float = 1.5
    """Nominal lateral offset, in units of the robot's body diameter (2*vehicle_width)."""

    detour_detour_ratio: float = 1.25
    """Reject a detour whose routed length exceeds this multiple of the straight-line
    waypoint chain -- that means the visibility planner went around a block rather than
    sidestepping."""

    # --- tier 3 --------------------------------------------------------------------
    agent_radius: Optional[float] = None
    """Disc radius the SIPP replan plans with. `None` derives `vehicle_width +
    vehicle_margin`, so the replan's clearance model matches the NMPC's fleet safe
    distance. Planning at AOC-CBS's smaller default would let the replan route robots
    past each other closer than the coordinator itself tolerates."""

    # --- scheduled waits -----------------------------------------------------------
    honour_waits: bool = True
    """Hold a robot whose reference speed has collapsed because its schedule tells it to
    wait, rather than letting the reference degenerate. Automatically disabled when the
    run ignores the speed reference."""

    wait_speed_floor_fraction: float = 0.05
    """Implied reference speed below this fraction of `lin_vel_max` counts as a
    scheduled wait."""
