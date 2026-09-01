# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for *"Combining High Level Scheduling and Low Level Control to Manage Fleets of Mobile Robots"* (Roselli, Zhang, Åkesson — IEEE/SICE SII 2026, [arXiv:2510.23129](https://arxiv.org/abs/2510.23129)).

It simulates a fleet of mobile robots (ATRs) in a plant that must complete transport tasks under **time windows, precedence constraints, and battery/recharging limits**. A high-level scheduler produces per-robot node/ETA timetables; a per-robot NMPC tracker then executes them while avoiding static obstacles and the other robots.

## Commands

There is no test suite, linter, or package manifest — just scripts. All commands must be run **from the project root**, not from `src/` (see "Path conventions" below).

```bash
pip install -r requirements.txt     # or: uv pip install -r requirements.txt
python src/build_solver.py          # ONLY for the PANOC / panoc_light backends: generate the OpEn solver into mpc_solver/
python src/main.py                  # run scheduler + simulation
python src/schedule_visualization.py  # compare planned vs. actual ETAs (Gantt / deviation plots)
```

External solvers that are not pip-only:
- **Gurobi** (routing and path-changing sub-problems) — needs a license (academic named-user works).
- **OpEn / PANOC** (NMPC, *only if* `solver_type: 'PANOC'` or `'PANOC_LIGHT'`) — needs a Rust toolchain; `build_solver.py` compiles a Rust crate into `mpc_solver/` with Python bindings. `mpc_solver/` is gitignored, so it must be rebuilt after cloning and after any change to `config/mpc_*.yaml` that alters problem dimensions or penalty count.

Z3 (`z3-solver`) and CasADi are normal pip dependencies and need no license. The default NMPC
backend is CasADi/IPOPT, which needs neither Rust nor a build step — see "Three NMPC backends".

### Running experiments

Everything is toggled by editing literals in `src/main.py`'s `__main__` block — there is no CLI. The flags on `general_funct` are:

- `problem` — test-case name from `data/test_cases/` (`4Small`, `10Large`, …).
- `scheduler` — run the scheduler and regenerate `schedule.csv` + `robot_start.json`. Only needs to run **once per test case**; afterwards it can be `False` and the controller reuses the saved CSV.
- `controller` — run the MPC simulation. If `False`, nothing is simulated.
- `naive_tracker` — use the simple proportional heading controller (`TrajectoryTracker.run_naive_step`) instead of the NMPC. This is the paper's baseline.
- `ignore_speed_ref` — drop the schedule's speed reference and track geometry only.
- `recording` — save an mp4 into `Demo/`.
- `scheduler_backend` — `"sp_comsat"` (default), `"occbs"`, or `"aoccbs"`; see "Second scheduler backend" and "Third scheduler backend" below.
- `mpc_backend` — `"casadi"`, `"panoc"`, or `"panoc_light"`, selecting the NMPC solver; `None` falls back to `solver_type` in the YAML. Ignored when `naive_tracker` is `True`. See "Three NMPC backends" below.

Simulation knobs that are *not* exposed through `main.py` live as module-level constants at the top of `run_mpc.py` (`CFG_FNAME`, `AUTORUN`, `MONITOR_COST`, `TIMEOUT`, `VERBOSE`).

## Architecture

### Two layers joined by one CSV

The scheduler and the controller are almost entirely decoupled; the whole interface between them is a flat CSV with columns **`robot_id`, `node_id`, `ETA`** (written by `main.py` to `data/schedule_demo2_data/schedule.csv`, plus `robot_start.json` for initial poses). Anything that can produce that CSV can drive the controller — that is why `pkg_motion_plan`'s Dijkstra/visibility global planner exists as a scheduler-free alternative.

### High-level scheduler — `src/pkg_sche/sp_comsat/`

`Compo_slim.Compo_slim(problem)` is the entry point. It does **not** solve one monolithic model; it runs a compositional CEGAR-style loop over four sub-solvers:

| Sub-problem | Module | Solver |
|---|---|---|
| Routing — assign tasks to vehicles, order them, respect autonomy/recharging | `E_Routing_Gurobi.py` | Gurobi MILP |
| Scheduling — visit/leave times, time windows, node & edge mutual exclusion | `scheduling_model.py` | Z3 |
| Path changing — propose alternative paths between the same nodes | `path_changer_Gurobi.py` | Gurobi |
| Route re-verification — re-validate routes after paths change | `route_checker_slim.py` | — |

The loop: solve routing → try to schedule those routes → if the schedule is UNSAT, either swap in different paths (inner loop, currently disabled via `bound = 0`) or push back to routing, which is re-solved with all previously returned route sets blocked. Capped at `routes_bound = 15` route sets. Overall verdict is a Z3 `sat`/`unsat`/`unknown`.

`scheduling_model.py` has a hard-coded `if True:` / `else:` switch choosing between `Optimize()` (minimize time-window centering + waiting at nodes + total visit times) and a plain feasibility `Solver()`. This is an experiment toggle, not dead code — check which branch is active before interpreting results.

**Open routes.** `E_Routing_Gurobi.routing` produces **open** routes: a vehicle leaves its depot
('start' task) and simply stops at the last task it serves — it no longer drives back. The 'end'
tasks that `support_functions.json_parser` still synthesizes per depot are now isolated in the
routing graph (`no_travel_to_end`), and the old closure constraint is replaced by a binary
`route_end[k, i]` — "vehicle `k` terminates at task `i`" — which enters flow conservation as
`in(i) == out(i) + route_end[k, i]`. A terminus must be a real job visit, never a depot or a
recharge dummy (which sits at a depot location anyway), and each vehicle that leaves its depot
terminates exactly once. Dropping the return legs cuts total travelling distance substantially
(4Small 336.0 → 238.0, 5Large 958.46 → 616.23).

One downstream consequence to keep in mind: `scheduling_model.schedule` asserts
`leave_node[last] == big_number`, i.e. a vehicle parks at its final node **forever**. With closed
routes that node was always the vehicle's own depot; with open routes it is an arbitrary task
node, so a parked robot permanently blocks that node (and can deadlock a corridor) for everyone
else under `one_node_at_a_time`. This is what makes `10Large` come back UNSAT from the scheduling
stage on every one of the 15 route sets — two vehicles there finish at nodes lying on each other's
paths. All the smaller instances (`2Small`, `3Small`, `4Small`, `4SmallNu`, `1Large`–`5Large`)
still solve, and faster than before.

Test cases (`data/test_cases/*.json`) hold `test_data` (graph nodes with `x`/`y`/`next`, `Environment` folder name, `Autonomy`, `charging_coefficient`, `Big_number`, `hub_nodes`), `ATRs` (robot id → start node), and `jobs` (each with `location`, `precedence`, `TW`, `Service`, `ATR`). `Compo_slim` synthesizes extra dummy recharge tasks from `start_*` jobs before building the instance.

### Second scheduler backend — `src/pkg_sche/occbs/`

An alternative to `sp_comsat` that drives **OC-CBS** (Optimal Continuous-time Conflict-Based Search) from [Adcombrink/Optimal-Continuous-CBS](https://github.com/Adcombrink/Optimal-Continuous-CBS). `runner.OCCBS(problem)` mirrors `Compo_slim`'s return shape, so `main.py` writes the same `schedule.csv` either way.

The upstream solver is C++ and is vendored, unbuilt, into the gitignored `external/`. Its `CMakeLists.txt` demands CMake ≥ 4.1, so on a machine with an older CMake compile it directly instead:

```bash
cd external/Optimal-Continuous-CBS
g++ -std=c++11 -O2 -I. -I/opt/homebrew/include main.cpp config.cpp tinyxml2.cpp \
    xml_logger.cpp map.cpp heuristic.cpp sipp.cpp task.cpp cbs.cpp simplex/*.cpp -o CCBS
```

Boost headers are the only external dependency. Generated XML lands in `data/occbs_work/<problem>/` (also gitignored).

**The two schedulers do not solve the same problem.** `sp_comsat` assigns tasks to vehicles and honours time windows, precedence, service times and battery autonomy. OC-CBS solves continuous-time MAPF: one start and one goal per agent, no tasks, no time windows, no battery. The `occbs` backend is therefore only valid on instances where assignment is already determined — every job pinned to a single ATR, with precedence forming one chain per robot. `runner.robot_task_chains` raises rather than guessing when an instance violates that; `4Small` satisfies it (and additionally has empty `TW` and zero `Service` throughout, with `Autonomy` far above what the map requires).

**Single-goal instances are the supported case.** `4SmallNu` is built for this: `4Small`'s graph and fleet, but one destination per robot (each robot crosses to the opposite corner, so all four contend for the centre). It solves as a single MAPF instance in well under a second, and none of the leg machinery below is exercised.

Because OC-CBS takes a single goal per agent while a robot may have a chain of task locations, `runner.plan_legs` cuts each chain into legs and solves all robots' k-th legs together as one MAPF instance, giving each agent a release time equal to its own arrival at the end of the previous leg. Release times are not upstream functionality — they are added by `src/pkg_sche/occbs/occbs_release_times.patch`, which must be reapplied after any re-clone of `external/` (`git apply` from the repo root, then rebuild). The patch adds `Agent::start_time`, parses a `start_time` task attribute, seeds the SIPP start node at that time, and prepends the pre-release wait to the path so conflict checking sees the agent parked at its start node.

**Multi-waypoint instances currently do not work.** With staggered release times, `4Small`'s leg 1 finds no solution even at a 300s limit, while the same leg solves instantly with all releases at zero, with any single agent released late, and with all agents released late uniformly — only the staggered combination fails. This has not been root-caused; it is not a floating-point artifact, and it reproduces with the pre-release wait section disabled. A plausible but unverified cause is CCBS's assumption that an agent which reaches its goal stays there forever, which turns early finishers into permanent obstacles for late-released agents. Until this is understood, use the `occbs` backend only on single-goal instances and `sp_comsat` for the rest.

Note also that the two backends differ in more than conflict handling. sp_comsat's routes used to be **closed** — every vehicle drove back to its depot — which is why it reported makespan 250.12 on `4SmallNu` against OC-CBS's 42; `E_Routing_Gurobi` now produces **open** routes (see "Open routes" below), so the return legs are gone and the two are closer, but sp_comsat still schedules under node/edge mutual exclusion and time windows, so the schedules remain not directly comparable on makespan.

### Third scheduler backend — `src/pkg_sche/aoccbs/`

A second alternative to `sp_comsat`, driving **AOC-CBS** (Anytime-Optimal Continuous-time CBS) from
[Adcombrink/AOC-CBS](https://github.com/Adcombrink/AOC-CBS). `runner.AOCCBS(problem)` mirrors
`OCCBS`'s and `Compo_slim`'s return shape, so `main.py` writes the same `schedule.csv` either way.
It shares `occbs`'s core restriction — no task-to-vehicle assignment, so by default an instance
must already have every job pinned to a single ATR with precedence forming one chain per robot
(`4Small` and `4SmallNu` both qualify) — and, like `occbs`, silently ignores `TW` rather than
checking it.

That restriction can be lifted with `AOCCBS(problem, assign_via_routing=True)` (or, from
`main.py`, the `assign_via_routing` flag on `general_funct`): it runs `sp_comsat`'s Gurobi
routing sub-solver (`E_Routing_Gurobi.routing`, the same MILP `Compo_slim` uses) over the
instance first to decide which robot gets which job and in what order — jobs may then list
several candidate ATRs, exactly as `sp_comsat` itself allows — and hands the resulting per-robot
chains to AOC-CBS, which only has to solve the trajectories. `_robot_task_specs_via_routing`
builds the routing sub-solver's `Instance` via `Compo_slim.build_instance` (factored out of
`Compo_slim` for this reuse) and keeps only each route's real job visits — the leading 'start'
depot task and any 'recharge' stops are dropped, since AOC-CBS has no notion of a battery (routes
are open, so there is no trailing 'end' task to drop any more). This still inherits the routing
MILP's own limits: it honours `TW`, precedence and autonomy, and it needs Gurobi even though the
plain `assign_via_routing=False` path does not.

Unlike OC-CBS, AOC-CBS is a pure-Python library and needs no compiler: it is vendored, unbuilt,
into the gitignored `external/AOC-CBS` and installed editable —

```bash
git clone https://github.com/Adcombrink/AOC-CBS.git external/AOC-CBS
pip install -e external/AOC-CBS
pip install sortedcontainers   # a real runtime dependency AOC-CBS's pyproject.toml omits
```

— then `src/pkg_sche/aoccbs/aoccbs_node_link_data.patch` must be applied (`cd external/AOC-CBS &&
git apply ../../src/pkg_sche/aoccbs/aoccbs_node_link_data.patch`), reapplied after any re-clone.
It fixes a genuine upstream/networkx incompatibility, not a project-specific behaviour change: at
networkx 3.4.2 (this project's pinned version) `json_graph.node_link_data()` still defaults to a
`'links'` key, but every AOC-CBS state-graph saver reads back `'edges'`, so saving any state graph
raises `KeyError: 'edges'` until the call sites pass `edges="edges"` explicitly.

**The key advantage over `occbs`: AOC-CBS agents natively carry an ordered task sequence with
per-task service times**, rather than one start and one goal. `runner.AOCCBS` hands each robot's
whole job chain to the solver as a single agent instead of cutting it into legs, so it does not
need `occbs`'s release-time machinery and is not subject to the "multi-waypoint instances
currently do not work" limitation documented above for `occbs` — `4Small`'s full multi-task
schedule solves directly. On `4SmallNu` (the single-goal case both backends support) the two
independent solvers agree exactly: both report makespan 42.

AOC-CBS keeps its own model library and preprocessing cache (state graphs, per-graph all-pairs
distance matrices, and pairwise intersection-interval files, the latter two expensive to
recompute) under `external/AOC-CBS/scratch/` and `external/AOC-CBS/cache/` — already inside the
gitignored `external/` tree. `runner._build_state_graph` names the state graph
`TrajPlan_<problem>` and skips rebuilding it (and thus recomputing its cache) if that id already
exists; if a test case's node graph changes, delete
`external/AOC-CBS/scratch/models/StateGraph_TrajPlan_<problem>.json` and the matching files under
`external/AOC-CBS/cache/`, or the new graph will silently solve against the old one's cache. Both
backends assume unit robot speed — an edge's travel time is its raw Euclidean length, matching
`sp_comsat`'s and OC-CBS's `support_functions.json_parser`/`roadmap.py` — so travel times are
comparable across all three backends.

### Low-level control — `src/pkg_mpc_tracker/`, `src/pkg_motion_plan/`

`run_mpc.run_mpc` is the simulation loop. Per tick, per robot:

1. `LocalTrajPlanner.get_local_ref` turns the schedule's (node, ETA) pairs into a **time-parameterized** reference over the horizon — the robot is asked to hit its scheduled ETA, not merely reach the node. This is where `ignore_speed_ref` takes effect.
2. `TrajectoryTracker.run_step` (NMPC, via whichever backend `solver_type` selects) or `run_naive_step` (proportional heading control) produces actions. Other robots are fed in as dynamic obstacles via `RobotManager.get_other_robot_states`; the inflated map supplies static ones.
3. `robot.step` advances a `UnicycleModel`, with a guard that suppresses motion when already within tolerance of the last reference (and a looser tolerance in `safe` mode).

The tracker is a small state machine (`work_mode` / `_mode`: `aligning`, `safe`, `work`, …) plus an `idle`/termination check per robot; the loop ends when every robot terminates or `TIMEOUT` ticks elapse. Actual arrival times are recorded and written to `data/schedule_demo2_data/Actual_<problem>.csv` for comparison against the planned schedule — that pairing is what `schedule_visualization.py` plots.

#### Recording actual arrivals — `pkg_motion_plan/arrival_logger.py`

`ArrivalLogger` is what produces `Actual_<problem>.csv`, in `schedule.csv`'s own
`robot_id,node_id,ETA` columns so the two join on `(robot_id, node_id)`. `run_mpc` builds one
from the schedule (`logger_from_schedule(gpc, robot_ids)`), calls `update(rid, t, xy)` once per
robot per tick — before the idle check, so a robot parked on its last node still gets that
arrival — then `finalize()` and `to_csv(...)`. Anything that can produce positions over time can
reuse it; nothing in it is MPC-specific.

Arrival is **geometric**: the robot's closest approach to the node, during the pass in which it
came within `arrival_tol` (default 0.5 m) of it. It deliberately does *not* read the local
planner's `_current_target_node`, which is what the previous inline version did: the planner
advances its target with a horizon of lookahead, so it leaves a node's index while the robot is
still short of it, and it targets the start node too briefly to register — which is why the
start node was missing from every `Actual_*.csv` produced before this. Three details matter:

- **Route order disambiguates repeated nodes.** A node only becomes eligible once its
  predecessor is settled, so a robot standing on its start node does not also timestamp the
  later revisit of that same node at t=0. (`4Small` revisits nodes on every robot.)
- **Cut corners still get a row**, timestamped at closest approach and flagged `exact=False`
  in `to_dataframe(include_flags=True)`; the CSV itself keeps only the three schedule columns.
- **The tolerance is auto-capped** at half the distance to a node's nearer route neighbour, so
  on dense roadmaps (MovingAI/CCBS graphs space nodes well under a metre apart) one node's ball
  cannot swallow the next and drop a row.

`unreached()` names the scheduled nodes that produced no arrival at all — empty on a clean run,
populated after a timeout or a failure; `run_mpc` prints it.

Configuration is YAML in `config/`, loaded through the dataclass-ish loaders in `src/configs.py`: `mpc_fast.yaml` / `mpc_default.yaml` (`MpcConfiguration`) and `robot_spec.yaml` (`CircularRobotSpecification`). Both `build_solver.py` and `run_mpc.py` name their config file independently — **keep them pointing at the same file**, or the compiled solver will not match the runtime problem dimensions.

#### Three NMPC backends — the `mpc_backend` flag, or `solver_type` in `config/mpc_*.yaml`

The tracker can drive any of three solvers over the *same* parameter vector; the block
layout in `casadi_impl.py`, `builder_panoc.py`, and `builder_panoc_light.py` deliberately
mirrors field for field, so a parameter packed for one backend is valid for the others.

| `mpc_backend` / `solver_type` | Module | Solver | Build step |
|---|---|---|---|
| `"casadi"` / `'Casadi'` (default) | `casadi_build/casadi_impl.py` | IPOPT via `ca.nlpsol`, direct multiple shooting | none — the NLP is constructed per robot in `load_motion_model` |
| `"panoc"` / `'PANOC'` | `casadi_build/builder_panoc.py` | PANOC/OpEn, compiled Rust | `python src/build_solver.py` |
| `"panoc_light"` / `'PANOC_LIGHT'` | `casadi_build/builder_panoc_light.py` | PANOC/OpEn, compiled Rust | `python src/build_solver.py` (with `panoc_builder = "panoc_light"`) |

Pick one per run with `general_funct(..., mpc_backend="panoc")`; `run_mpc.resolve_mpc_backend`
overwrites `config_mpc.solver_type` in place and prints the resolved choice. Passing `None`
keeps whatever the YAML says, so the config file remains the default and the flag is the
override. Selecting `"panoc"` or `"panoc_light"` without having run `build_solver.py` fails
immediately with a message naming the missing path, rather than as an `ImportError` deep inside
the tracker. `build_solver.py` has its own `panoc_builder` toggle (`"panoc"` or `"panoc_light"`,
independent of `solver_type` in whichever YAML it loads) and always builds whichever one that
says, so leaving the YAML on `'Casadi'` does not stop you rebuilding either PANOC variant.

`builder_panoc_light.PanocLightBuilder` is a distinct, simpler formulation ported from the
`MPC_light` reference project rather than folded into `builder_panoc.PanocBuilder`: fleet
collision costs (current-step and predictive) are active every horizon step with no
`critical_step` cutoff, and dynamic-obstacle costs are not wired in (upstream ships that code
commented out; the port keeps it commented for parity). `panoc` and `panoc_light` are built
from the *same* config file — same problem dimensions — so the two compiled solvers only need
different `optimizer_name`s to coexist under one `build_directory`; that name is derived
mechanically by `configs.panoc_light_optimizer_name` (appends `_light`) so `run_mpc.py` and
`build_solver.py` can never compute a different name for the same build. Concretely: build
both once via `build_solver.py`'s `panoc_builder` toggle (same `cfg_fname` both times), then
switch between them at run time with nothing but `mpc_backend`.

The CasADi backend puts the states in the decision vector (`w = [X, U]`, dynamics enforced as
equality constraints `g`), whereas PANOC keeps only the inputs and rolls the dynamics out inside
the cost. Consequences worth knowing:

- **Warm starts are not interchangeable.** `CasadiNMPC.shift_warm_start` shifts `[X, U]`; an
  all-zero guess would start the predicted trajectory at the origin rather than at the robot, so
  `run_solver` seeds `X` with the current state repeated over the horizon.
- **Headings must be continuous.** IPOPT sees a wrapped heading crossing ±π as a near-2π error.
  `set_current_state` accumulates measured heading, `_unwrap_reference_states` unwraps the
  reference along the horizon, and `_rebranch_warm_start` re-anchors a shifted guess onto the
  current angular branch. The cost itself uses `mpc_helper.angle_error` (an `atan2` wrap) on both
  backends.
- **Each `run_solver` call is a short penalty homotopy**, re-solving `max_outer_iter` times with
  `rho` starting at `rho_init` and scaled by `rho_factor` each pass, warm-started from the previous
  solve. All three are config keys (see below).

##### Obstacle costs use smooth surrogates, not PANOC's

Static and dynamic obstacle costs are active in `CasadiNMPC._stage_cost`, but they do **not**
reuse PANOC's expressions. PANOC's `inside_cvx_polygon` multiplies clamped half-space residuals,
so it is identically zero — gradient included — everywhere outside a polygon; PANOC tolerated
that because its ALM penalty constraints did the real work, but it leaves IPOPT with no descent
direction until a predicted state is already inside a wall. The CasADi path therefore routes
through smooth counterparts in `mpc_helper`/`mpc_cost`:

- `smooth_cvx_intrusion` summarises a polygon by its *smallest* half-space residual (positive
  only inside), giving a metric penetration depth. Residuals are divided by the normal's length
  first, because `polygon_halfspace_representation` does not return unit normals — without that
  the depth, and hence the meaning of `qstcobs`, would scale with obstacle size.
- `softplus` replaces `fmax`, in the overflow-safe form `max(x,0) + log1p(exp(-beta|x|))/beta`.
  The naive `log(1+exp(beta*x))` overflows past `beta*x ≈ 700`, easily reached with plant
  coordinates in the tens of metres.
- `cost_inside_cvx_polygon_smooth` / `cost_inside_ellipses_smooth` wrap those, keeping the
  originals' `alpha` handling and centimetre cost resolution.

`obstacle_beta` (default 10) sets sharpness: the repulsive tail outside an obstacle spans roughly
`1/beta` metres. The soft-min under-reports depth by `log(n_edges)/beta`, so a point at the centre
of a 1 m box measures 0.364 m rather than 0.5 m — harmless, since only the gradient matters, but
worth knowing when reading cost values. Lowering it widens the avoidance band up to a point and
then stops working: on a box straddling the reference, 10 gives 0.70 m of clearance and 5 gives
1.25 m, but 2 blurs the boundary so far that the robot stalls instead of going round.

Two consequences of the zero-filled parameter blocks are handled explicitly. Unused static slots
(`Nstcobs` = 10, zero-filled) would otherwise contribute a constant `softplus(0)` term per slot —
state-independent, so harmless to the optimum, but it obscures the reported cost; `_empty_slot_tol`
gates them out. Unused dynamic slots carry `alpha = 0` and so already contribute nothing.

Note also that `run_mpc.py` calls `run_step` with `full_dyn_obstacle_list=None`, so the
dynamic-obstacle cost is correct but inert in this project as it stands — other robots reach
the MPC through the fleet term, not through `o_d`.

##### Fleet collision avoidance: the predictive term is the one that matters

Both backends carry two robot-to-robot terms, and **the predictive one carries the heavy
weight** (PANOC: 1000 vs 10; CasADi: `qfleet_pred` 1000 vs `qfleet` 10):

- `cost_fleet_pred` compares the ego robot's predicted state at horizon step `k` against the
  other robots' *predicted* states at that same step (each robot publishes its last solved
  trajectory through `RobotManager.get_other_robot_states`). This is the only term that knows
  where the other robots are going.
- `cost_fleet` compares the ego robot's whole predicted horizon against the other robots'
  positions **frozen at solve time**. For two robots crossing paths this is satisfied simply by
  driving forward, since the frozen point falls behind. It is a cheap guard near `k = 0`, not
  an avoidance mechanism.

The weights used to be the other way round on both backends (and on CasADi `cost_fleet_pred`
was commented out altogether), which made robot-robot avoidance ineffective: on `4Small` with
the `aoccbs` schedule, A1 and A2 closed monotonically from 1.08 m to contact while the
1000-weight term read exactly 0.0 on every tick and the 10-weight term was correctly predicting
a 0.2 m overlap four steps ahead. Swapping them, plus the constraint below, resolves it — the
run completes with a closest approach of 1.12 m, at the cost of 4.4 s of lateness.

On PANOC, fleet overlap is additionally an **ALM penalty constraint**, not just a cost:
`step_cost` returns `penalty_constraints_fleet` (positive only once two bodies overlap, using
the same `2*vehicle_width` test as `run_mpc`'s collision checker) alongside the static- and
dynamic-obstacle residuals, so PANOC drives it toward zero rather than pricing it. This is what
makes "arrive late" preferable to "collide" rather than merely costlier. The CasADi backend has
no equivalent yet — there it remains a soft cost only.

Both are stated against the *predicted* positions, so both inherit the unused-slot convention:
empty other-robot slots are filled with `-10.0`, which is harmless only as long as no map puts
a robot near (-10, -10).

##### Casadi-only config keys

These were literals inside `casadi_impl.py`; they now live in `config/mpc_*.yaml` and are loaded
by `MpcConfiguration` (marked `[C]` there). **Both PANOC backends ignore every one of them** —
each keeps its own built-in constants — so changing them cannot invalidate a compiled OpEn
solver, and the backends are *not* automatically comparable on these settings.

| Key | Default | Replaces |
|---|---|---|
| `qfleet` / `qfleet_pred` | 10.0 / 1000.0 | `_w_fleet` / `_w_fleet_pred`; the predictive term is the heavy one (PANOC uses the same 10 / 1000 split) |
| `fleet_safe_distance` / `fleet_critical_distance` | `null` / `null` (derived: 1.107 / 0.907 m) | the two literals in `_stage_cost` |
| `critical_step` | 100 | `_critical_step` |
| `obstacle_beta` | 10.0 | `_obstacle_beta` |
| `rho_init` / `rho_factor` / `max_outer_iter` | 10.0 / 5.0 / 5 | the tracker's homotopy constants |
| `max_solver_iter` | 500 | the hard-coded `ipopt.max_iter` |

`max_solver_time` is now honoured on both backends: PANOC passes it to `with_max_duration_micros`,
CasADi converts it to `ipopt.max_cpu_time` (microseconds → seconds).

The fleet distances accept `null`, which derives them from `robot_spec.yaml` exactly as PANOC does
— `2*(vehicle_width+vehicle_margin)` = 1.107 m and `2*vehicle_width+vehicle_margin` = 0.907 m — and
`null` is now the shipped default for both. They previously carried upstream's literals (0.1 and
0.05 m), far below the robot's own body diameter of 0.707 m, so the fleet term only engaged once
the two robots already overlapped. Do not set them below 0.707 m for the shipped robot spec.

`CostMonitor` (`MONITOR_COST` in `run_mpc.py`) scores the *PANOC* cost expression, so it still
requires OpenGEN. It is built lazily in `set_monitor`, which keeps the CasADi backend usable on
installations without OpenGEN and raises a clear error if monitoring is switched on without it.

### Supporting packages

- `basic_map` — geometric maps, occupancy grids, and NetworkX graph wrappers, all serialized to/from JSON.
- `basic_obstacle`, `basic_motion_model`, `basic_casadi` — obstacle geometry, the unicycle model, and direct-multiple-shooting helpers.
- `pkg_robot` — `RobotManager`, the registry tying each robot id to its robot/controller/planner/visualizer and exposing cross-robot state.
- `visualizer` — live matplotlib animation (`MpcPlotInLoop`) and mp4 recording.

### Path conventions (a real trip hazard)

Two incompatible conventions coexist:
- `run_mpc.py`, `build_solver.py`, and `sche_to_csv.py` resolve paths from `pathlib.Path(__file__).parents[1]` — location-independent.
- `Compo_slim.py` and `schedule_visualization.py` use **relative** paths like `data/test_cases/{problem}.json`, so they only work with cwd = project root.

Meanwhile `python src/main.py` puts `src/` on `sys.path`, which is what makes the bare `from pkg_sche...` / `from configs import ...` imports resolve. So: **launch from the project root, targeting the script inside `src/`.** Running `cd src && python main.py` breaks the scheduler.

`src/pkg_sche/sp_comsat/` is the live scheduler package. The top-level `Scheduler_MPC/sp_comsat/` directory contains only stale `__pycache__` from an earlier layout — ignore it.
