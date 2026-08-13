# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for *"Combining High Level Scheduling and Low Level Control to Manage Fleets of Mobile Robots"* (Roselli, Zhang, Åkesson — IEEE/SICE SII 2026, [arXiv:2510.23129](https://arxiv.org/abs/2510.23129)).

It simulates a fleet of mobile robots (ATRs) in a plant that must complete transport tasks under **time windows, precedence constraints, and battery/recharging limits**. A high-level scheduler produces per-robot node/ETA timetables; a per-robot NMPC tracker then executes them while avoiding static obstacles and the other robots.

## Commands

There is no test suite, linter, or package manifest — just scripts. All commands must be run **from the project root**, not from `src/` (see "Path conventions" below).

```bash
pip install -r requirements.txt     # or: uv pip install -r requirements.txt
python src/build_solver.py          # generate the PANOC/OpEn solver into mpc_solver/ (do this first)
python src/main.py                  # run scheduler + simulation
python src/schedule_visualization.py  # compare planned vs. actual ETAs (Gantt / deviation plots)
```

Two external solvers are required and are not pip-only:
- **Gurobi** (routing and path-changing sub-problems) — needs a license (academic named-user works).
- **OpEn / PANOC** (NMPC) — needs a Rust toolchain; `build_solver.py` compiles a Rust crate into `mpc_solver/` with Python bindings. `mpc_solver/` is gitignored, so it must be rebuilt after cloning and after any change to `config/mpc_*.yaml` that alters problem dimensions or penalty count.

Z3 is a normal pip dependency (`z3-solver`) and needs no license.

### Running experiments

Everything is toggled by editing literals in `src/main.py`'s `__main__` block — there is no CLI. The flags on `general_funct` are:

- `problem` — test-case name from `data/test_cases/` (`4Small`, `10Large`, …).
- `scheduler` — run the scheduler and regenerate `schedule.csv` + `robot_start.json`. Only needs to run **once per test case**; afterwards it can be `False` and the controller reuses the saved CSV.
- `controller` — run the MPC simulation. If `False`, nothing is simulated.
- `naive_tracker` — use the simple proportional heading controller (`TrajectoryTracker.run_naive_step`) instead of the NMPC. This is the paper's baseline.
- `ignore_speed_ref` — drop the schedule's speed reference and track geometry only.
- `recording` — save an mp4 into `Demo/`.
- `scheduler_backend` — `"sp_comsat"` (default) or `"occbs"`; see "Second scheduler backend" below.

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

Note also that the two backends differ in more than conflict handling: sp_comsat routes each vehicle **back to its start**, so on `4SmallNu` it reports makespan 250.12 against OC-CBS's 42. The schedules are not comparable on makespan.

### Low-level control — `src/pkg_mpc_tracker/`, `src/pkg_motion_plan/`

`run_mpc.run_mpc` is the simulation loop. Per tick, per robot:

1. `LocalTrajPlanner.get_local_ref` turns the schedule's (node, ETA) pairs into a **time-parameterized** reference over the horizon — the robot is asked to hit its scheduled ETA, not merely reach the node. This is where `ignore_speed_ref` takes effect.
2. `TrajectoryTracker.run_step` (NMPC via the compiled PANOC solver) or `run_naive_step` (proportional heading control) produces actions. Other robots are fed in as dynamic obstacles via `RobotManager.get_other_robot_states`; the inflated map supplies static ones.
3. `robot.step` advances a `UnicycleModel`, with a guard that suppresses motion when already within tolerance of the last reference (and a looser tolerance in `safe` mode).

The tracker is a small state machine (`work_mode` / `_mode`: `aligning`, `safe`, `work`, …) plus an `idle`/termination check per robot; the loop ends when every robot terminates or `TIMEOUT` ticks elapse. Actual arrival times are recorded and written to `data/schedule_demo2_data/Actual_<problem>.csv` for comparison against the planned schedule — that pairing is what `schedule_visualization.py` plots.

Configuration is YAML in `config/`, loaded through the dataclass-ish loaders in `src/configs.py`: `mpc_fast.yaml` / `mpc_default.yaml` (`MpcConfiguration`) and `robot_spec.yaml` (`CircularRobotSpecification`). Both `build_solver.py` and `run_mpc.py` name their config file independently — **keep them pointing at the same file**, or the compiled solver will not match the runtime problem dimensions.

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
