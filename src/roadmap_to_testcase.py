"""Convert AOC-CBS's external MAPF benchmarks into this project's test_case.json format.

`external/AOC-CBS/data/external/` ships two unrelated benchmark families that this script turns
into `data/test_cases/<name>.json` files runnable through `main.py`'s `general_funct()` (with
`scheduler_backend="occbs"` or `"aoccbs"` -- both need every job pre-pinned to one robot with a
single-node chain, which is exactly what a start/goal MAPF instance already is, so no assignment
step is needed):

* **ccbs_roadmaps** (`dense`, `sparse`, `super-dense`): a GraphML `map.xml` roadmap plus 25
  `<n>_task.xml` files of `<agent start_id goal_id/>` pairs. This is already a continuous node
  graph in the same shape as `test_data.nodes`, so the conversion is graph-to-graph with no
  geometry recomputed: node ids and (x, y) coordinates are carried over verbatim, and a task's
  start/goal ids map straight onto node labels.

* **movingai** (e.g. `empty-16-16`, `den312d`, ...): a MovingAI `.map` grid plus `.scen` scenario
  files. There is no roadmap here to reuse, so this script builds one. `--method sampled`
  (the default) reuses AOC-CBS's own roadmap construction verbatim --
  `aoccbs.models.model1.roadmap.sample_roadmap_from_grid`, called directly on the map's
  blocked-cell mask rather than through `benchmarking.movingai.map_to_sampled_roadmap` so nothing
  is written into AOC-CBS's model library -- which is the dart-throwing + Voronoi-backbone +
  stretch-shortcut sampling method documented in that module and in the appendix of the AOC-CBS
  paper: points are placed by Poisson-disc dart throwing at a target density, joined along an
  obstacle-aware Voronoi backbone, then given shortcut edges wherever the backbone detours far
  around a straight line. Since a sampled vertex bears no relation to any grid cell, the
  scenario's (start, goal) cells are matched to their nearest sampled vertex instead of reused
  verbatim, skipping any pair whose start or goal vertex collides with one already claimed by an
  earlier agent. `--method grid` keeps the previous one-node-per-free-cell behaviour (edges to
  orthogonal, and optionally diagonal corner-cutting-excluded, free neighbours), for when an
  exact grid graph is wanted instead. Both methods flip MovingAI's y (which counts *downwards*
  from the map's top row) so the emitted (x, y) reads right-side up under this project's plotting
  convention (y up), unlike AOC-CBS's own modules, which deliberately keep MovingAI's orientation
  and flip only at drawing time. The grid method's node labels are `"{x}_{y}"` in the flipped
  frame; the sampled method keeps the roadmap's own `"v<i>"` labels.

Both converters emit the same test_case shape as `data/test_cases/4SmallNu.json`: one job per
robot (`location` = goal, empty `precedence`/`TW`, `Service` 0, single-candidate `ATR`).
`Big_number`/`Autonomy`/`charging_coefficient` are set generously high so battery constraints
never bind (these instances have no notion of recharging). `hub_nodes` is left empty.

`ccbs_roadmaps`' `map.xml` carries no obstacle geometry at all -- it is a bare weighted graph over
continuous coordinates, with no walls to reconstruct -- so its converter can only build a map, not
recover one: `data/schedule_demo2_data/<out>/map.json` is a rectangular boundary around the
roadmap's node extent, padded by `--margin` (default 10 units) on every side, with an empty
`obstacle_list`. That is enough to make `controller=True` runnable (open free space bounded by the
map extent, so only fleet/inter-robot avoidance is exercised, not static obstacle avoidance); pass
`--no-environment` to skip writing it and leave `Environment` `null`, as before (scheduler only).
`movingai` does have a grid to build a real map from, so its converter also writes
`data/schedule_demo2_data/<out>/{map.json,graph.json}` -- this project's MPC obstacle map and
roadmap format (see `src/run_mpc.py` and `src/basic_map/map_geometric.py`/`graph.py`) -- and
points the test case's `Environment` at it, so `controller=True` works out of the box. `map.json`'s
obstacles are a greedy tiling of the map's blocked cells into axis-aligned rectangles
(`_merge_blocked_rectangles`; not minimal, but far fewer than one per cell for MovingAI's
wall-like obstacle regions); `graph.json` mirrors the test case's own node graph exactly, since
`GlobalPathCoordinator.get_node_id` looks a scheduled node up by exact coordinate match against
it. Pass `--no-environment` to skip this and leave `Environment` `null`, as before.

**Corridor width vs. robot footprint.** A MovingAI cell is a 1x1 unit square, but this project's
default `robot_spec.yaml` inflates every obstacle outward by `vehicle_width + vehicle_margin` =
0.7 units (see `pkg_motion_plan/global_path_coordinate.py:inflate_map`). A single-cell-wide
corridor -- one free cell between two blocked ones, common in `maze-*`/`room-*`/`den*` -- is only
1 unit wide wall-to-wall, so it closes completely after inflation (`2*0.7 > 1`); MovingAI's own
agent convention (radius ~=0.354) does not have this problem, but this project's default robot
does. `--cell-size` (default 1.0) scales every coordinate, nodes and obstacles alike, before
writing, so `--cell-size 3` (say) widens every corridor to 3 units without touching
`robot_spec.yaml`; pick something the fleet can actually pass through, or lower
`vehicle_width`/`vehicle_margin` instead.

Usage (from the project root):

    python src/roadmap_to_testcase.py ccbs-list
    python src/roadmap_to_testcase.py ccbs --density dense --task 1 --n-agents 10 --out ccbs_dense_1_10
    python src/roadmap_to_testcase.py ccbs --density sparse --task 1 --n-agents 4 --out ccbs_sparse_1_4 \\
        --no-environment  # scheduler-only, as before -- no boundary/map written

    python src/roadmap_to_testcase.py movingai-list
    python src/roadmap_to_testcase.py movingai --map empty-16-16 --scenario empty-16-16-random-1 \\
        --n-agents 8 --cell-size 3 --out movingai_empty16_1_8
    python src/roadmap_to_testcase.py movingai --map empty-16-16 --scenario empty-16-16-random-1 \\
        --n-agents 8 --method grid --connectedness 4 --no-environment --out movingai_empty16_1_8_grid

Both subcommands write to `data/test_cases/<out>.json`, ready for
`general_funct(problem="<out>", scheduler_backend="occbs")` (or `"aoccbs"`).
"""

import argparse
import json
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
CCBS_ROADMAPS_DIR = PROJECT_ROOT / "external" / "AOC-CBS" / "data" / "external" / "ccbs_roadmaps"
MOVINGAI_DIR = PROJECT_ROOT / "external" / "AOC-CBS" / "data" / "external" / "movingai"
TEST_CASES_DIR = PROJECT_ROOT / "data" / "test_cases"
SCHEDULE_DATA_DIR = PROJECT_ROOT / "data" / "schedule_demo2_data"

_GRAPHML_NS = "{http://graphml.graphdrawing.org/xmlns}"

# Large enough that no generated instance ever hits an autonomy/recharge constraint -- these
# benchmarks have no notion of a battery, so the scheduler should never see one bind.
DEFAULT_BIG_NUMBER = 100000
DEFAULT_AUTONOMY = 100000
DEFAULT_CHARGING_COEFFICIENT = 1


def _build_test_case(nodes: dict, start_goal_pairs: list, n_agents: int, environment=None) -> dict:
    if n_agents > len(start_goal_pairs):
        raise ValueError(
            f"requested {n_agents} agents but only {len(start_goal_pairs)} start/goal pairs are available")
    pairs = start_goal_pairs[:n_agents]

    ATRs = {}
    jobs = {}
    for i, (start, goal) in enumerate(pairs, start=1):
        robot_id = f"A{i}"
        ATRs[robot_id] = start
        jobs[f"{i:03d}"] = {
            "location": goal,
            "precedence": [],
            "TW": [],
            "Service": 0,
            "ATR": [robot_id],
        }

    return {
        "test_data": {
            "Big_number": DEFAULT_BIG_NUMBER,
            "Autonomy": DEFAULT_AUTONOMY,
            "charging_coefficient": DEFAULT_CHARGING_COEFFICIENT,
            "Environment": environment,
            "nodes": nodes,
            "hub_nodes": [],
        },
        "ATRs": ATRs,
        "jobs": jobs,
    }


def _write_environment_files(env_name: str, nodes: dict, map_data: dict) -> None:
    """Write `data/schedule_demo2_data/<env_name>/{map.json,graph.json}`, this project's MPC
    obstacle map and roadmap format (see `src/run_mpc.py` and
    `src/basic_map/map_geometric.py`/`graph.py`). `graph.json` mirrors `nodes` exactly -- same
    labels, same coordinates -- since `GlobalPathCoordinator.get_node_id`
    (src/pkg_motion_plan/global_path_coordinate.py) looks a scheduled node up by exact coordinate
    match against it.
    """
    node_dict = {label: [d["x"], d["y"]] for label, d in nodes.items()}
    edges = sorted({tuple(sorted((label, nxt))) for label, d in nodes.items() for nxt in d["next"]})
    graph_data = {"node_dict": node_dict, "edge_list": [list(e) for e in edges]}

    env_dir = SCHEDULE_DATA_DIR / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    with open(env_dir / "map.json", "w") as f:
        json.dump(map_data, f, indent=4)
    with open(env_dir / "graph.json", "w") as f:
        json.dump(graph_data, f, indent=4)


def _write(out_name: str, test_case: dict) -> pathlib.Path:
    out_path = TEST_CASES_DIR / f"{out_name}.json"
    with open(out_path, "w") as f:
        json.dump(test_case, f, indent=4)
    n_nodes = len(test_case["test_data"]["nodes"])
    n_robots = len(test_case["ATRs"])
    print(f"wrote {out_path} ({n_nodes} nodes, {n_robots} robots)")
    environment = test_case["test_data"]["Environment"]
    if environment:
        print(f"  MPC environment: data/schedule_demo2_data/{environment}/ "
              f"(controller=True is ready to use)")
    return out_path


# ---------------------------------------------------------------------------
# ccbs_roadmaps: GraphML map.xml + <n>_task.xml
# ---------------------------------------------------------------------------

def _parse_ccbs_map(map_path: pathlib.Path) -> dict:
    """Node ids and (x, y) are carried over verbatim from the GraphML 'n<k>' labels."""
    root = ET.parse(map_path).getroot()
    nodes = {}
    for node_elem in root.iter(f"{_GRAPHML_NS}node"):
        label = node_elem.get("id")
        x_str, y_str = node_elem.find(f"{_GRAPHML_NS}data").text.strip().split(",")
        nodes[label] = {"x": float(x_str), "y": float(y_str), "next": []}
    for edge_elem in root.iter(f"{_GRAPHML_NS}edge"):
        nodes[edge_elem.get("source")]["next"].append(edge_elem.get("target"))
    return nodes


def _parse_ccbs_task(task_path: pathlib.Path) -> list:
    """A task's start_id/goal_id are bare integers matching the map's 'n<k>' node labels."""
    root = ET.parse(task_path).getroot()
    return [(f"n{a.get('start_id')}", f"n{a.get('goal_id')}") for a in root.iter("agent")]


def _ccbs_task_path(density_dir: pathlib.Path, task) -> pathlib.Path:
    task = str(task)
    if task.endswith(".xml"):
        return density_dir / task
    if task.endswith("_task"):
        return density_dir / f"{task}.xml"
    return density_dir / f"{task}_task.xml"


def _write_ccbs_environment(env_name: str, nodes: dict, margin: float) -> None:
    """Build the MPC map for a ccbs_roadmaps instance. Unlike MovingAI, `map.xml` gives no
    obstacle geometry to reconstruct from -- just a weighted graph over continuous coordinates --
    so the only map available is a rectangular boundary around the roadmap's own node extent,
    padded by `margin` on every side, with an empty `obstacle_list`. `margin` must comfortably
    exceed `robot_spec.yaml`'s default inflation (`vehicle_width + vehicle_margin` = 0.7), which
    shrinks the boundary inward (`GlobalPathCoordinator.inflate_map`), or the inflated map could
    clip nodes near the edge.
    """
    xs = [d["x"] for d in nodes.values()]
    ys = [d["y"] for d in nodes.values()]
    x_lo, x_hi = min(xs) - margin, max(xs) + margin
    y_lo, y_hi = min(ys) - margin, max(ys) + margin
    map_data = {
        "boundary_coords": [[x_lo, y_lo], [x_hi, y_lo], [x_hi, y_hi], [x_lo, y_hi]],
        "obstacle_list": [],
    }
    _write_environment_files(env_name, nodes, map_data)


def convert_ccbs_roadmap(density: str, task, n_agents: int, out_name: str,
                          environment: bool = True, margin: float = 10.0) -> pathlib.Path:
    density_dir = CCBS_ROADMAPS_DIR / density
    map_path = density_dir / "map.xml"
    task_path = _ccbs_task_path(density_dir, task)
    if not map_path.is_file():
        raise FileNotFoundError(f"no map.xml under {density_dir} -- known densities: {list_ccbs_densities()}")
    if not task_path.is_file():
        raise FileNotFoundError(f"no such task file: {task_path}")

    nodes = _parse_ccbs_map(map_path)
    pairs = _parse_ccbs_task(task_path)
    env_name = None
    if environment:
        env_name = out_name
        _write_ccbs_environment(env_name, nodes, margin)
    test_case = _build_test_case(nodes, pairs, n_agents, environment=env_name)
    return _write(out_name, test_case)


def list_ccbs_densities() -> list:
    if not CCBS_ROADMAPS_DIR.is_dir():
        return []
    return sorted(p.name for p in CCBS_ROADMAPS_DIR.iterdir() if (p / "map.xml").is_file())


def print_ccbs_catalog() -> None:
    densities = list_ccbs_densities()
    if not densities:
        print(f"no ccbs_roadmaps found under {CCBS_ROADMAPS_DIR}")
        return
    for density in densities:
        density_dir = CCBS_ROADMAPS_DIR / density
        n_nodes = len(_parse_ccbs_map(density_dir / "map.xml"))
        tasks = sorted(density_dir.glob("*_task.xml"))
        print(f"{density}: {n_nodes} nodes, {len(tasks)} task files")
        for task_path in tasks:
            n_agents = len(_parse_ccbs_task(task_path))
            print(f"    {task_path.stem}: {n_agents} start/goal pairs")


# ---------------------------------------------------------------------------
# movingai: .map grid + .scen scenario
# ---------------------------------------------------------------------------

_ORTHOGONAL_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIAGONAL_MOVES = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _read_map_rows(map_path: pathlib.Path) -> list:
    lines = []
    with open(map_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    rows = lines[4:]  # skip the type/height/width/map header
    if not rows:
        raise ValueError(f"no map rows found in {map_path}")
    return rows


def _movingai_node_label(x: int, y_flipped: int) -> str:
    return f"{x}_{y_flipped}"


def _parse_movingai_map(map_path: pathlib.Path, connectedness: int) -> tuple:
    """One node per free cell ('.' or 'G'); y is flipped so the result reads right-side up."""
    if connectedness not in (4, 8):
        raise ValueError(f"connectedness must be 4 or 8, got {connectedness}")

    rows = _read_map_rows(map_path)
    height = len(rows)
    free = set()
    for y, row in enumerate(rows):
        for x, char in enumerate(row):
            if char in ".G":
                free.add((x, y))

    nodes = {}
    for (x, y) in free:
        yf = height - 1 - y
        nodes[_movingai_node_label(x, yf)] = {"x": float(x), "y": float(yf), "next": []}

    for (x, y) in free:
        label = _movingai_node_label(x, height - 1 - y)
        for dx, dy in _ORTHOGONAL_MOVES:
            if (x + dx, y + dy) in free:
                nodes[label]["next"].append(_movingai_node_label(x + dx, height - 1 - (y + dy)))
        if connectedness == 8:
            for dx, dy in _DIAGONAL_MOVES:
                if ((x + dx, y + dy) in free
                        and (x + dx, y) in free       # horizontal side clear
                        and (x, y + dy) in free):     # vertical side clear
                    nodes[label]["next"].append(_movingai_node_label(x + dx, height - 1 - (y + dy)))

    return nodes, height


def _parse_movingai_scenario(scen_path: pathlib.Path, height: int) -> list:
    pairs = []
    with open(scen_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            sx, sy, gx, gy = (int(v) for v in parts[4:8])
            start = _movingai_node_label(sx, height - 1 - sy)
            goal = _movingai_node_label(gx, height - 1 - gy)
            pairs.append((start, goal))
    return pairs


def _parse_movingai_scenario_cells(scen_path: pathlib.Path, height: int) -> list:
    """Each pair's raw (start, goal) cells as (x, y) in the flipped, right-side-up frame --
    the coordinate space `_sample_movingai_roadmap`'s node positions are in, so the two can be
    matched by nearest neighbour."""
    pairs = []
    with open(scen_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            sx, sy, gx, gy = (int(v) for v in parts[4:8])
            pairs.append(((sx, height - 1 - sy), (gx, height - 1 - gy)))
    return pairs


def _sample_movingai_roadmap(map_path: pathlib.Path, density: float, clearance: float,
                              stretch: float, max_edge_length, speed: float, seed) -> tuple:
    """Sample a roadmap over the map's free space with AOC-CBS's own construction -- dart
    throwing at `density`, an obstacle-aware Voronoi backbone, then stretch-triggered shortcuts
    -- described in `aoccbs.models.model1.roadmap` and in the appendix of the AOC-CBS paper.

    Calls `sample_roadmap_from_grid` directly (rather than going through
    `aoccbs.benchmarking.movingai.map_to_sampled_roadmap`) so nothing is written into AOC-CBS's
    own model library; the roadmap is consumed here and discarded.
    """
    from aoccbs.benchmarking.movingai import load_grid
    from aoccbs.models.model1.roadmap import sample_roadmap_from_grid

    blocked = load_grid(map_path)  # [y, x], y counting downwards from the map's top row
    height = blocked.shape[0]
    state_graph = sample_roadmap_from_grid(
        blocked, density=density, clearance=clearance, stretch=stretch,
        max_edge_length=max_edge_length, speed=speed, seed=seed,
    )

    nodes = {}
    for label, data in state_graph.graph.nodes(data=True):
        x, y = data["pos"]
        nodes[label] = {"x": float(x), "y": float(height - 1 - y), "next": []}
    for u, v in state_graph.graph.edges():
        nodes[u]["next"].append(v)

    return nodes, height


def _match_scenario_to_roadmap(scen_cells: list, nodes: dict, n_agents: int) -> list:
    """Match a scenario's (start, goal) grid cells to their nearest sampled roadmap vertex.

    A sampled vertex bears no relation to any grid cell, so this is the simplest way to reuse an
    existing MovingAI scenario's start/goal cells on a roadmap sampled over the same map. Pairs
    are kept in scenario order and stop once `n_agents` are collected; a pair is skipped outright
    if its start and goal map to the same vertex, or if either vertex was already claimed by an
    earlier pair, so no two agents share a start or a goal.
    """
    labels = list(nodes)
    positions = np.array([[nodes[label]["x"], nodes[label]["y"]] for label in labels])
    tree = cKDTree(positions)

    used = set()
    pairs = []
    for start_cell, goal_cell in scen_cells:
        start_label = labels[tree.query(start_cell)[1]]
        goal_label = labels[tree.query(goal_cell)[1]]
        if start_label == goal_label or start_label in used or goal_label in used:
            continue
        used.add(start_label)
        used.add(goal_label)
        pairs.append((start_label, goal_label))
        if len(pairs) == n_agents:
            break
    return pairs


def _merge_blocked_rectangles(blocked) -> list:
    """Greedy tiling of the blocked cells into axis-aligned rectangles.

    Not the true minimum rectangle count, but far fewer than one per cell for the wall-like
    obstacle regions typical of MovingAI maps, which is what keeps `map.json` a manageable size
    for the MPC layer to inflate and check against. Scans row-major; each new rectangle grows
    first rightwards while the row stays blocked and uncovered, then downwards while the whole
    width stays blocked and uncovered.

    args:
        blocked: boolean mask indexed [y, x], True where blocked (as `aoccbs...load_grid` returns).

    returns:
        list of (x0, y0, x1, y1): half-open cell-index bounds in the raw MovingAI frame
        (row 0 = top row), i.e. covering cells x in [x0, x1) and y in [y0, y1).
    """
    height, width = blocked.shape
    covered = np.zeros_like(blocked)
    rects = []
    for y in range(height):
        x = 0
        while x < width:
            if blocked[y, x] and not covered[y, x]:
                w = 1
                while x + w < width and blocked[y, x + w] and not covered[y, x + w]:
                    w += 1
                h = 1
                while (y + h < height
                       and np.all(blocked[y + h, x:x + w])
                       and not np.any(covered[y + h, x:x + w])):
                    h += 1
                covered[y:y + h, x:x + w] = True
                rects.append((x, y, x + w, y + h))
                x += w
            else:
                x += 1
    return rects


def _rect_to_polygon(rect: tuple, height: int, cell_size: float) -> list:
    """A blocked-cell rectangle's corners in this module's flipped, right-side-up,
    cell-size-scaled frame -- the same frame `_parse_movingai_map`'s node positions are in.

    Each cell is the unit square centred on its integer coordinate, so the polygon spans half a
    cell beyond the rectangle's cell-index bounds on every side; a 1x1 rectangle at raw cell
    (x0, y0) is therefore centred on (x0, height-1-y0), matching `_movingai_node_label`'s flip.
    """
    x0, y0, x1, y1 = rect
    x_lo, x_hi = (x0 - 0.5) * cell_size, (x1 - 0.5) * cell_size
    yf_lo, yf_hi = (height - 0.5 - y1) * cell_size, (height - 0.5 - y0) * cell_size
    return [[x_lo, yf_lo], [x_hi, yf_lo], [x_hi, yf_hi], [x_lo, yf_hi]]


def _map_boundary(height: int, width: int, cell_size: float) -> list:
    """The full map extent in the flipped frame -- symmetric under the flip, so no separate
    case is needed for which edge ends up on which side."""
    x_lo, x_hi = -0.5 * cell_size, (width - 0.5) * cell_size
    y_lo, y_hi = -0.5 * cell_size, (height - 0.5) * cell_size
    return [[x_lo, y_lo], [x_hi, y_lo], [x_hi, y_hi], [x_lo, y_hi]]


def _write_movingai_environment(env_name: str, blocked, nodes: dict, cell_size: float) -> None:
    """Build the MPC map for a MovingAI grid: a boundary matching the map extent, and obstacles
    from a greedy tiling of the grid's blocked cells. `nodes` must already be scaled by
    `cell_size` (the same scale is applied here to the obstacle geometry, since the two have to
    agree).
    """
    height = blocked.shape[0]
    map_data = {
        "boundary_coords": _map_boundary(height, blocked.shape[1], cell_size),
        "obstacle_list": [
            _rect_to_polygon(rect, height, cell_size)
            for rect in _merge_blocked_rectangles(blocked)
        ],
    }
    _write_environment_files(env_name, nodes, map_data)


def _movingai_map_dir(map_name: str) -> pathlib.Path:
    map_dir = MOVINGAI_DIR / map_name
    if not map_dir.is_dir():
        raise FileNotFoundError(f"no such movingai map folder: {map_dir} -- known maps: {list_movingai_maps()}")
    return map_dir


def _movingai_map_file(map_dir: pathlib.Path) -> pathlib.Path:
    map_files = list(map_dir.glob("*.map"))
    if len(map_files) != 1:
        raise FileNotFoundError(f"expected exactly one .map file in {map_dir}, found {len(map_files)}")
    return map_files[0]


def _movingai_scenario_path(map_dir: pathlib.Path, scenario) -> pathlib.Path:
    scenario = str(scenario)
    if not scenario.endswith(".scen"):
        scenario = f"{scenario}.scen"
    direct = map_dir / scenario
    if direct.is_file():
        return direct
    return map_dir / "scenarios" / scenario


def convert_movingai(map_name: str, scenario, n_agents: int, out_name: str, method: str = "sampled",
                      connectedness: int = 4, density: float = 0.02, clearance=None,
                      stretch: float = 1.5, max_edge_length=None, speed: float = 1.0,
                      seed=None, environment: bool = True, cell_size: float = 1.0) -> pathlib.Path:
    map_dir = _movingai_map_dir(map_name)
    map_path = _movingai_map_file(map_dir)
    scen_path = _movingai_scenario_path(map_dir, scenario)
    if not scen_path.is_file():
        raise FileNotFoundError(f"no such scenario file: {scen_path}")

    if method == "grid":
        nodes, height = _parse_movingai_map(map_path, connectedness)
        pairs = _parse_movingai_scenario(scen_path, height)
    elif method == "sampled":
        if clearance is None:
            from aoccbs.models.model1.roadmap import DEFAULT_CLEARANCE
            clearance = DEFAULT_CLEARANCE
        nodes, height = _sample_movingai_roadmap(
            map_path, density, clearance, stretch, max_edge_length, speed, seed)
        pairs = _match_scenario_to_roadmap(
            _parse_movingai_scenario_cells(scen_path, height), nodes, n_agents)
    else:
        raise ValueError(f"unknown method {method!r}, expected 'sampled' or 'grid'")

    if cell_size != 1.0:
        nodes = {label: {**d, "x": d["x"] * cell_size, "y": d["y"] * cell_size}
                 for label, d in nodes.items()}

    env_name = None
    if environment:
        from aoccbs.benchmarking.movingai import load_grid
        env_name = out_name
        _write_movingai_environment(env_name, load_grid(map_path), nodes, cell_size)

    test_case = _build_test_case(nodes, pairs, n_agents, environment=env_name)
    return _write(out_name, test_case)


def list_movingai_maps() -> list:
    if not MOVINGAI_DIR.is_dir():
        return []
    return sorted(p.name for p in MOVINGAI_DIR.iterdir() if list(p.glob("*.map")))


def print_movingai_catalog() -> None:
    maps = list_movingai_maps()
    if not maps:
        print(f"no movingai maps found under {MOVINGAI_DIR}")
        return
    for map_name in maps:
        map_dir = MOVINGAI_DIR / map_name
        scen_dir = map_dir / "scenarios"
        scenarios = sorted(scen_dir.glob("*.scen")) if scen_dir.is_dir() else sorted(map_dir.glob("*.scen"))
        print(f"{map_name}: {len(scenarios)} scenario files")
        for scen_path in scenarios:
            n_agents = sum(1 for line in open(scen_path) if len(line.split()) == 9)
            print(f"    {scen_path.stem}: {n_agents} start/goal pairs")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ccbs = sub.add_parser("ccbs", help="convert a ccbs_roadmaps map + task file")
    p_ccbs.add_argument("--density", required=True, choices=["dense", "sparse", "super-dense"])
    p_ccbs.add_argument("--task", required=True, help="task file stem or number, e.g. 1 or 1_task")
    p_ccbs.add_argument("--n-agents", type=int, required=True)
    p_ccbs.add_argument("--out", required=True, help="output test case name (no .json)")
    p_ccbs.add_argument("--no-environment", dest="environment", action="store_false",
                         help="skip writing an MPC map under data/schedule_demo2_data/<out>/ "
                              "and leave test_data.Environment null (scheduler-only test case, "
                              "the previous default)")
    p_ccbs.add_argument("--margin", type=float, default=10.0,
                         help="--environment only: padding added around the roadmap's node "
                              "extent on every side to make the map boundary, since map.xml "
                              "carries no obstacle geometry to build a tighter map from")

    sub.add_parser("ccbs-list", help="list available ccbs_roadmaps densities/task files")

    p_mai = sub.add_parser("movingai", help="convert a movingai map + scenario file")
    p_mai.add_argument("--map", required=True, dest="map_name", help="map folder name, e.g. empty-16-16")
    p_mai.add_argument("--scenario", required=True, help="scenario file stem, e.g. empty-16-16-random-1")
    p_mai.add_argument("--n-agents", type=int, required=True)
    p_mai.add_argument("--out", required=True, help="output test case name (no .json)")
    p_mai.add_argument("--method", choices=["sampled", "grid"], default="sampled",
                        help="'sampled' (default): AOC-CBS's dart-throwing + Voronoi-backbone "
                             "roadmap sampling, per the appendix of the AOC-CBS paper. "
                             "'grid': one node per free cell.")
    p_mai.add_argument("--connectedness", type=int, choices=[4, 8], default=4,
                        help="--method grid only")
    p_mai.add_argument("--density", type=float, default=0.02,
                        help="--method sampled only: roadmap vertices per unit area")
    p_mai.add_argument("--clearance", type=float, default=None,
                        help="--method sampled only: clearance kept from obstacles; "
                             "defaults to AOC-CBS's standard circular agent radius")
    p_mai.add_argument("--stretch", type=float, default=1.5, help="--method sampled only")
    p_mai.add_argument("--max-edge-length", type=float, default=None,
                        dest="max_edge_length", help="--method sampled only")
    p_mai.add_argument("--speed", type=float, default=1.0, help="--method sampled only")
    p_mai.add_argument("--seed", type=int, default=None, help="--method sampled only")
    p_mai.add_argument("--no-environment", dest="environment", action="store_false",
                        help="skip writing an MPC obstacle map/graph under "
                             "data/schedule_demo2_data/<out>/ and leave test_data.Environment "
                             "null (scheduler-only test case)")
    p_mai.add_argument("--cell-size", type=float, default=1.0,
                        help="scale factor applied to every coordinate (nodes and obstacles) "
                             "before writing; see the module docstring's note on corridor width "
                             "vs. robot footprint")

    sub.add_parser("movingai-list", help="list available movingai maps/scenario files")

    args = parser.parse_args()

    if args.command == "ccbs":
        convert_ccbs_roadmap(args.density, args.task, args.n_agents, args.out,
                              environment=args.environment, margin=args.margin)
    elif args.command == "ccbs-list":
        print_ccbs_catalog()
    elif args.command == "movingai":
        convert_movingai(
            args.map_name, args.scenario, args.n_agents, args.out, method=args.method,
            connectedness=args.connectedness, density=args.density, clearance=args.clearance,
            stretch=args.stretch, max_edge_length=args.max_edge_length, speed=args.speed,
            seed=args.seed, environment=args.environment, cell_size=args.cell_size)
    elif args.command == "movingai-list":
        print_movingai_catalog()


if __name__ == "__main__":
    main()
