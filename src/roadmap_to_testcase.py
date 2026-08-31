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
import collections
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


def _clear_aoccbs_cache(out_name: str) -> None:
    """Delete any AOC-CBS state-graph/preprocessing cache for `out_name`.

    `pkg_sche.aoccbs.runner._build_state_graph` names the state graph `TrajPlan_<problem>` and
    reuses it by that id across runs rather than rebuilding it -- see that module's docstring.
    Regenerating a test case under the same `--out` name (a different `--density`/`--stride`, a
    different map, ...) leaves the old cache in place otherwise, and it doesn't always fail loudly:
    coordinate-based node labels from the old graph can coincidentally still exist in the new one,
    so `aoccbs` can silently solve the wrong graph instead of raising. Called unconditionally from
    `_write`, so every (re)generation under a given name starts with a clean slate for it.
    """
    try:
        from aoccbs import paths as aoccbs_paths
    except ImportError:
        return  # AOC-CBS isn't installed in this environment -- nothing cached to clear.

    sg_id = f"TrajPlan_{out_name}"
    removed = []

    try:
        model_file = aoccbs_paths.state_graph_model_file(sg_id)
        model_file.unlink()
        removed.append(model_file)
    except FileNotFoundError:
        pass

    try:
        dist_file = aoccbs_paths.state_graph_distance_file(sg_id)
        dist_file.unlink()
        removed.append(dist_file)
        index_file = dist_file.parent / f"{dist_file.stem}.index.json"
        if index_file.is_file():
            index_file.unlink()
            removed.append(index_file)
    except FileNotFoundError:
        pass

    if aoccbs_paths.INTERSECTION_INTERVAL_PP_DIR.is_dir():
        for path in aoccbs_paths.INTERSECTION_INTERVAL_PP_DIR.glob(f"*{sg_id}*.json"):
            path.unlink()
            removed.append(path)

    if removed:
        print(f"  cleared {len(removed)} stale AOC-CBS cache file(s) for {sg_id}")


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
    _clear_aoccbs_cache(out_name)
    return out_path


# ---------------------------------------------------------------------------
# ccbs_roadmaps: GraphML map.xml + <n>_task.xml
# ---------------------------------------------------------------------------

def _parse_ccbs_map(map_path: pathlib.Path) -> dict:
    """Node ids and (x, y) are carried over verbatim from the GraphML 'n<k>' labels.

    A handful of ccbs_roadmaps maps (e.g. `sparse`) contain a pair of distinct node ids placed at
    the exact same coordinates and an edge between them -- a zero-length edge with no well-defined
    direction. sp_comsat and OC-CBS tolerate it (they never normalise an edge vector), but
    AOC-CBS's preprocessing divides by edge length to get a direction and produces `NaN` on it,
    which later crashes as `ValueError: cannot convert float NaN to integer`. Such edges carry no
    information anyway -- both endpoints are the same point -- so they are dropped here rather
    than passed through, for every scheduler backend alike.
    """
    root = ET.parse(map_path).getroot()
    nodes = {}
    for node_elem in root.iter(f"{_GRAPHML_NS}node"):
        label = node_elem.get("id")
        x_str, y_str = node_elem.find(f"{_GRAPHML_NS}data").text.strip().split(",")
        nodes[label] = {"x": float(x_str), "y": float(y_str), "next": []}
    n_dropped = 0
    for edge_elem in root.iter(f"{_GRAPHML_NS}edge"):
        source, target = edge_elem.get("source"), edge_elem.get("target")
        if (nodes[source]["x"], nodes[source]["y"]) == (nodes[target]["x"], nodes[target]["y"]):
            n_dropped += 1
            continue
        nodes[source]["next"].append(target)
    if n_dropped:
        print(f"  dropped {n_dropped} zero-length edge(s) between coincident nodes in {map_path.name}")
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


def _bresenham_cells(x0: int, y0: int, x1: int, y1: int) -> list:
    """Integer cells from (x0, y0) to (x1, y1) inclusive, each step moving by exactly one cell
    orthogonally or diagonally -- i.e. never skipping past an intervening cell, which is what
    makes this usable as a line-of-sight check between two grid cells that are `stride` apart."""
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    cells = [(x, y)]
    while (x, y) != (x1, y1):
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
        cells.append((x, y))
    return cells


def _connected_components(nodes: dict) -> list:
    """Labels grouped by connected component, largest first is NOT guaranteed here -- callers
    that care about size order sort the result themselves."""
    seen = set()
    components = []
    for start in nodes:
        if start in seen:
            continue
        comp = {start}
        seen.add(start)
        queue = collections.deque([start])
        while queue:
            u = queue.popleft()
            for v in nodes[u]["next"]:
                if v not in seen:
                    seen.add(v)
                    comp.add(v)
                    queue.append(v)
        components.append(comp)
    return components


def _full_free_neighbors(free: set, x: int, y: int, connectedness: int):
    """Unit-step neighbours of (x, y) in the un-thinned free-cell grid, honouring the same
    diagonal corner-cutting rule as the rest of this module -- used for bridge-repair, which
    needs to route through cells `stride` skipped over rather than only the kept lattice."""
    moves = list(_ORTHOGONAL_MOVES)
    if connectedness == 8:
        moves += _DIAGONAL_MOVES
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if (nx, ny) not in free:
            continue
        if dx != 0 and dy != 0 and ((x + dx, y) not in free or (x, y + dy) not in free):
            continue
        yield nx, ny


def _bridge_path(free: set, connectedness: int, source_cells, target_cells):
    """Shortest path (list of cells, source-first) from any of `source_cells` to any of
    `target_cells`, breadth-first over every free cell -- not just kept ones -- so the bridge can
    cut through cells `stride` skipped over. Returns None if the two sets sit in different
    free-cell components of the map itself, a real map property thinning had nothing to do with.
    """
    target_cells = set(target_cells)
    prev = {cell: None for cell in source_cells}
    queue = collections.deque(source_cells)
    while queue:
        cur = queue.popleft()
        if cur in target_cells:
            path = [cur]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            path.reverse()
            return path
        for nb in _full_free_neighbors(free, *cur, connectedness):
            if nb not in prev:
                prev[nb] = cur
                queue.append(nb)
    return None


def _reconnect_thinned_components(nodes: dict, free: set, connectedness: int, height: int) -> None:
    """Repair fragmentation caused by `stride` thinning: bridge every disconnected component back
    to the largest one by inserting the shortest full-resolution path of free cells between them
    as extra nodes/edges (a targeted, local drop back to stride=1 rather than a global one).
    Mutates `nodes` in place.
    """
    components = _connected_components(nodes)
    if len(components) <= 1:
        return

    cell_by_label = {
        label: (int(info["x"]), height - 1 - int(info["y"])) for label, info in nodes.items()
    }

    components.sort(key=len, reverse=True)
    main = components[0]
    bridged, unreachable = 0, 0
    for comp in components[1:]:
        source_cells = [cell_by_label[label] for label in comp]
        target_cells = [cell_by_label[label] for label in main]
        path = _bridge_path(free, connectedness, source_cells, target_cells)
        if path is None:
            unreachable += 1
            continue

        prev_label = None
        for (x, y) in path:
            yf = height - 1 - y
            label = _movingai_node_label(x, yf)
            if label not in nodes:
                nodes[label] = {"x": float(x), "y": float(yf), "next": []}
                cell_by_label[label] = (x, y)
            if prev_label is not None:
                if label not in nodes[prev_label]["next"]:
                    nodes[prev_label]["next"].append(label)
                if prev_label not in nodes[label]["next"]:
                    nodes[label]["next"].append(prev_label)
            prev_label = label
        main = main | comp
        bridged += 1

    if bridged:
        print(f"  note: stride thinning fragmented the grid into {len(components)} components; "
              f"inserted {bridged} full-resolution bridge path(s) to reconnect them")
    if unreachable:
        print(f"  warning: {unreachable} component(s) could not be reconnected -- their "
              f"free-cell region has no path to the rest of the map regardless of stride")


def _grid_line_of_sight(free: set, x0: int, y0: int, x1: int, y1: int) -> bool:
    """Whether every cell on the Bresenham line between two grid cells is free, including the
    corner cells flanking each diagonal step (same corner-cutting check `_parse_movingai_map`
    always applied at 8-connectedness) -- needed once `stride` > 1 puts other cells' worth of
    wall between two kept nodes that a naive endpoint-only check would miss."""
    cells = _bresenham_cells(x0, y0, x1, y1)
    if any(cell not in free for cell in cells):
        return False
    for (ax, ay), (bx, by) in zip(cells, cells[1:]):
        if ax != bx and ay != by and ((ax, by) not in free or (bx, ay) not in free):
            return False
    return True


def _remove_redundant_rungs(nodes: dict) -> dict:
    """Drop a cardinal edge (P, Q) when both endpoints already have a full pass-through along the
    axis perpendicular to that edge -- i.e. both of P's and both of Q's neighbours on that other
    axis are present. This is the parallel-rail case in a corridor wider than one cell: picture a
    2-wide corridor as two rails with a rung (a straight cross-edge) at every row. A rung is only
    load-bearing where a rail's straight-through pass actually breaks -- a real corner, junction,
    dead end, or the run's open ends -- because that's the only place the rung is the sole
    remaining route between the two rails. Every interior rung, where both rails already run
    straight past it on both sides, is a duplicate of whatever rung sits at the run's boundary.

    Runs entirely off the *original* grid's neighbour structure (never off edges already removed
    by this same pass), so removal order can't cascade into removing a rung that turns out to be
    needed. Only cardinal edges are considered -- a diagonal edge (8-connectedness) is left alone,
    since "perpendicular axis" isn't well-defined for it here.

    Meant to run before `_simplify_collinear_chains`: once its rung is gone, an interior rail node
    is a plain degree-2 collinear pass-through and that pass collapses it away too. Together the
    two passes reduce a wide straight run to just its two end cross-sections -- exactly what a
    minimal roadmap through it needs, and no more.

    Only applies where the map is genuinely 1-D at that point, i.e. an actual corridor rail: a
    node is excluded from removal entirely if it has full pass-through in *both* cardinal axes at
    once (a real corridor rail never does -- it always has a wall on its outer side, capping it at
    3 neighbours). Without that guard, a true 2-D open room -- where interior cells are fully
    surrounded and so satisfy the perpendicular-pass-through test in both orientations at once --
    gets its edges stripped in both directions independently and collapses toward its bare
    perimeter, destroying interior cells a robot needs to be able to step into to let another
    robot pass. A corridor rail is never fully-surrounded, so this changes nothing there.
    """
    def has_neighbor(label: str, dx: float, dy: float) -> bool:
        target = _movingai_node_label(int(nodes[label]["x"] + dx), int(nodes[label]["y"] + dy))
        return target in nodes[label]["next"]

    def fully_surrounded(label: str) -> bool:
        return all(has_neighbor(label, dx, dy)
                   for dx, dy in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)))

    removable_edges = set()
    for label, info in nodes.items():
        if fully_surrounded(label):
            continue
        for nb in info["next"]:
            if fully_surrounded(nb):
                continue
            dx = nodes[nb]["x"] - info["x"]
            dy = nodes[nb]["y"] - info["y"]
            if dx != 0 and dy != 0:
                continue  # diagonal edge -- not handled here
            perp = [(0.0, 1.0), (0.0, -1.0)] if dx != 0 else [(1.0, 0.0), (-1.0, 0.0)]
            if (all(has_neighbor(label, pdx, pdy) for pdx, pdy in perp)
                    and all(has_neighbor(nb, pdx, pdy) for pdx, pdy in perp)):
                removable_edges.add(frozenset((label, nb)))

    result = {label: {"x": info["x"], "y": info["y"], "next": list(info["next"])}
              for label, info in nodes.items()}
    for edge in removable_edges:
        a, b = tuple(edge)
        result[a]["next"].remove(b)
        result[b]["next"].remove(a)
    return result


def _simplify_collinear_chains(nodes: dict) -> dict:
    """Drop redundant waypoints: a degree-2 node whose two incident edges run in exactly the same
    direction (it sits on a dead-straight stretch between its neighbours) is merged away, joining
    its two neighbours with one direct edge. Junctions (degree != 2) and turns (degree == 2 but
    the incoming/outgoing directions differ) are never touched.

    This is the density knob for `--method grid` that actually fits a maze-like map: `--stride`
    thins nodes by lattice position, which on a map dominated by 1-cell-wide corridors just moves
    the density problem into `_reconnect_thinned_components`'s full-resolution bridge repair --
    the corridor comes back at full density anyway, as one single-file chain with no alternative
    routing. Collapsing collinear runs instead removes exactly the nodes that were never adding
    path choice or a direction change, wherever a corridor happens to run straight, while leaving
    every junction and every turn -- and therefore every turn angle a planned path can ever
    present to the MPC -- completely unchanged. It also can't fragment the graph: it's a lossless
    topological contraction (same reachability, same turns), not a resample, so there is nothing
    for a repair pass to fix afterwards.
    """
    def direction(a_label: str, b_label: str) -> tuple:
        return (nodes[b_label]["x"] - nodes[a_label]["x"], nodes[b_label]["y"] - nodes[a_label]["y"])

    def is_collinear(label: str) -> bool:
        neighbors = nodes[label]["next"]
        if len(neighbors) != 2:
            return False
        a, b = neighbors
        return direction(a, label) == direction(label, b)

    removable = {label for label in nodes if is_collinear(label)}
    kept = {label: {"x": info["x"], "y": info["y"], "next": []}
            for label, info in nodes.items() if label not in removable}

    def walk_to_kept(prev: str, cur: str) -> str:
        while cur in removable:
            a, b = nodes[cur]["next"]
            prev, cur = cur, (b if a == prev else a)
        return cur

    seen_edges = set()
    for label in kept:
        for nb in nodes[label]["next"]:
            end = walk_to_kept(label, nb) if nb in removable else nb
            edge = frozenset((label, end))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            kept[label]["next"].append(end)
            kept[end]["next"].append(label)

    return kept


def _parse_movingai_map(map_path: pathlib.Path, connectedness: int, stride: int = 1,
                         simplify: bool = False) -> tuple:
    """One node per free cell ('.' or 'G'), or every `stride`-th free cell on both axes when
    `stride` > 1; y is flipped so the result reads right-side up.

    `stride` thins the dense one-node-per-cell grid down to a coarser lattice -- the grid
    equivalent of the sampled method's `--density`. Edges connect kept cells that are `stride`
    apart via `_grid_line_of_sight`, so a hop is only added when the straight path between the
    two nodes doesn't cross a wall that the thinning skipped over. Thinning a map with narrow
    corridors can fragment the lattice into disconnected components; `_reconnect_thinned_components`
    repairs that by bridging each one back to the largest with a full-resolution path.

    `simplify`, applied after thinning/repair, is the density knob that actually suits a map
    dominated by narrow corridors: it drops every waypoint that isn't a junction or a turn (see
    `_simplify_collinear_chains`), which is where `stride` mostly fails -- corridor cells rarely
    land on the thinned lattice, so they come back at full density via bridge repair anyway, as a
    single-file chain with no alternative routing. `simplify` alone (`stride=1`) already collapses
    those chains to their turns and junctions, usually without needing `stride` at all.
    """
    if connectedness not in (4, 8):
        raise ValueError(f"connectedness must be 4 or 8, got {connectedness}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    rows = _read_map_rows(map_path)
    height = len(rows)
    free = set()
    for y, row in enumerate(rows):
        for x, char in enumerate(row):
            if char in ".G":
                free.add((x, y))

    kept = {(x, y) for (x, y) in free if x % stride == 0 and y % stride == 0}
    if not kept:
        raise ValueError(f"stride {stride} leaves no free cells kept -- pick a smaller stride")

    nodes = {}
    for (x, y) in kept:
        yf = height - 1 - y
        nodes[_movingai_node_label(x, yf)] = {"x": float(x), "y": float(yf), "next": []}

    moves = list(_ORTHOGONAL_MOVES)
    if connectedness == 8:
        moves += _DIAGONAL_MOVES

    for (x, y) in kept:
        label = _movingai_node_label(x, height - 1 - y)
        for dx, dy in moves:
            nx, ny = x + dx * stride, y + dy * stride
            if (nx, ny) in kept and _grid_line_of_sight(free, x, y, nx, ny):
                nodes[label]["next"].append(_movingai_node_label(nx, height - 1 - ny))

    if stride > 1:
        _reconnect_thinned_components(nodes, free, connectedness, height)

    if simplify:
        nodes = _remove_redundant_rungs(nodes)
        nodes = _simplify_collinear_chains(nodes)

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
        print('direct')
        return direct
    return map_dir / "scenarios" / f'{map_dir.name}-{scenario}'


def convert_movingai(map_name: str, scenario, n_agents: int, out_name: str, method: str = "sampled",
                      connectedness: int = 4, stride: int = 1, simplify: bool = False,
                      density: float = 0.02, clearance=None,
                      stretch: float = 1.5, max_edge_length=None, speed: float = 1.0,
                      seed=None, environment: bool = True, cell_size: float = 1.0) -> pathlib.Path:
    map_dir = _movingai_map_dir(map_name)
    map_path = _movingai_map_file(map_dir)
    scen_path = _movingai_scenario_path(map_dir, scenario)
    if not scen_path.is_file():
        raise FileNotFoundError(f"no such scenario file: {scen_path}")

    if method == "grid":
        nodes, height = _parse_movingai_map(map_path, connectedness, stride, simplify)
        if stride == 1 and not simplify:
            pairs = _parse_movingai_scenario(scen_path, height)
        else:
            # A scenario's start/goal cells generally don't survive thinning (stride > 1) or
            # collinear-chain collapse (simplify), so snap them to their nearest surviving node,
            # same as the sampled method does for its own (unrelated) reason.
            pairs = _match_scenario_to_roadmap(
                _parse_movingai_scenario_cells(scen_path, height), nodes, n_agents)
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
    p_mai.add_argument("--stride", type=int, default=1,
                        help="--method grid only: keep only every stride-th free cell on each "
                             "axis (1 = one node per free cell, the previous default behaviour). "
                             "Raise it to thin a grid graph that has too many nodes; edges "
                             "between kept cells are added only when the straight path between "
                             "them doesn't cross a wall the thinning skipped over. On a map with "
                             "narrow corridors this mostly just relocates the density into "
                             "bridge-repair (see --simplify), so prefer --simplify there.")
    p_mai.add_argument("--simplify", action="store_true",
                        help="--method grid only: drop every waypoint that is neither a junction "
                             "nor a turn, merging dead-straight runs of collinear cells into one "
                             "edge. This is the density knob that suits maze-like/corridor-heavy "
                             "maps: unlike --stride it can't fragment the graph and never changes "
                             "any turn angle (junctions and turns are kept exactly as-is), so it's "
                             "safe with a tight MPC turning-angle limit. Combine with --stride if "
                             "you also want open areas thinned.")
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
            connectedness=args.connectedness, stride=args.stride, simplify=args.simplify,
            density=args.density, clearance=args.clearance,
            stretch=args.stretch, max_edge_length=args.max_edge_length, speed=args.speed,
            seed=args.seed, environment=args.environment, cell_size=args.cell_size)
    elif args.command == "movingai-list":
        print_movingai_catalog()


if __name__ == "__main__":
    main()
