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
  files. There is no roadmap here to reuse, so this script builds one: one graph node per free
  cell, edges to orthogonal (and optionally diagonal, corner-cutting excluded) free neighbours.
  MovingAI stores y counting *downwards* from the map's top row; that is flipped here so the
  emitted (x, y) reads right-side up under this project's plotting convention (y up), unlike
  AOC-CBS's own `movingai.py`, which deliberately keeps MovingAI's orientation and flips only at
  drawing time. Node labels are `"{x}_{y}"` in the flipped frame.

Both converters emit the same test_case shape as `data/test_cases/4SmallNu.json`: one job per
robot (`location` = goal, empty `precedence`/`TW`, `Service` 0, single-candidate `ATR`).
`Big_number`/`Autonomy`/`charging_coefficient` are set generously high so battery constraints
never bind (these instances have no notion of recharging); `Environment` is left `null` because
none of these benchmarks ships a matching obstacle map for the MPC layer -- run the generated
test cases with `controller=False` (scheduler only). `hub_nodes` is left empty.

Usage (from the project root):

    python src/roadmap_to_testcase.py ccbs-list
    python src/roadmap_to_testcase.py ccbs --density dense --task 1 --n-agents 10 --out ccbs_dense_1_10

    python src/roadmap_to_testcase.py movingai-list
    python src/roadmap_to_testcase.py movingai --map empty-16-16 --scenario empty-16-16-random-1 \\
        --n-agents 8 --connectedness 4 --out movingai_empty16_1_8

Both subcommands write to `data/test_cases/<out>.json`, ready for
`general_funct(problem="<out>", scheduler_backend="occbs")` (or `"aoccbs"`).
"""

import argparse
import json
import pathlib
import xml.etree.ElementTree as ET

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
CCBS_ROADMAPS_DIR = PROJECT_ROOT / "external" / "AOC-CBS" / "data" / "external" / "ccbs_roadmaps"
MOVINGAI_DIR = PROJECT_ROOT / "external" / "AOC-CBS" / "data" / "external" / "movingai"
TEST_CASES_DIR = PROJECT_ROOT / "data" / "test_cases"

_GRAPHML_NS = "{http://graphml.graphdrawing.org/xmlns}"

# Large enough that no generated instance ever hits an autonomy/recharge constraint -- these
# benchmarks have no notion of a battery, so the scheduler should never see one bind.
DEFAULT_BIG_NUMBER = 100000
DEFAULT_AUTONOMY = 100000
DEFAULT_CHARGING_COEFFICIENT = 1


def _build_test_case(nodes: dict, start_goal_pairs: list, n_agents: int) -> dict:
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
            "Environment": None,
            "nodes": nodes,
            "hub_nodes": [],
        },
        "ATRs": ATRs,
        "jobs": jobs,
    }


def _write(out_name: str, test_case: dict) -> pathlib.Path:
    out_path = TEST_CASES_DIR / f"{out_name}.json"
    with open(out_path, "w") as f:
        json.dump(test_case, f, indent=4)
    n_nodes = len(test_case["test_data"]["nodes"])
    n_robots = len(test_case["ATRs"])
    print(f"wrote {out_path} ({n_nodes} nodes, {n_robots} robots)")
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


def convert_ccbs_roadmap(density: str, task, n_agents: int, out_name: str) -> pathlib.Path:
    density_dir = CCBS_ROADMAPS_DIR / density
    map_path = density_dir / "map.xml"
    task_path = _ccbs_task_path(density_dir, task)
    if not map_path.is_file():
        raise FileNotFoundError(f"no map.xml under {density_dir} -- known densities: {list_ccbs_densities()}")
    if not task_path.is_file():
        raise FileNotFoundError(f"no such task file: {task_path}")

    nodes = _parse_ccbs_map(map_path)
    pairs = _parse_ccbs_task(task_path)
    test_case = _build_test_case(nodes, pairs, n_agents)
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


def convert_movingai(map_name: str, scenario, n_agents: int, out_name: str, connectedness: int = 4) -> pathlib.Path:
    map_dir = _movingai_map_dir(map_name)
    map_path = _movingai_map_file(map_dir)
    scen_path = _movingai_scenario_path(map_dir, scenario)
    if not scen_path.is_file():
        raise FileNotFoundError(f"no such scenario file: {scen_path}")

    nodes, height = _parse_movingai_map(map_path, connectedness)
    pairs = _parse_movingai_scenario(scen_path, height)
    test_case = _build_test_case(nodes, pairs, n_agents)
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

    sub.add_parser("ccbs-list", help="list available ccbs_roadmaps densities/task files")

    p_mai = sub.add_parser("movingai", help="convert a movingai map + scenario file")
    p_mai.add_argument("--map", required=True, dest="map_name", help="map folder name, e.g. empty-16-16")
    p_mai.add_argument("--scenario", required=True, help="scenario file stem, e.g. empty-16-16-random-1")
    p_mai.add_argument("--n-agents", type=int, required=True)
    p_mai.add_argument("--connectedness", type=int, choices=[4, 8], default=4)
    p_mai.add_argument("--out", required=True, help="output test case name (no .json)")

    sub.add_parser("movingai-list", help="list available movingai maps/scenario files")

    args = parser.parse_args()

    if args.command == "ccbs":
        convert_ccbs_roadmap(args.density, args.task, args.n_agents, args.out)
    elif args.command == "ccbs-list":
        print_ccbs_catalog()
    elif args.command == "movingai":
        convert_movingai(args.map_name, args.scenario, args.n_agents, args.out, args.connectedness)
    elif args.command == "movingai-list":
        print_movingai_catalog()


if __name__ == "__main__":
    main()
