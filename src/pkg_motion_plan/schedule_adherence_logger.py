"""Per-second actual-vs-expected position log, one row per (robot, whole second).

Complements `ArrivalLogger`: that answers "when did a robot reach each node", timestamped by
the robot's own geometry; this answers "how far off its schedule-implied trajectory was a
robot at wall-clock time t", by comparing the robot's actually measured (x, y) at each whole
second against where the schedule places it at that same instant. The expected position is
piecewise-linear interpolation between consecutive scheduled (node, ETA) pairs -- i.e. it
assumes constant speed across each ETA gap, which is the schedule's own implied average speed
for that leg (distance / (ETA_next - ETA)), and needs no separately configured speed.
"""

from typing import Any, Sequence

import numpy as np
import pandas as pd # type: ignore


Coord = tuple[float, float]


class ScheduleAdherenceLogger:
    """Turn a stream of per-tick robot positions into a per-second actual-vs-expected table.

    Usage mirrors `ArrivalLogger`: register every robot's schedule once, call `update` once
    per robot per tick, then `to_dataframe`/`to_csv` when the run ends.
    """

    def __init__(self) -> None:
        self._schedules: dict[Any, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._ticks: dict[Any, list[tuple[float, float, float]]] = {}

    # ------------------------------------------------------------------ setup

    def register_robot(self, robot_id: Any, path_coords: Sequence[Coord], path_times: Sequence[float]) -> None:
        """Declare the schedule (node coords + absolute ETAs) a robot's expected position is
        measured against, in the same order `schedule.csv` visits them.

        Outside the schedule's own time span the expected position is clamped to the first
        (before the robot's first scheduled ETA) or last (after its route ends -- a robot
        parks at its final node forever, see CLAUDE.md's "open routes" note) node.
        """
        times = np.asarray(path_times, dtype=float)
        coords = np.asarray(path_coords, dtype=float).reshape(-1, 2)
        if times.shape[0] != coords.shape[0]:
            raise ValueError(f"robot {robot_id}: {coords.shape[0]} coordinates but {times.shape[0]} ETAs")
        self._schedules[robot_id] = (times, coords[:, 0], coords[:, 1])
        self._ticks[robot_id] = []

    # -------------------------------------------------------------------- run

    def update(self, robot_id: Any, time: float, position: Sequence[float]) -> None:
        """Feed one robot's position at one instant. Cheap enough to call every tick."""
        ticks = self._ticks.get(robot_id)
        if ticks is None:
            return
        ticks.append((float(time), float(position[0]), float(position[1])))

    def expected_position(self, robot_id: Any, time: float) -> Coord:
        """Where the schedule places `robot_id` at `time` (see `register_robot`)."""
        times, xs, ys = self._schedules[robot_id]
        return float(np.interp(time, times, xs)), float(np.interp(time, times, ys))

    # ----------------------------------------------------------------- output

    def to_dataframe(self) -> pd.DataFrame:
        """One row per (robot, whole second), from t_s=0 up to the last tick recorded for
        that robot, with both the measured (interpolated between the two bracketing ticks)
        and the schedule-expected position at that second."""
        rows = []
        for robot_id, ticks in self._ticks.items():
            if not ticks or robot_id not in self._schedules:
                continue
            times = np.asarray([t for t, _, _ in ticks], dtype=float)
            xs = np.asarray([x for _, x, _ in ticks], dtype=float)
            ys = np.asarray([y for _, _, y in ticks], dtype=float)
            last_t = float(times[-1])
            for t_s in range(0, int(np.floor(last_t)) + 1):
                actual_x = float(np.interp(t_s, times, xs))
                actual_y = float(np.interp(t_s, times, ys))
                expected_x, expected_y = self.expected_position(robot_id, t_s)
                rows.append({
                    "robot_id": robot_id, "t_s": t_s,
                    "actual_x": round(actual_x, 6), "actual_y": round(actual_y, 6),
                    "expected_x": round(expected_x, 6), "expected_y": round(expected_y, 6),
                })
        columns = ["robot_id", "t_s", "actual_x", "actual_y", "expected_x", "expected_y"]
        df = pd.DataFrame(rows, columns=columns)
        return df.sort_values(["robot_id", "t_s"]).reset_index(drop=True)

    def to_csv(self, path: str) -> str:
        """Write the per-second actual-vs-expected table to `path` and return the path."""
        self.to_dataframe().to_csv(path, index=False)
        return path
