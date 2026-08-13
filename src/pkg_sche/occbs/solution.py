"""Read an OC-CBS solution log back into this project's schedule format.

The solver writes `<task>_log.xml` next to the task file it was given. Each
agent's path is a list of sections carrying endpoint coordinates and a duration
-- there are no node ids and no absolute times, so both are reconstructed here:
times by accumulating durations from zero, node ids by matching coordinates
against the roadmap.

A section whose start and goal coincide is a wait. The robot is already at that
node, so the wait extends its departure without producing a second arrival.
"""

import xml.etree.ElementTree as ET

from .roadmap import Roadmap


class NoSolution(Exception):
    """Raised when the log holds no path for an agent OC-CBS was asked about."""


def parse_log(log_path: str, roadmap: Roadmap, agent_names: list) -> dict:
    """Return {robot_id: [(node_name, eta), ...]} from a solution log.

    `agent_names` maps agent index to robot id, in the order the agents were
    written to the task file.
    """
    root = ET.parse(log_path).getroot()
    log = root.find('log')
    if log is None:
        raise NoSolution(f"{log_path} contains no <log> element")

    schedules = {}
    for agent_el in log.findall('agent'):
        index = int(agent_el.get('number'))
        try:
            robot_id = agent_names[index]
        except IndexError:
            raise NoSolution(f"log reports agent {index} but only "
                             f"{len(agent_names)} agents were submitted")

        path_el = agent_el.find('path')
        if path_el is None:
            raise NoSolution(f"no path found for agent {index} ({robot_id})")

        timetable = []
        time = 0.0
        for i, section in enumerate(path_el.findall('section')):
            start = (float(section.get('start_i')), float(section.get('start_j')))
            goal = (float(section.get('goal_i')), float(section.get('goal_j')))
            duration = float(section.get('duration'))

            if i == 0:
                timetable.append((roadmap.nearest_name(*start), 0.0))

            time += duration
            if goal != start:
                timetable.append((roadmap.nearest_name(*goal), time))

        schedules[robot_id] = timetable

    missing = set(agent_names) - set(schedules)
    if missing:
        raise NoSolution(f"log contains no path for {sorted(missing)}")
    return schedules


def parse_summary(log_path: str) -> dict:
    """Return the solver's own summary statistics for the run."""
    root = ET.parse(log_path).getroot()
    summary = root.find('log/summary')
    if summary is None:
        return {}
    return {k: float(v) for k, v in summary.attrib.items()}


def stitch(legs: list) -> dict:
    """Join per-leg schedules into one timetable per robot.

    Each leg is a {robot_id: [(node, eta), ...]} mapping. Because every agent is
    given a release time equal to its arrival at the end of the previous leg,
    the times are already on a common absolute clock and need no shifting; the
    only fix-up is dropping the hand-over node each leg repeats from the last.
    """
    combined: dict = {}
    for leg in legs:
        for robot_id, timetable in leg.items():
            if robot_id in combined and timetable:
                # The first node of this leg is where the previous leg ended.
                timetable = timetable[1:]
            combined.setdefault(robot_id, []).extend(timetable)
    return combined
