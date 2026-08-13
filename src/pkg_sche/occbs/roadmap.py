"""Convert a test-case graph into the GraphML roadmap OC-CBS expects.

The OC-CBS roadmap reader (`Map::get_roadmap` in map.cpp) is stricter than the
GraphML it nominally accepts, so two conventions matter here:

* Nodes are indexed by **document order**. The integer that a task file's
  ``start_id``/``goal_id`` refers to is the node's position in the file, so the
  ``n<k>`` id attribute must agree with that position.
* Edge endpoints are parsed by dropping the first character of ``source`` and
  ``target`` and reading the rest as an integer, and each edge is added in one
  direction only. An undirected connection therefore needs two edge elements.

Edge weights are written for readability but the solver ignores them: it derives
travel times from the node coordinates, which is what keeps the OC-CBS geometry
consistent with the MPC layer's view of the same map.
"""

import math
import xml.etree.ElementTree as ET


class Roadmap:
    """A test-case node graph paired with the integer ids OC-CBS uses."""

    def __init__(self, nodes: dict):
        """`nodes` is the `test_data.nodes` mapping: name -> {x, y, next}."""
        self.nodes = nodes
        # Document order fixes the solver-side ids; sort for reproducible files.
        self.names = sorted(nodes.keys())
        self.name_to_id = {name: i for i, name in enumerate(self.names)}
        self.id_to_name = {i: name for name, i in self.name_to_id.items()}

    def coord(self, name: str) -> tuple:
        node = self.nodes[name]
        return (float(node['x']), float(node['y']))

    def to_graphml(self) -> ET.ElementTree:
        graphml = ET.Element('graphml', {'xmlns': 'http://graphml.graphdrawing.org/xmlns'})
        ET.SubElement(graphml, 'key', {'id': 'key0', 'for': 'node',
                                       'attr.name': 'coords', 'attr.type': 'string'})
        ET.SubElement(graphml, 'key', {'id': 'key1', 'for': 'edge',
                                       'attr.name': 'weight', 'attr.type': 'double'})
        graph = ET.SubElement(graphml, 'graph', {'id': 'G', 'edgedefault': 'directed'})

        for name in self.names:
            node_el = ET.SubElement(graph, 'node', {'id': f'n{self.name_to_id[name]}'})
            x, y = self.coord(name)
            ET.SubElement(node_el, 'data', {'key': 'key0'}).text = f'{x},{y}'

        for name in self.names:
            for other in self.nodes[name]['next']:
                other = str(other)
                if other not in self.name_to_id:
                    raise KeyError(f"node {name!r} lists unknown neighbour {other!r}")
                edge = ET.SubElement(graph, 'edge', {
                    'source': f'n{self.name_to_id[name]}',
                    'target': f'n{self.name_to_id[other]}',
                })
                ET.SubElement(edge, 'data', {'key': 'key1'}).text = str(
                    math.dist(self.coord(name), self.coord(other)))

        return ET.ElementTree(graphml)

    def write_graphml(self, path: str) -> None:
        tree = self.to_graphml()
        ET.indent(tree, space='  ')
        tree.write(path, encoding='UTF-8', xml_declaration=True)

    def nearest_name(self, x: float, y: float, tol: float = 1e-3) -> str:
        """Map a solver-reported coordinate back to a node name.

        The solution log reports coordinates rather than ids, so recovering the
        node name means matching against the roadmap. Coordinates make a round
        trip through the XML as decimal text, hence the tolerance.
        """
        best, best_d = None, float('inf')
        for name in self.names:
            d = math.dist((x, y), self.coord(name))
            if d < best_d:
                best, best_d = name, d
        if best_d > tol:
            raise ValueError(f"coordinate ({x}, {y}) matches no roadmap node "
                             f"(closest {best} at distance {best_d})")
        return best
