import math

import numpy as np
import networkx as nx # type: ignore
import matplotlib as mpl
mpl.use('macosx')
import matplotlib.pyplot as plt # type: ignore
from matplotlib.patches import Polygon # type: ignore
import os
import pathlib
import json

def build_Environment(Env,PrintMap=False):

    DATA_NAME = "schedule_demo2_data"  # "schedule_demo_data"
    root_dir = pathlib.Path(__file__).resolve().parents[1]
    data_dir = os.path.join(root_dir, "data", DATA_NAME)

    with open(f'{data_dir}/{Env}/map.json','r') as read_file:
        data = json.load(read_file)
        boundary = data['boundary_coords']
        obstacle_list = data['obstacle_list']

    with open(f'{data_dir}/{Env}/graph.json','r') as read_file:
        data = json.load(read_file)
        nodes = data['node_dict']
        edges = data['edge_list']

    # min_x = min([i[0] for i in nodes.values()])
    # max_x = max([i[0] for i in nodes.values()])
    # min_y = min([i[1] for i in nodes.values()])
    # max_y = max([i[1] for i in nodes.values()])
    # print(f'min_x:{min_x}, max_x:{max_x}, min_y:{min_y}, max_y:{max_y}')

    edges_rev = [[i[1],i[0]] for i in edges]

    edges += edges_rev

    nodes_dict = {
        i:{
            'pos':nodes[i],
            'next':[j[1] for j in edges if j[0] == i]}
        for i in nodes
    }

    G = nx.Graph()
    for i in nodes_dict:
        G.add_node(i)
    for e in edges:
        G.add_edge(e[0], e[1])
    nx.set_node_attributes(G, nodes_dict)

    for node,coord in nodes.items():
        print(f'\"{node}\":{{\"x\":{coord[0]},\"y\":{coord[1]},\"next\":{nodes_dict[node]["next"]}}},')

    # for i in edges:
    #     print(f'\"{i[0]},{i[1]}\":[{round(math.dist((nodes[i[0]]),(nodes[i[1]])))},1],')

    fig, ax = plt.subplots()

    if PrintMap:
        ax.plot(np.array(boundary+[boundary[0]])[:,0], np.array(boundary+[boundary[0]])[:,1], 'r-')
        for obs in obstacle_list:
            ax.add_patch(Polygon(obs, closed=True, fill=True, color='gray'))
        ax.set_aspect('equal')
        ax.grid(False, which='both')
        ax.set_axis_on()

    nx.draw(G,
            nx.get_node_attributes(G,'pos'),
            ax=ax,
            # node_size = 10,
            # font_size = 6,
            with_labels=True)

    # Next 2 lines are if you want to see the coordinates of the nodes (testing)
    # for tx, pos in nodes.items():
    #     ax.text(pos[0], pos[1]-2, f'{pos}')

    plt.show()

if __name__ == "__main__":
    build_Environment('SmallEnv',True)
