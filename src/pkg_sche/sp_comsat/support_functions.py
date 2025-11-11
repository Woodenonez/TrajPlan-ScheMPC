import json
import networkx as nx
import matplotlib.pyplot as plt
from itertools import islice
from .classes import Path
import math
import collections
import itertools

# these next three functions are used to get the divisor of a number
def prime_factors(n):
    i = 2
    while i * i <= n:
        if n % i == 0:
            n /= i
            yield i
        else:
            i += 1

    if n > 1:
        yield n

def prod(iterable):
    result = 1
    for i in iterable:
        result *= i
    return result

def get_divisors(n):
    pf = prime_factors(n)

    pf_with_multiplicity = collections.Counter(pf)

    powers = [
        [factor ** i for i in range(count + 1)]
        for factor, count in pf_with_multiplicity.items()
    ]

    for prime_power_combo in itertools.product(*powers):
        yield prod(prime_power_combo)


# just a function to help me print dics and lists i use in the code
def print_support(iterable):
    if type(iterable) == dict:
        for i in iterable:
            print(i,iterable[i])
    elif type(iterable) == list:
        for i in iterable:
            print(i)
    else:
        print('this is not a dict nor a list, it is a ', type(iterable))
    return None

# just a function to help sorting lists
def take_second(elem):
    return elem[1]

# builds a graph structure out of nodes and edges
def make_graph(nodes,edges):
    graph = {
        i: [elem[1] for elem in edges  if i == elem[0]] for i in nodes
    }
    return graph

def print_graph(nodes,edges):
    nodes_dict = {
        i: {
            'pos': [j['x'],j['y']],
            'next': j['next']}
        for i,j in nodes.items()
    }

    G = nx.Graph()

    for i in nodes_dict:
        G.add_node(i)

    nx.set_node_attributes(G, nodes_dict)

    for i in edges:
        G.add_edge(i.split(',')[0],i.split(',')[1])

    nx.draw(G,nx.get_node_attributes(G,'pos'), with_labels=True)
    plt.show()

def json_parser(file_to_parse,monolithic = False):
    with open(file_to_parse,'r') as read_file:
        data = json.load(read_file)
    Big_number = data['test_data']['Big_number']
    hub_nodes = data['test_data']['hub_nodes']

    Autonomy = data['test_data']['Autonomy']
    charging_coefficient = data['test_data']['charging_coefficient']
    nodes = data['test_data']['nodes']
    jobs = data['jobs']
    ATRs = data['ATRs']

    start_list = []
    for i in ATRs.values():
        if i not in start_list:
            start_list.append(i)

    if monolithic == False:
        jobs.update(
            {
                "start_{}".format(j): {
                            "location": j,
                            "precedence": [],
                            "TW": [],
                            "Service": 0,
                            "ATR": [i for i,k in ATRs.items() if k == j]
                }
            for j in start_list
            }
        )
        jobs.update(
            {
                "end_{}".format(j): {
                            "location": j,
                            "precedence": "None",
                            "TW": [0, Big_number],
                            "Service": 0,
                            "ATR": [i for i,k in ATRs.items() if k == j]
                }
            for j in start_list
            }
        )
    edges = {f"{i},{j}":[math.dist((node['x'],node['y']),
                                   (nodes[j]['x'],nodes[j]['y'])),2] for i,node in nodes.items() for j in node['next']}

    if monolithic == False:
        return jobs,nodes,edges,Autonomy,ATRs,charging_coefficient,Big_number,hub_nodes
    else:
        return jobs,nodes,edges,Autonomy,ATRs,charging_coefficient,\
               Big_number,hub_nodes

# I need this function to generate k paths to connect any two points of interest
def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

def paths_formatter(current_paths,current_routes):

    # this dictonary contains the paths used to traverse the routes
    paths_combo = {
        route.vehicle: {
            (first.location, second.location):
            Path(
                (first.location, second.location),
                current_paths[(first.id, second.id)].length,
                current_paths[(first.id, second.id)].path_nodes
            )

            for first, second in zip(route.tasks[:-1], route.tasks[1:])
        }
        for route in current_routes
    }

    # convert the shortest path solution into a fromat that can be inputed into the path_changing function
    shortest_paths_solution = [
        (route, pair, (first, second))
        for route in paths_combo
        for pair_index, pair in enumerate(paths_combo[route])
        for first, second in zip(paths_combo[route][pair].path_nodes[:-1], paths_combo[route][pair].path_nodes[1:])
    ]

    return shortest_paths_solution, paths_combo

