from z3 import *
from gurobipy import *
from Scheduler_MPC.sp_comsat.classes import Path

def changer(graph, paths_combo, previous_paths=[]):

    m = Model('paths_changer')
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', 30)
    use_node = {}
    for route in paths_combo:
        for pair in paths_combo[route]:
            for i in graph.nodes:
                use_node[route.id,pair,i] = \
                    m.addVar(vtype=GRB.BINARY,name="use_node[{},{},{}]".format(route.id,pair,i))
    m.update()

    use_edge = {}
    for route in paths_combo:
        for pair in paths_combo[route]:
            for i in graph.edges:
                use_edge[route.id,pair,i] = \
                    m.addVar(vtype=GRB.BINARY,name="use_edge[{},{},{}]".format(route.id,pair,i))
    m.update()
    # print('variables')
    start_and_end_are_true = m.addConstrs(
        use_node[route.id,pair,node] == 1
        for route in paths_combo
        for pair in paths_combo[route]
        for node in graph.nodes
        if node in pair
    )
    # print('start and end true')

    only_one_edge_for_start = m.addConstrs(
        quicksum(use_edge[route.id,pair,edge]
                 for edge in graph.edges if edge in graph.out_edges(pair[0])) == 1
        for route in paths_combo
        for pair in paths_combo[route]
    )
    # print('only one edge for start')

    # -- 31.B Exactly zero incoming edges are used for the start node of each route (not every pair, only first)
    zero_incoming_edge_for_start = m.addConstrs(
        quicksum(use_edge[route.id,pair,edge] for edge in graph.edges
                 if edge in graph.in_edges(pair[0]) ) == 0
        for route in paths_combo
        for pair in paths_combo[route]
    )
    # print('zero incoming edges for start')

    only_one_edge_for_end = m.addConstrs(
        quicksum(use_edge[route.id,pair,edge] for edge in graph.edges
                 if edge in graph.in_edges(pair[1])) == 1
        for route in paths_combo
        for pair in paths_combo[route]
    )
    # print('only one edge for end')

    # -- 32.B Exactly zero outgoing edges are used for the end node of each route (not every pair, only first)
    zero_outgoing_edge_for_end = m.addConstrs(
        quicksum(use_edge[route.id,pair,edge] for edge in graph.edges
                 if edge in graph.out_edges(pair[1])) == 0
        for route in paths_combo
        for pair in paths_combo[route]
        # if pair == paths_combo[route][-1]
    )
    # print('zero outgoing edges for endsss')

    exactly_two_edges_1 = m.addConstrs(
        use_node[route.id,pair,node] ==
        quicksum(use_edge[route.id,pair,edge] for edge in graph.in_edges(node))
        for route in paths_combo
        for pair in paths_combo[route]
        for node in graph.nodes
        if node not in pair #!= pair[0] and node != pair[1]
    )
    exactly_two_edges_2 = m.addConstrs(
        use_node[route.id, pair, node] ==
        quicksum(use_edge[route.id, pair, edge] for edge in graph.out_edges(node))
        for route in paths_combo
        for pair in paths_combo[route]
        for node in graph.nodes
        if node not in pair #!= pair[0] and node != pair[1]
    )

    # print('exactly two edges')

    not_both_directions = [
        use_edge[route.id,pair,edge] + use_edge[route.id,pair,(edge[1],edge[0])] <= 1
        for route in paths_combo
        for pair in paths_combo[route]
        for edge in graph.edges
        if (edge[1],edge[0]) in graph.edges
    ]
    # print('not both directions')

    # print('add to model')

    if previous_paths != []:
        for single_solution in previous_paths:
            m.addConstr(
                quicksum(use_edge[route.id,pair,edge]
                       for edge in graph.edges
                            for route in paths_combo
                                for pair in paths_combo[route]
                                    if (route,pair,i) in single_solution
                ) <= len(single_solution) - 1
            )

    # ############# COST FUNCTION TERMS #############
    #
    # cumulative_length = Int('cumulative_length')
    # s.add(
    #       cumulative_length
    #       ==
    #       Sum([
    #         use_edge[route_i][pair_index][edge_index] * graph.get_edge_data(*i)['weight']  #edges[i][0]
    #         for edge_index,i in enumerate(graph.edges)
    #         for route_i,route in enumerate(paths_combo)
    #         for pair_index,_ in enumerate(paths_combo[route])
    #         ])
    #       )
    node_used = m.addVars(graph.nodes,vtype=GRB.BINARY)
    shared_nodes = m.addConstrs(
        node_used[node] * len(graph.nodes)
        >=
        quicksum(use_node[route.id,pair,node] for route in paths_combo for pair in paths_combo[route]) - 1
        for node in graph.nodes
    )
    # print('shared nodes')

    edge_used = m.addVars(graph.edges,vtype=GRB.BINARY)
    shared_edges = m.addConstrs(
        edge_used[edge] * len(graph.edges)
        >=
        quicksum(use_edge[route.id,pair,edge] for route in paths_combo for pair in paths_combo[route]) - 1
        for edge in graph.edges
    )
    # print('shared edges')

    m.setObjective(
        # cumulative_length
        # +
        quicksum(node_used[node] for node in graph.nodes)
        +
        quicksum(edge_used[edge] for edge in graph.edges)
    )

    m.optimize()

    if m.status != GRB.INFEASIBLE:

        PCF = sat

        # print("length: ",s.model()[cumulative_length])
        # print("nodes values: ", s.model()[sum_nodes_used])
        # print("edges values: ", s.model()[sum_edges_used])

        solution = [
                    ( route, pair, edge)
                                for edge in graph.edges
                                    for route in paths_combo
                                        for pair in paths_combo[route]
                                            if m.getVarByName('use_edge[{},{},{}]'.format(route.id,pair,edge)).X >= 0.5
                    ]

        # just an intermediate step to convert the paths_changing solution into a format readable by the route_checker
        buffer = {
            route: {
                pair: [sol[2] for sol in solution if sol[0] == route and sol[1] == pair]
                for pair in paths_combo[route]
            }
            for route in paths_combo
        }

        # keep manipulating the format
        new_paths = {}
        for route in buffer:
            new_paths.update({route: {}})
            for pair in buffer[route]:
                sequence = list(buffer[route][pair])
                path = [pair[0]]
                for _ in range(len(sequence)):
                    for i in sequence:
                        if i[0] == path[-1]:
                            path.append(i[1])
                new_paths[route].update({pair: path})

        # last step to get the new paths
        new_paths = {
            route:{
                pair:Path(
                    pair,
                    sum([ graph.get_edge_data(n1,n2)['weight'] for n1,n2 in zip(new_paths[route][pair][:-1],new_paths[route][pair][1:])]),
                    new_paths[route][pair]
                )
                for pair in new_paths[route]
            }
            for route in new_paths
        }

    else:
        solution = []
        new_paths = paths_combo
        PCF = unsat
    return PCF, solution, new_paths