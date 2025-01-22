from z3 import *
import networkx as nx
# import matplotlib.pyplot as plt
from support_functions import json_parser, paths_formatter
from time import time as tm
from classes import Task, Instance, ATR
import math
###### The sub-problems algorithms #########
from E_Routing_Gurobi import routing
from scheduling_model import schedule
from path_changer_Gurobi import changer
from route_checker_slim import routes_checking
import math
def Compo_slim(problem):

    print('COMPOSITIONAL ALGORITHM #### SLIM ####')
    print('instance',problem)

    starting_time = tm()

    # first of all, let's parse the json file with the plant layout and the tasks info
    jobs, nodes, edges, Autonomy, ATRs, charging_coefficient,Big_Number,hubs\
        = json_parser('test_cases/%s.json' % problem)
    # now let's build the graph out of nodes and edges
    graph = nx.DiGraph()

    for i in nodes:
        graph.add_node(i)
        for j in nodes[i]['next']:
            distance = math.sqrt((nodes[i]['x']-nodes[j]['x'])**2 + (nodes[i]['y']-nodes[j]['y'])**2)
            graph.add_edge(i,str(j), weight= distance, capacity=2)
    # for i in nodes:
    #     print(i)
    ########### In case I want to plot the graph ############
    # nx.draw(graph,with_labels = True)
    # plt.show()
    # print("The Graph is strongly connected...",nx.is_strongly_connected(graph))

    The_Instance = Instance(
        problem,
        graph,
        Autonomy,
        charging_coefficient,
        Big_Number,
        hubs
    )

    # this value repersents the number of dummy copies that i am making for each recharging station
    dummies = 2

    dum_buffer = {}
    for i in jobs:
        if i.split('_')[0] == 'start':
            for j in range(dummies):
                dum_buffer.update({
                    'recharge_{}_{}'.format(i.split('_')[1],j): {
                                'location': jobs[i]['location'],
                                'precedence': [],
                                'TW': [],
                                'Service': 0,
                                'ATR': jobs[i]['ATR']
                                }
                            })
    jobs.update(dum_buffer)

    for i in jobs:
        The_Instance.add_task(
            Task(
                i,
                i.split('_')[0],
                jobs[i]['location'],
                jobs[i]['precedence'],
                jobs[i]['TW'],
                jobs[i]['Service'],
                jobs[i]['ATR'],
            )
        )
    for i,j in ATRs.items():
            The_Instance.add_ATR(
                ATR(
                    i,
                    j
                )
            )

    ############# LET'S INITIALIZE A BUNCH OF STUFF #############
    node_sequence = []

    # decision upon the whole problem
    instance = unknown

    # initialize status of the routing problem
    routing_feasibility = unknown

    # initialize the list of used sets of routes
    previous_routes = []

    # lets's set a limit on the number of routes
    routes_bound = 15

    # I wanna know if paths are changed
    paths_changed = "NO"
    # this parameter is only for testing and it makes the "schedule check" UNSAT for a number of iterations to trigger
    routes_to_check = 2

    while routing_feasibility != unsat and instance == unknown and len(previous_routes) < routes_bound:

        #$$$$$$$$$$$ PRINTING $$$$$$$$$$$$$#
        print('length of previous routes', len(previous_routes))

        # let's solve the routing problem
        routing_start = tm()
        routing_feasibility, current_routes, routes_solution = routing(
                                                    The_Instance, previous_routes
                                                    )
        routing_end = tm()

        # print('CURRENT ROUTES - AFTER E-ROUTER')
        # for i in current_routes:
        #     print(i.display())
        # print('######################')

        previous_routes.append(routes_solution)

        #$$$$$$$$$$$ PRINTING $$$$$$$$$$$$$#
        print('routing', routing_feasibility, round(routing_end - routing_start,2))

        if routing_feasibility == unsat:
            break

        schedule_start = tm()
        schedule_feasibility, node_sequence, edge_sequence = schedule(The_Instance,current_routes)
        schedule_end = tm()

        ########### TEST #############
        # if len(previous_routes) < routes_to_check:
        #     schedule_feasibility = unsat
        # schedule_feasibility = unsat

        #$$$$$$$$$$$ PRINTING $$$$$$$$$$$$$#
        print('     schedule', schedule_feasibility, round(schedule_end - schedule_start,2))

        if schedule_feasibility == sat:
            instance = schedule_feasibility
            break

        # let's set the shortest paths as "current paths"
        ###### THIS SHOULD BE MADE CONSISTENT WITH THE OTHER PATH LISTS I.E. USING THE PATH CLASS ######
        current_paths = The_Instance.shortest_paths_list()
        # let's format the current paths so that I can use them as a previous solutions for the
        # path changing problem
        shortest_paths_solution, current_paths = paths_formatter(current_paths, current_routes)

        # initialize status of used paths for the current set of routes
        previous_paths = [shortest_paths_solution]

        # initialize status of the assignment problem
        paths_changing_feasibility = unknown

        # let's set a limit on the number of paths to try otherwise we'll get stuck in this loop
        bound = 0
        counter = 0
        while paths_changing_feasibility != unsat and instance == unknown and counter < bound:

            #$$$$$$$$$$$ PRINTING $$$$$$$$$$$$$#
            print('iteration', counter)

            path_change_start = tm()
            paths_changing_feasibility, paths_changing_solution, new_paths = changer(
                The_Instance.graph, current_paths, previous_paths
            )
            path_change_end = tm()

            ####### this is just to check whether paths are actually changed or not ############
            paths_changed = 'YES'
            ################## TEST ##################
            # paths_changing_feasibility = unsat

            # $$$$$$$$$$$ PRINTING $$$$$$$$$$$$$#
            print('         paths_changer', paths_changing_feasibility, round(path_change_end - path_change_start, 2))

            if paths_changing_feasibility == sat:

                previous_paths.append(paths_changing_solution)
                ### I don't really need this, but it is easier to understand what is going on
                current_paths = new_paths
                # for atr in current_paths:
                #     for i in current_paths[atr].values():
                #         print(i.path_id)
                #         print(i.length)
                #         print(i.path_nodes)

                # ROUTE VERIFICATION PROBLEM
                routes_check_start = tm()
                routes_checking_feasibility, current_routes = routes_checking(
                    The_Instance,current_paths, current_routes,
                )
                routes_check_end = tm()

                ########### TEST #############
                # routes_checking_feasibility = unsat

                #$$$$$$$$$$$ PRINTING $$$$$$$$$$$$$#
                print('         routes check', routes_checking_feasibility, round(routes_check_end
                                                                                  -
                                                                                  routes_check_start,2))

                if routes_checking_feasibility == sat:
                    pass

                    # print('CURRENT ROUTES - AFTER ROUTES-CHECKER')
                    # for i in current_routes:
                    #     print(i.display())

                    sched_2_start = tm()
                    schedule_feasibility, node_sequence, edge_sequence = schedule(The_Instance,current_routes)
                    sched_2_end = tm()

                    #### TEST ###########
                    # if len(previous_routes) < routes_to_check:
                    #     schedule_feasibility = unsat
                    # schedule_feasibility = unsat

                    #$$$$$$$$$$$ PRINTING $$$$$$$$$$$$$#
                    print('                 schedule check',schedule_feasibility, round(sched_2_end
                                                                                        -
                                                                                        sched_2_start,2))

                    if schedule_feasibility == sat:
                        instance = schedule_feasibility

            # if bound < 666:
            counter += 1
    #### the following parts must be commented ON if you have set up a limit on
    # the number of iterations of the routing problem and OFF if you relax that
    if routing_feasibility == unsat and len(previous_routes) < routes_bound:
        instance = routing_feasibility
    elif routing_feasibility == unsat and len(previous_routes) == routes_bound:
        instance = unknown

    running_time = round(tm()-starting_time,2)

    optimum = 'None'
    # just some output to check while running the instances in batches

    print('         feasibility', instance)
    print('         running time', running_time)
    if instance == sat:
        optimum = sum([i.length for i in current_routes])
        print('         travelling distance: ',optimum)

        print('CURRENT ROUTES - WHEN THE INSTANCE IS SOLVED')
        for i in current_routes:
            print(i.display())
        print('##########################################')
        for i in node_sequence:
            print('{} visits node {} at time {}'.format(i[0],node_sequence[i][0],node_sequence[i][1]))
            # print(i,node_sequence[i])
        # for i in edge_sequence:
        #     print(i)

    return instance,optimum,running_time,len(previous_routes),paths_changed, [node_sequence,current_routes]


