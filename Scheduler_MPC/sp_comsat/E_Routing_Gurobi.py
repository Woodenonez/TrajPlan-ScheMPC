from z3 import *
from gurobipy import *
from support_functions import json_parser
import networkx as nx
from itertools import permutations,combinations
from classes import Route
import math
from time import time as tm

def routing(the_instance, previous_routes = []):

    m = Model('router')
    m.setParam('OutputFlag', 0)

    # make lists with vehicles and task names to declare gurobi variables
    vehicles = the_instance.ATRs
    tasks = the_instance.tasks
    #compute the shortest paths between any two tasks
    shortest_paths = the_instance.shortest_paths_list()
    # binary variable that states whether a vehicle travels from customer i to customer j
    direct_travel = m.addVars([i.id for i in vehicles],
                              [i.id for i in tasks],
                              [i.id for i in tasks],vtype=GRB.BINARY,name='direct_travel')

    # continuous variable that keeps track of the arrival of vehicle k at a customer
    customer_served = m.addVars([i.id for i in tasks],vtype=GRB.CONTINUOUS,name='customer_served')
    # continuous variable that keeps track of the autonomy left when arriving at a customer
    autonomy_left = m.addVars([i.id for i in vehicles],
                              [i.id for i in tasks],vtype=GRB.CONTINUOUS,name='autonomy_left')
    # integer variable that keeps track of the needed to charge
    charging_time = m.addVars([i.id for i in vehicles],
                              [i.id for i in tasks], vtype=GRB.INTEGER, name='charging_time')

    # 28 the cost function is number of visits to the recharge station while at the same time minimizing
    m.setObjective(
        quicksum([
            1000*direct_travel[k.id, i.id, j.id]
            for k in vehicles
            for i in tasks
            for j in tasks
            if i.task_type == 'recharge'])
        +
        # lateness in serving the customers
        quicksum([
            customer_served[i.id] for i in tasks
            if i.task_type != 'start'
            and i.task_type != 'end'
            and i.task_type != 'recharge'
        ])
        +
        # the travelling distance
        quicksum([
            direct_travel[k.id, i.id, j.id] * shortest_paths[(i.id,j.id)].length
            for k in vehicles
            for i in tasks
            for j in tasks
            if i.id != j.id
        ])
        # THIS TERM IS ONLY USED FOR GPSS, where i want the vehicles to return as fast as possible to the depots
        # +
        # quicksum([customer_served[i.id] for i in tasks if i.task_type == 'recharge'])
    )

    # 29 no travel from and to the same spot
    not_travel_same_spot = m.addConstrs(
        direct_travel[i.id, j.id, j.id] == 0
        for i in vehicles
        for j in tasks
    )

    # 30 no travel to the starting point
    no_travel_to_start = m.addConstrs(
        direct_travel[k.id, i.id, j.id] == 0
        for k in vehicles
        for i in tasks
        for j in tasks
        if j.task_type == 'start'
    )

    # 31 no travel from the end point
    no_travel_from_end = m.addConstrs(
        direct_travel[k.id, i.id, j.id] == 0
        for k in vehicles
        for i in tasks
        for j in tasks
        if i.task_type == 'end'
    )

    # 32 tasks of mutual exclusive jobs cannot be executed on the same route (cause i need a different robot)
    mutual_exclusive = m.addConstrs(
        quicksum(direct_travel[k.id, i.id, j.id] for i in tasks) == 0
        for k in vehicles
        for j in tasks
        if k.id not in j.ATR
    )


    # 33 the operating range cannot exceed the given value
    domain = m.addConstrs(
        autonomy_left[i.id, j.id] <= the_instance.Autonomy for i in vehicles for j in tasks
    )

    # 34 constraint over arrival time when precedence is required
    # i assume that precedence constraints can only be among tasks of the same job
    precedence = m.addConstrs(
        customer_served[j1.id] <= customer_served[j2.id]
        for j1 in tasks
        for j2 in tasks
        if j1.id in j2.precedence
    )

    # 35.1 set the arrival time within the time window for each customer
    time_window_1 = m.addConstrs(
        customer_served[j.id] >= j.TW[0]
        for j in tasks
        if j.task_type != 'recharge'
        and j.TW != []
    )
    # 35.2 latest arrival time
    time_window_2 = m.addConstrs(
        customer_served[j.id] <= j.TW[1]
        for j in tasks
        if j.task_type != 'recharge'
        if j.TW != []
    )

    # 36 based on the direct travels, define the arrival time to a customer depending on the departure time from the previous
    infer_arrival_time = m.addConstrs(
        (direct_travel[k.id,i.id,j.id] == 1)
        >>
        (customer_served[j.id] >= customer_served[i.id] +
                                shortest_paths[(i.id,j.id)].length +
                                j.Service
         )
        for k in vehicles
        for i in tasks
        for j in tasks
        if i.id != j.id
        and i.task_type != 'recharge'
    )

    # 37 auxiliary constraint to connect charging time and state of charge
    charging_time_constraint = m.addConstrs(
        charging_time[k.id,i.id]
        >=
        (1 / the_instance.charging_coefficient) * (the_instance.Autonomy - autonomy_left[k.id, i.id])
        for k in vehicles
        for i in tasks
        if i.task_type == 'recharge'
    )

    # 38 arrival time for charging stations
    infer_arrival_time_2 = m.addConstrs(
        (direct_travel[k.id, i.id, j.id] == 1)
        >>
        ( customer_served[j.id] >= customer_served[i.id] +
                                shortest_paths[(i.id,j.id)].length +
                                charging_time[k.id,i.id]
          )
        for k in vehicles
        for i in tasks
        for j in tasks
        if i.id != j.id
        and i.task_type == 'recharge'
    )

    # 39 The following two constraints model how energy is consumed by travelling and accumulated again by visiting the
    # charging stations
    autonomy = m.addConstrs(
        (direct_travel[k.id,i.id,j.id] == 1)
        >>
        (autonomy_left[k.id,j.id] <= autonomy_left[k.id,i.id] - shortest_paths[(i.id,j.id)].length)
        for k in vehicles
        for i in tasks
        for j in tasks
        if i.id != j.id
        and i.task_type != 'recharge'
    )
    # 40
    autonomy_2 = m.addConstrs(
        (direct_travel[k.id, i.id, j.id] == 1)
        >>
        (autonomy_left[k.id, j.id] <= the_instance.Autonomy - shortest_paths[(i.id, j.id)].length)
        for k in vehicles
        for i in tasks
        for j in tasks
        if i.id != j.id
        and i.task_type == 'recharge'
    )

    # 41 one arrival at each customer
    one_arrival = m.addConstrs(
        quicksum([direct_travel[i.id, j.id, k.id] for i in vehicles for j in tasks]) == 1
        for k in tasks
        if k.task_type != 'start'
        and k.task_type != 'end'
        and k.task_type != 'recharge'
    )

    # 42 at most one arrival
    one_arrival_2 = m.addConstrs(
        quicksum([direct_travel[i.id, j.id, k.id] for i in vehicles for j in tasks]) <= 1
        for k in tasks
        if k.task_type == 'recharge'
    )

    # 43 guarantee the flow conservation between start and end
    flow = m.addConstrs(
        quicksum([direct_travel[k.id, i.id, j.id] for j in tasks if j.task_type != 'start'])
        ==
        quicksum([direct_travel[k.id, j.id, i.id] for j in tasks if j.task_type != 'end'])
        for k in vehicles
        for i in tasks
        if i.task_type != 'start'
        and i.task_type != 'end'
    )

    # 44 routes must be closed (i.e. every vehicle that goes out has to come back)
    route_continuity = m.addConstrs(
        quicksum([direct_travel[k.id, i.id, j.id] for j in tasks])
        ==
        quicksum([direct_travel[k.id, l.id, m.id] for l in tasks])

        for k in vehicles
        for i in tasks
        for m in tasks
        if i.task_type == 'start'
        and m.task_type == 'end'
        and i.location == m.location
    )

   # 45 if a number of tasks belongs to one job, they have to take place in sequence
    # TODO reformulate this constraint...it is wrong!
    mutual_exclusivity = m.addConstrs(
        quicksum([direct_travel[k.id, i.id, j.id] for k in vehicles ]) == 1
        for i in tasks
        for j in tasks
        if j.precedence != [] and i.id in j.precedence
    )

    # 47 excluding previous routes
    if previous_routes != []:
        blabla = {index:i for index,i in enumerate(previous_routes)}
        excluding_previous_routes = m.addConstrs(
            quicksum([ direct_travel[ k[0].id, k[1].id, k[2].id ] for k in blabla[j] ]) <= len(blabla[j]) - 1
            for j in blabla
        )

    # no travel from start to end (no empty routes)
    # NOT INCLUDED IN THE PAPER BECAUSE REDOUNDANT because of the cost function
    no_from_start_to_end = m.addConstrs(
        direct_travel[k.id, i.id, j.id] == 0
        for k in vehicles
        for i in tasks
        for j in tasks
        if i.task_type == 'start'
        and j.task_type == 'end'
    )

    # vehicles can only start at the depot the are assigned to
    vehicles_to_their_depots = m.addConstrs(
        direct_travel[k.id, i.id, j.id] == 0
        for k in vehicles
        for i in tasks
        for j in tasks
        if k.depot != i.location
        and i.task_type == 'start'
    )

    # there can at most be one route for each vehicle
    one_vehicle_one_route = m.addConstrs(
        quicksum(direct_travel[k.id, i.id, j.id] for j in tasks) <= 1
        for k in vehicles
        for i in tasks
        if k.depot == i.location
        and i.task_type == 'start'
    )

    m.optimize()

    # print(m.getVarByName('direct_travel[0,job_1_1,job_1_1]').X)
    # print('INFEASIBLE',m.status == GRB.INFEASIBLE)
    # print('OPTIMAL',m.status == GRB.OPTIMAL)
    routes_plus = []
    # this list will be use to store the current solution for future runs of the solver
    current_solution = []

    if m.status != GRB.INFEASIBLE:
        # print("TTD: ",m.getObjective().getValue())
        routing_feasibility = sat
        # route stores the info about each route that i generated with the VRPTW extended model
        routes = {}
        # segments stores the info o each direct travel from a task location to another
        segments = {}
        # here i populate the list based on the variable that evaluated to true in the model
        for k in vehicles:
            sub_segment = []
            for i in tasks:
                for j in tasks:
                    # print(k,i,j,m.getVarByName('direct_travel[{},{},{}]'.format(k,i,j)).X)
                    if m.getVarByName('direct_travel[{},{},{}]'.format(k.id,i.id,j.id)).X >= 0.5:
                        # print(m.getVarByName('direct_travel[{},{},{}]'.format(k,i,j)).VarName)
                        sub_segment.append((i,j))
                        current_solution.append([k, i, j])
            segments.update({k:sub_segment})
        ########### IN CASE I WANNA TAKE A LOOK AT THE SOLUTION ####################
        # print('############ CUSTOMER SERVED ################')
        # for i in tasks:
        #     print(i,round(m.getVarByName('customer_served[{}]'.format(i)).X))
        # print('############# AUTONOMY LEFT #################')
        # for k in vehicles:
        #     for i in tasks:
        #         print(k,i,round(m.getVarByName('autonomy_left[{},{}]'.format(k,i)).X))
        ############################################################################

        # here i start forming the routes by adding the direct travels that begin from the start point
        for k in segments:
            for i in segments[k]:
                if i[0].task_type == 'start':
                    routes.update({k:[i]})
        # # here i concatenate the routes by matching the last location of a direct travel with the first of another one
        for j in routes:
            while routes[j][-1][1].task_type != 'end':
                for i in segments[j]:
                    if i[0].id == routes[j][-1][1].id:
                        routes[j].append(i)
        # let's just change the format of storing the routes
        routes_2 = {route:[routes[route][0][0]] for route in routes}
        for route1, route2 in zip(routes, routes_2):
            for segment in routes[route1]:
                routes_2[route2].append(segment[1])
        # assign routes_2 to routes and keep using routes
        routes = routes_2
        # for i in routes_2:
        #     print(i.id)
        #     for j in routes_2[i]:
        #         print('         ', j.id)

        # this list tells the distance between each two locations based on the route
        # first step: calculate distance from one location to the following
        routes_length = {
            route:sum([
                    shortest_paths[(elem1.id, elem2.id)].length
                    for elem1, elem2 in zip(routes[route][:-1], routes[route][1:])
                ])
            for route in routes
        }

        # this list tells me the actual nodes that the vehicles must visit to execute the route and if
        # the node has a time window ([] otherwise)
        actual_nodes = {}
        for route in routes:
            points = [routes[route][0].location]
            tws = [[]]
            St = [0]
            Ct = [0]
            for segment1, segment2 in zip(routes[route][:-1], routes[route][1:]):
                points += shortest_paths[(segment1.id,segment2.id)].path_nodes[1:]
                for _ in shortest_paths[(segment1.id,segment2.id)].path_nodes[1:]:
                    tws.append([])
                    St.append(0)
                    Ct.append(0)
                tws[-1] = segment2.TW
                St[-1] =segment2.Service
                if segment2.task_type == 'recharge':
                    Ct[-1] = math.ceil((1 / the_instance.charging_coefficient) \
                    *\
                    (the_instance.Autonomy
                     - round(m.getVarByName('autonomy_left[{},{}]'.format(route.id, segment2.id)).X)))

            actual_nodes.update({route:(points, tws, St,Ct)})

        actual_nodes_2 = {
            elem:[(current, next) for current, next in zip(
                actual_nodes[elem][0],
                actual_nodes[elem][0][1:])
             ]
            for elem in actual_nodes
        }

        for route in routes:
            routes_plus.append(
                Route(
                    routes[route],
                    route,
                    routes_length[route],
                    actual_nodes[route][0],
                    actual_nodes_2[route],
                    actual_nodes[route][1],
                    actual_nodes[route][2],
                    actual_nodes[route][3]
                )
            )
    else:
        routing_feasibility = unsat

    return routing_feasibility, routes_plus, current_solution



