from z3 import *
from .classes import Route

def schedule(the_instance, current_routes):
    # i can now start building the model in z3. i am going to treat this part as a standard job shop problem
    # where each node/edge is a resource, each route a job and the nodes to visit are operations.
    # some operations i.e. the deliveries have time windows

    idle_atrs = [i for i in the_instance.ATRs if i not in [j.vehicle for j in current_routes]]
    # print([j.vehicle.id for j in current_routes])
    # print(idle_atrs)
    routes_plus_idle = current_routes.copy()
    for i in idle_atrs:
        routes_plus_idle.append(
            Route(
                [],
                i,
                0,
                [i.depot],
                [],
                [],
                [],
                []
            )
        )

    ### COMMENT THIS ON (AND THE LINE IN CONSTRAINT one_node_at_a_time TO ALLOW HUB NODES #####
    # hubs = [i.depot for i in idle_atrs]

    SafetyCoefficient = 20

    # a variable for each node (operation) of each route (job)
    visit_node = [[Real('vehicle_%s_VISITS_node_%s' % (i.vehicle, j_index))
                   for j_index,j in enumerate(i.nodes)]
                  for i in routes_plus_idle]

    leave_node = [[Real('vehicle_%s_LEAVES_node_%s' % (i.vehicle, j_index))
                   for j_index,j in enumerate(i.nodes)]
                  for i in routes_plus_idle]

    # i am going to declare a different set of variables for the edges
    visit_edge = [[Real('vehicle_%s_VISITS_edges_%s' % (i.vehicle, j_index))
                   for j_index,j in enumerate(i.edges)]
                  for i in routes_plus_idle]

    # all variable should be positive integer (really??? -.-)
    domain_scheduling_1 = [
        And(
            visit_node[i_index][j] >= 0,
            visit_node[i_index][j] <= the_instance.big_number
        )
        for i_index, i in enumerate(routes_plus_idle) for j,_ in enumerate(i.nodes)
    ]

    domain_scheduling_1_dot_2 = [
        And(
            leave_node[i_index][j] >= 0,
            leave_node[i_index][j] <= the_instance.big_number
        )
        for i_index, i in enumerate(routes_plus_idle) for j,_ in enumerate(i.nodes)
    ]

    domain_scheduling_2 = [
        And(
            visit_edge[i_index][j] >= 0,
            visit_edge[i_index][j] <= the_instance.big_number
        )
        for i_index, i in enumerate(routes_plus_idle) for j,_ in enumerate(i.edges)
    ]
    domain_scheduling = domain_scheduling_1 + domain_scheduling_1_dot_2 + domain_scheduling_2

    # the idle vehicle never leaves its initial location
    idle_vehicle = [
        And(
            visit_node[i_index][j] == 0,
            leave_node[i_index][j] == the_instance.big_number
        )
        for i_index, i in enumerate(routes_plus_idle) for j, _ in enumerate(i.nodes) if len(i.nodes) == 1
    ]

    # account for charging time
    charge_time = [
        visit_edge[i_index][j] >= visit_node[i_index][j] + i.CT[j]
        for i_index, i in enumerate(routes_plus_idle)
        for j,_ in enumerate(i.edges)
        if i.CT[j] > 0
    ]

    # establish precedence constraints among operations of the jobs
    visit_precedence = [
        And(
            visit_edge[i_index][j_index] >= visit_node[i_index][j_index] + i.ST[j_index],
            visit_node[i_index][j_index + 1] == visit_edge[i_index][j_index]
                                                + the_instance.graph.get_edge_data(*j)['weight']
        )
        for i_index, i in enumerate(routes_plus_idle)
        for j_index, j in enumerate(i.edges)
    ]

    # some operations have time windows
    visit_tw = [
        And(
            visit_node[i_index][j_index] >= j[0],
            visit_node[i_index][j_index] <= j[1]
        )
        for i_index, i in enumerate(routes_plus_idle)
        for j_index, j in enumerate(i.TW)
        if j != []
    ]

    # A vehicle leaves a node when it visits the following edge
    leaving_a_node = [
        leave_node[i_index][j] == visit_edge[i_index][j]
        for i_index, i in enumerate(routes_plus_idle)
        for j, actual_edge in enumerate(i.edges)
    ]

    # A vehicle does not leave the last node of a route until the end of the schedule
    staying_at_a_node = [
        leave_node[i_index][len(i.nodes)-1] == the_instance.big_number
        for i_index, i in enumerate(routes_plus_idle)
    ]

    # two operations cannot use the same node at the same time (unless the node is a hub)
    one_node_at_a_time = [
        Or(
            visit_node[i1][j1] >= leave_node[i2][j2] + 1 * SafetyCoefficient,
            visit_node[i2][j2] >= leave_node[i1][j1] + 1 * SafetyCoefficient
        )
        for i1, route1 in enumerate(routes_plus_idle)
        for j1, node1 in enumerate(route1.nodes)
        for i2, route2 in enumerate(routes_plus_idle)
        for j2, node2 in enumerate(route2.nodes)
        if route1 != route2
            and node1 == node2
            and node1 not in the_instance.hubs
    ]


    # if two operations are going to use the same edge (from the same side), they cannot start at the same time
    edges_direct = [
        Or(
            visit_edge[i1][j1] >= visit_edge[i2][j2] + 1 * SafetyCoefficient,
            visit_edge[i2][j2] >= visit_edge[i1][j1] + 1 * SafetyCoefficient,
        )

        for i1, route1 in enumerate(routes_plus_idle)
        for j1, edge1 in enumerate(route1.edges)
        for i2, route2 in enumerate(routes_plus_idle)
        for j2, edge2 in enumerate(route2.edges)
        if route1 != route2
        and edge1 == edge2
        # and the_instance.graph.get_edge_data(*edge1)['capacity'] == 1

    ]

    edges_inverse = [
        Or(
            visit_edge[i1][j1] >= visit_edge[i2][j2] + the_instance.graph.get_edge_data(*edge2)['weight'] + SafetyCoefficient,
            visit_edge[i2][j2] >= visit_edge[i1][j1] + the_instance.graph.get_edge_data(*edge1)['weight'] + SafetyCoefficient,
        )
        for i1, route1 in enumerate(routes_plus_idle)
        for j1, edge1 in enumerate(route1.edges)
        for i2, route2 in enumerate(routes_plus_idle)
        for j2, edge2 in enumerate(route2.edges)
        if route1 != route2
        and edge1[0] == edge2[1] and edge1[1] == edge2[0]
        # and the_instance.graph.get_edge_data(*edge1)['capacity'] == 1
    ]

    delayed_start = [
        And([
            Abs(visit_node[i][2] - visit_node[j][2]) > 20
            for j, route_j in enumerate(routes_plus_idle)
            if j != i and route_j.length > 1
        ])
        for i, route_i in enumerate(routes_plus_idle)
        if route_i.length > 0
    ]

    # HERE I BUILD UP THE MODEL FOR THE SCHEDULING PROBLEM
    set_option(rational_to_decimal=True)
    set_option(precision=2)
    if False:
        scheduling = Optimize()

        scheduling.minimize(
            # penalize vehicles from getting to the goal later/earlier than in the middle
            # of the time window
            Sum([
                Abs(visit_node[i_index][j_index] - ((j[1] - j[0]) / 2))
                for i_index, i in enumerate(routes_plus_idle)
                for j_index, j in enumerate(i.TW)
                if j != []
            ])
            +
            # discourage vehicels from waiting at nodes
            Sum([
                (leave_node[i_index][j] - visit_node[i_index][j])
                for i_index, i in enumerate(routes_plus_idle)
                for j, _ in enumerate(i.nodes)
                if j != 0 and j != len(i.nodes)
            ])
            # -
            # encourage delay between vehicles start
            # 1000*Sum([delayed_start[i] for i in range(len(routes_plus_idle))])
            +
            # encourage vehicle to be as fast as possible
            Sum([
                visit_node[i_index][j]
                for i_index, i in enumerate(routes_plus_idle)
                for j, _ in enumerate(i.nodes)
            ])
        )
    else:
        scheduling = Solver()

    # ASSERT THE CONSTRAINTS...
    scheduling.add(
        domain_scheduling +
        idle_vehicle +
        charge_time +
        visit_precedence +
        visit_tw +
        leaving_a_node +
        staying_at_a_node +
        one_node_at_a_time +
        edges_direct +
        edges_inverse +
        delayed_start
    )

    nodes_schedule = {}
    edges_schedule = []
    scheduling_feasibility = scheduling.check()
    if scheduling_feasibility == sat:
        m3 = scheduling.model()
        for i_index, i in enumerate(routes_plus_idle):
            # print(f'delay {i_index}:',m3[delayed_start[i_index]])
            for j_index, j in enumerate(i.nodes):
                # print(j_index)
                nodes_schedule.update(
                {
                    (i.vehicle.id ,j_index):(j, str(m3[visit_node[i_index][j_index]]).replace('?',''))
                }
                )
                # print(
                #     'vehicle_%s_visits_node_%s: ' % (i.vehicle.id, j),
                #     str(m3[visit_node[i_index][j_index]]).replace('?','')
                # )
        for i_index, i in enumerate(routes_plus_idle):
            for j_index, j in enumerate(i.edges):
                edges_schedule.append(
                    (
                        'vehicle_%s_visits_edge_%s: ' % (i, j),
                        m3[visit_edge[i_index][j_index]]
                    )
                )

    return scheduling_feasibility,nodes_schedule,edges_schedule
