from z3 import *


def routes_checking(the_instance,current_paths,current_routes):

    served = [[Real('vehicle_{}_serves_{}'.format(route.vehicle.id,customer.id)) for customer in route.tasks ]
                                                        for route in current_routes ]

    charge = [[Real('vehicle_{}_charge_at_{}'.format(route.vehicle.id,customer.id)) for customer in route.tasks ]
                                                        for route in current_routes ]
    # print('CURRENT ROUTES')
    # for i in current_routes:
    #     print(i)
    # print('CURRENT PATHS')
    # for i in current_paths:
    #     print(i)

    domain = [
        And(
            served[i][j] >= 0,
            charge[i][j] >= 0,
            charge[i][j] <= the_instance.Autonomy
        )
        for i,route in enumerate(current_routes)
        for j,_ in enumerate(route.tasks)
    ]

    infer_arrival_time = [
        served[i][j2+1] >= served[i][j1]
                        + current_paths[route.vehicle][(job1.location,job2.location)].length
                        + job1.Service
        for i, route in enumerate(current_routes)
            for (j1,job1),(j2,job2) in zip(enumerate(route.tasks[:-1]),
                                           enumerate(route.tasks[1:]))
    ]

    time_window = [
        And(
            served[i][j] >= job.TW[0],
            served[i][j] <= job.TW[1]
        )
        for i, route in enumerate(current_routes)
        for j, job in enumerate(route.tasks)
        if job.TW != []

    ]

    autonomy_1 = [
        charge[i][j2 + 1] <= charge[i][j1] - current_paths[route.vehicle][(job1.location,job2.location)].length
        for i, route in enumerate(current_routes)
        for (j1, job1), (j2, job2) in zip(enumerate(route.tasks[:-1]),
                                          enumerate(route.tasks[1:]))
        if job1.task_type != 'recharge'

    ]

    autonomy_2 = [
        charge[i][j2 + 1] <= the_instance.Autonomy - current_paths[route.vehicle][(job1.location,job2.location)].length
        for i, route in enumerate(current_routes)
        for (j1, job1), (j2, job2) in zip(enumerate(route.tasks[:-1]),
                                          enumerate(route.tasks[1:]))
        if job1.task_type == 'recharge'

    ]

    autonomy = autonomy_1 + autonomy_2

    checking = Optimize()

    checking.add(
        domain +
        infer_arrival_time +
        time_window +
        autonomy
    )

    checking_feasibility = checking.check()
    # routes_plus = []
    # locations_plus = {}
    if checking_feasibility == sat:

        # this list tells the distance between each two locations based on the route
        # first step: calculate distance from one location to the following
        routes_length = {
            route: sum([
                current_paths[route.vehicle][(elem1.location,elem2.location)].length
                for elem1, elem2 in zip(route.tasks[:-1], route.tasks[1:])
            ])
            for route in current_routes
        }

        # this manipulation I have done here is very weird, if I ever get the time, I'll scrap
        # it and redo it better
        # 20/10/2021: ....I WISH I HAD DONE IT BETTER :(
        # 10/11/2022: ... NAH...IT AIN'T GOING TO HAPPEN....LIVE WITH THAT!
        # 21/12/2022, 11.41: OK.....TODAY IS THE DAY.....
        # 21/12/2022, 12.30: I CAN'T BELIEVE IT....I FIXED IT

        actual_nodes = {}
        for i,route in enumerate(current_routes):
            points = [route.tasks[0].location]
            tws = [[]]
            St = [0]
            Ct = [0]
            for j,(segment1, segment2) in enumerate(zip(route.tasks[:-1], route.tasks[1:])):
                points += current_paths[route.vehicle][(segment1.location,segment2.location)].path_nodes[1:]
                for _ in current_paths[route.vehicle][(segment1.location,segment2.location)].path_nodes[1:]:
                    tws.append([])
                    St.append(0)
                    Ct.append(0)
                tws[-1] = segment2.TW
                St[-1] = segment2.Service
                if segment2.task_type == 'recharge':
                    Ct[-1] = math.ceil((1 / the_instance.charging_coefficient)
                                       * (the_instance.Autonomy - round(int(str(checking.model()[charge[i][j+1]])))))

            actual_nodes.update({route: (points, tws, St, Ct)})

        actual_nodes_2 = {
            elem: [(current, next) for current, next in zip(
                actual_nodes[elem][0],
                actual_nodes[elem][0][1:])
                   ]
            for elem in actual_nodes
        }

        for route in current_routes:
            route.length = routes_length[route]
            route.nodes = actual_nodes[route][0]
            route.edges = actual_nodes_2[route]
            route.TW = actual_nodes[route][1]
            route.ST = actual_nodes[route][2]
            route.CT = actual_nodes[route][3]

    return checking_feasibility, current_routes

