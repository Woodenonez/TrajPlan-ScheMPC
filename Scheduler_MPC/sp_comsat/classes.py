from itertools import combinations
import networkx as nx

class Instance:
    def __init__(self,name, graph, Autonomy, charging_coefficient,big_number,hubs):
        self.name = name
        self.tasks = []
        self.graph = graph
        self.Autonomy = Autonomy
        self.ATRs = []
        self.charging_coefficient = charging_coefficient
        self.big_number = big_number
        self.hubs = hubs

    def __str__(self):
        return 'instance {}'.format(self.name)
    def add_task(self,task):
        self.tasks.append(task)

    def add_ATR(self,ATR):
        self.ATRs.append(ATR)

    def shortest_paths_list(self):
        # here I compute the shortest paths between any two customers
        shortest_paths = {
                (i[0].id, i[1].id):
                Path(
                (i[0].id, i[1].id),
                nx.shortest_path_length(self.graph, i[0].location, i[1].location, weight='weight'),
                nx.shortest_path(self.graph, i[0].location, i[1].location, weight='weight')
                )
            for i in combinations(self.tasks, 2)
        }
        # also the other way around
        shortest_paths.update({

                (i[1].id, i[0].id):
                Path(
                    (i[1].id, i[0].id),
                    nx.shortest_path_length(self.graph, i[1].location, i[0].location, weight='weight'),
                    nx.shortest_path(self.graph, i[1].location, i[0].location, weight='weight')
                )
            for i in combinations(self.tasks, 2)
        })
        return shortest_paths

class Task:
    def __init__(self,identifier,task_type,location,precedence,TW,Service,ATR):
        self.id = identifier
        self.task_type = task_type
        self.location = location
        self.precedence = precedence
        self.TW = TW
        self.Service = Service
        self.ATR = ATR
    def __str__(self):
        return '{}'.format(self.id)

class ATR:
    def __init__(self,identifier,depot):
        self.id = identifier
        self.depot = depot
    def __str__(self):
        return '{}'.format(self.id)

class Path:
    def __init__(self,path_id,length,path_nodes):
        self.path_id = path_id
        self.length = length
        self.path_nodes = path_nodes
    def __str__(self):
        return "{}".format(self.path_id)

class Route:
    def __init__(self,tasks,vehicle,length,nodes,edges,TW,ST,CT):
        self.tasks = tasks
        self.vehicle = vehicle
        self.length = length
        self.nodes = nodes
        self.edges = edges
        self.TW = TW
        self.ST = ST
        self.CT = CT
    def display(self):
        print('Route of length {} executed by vehicle {}'.format(self.length,
                    self.vehicle.id))
        print('Tasks: ',[i.id for i in self.tasks])
        print('Nodes: ', self.nodes)
        # print('Edges: ',self.edges)
        # print('Time Windows: ',self.TW)
        # print('Service Time: ', self.ST)
        # print('Charging Time: ', self.CT)
        return ''

