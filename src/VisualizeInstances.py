from pkg_sche.sp_comsat.support_functions import print_graph,json_parser
import math
import networkx as nx

def visualize_instance(problem,labels):
# first of all, let's parse the json file with the plant layout and the tasks info
    jobs, nodes, edges, Autonomy, ATRs, charging_coefficient, Big_Number, hubs \
        = json_parser(f'data/test_cases/{problem}.json')
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
    print_graph(nodes,edges,labels)

if __name__ == "__main__":

    visualize_instance('movingai_empty16_1_8',False)