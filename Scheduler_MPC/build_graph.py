import networkx as nx
import matplotlib.pyplot as plt
import math

# this is the old graph where i assume that some nodes can be hubs
# nodes = {'d1': [12, 83], 'd2': [37, 83], 'N0': [12, 75], 'N1': [18, 75], 'N2': [32, 75], 'N3': [37, 75], 'N4': [18, 67], 'L1': [25, 67], 'N5': [32, 67], 'P1': [3, 58], 'N6': [12, 58], 'N7': [18, 58], 'N8': [32, 58], 'N9': [37, 58], 'P5': [47, 58], 'N10': [18, 49], 'L2': [25, 49], 'N11': [32, 49], 'P2': [3, 40], 'N12': [12, 40], 'N13': [18, 40], 'N14': [32, 40], 'N15': [37, 40], 'P6': [47, 40], 'N16': [18, 31], 'L3': [25, 31], 'N17': [32, 31], 'P3': [3, 22], 'N18': [12, 22], 'N19': [18, 22], 'N20': [32, 22], 'N21': [37, 22], 'P7': [47, 22], 'N22': [18, 13], 'L4': [25, 13], 'N23': [32, 13], 'P4': [3, 4], 'N24': [12, 4], 'N25': [18, 4], 'N26': [32, 4], 'N27': [37, 4], 'P8': [47, 4]}
# edges = [['d12', 'N0'], ['N0', 'N6'], ['N6', 'N12'], ['N12', 'N18'], ['N18', 'N24'], ['N1', 'N4'], ['N4', 'N7'], ['N7', 'N10'], ['N10', 'N13'], ['N13', 'N16'], ['N16', 'N19'], ['N19', 'N22'], ['N22', 'N25'], ['N2', 'N5'], ['N5', 'N8'], ['N8', 'N11'], ['N11', 'N14'], ['N14', 'N17'], ['N17', 'N20'], ['N20', 'N23'], ['N23', 'N26'], ['d2', 'N3'], ['N3', 'N9'], ['N9', 'N15'], ['N15', 'N21'], ['N21', 'N27'], ['N0', 'N1'], ['N2', 'N3'], ['N4', 'L1'], ['L1', 'N5'], ['P1', 'N6'], ['N6', 'N7'], ['N8', 'N9'], ['N9', 'P5'], ['N10', 'L2'], ['L2', 'N11'], ['P2', 'N12'], ['N12', 'N13'], ['N14', 'N15'], ['N15', 'P6'], ['N16', 'L3'], ['L3', 'N17'], ['P3', 'N18'], ['N18', 'N19'], ['N20', 'N21'], ['N21', 'P7'], ['N22', 'L4'], ['L4', 'N23'], ['P4', 'N24'], ['N24', 'N25'], ['N26', 'N27'], ['N27', 'P8']]


nodes = {
        'D01': [4,90],'D02': [7, 90],'D03': [10, 90],'D04': [13, 90],'D05': [16, 90],
        'D11': [4, 83],'D12': [7, 83],'D13': [10, 83],'D14': [13, 83],'D15':[16,83],
        'D21': [33, 90],'D22': [36, 90],'D23': [39, 90],'D24': [42, 90],'D25': [45, 90],
        'D31': [33, 83],'D32': [36, 83],'D33': [39, 83],'D34': [42, 83],'D35':[45,83],
        'N0': [12, 75],'N00': [8, 75], 'N1': [18, 75], 'N2': [32, 75], 'N3': [37, 75], 'N30': [41, 75], 'N4': [18, 67],
        'L1': [23, 67],'L12': [27, 67],'L13': [23, 60],'L14': [27, 60],
        'N5': [32, 67], 'P1': [3, 58],'P12': [0, 58],'P13': [3, 65],'P14': [0, 65],
        'N6': [12, 58], 'N60': [8, 58], 'N7': [18, 58], 'N8': [32, 58], 'N9': [37, 58], 'N90': [41, 58],
        'P5': [47, 58],'P52': [50, 58],'P53': [47, 65],'P54': [50, 65], 'N10': [18, 49], 'L2': [23, 49],'L22': [27, 49],
        'L23': [23, 42],'L24': [27, 42], 'N11': [32, 49], 'P2': [3, 40],'P22': [0, 40],'P23': [3, 47],'P24': [0, 47], 'N12': [12, 40],
        'N120': [8, 40], 'N13': [18, 40], 'N14': [32, 40], 'N15': [37, 40], 'N150': [41, 40], 'P6': [47, 40],'P62': [50, 40],'P63': [47, 47],
        'P64': [50, 47], 'N16': [18, 31], 'L3': [23, 31],'L32': [27, 31],'L33': [23, 24],'L34': [27, 24], 'N17': [32, 31], 'P3': [3, 22],'P32': [0, 22],
        'P33': [3, 29],'P34': [0, 29], 'N18': [12, 22], 'N180': [8, 22], 'N19': [18, 22], 'N20': [32, 22], 'N21': [37, 22], 'N210': [41, 22],
        'P7': [47, 22],'P72': [50, 22],'P73': [47, 29],'P74': [50, 29], 'N22': [18, 13], 'L4': [23, 13],'L42': [27, 13],'L43': [23, 6],'L44': [27, 6],
        'N23': [32, 13], 'P4': [3, 4],'P42': [0, 4],'P43': [3, 11],'P44': [0, 11], 'N24': [12, 4], 'N240': [8, 4], 'N25': [18, 4], 'N26': [32, 4],
        'N27': [37, 4], 'N270': [41, 4], 'P8': [47, 4],'P82': [50, 4],'P83': [47, 11],'P84': [50,11]
    }
edges = [
    ['D01','D02'],['D02','D03'],['D03','D04'],['D04','D05'],
    ['D11','D12'],['D12','D13'],['D13','D14'],['D14','D15'],
    ['D01','D11'],['D02','D12'],['D03','D13'],['D04','D14'],['D05','D15'],
    ['D13', 'N0'],['D13', 'N00'],['N0','N00'],
    ['N0', 'N6'],['N00', 'N60'], ['N6', 'N12'],['N60', 'N120'], ['N12', 'N18'],['N120', 'N180'], ['N18', 'N24'],['N180', 'N240'],
    ['N1', 'N4'], ['N4', 'N7'], ['N7', 'N10'], ['N10', 'N13'], ['N13', 'N16'], ['N16', 'N19'], ['N19', 'N22'], ['N22', 'N25'], ['N2', 'N5'], ['N5', 'N8'], ['N8', 'N11'], ['N11', 'N14'], ['N14', 'N17'], ['N17', 'N20'], ['N20', 'N23'], ['N23', 'N26'],
    ['D21','D22'],['D22','D23'],['D23','D24'],['D24','D25'],
    ['D31','D32'],['D32','D33'],['D33','D34'],['D34','D35'],
    ['D21','D31'],['D22','D32'],['D23','D33'],
    ['D33', 'N3'],['D33', 'N30'],['N3','N30'],
    ['N3', 'N9'],['N30', 'N90'], ['N9', 'N15'],['N90', 'N150'], ['N15', 'N21'],['N150', 'N210'], ['N21', 'N27'],['N210', 'N270'],
    ['N0', 'N1'], ['N2', 'N3'], ['N60', 'N6'], ['N6', 'N7'], ['N8', 'N9'], ['N9', 'N90'], ['N120', 'N12'], ['N12', 'N13'], ['N14', 'N15'], ['N15', 'N150'], ['N180', 'N18'], ['N18', 'N19'], ['N20', 'N21'], ['N21', 'N210'], ['N240', 'N24'], ['N24', 'N25'], ['N26', 'N27'], ['N27', 'N270'],
    ['N4', 'L1'], ['L12', 'N5'],
    ['N10', 'L2'], ['L22', 'N11'],
    ['N16', 'L3'], ['L32', 'N17'],
    ['N22', 'L4'], ['L42', 'N23'],
    ['L1','L12'],['L1','L13'],['L12','L14'],['L13','L14'],
    ['L2','L22'],['L2','L23'],['L22','L24'],['L23','L24'],
    ['L3','L32'],['L3','L33'],['L32','L34'],['L33','L34'],
    ['L4','L42'],['L4','L43'],['L42','L44'],['L43','L44'],
    ['P1','N60'],['P1','P12'],['P12','P14'],['P14','P13'],['P13','P1'],
    ['P2','N120'],['P2','P22'],['P22','P24'],['P24','P23'],['P23','P2'],
    ['P3','N180'],['P3','P32'],['P32','P34'],['P34','P33'],['P33','P3'],
    ['P4','N240'],['P4','P42'],['P42','P44'],['P44','P43'],['P43','P4'],
    ['N90','P5'],['P5','P52'],['P52','P54'],['P54','P53'],['P53','P5'],
    ['N150','P6'],['P6','P62'],['P62','P64'],['P64','P63'],['P63','P6'],
    ['N210','P7'],['P7','P72'],['P72','P74'],['P74','P73'],['P73','P7'],
    ['N270','P8'],['P8','P82'],['P82','P84'],['P84','P83'],['P83','P8'],
    ]
edges_rev = [[i[1],i[0]] for i in edges]

edges += edges_rev

nodes_dict = {
    i:{
        'pos':nodes[i],
        'next':[j[1] for j in edges if j[0] == i]}
    for i in nodes
}

for node,coord in nodes.items():
    print(f'\"{node}\":{{\"x\":{coord[0]},\"y\":{coord[1]},\"next\":{nodes_dict[node]["next"]}}},')

for i in edges:
    print(f'\"{i[0]},{i[1]}\":[{round(math.dist((nodes[i[0]]),(nodes[i[1]])))},1],')

G = nx.Graph()
for i in nodes_dict:
    G.add_node(i)
for i in edges:
    G.add_edge(i[0],i[1])

nx.set_node_attributes(G,nodes_dict)
nx.draw(G,nx.get_node_attributes(G,'pos'), with_labels=True)
plt.show()
