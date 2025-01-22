import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

boundary = [0.0, 0.0, 50.0, 80.0] # [x_min, y_min, x_max, y_max]
boundary_coords = [[boundary[0], boundary[1]], [boundary[2], boundary[1]], [boundary[2], boundary[3]], [boundary[0], boundary[3]]]

obstacle_list = [[[0.0, 61.0], [0.0, 80.0], [6.0, 80.0], [6.0, 61.0]],
                 [[50.0, 61.0], [50.0, 80.0], [44.0, 80.0], [44.0, 61.0]],

                 [[22.0, 17.0], [22.0, 27.0], [28.0, 27.0], [28.0, 17.0]],
                 [[22.0, 35.0], [22.0, 45.0], [28.0, 45.0], [28.0, 35.0]],
                 [[22.0, 53.0], [22.0, 63.0], [28.0, 63.0], [28.0, 53.0]],
                 [[22.0, 71.0], [22.0, 80.0], [28.0, 80.0], [28.0, 71.0]],
                 [[22.0, 9.0], [22.0, 0.0], [28.0, 0.0], [28.0, 9.0]],
                 
                 [[19.0, 14.0], [22.0, 14.0], [22.0, 30.0], [19.0, 30.0]],
                 [[19.0, 32.0], [22.0, 32.0], [22.0, 48.0], [19.0, 48.0]],
                 [[19.0, 50.0], [22.0, 50.0], [22.0, 66.0], [19.0, 66.0]],
                 [[19.0, 68.0], [22.0, 68.0], [22.0, 80.0], [19.0, 80.0]],
                 [[19.0, 12.0], [22.0, 12.0], [22.0, 0.0], [19.0, 0.0]],
                 [[28.0, 14.0], [31.0, 14.0], [31.0, 30.0], [28.0, 30.0]],
                 [[28.0, 32.0], [31.0, 32.0], [31.0, 48.0], [28.0, 48.0]],
                 [[28.0, 50.0], [31.0, 50.0], [31.0, 66.0], [28.0, 66.0]],
                 [[28.0, 68.0], [31.0, 68.0], [31.0, 80.0], [28.0, 80.0]],
                 [[28.0, 12.0], [31.0, 12.0], [31.0, 0.0], [28.0, 0.0]],

                 [[17.0, 23.0], [17.0, 39.0], [15.0, 39.0], [15.0, 23.0]],
                 [[17.0, 57.0], [17.0, 41.0], [15.0, 41.0], [15.0, 57.0]],
                 [[17.0, 59.0], [17.0, 74.0], [15.0, 74.0], [15.0, 59.0]],
                 [[17.0, 21.0], [17.0, 5.0], [15.0, 5.0], [15.0, 21.0]],
                 [[33.0, 23.0], [33.0, 39.0], [35.0, 39.0], [35.0, 23.0]],
                 [[33.0, 57.0], [33.0, 41.0], [35.0, 41.0], [35.0, 57.0]],
                 [[33.0, 59.0], [33.0, 74.0], [35.0, 74.0], [35.0, 59.0]],
                 [[33.0, 21.0], [33.0, 5.0], [35.0, 5.0], [35.0, 21.0]],
                 
                 [[0.0, 7.0], [6.0, 7.0], [6.0, 19.0], [0.0, 19.0]],
                 [[0.0, 25.0], [6.0, 25.0], [6.0, 37.0], [0.0, 37.0]],
                 [[0.0, 43.0], [6.0, 43.0], [6.0, 55.0], [0.0, 55.0]],
                 [[50.0, 7.0], [44.0, 7.0], [44.0, 19.0], [50.0, 19.0]],
                 [[50.0, 25.0], [44.0, 25.0], [44.0, 37.0], [50.0, 37.0]],
                 [[50.0, 43.0], [44.0, 43.0], [44.0, 55.0], [50.0, 55.0]],

                 [[6.0, 76.0], [6.0, 59.0], [11.0, 59.0], [11.0, 76.0]],
                 [[6.0, 57.0], [6.0, 41.0], [11.0, 41.0], [11.0, 57.0]],
                 [[6.0, 39.0], [6.0, 23.0], [11.0, 23.0], [11.0, 39.0]],
                 [[6.0, 21.0], [6.0, 5.0], [11.0, 5.0], [11.0, 21.0]],
                 [[44.0, 76.0], [44.0, 59.0], [39.0, 59.0], [39.0, 76.0]],
                 [[44.0, 57.0], [44.0, 41.0], [39.0, 41.0], [39.0, 57.0]],
                 [[44.0, 39.0], [44.0, 23.0], [39.0, 23.0], [39.0, 39.0]],
                 [[44.0, 21.0], [44.0, 5.0], [39.0, 5.0], [39.0, 21.0]],

                 [[6.0, 0.0], [6.0, 3.0], [19.0, 3.0], [19.0, 0.0]],
                 [[44.0, 0.0], [44.0, 3.0], [21.0, 3.0], [21.0, 0.0]],
                 ]

node_dict = {"d_1": [12, 78], "d_2": [37, 78], "node_0": [12, 75], "node_1": [18, 75], "node_2": [32, 75], "node_3": [37, 75], "node_4": [18, 67], "L_1": [25, 67], "node_5": [32, 67], "P_1": [3, 58], "node_6": [12, 58], "node_7": [18, 58], "node_8": [32, 58], "node_9": [37, 58], "P_5": [47, 58], "node_10": [18, 49], "L_2": [25, 49], "node_11": [32, 49], "P_2": [3, 40], "node_12": [12, 40], "node_13": [18, 40], "node_14": [32, 40], "node_15": [37, 40], "P_6": [47, 40], "node_16": [18, 31], "L_3": [25, 31], "node_17": [32, 31], "P_3": [3, 22], "node_18": [12, 22], "node_19": [18, 22], "node_20": [32, 22], "node_21": [37, 22], "P_7": [47, 22], "node_22": [18, 13], "L_4": [25, 13], "node_23": [32, 13], "P_4": [3, 4], "node_24": [12, 4], "node_25": [18, 4], "node_26": [32, 4], "node_27": [37, 4], "P_8": [47, 4]} # TODO: fill in the node dict
edge_list = [["d_1", "node_0"], ["node_0", "node_6"], ["node_6", "node_12"], ["node_12", "node_18"], ["node_18", "node_24"], ["node_1", "node_4"], ["node_4", "node_7"], ["node_7", "node_10"], ["node_10", "node_13"], ["node_13", "node_16"], ["node_16", "node_19"], ["node_19", "node_22"], ["node_22", "node_25"], ["node_2", "node_5"], ["node_5", "node_8"], ["node_8", "node_11"], ["node_11", "node_14"], ["node_14", "node_17"], ["node_17", "node_20"], ["node_20", "node_23"], ["node_23", "node_26"], ["d_2", "node_3"], ["node_3", "node_9"], ["node_9", "node_15"], ["node_15", "node_21"], ["node_21", "node_27"], ["node_0", "node_1"], ["node_2", "node_3"], ["node_4", "L_1"], ["L_1", "node_5"], ["P_1", "node_6"], ["node_6", "node_7"], ["node_8", "node_9"], ["node_9", "P_5"], ["node_10", "L_2"], ["L_2", "node_11"], ["P_2", "node_12"], ["node_12", "node_13"], ["node_14", "node_15"], ["node_15", "P_6"], ["node_16", "L_3"], ["L_3", "node_17"], ["P_3", "node_18"], ["node_18", "node_19"], ["node_20", "node_21"], ["node_21", "P_7"], ["node_22", "L_4"], ["L_4", "node_23"], ["P_4", "node_24"], ["node_24", "node_25"], ["node_26", "node_27"], ["node_27", "P_8"]] # TODO: fill in the edge list

tunnel_list = [[[11.0, 76.0], [11.0, 3.0], [15.0, 3.0], [15.0, 76.0]],
               [[35.0, 76.0], [35.0, 3.0], [39.0, 3.0], [39.0, 76.0]],]

storage_rooms = [[[0.0, 1.0], [0.0, 7.0], [6.0, 7.0], [6.0, 1.0]],
                 [[0.0, 19.0], [0.0, 25.0], [6.0, 25.0], [6.0, 19.0]],
                 [[0.0, 37.0], [0.0, 43.0], [6.0, 43.0], [6.0, 37.0]],
                 [[0.0, 55.0], [0.0, 61.0], [6.0, 61.0], [6.0, 55.0]],
                 [[50.0, 1.0], [50.0, 7.0], [44.0, 7.0], [44.0, 1.0]],
                 [[50.0, 19.0], [50.0, 25.0], [44.0, 25.0], [44.0, 19.0]],
                 [[50.0, 37.0], [50.0, 43.0], [44.0, 43.0], [44.0, 37.0]],
                 [[50.0, 55.0], [50.0, 61.0], [44.0, 61.0], [44.0, 55.0]],]

work_rooms = [[[22.0, 9.0], [28.0, 9.0], [28.0, 17.0], [22.0, 17.0]],
              [[22.0, 27.0], [28.0, 27.0], [28.0, 35.0], [22.0, 35.0]],
              [[22.0, 45.0], [28.0, 45.0], [28.0, 53.0], [22.0, 53.0]],
              [[22.0, 63.0], [28.0, 63.0], [28.0, 71.0], [22.0, 71.0]],]
              
rest_rooms = [[[6.0, 76.0], [6.0, 80.0], [19.0, 80.0], [19.0, 76.0]],
              [[44.0, 76.0], [44.0, 80.0], [31.0, 80.0], [31.0, 76.0]],]


if __name__ == "__main__":
    # Create figure and axes
    def generate_random_point(rectangle):
        """Generate a random point within the rectangle defined by four points.
        
        Args:
            rectangle: a list of four points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        
        Returns: 
            (x, y) representing a random point within the rectangle
        """
        import random
        x_lim = (min(np.array(rectangle)[:, 0]), max(np.array(rectangle)[:, 0]))
        y_lim = (min(np.array(rectangle)[:, 1]), max(np.array(rectangle)[:, 1]))
        x = random.uniform(x_lim[0], x_lim[1])
        y = random.uniform(y_lim[0], y_lim[1])
        return x, y

    n_robots = 5

    test_robots_1 = [patches.Circle(generate_random_point(tunnel_list[0]), 1, facecolor='red') for _ in range(n_robots)]
    test_robots_2 = [patches.Circle(generate_random_point(tunnel_list[1]), 1, facecolor='blue') for _ in range(n_robots)]

    goals_1 = [generate_random_point(tunnel_list[0]) for _ in range(n_robots)]
    goals_2 = [generate_random_point(tunnel_list[1]) for _ in range(n_robots)]

    fig, ax = plt.subplots()
    ax.set_xlim(boundary[0]-1, boundary[2]+1)
    ax.set_ylim(boundary[1]-1, boundary[3]+1)
    ax.set_aspect('equal')

    ax.plot(np.array(boundary_coords+[boundary_coords[0]])[:,0], np.array(boundary_coords+[boundary_coords[0]])[:,1], 'k--')
    for obstacle in obstacle_list:
        obs_patch = patches.Polygon(obstacle, facecolor='k')
        ax.add_patch(obs_patch)

    for tunnel in tunnel_list:
        tunnel_patch = patches.Polygon(tunnel, facecolor='gray', alpha=0.3)
        ax.add_patch(tunnel_patch)

    for storage_room in storage_rooms:
        ax.plot(np.array(storage_room+[storage_room[0]])[:,0], np.array(storage_room+[storage_room[0]])[:,1], 'r--')
    for work_room in work_rooms:
        ax.plot(np.array(work_room+[work_room[0]])[:,0], np.array(work_room+[work_room[0]])[:,1], 'b--')
    for rest_room in rest_rooms:
        ax.plot(np.array(rest_room+[rest_room[0]])[:,0], np.array(rest_room+[rest_room[0]])[:,1], 'g--')

    for test_robot_1, test_robot_2 in zip(test_robots_1, test_robots_2):
        ax.add_patch(test_robot_1)
        ax.add_patch(test_robot_2)

    num_frames = 20
    wpts_1 = [np.linspace(test_robot_1.center, goal_1, num_frames) for test_robot_1, goal_1 in zip(test_robots_1, goals_1)]
    wpts_2 = [np.linspace(test_robot_2.center, goal_2, num_frames) for test_robot_2, goal_2 in zip(test_robots_2, goals_2)]

    for j in range(5):
        for i in range(num_frames):
            for test_robot_1, test_robot_2, wpt_1, wpt_2 in zip(test_robots_1, test_robots_2, wpts_1, wpts_2):
                test_robot_1.center = wpt_1[i]
                test_robot_2.center = wpt_2[i]
            fig.canvas.draw()
            plt.pause(0.1)

    plt.show()

