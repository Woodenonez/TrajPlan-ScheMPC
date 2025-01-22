from Compo_slim import Compo_slim

###################### HERE IS WHERE THE ACTUAL TESTING STARTS ######################

# problem = 'Volvo_gpss_2'
# problem = 'testPlant'
problem = 'PlantNoHubs'

print(f'Problem: {problem}')

instance,optimum,running_time,len_previous_routes,paths_changed, _ = Compo_slim(problem)

################## A BUNCH OF STUFF TO MAKE THINGS INTERACTIVE #####################


# print(f"The current problem to be solved is {problem}")
# while False:
#     answer = input("do you want to change it? \n")
#     if answer == 'no':
#         break
#     elif answer == 'yes':
#         nodes = input('how many nodes? (integer)\n')
#         nodes = int(nodes)
#         vehicles = input('how many vehicles? (integer)\n')
#         vehicles = int(vehicles)
#         tasks = input('how many tasks? (integer)\n')
#         tasks = int(tasks)
#         edge_reduction = input('what is the EDGE REDUCTION? (real between 0 and 1)\n')
#         edge_reduction = float(edge_reduction)
#         time_horizon = input('what is the TIME HORIZON? (real)\n')
#         time_horizon = int(time_horizon)
#         seed = input('what is the SEED? (integer)\n')
#         seed = int(seed)
#         print('I AM MAKING THE INSTANCE')
#         make_an_instance(nodes, vehicles, tasks, edge_reduction, time_horizon, seed)
#         problem = f'MM_{str(nodes) + str(vehicles) + str(tasks)}_{edge_reduction}_{time_horizon}_{seed}'
#         break
#     else:
#         print('say either yes or no')
# while False:
#     # this is the part where I update the Rviz related files
#     answer2 = input("Do you want to updated the Rviz configuration files? \n")
#     if answer2 == 'no':
#         break
#     elif answer2 == 'yes':
#         set_up_Rviz('test_cases/{}.json'.format(problem))
#         ####### I am actually not able to launch Rviz....solve this!!!! ###########
#         # print("Files updated, launching Rviz")
#         # os.system('cd /home/sabino/Documents/gpss')
#         # os.system('source /opt/ros/humble/setup.bash')
#         # os.system('ros2 launch gpss_visualization_bringup gpss.launch.py')
#         break
#     else:
#         print('say either yes or no')


