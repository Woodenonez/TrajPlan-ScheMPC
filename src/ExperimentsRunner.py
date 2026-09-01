from main import general_funct
from roadmap_to_testcase import convert_movingai
import csv

maps = ['den312d',
        # 'den520d',
        # 'emtpy-16-16',
        # 'maze-128-128-2',
        # 'maze-32-32-2',
        # 'random-64-64-8',
        # 'room-32-32-4',
        # 'room-64-64-8',
        # 'warehhouse-10-20-10-2-2'
        ]

scenarios = ['1']

n_agents = [
    '4',
    # '5','6','7','8','9','10',
    # '11','12','13','14','15','16','17','18','19','20'
]

seeds = [
    '3',
    # '4','5','6','7',
    # '8','9','10','11','12'
]

# python src/roadmap_to_testcase.py movingai --map den312d --scenario random-1 --n-agents 7 --cell-size 2 --seed 7 --clearance 0.7 --out test_9

def ExpRunner(maps,scenarios,n_agents,seeds):

    for map in maps:
        for scenario in scenarios:
            for n_agent in n_agents:
                for seed in seeds:
                    # create instance
                    convert_movingai(
                        map_name= map,
                        n_agents= n_agent,
                        scenario= f'{map}-random-{scenario}',
                        seed= seed,
                        method= "grid",
                        connectedness= 8,
                        simplify= True,
                        cell_size= 2,
                        out_name= f'{map}_scenario-{scenario}_{n_agent}_{seed}',

                    )

                    general_funct(
                        f'{map}_scenario-{scenario}_{n_agent}_{seed}',
                        scheduler=True,
                        controller=True,
                        naive_tracker=False,  # True = proportional baseline, False = NMPC (see mpc_backend)
                        ignore_speed_ref=False,
                        recording=False,
                        scheduler_backend="sp_comsat",  # "sp_comsat", "occbs", or "aoccbs"
                        assign_via_routing=False,
                        first_solution_only=False,
                        mpc_backend="panoc",
                        headless=True,
                        late_threshold_s=False,
                        stuck_timeout_s=False,
                        collision_check=False,
                        collision_margin=False,
                    )