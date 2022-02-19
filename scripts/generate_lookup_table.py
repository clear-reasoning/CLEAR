"""
Instructions:

Setup the 9 parameters below (bounds and discretization steps)
Default parameters are generally fine but finer discretization
might be needed if your controller's outputted acceleration
varies strongly wrt. some of the input variables.

Modify the get_accel function so that it returns the acceleration output
of your controller as a function of the AV speed, leader speed and space gap
(everything is in standard SI units)

Run python scripts/generate_lookup_table.py, this generate controller_data.csv

Run Benni's transform_controller_into_3d_array.m (you might need to uncomment
or modify the first line so that it links to the .csv file) to convert the
.csv file into a controller_data.mat file

Run Benni's controller_properties.m so that it generates plots from the data
contained in controller_data.mat
"""

import numpy as np
import csv
from pathlib import Path
import json
import re
import sys
import importlib
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""  # disable GPU

# bounds and discretization parameters
min_ego_speed = 0
max_ego_speed = 30
interval_ego_speed = 0.5

min_leader_speed = 0
max_leader_speed = 30
interval_leader_speed = 0.5

min_space_gap = 0
max_space_gap = 100
interval_space_gap = 1

controller_file_name = 'controller_data.csv'

# code to load an RL controller
if True:
    # load config file
    cp_path = Path('checkpoints/2021_08_vandertest_controller/vandertest_controller/checkpoints/800.zip')
    with open(cp_path.parent.parent / 'configs.json', 'r') as fp:
        configs = json.load(fp)
    env_config = configs['env_config']

    # retrieve algorithm
    alg_module, alg_class = re.match("<class '(.+)\\.([a-zA-Z\\_]+)'>", configs['algorithm']).group(1, 2)
    assert (alg_module.split('.')[0] in ['stable_baselines3', 'algos'] or alg_module.split('.')[1] == 'algos')
    sys.path.append(os.path.join(sys.path[0], '..'))
    sys.path.append(os.path.join(sys.path[0], '..', 'trajectory'))
    algorithm = getattr(importlib.import_module(alg_module), alg_class)

    # load checkpoint into model
    model = algorithm.load(cp_path)

    # calibrate number of vf states
    n_vf_states = 0
    state = [0, 0, 0]
    while True:
        try:
            model.predict(state)
        except ValueError:
            n_vf_states += 1
            state.append(0)
        else:
            break


def get_accel(ego_speed, leader_speed, space_gap):
    # return the acceleration output of your controller
    # the following code is for the RL controller
    state = np.array([ego_speed / 40.0, leader_speed / 40.0, space_gap / 100.0] + ([0] * n_vf_states))
    return model.predict(state, deterministic=True)[0][0]


with open(controller_file_name, 'w', newline='') as csvfile:
    fieldnames = ['ego_speed', 'leader_speed', 'space_gap', 'accel']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for ego_speed in np.arange(min_ego_speed, max_ego_speed + interval_ego_speed, interval_ego_speed):
        for leader_speed in np.arange(min_leader_speed, max_leader_speed + interval_leader_speed, interval_leader_speed):
            for space_gap in np.arange(min_space_gap, max_space_gap + interval_space_gap, interval_space_gap):
                accel = get_accel(ego_speed, leader_speed, space_gap)
                writer.writerow({
                    'ego_speed': ego_speed,
                    'leader_speed': leader_speed,
                    'space_gap': space_gap,
                    'accel': accel,
                })

print(f'Controller data written at ./{controller_file_name}')
