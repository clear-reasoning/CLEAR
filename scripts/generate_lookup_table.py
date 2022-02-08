"""
Instructions:

Setup the 9 parameters below (bounds and discretization steps)
Default parameters are generally fine but finer discretization 
might be needed if your controller's outputted acceleration
varies strongly wrt. some of the input variables.

Run python scripts/generate_lookup_table.py, tis generate controller_data.csv

Benni's transform_controller_into_3d_array.m allows to transform
this file into controller_data.mat, which can be fed into the
MATLAB scripts.-
"""

import numpy as np
import csv

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

def get_accel(ego_speed, leader_speed, space_gap):
    pass   # return the acceleration of your controller

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