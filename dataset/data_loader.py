import pandas as pd
from math import cos, sin, radians, atan2, sqrt
import numpy as np


DATA_PATH = 'dataset/RL_data_example_030921.csv'


def get_distance_lat_long(lat1, long1, lat2, long2):
    """Returns distance in meters between two (latitude, longitude) points (Haversine formula)"""
    R = 6371e3  # radius of the Earth in meters
    dlat = radians(lat2 - lat1)
    dlong = radians(long2 - long1)
    a = sin(dlat / 2) * sin(dlat / 2)
    a += cos(radians(lat1)) * cos(radians(lat2)) * sin(dlong / 2) * sin(dlong / 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def load_data():
    """Returns the time step dt, a list of leader positions and a list of leader velocities"""
    df = pd.read_csv(DATA_PATH)

    times = np.array(df['Time'])
    velocities = np.array(df['Velocity'])
    space_gaps = np.array(df['SpaceGap'])
    relative_velocities = np.array(df['RelativeVelocity'])
    longs = np.array(df['Long'])
    lats = np.array(df['Lat'])

    # make sure all data has the same length
    assert(len(set([len(times), len(velocities), len(space_gaps), len(relative_velocities), len(longs), len(lats)])) == 1)

    time_step = df['Time'][1] - df['Time'][0]
    dist_from_start = 0
    leader_positions = [space_gaps[0]]
    leader_velocities = np.maximum((velocities + relative_velocities) / 3.6, 0)  # convert km/h to m/s

    for i in range(1, len(times)):
        # keep track of driving distance from start
        dist_from_start += get_distance_lat_long(lats[i-1], longs[i-1], lats[i], longs[i])
        leader_positions.append(dist_from_start + space_gaps[i])

        # make sure timesteps are consistant
        assert(abs(times[i] - times[i-1] - time_step) < 1e-5)

    return time_step, leader_positions, leader_velocities


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    df = pd.read_csv(DATA_PATH)

    times = np.array(df['Time'])
    velocities = np.array(df['Velocity'])
    space_gaps = np.array(df['SpaceGap'])
    relative_velocities = np.array(df['RelativeVelocity'])
    longs = np.array(df['Long'])
    lats = np.array(df['Lat'])

    driving_length = 0
    for i in range(lats.shape[0]-1):
        driving_length += get_distance_lat_long(lats[i], longs[i], lats[i+1], longs[i+1])

    birds_fly_length = get_distance_lat_long(lats[0], longs[0], lats[-1], longs[-1])

    print(f'Total driving length: {round(driving_length)}m')
    print(f'Total birds fly length: {round(birds_fly_length)}m')

    total_time = times[-1] - times[0]
    print(f'Total trajectory time: {round(total_time)}s')

    time_step, leader_positions, leader_velocities = load_data()
    positions_from_velocity = [leader_positions[0]]
    for vel in leader_velocities[:-1]:
        positions_from_velocity.append(positions_from_velocity[-1] + vel * 0.1)

    # plot ego and leader velocities
    plt.figure()
    plt.plot(times - times[0], velocities / 3.6, label='Ego')
    plt.plot(times - times[0], (velocities + relative_velocities) / 3.6, label='Leader')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.title('Velocity of ego and leading vehicles along trajectory')
    plt.savefig('figs/trajectory_data/velocities.png')

    # plot headways
    plt.figure()
    plt.plot(times - times[0], space_gaps)
    plt.xlabel('Time (s)')
    plt.ylabel('Headway (m)')
    plt.title('Headways of ego vehicle along trajectory')
    plt.savefig('figs/trajectory_data/headways.png')

    # plot time headways
    plt.figure()
    plt.plot(times - times[0], space_gaps / (velocities + 1e-9))
    plt.xlabel('Time (s)')
    plt.ylabel('Time headway (s)')
    plt.ylim(ymin=0, ymax=10)
    plt.title('Time headways (headway / velocity) of ego vehicle along trajectory')
    plt.savefig('figs/trajectory_data/time_headways.png')

    # plot ego and leader positions
    plt.figure()
    plt.plot(times - times[0], leader_positions - space_gaps, label='Ego')
    plt.plot(times - times[0], leader_positions, label='Leader')
    plt.plot(times - times[0], positions_from_velocity, label='Leader from velocities')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.title('Position of ego and leader vehicles along trajectory')
    plt.savefig('figs/trajectory_data/positions.png')
