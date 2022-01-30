from pathlib import Path
import numpy as np
import pandas as pd
import os
import random
import onnx  # pip install onnx
import onnxruntime as ort  # pip install onnxruntime
import matplotlib.pyplot as plt
from trajectory.utils import counter

DATA_PATH = '../dataset/data_v2_preprocessed_west'
CONTROLLER_PATH = "./vandertest_controller.onnx"


class DataLoader(object):
    def __init__(self):
        self.trajectories = []
        for fp, data in self.get_raw_data():
            positions = [0]
            for speed in np.array(data['Velocity'])[:-1] / 3.6:
                positions.append(positions[-1] + 0.1 * speed)
            self.trajectories.append({
                'path': fp,
                'timestep': round(data['Time'][1] - data['Time'][0], 3),
                'duration': round(data['Time'].max() - data['Time'].min(), 3),
                'size': len(data['Time']),
                'times': np.array(data['Time']) - data['Time'][0],
                'positions': positions,  #  np.array(data['DistanceGPS']),
                'velocities': np.array(data['Velocity']) / 3.6,
                'accelerations': np.array(data['Acceleration'])
            })

    def get_raw_data(self):
        file_paths = list(Path(os.path.join(DATA_PATH)).glob('**/*.csv'))
        data = map(pd.read_csv, file_paths)
        return zip(file_paths, data)

    def get_all_trajectories(self):
        random.shuffle(self.trajectories)
        return iter(self.trajectories)

    def get_trajectories(self, chunk_size=None, count=None):
        for _ in counter(count):
            traj = random.sample(self.trajectories, k=1)[0]
            if chunk_size is None:
                yield dict(traj)
            start_idx = random.randint(0, traj['size'] - chunk_size)
            traj_chunk = {
                k: traj[k][start_idx:start_idx + chunk_size]
                for k in ['times', 'positions', 'velocities', 'accelerations']
            }
            traj_chunk.update({
                'path': traj['path'],
                'timestep': traj['timestep'],
                'duration': round(np.max(traj_chunk['times']) - np.min(traj_chunk['times']), 3),
                'size': len(traj_chunk['times']),
            })
            yield traj_chunk


# load i24 trajectory data
data = DataLoader()
trajectories = data.get_all_trajectories()
traj = next(trajectories)

print(traj.keys())

# load RL controller
model = onnx.load_model(CONTROLLER_PATH)
ort_session = ort.InferenceSession(CONTROLLER_PATH)


def get_accel(state):
    # state is [av speed, leader speed, headway] (no normalization needed)
    # output is instant acceleration to apply to the AV
    data = np.array([state]).astype(np.float32)
    outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: data})
    return outputs[0][0][0]


# initialize AV
av_positions = [traj['positions'][0] - 50.0]  # set initial headway
av_speeds = [traj['velocities'][0]]  # set initial speed

# run AV behind trajectory
dt = traj['timestep']
print(f'dt={dt}s')
for time, leader_pos, leader_speed in zip(traj['times'], traj['positions'], traj['velocities']):
    # get AV accel
    av_pos = av_positions[-1]
    av_speed = av_speeds[-1]
    av_space_gap = leader_pos - av_pos
    av_accel = get_accel([av_speed, leader_speed, av_space_gap])

    # update AV
    new_av_speed = av_speed + dt * av_accel
    new_av_pos = av_pos + dt * new_av_speed
    av_speeds.append(new_av_speed)
    av_positions.append(new_av_pos)

print(f'Trajectory ending at time t={round(time, 1)}s')

plt.figure()
plt.plot(traj['times'], av_speeds[1:], label='AV')
plt.plot(traj['times'], traj['velocities'], label='leader')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.legend()
plt.show()
