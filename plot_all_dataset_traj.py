
from trajectory.data_loader import DataLoader
import matplotlib.pyplot as plt

trajs = list(DataLoader().get_all_trajectories())

fig, axs = plt.subplots(len(trajs), figsize=(20, 5*len(trajs)))
fig.suptitle('I24 dataset velocity trajectories')

for i, traj in enumerate(trajs):
    title = traj['path']
    title = f'Traj #{i} - ' + str(title).split('/')[-1]

    axs[i].plot(traj['times'], traj['velocities'])
    axs[i].set_xlim(traj['times'].min(), traj['times'].max())
    axs[i].set_ylim(0, 40)
    axs[i].set_title(title)
    axs[i].set_xlabel('Time (s)')
    axs[i].set_ylabel('AV velocity (m/s)')

plt.savefig('dataset_velocities.png')