import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors
import numpy as np

csv_path = './test.csv'
df = pd.read_csv(csv_path)
#  time,env_step,trajectory_index,veh_types,veh_positions,veh_speeds

start = 0
end = 10000

timesteps = df['time'][start:end]
positions = np.array(list(map(lambda s: list(map(float, s.split(';'))), df['veh_positions'])))[start:end]
speeds = np.array(list(map(lambda s: list(map(float, s.split(';'))), df['veh_speeds'])))[start:end]
assert (positions.shape == speeds.shape)

segments = np.array([np.column_stack([timesteps, positions[:,i], speeds[:,i]]) for i in range(positions.shape[1])])
segments = np.array(np.array_split(segments, (end - start) // 10, axis=1))
segments = segments.reshape(-1, *segments.shape[2:])
segments, speeds = segments[:,:,:2], segments[:,:,2]
speeds = np.mean(speeds, axis=1)

print(segments.shape, speeds.shape)

fig, ax = plt.subplots()
ax.set_xlim(np.min(timesteps), np.max(timesteps))
ax.set_ylim(np.min(positions[0]), np.max(positions[-1]))
# norm = plt.Normalize(np.min(colors), np.max(colors))
cdict = {
    'red': ((0, 0, 0), (0.2, 1, 1), (0.6, 1, 1), (1, 0, 0)),
    'green': ((0, 0, 0), (0.2, 0, 0), (0.6, 1, 1), (1, 1, 1)),
    'blue': ((0, 0, 0), (0.2, 0, 0), (0.6, 0, 0), (1, 0, 0))
}
cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
line_segments = LineCollection(segments=segments, cmap=cmap, # norm=norm, 
                               linewidth=1.0)
line_segments.set_array(speeds)

ax.add_collection(line_segments)
axcb = fig.colorbar(line_segments)

ax.set_title('Time-space diagram on trajectory env')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position (m)')
axcb.set_label('Velocity (m/s)')

plt.show()