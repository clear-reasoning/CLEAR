import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors
import numpy as np
import sys



print('Reading emissions data')

emissions_path = sys.argv[1]
df = pd.read_csv(emissions_path)
#  time,step,id,position,speed,accel,headway,leader_speed,speed_difference,leader_id,follower_id,instant_energy_consumption,total_energy_consumption,total_distance_traveled,total_miles,total_gallons,avg_mpg

start = 0
end = 10000

timesteps = sorted(list(set(map(lambda x: round(x, 1), df['time']))))[start:end]

car_types = []
positions = []
speeds = []

for ts in timesteps:
    ts_data = df.loc[df['time'] == ts]
    car_types.append(list(ts_data['id']))
    positions.append(list(ts_data['position']))
    speeds.append(list(ts_data['speed']))

timesteps = np.array(timesteps)
positions = np.array(positions)
speeds = np.array(speeds)

print('Done')



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