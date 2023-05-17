import os
import pandas as pd
import matplotlib.pyplot as plt
import json

# Set the root folder
root_folder = 'data_v2_preprocessed_west'

# Traverse the subfolders
for subdir, dirs, files in os.walk(root_folder):
    if not os.path.exists(os.path.join(subdir, 'trajectory.csv')):
        continue

    print(subdir)

    # load files
    trajectory_path = os.path.join(subdir, 'trajectory.csv')
    segments_path = os.path.join(subdir, 'segments.json')
    speeds_path = os.path.join(subdir, 'speed.csv')

    # Read the CSV file using pandas
    trajectory_df = pd.read_csv(trajectory_path)
    speeds_df = pd.read_csv(speeds_path)
    with open(segments_path, 'r') as f:
        segments = json.load(f)

    nplots = 1 + len(speeds_df)
    fig, ax = plt.subplots(nrows=nplots, figsize=(10, 3 * nplots))

    for i in range(nplots):
        # Plot speed vs time
        traj_time = trajectory_df['Time'] - trajectory_df['Time'][0]
        traj_pos = trajectory_df['DistanceGPS'] - trajectory_df['DistanceGPS'][0]
        ax[i].plot(traj_pos, trajectory_df['Velocity'] / 3.6)
        ax[i].set_xlim(traj_pos.min(), traj_pos.max())
        ax[i].set_xlabel('Position (m)')
        # plt.plot(traj_time, trajectory_df['Velocity'] / 3.6)
        # plt.xlim(traj_time.min(), traj_time.max())
        # plt.xlabel('Time (s)')

        if i > 0:
            row = speeds_df.loc[i-1]
            inrix_time, *inrix_speeds = list(row)

            segments_plot = []
            for segment in segments[1:-1]:
                segments_plot.append(segment)
                segments_plot.append(segment)
            segments_plot = [segments[0]] + segments_plot + [segments[-1]]
            inrix_speeds_plot = []
            for inrix_speed in inrix_speeds[:-1]:
                inrix_speeds_plot.append(inrix_speed)
                inrix_speeds_plot.append(inrix_speed)
            ax[i].plot(segments_plot, inrix_speeds_plot)

        ax[i].set_ylabel('Speed (m/s)')
        ax[i].set_title(f't = {inrix_time}s' if i > 0 else '')
        ax[i].grid()

    plt.tight_layout()

    # Save the plot in the same subfolder
    plot_path = os.path.join(subdir, 'speed_vs_time.png')
    plt.savefig(plot_path)
    plt.close()
