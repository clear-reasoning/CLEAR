import os
import pandas as pd
import matplotlib.pyplot as plt

# Set the root folder
root_folder = 'data_v2_preprocessed_west'

# Traverse the subfolders
for subdir, dirs, files in os.walk(root_folder):
    for file in files:
        # Check if the file is 'trajectory.csv'
        if file == 'trajectory.csv':
            # Construct the file path
            file_path = os.path.join(subdir, file)

            # Read the CSV file using pandas
            df = pd.read_csv(file_path)

            # Plot speed vs time
            plt.figure()
            plt.plot(df['Time'], df['Velocity'] / 3.6)
            plt.xlabel('Time')
            plt.ylabel('Speed')
            plt.title('Speed vs Time')

            # Save the plot in the same subfolder
            plot_path = os.path.join(subdir, 'speed_vs_time.png')
            plt.savefig(plot_path)
            plt.close()