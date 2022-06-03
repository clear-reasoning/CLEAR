"""Generate time-space diagram."""
import matplotlib as mpl
from matplotlib import colors
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandas as pd
import sys
import json

ROUTE_START = 0
ROUTE_END = 14000


def plot_time_space_diagram(emissions_path, save_path, inrix_path=None):
    """Plot time-space diagram."""
    # load emissions
    df = pd.read_csv(emissions_path)

    xmin, xmax = df['time'].min(), df['time'].max()
    ymin, ymax = df['position'].min(), df['position'].max()

    # compute line segment ends by shifting dataframe by 1 row
    df[['next_pos', 'next_time']] = df.groupby('id')[['position', 'time']].shift(-1)
    # remove nans from data
    df = df[df['next_time'].notna()]
    # generate segments for line collection
    segs = df[['time', 'position', 'next_time', 'next_pos']].values.reshape((len(df), 2, 2))

    # create figure
    fig, ax = plt.subplots()
    ax.set_xlim(df['time'].min(), df['time'].max())
    cdict = {
        'red': ((0, 0, 0), (0.2, 1, 1), (0.6, 1, 1), (1, 0, 0)),
        'green': ((0, 0, 0), (0.2, 0, 0), (0.6, 1, 1), (1, 1, 1)),
        'blue': ((0, 0, 0), (0.2, 0, 0), (0.6, 0, 0), (1, 0, 0))
    }
    cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

    # plot grid lines
    if inrix_path is not None and inrix_path != 'traj_default':
        with open(f'{inrix_path}/segments.json') as f:
            segments = json.load(f)
            for s in segments:
                if s > ROUTE_START and s < ROUTE_END:
                    ax.axhline(s, linestyle='-', color='grey', linewidth=0.5, alpha=0.7)
        time_lines = pd.read_csv(f'{inrix_path}/speed.csv')['time']
        for t in time_lines:
            if t > 0:
                ax.axvline(t, linestyle='-', color='grey', linewidth=0.5, alpha=0.7)

    # plot line segments
    lc = LineCollection(segs, cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=40))
    lc.set_array(df['speed'].values)
    lc.set_linewidth(1)
    ax.add_collection(lc)
    ax.autoscale()

    # plot grey rectangles
    rects = []
    # rectangle for lower ghost edge
    rects.append(Rectangle((xmin, ymin), xmax - xmin, ROUTE_START - ymin))
    # rectangle for upper ghost edge
    rects.append(Rectangle((xmin, ROUTE_END), xmax - xmin, ymax - ROUTE_END))

    pc = PatchCollection(rects, facecolor='grey', alpha=0.5, edgecolor=None)
    pc.set_zorder(20)
    ax.add_collection(pc)

    # add colorbar
    lc.set_clim(0, 40)
    axcb = fig.colorbar(lc)

    # set title and axes labels
    ax.set_title('Time-space diagram on trajectory env')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    axcb.set_label('Velocity (m/s)')

    # save
    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == '__main__':
    emissions_path, save_path = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:
        inrix_path = sys.argv[3]
    else:
        inrix_path = None
    plot_time_space_diagram(emissions_path, save_path, inrix_path)
