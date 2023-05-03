"""Generate time-space diagram."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

from matplotlib import colors
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle

ROUTE_START = 0
ROUTE_END = 14000
MPH_TO_MS = 0.44704


def plot_time_space_diagram(emissions_path, save_path, inrix_path=None):
    """Plot time-space diagram."""
    # load emissions
    df = pd.read_csv(emissions_path)
    df = df.assign(is_av=df['id'].str.contains('av'))

    system_mpg, total_gallons, total_miles, system_speed, speed_std, \
        av_mpg, total_gallons_av, total_miles_av, av_speed, av_speed_std, \
        throughput = compute_metrics(df)

    xmin, xmax = df['time'].min(), df['time'].max()
    ymin, ymax = df['position'].min(), df['position'].max()

    # compute line segment ends by shifting dataframe by 1 row
    df[['next_pos', 'next_time']] = df.groupby('id')[['position', 'time']].shift(-1)
    # remove nans from data
    df = df[df['next_time'].notna()]
    # generate segments for line collection
    segs = df[['time', 'position', 'next_time', 'next_pos']].values.reshape((len(df), 2, 2))
    av_df = df[df['is_av']]
    av_segs = av_df[['time', 'position', 'next_time', 'next_pos']].values.reshape((len(av_df), 2, 2))

    # create figure
    fig = plt.figure(figsize=(20, 10))
    spec = fig.add_gridspec(nrows=5, ncols=2)
    ax = fig.add_subplot(spec[:, 0])
    ax.set_xlim(df['time'].min(), df['time'].max())
    cdict = {
        'red': ((0, 0, 0), (0.2, 1, 1), (0.6, 1, 1), (1, 0, 0)),
        'green': ((0, 0, 0), (0.2, 0, 0), (0.6, 1, 1), (1, 1, 1)),
        'blue': ((0, 0, 0), (0.2, 0, 0), (0.6, 0, 0), (1, 0, 0))
    }
    cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

    for line, tput in throughput['lines']['total'].items():
        ax.axhline(line, linestyle='--')
        ax.text(x=10, y=line+50, s='{} veh/hr'.format(round(tput, 2)),
                fontsize=12,
                horizontalalignment='left', verticalalignment='bottom')
    ax.text(x=10, y=20000,
            s='System MPG: {} mpg\nSystem VMT: {} miles\nFuel: {} gals\nSystem avg speed: {} mph\nSpeed stddev: {} mph'.format(
                system_mpg.round(2), total_miles.round(2), total_gallons.round(2), system_speed.round(2), (speed_std/MPH_TO_MS).round(2)),
            fontsize=12, horizontalalignment='left', verticalalignment='top')
    # ax.text(x=500, y=20000,
    #         s='AV MPG: {} mpg\nAV VMT: {} miles\nAV fuel: {} gals\nAV avg speed: {} mph\nAV speed stddev: {} mph'.format(
    #             av_mpg.round(2), total_miles_av.round(2), total_gallons_av.round(2),
    #             av_speed.round(2), (av_speed_std/MPH_TO_MS).round(2)),
    #         fontsize=12, horizontalalignment='left', verticalalignment='top')

    # plot grid lines
    if inrix_path is not None and inrix_path != 'traj_default':
        # with open(f'{inrix_path}/segments.json') as f:
        #     segments = json.load(f)
        #     for s in segments:
        #         if s > ROUTE_START and s < ROUTE_END:
        #             ax.axhline(s, linestyle='-', color='grey', linewidth=0.5, alpha=0.7)
        file_path = os.path.join(inrix_path, "speed.csv")
        if os.path.exists(file_path):
            time_lines = pd.read_csv(file_path)['time']
            for t in time_lines:
                if t > 0:
                    ax.axvline(t, linestyle='-', color='grey', linewidth=0.5, alpha=0.7)

    # plot line segments
    lc = LineCollection(segs, cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=40))
    lc.set_array(df['speed'].values)
    lc.set_linewidth(1)
    ax.add_collection(lc)
    av_lc = LineCollection(av_segs, colors='k')
    av_lc.set_linewidth(1)
    ax.add_collection(av_lc)
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
    axcb = plt.colorbar(lc, ax=ax)

    # set title and axes labels
    ax.set_title('Time-Space Diagram', fontsize=26)
    ax.set_xlabel('Time (s)', fontsize=20)
    ax.set_ylabel('Position (m)', fontsize=20)
    bottom, top = ax.get_ylim()
    ax.set_ylim(max(bottom, -8000), min(top, 20000))
    axcb.set_label('Velocity (m/s)', fontsize=16)

    # i = 0
    # t_range = throughput['time_range']
    # axleft = {}
    # max_tput = max([max(tput) for _, tput in throughput['lines']['time_series'].items()])
    # for line, tput_line in throughput['lines']['time_series'].items():
    #     axleft[i] = fig.add_subplot(spec[4 - i, 1])
    #     axleft[i].plot(t_range, tput_line, linewidth=2, label='Pos: {}m'.format(line))
    #     axleft[i].legend(loc='upper right')
    #     axleft[i].set_ylim(0, max_tput*1.05)
    #     if i == 2:
    #         axleft[i].set_ylabel('Throughput (veh/hr)', fontsize=20)
    #     if i == 0:
    #         axleft[i].set_xlabel('Time (s)', fontsize=20)
    #     i += 1
    # ax2 = plt.subplot(122)
    # ax2.plot(queue_length[:, 0], queue_length[:, 1], linewidth=2)
    # ax2.set_ylim(0, 1200)
    # ax2.set_ylabel('Queue Length (m)', fontsize=20)
    # ax2.set_xlabel('Time (s)', fontsize=20)
    # ax2.set_title('Queue Length over Time', fontsize=26)

    # save
    plt.tight_layout()
    plt.savefig(save_path)


def compute_metrics(df):
    """Compute metrics from dataframe."""
    # get throughput at various lines
    lines = [0, 3500, 7000, 10500, 14000]
    t_end = int(df['time'].max() / 60) * 60
    t_range = np.arange(0, t_end, 60)
    half_width = 180  # 3 minutes on either side

    throughput = {
        'time_range': t_range,
        'lines': {
            'time_series': {},
            'total': {},
        }
    }
    for line in lines:
        tput_line = np.zeros(len(t_range))
        for i, t_mid in enumerate(t_range):
            t_start = t_mid - half_width
            t_stop = t_mid + half_width
            before_ids = set(df.loc[(df['position'] < line) & (df['time'].between(t_start, t_stop)), 'id'].unique())
            after_ids = set(df.loc[(df['position'] > line) & (df['time'].between(t_start, t_stop)), 'id'].unique())
            crossing_ids = before_ids.intersection(after_ids)

            first_time = df.loc[(df['id'].isin(crossing_ids)) & (df['position'] > line), 'time'].min()
            last_time = df.loc[(df['id'].isin(crossing_ids)) & (df['position'] < line), 'time'].max()

            if len(crossing_ids) > 0:
                tput_line[i] = len(crossing_ids) / (last_time - first_time) * 3600
        throughput['lines']['time_series'][line] = tput_line

        before_ids = set(df.loc[df['position'] < line, 'id'].unique())
        after_ids = set(df.loc[df['position'] > line, 'id'].unique())
        crossing_ids = before_ids.intersection(after_ids)

        first_time = df.loc[(df['id'].isin(crossing_ids)) & (df['position'] > line), 'time'].min()
        last_time = df.loc[(df['id'].isin(crossing_ids)) & (df['position'] < line), 'time'].max()

        throughput['lines']['total'][line] = len(crossing_ids) / (last_time - first_time) * 3600

    df = df[(df['position'] >= ROUTE_START) & (df['position'] <= ROUTE_END)]

    # get queue length over time
    # slow_threshold = 12
    # df = df.assign(slow=df['speed'] < slow_threshold)
    # queue_length = []
    # for time, group in df.groupby('time'):
    #     grp = group.sort_values('position', ascending=True)
    #     grp['queue_num'] = (grp['slow'] != grp['slow'].shift(1)).cumsum()
    #     queue_len = 0
    #     for queue_num, queue in grp.groupby('queue_num'):
    #         if queue['slow'].values[0]:
    #             queue_len += queue['position'].max() - queue['position'].min()
    #     queue_length.append(np.array([time, queue_len]))
    # queue_length = np.array(queue_length)

    speed_std = df['speed'].std()
    av_speed_std = df[df['id'].str.contains('av')]['speed'].std()

    # get mpg and speed metrics
    total_miles = 0
    total_gallons = 0
    total_seconds = 0
    total_miles_av = 0
    total_gallons_av = 0
    total_seconds_av = 0
    for veh_id, group in df.groupby('id'):
        total_gallons += group['total_gallons'].max() - group['total_gallons'].min()
        total_miles += group['total_miles'].max() - group['total_miles'].min()
        total_seconds += len(group) * 0.1
        if 'av' in veh_id:
            total_gallons_av += group['total_gallons'].max() - group['total_gallons'].min()
            total_miles_av += group['total_miles'].max() - group['total_miles'].min()
            total_seconds_av += len(group) * 0.1
    system_mpg = total_miles / total_gallons
    system_speed = total_miles / total_seconds * 3600
    av_mpg = total_miles_av / total_gallons_av
    av_speed = total_miles_av / total_seconds_av * 3600

    return (system_mpg, total_gallons, total_miles, system_speed, speed_std,
            av_mpg, total_gallons_av, total_miles_av, av_speed, av_speed_std,
            throughput)


if __name__ == '__main__':
    emissions_path, save_path = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:
        inrix_path = sys.argv[3]
    else:
        inrix_path = None
    plot_time_space_diagram(emissions_path, save_path, inrix_path)