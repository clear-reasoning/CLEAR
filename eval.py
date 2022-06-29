# script to evaluate controllers in an experiment logdir (handles grid searches)
# no lane change / road grade functionality for now

from pathlib import Path
import argparse
from trajectory.env.trajectory_env import TrajectoryEnv, DEFAULT_ENV_CONFIG
from trajectory.visualize.time_space_diagram import plot_time_space_diagram
import pandas as pd
import copy
from collections import defaultdict
import prettytable
import matplotlib.pyplot as plt
import multiprocessing
from itertools import repeat
from os.path import join as opj
import trajectory.config as tc


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate controllers in an experiment logdir.')

    parser.add_argument('--logdir', type=str, required=True,
                        help='Experiment logdir (eg. log/09May22/test_18h42m0gamma4s) OR '
                             'sweep dir (e.g. log/09May22/test_18h42m04s/gamma=0.999/ OR '
                             'a folder containing one configs.json and one checkpoint.zip file.')
    parser.add_argument('--n_cpus', type=int, default=1,
                        help='Set to the number of parallel processes you wish to run.')

    # If no paths specified, will evaluate on the 7050 trajectory only
    parser.add_argument('--trajectories', default='one_traj', type=str, nargs='?',
                        choices=['one_traj', 'low_speed', 'high_speed', 'west', 'east', 'all'],
                        help='Which set of trajectories to evaluate on')
    parser.add_argument('--traj_path', default='dataset/data_v2_preprocessed_west/'
                                               '2021-04-22-12-47-13_2T3MWRFVXLW056972_masterArray_0_7050/trajectory.csv',
                        help="if --trajectories is 'one_traj', which trajectory to evaluate on")

    args = parser.parse_args()
    return args


def run_eval(env_config, traj_dir):
    av_name = env_config['av_controller'] if env_config['av_controller'] != 'av' \
        else eval(env_config['av_kwargs'])['cp_path']

    # create env
    env = TrajectoryEnv(config=env_config, _simulate=True, _verbose=False)
    env.reset()

    # step through the whole trajectory
    done = False
    while not done:
        _, _, done, _ = env.step(None)

    # create controller dir
    controller_dir = traj_dir / av_name.replace('/', '_')
    controller_dir.mkdir(exist_ok=True)

    # generate emission file
    emissions_path = controller_dir / 'emissions.csv'
    env.gen_emissions(emissions_path=emissions_path, upload_to_leaderboard=False)

    # compute tsd
    tsd_path = controller_dir / 'tsd.png'
    plot_time_space_diagram(emissions_path, save_path=tsd_path)
    print('>', tsd_path)

    # load emissions data
    df = pd.read_csv(emissions_path)
    timestep = 0.1

    # compute trajectory plots
    traj_leader_id = [vid for vid in df['id'].unique() if 'leader' in vid][0]
    av_ids = [vid for vid in df['id'].unique() if 'av' in vid]

    # plot speed of leader and all avs
    plt.figure(figsize=(15, 3))
    for veh_id in [traj_leader_id] + av_ids:
        # color red for leader and blue for AVs
        if veh_id == traj_leader_id:
            color = (0.8, 0.2, 0.2, 1.0)
        else:
            # hardcoded for a platoon of (av human*24)*8
            av_number = int(veh_id.split('_')[0])
            color = (0.2, 0.2, 0.8, 1.0) if av_number == 1 \
                else (0.2, 0.8, 0.2, 1.0) if av_number == 176 \
                else (0.2, 0.2, 1.0, 0.2)
        df_av = df[df['id'] == veh_id]
        plt.plot(df_av['time'] / timestep, df_av['speed'], label=veh_id, linewidth=2.0, color=color)
    plt.title('platoon speeds')
    plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.01, 0.5))
    plt.grid()
    plt.xlim(0, (df['time'] / timestep).max())
    fig_path = controller_dir / 'speed_avs_leader.png'
    plt.tight_layout()
    plt.savefig(fig_path)
    print('>', fig_path)

    # trajectory plots
    # plot av speed+leader speed, av accel and av gap as a function of time in 3 separate subplots
    for av_id in av_ids:
        df_av = df[df['id'] == av_id]
        plt.figure(figsize=(12, 7))
        plt.subplot(311)
        plt.plot(df_av['time'], df_av['speed'], label=av_id, linewidth=2.0)
        plt.plot(df_av['time'], df_av['leader_speed'], label='leader', linewidth=2.0)
        plt.grid()
        plt.xlim(0, df_av['time'].max())
        plt.title('speeds')
        plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.01, 0.5))
        plt.subplot(312)
        plt.plot(df_av['time'], df_av['accel'], label=av_id, linewidth=2.0)
        plt.grid()
        plt.xlim(0, df_av['time'].max())
        plt.title('accels')
        plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.01, 0.5))
        plt.subplot(313)
        plt.plot(df_av['time'], df_av['headway'], label=av_id, linewidth=2.0)
        gap_closing_threshold = [max(env.max_headway, env.max_time_headway * vel)
                                 for vel in df_av['speed']]
        failsafe_threshold = [6 * ((this_vel + 1 + this_vel * 4 / 30) - lead_vel)
                              for this_vel, lead_vel in zip(df_av['speed'], df_av['leader_speed'])]
        plt.plot(df_av['time'], gap_closing_threshold, label='gap closing threshold', linewidth=2.0)
        plt.plot(df_av['time'], failsafe_threshold, label='failsafe threshold', linewidth=2.0)
        plt.grid()
        plt.xlim(0, df_av['time'].max())
        plt.title('headway')
        plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.01, 0.5))
        plt.tight_layout()
        fig_path = controller_dir / f'traj_{av_id}.png'
        plt.savefig(fig_path)
        print('>', fig_path)

    # compute MPG metrics (AV, platoon, system ; low speeds vs high speeds)
    def meters_per_second_to_miles(meters_per_second):
        return meters_per_second / 1609.34 * timestep

    def gallons_per_hour_to_gallons(gallons_per_hour):
        return gallons_per_hour / 3600.0 * timestep

    def extract_mpg(df):
        miles = meters_per_second_to_miles(df['speed'].sum())
        gallons = gallons_per_hour_to_gallons(df['instant_energy_consumption'].sum())
        mpg = miles / gallons if gallons > 0 else None

        return mpg

    def extract_mpg_metrics(df):
        # system MPG: for all vehicles in the simulation
        system_mpg = extract_mpg(df)
        # print('SYSTEM MPG', av_name, system_mpg)

        # AV MPG: for all AVs in the simulation
        df_avs = df[df['id'].str.contains('av')]
        avs_mpg = extract_mpg(df_avs)

        # platoon MPG: for all AVs + up to 5 human followers for each AV
        # TODO(nl): if incorporating lane-changing, this will need to be changed to
        # account for platoon changes (use the 'follower_id' at each time step)
        # platoon_ids = []
        # veh_ids = df['id'].unique()
        # for av_id in [vid for vid in veh_ids if 'av' in vid]:
        #     av_num = av_id.split('_')[0]
        #     for i in range(5):
        #         follower_num = int(av_num) + 1 + i
        #         follower_id = [vid for vid in veh_ids if vid.startswith(str(follower_num))][0]
        #         platoon_ids.append(follower_id)
        # df_platoons = df[df['id'].isin(platoon_ids)]
        # platoons_mpg = extract_mpg(df_platoons)
        # platoons_mpg = None
        return system_mpg, avs_mpg  # , platoons_mpg

    mpgs = extract_mpg_metrics(df)

    speed_threshold = 20  # threshold for low speeds/high speeds

    low_speed_times = df[(df['id'].str.contains('trajectory')) & (df['speed'] < speed_threshold)]['time']
    df_low_speeds = df[df['time'].isin(low_speed_times)]
    mpgs_low_speeds = extract_mpg_metrics(df_low_speeds)

    high_speed_times = df[(df['id'].str.contains('trajectory')) & (df['speed'] >= speed_threshold)]['time']
    df_high_speeds = df[df['time'].isin(high_speed_times)]
    mpgs_high_speeds = extract_mpg_metrics(df_high_speeds)

    # delete emission file (heavy)
    emissions_path.unlink()

    # return metrics
    return (av_name, [*mpgs, *mpgs_low_speeds, *mpgs_high_speeds])


def generate_metrics(eval_dir, lane_changing, eval_trajectories):
    eval_dir.mkdir(exist_ok=True)
    print('>', eval_dir)

    metrics = defaultdict(dict)
    # for each eval trajectory
    for eval_traj in eval_trajectories:
        # create env config
        abstract_env_config = DEFAULT_ENV_CONFIG
        abstract_env_config.update({
            'whole_trajectory': True,
            'platoon': '(av human*24)*8',
            'fixed_traj_path': str(eval_traj),
            'human_controller': 'idm',
            'human_kwargs': 'dict()',
            'lane_changing': lane_changing,
            'road_grade': None,
        })

        # create trajectory logdir
        traj_name = eval_traj.parent.name
        traj_dir = eval_dir / traj_name
        traj_dir.mkdir(exist_ok=True)
        print('>', traj_dir)

        # plot velocity of trajectory
        df = pd.read_csv(eval_traj)
        traj_fig_path = traj_dir / 'trajectory.png'
        plot = df.plot(x="Time", y=["Velocity"])
        plot.get_figure().savefig(traj_fig_path)
        print('>', traj_fig_path)

        # run controllers
        av_configs = [
            {'av_controller': baseline_controller, 'av_kwargs': 'dict(noise=0)'},
            *[{'av_controller': 'av', 'av_kwargs': f'dict(config_path="{config_path}", cp_path="{cp_path}")'}
              for (config_path, cp_path) in rl_paths],
        ]
        av_env_configs = []
        for av_config in av_configs:
            env_config = copy.deepcopy(abstract_env_config)
            env_config.update(av_config)
            av_env_configs.append(env_config)

        with multiprocessing.Pool(processes=args.n_cpus) as pool:
            iterable = zip(av_env_configs, repeat(traj_dir))
            data = pool.starmap(run_eval, iterable)
            for av_name, av_metrics in data:
                metrics[eval_traj][av_name] = av_metrics

    field_names = [
        'System MPG', 'AVs MPG',  # 'Platoons MPG',
        'System MPG (LS)', 'AVs MPG (LS)',  # 'Platoons MPG (LS)',
        'System MPG (HS)', 'AVs MPG (HS)',  # 'Platoons MPG (HS)',
        'AV controller',
    ]

    def parse_mpg(mpg, baseline_mpg):
        if mpg is None or baseline_mpg is None:
            return None
        improvement = (mpg / baseline_mpg - 1) * 100
        return f'{mpg:.2f} ({"+" if improvement >= 0 else ""}{improvement:.2f}%)'

    tables = []
    metrics_sum_count = defaultdict(lambda: [(0, 0)] * 6)
    for traj in metrics:
        x = prettytable.PrettyTable()
        x.field_names = field_names
        x.sortby = "System MPG"
        x.reversesort = True

        for controller in metrics[traj]:
            mpg_metrics = list(map(parse_mpg, metrics[traj][controller], metrics[traj][baseline_controller]))
            x.add_row([*mpg_metrics, controller])

            for i, mpg in enumerate(metrics[traj][controller]):
                if mpg is not None:
                    current_sum, current_count = metrics_sum_count[controller][i]
                    metrics_sum_count[controller][i] = (current_sum + mpg, current_count + 1)
        tables.append((traj, x))

    avg_metrics_path = eval_dir / 'metrics.txt'
    x = prettytable.PrettyTable()
    x.field_names = field_names
    x.sortby = "System MPG"
    x.reversesort = True
    metrics_avg_baseline = [s / c if c > 0 else None for s, c in metrics_sum_count[baseline_controller]]
    for controller in metrics_sum_count:
        metrics_avg = [s / c if c > 0 else None for s, c in metrics_sum_count[controller]]
        mpg_metrics = list(map(parse_mpg, metrics_avg, metrics_avg_baseline))
        x.add_row([*mpg_metrics, controller])

    with open(avg_metrics_path, 'w') as fp:
        fp.write('Average metrics\n\n' + str(x))
        fp.write('\n\n' + 'Metrics for each trajectory')
        for traj, x_traj in tables:
            fp.write('\n\n' + str(traj) + '\n' + str(x_traj))
    print('>', avg_metrics_path)


if __name__ == '__main__':
    baseline_controller = 'idm'

    # parse args
    args = parse_args()
    logdir = Path(args.logdir)
    print("Evaluating all checkpoints in", logdir)

    eval_name = "eval_" + args.trajectories

    # If running eval on all sweeps of a run
    if (logdir / "params.json").exists():
        exp_dir = logdir
        # create an eval folder within the exp logdir
        eval_root = exp_dir / eval_name
        eval_root.mkdir(exist_ok=True)
        config_paths = exp_dir.rglob('configs.json')
    # If running eval on a specific sweep
    elif (logdir / "configs.json").exists():
        exp_dir = logdir.parent
        # create an eval folder within the sweep logdir
        eval_root = logdir / eval_name
        eval_root.mkdir(exist_ok=True)
        config_paths = [logdir / "configs.json"]
    else:
        raise ValueError("Invalid logdir", logdir)

    print('>', eval_root)

    rl_paths = []
    # get all grid searches
    for config_path in config_paths:
        # find latest checkpoint
        checkpoints_path = config_path.parent / 'checkpoints'
        cp_numbers = [int(f.stem) for f in checkpoints_path.glob('*.zip')]
        if len(cp_numbers) > 0:
            # get the latest checkpoint
            latest_cp_number = sorted(cp_numbers)[-1]
            latest_cp_path = checkpoints_path / f'{latest_cp_number}.zip'
        elif (config_path.parent / 'checkpoint.zip').exists():
            # if no checkpoints, but a checkpoint.zip exists, use that
            latest_cp_path = config_path.parent / 'checkpoint.zip'
        else:
            # if no checkpoints, and no checkpoint.zip, skip this config
            raise ValueError('No checkpoints found in', config_path.parent)
        rl_paths.append((config_path, latest_cp_path))

    EVAL_TRAJECTORIES = []
    if args.trajectories == 'one_traj':
        EVAL_TRAJECTORIES = [Path(args.traj_path)]
        print(f"Evaluating on {args.traj_path}")
    else:
        paths = {
            'low_speed': ['dataset/data_v2_preprocessed_west_low_speed/'],
            'high_speed': ['dataset/data_v2_preprocessed_west_high_speed/'],
            'west': ['dataset/data_v2_preprocessed_west/'],
            'east': ['dataset/data_v2_preprocessed_east/'],
            'all': ['dataset/data_v2_preprocessed_west/', 'dataset/data_v2_preprocessed_east/']
        }
        print("Evaluating on trajectories in the following directories:", paths[args.trajectories])
        for path in paths[args.trajectories]:
            EVAL_TRAJECTORIES += list(Path(opj(tc.PROJECT_PATH, path)).glob('*/trajectory.csv'))

    lc_dir = eval_root / "lc"
    no_lc_dir = eval_root / "no_lc"
    generate_metrics(eval_dir=lc_dir, lane_changing=True, eval_trajectories=EVAL_TRAJECTORIES)
    generate_metrics(eval_dir=no_lc_dir, lane_changing=False, eval_trajectories=EVAL_TRAJECTORIES)
