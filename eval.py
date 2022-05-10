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
import telegram  # TODO(nl)


# TODO: remove from train set one we get synthetic trajectories merged
EVAL_TRAJECTORIES = map(Path, [
    'dataset/data_v2_preprocessed_west/2021-04-22-12-47-13_2T3MWRFVXLW056972_masterArray_0_7050/trajectory.csv',
    'dataset/data_v2_preprocessed_west/2021-04-12-21-34-57_2T3MWRFVXLW056972_masterArray_1_4436/trajectory.csv',
])

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate controllers in an experiment logdir.')

    parser.add_argument('--logdir', type=str, required=True,
                        help='Experiment logdir (eg. log/09May22/test_18h42m04s)')
    parser.add_argument('--telegram', default=False, action='store_true',
                        help='If set, you will receive a Telegram notification once eval is complete.')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    baseline_controller = 'idm'

    # parse args
    args = parse_args()
    exp_dir = Path(args.logdir)

    # create an eval folder within the exp logdir
    eval_dir = exp_dir / 'eval'
    eval_dir.mkdir()
    print('>', eval_dir)

    # get all grid searches
    rl_paths = []
    for config_path in exp_dir.rglob('configs.json'):
        # find latest checkpoint
        checkpoints_path = config_path.parent / 'checkpoints'
        cp_numbers = [int(f.stem) for f in checkpoints_path.glob('*.zip')]
        latest_cp_number = sorted(cp_numbers)[-1]
        latest_cp_path = checkpoints_path / f'{latest_cp_number}.zip'
        rl_paths.append((config_path, latest_cp_path))

    metrics = defaultdict(dict)

    # for each eval trajectory
    for eval_traj in EVAL_TRAJECTORIES:
        # create env config
        abstract_env_config = DEFAULT_ENV_CONFIG
        abstract_env_config.update({
            'whole_trajectory': True,
            'platoon': '(av human*7)*2',
            'fixed_traj_path': str(eval_traj),
            'human_controller': 'idm',
            'human_kwargs': 'dict()',
            'lane_changing': False,
            'road_grade': None,
        })

        # create trajectory logdir
        traj_name = eval_traj.parent.name
        traj_dir = eval_dir / traj_name
        traj_dir.mkdir()
        print('>', traj_dir)
        
        # plot velocity of trajectory
        df = pd.read_csv(eval_traj)
        traj_fig_path = traj_dir / 'trajectory.png'
        plot = df.plot(x="Time", y=["Velocity"])
        plot.get_figure().savefig(traj_fig_path)
        print('>', traj_fig_path)
        
        # run RL and IDM 
        for av_config in [
            {'av_controller': baseline_controller, 'av_kwargs': f'dict(noise=0)'},
            *[{'av_controller': 'av', 'av_kwargs': f'dict(config_path="{config_path}", cp_path="{cp_path}")'}
              for (config_path, cp_path) in rl_paths],
        ]:  
            av_name = av_config['av_controller'] if av_config['av_controller'] != 'av' \
                else eval(av_config['av_kwargs'])['cp_path']

            # create particular env config
            env_config = copy.deepcopy(abstract_env_config)
            env_config.update(av_config)

            # create env
            env = TrajectoryEnv(config=env_config, _simulate=True, _verbose=False)
            env.reset()

            # step through the whole trajectory
            done = False
            while not done:
                _, _, done, _ = env.step(None)

            # generate emission file
            emissions_path = traj_dir / f"emissions_{av_name.replace('/', '_')}.csv"
            env.gen_emissions(emissions_path=emissions_path, upload_to_leaderboard=False)

            # compute tsd
            tsd_path = traj_dir / f"tsd_{av_name.replace('/', '_')}.png"
            plot_time_space_diagram(emissions_path, save_path=tsd_path)
            print('>', tsd_path)

            # compute MPG metrics (AV, platoon, system ; low speeds vs high speeds)
            df = pd.read_csv(emissions_path)
            timestep = 0.1

            def extract_mpg(df):
                meters_per_second_to_miles = lambda meters_per_second: meters_per_second / 1609.34 * timestep
                gallons_per_hour_to_gallons = lambda gallons_per_hour: gallons_per_hour / 3600.0 * timestep

                miles = meters_per_second_to_miles(df['speed'].sum())
                gallons = gallons_per_hour_to_gallons(df['instant_energy_consumption'].sum())
                mpg = miles / gallons if gallons > 0 else None

                return mpg

            def extract_mpg_metrics(df):
                # system MPG: for all vehicles in the simulation
                system_mpg = extract_mpg(df)

                # AV MPG: for all AVs in the simulation
                df_avs = df[df['id'].str.contains('av')]
                avs_mpg = extract_mpg(df_avs)

                # platoon MPG: for all AVs + up to 5 human followers for each AV
                # TODO(nl): if incorporating lane-changing, this will need to be changed to 
                # account for platoon changes (use the 'follower_id' at each time step)
                platoon_ids = []
                veh_ids = df['id'].unique()
                for av_id in [vid for vid in veh_ids if 'av' in vid]:
                    platoon_ids.append(av_id)
                    av_num = av_id.split('_')[0]
                    for i in range(5):
                        follower_num = int(av_num) + 1 + i
                        follower_id = [vid for vid in veh_ids if vid.startswith(str(follower_num))][0]
                        platoon_ids.append(follower_id)
                df_platoons = df[df['id'].isin(platoon_ids)]
                platoons_mpg = extract_mpg(df_platoons)

                return system_mpg, avs_mpg, platoons_mpg

            mpgs = extract_mpg_metrics(df)

            speed_threshold = 20  # threshold for low speeds/high speeds

            low_speed_times = df[(df['id'].str.contains('trajectory')) & (df['speed'] < speed_threshold)]['time']
            df_low_speeds = df[df['time'].isin(low_speed_times)]
            mpgs_low_speeds = extract_mpg_metrics(df_low_speeds)

            high_speed_times = df[(df['id'].str.contains('trajectory')) & (df['speed'] >= speed_threshold)]['time']
            df_high_speeds = df[df['time'].isin(high_speed_times)]
            mpgs_high_speeds = extract_mpg_metrics(df_high_speeds)

            metrics[eval_traj][av_name] = [*mpgs, *mpgs_low_speeds, *mpgs_high_speeds]

            # delete emission file (heavy)
            emissions_path.unlink()

    field_names = [
        'System MPG', 'AVs MPG', 'Platoons MPG',
        'System MPG (LS)', 'AVs MPG (LS)', 'Platoons MPG (LS)',
        'System MPG (HS)', 'AVs MPG (HS)', 'Platoons MPG (HS)',
        'AV controller',
    ]

    def parse_mpg(mpg, baseline_mpg):
        if mpg is None or baseline_mpg is None:
            return None
        improvement = (mpg / baseline_mpg - 1) * 100
        return f'{mpg:.2f} ({"+" if improvement >= 0 else ""}{improvement:.2f}%)'

    tables = []
    metrics_sum_count = defaultdict(lambda: [(0,0)] * 9)
    for traj in metrics:
        x = prettytable.PrettyTable()
        x.field_names = field_names
        
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
