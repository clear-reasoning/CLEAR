import argparse
import traceback
from collections import defaultdict
from datetime import datetime
import importlib
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
from pathlib import Path
import os
import re

import trajectory.config as tc
from trajectory.callbacks import TensorboardCallback
from trajectory.env.trajectory_env import DEFAULT_ENV_CONFIG, PLATOON_PRESETS, TrajectoryEnv
from trajectory.env.utils import get_first_element
from trajectory.visualize.plotter import Plotter
from trajectory.visualize.time_space_diagram import plot_time_space_diagram
from trajectory.visualize.render import Renderer


def parse_args_simulate(return_defaults=False):
    """Parse arguments to simulate.py
    return_defaults -- If enabled, return the default values rather than parsing from command line
    """
    parser = argparse.ArgumentParser(description='Simulate a trained controller or baselines on the trajectory env.')

    # trajectory
    parser.add_argument('--horizon', type=int, default=None,
                        help='Number of environment steps to simulate. If None, use a whole trajectory.')
    parser.add_argument('--traj_path', type=str,
                        default='dataset/data_v2_preprocessed_west/2021-04-22-12-47-13_2T3MWRFVXLW056972_masterArray_0_7050.csv',
                        help='Use a specific trajectory by default. Set to None to use a random trajectory.')
    parser.add_argument('--n_runs', type=int, default=1,
                        help='How many times to run the experiment. If > 1, will run several times the same experiment '
                             '(on the same trajectory if --traj_path is set) and compute average and variance of collected metrics.')
    parser.add_argument('--fast', default=False, action='store_true',
                        help='If set, emission files and plots will not be generated.')
    # leaderboard
    parser.add_argument('--data_pipeline', default=None, nargs=3,
                        help='If set, the emission file and metadata will be uploaded to leaderboard. '
                             'Arguments are [author] [strategy name] [is baseline]. '
                             'ie. --data_pipeline "Your name" "Your training strategy/controller name" True|False. '
                             'Note that [is baseline] should by default be set to False (or 0).')
    # render
    parser.add_argument('--render', default=False, action='store_true',
                        help='If set, the experiment will be rendered in a window (which is slower).')
    parser.add_argument('--keyboard', default=False, action='store_true',
                        help='[NOT IMPLEMENTED YET] '
                             'If set, the leader vehicle will be controlled using keyboard instead of following '
                             'a trajectory. Control: A to brake, D to accelerate, space bar + A or D to brake slower '
                             'or accelerate faster.')
    # vehicles
    parser.add_argument('--platoon', type=str, default='av human*5',
                        help='Platoon of vehicles following the leader. Can contain either "human"s or "av"s. '
                             '"(av human*2)*2" is a shortcut for "av human human av human human". '
                             'Vehicle tags can be passed with hashtags, eg. "av#tag", "human#tag*3". '
                             'Available presets: ' + ', '.join(
                            [f'{k} ({v})' for k, v in PLATOON_PRESETS.items()]).replace('%', '%%'))
    # avs controller
    parser.add_argument('--av_controller', type=str, default='idm',
                        help='Controller to control the AV(s) with. Can be either one of "rl", "idm" or "fs".')
    parser.add_argument('--av_kwargs', type=str, default='{}',
                        help='Kwargs to pass to the AV controller, as a string that will be evaluated into a dict. '
                             'For instance "{\'a\':1, \'b\': 2}" or "dict(a=1, b=2)" for IDM.')
    parser.add_argument('--cp_path', type=str, default=None,
                        help='Path to a saved model checkpoint when using --av_controller rl. '
                             'Checkpoint must be a .zip file and have a configs.json file in its parent directory.')
    parser.add_argument('--cp_dir', type=str, default=None,
                        help='Path to a directory of checkpoints when using --av_controller rl. '
                             'Directory must contain params.json')

    # humans controller
    parser.add_argument('--human_controller', type=str, default='idm',
                        help='Controller to control the humans(s) with. Can be either one of "idm" or "fs".')
    parser.add_argument('--human_kwargs', type=str, default='{}',
                        help='Kwargs to pass to the human vehicles, as a string that will be evaluated into a dict. '
                             'For instance "{\'a\':1, \'b\': 2}" or "dict(a=1, b=2)" for IDM.')
    # road
    parser.add_argument('--no_lc', default=False, action='store_true',
                        help='If set, disables the lane-changing model.')
    parser.add_argument('--road_grade', type=str, default=None,
                        help='Can be set to i24 or i680. If set, road grade will be included in the energy function computations.')

    if return_defaults:
        return parser.parse_args([])
    else:
        return parser.parse_args()
    # return args


logs_str = ''


def print_and_log(*args, output=True):
    global logs_str
    for string in args:
        logs_str += string
    logs_str += '\n'
    if output:
        print(*args)


def save_logs(exp_dir):
    logs_path = exp_dir / 'logs.txt'
    with open(logs_path, 'w') as f:
        f.write(logs_str)

    print(f'\nExperiment logs have been saved at {logs_path}')
    print(f'Experiment folder is {exp_dir}')


def simulate(args, cp_path=None, select_policy=False, df=None):
    assert args.human_controller in ['idm', 'fs']
    assert args.av_controller in ['rl', 'idm', 'fs']

    assert args.data_pipeline is None or args.n_runs == 1

    # logging function

    # generate env config
    env_config = DEFAULT_ENV_CONFIG

    # load AV controller
    if 'rl' in args.av_controller.lower():
        # load config file
        if not cp_path:
            cp_path = Path(args.cp_path)

        with open(cp_path.parent.parent / 'configs.json', 'r') as fp:
            configs = json.load(fp)
        env_config.update(configs['env_config'])

        # retrieve algorithm
        alg_module, alg_class = re.match("<class '(.+)\\.([a-zA-Z\\_]+)'>", configs['algorithm']).group(1, 2)
        assert (alg_module.split('.')[0] in ['stable_baselines3', 'algos'] or alg_module.split('.')[1] == 'algos')
        algorithm = getattr(importlib.import_module(alg_module), alg_class)

        # load checkpoint into model
        model = algorithm.load(cp_path)

        print_and_log(f'\nLoaded model checkpoint at {cp_path}')
        if not select_policy:
            print_and_log(f'\n\ttrained for {model.num_timesteps} timesteps'
                          f'\n\talgorithm = {alg_module}.{alg_class}'
                          f'\n\tobservation space = {model.observation_space}'
                          f'\n\taction space = {model.action_space}'
                          f'\n\tpolicy = {model.policy_class}'
                          f'\n\n{model.policy}')

        def get_action(state):
            return model.predict(state, deterministic=True)[0]

    env_config.update({
        'platoon': args.platoon,
        'whole_trajectory': True,
        'fixed_traj_path': (os.path.join(tc.PROJECT_PATH, args.traj_path) if args.traj_path != 'None' else None),
        'av_controller': args.av_controller,
        'av_kwargs': args.av_kwargs,
        'human_controller': args.human_controller,
        'human_kwargs': args.human_kwargs,
        'lane_changing': not args.no_lc,
        'road_grade': args.road_grade
    })

    if args.horizon is not None:
        env_config.update({
            'whole_trajectory': False,
            'horizon': args.horizon,
        })

    if not select_policy:
        now = datetime.now().strftime('%d%b%y_%Hh%Mm%Ss')
        timestamp = datetime.now().timestamp()
        exp_dir = Path(f'data/simulate/{int(timestamp)}_{now}/')
        exp_dir.mkdir(parents=True, exist_ok=False)
        print_and_log(f'Created experiment folder at {exp_dir}\n')

    exp_metrics = defaultdict(list)

    for i in range(args.n_runs):
        # create env
        test_env = TrajectoryEnv(config=env_config, _simulate=True, _verbose=False)
        if i == 0 and not select_policy:
            print_and_log('Running experiment with the following platoon:',
                          ' '.join([v.name for v in test_env.sim.vehicles]))
            print_and_log(f'with av controller {args.av_controller} (kwargs = {args.av_kwargs})')
            print_and_log(f'with human controller {args.human_controller} (kwargs = {args.human_kwargs})\n')

        state = test_env.reset()

        traj_path = test_env.traj['path']
        horizon = test_env.horizon
        print_and_log(f'Running experiment {i + 1}/{args.n_runs}, lasting {horizon} timesteps.')
        print_and_log(f'Using trajectory {traj_path}')

        # run one rollout
        test_env.start_collecting_rollout()
        done = False
        if args.render:
            renderer = Renderer()
        while not done:
            if 'rl' in args.av_controller:
                # get RL action
                action = [
                    get_first_element(model.predict(test_env.get_state(av_idx=i), deterministic=True))
                    for i in range(len(test_env.avs))
                ]
            else:
                # other controllers should be implemented via Vehicle objects
                action = 0
            state, reward, done, infos = test_env.step(action)

            if args.render:
                time = test_env.sim.time_counter
                veh_types = [v.kind for v in test_env.sim.vehicles]
                veh_positions = [v.pos for v in test_env.sim.vehicles]
                veh_speeds = [v.speed for v in test_env.sim.vehicles]
                renderer.step(time, veh_types, veh_positions, veh_speeds)
                renderer.render()

        test_env.stop_collecting_rollout()

        # generate emissions file and optionally upload to leaderboard
        if not args.fast:
            emissions_path = exp_dir / f'emissions/emissions_{i + 1}.csv'
            if args.data_pipeline is not None:
                metadata = {
                    'is_baseline': int(args.data_pipeline[2].lower() in ['true', '1', 't', 'y', 'yes']),
                    'author': args.data_pipeline[0],
                    'strategy': args.data_pipeline[1]}
                if len(match := re.findall('2avs_([0-9]+)%', args.platoon)) > 0:
                    pr = match[0]
                    if '.' not in pr:
                        pr += '.0'
                    metadata['penetration_rate'] = pr
                metadata['version'] = '4.0 wo LC' if args.no_lc else '4.0 w LCv0'
                print_and_log(f'Data will be uploaded to leaderboard with metadata {metadata}')
                if not args.fast:
                    test_env.gen_emissions(emissions_path=emissions_path, upload_to_leaderboard=True,
                                           additional_metadata=metadata)
            else:
                if not args.fast:
                    test_env.gen_emissions(emissions_path=emissions_path, upload_to_leaderboard=False)

        # gen metrics
        tb_callback = TensorboardCallback(eval_freq=0, eval_at_end=True)
        rollout_dict = tb_callback.get_rollout_dict(test_env)

        if not args.fast:
            # plot metrics
            plotter = Plotter(exp_dir / 'figs')
            for group, metrics in rollout_dict.items():
                for k, v in metrics.items():
                    plotter.plot(v, title=k, grid=True, linewidth=1.0)
                fig_name = f'{group}_{i + 1}'
                plotter.save(fig_name, log=False)
                if args.n_runs == 1:
                    print_and_log(f'Wrote {exp_dir / "figs" / fig_name}.png')

            # plot speed and accel profiles
            if args.no_lc:
                if args.platoon == 'av human*8':
                    # special plot for slides
                    veh_lst = [veh for veh in test_env.sim.vehicles[::-1] if veh.vid in [0, 1, 2, 5, 9]]
                    figsize = (7, 6)

                    with plotter.subplot(title='Velocity profiles', xlabel='Time (s)', ylabel='Velocity (m/s)',
                                         grid=True, legend=False):
                        for veh in veh_lst:
                            times = test_env.sim.get_data(veh, 'time')
                            speeds = test_env.sim.get_data(veh, 'speed')
                            plotter.plot(times, speeds, label=veh.name)
                    with plotter.subplot(title='Acceleration profiles', xlabel='Time (s)', ylabel='Acceleration (m/s²)',
                                         grid=True, legend=True):
                        for veh in veh_lst:
                            times = test_env.sim.get_data(veh, 'time')
                            speeds = test_env.sim.get_data(veh, 'target_accel_no_noise_with_failsafe')
                            plotter.plot(times, speeds, label=veh.name)
                else:
                    # plot profiles of AVs and their respective leaders
                    figsize = None
                    for k, veh in enumerate(test_env.avs):
                        veh_lst = [veh, veh.leader]
                        with plotter.subplot(title=f'Velocity profiles (AV {k + 1})', xlabel='Time (s)',
                                             ylabel='Velocity (m/s)', grid=True, legend=False):
                            for veh in veh_lst:
                                times = test_env.sim.get_data(veh, 'time')
                                speeds = test_env.sim.get_data(veh, 'speed')
                                plotter.plot(times, speeds, label=veh.name)
                        with plotter.subplot(title=f'Acceleration profiles (AV {k + 1})', xlabel='Time (s)',
                                             ylabel='Acceleration (m/s²)', grid=True, legend=True):
                            for veh in veh_lst:
                                times = test_env.sim.get_data(veh, 'time')
                                speeds = test_env.sim.get_data(veh, 'target_accel_no_noise_with_failsafe')
                                plotter.plot(times, speeds, label=veh.name)

                fig_name = f'speed_accel_profiles_{i + 1}'
                plotter.save(fig_name, log=False, figsize=figsize, legend_pos='auto')
                if args.n_runs == 1:
                    print_and_log(f'Wrote {exp_dir / "figs" / fig_name}.png')

            output_tsd_path = exp_dir / f'figs/time_space_diagram_{i + 1}.png'
            plot_time_space_diagram(emissions_path, output_tsd_path)
            if args.n_runs == 1:
                print_and_log(f'Wrote {output_tsd_path}\n')

        # accumulate metrics
        exp_metrics['system_mpg'].append(rollout_dict['system']['avg_mpg'][-1])
        exp_metrics['system_speed'].append(rollout_dict['system']['speed'][-1])
        exp_metrics['av_mpg'].append(rollout_dict['sim_data_av']['avg_mpg'][-1])
        for j in range(len(test_env.avs)):
            exp_metrics[f'platoon_{j}_mpg'].append(rollout_dict[f'platoon_{j}']['platoon_mpg'][-1])

        for penalty in ['crash', 'low_headway_penalty', 'large_headway_penalty', 'low_time_headway_penalty']:
            count_penalty = sum(rollout_dict['custom_metrics'][penalty])
            exp_metrics[f'count_{penalty}'].append(count_penalty)

        stat_fns = [('mean', np.mean), ('std', np.std), ('min', np.min), ('max', np.max)]
        for (name, array) in [
                                 ('av_headway', rollout_dict['sim_data_av']['headway']),
                                 ('av_speed', rollout_dict['base_state']['speed'])
                             ] \
                             + [(f'platoon_{j}_speed', rollout_dict[f'platoon_{j}']['platoon_speed']) for j in
                                range(len(test_env.avs))] \
                             + [
                                 ('av_leader_speed_difference', rollout_dict['sim_data_av']['speed_difference']),
                                 ('instant_energy_consumption',
                                  rollout_dict['sim_data_av']['instant_energy_consumption']),
                                 ('rl_reward', rollout_dict['training']['rewards']),
                             ]:
            for fn_name, fn in stat_fns:
                exp_metrics[f'{name} ({fn_name})'].append(fn(array))

        exp_metrics['rl_episode_reward'].append(np.sum(rollout_dict['training']['rewards']))

        exp_metrics['n_cutins'].append(test_env.sim.n_cutins)
        exp_metrics['n_cutouts'].append(test_env.sim.n_cutouts)
        for fn_name, fn in stat_fns:
            exp_metrics[f'n_vehicles ({fn_name})'].append(fn(test_env.sim.n_vehicles))

        plt.close('all')
        print()

    if select_policy:
        for i in range(args.n_runs):
            values = {'cp_path': cp_path, 'run': i}
            values.update({k: v[i] for k, v in exp_metrics.items()})

            # Append metrics to dataframe
            if not df.columns.any():
                row_index = df.shape[0]
                for k, v in values.items():
                    df.at[row_index, k] = v
            else:
                df.loc[df.shape[0]] = values

    else:
        print_and_log(f'Metrics aggregated over {args.n_runs} runs:\n')
        for k, v in exp_metrics.items():
            print_and_log(f'{k}: {np.mean(v):.2f} ± {np.std(v):.2f} (min = {np.min(v):.2f}, max = {np.max(v):.2f})')
        save_logs(exp_dir)


def find_best_policies(df, n=3):
    if isinstance(df, Path):
        df = pandas.read_pickle(df)

    mean_df = df.groupby('cp_path').mean()

    good_policies = mean_df[(mean_df["av_headway (max)"] < 350) & (mean_df["av_headway (min)"] > 0.5) & (
                mean_df["count_crash"] == 0)].sort_values(
        'system_mpg').head(n=n).index

    good_policy_metrics = df[df['cp_path'].isin(good_policies)].drop("run", axis=1).groupby('cp_path')

    means = good_policy_metrics.mean()
    stds = good_policy_metrics.std().replace(np.nan, 0)
    mins = good_policy_metrics.min()
    maxs = good_policy_metrics.max()

    print_and_log(f"Reasonable policies (positive reward, no crashes) by System MPG, best first:")

    for cp in good_policies:
        print_and_log("\n" + str(cp))
        for metric in means.loc[cp].keys():
            print_and_log(
                f'{metric}: {means.loc[cp][metric]:.2f} ± {stds.loc[cp][metric]:.2f} (min = {mins.loc[cp][metric]:.2f}, max = {maxs.loc[cp][metric]:.2f})')


def simulate_dir(args):
    cp_dir = Path(args.cp_dir).expanduser()
    args.fast = True

    assert cp_dir.is_dir(), f"{cp_dir} is not a directory"
    assert len(list(cp_dir.iterdir())) > 0

    now = datetime.now().strftime('%d%b%y_%Hh%Mm%Ss')
    timestamp = datetime.now().timestamp()
    exp_dir = Path(f'data/simulate/{int(timestamp)}_{now}/')
    exp_dir.mkdir(parents=True, exist_ok=False)

    df = pd.DataFrame().astype('object')

    for hparams_dir in cp_dir.iterdir():
        if str(hparams_dir).split("/")[-1] == "checkpoints":
            checkpoints = hparams_dir
        else:
            checkpoints = hparams_dir / 'checkpoints'

        if checkpoints.is_dir() and len(list(checkpoints.iterdir())) > 0:
            get_path = lambda path: re.search("\d+", str(path).split("/")[-1]).group(0)
            try:
                highest_cp_path = max(list(checkpoints.iterdir()), key=get_path)
            except ValueError:
                print_and_log(f"Issue with {checkpoints} directory")
                traceback.print_exc()

            simulate(args, highest_cp_path, select_policy=True, df=df)

    find_best_policies(df)

    pkl_file = str(exp_dir / "df.pkl")
    df.to_pickle(pkl_file)
    print_and_log(f"\nFull metrics (including data for each run) can be found at Pandas DataFrame saved to {pkl_file}")
    save_logs(exp_dir)


if __name__ == "__main__":
    # parse command line arguments
    args = parse_args_simulate()
    if args.cp_dir:
        simulate_dir(args)
    elif args.cp_path:
        simulate(args, Path(args.cp_path))
    else:
        simulate(args)
