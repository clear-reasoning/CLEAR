import argparse
from collections import defaultdict
from datetime import datetime
import importlib
import json
import numpy as np
from pathlib import Path
import os
import re

import trajectory.config as tc
from trajectory.callbacks import TensorboardCallback
from trajectory.env.trajectory_env import DEFAULT_ENV_CONFIG, TrajectoryEnv
from trajectory.env.utils import get_first_element
from trajectory.visualize.plotter import Plotter


def parse_args_simulate():
    parser = argparse.ArgumentParser(description='Simulate a trained controller or baselines on the trajectory env.')

    parser.add_argument('--cp_path', type=str, default=None,
        help='Path to a saved model checkpoint. '
             'Checkpoint must be a .zip file and have a configs.json file in its parent directory.')
    parser.add_argument('--verbose', default=False, action='store_true',  # not needed
        help='If set, print information about the loaded controller when {av_controller} is "rl".')
    parser.add_argument('--gen_emissions', default=False, action='store_true',  # by default yes, otherwise --fast, save all in one folder
        help='If set, a .csv emission file will be generated.')
    parser.add_argument('--gen_metrics', default=False, action='store_true',
        help='If set, some figures will be generated and some metrics printed.')
    parser.add_argument('--data_pipeline', default=None, nargs=3,
        help='If set, the emission file and metadata will be uploaded to leaderboard. '
             'Arguments are [author] [strategy name] [is baseline]. '
             'ie. --data_pipeline "Your name" "Your training strategy/controller name" True|False. '
             'Note that [is baseline] should by default be set to False (or 0).')

    parser.add_argument('--horizon', type=int, default=None,
        help='Number of environment steps to simulate. If None, use a whole trajectory.')
    parser.add_argument('--traj_path', type=str, default='dataset/data_v2_preprocessed/2021-03-26-21-26-45_2T3MWRFVXLW056972_masterArray_1_6131.csv',
        help='Use a specific trajectory by default. Set to None to use a random trajectory.')
    parser.add_argument('--platoon', type=str, default='av human*5',
        help='Platoon of vehicles following the leader. Can contain either "human"s or "av"s. '
             '"(av human*2)*2" can be used as a shortcut for "av human human av human human". '
             'Vehicle tags can be passed with hashtags, eg "av#tag" "human#tag*3". '
             'Available presets: "scenario1".')
    parser.add_argument('--av_controller', type=str, default='idm',
        help='Controller to control the AV(s) with. Can be either one of "rl", "idm" or "fs".')
    parser.add_argument('--av_kwargs', type=str, default='{}',
        help='Kwargs to pass to the AV controller, as a string that will be evaluated into a dict. '
             'For instance "{\'a\':1, \'b\': 2}" or "dict(a=1, b=2)" for IDM.')
    parser.add_argument('--human_controller', type=str, default='idm',
        help='Controller to control the humans(s) with. Can be either one of "idm" or "fs".')
    parser.add_argument('--human_kwargs', type=str, default='{}',
        help='Kwargs to pass to the human vehicles, as a string that will be evaluated into a dict. '
             'For instance "{\'a\':1, \'b\': 2}" or "dict(a=1, b=2)" for IDM.')

    parser.add_argument('--no_lc', default=False, action='store_true',
        help='If set, disables the lane-changing model.')

    parser.add_argument('--all_trajectories', default=False, action='store_true',
        help='If set, the script will be ran for all the trajectories in the dataset.')

    args = parser.parse_args()
    return args


# parse command line arguments
args = parse_args_simulate()

# load AV controller
if 'rl' in args.av_controller.lower():
    # load config file
    cp_path = Path(args.cp_path)
    with open(cp_path.parent.parent / 'configs.json', 'r') as fp:
        configs = json.load(fp)
    env_config = DEFAULT_ENV_CONFIG
    env_config.update(configs['env_config'])

    if args.av_controller == 'rl_fs':
        env_config['av_controller'] = 'rl_fs'


    # retrieve algorithm
    alg_module, alg_class = re.match("<class '(.+)\.([a-zA-Z\_]+)'>", configs['algorithm']).group(1, 2)
    assert (alg_module.split('.')[0] in ['stable_baselines3', 'algos'])
    algorithm = getattr(importlib.import_module(alg_module), alg_class)

    # load checkpoint into model
    model = algorithm.load(cp_path)

    print(f'\nLoaded model checkpoint at {cp_path}\n')
    if args.verbose:
        print(f'trained for {model.num_timesteps} timesteps')
        print(f'algorithm = {alg_module}.{alg_class}')
        print(f'observation space = {model.observation_space}')
        print(f'action space = {model.action_space}')
        print(f'policy = {model.policy_class}')
        print(f'\n{model.policy}\n')

    get_action = lambda state: model.predict(state, deterministic=True)[0]

else:
    env_config = DEFAULT_ENV_CONFIG
    env_config.update({
        'use_fs': False,
        'discrete': False,
        'human_controller': 'idm',
    })

env_config.update({
    'platoon': args.platoon,
    'whole_trajectory': True,
    'fixed_traj_path': (os.path.join(tc.PROJECT_PATH, args.traj_path)
                        if not args.all_trajectories and args.traj_path != 'None' else None),
    'av_controller': args.av_controller,
    'av_kwargs': args.av_kwargs,
    'human_kwargs': args.human_kwargs,
    'lane_changing': not args.no_lc
})

if args.horizon is not None:
    env_config.update({
        'whole_trajectory': False,
        'horizon': args.horizon,
    })

# create env
test_env = TrajectoryEnv(config=env_config, _simulate=True)

# execute controller on traj
mpgs = defaultdict(list)
now = datetime.now().strftime('%d%b%y_%Hh%Mm%Ss')

while True:
    try:
        state = test_env.reset()
    except StopIteration:
        print('Done.')
        break

    print('Using trajectory', test_env.traj['path'], '\n')
    done = False
    test_env.start_collecting_rollout()
    while not done:
        if 'rl' in args.av_controller:
            if False:
                state = test_env.get_state(av_idx=0)
                state[3:] = [0,0,0]
                output = model.predict(state, deterministic=True)
                print('model input', state)
                print('model output', output)
            action = [
                get_first_element(model.predict(test_env.get_state(av_idx=i), deterministic=True))
                for i in range(len(test_env.avs))
            ]
        else:
            action = 0  # do not change (controllers should be implemented via Vehicle objects)
        state, reward, done, infos = test_env.step(action)
    test_env.stop_collecting_rollout()

    # generate_emissions
    if args.all_trajectories:
        traj_name = Path(test_env.traj['path']).stem
        emissions_path = f'emissions/{now}/emissions_{args.av_controller}_{traj_name}.csv'
    else:
        emissions_path = None

    if args.gen_emissions or args.data_pipeline is not None:
        print('Generating emissions...')
        if args.data_pipeline is not None:
            metadata = {
                'is_baseline': int(args.data_pipeline[2].lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'ya']),
                'author': args.data_pipeline[0],
                'strategy': args.data_pipeline[1]
            }
            if len(match := re.findall('2avs_([0-9]+)%', args.platoon)) > 0:
                pr = match[0]
                if '.' not in pr:
                    pr += '.0'
                metadata['penetration_rate'] = pr
            metadata['version'] = '4.0 wo LC' if args.no_lc else '4.0 w LCv0'
            print(f'Data will be uploaded to leaderboard with metadata {metadata}')
            test_env.gen_emissions(emissions_path=emissions_path,
                                   upload_to_leaderboard=True,
                                   additional_metadata=metadata)
        else:
            test_env.gen_emissions(emissions_path=emissions_path,
                                   upload_to_leaderboard=False)

    if args.gen_metrics:
        tb_callback = TensorboardCallback(eval_freq=0, eval_at_end=True)  # temporary shortcut
        rollout_dict = tb_callback.get_rollout_dict(test_env)

        # plot stuff
        print()
        now = datetime.now().strftime('%d%b%y_%Hh%Mm%Ss')
        plotter = Plotter(f'figs/simulate/{now}')
        for group, metrics in rollout_dict.items():
            for k, v in metrics.items():
                plotter.plot(v, title=k, grid=True, linewidth=1.0)
            plotter.save(group, log='\t')
        print()

        # print stuff
        print('\nMetrics:')
        episode_reward = np.sum(rollout_dict['training']['rewards'])
        av_mpg = rollout_dict['sim_data_av']['avg_mpg'][-1]
        print('\tepisode_reward', episode_reward)
        print('\tav_mpg', av_mpg)
        for penalty in ['crash', 'low_headway_penalty', 'large_headway_penalty', 'low_time_headway_penalty']:
            has_penalty = int(any(rollout_dict['custom_metrics'][penalty]))
            print(f'\thas_{penalty}', has_penalty)

        for (name, array) in [
            ('reward', rollout_dict['training']['rewards']),
            ('headway', rollout_dict['sim_data_av']['headway']),
            ('speed_difference', rollout_dict['sim_data_av']['speed_difference']),
            ('instant_energy_consumption', rollout_dict['sim_data_av']['instant_energy_consumption']),
        ]:
            print(f'\tmin_{name}', np.min(array))
            print(f'\tmax_{name}', np.max(array))

    if not args.all_trajectories:
        break


