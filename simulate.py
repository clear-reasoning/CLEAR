from datetime import datetime
import importlib
import json
import numpy as np
from pathlib import Path
import re

from args import parse_args_simulate
from callbacks import TensorboardCallback
from env.trajectory_env import DEFAULT_ENV_CONFIG, TrajectoryEnv
from env.utils import get_first_element
from env.simulation import Simulation
from visualize.plotter import Plotter


# parse command line arguments
args = parse_args_simulate()

# load AV controller
if args.av_controller.lower() == 'rl':
    # load config file
    cp_path = Path(args.cp_path)
    with open(cp_path.parent.parent / 'configs.json', 'r') as fp:
        configs = json.load(fp)
    env_config = DEFAULT_ENV_CONFIG
    env_config.update(configs['env_config'])

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
        'av_controller': args.av_controller,
        'av_kwargs': args.av_kwargs,
        'human_controller': 'idm',
        'human_kwargs': args.human_kwargs,
    })

env_config.update({
    'platoon': args.platoon,
    'whole_trajectory': True,
})

# create env
test_env = TrajectoryEnv(config=env_config)

# execute controller on traj
state = test_env.reset()
done = False
test_env.start_collecting_rollout()
while not done:
    if args.av_controller == 'rl':
        action = [
            get_first_element(model.predict(test_env.get_state(av_idx=i), deterministic=True))
            for i in range(len(test_env.avs))
        ]
    else:
        action = 0  # do not change (controllers should be implemented via Vehicle objects)
    state, reward, done, infos = test_env.step(action)
test_env.stop_collecting_rollout()

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

# generate_emissions
if args.gen_emissions:
    test_env.gen_emissions(upload_to_leaderboard=args.s3, platoon=args.platoon)

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