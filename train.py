from args import parse_args_train
from env.trajectory_env import TrajectoryEnv
from callbacks import CheckpointCallback, TensorboardCallback, ProgressBarCallback, LoggingCallback

import numpy as np
import matplotlib.pyplot as plt

import torch
from stable_baselines3.ppo import PPO
from stable_baselines3.td3 import TD3
from algos.ppo.policies import PopArtActorCriticPolicy
# from algos.ppo.ppo import PPO as AugmentedPPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import (
    register_policy,
)

from datetime import datetime
import os.path
import os
import json
import sys
import subprocess
import multiprocessing
import itertools
import platform

from train_setup import start_training


if __name__ == '__main__':
    # fix for macOS
    if platform.system() == 'Darwin':
        multiprocessing.set_start_method('spawn')

    args = parse_args_train()

    # parse command line args to separate grid searches from regular values
    fixed_config = {}
    grid_searches = {}
    for arg, value in vars(args).items():
        if type(value) is list:
            if len(value) == 0:
                raise ValueError('empty list in args')
            elif len(value) == 1:
                fixed_config[arg] = value[0]
            else:
                grid_searches[arg] = value
        else:
            fixed_config[arg] = value
    
    # generate cartesian product of grid search to generate all configs
    product_raw = itertools.product(*grid_searches.values())
    product_dicts = [dict(zip(grid_searches.keys(), values)) for values in product_raw]
    configs = [(fixed_config, gs_config) for gs_config in product_dicts]

    # print config and grid searches
    print('\nRunning experiment with the following config:\n')
    for k, v in fixed_config.items():
        print(f'\t{k}: {v}')
    if len(grid_searches) > 0:
        print(f'\nwith a total of {len(configs)} grid searches across the following parameters:\n')
        for k, v in grid_searches.items():
            print(f'\t{k}: {v}')
    print()

    # create exp logdir
    now = datetime.now().strftime('%d%b%y_%Hh%Mm%Ss')
    exp_logdir = os.path.join(args.logdir, f'{args.expname}_{now}')
    os.makedirs(exp_logdir, exist_ok=True)
    print(f'Created experiment logdir at {exp_logdir}')

    with open(os.path.join(exp_logdir, 'params.json'), 'w') as fp:
        git_branch = subprocess.check_output(['git', 'branch', '--show-current']).decode('utf8').split()[0]
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf8').split()[0]
        
        exp_dict = {
            'full_command': 'python ' + ' '.join(sys.argv),
            'timestamp': datetime.timestamp(datetime.now()),
            'git_branch': git_branch,
            'git_commit': git_commit,
            'args': vars(args),
        }

        class Encoder(json.JSONEncoder):
            def default(self, obj):
                try:
                    return json.JSONEncoder.default(self, obj)
                except TypeError:
                    return str(obj)

        json.dump(exp_dict, fp, indent=4, cls=Encoder)
        print(f'Saved exp params to {fp.name}')

    # save git diff to account for uncommited changes
    ps = subprocess.Popen(('git', 'diff', 'HEAD'), stdout=subprocess.PIPE)
    git_diff = subprocess.check_output(('cat'), stdin=ps.stdout).decode('utf8')
    ps.wait()
    if len(git_diff) > 0:
        with open(os.path.join(exp_logdir, 'git_diff.txt'), 'w') as fp:
            print(git_diff, file=fp)
            print(f'Saved git diff to {fp.name}')
    print()

    if len(configs) == 1:
        start_training((configs[0], exp_logdir))
    else:
        print(f'Starting training with {fixed_config["n_processes"]} parallel processes')
        with multiprocessing.Pool(processes=fixed_config['n_processes']) as pool:
            pool.map(start_training, zip(configs, [exp_logdir] * len(configs)))
        pool.close()
        pool.join()

    print('\nTraining terminated')