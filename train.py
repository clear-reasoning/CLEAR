from args import parse_args
from env.trajectory_env import TrajectoryEnv
from callbacks import CheckpointCallback, TensorboardCallback, ProgressBarCallback, LoggingCallback

import numpy as np
import matplotlib.pyplot as plt

import torch
from stable_baselines3.ppo import PPO
from stable_baselines3.td3 import TD3
from algos.ppo.policies import PopArtActorCriticPolicy
from algos.ppo.ppo import PPO as AugmentedPPO
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
from multiprocessing import Pool
import itertools


if __name__ == '__main__':
    args = parse_args()

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

    # save args and metadata
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

    def start_training(configs):
        fixed_config, grid_search_config = configs
        config = {**fixed_config, **grid_search_config}

        if len(grid_search_config) > 0:
            gs_str = '_' + '_'.join([f'{k}={v}' for k, v in grid_search_config.items()])
        else:
            gs_str = ''

        env_config = {
            'max_accel': 1.5,
            'max_decel': 3.0,
            'horizon': 300,
            'min_speed': 0,
            'max_speed': 40,
            'max_headway': 80,
            'discrete': config['env_discrete'],
            'num_actions': config['env_num_actions'],
            'use_fs': config['use_fs'],
            'extra_obs': config['augment_vf'],
            # how close we need to be at the end to get the reward
            'closing_gap': .85,
            # if we get closer then this time headway we are forced to break with maximum decel
            'minimal_time_headway': 1.5
        }

        multi_env = make_vec_env(TrajectoryEnv, n_envs=config['n_envs'], env_kwargs=dict(config=env_config))

        callbacks = []
        if len(grid_search_config) == 0:
            callbacks.append(ProgressBarCallback())
        
        callbacks += [
            CheckpointCallback(
                save_path=os.path.join(exp_logdir, f'checkpoints{gs_str}'),
                save_freq=config['cp_frequency'],
                save_at_end=True),
            TensorboardCallback(
                eval_freq=args.eval_frequency,
                eval_at_start=True,
                eval_at_end=True),
            LoggingCallback(
                grid_search_config=grid_search_config,
                log_metrics=len(grid_search_config) > 0),
        ]
        callbacks = CallbackList(callbacks)

        if config['augment_vf']:
            from algos.ppo.policies import SplitActorCriticPolicy
            policy = SplitActorCriticPolicy
        else:
            register_policy("PopArtMlpPolicy", PopArtActorCriticPolicy)
            policy = PopArtActorCriticPolicy

        if config['augment_vf']:
            algorithm = AugmentedPPO
        else:
            algorithm = {
                'ppo': PPO,
            }[args.algorithm.lower()]

        train_config = {
            'env': multi_env,
            'tensorboard_log': exp_logdir,
            'verbose': 0 if len(grid_search_config) > 0 else 1,  # 0 no output, 1 info, 2 debug
            'seed': None,  # only concerns PPO and not the environment
            'device': 'cpu',  # 'cpu', 'cuda', 'auto'

            # policy params
            'policy': policy,
            'policy_kwargs': {
                'activation_fn': {
                    'tanh': torch.nn.Tanh,
                    'relu': torch.nn.ReLU,
                }[args.activation.lower()],
                'net_arch': [{
                    'pi': [config['hidden_layer_size']] * config['network_depth'], 
                    'vf': [config['hidden_layer_size']] * config['network_depth'],
                }],
                'optimizer_class': {
                    'adam': torch.optim.Adam,
                }[args.optimizer.lower()],
            },

            # PPO params
            'learning_rate': config['lr'],  # lr (*)
            'n_steps': config['n_steps'],  # rollout size is n_steps * n_envs (distinct from env horizon which can span across several rollouts)
            'batch_size': config['batch_size'],  #64 # minibatch size
            'n_epochs': config['n_epochs'],  # num sgd iter
            'gamma': config['gamma'],  # discount factor
            'gae_lambda': config['gae_lambda'],  # factor for trade-off of bias vs variance for Generalized Advantage Estimator
            'clip_range': 0.2,  # clipping param (*)
            'clip_range_vf': None,  # clipping param for the vf (*)
            'ent_coef': 0.0,  # entropy coef in loss function
            'vf_coef': 0.5,  # vf coef in loss function
            'max_grad_norm': 0.5,  # max value of grad clipping
            # (*) can be a function of the current progress remaining (from 1 to 0)
        }

        learn_config = {
            'total_timesteps': config['iters'] * config['n_steps'] * config['n_envs'],
            'tb_log_name': f'tb{gs_str}',
            'callback': callbacks,
            'log_interval': 1,  # print metrics every n rollouts
        }

        model = algorithm(**train_config)
        model.learn(**learn_config)

    with Pool(processes=fixed_config['n_processes']) as pool:
        pool.map(start_training, configs)
    pool.close()
    pool.join()

    print('\nTraining terminated')