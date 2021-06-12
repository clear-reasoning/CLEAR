from env.trajectory_env import TrajectoryEnv
from callbacks import CheckpointCallback, TensorboardCallback, ProgressBarCallback, LoggingCallback

import numpy as np
import matplotlib.pyplot as plt

import torch
from stable_baselines3.ppo import PPO
from stable_baselines3.td3 import TD3
from algos.td3.policies import CustomTD3Policy
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
import multiprocessing
import itertools
import platform



def start_training(args):
    configs, exp_logdir = args
    fixed_config, grid_search_config = configs
    config = {**fixed_config, **grid_search_config}

    if len(grid_search_config) > 0:
        gs_str = '_' + '_'.join([f'{k}={v}' for k, v in grid_search_config.items()])
    else:
        gs_str = ''

    env_config = {
        'max_accel': 1.5,
        'max_decel': 3.0,
        'horizon': config['env_horizon'],
        'min_speed': 0,
        'max_speed': 40,
        'max_headway': config['env_max_headway'],
        'minimal_headway': 7,
        'whole_trajectory': False,
        'discrete': config['env_discrete'],
        'num_actions': config['env_num_actions'],
        'use_fs': config['use_fs'],
        'extra_obs': config['augment_vf'],
        # if we get closer then this time headway we are forced to break with maximum decel
        'minimal_time_headway': config['env_minimal_time_headway'],
        # if false, we only include the AVs mpg in the calculation
        'include_idm_mpg': config['env_include_idm_mpg'],
        'num_idm_cars': config['env_num_idm_cars'],
        'num_concat_states': config['env_num_concat_states'],
        'num_steps_per_sim': config['env_num_steps_per_sim']
    }
    gs_dir_path = exp_logdir if len(gs_str) == 0 else os.path.join( exp_logdir, gs_str[1:])
    os.makedirs(gs_dir_path, exist_ok=True)
    with open(os.path.join(gs_dir_path, 'env_config.json'), 'w') as fp:
        print(f'saved env config to {fp.name}')
        json.dump(env_config, fp)

    multi_env = make_vec_env(TrajectoryEnv, n_envs=config['n_envs'], env_kwargs=dict(config=env_config))

    if config['algorithm'].lower() == 'ppo':
        callbacks = []
        if len(grid_search_config) == 0:
            callbacks.append(ProgressBarCallback())
        
        if not config['no_eval']:
            callbacks.append(TensorboardCallback(
                eval_freq=config['eval_frequency'],
                eval_at_start=True,
                eval_at_end=True))
        callbacks += [
            LoggingCallback(
                grid_search_config=grid_search_config,
                log_metrics=len(grid_search_config) > 0),
            CheckpointCallback(
                save_path=os.path.join(exp_logdir, f'checkpoints{gs_str}'),
                save_freq=config['cp_frequency'],
                save_at_end=True),
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
            }[config['algorithm'].lower()]

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
                }[config['activation'].lower()],
                'net_arch': [{
                    'pi': [config['hidden_layer_size']] * config['network_depth'],
                    'vf': [config['hidden_layer_size']] * config['network_depth'],
                }],
                'optimizer_class': {
                    'adam': torch.optim.Adam,
                }[config['optimizer'].lower()],
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
    elif config['algorithm'].lower() == 'td3':
        callbacks = []

        callbacks += [
            TensorboardCallback(
                eval_freq=config['eval_frequency'],
                eval_at_start=True,
                eval_at_end=True),
            LoggingCallback(
                grid_search_config=grid_search_config,
                log_metrics=len(grid_search_config) > 0),
            CheckpointCallback(
                save_path=os.path.join(exp_logdir, f'checkpoints{gs_str}'),
                save_freq=config['cp_frequency'],
                save_at_end=True),
        ]
        callbacks = CallbackList(callbacks)
        algorithm = TD3
        if config['augment_vf']:
            policy = CustomTD3Policy
        else:
            policy = 'MlpPolicy'
        train_config = {
            'env': multi_env,
            'tensorboard_log': exp_logdir,
            'verbose': 0 if len(grid_search_config) > 0 else 1,  # 0 no output, 1 info, 2 debug
            'seed': None,  # only concerns PPO and not the environment
            'device': 'cpu',  # 'cpu', 'cuda', 'auto'

            # policy params
            'policy': policy,

            # # TD3 params
            'gamma':0.99,
            'learning_rate': 0.0003,
            'buffer_size': 1000000,
            'learning_starts': 100,
            'train_freq': 100,
            'gradient_steps': 100,
            'batch_size': 128,
            'tau': 0.005,
            'policy_delay': 2,
            'action_noise': None,
            'target_policy_noise': 0.2,
            'target_noise_clip': 0.5,
        }


    learn_config = {
        'total_timesteps': config['iters'] * config['n_steps'] * config['n_envs'],
        'tb_log_name': f'tb{gs_str}',
        'callback': callbacks,
        'log_interval': 1,  # print metrics every n rollouts
    }

    model = algorithm(**train_config)
    model.learn(**learn_config)
