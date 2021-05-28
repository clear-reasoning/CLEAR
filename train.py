from args import parse_args
from env.trajectory_env import TrajectoryEnv
from callbacks import CheckpointCallback, TensorboardCallback, ProgressBarCallback, LoggingCallback

import numpy as np
import matplotlib.pyplot as plt

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env

from datetime import datetime
import os.path
import os
import json
import sys
import subprocess


if __name__ == '__main__':
    args = parse_args()

    now = datetime.now().strftime('%d%b%y_%Hh%Mm%Ss')
    exp_logdir = os.path.join(args.logdir, f'{args.expname}_{now}')
    os.makedirs(exp_logdir, exist_ok=True)

    env_config = {
        'max_accel': 1.5,
        'max_decel': 3.0,
        'horizon': 500,
        'min_speed': 0,
        'max_speed': 40,
        'max_headway': 120,
        # 'use_fs': args.use_fs,
    }

    multi_env = make_vec_env(TrajectoryEnv, n_envs=args.n_envs, env_kwargs=dict(config=env_config))

    callbacks = CallbackList([
        ProgressBarCallback(),
        CheckpointCallback(   
            save_path=os.path.join(exp_logdir, 'checkpoints'),
            save_freq=args.cp_frequency,
            save_at_end=True),
        TensorboardCallback(
            eval_freq=args.eval_frequency,
            eval_at_start=True,
            eval_at_end=True),
        LoggingCallback(),
    ])

    algorithm = {
        'ppo': PPO,
    }[args.algorithm.lower()]

    train_config = {
        'env': multi_env,
        'tensorboard_log': exp_logdir,
        'verbose': 1,  # 0 no output, 1 info, 2 debug
        'seed': None,  # only concerns PPO and not the environment
        'device': 'cpu',  # 'cpu', 'cuda', 'auto'

        # policy params
        'policy': 'MlpPolicy',
        'policy_kwargs': {
            'activation_fn': {
                'tanh': torch.nn.Tanh,
                'relu': torch.nn.ReLU,
            }[args.activation.lower()],
            'net_arch': [{
                'pi': args.hidden_layers, 
                'vf': args.hidden_layers,
            }],
            'optimizer_class': {
                'adam': torch.optim.Adam,
            }[args.optimizer.lower()],
        },

        # PPO params
        'learning_rate': args.lr,  # lr (*)
        'n_steps': args.n_steps,  # rollout size is n_steps * n_envs (distinct from env horizon which can span across several rollouts)
        'batch_size': args.batch_size,  #64 # minibatch size
        'n_epochs': args.n_epochs,  # num sgd iter
        'gamma': args.gamma,  # discount factor
        'gae_lambda': 0.95,  # factor for trade-off of bias vs variance for Generalized Advantage Estimator
        'clip_range': 0.2,  # clipping param (*)
        'clip_range_vf': None,  # clipping param for the vf (*)
        'ent_coef': 0.0,  # entropy coef in loss function
        'vf_coef': 0.5,  # vf coef in loss function
        'max_grad_norm': 0.5,  # max value of grad clipping
        # (*) can be a function of the current progress remaining (from 1 to 0)
    }

    learn_config = {
        'total_timesteps': args.iters * args.n_steps * args.n_envs,
        'tb_log_name': 'tb',
        'callback': callbacks,
        'log_interval': 1,  # print metrics every n rollouts
    }

    with open(os.path.join(exp_logdir, 'params.json'), 'w') as fp:
        git_branch = subprocess.check_output(['git', 'branch', '--show-current']).decode('utf8').split()[0]
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf8').split()[0]
        
        exp_dict = {
            'full_command': 'python ' + ' '.join(sys.argv),
            'timestamp': datetime.timestamp(datetime.now()),
            'git_branch': git_branch,
            'git_commit': git_commit,
            'args': vars(args),
            'env_config': env_config,
            'train_config': train_config,
            'learn_config': learn_config,
        }

        class Encoder(json.JSONEncoder):
            def default(self, obj):
                try:
                    return json.JSONEncoder.default(self, obj)
                except TypeError:
                    return str(obj)

        json.dump(exp_dict, fp, indent=4, cls=Encoder)
        print(f'Saved exp params to {fp.name}')

    model = algorithm(**train_config)
    model.learn(**learn_config)
