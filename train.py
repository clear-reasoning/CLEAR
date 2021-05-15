import ray
from ray import tune
from ray.tune import CLIReporter

import numpy as np
import matplotlib.pyplot as plt

from callbacks import MPGCallback, PlotTrajectoryCallback
from env.trajectory_env import TrajectoryEnv
from args import parse_args


if __name__ == '__main__':
    args = parse_args()

    exp_config = {
        'run_or_experiment': 'PPO',
        'name': args.expname,
        'config': {
            'env': TrajectoryEnv,
            'env_config': {
                'max_accel': 1.5,
                'max_decel': 3.0,
                'horizon': 500,
                'min_speed': 0,
                'max_speed': 40,
                'max_headway': 120,
                'use_fs': args.use_fs,
            },
            'num_gpus': 0,
            'model': {
                'vf_share_layers': True,
                'fcnet_hiddens': [64, 64],
                'use_lstm': False,
            },
            'vf_loss_coeff': 1e-7,
            'lr': 5e-5,
            'gamma': 0.99,
            'num_workers': 3,
            # 'vf_clip_param': 100,
            'framework': 'torch',
            'train_batch_size': 40000,
            'sgd_minibatch_size': 4000,
            'num_sgd_iter': 10,
            'batch_mode': 'complete_episodes',
            'explore': True,
            'normalize_actions': True,
            'clip_actions': True,
            "grad_clip": 40.0,
            "callbacks": MPGCallback if not args.plot_trajectory else PlotTrajectoryCallback,
        },
        'stop': {
            'training_iteration': args.iters,
        },
        'local_dir': './ray_results',
        'checkpoint_freq': 20,
        'checkpoint_at_end': True,
        'verbose': 1,
        'log_to_file': False,
        'restore': None,
        'progress_reporter': CLIReporter(metric_columns={
            'training_iteration': 'iter',
            'time_this_iter_s': 'time iter (s)',
            'time_total_s': 'total',
            'timesteps_total': 'ts',
            'episodes_this_iter': 'ep iter',
            'episodes_total': 'total',
            'episode_reward_mean': 'ep rwd mean',
            'episode_reward_min': 'min',
            'episode_reward_max': 'max',
            'episode_len_mean': 'ep len mean',
            'info/learner/default_policy/learner_stats/policy_loss': 'policy loss',
            'info/learner/default_policy/learner_stats/vf_loss': 'vf loss',
            'info/learner/default_policy/learner_stats/kl': 'kl',
            'info/learner/default_policy/learner_stats/entropy': 'entropy',
            'custom_metrics/avg_mpg_mean': 'avg_mpg'
        }),
    }

    ray.init()
    ray.tune.run(**exp_config)
    ray.shutdown()
