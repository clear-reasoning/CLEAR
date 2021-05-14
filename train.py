import ray

from env.trajectory_env import TrajectoryEnv
from args import parse_args

import numpy as np
import matplotlib.pyplot as plt

from ray.tune import grid_search
from progress_reporter import CLIReporter

from ray.rllib.agents.callbacks import DefaultCallbacks


class PlotTrajectoryCallback(DefaultCallbacks):

    def __init__(self, legacy_callbacks_dict=None):

        self.rewards_1 = []
        self.rewards_2 = []

        self.exploit_1 = []
        self.exploit_2 = []

        self.play_pref_1 = []
        self.play_pref_2 = []

        self.max_reward = []

        super().__init__(legacy_callbacks_dict)

    def on_sample_end(self, *, worker, samples, **kwargs):

        if worker.worker_index == 0 or worker.worker_index == 1:

            rewards = samples['rewards']
            actions = samples['actions'][:500]
            actions = np.clip(actions, -1, 1)
            observations = samples['obs']
            headway = observations[:500, 2] * 100.0
            speed = observations[:500, 0] * 50.0
            lead_speed = observations[:500, 1] * 50.0

            if observations.shape[-1] == 4:
                num_plots = 5
                plt.figure(figsize=(5, 16))
                plt.tight_layout()
            else:
                plt.figure(figsize=(5, 13))
                plt.tight_layout()
                num_plots = 4

            plt.subplot(num_plots, 1, 1)
            plt.cla()
            plt.plot(range(1, len(headway) + 1), headway)
            plt.title("time-step vs. headway")

            plt.subplot(num_plots, 1, 2)
            plt.cla()
            plt.plot(range(1, len(speed) + 1), speed)
            plt.title("time-step vs. speed")

            plt.subplot(num_plots, 1, 3)
            plt.cla()
            plt.plot(range(1, len(lead_speed) + 1), lead_speed)
            plt.title("time-step vs. lead speed")

            plt.subplot(num_plots, 1, 4)
            plt.cla()
            plt.plot(range(1, len(actions) + 1), actions, label='action')
            plt.plot(range(1, len(actions) + 1), 0.5 * (actions ** 2), label='action penalty')
            print('action penalty was ', np.sum(0.5 * (actions ** 2)))
            ax = plt.gca()
            ax.legend(loc="upper right")
            plt.title("time-step vs. lead speed")

            if num_plots == 5:
                plt.subplot(num_plots, 1, 5)
                plt.cla()
                plt.plot(range(1, len(lead_speed) + 1), observations[:500, 3] * 50.0, label='v_des')
                plt.plot(range(1, len(speed) + 1), speed, label='speed')
                ax = plt.gca()
                ax.legend(loc="upper right")
                plt.title("time-step vs. v._des")

            local_lr = worker.creation_args()['policy_config']['lr']
            path = './figs/env_trajectory/test.png'
            plt.savefig(path)

            plt.close()

            # plt.pause(0.05)

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
                'max_headway': 80,
                'use_fs': args.use_fs,
            },
            'num_gpus': 0,
            'model': {
                'vf_share_layers': True,
                'fcnet_hiddens': [64, 64],
                'use_lstm': False,
            },
            # 'vf_loss_coeff': 1e-5,
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
            "grad_clip": 40.0
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
        }),
    }

    if args.plot_trajectory:
        exp_config['config']['callbacks'] = PlotTrajectoryCallback

    ray.init()
    ray.tune.run(**exp_config)
    ray.shutdown()
