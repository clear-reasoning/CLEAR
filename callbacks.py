from typing import Dict
import argparse
import numpy as np
import os

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

from env.trajectory_env import TrajectoryEnv, SPEED_SCALE, DISTANCE_SCALE
from env.accel_controllers import IDMController


class MPGCallback(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        # episode.user_data['metric'] = []
        pass

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        # Make sure this episode is ongoing.
        assert episode.length > 0, \
            "ERROR: `on_episode_step()` callback should not be called right " \
            "after env reset!"
        # env = base_env.get_unwrapped()[0]
        # episode.user_data['metric'].append(0)
        pass

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        env = base_env.get_unwrapped()[0]

        for controller in ['rl', 'idm']:
            test_env = TrajectoryEnv(config=env.config)
            test_env.whole_trajectory = True
            state = test_env.reset()
            done = False
            total_reward = 0
            total_distance = 0
            total_energy = 0

            actions = []
            headways = []
            speed_deltas = []

            if controller == 'idm':
                idm = IDMController(a=env.max_accel, b=env.max_decel)
                test_env.use_fs = False  # for IDM only

            while not done:
                if controller == 'rl':
                    action = policies['default_policy'].compute_single_action(state, clip_actions=True, explore=False)[0][0]
                elif controller == 'idm':
                    action = idm.get_accel(state[0] * SPEED_SCALE, state[1] * SPEED_SCALE, state[2] * DISTANCE_SCALE)
                state, reward, done, _ = test_env.step(action)

                actions.append(action)
                headways.append(state[2] * DISTANCE_SCALE)
                speed_deltas.append( np.abs(state[0] - state[1]) * SPEED_SCALE )

                total_reward += reward
                total_distance += test_env.av['speed']
                total_energy += test_env.energy_model.get_instantaneous_fuel_consumption(test_env.av['last_accel'], test_env.av['speed'], grade=0)

            episode.custom_metrics[f"{controller}_traj_mpg"] = (total_distance / 1609.34) / (total_energy / 3600 + 1e-6)
            episode.custom_metrics[f"{controller}_traj_reward"] = total_reward

            episode.custom_metrics[f"{controller}_avg_abs_action"] = np.mean(np.abs(actions))
            episode.custom_metrics[f"{controller}_std_abs_action"] = np.std(np.abs(actions))
            episode.custom_metrics[f"{controller}_min_action"] = np.min(actions)
            episode.custom_metrics[f"{controller}_max_action"] = np.max(actions)

            episode.custom_metrics[f"{controller}_avg_headway"] = np.mean(headways)
            episode.custom_metrics[f"{controller}_std_headway"] = np.std(headways)
            episode.custom_metrics[f"{controller}_min_headway"] = np.min(headways)
            episode.custom_metrics[f"{controller}_max_headway"] = np.max(headways)

            episode.custom_metrics[f"{controller}_avg_speed_delta"] = np.mean(speed_deltas)
            episode.custom_metrics[f"{controller}_std_speed_delta"] = np.std(speed_deltas)
            episode.custom_metrics[f"{controller}_min_speed_delta"] = np.min(speed_deltas)
            episode.custom_metrics[f"{controller}_max_speed_delta"] = np.max(speed_deltas)


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