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


class MPGCallback(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        episode.user_data["curr_distance"] = []
        episode.user_data["curr_energy"] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        # Make sure this episode is ongoing.
        assert episode.length > 0, \
            "ERROR: `on_episode_step()` callback should not be called right " \
            "after env reset!"
        env = base_env.get_unwrapped()[0]
        distances = []
        energies = []
        for car in [env.av, *env.idm_followers]:
            # don't multiply by env.time_step because they will cancel out
            distances.append(car['speed'])
            energies.append(env.energy_model.get_instantaneous_fuel_consumption(car['last_accel'], car['speed'], grade=0))
        episode.user_data["curr_distance"].append(sum(distances))
        episode.user_data["curr_energy"].append(sum(energies))

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        cumulative_gallons = np.sum(episode.user_data["curr_energy"]) / 3600.0
        cumulative_distance = np.sum(episode.user_data["curr_distance"]) / 1609.34
        # miles / gallon is (distance_dot * \delta t) / (gallons_dot * \delta t)
        mpg = cumulative_distance / (cumulative_gallons + 1e-6)
        episode.custom_metrics["avg_mpg"] = mpg


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