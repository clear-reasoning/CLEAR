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
        dist_travelled = abs(episode.last_observation_for()[0] * 50 * env.time_step)
        energy_consumed = sum([env.energy_model.get_instantaneous_fuel_consumption(car['last_accel'], car['speed'], grade=0)
                        for car in [env.av] + env.idm_followers]) / (1 + len(env.idm_followers))
        episode.user_data["curr_distance"].append(dist_travelled)
        episode.user_data["curr_energy"].append(energy_consumed)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        cumulative_gallons = np.sum(episode.user_data["curr_energy"])
        cumulative_distance = np.sum(episode.user_data["curr_distance"])
        cumulative_gallons /= 3600.0
        cumulative_distance /= 1609.34
        # miles / gallon is (distance_dot * \delta t) / (gallons_dot * \delta t)
        mpg = cumulative_distance / (cumulative_gallons + 1e-6)
        episode.custom_metrics["avg_mpg"] = mpg
