import gym
import time
import multiprocessing

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class TimePerIterCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.t0 = time.time()

    def _on_rollout_start(self):
        self.logger.record('time/iter_sgd_duration', time.time() - self.t0)

    def _on_rollout_end(self):
        self.t0 = time.time()

    def _on_step(self):
        return True

env = gym.make('CartPole-v1')
model = PPO('MlpPolicy', env, verbose=1, n_steps=640, n_epochs=100)
model.learn(total_timesteps=640*3, callback=TimePerIterCallback())

