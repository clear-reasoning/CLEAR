# TODO

from env import SimpleRoad
import ray
from ray.rllib.agents import ppo

from ray.rllib.models import ModelCatalog
from model import MyFullyConnectedNetwork
import numpy as np

class Model(object):
    def __init__(self, ckpt_path):
        ray.init()
        ModelCatalog.register_custom_model('my_fcnet', MyFullyConnectedNetwork)

        self.config = {
            'env': SimpleRoad,
            'env_config': {
                'max_accel': 1.5,
                'max_decel': 3.0,
                'road_length': 1000,
                'sim_step': 0.3,
            },
            'num_gpus': 0,
            'model': {
                'custom_model': 'my_fcnet',
                'vf_share_layers': False,
                'fcnet_hiddens': [64, 64],
            },
            'lr': 1e-4,
            'gamma': 0.9,
            'num_workers': 2, 
            'framework': 'torch',
        }

        agent = ppo.PPOTrainer(self.config, env=SimpleRoad)
        agent.restore(ckpt_path)
        self.policy = agent.workers.local_worker().get_policy()

    def get_state(self):
        av_speed = self.av.speed / 30.0
        leader_speed = self.leader.speed / 30.0
        headway = (self.leader.pos - self.av.pos) / self.road_length
        state = np.array([av_speed, leader_speed, headway])
        return state

    def act(self, speed, leader_speed, headway):
        state = np.array([speed / 30.0, leader_speed / 30.0, headway / self.config['env_config']['road_length']])
        return self._act(state)

    def _act(self, state):
        action = self.policy.compute_actions([state])
        return action[0][0][0]  # other interesting things in there

cp = './ray_results/alpha_v0.1/PPO_SimpleRoad_98e53_00000_0_2021-05-04_12-15-06/checkpoint_001000/checkpoint-1000'
model = Model(cp)

import matplotlib.pyplot as plt

ego_speed = 5
lead_speed_range = np.linspace(0, 10, 50)
headway_range = np.linspace(0, 30, 50)

lead_speeds, headways = np.meshgrid(lead_speed_range, headway_range)

accels = np.zeros_like(lead_speeds)
for i in range(lead_speeds.shape[0]):
    for j in range(lead_speeds.shape[1]):
        accels[-1-i,j] = model.act(ego_speed, lead_speeds[i,j], headways[i,j])


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))

im = ax.imshow(accels, cmap=plt.cm.RdBu, interpolation='bilinear')
fig.colorbar(im, ax=ax)

ax.set_title(f'Ego speed 5 m/s')
ax.set_xlabel('Leader speed (m/s)')
ax.set_ylabel('Headway (m)')

plt.savefig('accels.png', dpi=300)
