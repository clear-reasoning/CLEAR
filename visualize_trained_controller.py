# TODO

from env.trajectory_env import TrajectoryEnv
import ray
from ray.rllib.agents import ppo

import matplotlib.pyplot as plt
import numpy as np
from env.failsafes import safe_velocity

class Model(object):
    def __init__(self, ckpt_path):
        ray.init()

        self.config = {
            'env': TrajectoryEnv,
            'env_config': {
                'max_accel': 1.5,
                'max_decel': 3.0,
                'horizon': 500,
                'min_speed': 0,
                'max_speed': 40,
                'max_headway': 70,
            },
            'num_gpus': 0,
            'model': {
                # 'vf_share_layers': True,
                'fcnet_hiddens': [64, 64],
                # 'use_lstm': True,
            },
            'vf_loss_coeff': 1e-5,
            'lr': 1e-4,
            'gamma': 0.95,
            'num_workers': 2, 
            'framework': 'torch',
            'train_batch_size': 5000,
            'batch_mode': 'complete_episodes',
            'explore': True,
        }

        agent = ppo.PPOTrainer(self.config, env=TrajectoryEnv)
        agent.restore(ckpt_path)
        self.agent = agent
        # self.policy = agent.workers.local_worker().get_policy()

    def act(self, speed, leader_speed, headway):
        state = np.array([speed / 50.0, leader_speed / 50.0, headway / 100])
        return self._act(state)

    def _act(self, state):
        # with lstm cf https://github.com/ray-project/ray/issues/9220
        action = self.agent.compute_action(state, explore=False)
        return action[0]  # other interesting things in there

cp = '/Users/eugenevinitsky/Desktop/Research/Code/trajectory_training/ray_results/trajectory_env/PPO_TrajectoryEnv_0c5a2_00000_0_2021-05-10_14-17-58/checkpoint_000100/checkpoint-100'
model = Model(cp)

ego_speed = 15
max_decel = 3.0
max_accel = 1.5
sim_step = 0.1
lead_speed_range = np.linspace(20, 10, 50)
headway_range = np.linspace(0, 100, 100)

lead_speeds, headways = np.meshgrid(lead_speed_range, headway_range)

accels = np.zeros_like(lead_speeds)
for i in range(lead_speeds.shape[0]):
    for j in range(lead_speeds.shape[1]):
        accels[-1-i,j] = model.act(ego_speed, lead_speeds[i,j], headways[i,j])

        # v_safe = safe_velocity(ego_speed, lead_speeds[i,j],
        #                        headways[i,j], max_decel, sim_step)
        # v_next = accels[-1-i,j] * sim_step + ego_speed
        # if v_next > v_safe:
        #     accel = np.clip((v_safe - ego_speed) / sim_step, -np.abs(max_decel), max_accel)
        #     accels[-1-i, j] = accel


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))

im = ax.imshow(accels, cmap=plt.cm.RdBu, interpolation='bilinear')
fig.colorbar(im, ax=ax)

ax.set_title(f'Ego speed {ego_speed} m/s')
ax.set_xlabel('Leader speed (m/s)')
ax.set_ylabel('Headway (m)')

plt.savefig('accels.png', dpi=300)
