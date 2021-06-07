from random import randint
import gym
from gym.spaces import Discrete, Box
import numpy as np
from collections import defaultdict

from dataset.data_loader import load_data
from env.accel_controllers import IDMController, TimeHeadwayFollowerStopper
from env.energy_models import PFMMidsizeSedan
from env.failsafes import safe_velocity


DISTANCE_SCALE = 100
SPEED_SCALE = 40


class TrajectoryEnv(gym.Env):
    def __init__(self, config):
        super(TrajectoryEnv, self).__init__()

        self.config = config

        self.max_accel = config['max_accel']
        self.max_decel = config['max_decel']
        self.horizon = config.get('horizon', 500)
        # set to some low value for a curriculum over horizon
        self.horizon_counter = config.get('horizon', 500)

        self.min_speed = config.get('min_speed', 0)
        self.max_speed = config.get('max_speed', 40)
        self.use_fs = config.get('use_fs')
        self.max_headway = config.get('max_headway', 120)
        self.extra_obs = config.get('extra_obs')
        # what percentage of the leaders trajectory we need to have covered to get the final reward
        self.closing_gap = config.get('closing_gap')
        self.minimal_time_headway = config.get('minimal_time_headway')

        self.whole_trajectory = config.get('whole_trajectory', False)
        self.step_num = 0

        self.time_step, self.leader_positions, self.leader_speeds = load_data()
        # for now get positions from velocities to ignore in-lane-changes
        self.leader_positions = [self.leader_positions[0]]
        for vel in self.leader_speeds[:-1]:
            self.leader_positions.append(self.leader_positions[-1] + vel * self.time_step)
        assert(len(self.leader_positions) == len(self.leader_speeds))

        if config.get('discrete'):
            self.use_discrete = True
            self.num_actions = config.get('num_actions', 7)
            self.action_space = Discrete(self.num_actions)
            self.action_set = np.linspace(-1, 1, self.num_actions)
        else:
            self.use_discrete = False
            self.action_space = Box(low=-3.0, high=1.5, shape=(1,), dtype=np.float32)

        if self.use_fs:
            obs_shape = 4
            if self.extra_obs:
                obs_shape *= 2
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
            self.state_names = ['speed', 'leader_speed', 'headway', 'vdes']
            self.state_scales = [SPEED_SCALE, SPEED_SCALE, DISTANCE_SCALE, SPEED_SCALE]
        else:
            obs_shape = 3
            if self.extra_obs:
                obs_shape *= 2
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
            self.state_names = ['speed', 'leader_speed', 'headway']
            self.state_scales = [SPEED_SCALE, SPEED_SCALE, DISTANCE_SCALE]

        self.idm_controller = IDMController(a=self.max_accel, b=self.max_decel, noise=0.5)
        self.follower_stopper = TimeHeadwayFollowerStopper(max_accel=self.max_accel, max_deaccel=self.max_decel)
        self.energy_model = PFMMidsizeSedan()

        self.emissions = False
        if self.emissions:
            self.emissions_data = defaultdict(list)

        self.reset()

    def normalize_state(self, state):
        return np.array([state[name] / scale
                         for name, scale in zip(self.state_names, self.state_scales)])

    def unnormalize_state(self, state):
        return {name: state[i] * scale
                for i, (name, scale) in enumerate(zip(self.state_names, self.state_scales))}
    
    def get_state(self):
        state = {
            'speed': self.av['speed'],
            'leader_speed': self.leader_speeds[self.traj_idx],
            'headway': self.leader_positions[self.traj_idx] - self.av['pos'],
        }

        if self.use_fs:
            state['vdes'] = self.follower_stopper.v_des

        return self.normalize_state(state)
    
    def reset(self):
        self.step_num += 1
        # start at random time in trajectory
        total_length = len(self.leader_positions)
        if self.whole_trajectory:
            self.traj_idx = 0
            self.horizon = total_length - 1
        else:
            self.traj_idx = randint(0, total_length - self.horizon - 1)
        self.env_step = 0

        # create av behind leader
        self.av = {
            'pos': self.leader_positions[self.traj_idx] - max(2.1 * self.leader_speeds[self.traj_idx], 20),
            'speed': self.leader_speeds[self.traj_idx],
            'last_accel': -1,
        }
        self.init_leader_pos = self.leader_positions[self.traj_idx]
        self.accumulated_headway = 0
        self.accumulated_pos = 0
        self.average_speed = 0
        if self.use_fs:
            self.follower_stopper.v_des = self.leader_speeds[self.traj_idx]

        # create idm followers behind av
        self.idm_followers = [{
            'pos': self.av['pos'] - 20 * (i + 1),
            'speed': self.av['speed'],
            'last_accel': -1,
        } for i in range(5)]

        if self.emissions:
            self.emissions_data = defaultdict(list)
            self.step_emissions()

        self.energy_consumption = [0 for _ in range(len(self.idm_followers) + 1)]
        self.init_pos = [car['pos'] for car in [self.av] + self.idm_followers]

        if self.extra_obs:
            return np.concatenate((self.get_state(), np.zeros(int(self.observation_space.low.shape[0] / 2))))
        else:
            return self.get_state()

    def step_emissions(self):

        veh_types = ['idm'] * len(self.idm_followers) + ['av'] + ['leader']
        veh_positions = [idm['pos'] for idm in self.idm_followers] + [self.av['pos']] + [self.leader_positions[self.traj_idx]]
        veh_speeds = [idm['speed'] for idm in self.idm_followers] + [self.av['speed']] + [self.leader_speeds[self.traj_idx]]

        step_dict = {
            'time': round(self.env_step * self.time_step, 3),
            'env_step': self.env_step,
            'trajectory_index': self.traj_idx,
            'veh_types': ';'.join(veh_types),
            'veh_positions': ';'.join(map(str, veh_positions)),
            'veh_speeds': ';'.join(map(str, veh_speeds)),
        }

        for k, v in step_dict.items():
            self.emissions_data[k].append(v)

    def generate_emissions(self):
        import pandas as pd
        path = 'test.csv'
        print(f'Saving emissions at {path}')
        df = pd.DataFrame(self.emissions_data)  # {key: pd.Series(value) for key, value in dictmap.items()})
        df.to_csv(path, encoding='utf-8', index=False)

    def step(self, actions):
        # get av accel

        # additional trajectory data that will be plotted in tensorboard
        infos = {
            'test': 2,
        }

        # assert self.action_space.contains(action), f'Action {action} not in action space'
        # careful should not be rescaled when this method is called for IDM/FS baseline in callback
        if self.use_discrete and isinstance(actions, np.int64):
            action = self.action_set[actions]
        else:
            action = float(actions)
        
        # action = np.clip(action, -1, 1)
        # action *= self.max_accel if action > 0 else self.max_decel
        if self.use_fs:
            self.follower_stopper.v_des += action
            self.follower_stopper.v_des = max(self.follower_stopper.v_des, 0)
            self.follower_stopper.v_des = min(self.follower_stopper.v_des, self.max_speed)
            # TODO(eugenevinitsky) decide on the integration scheme, whether we want this to depend on current or next pos
            accel = self.follower_stopper.get_accel(self.av['speed'], self.leader_speeds[self.traj_idx],
                                                    self.leader_positions[self.traj_idx] - self.av['pos'],
                                                    self.time_step)
        else:
            accel = action
            v_safe = safe_velocity(self.av['speed'], self.leader_speeds[self.traj_idx],
                                self.leader_positions[self.traj_idx] - self.av['pos'], self.max_decel, self.time_step)
            v_next = accel * self.time_step + self.av['speed']
            if v_next > v_safe:
                accel = np.clip((v_safe - self.av['speed']) / self.time_step, -np.abs(self.max_decel), self.max_accel)

        av_headway = self.leader_positions[self.traj_idx] - self.av['pos']

        # forcibly prevent the car from getting within a headway
        time_headway = av_headway / np.maximum(self.av['speed'], 0.01)
        if time_headway < 1.5:
            accel = -np.abs(self.max_decel)
        self.av['last_accel'] = accel

        for i, car in enumerate([self.av] + self.idm_followers):
            curr_consumption = self.energy_model.get_instantaneous_fuel_consumption(car['last_accel'], car['speed'], grade=0)
            self.energy_consumption[i] += curr_consumption
            infos['energy_consumption_{}'.format(i)] = curr_consumption
            infos['speed_{}'.format(i)] = car['speed']

        # compute idms accels
        for i, idm in enumerate(self.idm_followers):
            if i == 0:
                # idm right behind av
                leader_speed = self.av['speed']
                headway = self.av['pos'] - idm['pos']
            else:
                leader_speed = self.idm_followers[i - 1]['speed']
                headway = self.idm_followers[i - 1]['pos'] - idm['pos']
            assert(headway > 0)
            idm['last_accel'] = self.idm_controller.get_accel(idm['speed'], leader_speed, headway, self.time_step)
        
        # step cars
        for car in [self.av] + self.idm_followers:
            car['speed'] += car['last_accel'] * self.time_step
            car['speed'] = min(max(car['speed'], self.min_speed), self.max_speed)
            car['pos'] += car['speed'] * self.time_step

        # compute reward/done
        av_headway = self.leader_positions[self.traj_idx] - self.av['pos']
        done = False

        self.env_step += 1
        self.traj_idx += 1

        reward = 0
        if av_headway <= 0:
            # crash
            reward -= 50
            done = True

        if self.whole_trajectory:
            if self.traj_idx >= len(self.leader_positions) - 1:
                done = True
        else:
            if self.env_step % int(self.horizon) == 0:
                done = True

        if self.emissions:
            self.step_emissions()
            if done:
                self.generate_emissions()

        # reward -= (action ** 2) * 0.5
        # we have travelled at least 0.9 times as far as the lead car did
        leader_pos_change = self.leader_positions[self.traj_idx] - self.init_leader_pos
        reward = 0
        if (self.env_step % int(self.horizon) == 0 and
                self.av['pos'] - self.init_pos[0] > (self.closing_gap * leader_pos_change)):
            reward += np.mean([((car['pos'] - self.init_pos[i]) / 1609.34) / (
                        np.maximum(self.energy_consumption[i], 10.0) / 3600 + 1e-6)
                           for i, car in enumerate([self.av] + self.idm_followers)])
            reward /= self.time_step
            # reward -= 0.002 * (self.accumulated_headway)
            self.env_step = 0
            self.energy_consumption = [0 for _ in range(len(self.idm_followers) + 1)]
            self.init_pos = [car['pos'] for car in [self.av] + self.idm_followers]

        # reward -= (action ** 2) * 0.5
        returned_state = self.get_state()
        if self.extra_obs:
            vec = np.zeros(int(self.observation_space.low.shape[0] / 2))
            # do the feature engineering: is the episode over, have we satisfied the criterion
            vec[0] = ((self.env_step / self.horizon) >= 1)
            # vec[1] = (self.av['pos'] - self.init_pos) / (0.95 * leader_pos_change) - 1
            vec[1] = self.av['pos'] / 1000.0
            vec[2] = leader_pos_change / 1000.0
            returned_state = np.concatenate((returned_state, vec))
        return returned_state, reward, done, infos