from random import randint
import gym
from gym.spaces import Discrete, Box
import numpy as np

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

        self.min_speed = config.get('min_speed', 0)
        self.max_speed = config.get('max_speed', 40)
        self.use_fs = config.get('use_fs')
        self.max_headway = config.get('max_headway', 120)

        self.whole_trajectory = config.get('whole_trajectory', False)

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
            self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        if self.use_fs:
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
            self.state_names = ['speed', 'leader_speed', 'headway', 'vdes']
            self.state_scales = [SPEED_SCALE, SPEED_SCALE, DISTANCE_SCALE, SPEED_SCALE]
        else:
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
            self.state_names = ['speed', 'leader_speed', 'headway']
            self.state_scales = [SPEED_SCALE, SPEED_SCALE, DISTANCE_SCALE]

        self.idm_controller = IDMController(a=self.max_accel, b=self.max_decel)
        if self.use_fs:
            self.follower_stopper = TimeHeadwayFollowerStopper(max_accel=self.max_accel, max_deaccel=self.max_decel)
        self.energy_model = PFMMidsizeSedan()

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
        # start at random time in trajectory
        total_length = len(self.leader_positions)
        if self.whole_trajectory:
            self.traj_idx = 0
        else:
            self.traj_idx = randint(0, total_length - self.horizon - 1)
        self.env_step = 0

        # create av behind leader
        self.av = {
            'pos': self.leader_positions[self.traj_idx] - 20, 
            'speed': self.leader_speeds[self.traj_idx],
            'last_accel': -1,
        }
        if self.use_fs:
            self.follower_stopper.v_des = self.leader_speeds[self.traj_idx]

        # create idm followers behind av
        self.idm_followers = [{
            'pos': self.av['pos'] - 20 * (i + 1),
            'speed': self.av['speed'],
            'last_accel': -1,
        } for i in range(5)]

        return self.get_state()

    def step(self, actions):
        # get av accel

        # additional trajectory data that will be plotted in tensorboard
        infos = {
            'test': 2,
        }

        # assert self.action_space.contains(action), f'Action {action} not in action space'
        # careful should not be rescaled when this method is called for IDM/FS baseline in callback
        if self.use_discrete and isinstance(actions, int):
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

        self.av['last_accel'] = accel

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
            idm['last_accel'] = self.idm_controller.get_accel(idm['speed'], leader_speed, headway)
        
        # step cars
        for car in [self.av] + self.idm_followers:
            car['speed'] += car['last_accel'] * self.time_step
            car['speed'] = min(max(car['speed'], self.min_speed), self.max_speed)
            car['pos'] += car['speed'] * self.time_step

        # compute reward/done
        av_headway = self.leader_positions[self.traj_idx] - self.av['pos']
        done = False
        reward = sum([- self.energy_model.get_instantaneous_fuel_consumption(car['last_accel'], car['speed'], grade=0)
                        for car in [self.av] + self.idm_followers]) / (1 + len(self.idm_followers))
        
        energy_consumption = self.energy_model.get_instantaneous_fuel_consumption(self.av['last_accel'], self.av['speed'], grade=0)
        # reward = - energy_consumption / 10

        infos['energy_consumption'] = energy_consumption


        # reward -= 1.0 * (np.abs(av_headway) ** 0.2)
        # reward -= 0.1 * (action ** 2)

        self.env_step += 1
        self.traj_idx += 1

        if av_headway <= 0:
            # crash
            reward -= 50
            done = True
        elif av_headway >= self.max_headway:
            # headway penalty
            reward -= 10

        if self.whole_trajectory:
            if self.traj_idx >= len(self.leader_positions) - 1:
                done = True
        else:
            if self.env_step >= self.horizon:
                done = True

        # reward -= (action ** 2) * 0.5

        return self.get_state(), reward, done, infos