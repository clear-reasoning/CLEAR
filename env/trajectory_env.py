from random import randint
import gym
from gym.spaces import Discrete, Box
import numpy as np

from env.data_loader import load_data
from env.accel_controllers import IDMController, TimeHeadwayFollowerStopper
from env.energy_models import PFMMidsizeSedan
from env.failsafes import safe_velocity

DISTANCE_SCALE = 100
SPEED_SCALE = 50


class TrajectoryEnv(gym.Env):
    def __init__(self, config):
        self.max_accel = config.get('max_accel', 1.5)
        self.max_decel = config.get('max_decel', 3.0)
        self.horizon = config.get('horizon', 1000)

        self.min_speed = config.get('min_speed', 0)
        self.max_speed = config.get('max_speed', 40)
        self.use_fs = config.get('use_fs')

        self.max_headway = config.get('max_headway', 80)  # TODO maybe do both max time headway for high speeds and space headway for low speeds

        self.whole_trajectory = config.get('whole_trajectory', False)

        self.time_step, self.leader_positions, self.leader_speeds = load_data()
        # for now get positions from velocities to ignore in-lane-changes
        self.leader_positions = [self.leader_positions[0]]
        for vel in self.leader_speeds[:-1]:
            self.leader_positions.append(self.leader_positions[-1] + vel * self.time_step)
        assert(len(self.leader_positions) == len(self.leader_speeds))

        self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        if self.use_fs:
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        else:
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self.idm_controller = IDMController(a=self.max_accel, b=self.max_decel)
        if self.use_fs:
            self.follower_stopper = TimeHeadwayFollowerStopper(max_accel=self.max_accel, max_deaccel=self.max_decel)
        self.energy_model = PFMMidsizeSedan()

        self.reset()
    
    def get_state(self):
        speed = self.av['speed'] / SPEED_SCALE
        leader_speed = self.leader_speeds[self.traj_idx] / SPEED_SCALE
        headway = (self.leader_positions[self.traj_idx] - self.av['pos']) / DISTANCE_SCALE
        if self.use_fs:
            state = np.array([speed, leader_speed, headway, self.follower_stopper.v_des / 50.0])
        else:
            state = np.array([speed, leader_speed, headway])
        return state
    
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

    def step(self, action):
        self.env_step += 1
        self.traj_idx += 1

        # get av accel
        action = float(action)
        # action = np.clip(action, -1, 1)
        # action *= self.max_accel if action > 0 else self.max_decel
        if self.use_fs:
            self.follower_stopper.v_des += action
            self.follower_stopper.v_des = max(self.follower_stopper.v_des, 0)
            self.follower_stopper.v_des = min(self.follower_stopper.v_des, self.max_speed)
            # TODO(eugenevinitsky) decide on the integration scheme, whether we want this to depend on current or next pos
            accel = self.follower_stopper.get_accel(self.av['speed'], self.leader_speeds[self.traj_idx],
                                                    self.leader_positions[self.traj_idx] - self.av['pos'],
                                                    self)
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

        return self.get_state(), reward, done, {}