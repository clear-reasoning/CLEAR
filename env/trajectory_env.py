import gym
from gym.spaces import Discrete, Box
import numpy as np

from data_loader import DataLoader
from env.simulation import Simulation

from env.accel_controllers import * # TMP
from env.energy_models import PFMMidsizeSedan  # TMP


# env params that will be used except for params explicitely set in the command-line arguments
DEFAULT_ENV_CONFIG = {
    'horizon': 1000,
    'max_accel': 1.5,
    'max_decel': 3.0,
    'min_speed': 0.0,
    'max_speed': 40.0,
    'min_headway': 7.0,
    'max_headway': 120.0,
    'whole_trajectory': False,
    'discrete': False,
    'num_actions': 7,
    'use_fs': False,
    # extra observations for the value function
    'augment_vf': True,
    # if we get closer then this time headway we are forced to break with maximum decel
    'minimal_time_headway': 1.0,
    # if false, we only include the AVs mpg in the calculation
    'include_idm_mpg': False,
    'num_idm_cars': 5,
    'num_concat_states': 1,
    'num_steps_per_sim': 1,
    # scales to normalize the action space
    'speed_scale': 40,  # for vehicle speeds or desired speeds
    'distance_scale': 100,  # for headways
}


class TrajectoryEnv(gym.Env):
    def __init__(self, config):
        super().__init__()

        # extract params from config
        self.config = config
        for k, v in self.config.items():
            setattr(self, k, v)

        # instantiate generator of dataset trajectories
        self.data_loader = DataLoader()
        if self.whole_trajectory:
            self.trajectories = self.data_loader.get_all_trajectories()
        else:
            self.trajectories = self.data_loader.get_trajectories(chunk_size=self.horizon * self.num_steps_per_sim)

        # create simulation
        self.create_simulation()

        # define action space
        if self.discrete:
            self.action_space = Discrete(self.num_actions)
            self.action_set = np.linspace(-1, 1, self.num_actions)
        else:
            self.action_space = Box(low=-3.0, high=1.5, shape=(1,), dtype=np.float32)

        # get number of states
        n_states = len(self.get_base_state())
        n_additional_vf_states = len(self.get_base_additional_vf_state())
        if self.augment_vf:
            n_additional_vf_states = 0
        assert (n_additional_vf_states <= n_states)

        # create buffer to concatenate past states
        n_states *= self.num_concat_states
        self.past_states = np.zeros(n_states)

        # define observation space
        n_obs = n_states
        if self.augment_vf:
            n_obs *= 2  # additional room for vf states
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

        # TMP
        self.idm_controller = IDMController(a=self.max_accel, b=self.max_decel, noise=0.5)
        self.follower_stopper = TimeHeadwayFollowerStopper(max_accel=self.max_accel, max_deaccel=self.max_decel)
        self.energy_model = PFMMidsizeSedan()


    def get_base_state(self):
        state = {
            'speed': self.av['speed'] / self.speed_scale,
            'leader_speed': self.leader_speeds[self.traj_idx] / self.speed_scale,
            'headway': (self.leader_positions[self.traj_idx] - self.av['pos']) / self.distance_scale,
        }
        
        if self.use_fs:
            state['vdes'] = self.follower_stopper.v_des / self.speed_scale

        return state

    def get_base_additional_vf_state(self):
        vf_state = {
            'time': self.env_step / self.horizon,
            'mpg_top': np.mean([(car['pos'] - self.init_pos[i]) / 1609.34
                             for i, car in enumerate(self.car_list)]),
            'mpg_bot': np.mean([np.maximum(self.energy_consumption[i], 10.0) / 3600 + 1e-6
                             for i, car in enumerate(self.car_list)]),
        }

        return vf_state

    def get_state(self, _store_state=False):
        if _store_state:
            # preprend new state to the saved past states
            state = list(self.get_base_state().values())
            self.past_states = np.roll(self.past_states, len(state))
            self.past_states[:len(state)] = state

        if self.augment_vf:
            additional_vf_state = list(self.get_base_additional_vf_state().values())
            additional_vf_state += [0] * (len(self.past_states) - len(additional_vf_state))
            state = np.concatenate((self.past_states, additional_vf_state))
        else:
            state = self.past_states

        return state
    
    def create_simulation(self):
        # start at random time in trajectory

        # total_length = len(self.leader_positions)
        # if self.whole_trajectory:
        #     self.traj_idx = 0
        #     self.horizon = total_length - 1
        # else:
        #     self.traj_idx = randint(0, total_length - self.horizon - 1)
        traj = next(self.trajectories)
        self.leader_positions, self.leader_speeds = traj['positions'], traj['velocities']
        self.traj_idx = 0
        self.env_step = 0
        self.time_step = traj['timestep']

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
            'pos': self.av['pos'] - 20 * (i + 1),#- max(2 * self.av['speed'] + 20 * (-0.5 + np.random.uniform(low=0, high=1)), 20) * (i + 1),
            'speed': self.av['speed'],
            'last_accel': -1,
        } for i in range(5)]

        if self.include_idm_mpg:
            self.energy_consumption = [0 for _ in range(len(self.idm_followers) + 1)]
            self.init_pos = [car['pos'] for car in [self.av] + self.idm_followers]
        else:
            self.energy_consumption = [0 for _ in range(1)]
            self.init_pos = [car['pos'] for car in [self.av]]
            
        if self.include_idm_mpg:
            self.car_list = [self.av] + self.idm_followers
        else:
            self.car_list = [self.av]

    def reset(self):
        self.create_simulation()
        return self.get_state(_store_state=True)

    def step(self, actions):
        # get av accel

        # additional trajectory data that will be plotted in tensorboard
        infos = {
            'test': 2,
        }

        # assert self.action_space.contains(action), f'Action {action} not in action space'
        # careful should not be rescaled when this method is called for IDM/FS baseline in callback
        if self.discrete and isinstance(actions, np.int64):
            action = self.action_set[actions]
        else:
            action = float(actions)
        
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

        self.av['last_accel'] = accel

        if self.include_idm_mpg:
            self.car_list = [self.av] + self.idm_followers
        else:
            self.car_list = [self.av]
        for i, car in enumerate(self.car_list):
            curr_consumption = self.energy_model.get_instantaneous_fuel_consumption(car['last_accel'], car['speed'], grade=0)
            self.energy_consumption[i] += curr_consumption
            infos['energy_consumption_{}'.format(i)] = curr_consumption
            infos['speed_car_{}'.format(i)] = car['speed']

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
        self.traj_idx += self.num_steps_per_sim

        reward = 0
        if av_headway <= 0:
            # crash
            reward -= 50
            done = True

        if av_headway <= self.min_headway:
            reward -= 2

        # forcibly prevent the car from getting within a headway
        time_headway = av_headway / np.maximum(self.av['speed'], 0.01)
        if time_headway < self.minimal_time_headway:
            reward -= 2.0

        if av_headway > self.max_headway:
            reward -= 2.0


        state = self.get_state(_store_state=True)

        if self.whole_trajectory:
            if self.traj_idx >= len(self.leader_positions) - 1:
                done = True
        else:
            if (self.env_step + 1) % int(self.horizon) == 0:
                done = True

        if ((self.env_step + 1) % int(self.horizon) == 0) or self.traj_idx >= len(self.leader_positions) - 1:
            avg_mpg = np.mean([((car['pos'] - self.init_pos[i]) / 1609.34) / (
                        np.maximum(self.energy_consumption[i], 1.0) / 3600 + 1e-6)
                           for i, car in enumerate(self.car_list)])
            avg_mpg /= self.time_step
            reward += avg_mpg
            # reward -= 0.002 * (self.accumulated_headway)
            self.energy_consumption = [0 for _ in range(len(self.car_list))]
            self.init_pos = [car['pos'] for car in self.car_list]
            infos['avg_horizon_mpg'] = avg_mpg



        return state, reward, done, infos