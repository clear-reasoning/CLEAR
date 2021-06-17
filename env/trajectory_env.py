from collections import defaultdict
from datetime import datetime
import gym
from gym.spaces import Discrete, Box
import numpy as np
import pandas as pd
from pathlib import Path
import uuid

from data_loader import DataLoader
from env.simulation import Simulation
from env.utils import upload_to_s3


# env params that will be used except for params explicitely set in the command-line arguments
DEFAULT_ENV_CONFIG = {
    'horizon': 1000,
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
    'num_concat_states': 1,
    'num_steps_per_sim': 1,
    # controller to use for the AV (available options: rl, idm, fs)
    'av_controller': 'rl',
    'av_kwargs': '{}',
    # idm platoon
    'num_idm_cars': 5,
    'idms_kwargs': '{}',
}


class TrajectoryEnv(gym.Env):
    def __init__(self, config):
        super().__init__()

        # extract params from config
        self.config = config
        for k, v in self.config.items():
            setattr(self, k, v)
        self.collect_rollout = False

        assert (self.use_fs == False)  # TODO(nl) need an FS wrapper in the vehicle class

        # instantiate generator of dataset trajectories
        self.data_loader = DataLoader()
        if self.whole_trajectory:
            self.trajectories = self.data_loader.get_all_trajectories()
        else:
            # 1 more than horizon to get the next_state at the last env step
            self.trajectories = self.data_loader.get_trajectories(chunk_size=(self.horizon + 1) * self.num_steps_per_sim)

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


    def get_base_state(self):
        """Dict of state_name: (state_value, state_normalization_scale)"""
        state = {
            'speed': (self.av.speed, 40.0),
            'leader_speed': (self.av.get_leader_speed(), 40.0),
            'headway': (self.av.get_headway(), 100.0),
        }
        
        if self.use_fs:
            state['vdes'] = (self.follower_stopper.v_des, 40.0)

        return state

    def get_base_additional_vf_state(self):
        """Dict of state_name: (state_value, state_normalization_scale)
        This state will only be used if augment_vf is set"""
        vf_state = {
            'time': (self.sim.step_counter, self.horizon),
            'avg_miles': (np.mean([self.sim.get_data(veh, 'total_miles')[-1] for veh in self.mpg_cars]), 50.0),
            'avg_gallons': (np.mean([self.sim.get_data(veh, 'total_gallons')[-1] + 1e-6 for veh in self.mpg_cars]), 100.0),
        }

        return vf_state

    def get_state(self, _store_state=False):
        if _store_state:
            # preprend new state to the saved past states
            state = [value / scale for value, scale in self.get_base_state().values()]
            self.past_states = np.roll(self.past_states, len(state))
            self.past_states[:len(state)] = state

        if self.augment_vf:
            additional_vf_state = [value / scale for value, scale in self.get_base_additional_vf_state().values()]
            additional_vf_state += [0] * (len(self.past_states) - len(additional_vf_state))
            state = np.concatenate((self.past_states, additional_vf_state))
        else:
            state = self.past_states

        return state
    
    def create_simulation(self):
        # collect the next trajectory
        self.traj = next(self.trajectories)

        # create a simulation object
        self.time_step = self.traj['timestep']
        self.sim = Simulation(timestep=self.time_step)

        # populate simulation with a trajectoy leader
        self.sim.add_vehicle(controller='trajectory', kind='leader',
            trajectory=zip(self.traj['positions'], self.traj['velocities'], self.traj['accelerations']))
        # an AV
        av_initial_gap = max(2.1 * self.traj['velocities'][0], 20)
        self.av = self.sim.add_vehicle(controller=self.av_controller, kind='av', gap=av_initial_gap,
                                       **eval(self.av_kwargs))
        # and a platoon of IDMs
        self.idm_platoon = [self.sim.add_vehicle(controller='idm', kind='platoon', gap=20, **eval(self.idms_kwargs))
                            for _ in range(self.num_idm_cars)]

        # define which vehicles are used for the MPG reward
        self.mpg_cars = [self.av] + (self.idm_platoon if self.include_idm_mpg else [])

        # initialize one data collection step
        self.sim.collect_data()

    def reset(self):
        self.create_simulation()
        return self.get_state(_store_state=True)

    def step(self, action):
        # additional trajectory data that will be plotted in tensorboard
        metrics = {}

        # apply acceleration action to AV
        if self.av.controller == 'rl':
            accel = self.action_set[action] if self.discrete else float(action)
            metrics['rl_accel_before_failsafe'] = accel
            accel = self.av.set_accel(accel)  # returns accel with failsafes applied
            metrics['rl_accel_after_failsafe'] = accel

        # execute one simulation step
        end_of_horizon = not self.sim.step()

        # compute reward & done
        h = self.av.get_headway()
        th = self.av.get_time_headway()

        reward = 0

        # prevent crashes
        crash = (h <= 0)
        if crash:  
            reward -= 50.0

        # forcibly prevent the car from getting too small or large headways
        headway_penalties = {
            'low_headway_penalty': h < self.min_headway, 
            'large_headway_penalty': h > self.max_headway, 
            'low_time_headway_penalty': th < self.minimal_time_headway}
        if any(headway_penalties.values()):
            reward -= 2.0


        # give average MPG reward at the end
        if end_of_horizon:
            mpgs = [self.sim.get_data(veh, 'avg_mpg')[-1] for veh in self.mpg_cars]
            reward += np.mean(mpgs)

        # log some metrics
        metrics['crash'] = int(crash)
        for k, v in headway_penalties.items():
            metrics[k] = int(v)

        # get next state & done
        next_state = self.get_state(_store_state=True)
        done = (end_of_horizon or crash)
        infos = { 'metrics': metrics }

        if self.collect_rollout:
            self.collected_rollout['actions'].append(action)
            self.collected_rollout['base_states'].append(self.get_base_state())
            self.collected_rollout['base_states_vf'].append(self.get_base_additional_vf_state())
            self.collected_rollout['rewards'].append(reward)
            self.collected_rollout['dones'].append(done)
            self.collected_rollout['infos'].append(infos)

        return next_state, reward, done, infos

    def start_collecting_rollout(self):
        self.collected_rollout = defaultdict(list)
        self.collect_rollout = True

    def stop_collecting_rollout(self):
        self.collot_rollout = False

    def get_collected_rollout(self):
        return self.collected_rollout

    def gen_emissions(self, emissions_dir='emissions', upload_to_leaderboard=True):
        # create emissions dir if it doesn't exist
        now = datetime.now().strftime('%d%b%y_%Hh%Mm%Ss')
        path = Path(emissions_dir, now)
        path.mkdir(parents=True, exist_ok=True)
        emissions_path = path / 'emissions.csv'

        # generate emissions dict
        self.emissions = defaultdict(list)
        for veh in self.sim.vehicles:
            for k, v in self.sim.data_by_vehicle[veh.name].items():
                self.emissions[k] += v
        
        # custom columns
        self.emissions['x'] = self.emissions['position']
        self.emissions['y'] = [0] * len(self.emissions['x'])
        self.emissions['leader_rel_speed'] = self.emissions['speed_difference']
        self.emissions['road_grade'] = [0] * len(self.emissions['x'])
        self.emissions['edge_id'] = ['edge0'] * len(self.emissions['x'])
        self.emissions['lane_number'] = [0] * len(self.emissions['x'])
        self.emissions['distance'] = self.emissions['total_distance_traveled']
        self.emissions['relative_position'] = self.emissions['total_distance_traveled']
        self.emissions['realized_accel'] = self.emissions['accel']
        self.emissions['target_accel_with_noise_with_failsafe'] = self.emissions['accel']
        self.emissions['target_accel_no_noise_no_failsafe'] = self.emissions['accel']
        self.emissions['target_accel_with_noise_no_failsafe'] = self.emissions['accel']
        self.emissions['target_accel_no_noise_with_failsafe'] = self.emissions['accel']

        # sort and save emissions file
        pd.DataFrame(self.emissions) \
            .sort_values(by=['time', 'id']) \
            .to_csv(emissions_path, index=False)
        print(f'Saved emissions file at {emissions_path}')

        if upload_to_leaderboard:
            # get date & time in appropriate format
            now = datetime.now()
            date_now = now.date().isoformat()
            time_now = now.time().isoformat()

            # create metadata file
            source_id = f'trajectory_{uuid.uuid4().hex}'
            metadata = pd.DataFrame({
                'source_id': [source_id],
                'submission_time': [time_now],
                'network': ['Single-Lane Trajectoy'],
                'is_baseline': [False],
                'submitter_name': ['Nathan'],
                'strategy': ['Strategy'],
                'version': ['1.0'],
                'on_ramp': [False],
                'penetration_rate': [0],
                'road_grade': [False],
                'is_benchmark': [False],
            })
            metadata_path = path / 'metadata.csv'
            metadata.to_csv(metadata_path, index=False)

            # upload emissions and metadata to S3
            print()
            upload_to_s3(
                'circles.data.pipeline',
                f'metadata_table/date={date_now}/partition_name={source_id}_METADATA/{source_id}_METADATA.csv',
                metadata_path, log=True
            )
            upload_to_s3(
                'circles.data.pipeline',
                f'fact_vehicle_trace/date={date_now}/partition_name={source_id}/{source_id}.csv',
                emissions_path, log=True
            )

            # TODO generate time space diagram and upload to
            # upload_to_s3(
            #     'circles.data.pipeline',
            #     'time_space_diagram/date={0}/partition_name={1}/'
            #     '{1}.png'.format(cur_date, source_id),
            #     emission_files[0].replace('csv', 'png')
            # ).