"""Trajectory environment."""
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from datetime import timezone
from pathlib import Path

import gym
import numpy as np
import pandas as pd
from gym.spaces import Discrete, Box, MultiDiscrete

from trajectory.data_loader import DataLoader
from trajectory.env.megacontroller import MegaController
from trajectory.env.simulation import Simulation
from trajectory.env.utils import get_first_element
from trajectory.visualize.time_space_diagram import plot_time_space_diagram

# env params that will be used except for params explicitly set in the command-line arguments
MPH_TO_MS = 0.44704
DEFAULT_ENV_CONFIG = {
    'horizon': 1000,
    'min_headway': 10.0,
    'max_headway': 120.0,
    'max_time_headway': 0.0,
    'whole_trajectory': False,
    'discrete': False,
    'num_actions': 7,
    'min_accel': -3.0,
    'max_accel': 1.5,
    'use_fs': False,
    # extra observations for the value function
    'augment_vf': True,
    # if we get closer then this time headway we are forced to break with maximum decel
    'minimal_time_headway': 1.0,
    'minimal_time_to_collision': 6.0,
    'headway_penalty': 0.0,
    'min_headway_penalty_gap': 10.0,
    'min_headway_penalty_speed': 1.0,
    'accel_penalty': 0.2,
    'intervention_penalty': 0,
    'penalize_energy': 1,
    # if false, we only include the AVs mpg in the calculation
    'include_idm_mpg': False,
    'num_concat_states': 1,
    # number of larger interval concat states
    'num_concat_states_large': 0,
    # number of leader speed memory in state
    'num_leader_speed_memory': 0,
    # platoon (combination of avs and humans following the leader car)
    'platoon': 'av human*5',
    # controller to use for the AV (available options: rl, idm, fs)
    'av_controller': 'rl',
    'av_kwargs': '{}',
    # human controller & params
    'human_controller': 'idm',
    'human_kwargs': '{}',
    # set to use one specific trajectory
    'fixed_traj_path': None,
    # set to use one specific set of trajectories
    'traj_dir': None,
    # Whether to gradually add trajectories to training
    'traj_curriculum': 0,
    # which set of trajectories to draw from for curriculum
    'traj_curriculum_dir': None,
    # frequency of introducing curriculum trajectories
    'traj_curriculum_freq': 100,
    # enable lane changing
    'lane_changing': True,
    # set probability of lane changing model enabled in a given rollout
    'lc_prob': 0,
    # set time step after which lane changing is enabled randomly
    'lc_curriculum_steps': 0,
    # enable road grade in energy function
    'road_grade': '',
    # set size of platoon for observation
    'platoon_size': 5,
    # whether to add downstream speeds to state / use them
    'downstream': 0,
    # how many segments of downstream info to add to state
    'downstream_num_segments': 10,
    # whether to include INRIX data for local segment (only applies if downstream is set)
    'include_local_segment': 0,
    # whether to include thresholds for downstream and gap-closing in state
    'include_thresholds': False,
    # whether inrix portion of state is included in memory (if set to 1, included)
    'inrix_mem': 1,
    # whether inrix portion of state is included in memory (if set to 1, included)
    'vf_include_chunk_idx': 0,
    # whether to add speed planner to state (if set to 1, included)
    'speed_planner': 0,
    # params for acc controller (rl_acc vehicle)
    'output_acc': False,  # If set, output acc gap and speed settings instead of accel
    'acc_num_gap_settings': 3,
    'acc_min_speed': 20 * MPH_TO_MS,
    'acc_max_speed': 80 * MPH_TO_MS,
    'acc_speed_step': 1 * MPH_TO_MS,
    # whether to add current speed and gap ACC settings to the state
    'acc_states': 0,
    # whether to set the neural network output as continuous (and clipped/rounded)
    # by default the output is discrete
    'acc_continuous': 0,
    # args for stripped / reduced state
    'stripped_state': 0,
    'leader_present': 0,
    'leader_present_threshold': 80,
    'leader_faster': 0,
    'dummy_states': 0,
    # past leader speeds and AV accels
    'past_vels_state': False,
    'past_accels_state': False,
}

# platoon presets that can be passed to the "platoon" env param
PLATOON_PRESETS = {
    'scenario1': 'human#sensor human*5 (human#sensor human*5 av human*5)*4 human#sensor human*5 human#sensor',
    '2avs_4%': 'av human*24 av human*24',
    '2avs_5%': 'av human*19 av human*19',
    '2avs_7%': 'av human*13 av human*13',
    '2avs_10%': 'av human*9 av human*9',
    '2avs_12.5%': 'av human*7 av human*7',
    '33avs_4%': '(av human*24)*33',
}


class TrajectoryEnv(gym.Env):
    """Trajectory Environment."""

    def __init__(self, config, _simulate=False, _verbose=True):
        super().__init__()

        # Keep track of total number of training steps
        self.step_count = 0

        # extract params from config
        self.config = dict(DEFAULT_ENV_CONFIG)
        self.config.update(config)
        for k, v in self.config.items():
            setattr(self, k, v)
        self.collect_rollout = False
        self.simulate = _simulate
        self.log_time_counter = time.time()

        if self.use_fs:
            assert self.av_controller == 'rl'
            assert not self.discrete
            self.av_controller = 'rl_fs'

        if self.output_acc:
            assert 'rl' in self.av_controller
            self.av_controller = 'rl_acc'

        # instantiate generator of dataset trajectories
        self.data_loader = DataLoader(traj_path=self.fixed_traj_path, traj_dir=self.traj_dir,
                                      curriculum_dir=self.traj_curriculum_dir)

        self.chunk_size = None if self.whole_trajectory else self.horizon
        self.trajectories = self.data_loader.get_trajectories(
            chunk_size=self.chunk_size,
            # fixed_traj_path=self.fixed_traj_path,
        )
        self.traj = None
        self.traj_idx = -1
        self.chunk_idx = -1
        
        self.past_av_speeds = [-40] * 10
        self.past_requested_speed_setting = [-40] * 10

        self.megacontroller = MegaController(output_acc=False)

        # create simulation
        self.create_simulation(self.lane_changing)

        self._verbose = _verbose
        if self._verbose:
            print('\nRunning experiment with the following platoon:', ' '.join([v.name for v in self.sim.vehicles]))
            print(f'\twith av controller {self.av_controller} (kwargs = {self.av_kwargs})')
            print(f'\twith human controller {self.human_controller} (kwargs = {self.human_kwargs})\n')
            if not self.simulate and len([v for v in self.sim.vehicles if v.kind == 'av']) > 1:
                raise ValueError('Training is only supported with 1 AV in the platoon.')

        # define action space
        a_min = self.min_accel
        a_max = self.max_accel
        if self.output_acc:
            self.acc_num_speed_settings = int((self.acc_max_speed - self.acc_min_speed)/ self.acc_speed_step + 1)
            if self.action_delta:
                self.action_space = MultiDiscrete([4, self.acc_num_gap_settings])
                self.action_mapping = {0: -5 * MPH_TO_MS, 1: -1 * MPH_TO_MS, 2: 1 * MPH_TO_MS, 3: 5 * MPH_TO_MS} # in m/s
            elif self.acc_continuous:
                self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            else:
                self.action_space = MultiDiscrete([self.acc_num_speed_settings, self.acc_num_gap_settings])
            self.gap_action_set = np.array([1, 2, 3])
            self.speed_action_set = np.arange(self.acc_min_speed,
                                              self.acc_max_speed + self.acc_speed_step,
                                              self.acc_speed_step)

        elif self.discrete:
            self.action_space = Discrete(self.num_actions)
            self.action_set = np.linspace(a_min, a_max, self.num_actions)
        else:
            if self.use_fs:
                self.action_space = Box(low=a_min, high=a_max, shape=(1,), dtype=np.float32)
            else:
                self.action_space = Box(low=a_min, high=a_max, shape=(1,), dtype=np.float32)

        # get number of states
        n_states = len(self.get_base_state())
        n_additional_vf_states = len(self.get_base_additional_vf_state())
        assert n_additional_vf_states <= n_states
        if self.augment_vf:
            n_additional_vf_states = 0
        assert (n_additional_vf_states <= n_states)

        # create buffer to concatenate past states
        assert self.num_concat_states >= 0  # we store at least the current state
        assert self.num_concat_states_large >= 0
        n_states *= (self.num_concat_states + self.num_concat_states_large)
        self.n_states = n_states
        self.past_states = {
            i: np.zeros(self.n_states)
            for i in range(len(self.avs))
        }

        # Additional state that is not included in past states (so not set to self.n_states)
        if self.downstream and not self.inrix_mem:
            n_states += len(self.get_downstream_state())

        # define observation space
        n_obs = n_states
        if self.augment_vf:
            n_obs *= 2  # additional room for vf states
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

    def get_acc_input(self, action):
        if self.acc_continuous:
            action = np.clip(action, -1.0, 1.0)
            speed_setting = int((action[0] + 1.0) * 20.0)
            if action[1] > 1.0 / 3.0:
                gap_setting = 1
            elif action[1] > -1.0 / 3.0:
                gap_setting = 2
            else:
                gap_setting = 3
        else:
            speed_setting = self.speed_action_set[action[0]]
            gap_setting = self.gap_action_set[action[1]]
        return speed_setting, gap_setting

    def get_base_state(self, av_idx=None):
        """Get base state.

        Dict of state_name: (state_value, state_normalization_scale)
        """
        av = self.avs[av_idx if av_idx is not None else 0]

        state = {'speed': (av.speed, 40.0)}
        if not self.stripped_state:
            state.update({
                'leader_speed': (av.get_leader_speed(), 40.0),
                'headway': (av.get_headway(), 100.0),
            })
        
        if self.past_vels_state:
            past_vels = self.sim.get_data(av, 'speed')[-self.past_vels_state:]
            past_vels = [0] * (self.past_vels_state - len(past_vels)) + past_vels
            state.update({
                f'past_vel_{i}': (past_vels[-i], 40.0)
                for i in range(1, self.past_vels_state + 1)
            })

        if self.past_accels_state:
            past_accels = self.sim.get_data(av, 'accel')[-self.past_accels_state:]
            past_accels = [0] * (self.past_accels_state - len(past_accels)) + past_accels
            state.update({
                f'past_accel_{i}': (past_accels[-i], 4)
                for i in range(1, self.past_accels_state + 1)
            })

        if self.leader_present:
            state.update({
                # 1 if leader within headway threshold, 0 otherwise
                'leader_present': (int(av.get_headway() < self.leader_present_threshold), 1),
            })

        if self.leader_faster:
            state.update({
                # 1 if leader is faster than the av, 0 otherwise
                'leader_faster': (int(av.get_leader_speed() > av.speed), 1),
            })

        if self.include_thresholds:
            state.update({
                'gap_closing': (self.gap_closing_threshold(av), 100.0),
                'failsafe': (av.failsafe_threshold(), 100.0)
            })

        # Add inrix data to base state if downstream set and including in memory
        if self.downstream and self.inrix_mem:
            state.update(self.get_downstream_state(av_idx))

        if self.num_leader_speed_memory:
            n_mem = self.num_leader_speed_memory

            # Get past leader speeds from simulation (all but current), and ensure that
            # past leader speed is at least as long as n_mem
            past_leader_speeds = [0] * n_mem + self.sim.get_data(av, 'leader_speed')[:-1]
            state.update({
                f'leader_speed_{i}': (past_leader_speeds[-i], 40.0)
                for i in range(1, n_mem+1)
            })

        if self.speed_planner:
            self.megacontroller.run_speed_planner(av)
            target_speed, max_headway = self.megacontroller.get_target(av)
            state.update({
                'target_speed': (target_speed, 40.0),
                'max_headway': (max_headway, 1.0),
            })
            
        for pos_delta in [200, 500, 1000]:  # ]list(range(100, 1001, 100)) + [2000]:
            target_speed_delta, _ = self.megacontroller.get_target(av, pos_delta=pos_delta)
            state.update({
                f'target_speed_{pos_delta}': (target_speed_delta, 40.0),
            })
            
        if self.acc_states:
            state.update({
                'speed_setting': (av.megacontroller.speed_setting, 40.0),
                'gap_setting': (av.megacontroller.gap_setting, 3.0),                
            })

        if self.dummy_states > 0:
            state.update({f"dummy_{i}": (0.0, 1.0) for i in range(self.dummy_states)})

        for i in range(10):
            state.update({
                f'past_av_speeds_{i}': (self.past_av_speeds[-i-1], 40.0),
                f'past_requested_speed_setting_{i}': (self.past_requested_speed_setting[-i-1], 40.0),           
            })

        return state

    def get_downstream_state(self, av_idx=0):
        av = self.avs[av_idx if av_idx is not None else 0]

        """Get downstream state."""
        state = {}
        # Get extra speed because 0th speed is the local speed
        downstream_speeds = av.get_downstream_avg_speed(k=self.downstream_num_segments + 1)
        downstream_distances = av.get_distance_to_next_segments(k=self.downstream_num_segments)

        downstream_obs = 0  # Number of non-null downstream datapoints in tse info
        local_speed = -1
        if downstream_speeds:
            local_speed = downstream_speeds[1][0]  # Extract local segment speed
            downstream_speeds = downstream_speeds[1][1:]  # Remove local segment to align speeds and distances
            downstream_obs = min(len(downstream_speeds), len(downstream_distances))

        if self.include_local_segment:
            if local_speed > -1:
                state.update({
                    "local_speed": (local_speed, 40.0),
                })
            else:
                state.update({
                    "local_speed": (-1.0, 1.0),
                })

        # for the segments that TSE info is available
        for i in range(downstream_obs):
            state.update({
                f"seg_{i}_speed": (downstream_speeds[i], 40.0),
                f"seg_{i}_dist": (downstream_distances[i], 5000.0)
            })

        # for segments where TSE info is not available
        for i in range(downstream_obs, self.downstream_num_segments):
            state.update({
                f"seg_{i}_speed": (-1.0, 1.0),
                f"seg_{i}_dist": (-1.0, 1.0)
            })
        return state

    def get_base_additional_vf_state(self, av_idx=None):
        """Get base additional vf state.

        Dict of state_name: (state_value, state_normalization_scale)
        This state will only be used if augment_vf is set
        """
        vf_state = {
            'time': (self.sim.step_counter, self.horizon),
            'avg_miles': (np.mean([self.sim.get_data(veh, 'total_miles')[-1] for veh in self.mpg_cars]), 50.0),
            'avg_gallons': (
                np.mean([self.sim.get_data(veh, 'total_gallons')[-1] + 1e-6 for veh in self.mpg_cars]), 100.0)
        }

        if self.stripped_state:
            av = self.avs[av_idx if av_idx is not None else 0]
            vf_state.update({
                'leader_speed': (av.get_leader_speed(), 40.0),
                'headway': (av.get_headway(), 100.0),
            })

        if self.vf_include_chunk_idx:
            vf_state.update({'traj_idx': (self.traj_idx, 10.0),
                             'chunk_idx': (self.chunk_idx, 10000.0)})

        av = self.avs[av_idx if av_idx is not None else 0]
        vf_state.update({
            'av_pos': (av.pos, 5000.0),
        })

        return vf_state

    def get_platoon_state(self, veh):
        """Return the platoon state of veh."""
        platoon = self.sim.get_platoon(veh, self.platoon_size)

        state = {
            'platoon_speed': np.mean([self.sim.get_data(veh, 'speed')[-1] for veh in platoon]),
            'platoon_mpg':
                np.sum([self.sim.get_data(veh, 'total_miles')[-1] for veh in platoon]) /
                np.sum([self.sim.get_data(veh, 'total_gallons')[-1] for veh in platoon])
        }
        return state

    def get_state(self, av_idx=None):
        """Get state."""
        # during training (always single-agent), this is called with av_idx=None
        # av_idx is set from simulate.py when evaluating with several AVs
        if av_idx is None:
            av_idx = 0

        # self.past_states[av_idx] is an array of size len(state) * (self.num_concat_states + self.num_concat_states_large)
        # self.past_states[av_idx][0:len(state)*self.num_concat_states] stores past states at a 0.1s interval (short-term)
        # self.past_states[av_idx][len(state)*self.num_concat_states:] stores past states at a 1s interval (long-term)

        # get current state
        state = [value / scale for value, scale in self.get_base_state(av_idx=av_idx).values()]

        # roll short-term memory and preprend new state
        index = self.num_concat_states * len(state)
        self.past_states[av_idx][:index] = np.roll(self.past_states[av_idx][:index], len(state))
        self.past_states[av_idx][:len(state)] = state

        if self.num_concat_states_large > 0:
            # roll long-term memory and preprend new state every second
            if round(self.sim.time_counter, 1) % 1 == 0:
                self.past_states[av_idx][index:] = np.roll(self.past_states[av_idx][index:], len(state))
                self.past_states[av_idx][index: index + len(state)] = state

        # use past states (including current state) as state
        state = self.past_states[av_idx]

        # If not storing inrix data in memory, add after concatenating states
        if self.downstream and not self.inrix_mem:
            downstream_state = [value / scale for value, scale in self.get_downstream_state(av_idx).values()]
            state = np.concatenate((state, downstream_state))

        if self.augment_vf:
            # add value function augmentation to states
            additional_vf_state = [value / scale for value, scale in self.get_base_additional_vf_state().values()]
            # vf additional state should be the same size as base policy state
            # because policy network keeps the first half of the whole state (including augmentation)
            additional_vf_state += [0] * (len(state) - len(additional_vf_state))
            state = np.concatenate((state, additional_vf_state))

        return state

    def reward_function(self, av, action):
        # reward should be positive, otherwise controller would learn to crash

        reward = 1
        # reward = 0

        # crash penalty
        if av.get_headway() < 0:
            reward -= 50

        # penalize instant energy consumption for the AV or AV + platoon
        energy_reward = 0
        if self.penalize_energy:
            energy_reward = -np.mean([max(self.sim.get_data(veh, 'instant_energy_consumption')[-1], 0)
                                      for veh in self.mpg_cars]) / 10.0
            reward += energy_reward

        # penalize acceleration amplitude
        accel_reward = -self.accel_penalty * (action ** 2)
        reward += accel_reward

        # penalize use of interventions
        gap_closing = av.get_headway() > self.gap_closing_threshold(av)
        failsafe = av.get_headway() < av.failsafe_threshold()

        intervention_reward = 0
        if gap_closing or failsafe:
            intervention_reward = -self.accel_penalty * self.intervention_penalty
            reward += intervention_reward

        # penalize large headways
        headway_reward = 0
        if av.get_headway() > self.min_headway_penalty_gap and av.speed > self.min_headway_penalty_speed:
            headway_reward = -self.headway_penalty * av.get_time_headway()
            reward += headway_reward
    
        # speed planner and curr speed diff
        speed_diff_reward = 0

        if self.speed_planner:
            speed_diff_reward = 0
            target_speed, _ = self.megacontroller.get_target(av)
            speed_diff_reward = -self.speed_diff_reward_weight * (target_speed - av.speed)**2
            reward += speed_diff_reward

        return reward, energy_reward, accel_reward, intervention_reward, headway_reward, speed_diff_reward

    def gap_closing_threshold(self, av):
        return max(self.max_headway, self.max_time_headway * av.speed)

    def create_simulation(self, lc):
        """Create simulation."""
        # collect the next trajectory
        (self.traj_idx, self.chunk_idx), self.traj = next(self.trajectories)
        self.horizon = len(self.traj['positions'])

        # create a simulation object
        self.time_step = self.traj['timestep']

        downstream_path = os.path.dirname(self.traj["path"])
        if self.downstream and not os.path.exists(os.path.join(downstream_path, "speed.csv")):
            raise ValueError("No inrix data available in {}".format(downstream_path))

        self.sim = Simulation(
            timestep=self.time_step,
            enable_lane_changing=lc,
            road_grade=self.road_grade,
            downstream_path=downstream_path
        )

        # populate simulation with a trajectory leader
        self.sim.add_vehicle(
            controller='trajectory',
            kind='leader',
            trajectory=zip(
                self.traj['positions'],
                self.traj['velocities'],
                self.traj['accelerations']))

        # parse platoons
        if self.platoon in PLATOON_PRESETS:
            print(f'Setting scenario preset "{self.platoon}"')
            self.platoon = PLATOON_PRESETS[self.platoon]

        # replace (subplatoon)*n into subplatoon ... subplatoon (n times)
        def replace1(match):
            return ' '.join([match.group(1)] * int(match.group(2)))

        self.platoon = re.sub(r'\(([a-z0-9\s\*\#]+)\)\*([0-9]+)', replace1, self.platoon)
        # parse veh#tag1...#tagk*n into (veh, [tag1, ..., tagk], n)
        self.platoon_lst = re.findall(r'([a-z]+)((?:\#[a-z]+)*)(?:\*?([0-9]+))?', self.platoon)
        # spawn vehicles
        self.avs = []
        self.humans = []
        for vtype, vtags, vcount in self.platoon_lst:
            for _ in range(int(vcount) if vcount else 1):
                tags = vtags.split('#')[1:]
                if vtype == 'av':
                    self.avs.append(
                        self.sim.add_vehicle(controller=self.av_controller, kind='av',
                                             tags=tags, gap=-1, **eval(self.av_kwargs),
                                             default_time_headway=3.0, stripped_state=self.stripped_state,
                                             no_acc_failsafe=self.no_acc_failsafe, no_acc_gap_closing=self.no_acc_gap_closing)
                    )
                elif vtype == 'human':
                    self.humans.append(
                        self.sim.add_vehicle(controller=self.human_controller, kind='human',
                                             tags=tags, gap=-1, **eval(self.human_kwargs))
                    )
                else:
                    raise ValueError(f'Unknown vehicle type: {vtype}. Allowed types are "human" and "av".')

        # define which vehicles are used for the MPG reward
        self.mpg_cars = self.avs + (self.humans if self.include_idm_mpg else [])

        # initialize one data collection step
        self.sim.collect_data()

    def reset(self):
        """Reset."""

        # Create simulation with lc enabled with a probability of lc_prob (if lane changing is disabled overall)
        lc = self.lane_changing
        if not self.lane_changing and not self.simulate and self.step_count > self.lc_curriculum_steps:
            lc = np.random.rand() < self.lc_prob
        self.create_simulation(lc)

        # reset memory
        self.past_states = {
            i: np.zeros(self.n_states)
            for i in range(len(self.avs))
        }
        return self.get_state() if not self.simulate else None  # don't stack memory here during eval

    def step(self, actions):
        """Step forward."""
        # additional trajectory data that will be plotted in tensorboard
        metrics = {}

        # apply action to AV
        accel = None
        if self.av_controller == 'rl':
            if type(actions) not in [list, np.ndarray]:
                actions = [actions]
            for av, action in zip(self.avs, actions):
                accel = self.action_set[action] if self.discrete else float(action)
                metrics['rl_controller_accel'] = accel
                accel = av.set_accel(accel, large_gap_threshold=self.gap_closing_threshold(av))
                metrics['rl_processed_accel'] = accel
        elif self.av_controller == 'rl_fs':
            # RL with FS wrapper
            if type(actions) not in [list, np.ndarray]:
                actions = [actions]
            for av, action in zip(self.avs, actions):
                vdes_command = av.speed + float(action) * self.time_step
                metrics['rl_accel'] = float(action)
                metrics['vdes_command'] = vdes_command
                metrics['vdes_delta'] = float(action) * self.time_step
                av.set_vdes(vdes_command)  # set v_des = v_av + accel * dt
        elif self.av_controller == 'rl_acc':
            if type(actions) not in [list, np.ndarray]:
                actions = [actions]
            elif type(actions) is np.ndarray and len(actions.shape) < 2:
                actions = np.array([actions])

            for av, action in zip(self.avs, actions):
                # # <================= copy and pasting from megacontroller.py for sanity check that something isn't 
                # # straight up not using action so it better fucking work
                # target_speed, max_headway = self.megacontroller.get_target(av)
                # lead_vel = self.avs[0].get_leader_speed()
                # if target_speed < lead_vel:
                #     speed_setting = target_speed * 0.6 + lead_vel * 0.4
                # else:
                #     speed_setting = target_speed
                # if max_headway:
                #     gap_setting = 1
                # else:
                #     gap_setting = 3
                
                # speed_setting = round(speed_setting / MPH_TO_MS) * MPH_TO_MS
                # speed_setting = max(speed_setting, MPH_TO_MS * 20)
                # # =================>

                speed_setting, gap_setting = self.get_acc_input(action)
                if self.action_delta:
                    delta = self.action_mapping[action[0]]
                    if curr_speed := av.get_speed_setting():
                        speed_setting = curr_speed + delta
                elif self.jonny_style:
                    lead_vel = self.avs[0].get_leader_speed()
                    # self.megacontroller.run_speed_planner(av)
                    target_speed, _ = self.megacontroller.get_target(av)

                    if target_speed < lead_vel:
                        speed_setting = target_speed * 0.6 + lead_vel * 0.4
                    else:
                        speed_setting = target_speed
                    # Apply delta
                    delta = self.action_mapping[action[0]]
                    speed_setting += delta
                
                self.past_requested_speed_setting.append(speed_setting)
                self.past_av_speeds.append(av.speed)

                av.set_speed_setting(speed_setting)
                av.set_gap_setting(gap_setting)
                metrics['rl_acc_speed_setting'] = speed_setting
                metrics['rl_acc_gap_setting'] = gap_setting
                accel = av.set_acc(large_gap_threshold=self.gap_closing_threshold(av))
                metrics['rl_controller_accel'] = accel
                metrics['rl_processed_accel'] = accel

        # compute reward, store reward components for rollout dict
        reward, energy_reward, accel_reward, intervention_reward, headway_reward, speed_diff_reward \
            = self.reward_function(av=self.avs[0], action=accel) if accel is not None else (0, 0, 0, 0, 0)

        # print crashes
        crash = False
        crashes = [av.get_headway() <= 0 for av in self.avs]
        for i, crashed in enumerate(crashes):
            if crashed:
                print(f'Crash {i}')
                crash = True

        metrics['crash'] = int(crash)

        # execute one simulation step
        end_of_horizon = not self.sim.step(self)

        # print progress every 5s if running from simulate.py
        if self.simulate and self._verbose:
            if end_of_horizon or time.time() - self.log_time_counter > 5.0:
                steps, max_steps = self.sim.step_counter, self.traj['size']
                print(f'Progress: {round(steps / max_steps * 100, 1)}% ({steps}/{max_steps} env steps)')
                self.log_time_counter = time.time()

        # get next state & done
        next_state = self.get_state() if not self.simulate else None  # don't stack memory here during eval
        done = (end_of_horizon or crash)
        infos = {'metrics': metrics}

        if self.collect_rollout:
            base_state = self.get_base_state()
            self.collected_rollout['actions'].append(get_first_element(actions))
            if self.output_acc:
                speed_setting, gap_setting = self.get_acc_input(actions[0])
                self.collected_rollout['speed_actions'].append(speed_setting)
                self.collected_rollout['gap_actions'].append(gap_setting)
                self.collected_rollout['speed_setting'].append(self.avs[0].megacontroller.speed_setting)
                self.collected_rollout['gap_setting'].append(self.avs[0].megacontroller.gap_setting)
            if self.speed_planner:
                target_speed = float(base_state['target_speed'][0])
                self.collected_rollout['target_speed'].append(target_speed)

            self.collected_rollout['actions'].append(get_first_element(actions))

            self.collected_rollout['base_states'].append(base_state)
            self.collected_rollout['base_states_vf'].append(self.get_base_additional_vf_state())
            self.collected_rollout['rewards'].append(reward)
            self.collected_rollout['energy_rewards'].append(energy_reward)
            self.collected_rollout['accel_rewards'].append(accel_reward)
            self.collected_rollout['intervention_rewards'].append(intervention_reward)
            self.collected_rollout['headway_rewards'].append(headway_reward)
            self.collected_rollout['speed_diff_reward'].append(speed_diff_reward)
            self.collected_rollout['dones'].append(done)
            self.collected_rollout['infos'].append(infos)
            self.collected_rollout['system'].append({
                'avg_mpg':
                    np.sum([self.sim.get_data(veh, 'total_miles')[-1] for veh in self.sim.vehicles]) /
                    np.sum([self.sim.get_data(veh, 'total_gallons')[-1] for veh in self.sim.vehicles]),
                'speed':
                    np.mean([self.sim.get_data(veh, 'speed')[-1] for veh in self.sim.vehicles])})
            self.collected_rollout['lane_changes'].append({
                'n_cutins': self.sim.n_cutins,
                'n_cutouts': self.sim.n_cutouts,
                'n_vehicles': self.sim.n_vehicles[-1],
            })
            for i, av in enumerate(self.avs):
                self.collected_rollout[f'platoon_{i}'].append(self.get_platoon_state(av))

        # Track total number of steps
        self.step_count += 1

        # Update curriculum if applicable
        if self.traj_curriculum and self.step_count % self.traj_curriculum_freq == 0:
            self.data_loader.update_curriculum()
            self.trajectories = self.data_loader.get_trajectories(chunk_size=self.chunk_size)

        return next_state, reward, done, infos

    def start_collecting_rollout(self):
        """Start collecting rollout."""
        self.collected_rollout = defaultdict(list)
        self.collect_rollout = True

    def stop_collecting_rollout(self):
        """Stop collecting rollout."""
        self.collot_rollout = False

    def get_collected_rollout(self):
        """Get collected rollout."""
        return self.collected_rollout

    def gen_emissions(self, emissions_path='emissions', upload_to_leaderboard=True, large_tsd=False,
                      additional_metadata={}):
        """Generate emissions output."""
        # create emissions dir if it doesn't exist
        if emissions_path is None:
            emissions_path = 'emissions'
        emissions_path = Path(emissions_path)
        if emissions_path.suffix == '.csv':
            dir_path = emissions_path.parent
        else:
            now = datetime.now().strftime('%d%b%y_%Hh%Mm%Ss')
            dir_path = Path(emissions_path, now)
            emissions_path = dir_path / 'emissions.csv'
        dir_path.mkdir(parents=True, exist_ok=True)
        self.emissions_path = emissions_path
        self.dir_path = dir_path

        # generate emissions dict
        self.emissions = defaultdict(list)
        for veh in self.sim.data_by_vehicle.keys():
            for k, v in self.sim.data_by_vehicle[veh].items():
                self.emissions[k] += v

        # sort and save emissions file
        pd.DataFrame(self.emissions) \
            .sort_values(by=['time', 'id']) \
            .to_csv(emissions_path, index=False, float_format="%g")
        if self._verbose:
            print(f'Saved emissions file at {emissions_path}')

        if upload_to_leaderboard:
            # get date & time in appropriate format
            now = datetime.now(timezone.utc)
            date_now = now.date().isoformat()

            # create metadata file
            source_id = additional_metadata.get('source_id', 'blank')
            is_baseline = additional_metadata.get('is_baseline', 0)
            submitter_name = additional_metadata.get('author', 'blank')
            strategy = additional_metadata.get('strategy', 'blank')
            penetration_rate = additional_metadata.get('penetration_rate', 'x')
            version = additional_metadata.get('version', '4.0')
            traj_name = additional_metadata.get('traj_name', 'traj_default')
            metadata = pd.DataFrame({
                'source_id': [source_id],
                'submission_date': [date_now],
                'network': ['Single-Lane Trajectory'],
                'is_baseline': [is_baseline],
                'submitter_name': [submitter_name],
                'strategy': [strategy],
                'version': [version],
                'on_ramp': [0],
                'penetration_rate': [penetration_rate],
                'road_grade': [1],
                'is_benchmark': [0],
                'traj_name': [traj_name],
            })
            print('Metadata:', metadata)

            metadata_path = dir_path / f'{source_id}_metadata.csv'
            metadata.to_csv(metadata_path, index=False)

            # custom emissions for leaderboard
            def change_id(vid):
                if vid is None:
                    return None
                elif '#sensor' in vid:
                    return f'sensor_{vid}'
                elif 'human' in vid:
                    return f'human_{vid}'
                elif 'av' in vid:
                    return f'av_{vid}'
                elif 'trajectory' in vid:
                    return f'human_{vid}'
                else:
                    raise ValueError(f'Unknown vehicle type: {vid}')

            self.emissions['id'] = list(map(change_id, self.emissions['id']))
            self.emissions['leader_id'] = list(map(change_id, self.emissions['leader_id']))
            self.emissions['follower_id'] = list(map(change_id, self.emissions['follower_id']))
            self.emissions['x'] = self.emissions['position']
            self.emissions['y'] = [0] * len(self.emissions['x'])
            self.emissions['leader_rel_speed'] = self.emissions['speed_difference']
            self.emissions['road_grade'] = self.emissions['road_grade']
            self.emissions['edge_id'] = ['edge0'] * len(self.emissions['x'])
            self.emissions['lane_id'] = [0] * len(self.emissions['x'])
            self.emissions['distance'] = self.emissions['total_distance_traveled']
            self.emissions['relative_position'] = self.emissions['total_distance_traveled']
            self.emissions['realized_accel'] = self.emissions['realized_accel']
            self.emissions['target_accel_with_noise_with_failsafe'] = self.emissions['accel']
            self.emissions['target_accel_no_noise_no_failsafe'] = self.emissions['target_accel_no_noise_no_failsafe']
            self.emissions['target_accel_with_noise_no_failsafe'] = self.emissions[
                'target_accel_with_noise_no_failsafe']
            self.emissions['target_accel_no_noise_with_failsafe'] = self.emissions[
                'target_accel_no_noise_with_failsafe']
            self.emissions['source_id'] = [source_id] * len(self.emissions['x'])
            self.emissions['run_id'] = ['run_0'] * len(self.emissions['x'])
            self.emissions['submission_date'] = [date_now] * len(self.emissions['x'])

            emissions_df = pd.DataFrame(self.emissions).sort_values(by=['time', 'id'])
            emissions_df = emissions_df[['time',
                                         'id',
                                         'x',
                                         'y',
                                         'speed',
                                         'headway',
                                         'leader_id',
                                         'follower_id',
                                         'leader_rel_speed',
                                         'target_accel_with_noise_with_failsafe',
                                         'target_accel_no_noise_no_failsafe',
                                         'target_accel_with_noise_no_failsafe',
                                         'target_accel_no_noise_with_failsafe',
                                         'realized_accel',
                                         'road_grade',
                                         'edge_id',
                                         'lane_id',
                                         'distance',
                                         'relative_position',
                                         'source_id',
                                         'run_id',
                                         'submission_date']]
            leaderboard_emissions_path = dir_path / f'{source_id}_emissions_leaderboard.csv'
            emissions_df.to_csv(leaderboard_emissions_path, index=False)

            # platoon_mpg_path = dir_path / 'platoon_mpg.png'
            # print(f'Generating platoon MPG plot at {platoon_mpg_path}')
            # plot_platoon_mpg(emissions_path, save_path=platoon_mpg_path)

            tsd_dir_path = Path(f'/home/circles/sdb/tsd/{strategy}/')
            tsd_dir_path.mkdir(parents=True, exist_ok=True)
            if large_tsd:
                if 'wo LC' in version:
                    tsd_path = tsd_dir_path / 'large_tsd.png'
                else:
                    tsd_path = tsd_dir_path / 'large_tsd_lc.png'
            else:
                tsd_path = tsd_dir_path / f'{source_id}.png'
            print(f'Generating time-space diagram plot at {tsd_path}')
            plot_time_space_diagram(emissions_path, save_path=tsd_path)

            print()

            os.remove(leaderboard_emissions_path)
            os.remove(emissions_path)

        return emissions_path
