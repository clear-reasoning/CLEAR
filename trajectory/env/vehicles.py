"""Vehicles."""
from gym.spaces import Discrete
from trajectory.env.accel_controllers import TimeHeadwayFollowerStopper, IDMController
from trajectory.env.failsafes import safe_velocity, safe_ttc_velocity
from trajectory.env.utils import get_first_element
import numpy as np
import bisect


class Vehicle(object):
    """Vehicle object."""

    def __init__(self, vid, controller, kind=None, tags=None,
                 pos=0, speed=0, accel=0,
                 length=5.0, max_accel=1.5, max_decel=3.0,
                 timestep=None, leader=None, follower=None,
                 **controller_args):

        self.vid = vid  # eg 3
        self.controller = controller  # eg rl, idm
        self.kind = kind  # eg av, human
        self.tags = tags  # list of strings
        self._tse = {
            "time": None,
            "segments": None,
            "avg_speed": None,
            "confidence": None,
            "cvalue": None,
        }

        self.name = f'{self.vid}_{self.controller}'  # eg 2_idm_human#metrics
        if self.kind is not None:
            self.name += f'_{self.kind}'
        if self.tags is not None:
            self.name += ''.join([f'#{tag}' for tag in tags])

        self.pos = pos
        self.speed = speed
        self.prev_speed = 0
        self.accel = accel
        self.dt = timestep
        self.length = length
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.leader = leader
        self.follower = follower

        self.accel_no_noise_no_failsafe = accel
        self.accel_with_noise_no_failsafe = accel
        self.accel_no_noise_with_failsafe = accel

        self.controller_args = controller_args

        assert (timestep is not None)

    def step(self, accel=None, ballistic=False, tse=None):
        """Step forward."""
        if accel is not None:
            self.accel = accel

        # save previous speed
        self.prev_speed = self.speed

        # clip accel
        self.accel = min(max(self.accel, -self.max_decel), self.max_accel)

        if ballistic:
            self.pos += max(self.dt * self.speed + self.dt * self.dt * self.accel / 2.0, 0)
            self.speed = max(self.speed + self.dt * self.accel, 0)
        else:
            self.speed = max(self.speed + self.dt * self.accel, 0)
            self.pos += self.dt * self.speed

        # Update stored traffic state estimates.
        self._tse = tse

        return True

    def get_headway(self):
        """Get headway."""
        if self.leader is None:
            return None
        return self.leader.pos - self.pos - self.length

    def get_time_headway(self):
        """Get time headway."""
        if self.leader is None:
            return None
        return np.inf if self.speed == 0 else self.get_headway() / self.speed

    def get_leader_speed(self):
        """Get leader speed."""
        if self.leader is None:
            return None
        return self.leader.speed

    def get_speed_difference(self):
        """Get speed difference."""
        if self.leader is None:
            return None
        return self.speed - self.leader.speed

    def get_time_to_collision(self):
        """Get time to collision."""
        if self.leader is None:
            return None
        return np.inf if self.get_speed_difference() <= 0 else self.get_headway() / self.get_speed_difference()

    def apply_failsafe(self, accel):
        """Apply failsafe."""
        # TODO hardcoded max decel to be conservative
        v_safe = safe_velocity(self.speed, self.leader.speed, self.get_headway(), self.max_decel, self.dt)
        v_next = self.speed + accel * self.dt
        if v_next > v_safe:
            safe_accel = np.clip((v_safe - self.speed) / self.dt, - np.abs(self.max_decel), self.max_accel)
        else:
            safe_accel = accel
        return safe_accel

    def get_segments(self):
        """Return the starting position of every segment whose macroscopic state is approximated."""
        return self._tse["segments"] if self._tse else None

    def get_avg_speed(self):
        """Return a tuple of traffic state estimation data.

        This tuple consists of the following elements:

        1. the time (relative to the start of the simulation) when the estimate
           was taken (in seconds)
        2. the average speed of every segment (in m/s).
        3. Confidence score — This is a simple confidence factor. 30 — high
           confidence, based on real-time data for that specific segment 20 —
           medium confidence
        4. Indicates the probability that the current probe reading represents
           the actual roadway conditions based on recent and historic trends.
           This value is presented only when the confidence score is 30.
           (0 = low probability, 100 = high probability)
        """
        return (self._tse["time"],
                self._tse["avg_speed"],
                self._tse["confidence"],
                self._tse["cvalue"])

    def get_distance_to_next_segment(self):
        """Return the distance to the next segment."""
        if self.get_segments() is None:
            return None

        index = bisect.bisect(self.get_segments(), self.pos)

        return self.get_segments()[index] - self.pos

    def get_distance_to_next_segments(self, k=1):
        """Return the distance to the next k segments."""
        if self.get_segments() is None or k < 0:
            return None

        index = bisect.bisect(self.get_segments(), self.pos)
        index = np.arange(index, min(len(self.get_segments()), index+k))
        dists = [self.get_segments()[index[i]] - self.pos for i in range(len(index))]

        return dists

    def get_distance_to_previous_segments(self, k=1):
        """Return the distance to the start of previous k segments.

        Doesn't include current segment.
        """
        if self.get_segments() is None or k < 0:
            return None

        index = bisect.bisect(self.get_segments(), self.pos) - 1
        index = np.arange(max(0, index-k), index)
        dists = [self.get_segments()[index[i]] - self.pos for i in range(len(index))]

        return dists

    def get_upstream_avg_speed(self, k=10):
        """Return traffic-state info of the k closest upstream segments.

        See the docstring for `avg_speed` to learn more about what each element
        in the tuple consists of.
        """
        if self.get_segments() is None:
            return None

        index = bisect.bisect(self.get_segments(), self.pos) - 1

        t, avg_speed, confidence, cvalue = self.get_avg_speed()

        min_indx = max(index - k, 0)
        max_indx = min(index, len(avg_speed))

        return (t,
                avg_speed[min_indx: max_indx],
                confidence[min_indx: max_indx],
                cvalue[min_indx: max_indx])

    def get_downstream_avg_speed(self, k=10):
        """Return traffic-state info of the k closest downstream segments.

        See the docstring for `avg_speed` to learn more about what each element
        in the tuple consists of.
        """
        if self.get_segments() is None:
            return None

        index = bisect.bisect(self.get_segments(), self.pos) - 1

        t, avg_speed, confidence, cvalue = self.get_avg_speed()

        min_indx = max(index, 0)
        max_indx = min(index + k, len(avg_speed))

        return (t,
                avg_speed[min_indx: max_indx],
                confidence[min_indx: max_indx],
                cvalue[min_indx: max_indx])

    def get_local_avg_speed(self, k=10):
        """Return traffic-state info within k segments from your position.

        See the docstring for `avg_speed` to learn more about what each element
        in the tuple consists of.
        """
        if self.get_segments() is None:
            return None

        index = bisect.bisect(self.get_segments(), self.pos) - 1

        t, avg_speed, confidence, cvalue = self.get_avg_speed()

        min_indx = max(index - k, 0)
        max_indx = min(index + k, len(avg_speed))

        return (t,
                avg_speed[min_indx: max_indx],
                confidence[min_indx: max_indx],
                cvalue[min_indx: max_indx])


class IDMVehicle(Vehicle):
    """IDM Vehicle."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.idm = IDMController(**self.controller_args)

    def step(self, accel=None, ballistic=False, tse=None):
        """See parent."""
        accel = self.idm.get_accel(self.speed, self.get_leader_speed(), self.get_headway(), self.dt)
        self.accel_with_noise_no_failsafe = accel
        self.accel_no_noise_no_failsafe = self.idm.get_accel_without_noise()
        self.accel_no_noise_with_failsafe = self.apply_failsafe(self.accel_no_noise_no_failsafe)
        accel = self.apply_failsafe(accel)

        return super().step(accel=accel, ballistic=True, tse=tse)


class FSVehicle(Vehicle):
    """Follower-Stopper Vehicle."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.fs = TimeHeadwayFollowerStopper(**self.controller_args)

    def step(self, accel=None, ballistic=False, tse=None):
        """See parent."""
        self.fs.v_des = self.get_leader_speed()

        accel = self.fs.get_accel(self.speed, self.get_leader_speed(), self.get_headway(), self.dt)
        self.accel_with_noise_no_failsafe = accel
        self.accel_no_noise_no_failsafe = self.fs.get_accel_without_noise()
        self.accel_no_noise_with_failsafe = self.apply_failsafe(self.accel_no_noise_no_failsafe)
        accel = self.apply_failsafe(accel)

        return super().step(accel=accel, ballistic=True, tse=tse)


class TrajectoryVehicle(Vehicle):
    """Trajectory Vehicle."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.trajectory = self.controller_args['trajectory']
        self.step()

    def step(self, accel=None, ballistic=False, tse=None):
        """See parent."""
        traj_data = next(self.trajectory, None)
        if traj_data is None:
            return False
        self.pos, self.speed, self.accel = traj_data
        self.accel_no_noise_with_failsafe = self.accel
        self.accel_with_noise_no_failsafe = self.accel
        self.accel_no_noise_no_failsafe = self.accel
        return True


class RLVehicle(Vehicle):
    """RL Vehicle."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.idm = IDMController()

    def step(self, accel=None, ballistic=False, tse=None):
        return super().step(accel=self.accel, ballistic=True, tse=tse)

    def set_accel(self, accel, large_gap_threshold=120):
        """Set acceleration."""
        # If position < 0, use IDM instead of RL accel
        if self.pos < 0:
            self.accel_with_noise_no_failsafe = self.idm.get_accel(self.speed, self.get_leader_speed(), self.get_headway(), self.dt)
            self.accel_no_noise_no_failsafe = self.idm.get_accel_without_noise()
            self.accel_no_noise_with_failsafe = IDMVehicle.apply_failsafe(self, self.accel_no_noise_no_failsafe)
            self.accel = self.accel_no_noise_with_failsafe
        else:
            # hardcoded gap closing ~(linearly increasing from 0.1 to 0.5 up to 100m)~
            if self.get_headway() >= large_gap_threshold:
                # gap_over_threshold = min(self.get_headway() - large_gap_threshold, 100.0)  # between 0 and 100
                # accel_gap_closing = 0.5 * gap_over_threshold / 100.0
                accel_gap_closing = 1.0
                # maxed with controller accel (can go faster than hardcoded)
                accel = max(accel, accel_gap_closing)

            self.accel_with_noise_no_failsafe = accel
            self.accel_no_noise_no_failsafe = accel
            self.accel = self.apply_failsafe(accel)
            self.accel_no_noise_with_failsafe = self.accel
        return self.accel

    def apply_failsafe(self, accel):
        # TODO hardcoded max decel to be conservative
        v_safe = safe_velocity(self.speed, self.leader.speed, self.get_headway(), self.max_decel, self.dt)
        v_safe = min(v_safe, safe_ttc_velocity(self.speed, self.leader.speed, self.get_headway(), self.max_decel, self.dt))
        v_next = self.speed + accel * self.dt
        if v_next > v_safe:
            safe_accel = np.clip((v_safe - self.speed) / self.dt, - np.abs(self.max_decel), self.max_accel)
        else:
            safe_accel = accel
        return safe_accel


class FSWrappedRLVehicle(Vehicle):
    """Follower-Stopper-wrapped RL Vehicle."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.fs = TimeHeadwayFollowerStopper(**self.controller_args)
        self.fs.v_des = self.speed

    def step(self, accel=None, ballistic=False, tse=None):
        """See parent."""
        accel = self.fs.get_accel(self.speed, self.get_leader_speed(), self.get_headway(), self.dt)
        self.accel_with_noise_no_failsafe = accel
        self.accel_no_noise_no_failsafe = self.fs.get_accel_without_noise()
        self.accel_no_noise_with_failsafe = self.apply_failsafe(self.accel_no_noise_no_failsafe)
        accel = self.apply_failsafe(accel)

        return super().step(accel=accel, ballistic=True, tse=tse)

    def set_vdes(self, vdes):
        """Set v-des."""
        self.fs.v_des = vdes


class AvVehicle(Vehicle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # import libs
        import json
        import re
        import importlib

        # load kwargs
        config_path = kwargs['config_path']
        cp_path = kwargs['cp_path']

        # load configs.json
        with open(config_path, 'r') as fp:
            self.config = json.load(fp)

        if self.config['env_config']['discrete']:
            try:
                a_min = self.config['env_config']['min_accel']
                a_max = self.config['env_config']['max_accel']
            except KeyError:
                # config.json does not specify a_min and/or a_max, setting to defaults for backward compatibility
                a_min = -3.0
                a_max = 1.5

            self.action_space = Discrete(self.config['env_config']['num_actions'])
            self.action_set = np.linspace(a_min, a_max, self.config['env_config']['num_actions'])

        self.max_headway = self.config['env_config']['max_headway']
        self.num_concat_states = self.config['env_config']['num_concat_states']
        self.num_concat_states_large = self.config['env_config']['num_concat_states_large']
        self.augment_vf = self.config['env_config']['augment_vf']
        self.n_base_states = len(self.get_base_state())
        n_states = self.n_base_states * (self.num_concat_states + self.num_concat_states_large) * (2 if self.augment_vf else 1)
        self.states = np.zeros(n_states)
        self.step_counter = 0
        self.idm = IDMController()  # Instantiate IDM controller for use when self.pos < 0

        # retrieve algorithm
        alg_module, alg_class = re.match("<class '(.+)\\.([a-zA-Z\\_]+)'>", self.config['algorithm']).group(1, 2)
        assert (alg_module.split('.')[0] in ['stable_baselines3', 'algos'] or alg_module.split('.')[1] == 'algos')
        algorithm = getattr(importlib.import_module(alg_module), alg_class)

        # load checkpoint into model
        self.model = algorithm.load(cp_path)

    def get_action(self, state):
        return get_first_element(self.model.predict(state, deterministic=True))

    def apply_failsafe(self, accel):
        # TODO hardcoded max decel to be conservative
        v_safe = safe_velocity(self.speed, self.leader.speed, self.get_headway(), self.max_decel, self.dt)
        v_safe = min(v_safe, safe_ttc_velocity(self.speed, self.leader.speed, self.get_headway(), self.max_decel, self.dt))
        v_next = self.speed + accel * self.dt
        if v_next > v_safe:
            safe_accel = np.clip((v_safe - self.speed) / self.dt, - np.abs(self.max_decel), self.max_accel)
        else:
            safe_accel = accel
        return safe_accel

    def get_base_state(self):
        state = [
            self.speed / 40.0,
            self.leader.speed / 40.0,
            self.get_headway() / 100.0,
        ]

        if self.config['env_config']['downstream']:
            num_segments = self.config['env_config']['downstream_num_segments']
            downstream_speeds = self.get_downstream_avg_speed(k=num_segments)
            downstream_distances = self.get_distance_to_next_segments(k=num_segments)

            downstream_obs = 0  # Number of non-null downstream datapoints in tse info
            if downstream_speeds and downstream_distances:
                downstream_speeds = downstream_speeds[1]
                downstream_obs = min(len(downstream_speeds), len(downstream_distances))

            # for the segments that TSE info is available
            for i in range(downstream_obs):
                state.append(downstream_speeds[i] / 40.0)
                state.append(downstream_distances[i] / 5000.0)

            # for segments where TSE info is not available
            for i in range(downstream_obs, num_segments):
                state.append(-1.0)
                state.append(-1.0)

        return state

    def get_state(self):
        new_state = self.get_base_state()

        # roll short-term memory and preprend new state
        index = self.num_concat_states * self.n_base_states
        self.states[:index] = np.roll(self.states[:index], self.n_base_states)
        self.states[:self.n_base_states] = new_state

        if self.num_concat_states_large > 0:
            # roll long-term memory and preprend new state every second
            index2 = index + self.num_concat_states_large * self.n_base_states
            if self.step_counter % 10 == 0:
                self.states[index:index2] = np.roll(self.states[index:index2], self.n_base_states)
                self.states[index:index + self.n_base_states] = new_state

        return self.states

    def step(self, accel=None, ballistic=False, tse=None):
        self.step_counter += 1

        # Only use RL after pos is 0, use IDM before
        if self.pos > 0:
            # get action from model
            action = self.get_action(self.get_state())
            accel = self.action_set[action] if self.config['env_config']['discrete'] else float(action)

            # hardcoded gap closing
            if self.get_headway() > self.max_headway:
                accel = 1.0

            # failsafe
            accel = self.apply_failsafe(accel)
        else:
            accel = self.idm.get_accel(self.speed, self.get_leader_speed(), self.get_headway(), self.dt)
            accel = IDMVehicle.apply_failsafe(self, accel)

        return super().step(accel=accel, ballistic=True, tse=tse)
