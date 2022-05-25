from trajectory.env.accel_controllers import TimeHeadwayFollowerStopper, IDMController
from trajectory.env.failsafes import safe_velocity, safe_ttc_velocity
import numpy as np
import bisect


class Vehicle(object):
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
        if self.leader is None:
            return None
        return self.leader.pos - self.pos - self.length

    def get_time_headway(self):
        if self.leader is None:
            return None
        return np.inf if self.speed == 0 else self.get_headway() / self.speed

    def get_leader_speed(self):
        if self.leader is None:
            return None
        return self.leader.speed

    def get_speed_difference(self):
        if self.leader is None:
            return None
        return self.speed - self.leader.speed

    def get_time_to_collision(self):
        if self.leader is None:
            return None
        return np.inf if self.get_speed_difference() <= 0 else self.get_headway() / self.get_speed_difference()

    def apply_failsafe(self, accel):
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
        return self._tse["segments"]

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.idm = IDMController(**self.controller_args)

    def step(self, accel=None, ballistic=False, tse=None):
        accel = self.idm.get_accel(self.speed, self.get_leader_speed(), self.get_headway(), self.dt)
        self.accel_with_noise_no_failsafe = accel
        self.accel_no_noise_no_failsafe = self.idm.get_accel_without_noise()
        self.accel_no_noise_with_failsafe = self.apply_failsafe(self.accel_no_noise_no_failsafe)
        accel = self.apply_failsafe(accel)

        return super().step(accel=accel, ballistic=True, tse=tse)


class FSVehicle(Vehicle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.fs = TimeHeadwayFollowerStopper(**self.controller_args)

    def step(self, accel=None, ballistic=False, tse=None):
        self.fs.v_des = self.get_leader_speed()

        accel = self.fs.get_accel(self.speed, self.get_leader_speed(), self.get_headway(), self.dt)
        self.accel_with_noise_no_failsafe = accel
        self.accel_no_noise_no_failsafe = self.fs.get_accel_without_noise()
        self.accel_no_noise_with_failsafe = self.apply_failsafe(self.accel_no_noise_no_failsafe)
        accel = self.apply_failsafe(accel)

        return super().step(accel=accel, ballistic=True, tse=tse)


class TrajectoryVehicle(Vehicle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.trajectory = self.controller_args['trajectory']
        self.step()

    def step(self, accel=None, ballistic=False, tse=None):
        traj_data = next(self.trajectory, None)
        if traj_data is None:
            return False
        self.pos, self.speed, self.accel = traj_data
        self.accel_no_noise_with_failsafe = self.accel
        self.accel_with_noise_no_failsafe = self.accel
        self.accel_no_noise_no_failsafe = self.accel
        return True


class RLVehicle(Vehicle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, accel=None, ballistic=False, tse=None):
        return super().step(accel=self.accel, ballistic=True, tse=tse)

    def set_accel(self, accel, large_gap_threshold=120):
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.fs = TimeHeadwayFollowerStopper(**self.controller_args)
        self.fs.v_des = self.speed

    def step(self, accel=None, ballistic=False, tse=None):
        accel = self.fs.get_accel(self.speed, self.get_leader_speed(), self.get_headway(), self.dt)
        self.accel_with_noise_no_failsafe = accel
        self.accel_no_noise_no_failsafe = self.fs.get_accel_without_noise()
        self.accel_no_noise_with_failsafe = self.apply_failsafe(self.accel_no_noise_no_failsafe)
        accel = self.apply_failsafe(accel)

        return super().step(accel=accel, ballistic=True, tse=tse)

    def set_vdes(self, vdes):
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
            raise NotImplementedError

        self.max_headway = self.config['env_config']['max_headway']
        self.num_concat_states = self.config['env_config']['num_concat_states']
        self.num_concat_states_large = self.config['env_config']['num_concat_states_large']
        self.augment_vf = self.config['env_config']['augment_vf']
        self.n_base_states = len(self.get_base_state())
        n_states = self.n_base_states * (self.num_concat_states + self.num_concat_states_large) * (2 if self.augment_vf else 1)
        self.states = np.zeros(n_states)
        self.step_counter = 0

        # retrieve algorithm
        alg_module, alg_class = re.match("<class '(.+)\\.([a-zA-Z\\_]+)'>", self.config['algorithm']).group(1, 2)
        assert (alg_module.split('.')[0] in ['stable_baselines3', 'algos'] or alg_module.split('.')[1] == 'algos')
        algorithm = getattr(importlib.import_module(alg_module), alg_class)

        # load checkpoint into model
        self.model = algorithm.load(cp_path)

    def get_action(self, state):
        return self.model.predict(state, deterministic=True)[0][0]

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
        return [
            self.speed / 40.0,
            self.leader.speed / 40.0,
            self.get_headway() / 100.0,
        ]

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
                self.states[index:index+self.n_base_states] = new_state

        return self.states

    def step(self, accel=None, ballistic=False, tse=None):
        self.step_counter += 1

        # get action from model
        accel = self.get_action(self.get_state())

        # hardcoded gap closing
        if self.get_headway() > self.max_headway:
            accel = 1.0

        # failsafe
        accel = self.apply_failsafe(accel)

        return super().step(accel=accel, ballistic=True, tse=tse)
