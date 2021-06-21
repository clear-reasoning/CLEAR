from env.accel_controllers import TimeHeadwayFollowerStopper, IDMController
from env.failsafes import safe_velocity
import numpy as np


class Vehicle(object):
    def __init__(self, vid, controller, kind=None,
                 pos=0, speed=0, accel=0,
                 length=5.0, max_accel=1.5, max_decel=3.0,
                 timestep=None, leader=None, follower=None,
                 **controller_args):
        self.vid = vid
        self.controller = controller
        self.kind = kind
        self.name = '_'.join(([str(self.kind)] if self.kind is not None else []) + [str(self.vid), self.controller])

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

    def step(self, accel=None, ballistic=False):
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

    def apply_failsafe(self, accel):
        # TODO hardcoded max decel to be conservative
        v_safe = safe_velocity(self.speed, self.leader.speed, self.get_headway(), self.max_decel, self.dt)
        v_next = self.speed + accel * self.dt
        if v_next > v_safe:
            safe_accel = np.clip((v_safe - self.speed) / self.dt, - np.abs(self.max_decel), self.max_accel)
        else:
            safe_accel = accel
        return safe_accel


class IDMVehicle(Vehicle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.idm = IDMController(**self.controller_args)

    def step(self):
        accel = self.idm.get_accel(self.speed, self.get_leader_speed(), self.get_headway(), self.dt)
        self.accel_with_noise_no_failsafe = accel
        self.accel_no_noise_no_failsafe = self.idm.get_accel_without_noise()
        self.accel_no_noise_with_failsafe = self.apply_failsafe(self.accel_no_noise_no_failsafe)
        accel = self.apply_failsafe(accel)

        return super().step(accel=accel, ballistic=True)


class FSVehicle(Vehicle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.fs = TimeHeadwayFollowerStopper(**self.controller_args)

    def step(self):
        self.fs.v_des = self.get_leader_speed()

        accel = self.fs.get_accel(self.speed, self.get_leader_speed(), self.get_headway(), self.dt)
        self.accel_with_noise_no_failsafe = accel
        self.accel_no_noise_no_failsafe = self.fs.get_accel_without_noise()
        self.accel_no_noise_with_failsafe = self.apply_failsafe(self.accel_no_noise_no_failsafe)
        accel = self.apply_failsafe(accel)

        return super().step(accel=accel, ballistic=True)


class TrajectoryVehicle(Vehicle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.trajectory = self.controller_args['trajectory']
        self.step()

    def step(self):
        traj_data = next(self.trajectory, None)
        if traj_data is None:
            return False
        self.pos, self.speed, self.accel = traj_data
        self.accel_no_noise_with_failsafe = self.accel_with_noise_no_failsafe = self.accel_no_noise_no_failsafe = self.accel
        return True


class RLVehicle(Vehicle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self):
        self.accel_no_noise_with_failsafe = self.accel_with_noise_no_failsafe = self.accel_no_noise_no_failsafe = self.accel
        return super().step(ballistic=True)

    def set_accel(self, accel):
        self.accel = self.apply_failsafe(accel)
        return self.accel

class FSVehicle(Vehicle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.fs = TimeHeadwayFollowerStopper(**self.controller_args)

    def step(self):
        accel = self.fs.get_accel(self.speed, self.get_leader_speed(), self.get_headway(), self.dt)
        self.accel_with_noise_no_failsafe = accel
        self.accel_no_noise_no_failsafe = self.fs.get_accel_without_noise()
        self.accel_no_noise_with_failsafe = self.apply_failsafe(self.accel_no_noise_no_failsafe)
        accel = self.apply_failsafe(accel)

        return super().step(accel=accel, ballistic=False)

# TODO(nl) add FS-wrapped RL vehicle

# if self.use_fs:
#     self.follower_stopper.v_des += action
#     self.follower_stopper.v_des = max(self.follower_stopper.v_des, 0)
#     self.follower_stopper.v_des = min(self.follower_stopper.v_des, self.max_speed)
#     # TODO(eugenevinitsky) decide on the integration scheme, whether we want this to depend on current or next pos
#     accel = self.follower_stopper.get_accel(self.av['speed'], self.leader_speeds[self.traj_idx],
#                                             self.leader_positions[self.traj_idx] - self.av['pos'],
#                                             self.time_step)
