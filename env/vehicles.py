from env.accel_controllers import IDMController, TimeHeadwayFollowerStopper
from env.failsafes import safe_velocity
import numpy as np


class Vehicle(object):
    def __init__(self, vid, name=None,
                 pos=0, speed=0, accel=0,
                 length=5.0, max_accel=1.3, max_decel=2.0,
                 timestep=None, leader=None,
                 **controller_args):
        self.vid = vid
        self.name = name
        self.id_name = f'{vid}_{name}'
        
        self.pos = pos
        self.speed = speed
        self.accel = accel
        self.dt = timestep
        self.length = length
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.leader = leader
        

        self.controller_args = controller_args

        assert (timestep is not None)

    def step(self, accel=None, ballistic=False):
        if accel is not None:
            self.accel = accel

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

    def get_leader_speed(self):
        if self.leader is None:
            return None
        return self.leader.speed

    def apply_failsafe(self, accel):
        v_safe = safe_velocity(self.speed, self.leader.speed, self.get_headway(), 5, self.dt)
        v_next = self.speed + accel * self.dt
        if v_next > v_safe:
            accel = np.clip((v_safe - self.speed) / self.dt, np.abs(self.max_decel), self.max_accel)
        return accel


class IDMVehicle(Vehicle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.idm = IDMController(**self.controller_args)

    def step(self):
        accel = self.idm.get_accel(self.speed, self.get_leader_speed(), self.get_headway(), self.dt)
        accel = self.apply_failsafe(accel)

        return super().step(accel=accel, ballistic=False)


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
        return True


class RLVehicle(Vehicle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self):
        raise NotImplementedError
        accel = 0

        return super().step(accel=accel, ballistic=False)

class FSVehicle(Vehicle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.fs = TimeHeadwayFollowerStopper(**self.controller_args)

    def step(self):
        accel = self.fs.get_accel(self.speed, self.get_leader_speed(), self.get_headway(), self.dt)
        accel = self.apply_failsafe(accel)

        return super().step(accel=accel, ballistic=False)


