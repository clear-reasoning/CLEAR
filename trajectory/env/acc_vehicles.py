from trajectory.env.acc_controller import ACCController
from trajectory.env.megacontroller import MegaController
from trajectory.env.vehicles import Vehicle

import pandas as pd


class ACCVehicle(Vehicle):
    """ACCVehicle."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.acc = ACCController(**self.controller_args)
        self.tse_log = pd.DataFrame()

    def step(self, accel=None, ballistic=False, tse=None, tse_log=None):
        """See parent."""
        accel = self.acc.get_accel(self.speed, self.get_leader_speed(), self.get_headway())
        self.accel_with_noise_no_failsafe = accel
        self.accel_no_noise_no_failsafe = accel
        self.accel_no_noise_with_failsafe = accel

        return super().step(accel=accel, ballistic=True, tse=tse)


class ACCWrappedRLVehicle(Vehicle):
    """ACCWrappedRLVehicle."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs, max_decel=4.0)
        # self.acc = ACCController(**self.controller_args)
        self.max_speed = 35.76  # 80 mph
        self.megacontroller = MegaController(output_acc=True, speed_setting=self.max_speed)
        self.tse_log = pd.DataFrame()

    # Used for failsafe penalty, not necessarily what's applied
    def failsafe_threshold(self):
        return 6 * ((self.speed + 1 + self.speed * 4 / 30) - self.leader.speed)

    def step(self, accel=None, ballistic=False, tse=None, tse_log=None):
        return super().step(accel=self.accel, ballistic=True, tse=tse)

    def set_acc(self, speed_setting, gap_setting, large_gap_threshold=120):
        if self.get_headway() >= large_gap_threshold:
            speed_setting = self.max_speed
        if self.get_headway() <= self.failsafe_threshold():
            speed_setting = 0

        accel = self.megacontroller.get_acc_accel(self.speed, self.get_leader_speed(), self.get_headway(),
                                                  speed_setting, gap_setting)
        self.accel_with_noise_no_failsafe = accel
        self.accel_no_noise_no_failsafe = accel
        # Note that failsafe applied isn't necessarily the same as failsafe_threshold
        self.accel_no_noise_with_failsafe = self.apply_failsafe(self.accel_no_noise_no_failsafe)
        self.accel = self.accel_no_noise_with_failsafe
        return self.accel
