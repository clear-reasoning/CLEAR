from trajectory.env.acc_controller import ACCController
from trajectory.env.megacontroller import MegaController
from trajectory.env.vehicles import Vehicle


class ACCVehicle(Vehicle):
    """ACCVehicle."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.acc = ACCController(**self.controller_args)

    def step(self, accel=None, ballistic=False, tse=None):
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
        self.speed_setting = None
        self.gap_setting = None

    # Used for failsafe penalty, not necessarily what's applied
    def failsafe_threshold(self):
        return 6 * ((self.speed + 1 + self.speed * 4 / 30) - self.leader.speed)

    def step(self, accel=None, ballistic=False, tse=None):
        return super().step(accel=self.accel, ballistic=True, tse=tse)

    def get_speed_setting(self):
        """Pretty sure this is in m/s."""
        return self.speed_setting

    def get_gap_setting(self):
        return self.gap_setting

    def set_speed_setting(self, speed_setting):
        self.speed_setting = speed_setting
    
    def set_gap_setting(self, gap_setting):
        self.gap_setting = gap_setting

    def set_acc(self, large_gap_threshold=120):
        speed_setting = self.speed_setting
        gap_setting = self.gap_setting
        # if self.get_headway() >= large_gap_threshold:
        #     speed_setting = self.max_speed
        # if self.get_headway() <= self.failsafe_threshold():
        #     speed_setting = 0

        accel = self.megacontroller.get_acc_accel(self.speed, self.get_leader_speed(), self.get_headway(),
                                                  speed_setting, gap_setting)
        self.accel_with_noise_no_failsafe = accel
        self.accel_no_noise_no_failsafe = accel
        # Note that failsafe applied isn't necessarily the same as failsafe_threshold
        self.accel_no_noise_with_failsafe = self.apply_failsafe(self.accel_no_noise_no_failsafe)
        self.accel = self.accel_no_noise_with_failsafe
        return self.accel
