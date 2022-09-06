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
        super().__init__(**kwargs)
        # self.acc = ACCController(**self.controller_args)
        self.megacontroller = MegaController(output_acc=True)
        self.max_speed = 35.76  # 80 mph

    def step(self, accel=None, ballistic=False, tse=None):
        return super().step(accel=self.accel, ballistic=True, tse=tse)

    def set_acc(self, speed_setting, gap_setting, large_gap_threshold=120):
        if self.get_headway() >= large_gap_threshold:
            speed_setting = self.max_speed
            gap_setting = 3

        accel = self.megacontroller.get_acc_accel(self.speed, self.get_leader_speed(), self.get_headway(),
                                                  speed_setting, gap_setting)
        self.accel_with_noise_no_failsafe = accel
        self.accel_no_noise_no_failsafe = accel
        self.accel_no_noise_with_failsafe = self.apply_failsafe(self.accel_no_noise_no_failsafe)
        self.accel = self.accel_no_noise_with_failsafe
        return self.accel
