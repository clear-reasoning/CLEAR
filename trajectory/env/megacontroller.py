"""MegaController Submissions."""
import abc
from copy import deepcopy
from collections import defaultdict


import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spi

MPH_TO_MS = 0.44704


class AbstractMegaController(metaclass=abc.ABCMeta):
    """Abstract MegaController."""

    def __init__(self, lc_flag, speed_setting=None, gap_setting=3, output_acc=False):
        if output_acc:
            from trajectory.env.acc_controller import ACCController

            # Instantiate ACCController for use when output acc is enabled
            self.acc = ACCController()
            self.speed_setting = speed_setting
            self.gap_setting = gap_setting
            self.time_counter = 0
            self.gap_transitions = {
                1: 2,
                2: 3,
                3: 1,
            }
        self.accel_without_noise = 0
        self.lc_flag = lc_flag

    @abc.abstractmethod
    def run_speed_planner(self, veh):
        """Get target_speed and max_headway from Speed Planner.

        Parameters
        ----------
        veh : Vehicle
            The AvVehicle vehicle object

        Returns
        -------
        target_speed_profile : list of float
            Profile of suggested speed setting.
        max_headway : list of bool
            Profile of suggested headway.
            Tentatively, True if opening a gap is not suggested, False if otherwise.
        """
        pass

    def get_accel(
        self,
        this_vel,
        lead_vel,
        headway,
        prev_vels=[-1] * 640,
        target_speed=-1,
        max_headway=True,
        sim_step=0.1,
    ):
        """Get the requested acceleration.

        Parameters
        ----------
        this_vel : float
            Current ego speed in m/s.
        lead_vel : float
            Current leader speed in m/s.
        headway : float
            Current space gap in m.
        prev_vels : list of floats
            Buffered list of 64 seconds worth of previous lead_vel observations.
        target_speed : float
            Recommended target speed from speed planner.
        max_headway : bool
            Recommendation on whether to close gap (True) or open gap (False).
        sim_step : float
            Simulation step size in seconds.

        Returns
        -------
        acceleration : float
            Requested acceleration in m/s/s.
        """
        accel = self.get_main_accel(
            this_vel, lead_vel, headway, prev_vels, target_speed, max_headway, sim_step
        )
        if self.previous_accel is None:
            self.previous_accel = accel
            self.previous_headway = headway

        if self.lc_flag == 0:
            self.lc_flag = self.detect_lane_change(accel, headway, this_vel, lead_vel, sim_step)

        if self.lc_flag < 0:
            accel = self.get_cutin_accel(
                this_vel,
                lead_vel,
                headway,
                prev_vels,
                target_speed,
                max_headway,
                sim_step,
                accel,
            )
        elif self.lc_flag > 0:
            accel = self.get_cutout_accel(
                this_vel,
                lead_vel,
                headway,
                prev_vels,
                target_speed,
                max_headway,
                sim_step,
                accel,
            )

        self.accel_without_noise = accel
        self.previous_accel = accel
        self.previous_headway = headway
        return accel

    @abc.abstractmethod
    def get_main_accel(
        self,
        this_vel,
        lead_vel,
        headway,
        prev_vels=[-1] * 640,
        target_speed=-1,
        max_headway=True,
        sim_step=0.1,
    ):
        """Get main controller acceleration.

        Parameters
        ----------
        See get_accel() docstring.

        Returns
        -------
        acceleration : float
            Requested acceleration in m/s/s.
        """
        pass

    @abc.abstractmethod
    def detect_lane_change(self, accel, headway, this_vel, lead_vel, sim_step=0.1):
        """Detect lane change from changes in space gap.

        Returns
        -------
        lc_flag : {-1, 0, 1}
            Flag whether a lane change was detected.
        """
        pass

    @abc.abstractmethod
    def get_cutout_accel(
        self,
        this_vel,
        lead_vel,
        headway,
        prev_vels=[-1] * 640,
        target_speed=-1,
        max_headway=True,
        sim_step=0.1,
        main_accel=0,
    ):
        """Get cut-out recovery acceleration.

        Parameters
        ----------
        See get_accel() docstring.

        Returns
        -------
        acceleration : float
            Requested acceleration in m/s/s.
        """
        pass

    @abc.abstractmethod
    def get_cutin_accel(
        self,
        this_vel,
        lead_vel,
        headway,
        prev_vels=[-1] * 640,
        target_speed=-1,
        max_headway=True,
        sim_step=0.1,
        main_accel=0,
    ):
        """Get cut-in recovery acceleration.

        Parameters
        ----------
        See get_accel() docstring.

        Returns
        -------
        acceleration : float
            Requested acceleration in m/s/s.
        """
        pass

    def get_acc_accel(
        self,
        this_vel,
        lead_vel,
        headway,
        target_speed_setting=None,
        target_gap_setting=None,
        sim_step=0.1,
        force=False,
    ):
        """Get ACC acceleration.

        Parameters
        ----------
        See get_accel() docstring.

        Returns
        -------
        acceleration : float
            Requested acceleration in m/s/s.
        """
        if self.time_counter >= 0.5 or force:
            assert target_speed_setting is not None
            assert target_gap_setting is not None
            self.time_counter = 0

            # Set target speed setting to at least 20 mph.
            target_speed_setting_mph = target_speed_setting / MPH_TO_MS
            target_speed_setting_mph = max(target_speed_setting_mph, 20)
            if self.speed_setting is None:
                current_speed_setting_mph = target_speed_setting_mph
            else:
                # Set speed setting of controller.
                current_speed_setting_mph = round(self.speed_setting / MPH_TO_MS)

                # Simulate holding the up button.
                def hold_up(curr_speed_mph: int):
                    if (curr_speed_mph + 1) % 5 == 0:
                        return curr_speed_mph + 6
                    elif curr_speed_mph % 5 == 0:
                        return curr_speed_mph + 5
                    return max((curr_speed_mph // 5 + 1) * 5, 20)

                # Simulate clicking the up button.
                def click_up(curr_speed_mph: int):
                    return max(curr_speed_mph + 1, 20)

                # Simulate holding the down button.
                def hold_down(curr_speed_mph: int):
                    if (curr_speed_mph - 1) % 5 == 0:
                        return curr_speed_mph - 6
                    elif curr_speed_mph % 5 == 0:
                        return curr_speed_mph - 5
                    return max((curr_speed_mph // 5) * 5, 20)

                # Simulate clicking the down button.
                def click_down(curr_speed_mph: int):
                    return max(curr_speed_mph - 1, 20)

                difference_mph = (
                    target_speed_setting_mph - current_speed_setting_mph)

                if difference_mph > 0:
                    hold_up_speed = hold_up(current_speed_setting_mph)
                    click_up_speed = click_up(current_speed_setting_mph)
                    hold_diff = abs(hold_up_speed - target_speed_setting_mph)
                    click_diff = abs(click_up_speed - target_speed_setting_mph)
                    if click_diff > hold_diff:
                        new_speed_mph = hold_up_speed
                    else:
                        new_speed_mph = click_up_speed
                elif difference_mph < 0:
                    hold_down_speed = hold_down(current_speed_setting_mph)
                    click_down_speed = click_down(current_speed_setting_mph)
                    hold_diff = abs(hold_down_speed - target_speed_setting_mph)
                    click_diff = abs(click_down_speed - target_speed_setting_mph)
                    if click_diff > hold_diff:
                        new_speed_mph = hold_down_speed
                    else:
                        new_speed_mph = click_down_speed
                else:
                    new_speed_mph = target_speed_setting_mph
            self.speed_setting = new_speed_mph * MPH_TO_MS

            # Set gap setting of controller.
            if target_gap_setting != self.gap_setting:
                self.gap_setting = self.gap_transitions[self.gap_setting]
        else:
            self.time_counter += sim_step

        if force:
            accel = self.acc.get_accel(
                this_vel=this_vel,
                lead_vel=lead_vel,
                headway=headway,
                speed_setting=target_speed_setting,
                time_gap_setting=target_gap_setting,
            )
        else:
            accel = self.acc.get_accel(
                this_vel=this_vel,
                lead_vel=lead_vel,
                headway=headway,
                speed_setting=self.speed_setting,
                time_gap_setting=self.gap_setting,
            )
        self.accel_without_noise = accel
        return accel

    @abc.abstractmethod
    def get_acc_settings(
        self,
        this_vel,
        lead_vel,
        headway,
        prev_vels=[-1] * 640,
        target_speed=-1,
        max_headway=True,
        sim_step=0.1,
    ):
        """Get ACC settings.

        Parameters
        ----------
        See get_accel() docstring.

        Returns
        -------
        speed_setting : float
            Requested speed setting in m/s.
        gap_setting : int
            Requested time gap setting from {1, 2, 3}.
        """
        pass

    def get_accel_without_noise(
        self,
        this_vel,
        lead_vel,
        headway,
        prev_vels=[-1] * 640,
        target_speed=-1,
        max_headway=True,
        sim_step=0.1,
    ):
        """Return the accel without applying any noise.

        Must be called after get_accel or get_acc_accel to updated result.
        """
        return self.accel_without_noise


class MegaController(AbstractMegaController):
    """MegaController."""

    def __init__(
        self,
        speed_setting=None,
        gap_setting=3,
        output_acc=False,
        max_decel_lead=-6,
        max_accel=1,
        max_decel_comfort=-0.75,
        max_decel_des=-0.4,
        max_decel=-3,
        v_limit=35,
        s0=2,
        k=1,
        long_time_head=5,
        gap_pen=1,
        lc_flag=0,
    ):
        super().__init__(lc_flag, speed_setting=speed_setting, gap_setting=gap_setting, output_acc=output_acc)
        self.max_accel = max_accel
        self.max_decel_des = max_decel_des
        self.max_decel_comfort = max_decel_comfort
        self.max_decel = max_decel
        self.max_decel_lead = max_decel_lead
        self.v_limit = v_limit
        self.long_time_head = long_time_head
        self.gap_pen = gap_pen
        self.s0 = s0
        self.k = k
        self.previous_dx = None
        self.previous_v_des = None
        self.previous_v_max = None
        self.previous_lead_acc = 0
        self.previous_accel = None
        self.previous_headway = None
        self.time_since_lc = 0
        self.target_speed_profile = defaultdict(lambda: -1)
        self.max_headway_profile = defaultdict(lambda: -1)
        
        self.latest_tse = {}

    def run_speed_planner(self, veh):
        """See parent class."""
        new_dx = 100  # 100 meter resolution on resampled profile
        x_range = np.arange(-20000, 21000, new_dx)

        if veh._tse["avg_speed"] is not None:
            if veh not in self.latest_tse or (self.latest_tse[veh] != veh._tse["avg_speed"]).any():
                self.latest_tse[veh] = veh._tse["avg_speed"]
                x = np.array(veh._tse["segments"])
                speed = np.array(veh._tse["avg_speed"])

                # resample to finer spatial grid
                speed_interp = spi.interp1d(
                    x, speed, kind="linear", fill_value="extrapolate"
                )
                speed = speed_interp(x_range)

                # apply gaussian smoothing
                gaussian_smoothed_speed = np.array(
                    [
                        gaussian(x_range[i], x_range, np.array(speed), sigma=250)
                        for i in range(len(x_range))
                    ]
                )

                (
                    target_speed_profile,
                    max_headway_profile,
                ) = get_target_profiles_with_fixed_decel(
                    x_range, gaussian_smoothed_speed, decel=-0.5
                )
                self.target_speed_profile[veh] = target_speed_profile
                self.max_headway_profile[veh] = max_headway_profile

            return

        # Update with free speed profile if veh._tse is disabled or no bottleneck observed
        free_speed_profile = tuple([x_range, np.ones(x_range.shape) * 33, np.array(1)])
        free_headway_profile = tuple(
            [x_range, np.array([True] * len(x_range)), np.array(1)]
        )
        self.target_speed_profile[veh] = free_speed_profile
        self.max_headway_profile[veh] = free_headway_profile
        return

    def get_target(self, veh):
        """Get target speed and max headway."""
        target_speed = get_target_by_position(self.target_speed_profile[veh], veh.pos, float)
        max_headway = get_target_by_position(self.max_headway_profile[veh], veh.pos, bool)
        return target_speed, max_headway

    def get_main_accel(
        self,
        this_vel,
        lead_vel,
        headway,
        prev_vels=[-1] * 640,
        target_speed=-1,
        max_headway=True,
        sim_step=0.1,
        gap_target=2,
    ):
        """See parent class."""
        # TODO (Amaury & Shengquan): update as needed
        if target_speed == -1:
            filtered_vel = []
            for vel in prev_vels:
                if vel != -1:
                    filtered_vel.append(vel)
            if len(filtered_vel) > 0:
                target_speed = np.mean(filtered_vel)
            else:
                target_speed = lead_vel
            # max_headway = True
        else:
            # TODO Amaury, change the gap penalization
            # if target_speed < lead_vel:
            # target_speed = target_speed + (
            #     0.1 * (max(0, 5 * (headway - gap_target) / max(1, this_vel))) ** 2
            # )
            # Mode 2
            target_speed = max(min(target_speed, 1.2 * lead_vel), 0.8 * lead_vel)

        # Penalise too large headway
        # target_speed = target_speed * 0.6 + lead_vel * 0.4
        # max_headway = True

        # in order to deal with ZeroDivisionError
        if abs(headway) < 1e-3:
            headway = 1e-3

        # Create leader's accel #TODO can do a better smoothing
        if prev_vels[0] == -1:
            lead_acc = 0
        elif (lead_vel - prev_vels[0]) < 3.3 * sim_step:
            lead_acc = (lead_vel - prev_vels[0]) / sim_step
        else:
            lead_acc = self.previous_lead_acc

        self.previous_lead_acc = lead_acc  # Update lead_acc for next timestep

        # Create the acceleration
        # TODO add the patch for spotting the lack of signal.
        # TODO maybe issues when there is a lead veh then no lead veh then again a lead veh for the memory quantity

        # Create v_des from the motion planner

        v_des = target_speed
        assert v_des != -1
        if self.previous_v_des is None:  # Initial step
            v_des_dot = 0
        else:
            v_des_dot = (
                v_des - self.previous_v_des
            ) / sim_step  # Create the derivative of V_DES

        # update previous v_des
        self.previous_v_des = v_des

        if lead_vel is None:  # no car ahead
            a_mng = self.max_accel
            v_max_dot = 0
            v_max = self.v_limit
        else:
            dx = headway
            if self.previous_dx is None:  # Initial step
                self.previous_dx = dx

            # Compute v_max and its derivative
            v_max = np.sqrt(
                2
                * np.abs(self.max_decel)
                * (
                    max(dx - self.s0, 0)
                    + 0.5 * ((lead_vel) ** 2) / np.abs(self.max_decel_lead)
                )
            )

            if self.previous_v_max is None:  # Initial step
                v_max_dot = 0
            elif (
                np.abs(dx - self.previous_dx) > sim_step * 50
            ):  # Exclude the discontinuities in headway
                v_max_dot = 0
            else:
                v_max_dot = (
                    v_max - self.previous_v_max
                ) / sim_step  # Compute the derivative of v_max

            self.previous_v_max = v_max  # update previous_v_max

            # Compute a_mng (acceleration managment)

            if lead_acc < 0:
                a_0 = lead_acc * this_vel / (lead_vel + 0.001)
                a_12 = (
                    -0.5
                    * (this_vel) ** 2
                    / (
                        max(0, dx - self.s0)
                        + 0.5 * ((lead_vel + 0.00001) ** 2) / abs(lead_acc - 0.01)
                    )
                )
                if a_0 < a_12:
                    a_mng = a_12
                else:
                    if lead_vel >= this_vel:
                        a_mng = a_0
                    else:
                        a_mng = lead_acc - ((lead_vel - this_vel) ** 2) / (
                            2 * max(dx - self.s0, 0.0001)
                        )
                    assert a_mng <= 0, (a_mng, a_0)  # changed to <= 0
            elif lead_acc >= 0:
                if lead_vel <= this_vel:
                    a_mng = lead_acc - ((max(0, this_vel - lead_vel)) ** 2) / (
                        2 * max(dx - self.s0, 0.001)
                    )
                else:
                    a_mng = min(
                        self.max_accel, lead_acc + (lead_acc) * (lead_vel - this_vel),
                    )
                    # TODO this can be changed / might be the source of some oscillations. together with long time head
            else:
                a_mng = None
                print("This is an error")

            # Unactivate a_mng when the vehicle is above a certain distance
            assert self.long_time_head > 1 and self.long_time_head >= gap_target

            if dx > this_vel * (self.long_time_head - 1):
                a_mng = a_mng + self.gap_pen * (
                    dx - (self.long_time_head - 1) * this_vel
                ) / max(this_vel, 0.001)

            self.previous_dx = dx  # Update the previous headway

        # Final acceleration
        a_mng = max(a_mng, -np.abs(self.max_decel_comfort))

        a_vdes = max(
            -self.k * (this_vel - v_des) + v_des_dot, -np.abs(self.max_decel_des)
        )
        a_vmax = -self.k * (this_vel - v_max) + v_max_dot
        accel = min(min(a_vdes, a_vmax), a_mng)
        accel = min(max(accel, -np.abs(self.max_decel)), self.max_accel)

        if this_vel >= self.v_limit:
            accel = min(0, accel)

        return accel

    def detect_lane_change(self, main_accel, headway, this_vel, lead_vel, sim_step=0.1):
        """Check the criteria for detecting lane change and applying the lane change controller.

        Parameters
        ----------
        main_accel : float
            The main acceleration m/s/s
        headway : float
            Current space gap in m.
        this_vel : float
            Current ego velocity in m/s.
        lead_vel : float
            Current leader velocity in m/s.

        Returns
        -------
        See parent class.
        """
        lc_thresh = 3.5
        a_comfort = 0.9 * sim_step
        th_safe = 2
        ttc_safe = 4.5

        rv = lead_vel - this_vel
        da = np.abs(main_accel - self.previous_accel)
        th = headway / this_vel if this_vel > 0 else 60
        ttc = -headway / rv if rv < 0 else 60
        difference = headway - self.previous_headway

        if difference <= -lc_thresh:
            if da > a_comfort:
                if rv > 0 and th > th_safe:
                    return -1
                elif rv < 0 and ttc > ttc_safe:
                    return -1
        elif difference >= lc_thresh:
            if da > a_comfort:
                if rv > 0 and th > th_safe:
                    return 1
                elif rv < 0 and ttc > ttc_safe:
                    return 1
        return 0

    def get_lc_recovery_accel(
        self, main_accel, headway, this_vel, lead_vel, sim_step=0.1
    ):
        """Get recovery acceleration when a lane change is detected.

        Parameters
        ----------
        main_accel : float
            The main acceleration m/s/s
        headway : float
            Current space gap in m.
        this_vel : float
            Current ego velocity in m/s.
        lead_vel : float
            Current leader velocity in m/s.

        Returns
        -------
        acceleration : float
            Requested acceleration in m/s/s.
        """
        # Define controller params #
        h_star = 1.32
        rv_star_positive = 10.34/headway
        rv_star_negative = 10.34/headway
        epsilon = 0.3 * sim_step
        f_h = 0.75
        f_rv = 0.25

        # Controller logic #
        th = headway / this_vel if this_vel > 0 else 60
        rv = lead_vel - this_vel
        rv_star = rv_star_positive * (rv >= 0) + rv_star_negative * (rv < 0)
        alpha = f_h * np.tanh(h_star * th) + f_rv * (0.5 * np.tanh(rv_star * rv) + 0.5)
        # alpha = alpha / (1+np.sqrt(self.time_since_lc))
        accel = alpha * self.previous_accel + (1 - alpha) * main_accel
        if np.abs(accel - main_accel) < epsilon:
            self.lc_flag = 0
            self.time_since_lc = 0
            return main_accel
        self.time_since_lc += 1
        return accel

    def get_cutout_accel(
        self,
        this_vel,
        lead_vel,
        headway,
        prev_vels=[-1] * 640,
        target_speed=-1,
        max_headway=True,
        sim_step=0.1,
        main_accel=0,
    ):
        """See parent class."""
        return self.get_lc_recovery_accel(main_accel, headway, this_vel, lead_vel)

    def get_cutin_accel(
        self,
        this_vel,
        lead_vel,
        headway,
        prev_vels=[-1] * 640,
        target_speed=-1,
        max_headway=True,
        sim_step=0.1,
        main_accel=0,
    ):
        """See parent class."""
        return self.get_lc_recovery_accel(main_accel, headway, this_vel, lead_vel)

    def get_acc_settings(
        self,
        this_vel,
        lead_vel,
        headway,
        prev_vels=[-1] * 640,
        target_speed=-1,
        max_headway=True,
        sim_step=0.1,
    ):
        """See parent class."""
        # TODO (Eugene, Nathan, Adit, Fahd, Kathy): replace with trained controller
        if target_speed == -1:
            if lead_vel < 20 * MPH_TO_MS:
                speed_setting = lead_vel
                if abs(this_vel - lead_vel) < 2.0:
                    gap_setting = 2
                elif this_vel > lead_vel:
                    gap_setting = 1
                else:
                    gap_setting = 3
            else:
                if this_vel > lead_vel + 2:
                    gap_setting = 1
                    speed_setting = lead_vel
                else:
                    gap_setting = 3
                    speed_setting = (this_vel + lead_vel) / 2.0
        else:
            if target_speed < lead_vel:
                speed_setting = target_speed * 0.6 + lead_vel * 0.4
            else:
                speed_setting = target_speed
            if max_headway:
                gap_setting = 1
            else:
                gap_setting = 3
        speed_setting = round(speed_setting / MPH_TO_MS) * MPH_TO_MS
        speed_setting = max(speed_setting, MPH_TO_MS * 20)
        return speed_setting, gap_setting


def get_sl_by_distance(x, v_0=33, v_bn=28 / 3.6, D=16000):
    """Get SL by distance."""
    a = (v_bn * v_bn - v_0 * v_0) / (2 * D)
    if (v_0 * v_0 + 2 * a * x) > 0:
        t = -(v_0 - (v_0 * v_0 + 2 * a * x) ** (1 / 2)) / a
        sl = v_0 + a * t
        sl = max(sl, 20 / 3.6)
    else:
        sl = None
    return sl


def get_target_profile_with_constant_decel(
    x, speed, start_loc: float = None, bn_loc: float = None
):
    """Get target profile with constant decel."""
    start_loc = x[int(x.size * 0.12)] if start_loc is None else start_loc
    bn_loc = min(x[speed == speed[speed > 0].min()]) if bn_loc is None else bn_loc

    # to fix the shift of target profile
    unique_seg, unique_index = np.unique(x, return_index=True)
    speed_vsl = speed[unique_index]

    v_0 = speed_vsl[unique_seg == start_loc]
    v_bn = speed_vsl[unique_seg == bn_loc]

    seg_vsl_downstream = unique_seg[(unique_seg >= bn_loc) & (speed_vsl > 0)]
    speed_vsl_downstream = speed_vsl[(unique_seg >= bn_loc) & (speed_vsl > 0)]

    seg_vsl_upstream = unique_seg[(unique_seg <= start_loc) & (speed_vsl > 0)]
    speed_vsl_upstream = speed_vsl[(unique_seg <= start_loc) & (speed_vsl > 0)]

    smoothed_x = np.arange(start_loc + 1, bn_loc, 1)
    smoothed_v = []
    for i in smoothed_x:
        # Coverting unit for INRIX API, but eventually get the same return value
        # smoothed_v.append(get_sl_by_distance((i - start_loc) * 1609, v_0[0] * 0.45, v_bn[0] * 0.45,
        #                                      (bn_loc - start_loc) * 1609) * 2.24)
        smoothed_v.append(
            get_sl_by_distance((i - start_loc), v_0[0], v_bn[0], (bn_loc - start_loc))
        )
    smoothed_v = np.array(smoothed_v)

    x = np.hstack((seg_vsl_upstream, smoothed_x, seg_vsl_downstream))
    y = np.hstack((speed_vsl_upstream, smoothed_v, speed_vsl_downstream))

    return tuple([x, y, 1])


def get_target_profiles_with_fixed_decel(x, speed, decel):
    """Get target profile with fixed decel."""
    dx = x[1] - x[0]
    headway_profile = [True] * len(speed)
    next_speed = speed[-1]
    for i in range(len(x) - 1, 0, -1):
        decel_speed = np.sqrt(next_speed ** 2 - dx * decel * 2.0)
        if decel_speed < speed[i]:
            speed[i] = decel_speed
            headway_profile[i] = False
        next_speed = speed[i]
    return tuple([x, speed, np.array(1)]), tuple([x, headway_profile, np.array(1)])


def get_target_by_position(profile, position, dtype=float):
    """Get target speed by position."""
    if dtype == bool:
        kind = "previous"
    else:
        kind = "linear"
    interp = spi.interp1d(profile[0], profile[1], kind=kind, fill_value="extrapolate")
    return interp(position)


def naive_bn_identify(inrix):
    """Identify bottleneck naively."""
    congest_range = np.where(inrix < 18)
    return congest_range[0][0] if len(congest_range[0]) > 0 else None


def simlified_inverse_pems_bn_identify(inrix):
    """Identify bottleneck with simplified inverse PeMS."""
    inrix_diff = np.diff(inrix)
    gap_loc = np.where(inrix_diff < -8)
    return gap_loc[0][0] + 1 if len(gap_loc) > 0 else None


def anneal(x0, x, z, forward_width, backward_width):
    """Anneal."""
    ix0 = next(i for i in range(len(x)) if x[i] >= x0 - backward_width)
    width = forward_width + backward_width

    x = deepcopy(x[ix0 - 1:])
    z = deepcopy(z[ix0 - 1:])

    # Replace starting point with a cutoff.
    z[0] = z[0] + (z[1] - z[0]) * (x0 - backward_width - x[0]) / (x[1] - x[0])
    x[0] = x0 - backward_width

    try:
        ix1 = next(i for i in range(len(x)) if x[i] >= x0 + forward_width)

        x = deepcopy(x[:ix1])
        z = deepcopy(z[:ix1])

        z[-1] = z[-2] + (z[-1] - z[-2]) * (x0 + forward_width - x[-2]) / (x[-1] - x[-2])
        x[-1] = x0 + forward_width

    except StopIteration:  # doesn't exist

        width = x[-1] - x0
        pass  # TODO

    answer = 0
    for i in range(len(x) - 1):
        x1_p = x[i]
        v1_p = z[i]

        x2_p = x[i + 1]
        v2_p = z[i + 1]

        u1 = x0 - backward_width
        u2 = x0 + forward_width
        w = v1_p * x2_p - v2_p * x1_p

        a1 = u1 * w
        b1 = u1 * (v2_p - v1_p) - w
        c = v2_p - v1_p

        a2 = u2 * w
        b2 = u2 * (v2_p - v1_p) - w

        if x[i + 1] <= x0:
            answer += (
                2
                / (width * backward_width * (x2_p - x1_p))
                * (-a1 * x2_p - b1 * x2_p ** 2 / 2 + c * x2_p ** 3 / 3)
            )
            answer -= (
                2
                / (width * backward_width * (x2_p - x1_p))
                * (-a1 * x1_p - b1 * x1_p ** 2 / 2 + c * x1_p ** 3 / 3)
            )
        elif x[i] < x0:
            answer += (
                2
                / (width * backward_width * (x2_p - x1_p))
                * (-a1 * x0 - b1 * x0 ** 2 / 2 + c * x0 ** 3 / 3)
            )
            answer -= (
                2
                / (width * backward_width * (x2_p - x1_p))
                * (-a1 * x1_p - b1 * x1_p ** 2 / 2 + c * x1_p ** 3 / 3)
            )
            answer += (
                2
                / (width * forward_width * (x2_p - x1_p))
                * (a2 * x2_p + b2 * x2_p ** 2 / 2 - c * x2_p ** 3 / 3)
            )
            answer -= (
                2
                / (width * forward_width * (x2_p - x1_p))
                * (a2 * x0 + b2 * x0 ** 2 / 2 - c * x0 ** 3 / 3)
            )
        else:
            answer += (
                2
                / (width * forward_width * (x2_p - x1_p))
                * (a2 * x2_p + b2 * x2_p ** 2 / 2 - c * x2_p ** 3 / 3)
            )
            answer -= (
                2
                / (width * forward_width * (x2_p - x1_p))
                * (a2 * x1_p + b2 * x1_p ** 2 / 2 - c * x1_p ** 3 / 3)
            )

    return answer


def gaussian(x0, x, z, sigma):
    """Perform a kernel smoothing operation on future average speeds."""
    # Collect relevant traffic-state info.
    # - ix0: segments in front of the ego vehicle
    # - ix1: final segment, or clipped to estimate congested speeds
    ix0 = 0  # next(i for i in range(len(x)) if x[i] >= x0)
    ix1 = len(x)
    x = x[ix0:ix1]
    z = z[ix0:ix1]

    densities = (
        1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-np.square(x - x0) / (2 * sigma ** 2))
    )

    densities = densities / sum(densities)

    return sum(densities * z)


def visualize_profile(
    x: np.array,
    inrix: np.array,
    currrent_profile: tuple,
    target_profile: tuple,
    file_name: str,
):
    """Save plots for debug purpose."""
    x0 = x
    inrix = inrix

    plt.figure()
    plt.plot(x0, inrix, "o", label="INRIX Points")
    plt.plot(
        x0, spi.splev(x0, currrent_profile), color="r", label="Current Speed Profile"
    )
    plt.plot(x0, spi.splev(x0, target_profile), color="b", label="Target Speed Profile")
    plt.xlabel("Position (m)")
    plt.ylabel("speed (meter/s)")
    plt.legend()

    plt.savefig(file_name, dpi=360)
