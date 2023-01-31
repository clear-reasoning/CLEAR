"""MegaController Submissions."""
import abc
from copy import deepcopy
from collections import defaultdict

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spi

from trajectory.env.planners import Zhe as SpeedPlanner

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

    # @abc.abstractmethod
    # def run_speed_planner(self, veh):
    #     """Get target_speed and max_headway from Speed Planner.

    #     Parameters
    #     ----------
    #     veh : Vehicle
    #         The AvVehicle vehicle object

    #     Returns
    #     -------
    #     target_speed_profile : list of float
    #         Profile of suggested speed setting.
    #     max_headway : list of bool
    #         Profile of suggested headway.
    #         Tentatively, True if opening a gap is not suggested, False if otherwise.
    #     """
    #     pass

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
        
        self.speed_planner = SpeedPlanner()

    def get_target(self, veh, pos_delta=0):
        """See parent class."""
        x_av = np.array([veh.pos + pos_delta])  # position of AV
        x_seg = np.array(veh._tse["segments"])  # position of every segment
        v_seg = np.array(veh._tse["avg_speed"])  # approximated speed at every segment

        if veh._tse["avg_speed"] is not None:
            target_speeds, max_headways = self.speed_planner.get_target(x_av, x_seg, v_seg)
        else:
            target_speeds, max_headways = [33.0], [True]

        return target_speeds[0], max_headways[0]
    
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
