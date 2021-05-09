import math
import numpy as np

class TimeHeadwayFollowerStopper(object):
    """New FollowerStopper with safety envelopes based on time-headways.

    Usage
    -----
    See base class for example.

    Parameters
    ----------
    veh_id : str
        unique vehicle identifier
    v_des : float, optional
        desired speed of the vehicles (m/s)
    no_control_edges : [str]
        list of edges that we should not apply control on
    """

    def __init__(self,
                 v_des=15,
                 max_accel=1.5,
                 max_deaccel=3.0):

        # other parameters
        self.h_1 = 0.4
        self.h_2 = 1.2
        self.h_3 = 1.8
        self.dx_1_0 = 4.5
        self.dx_2_0 = 5.25
        self.dx_3_0 = 6.0
        self.d_1 = 1.5
        self.d_2 = 1.0
        self.d_3 = 0.5

        self.v_des = v_des
        self.max_accel = max_accel
        self.max_deaccel = max_deaccel

    def get_accel(self, this_vel, lead_vel, headway, env):
        """See parent class."""

        dx = headway
        dv_minus = min(lead_vel - this_vel, 0)

        dx_1 = self.dx_1_0 + 1 / (2 * self.d_1) * dv_minus ** 2 + self.h_1 * this_vel
        dx_2 = self.dx_2_0 + 1 / (2 * self.d_2) * dv_minus ** 2 + self.h_2 * this_vel
        dx_3 = self.dx_3_0 + 1 / (2 * self.d_3) * dv_minus ** 2 + self.h_3 * this_vel
        v = min(max(lead_vel, 0), self.v_des)
        # compute the desired velocity
        if dx <= dx_1:
          v_cmd = 0
        elif dx <= dx_2:
          v_cmd = v * (dx - dx_1) / (dx_2 - dx_1)
        elif dx <= dx_3:
          v_cmd = v + (self.v_des - v) * (dx - dx_2) \
                  / (dx_3 - dx_2)
        else:
          v_cmd = self.v_des

        v_safe = self.safe_velocity(this_vel, lead_vel, headway, env)
        desired_accel = np.clip((v_cmd - this_vel) / env.time_step, -np.abs(self.max_deaccel), self.max_accel)
        v_next = desired_accel * env.time_step + this_vel
        if v_next > v_safe:
          return np.clip((v_safe - this_vel) / env.time_step, -np.abs(self.max_deaccel), self.max_accel)
        else:
          return np.clip((v_cmd - this_vel) / env.time_step, -np.abs(self.max_deaccel), self.max_accel)

    def safe_velocity(self, this_vel, lead_vel, headway, env):
        """Compute a safe velocity for the vehicles.

        Finds maximum velocity such that if the lead vehicle were to stop
        instantaneously, we can bring the following vehicle to rest at the point at
        which the headway is zero.

        WARNINGS:
        1. We assume the lead vehicle has the same deceleration capabilities as our vehicles
        2. We solve for this value using the discrete time approximation to the dynamics. We assume that the
           integration scheme induces positive error in the position, which leads to a slightly more conservative
           driving behavior than the continuous time approximation would induce. However, the continuous time
           safety rule would not be strictly safe.

        Parameters
        ----------
        env : flow.envs.Env
            current environment, which contains information of the state of the
            network at the current time step

        Returns
        -------
        float
            maximum safe velocity given a maximum deceleration, delay in
            performing the breaking action, and speed limit
        """
        # TODO(eugenevinitsky) hardcoding
        min_gap = 2.5
        brake_distance = self.brake_distance(lead_vel, self.max_deaccel,
                                             env.time_step)
        v_safe = self.maximum_safe_stop_speed(headway + brake_distance - min_gap, this_vel, env.time_step)

        if this_vel > v_safe:
            if self.display_warnings:
                print(
                    "=====================================\n"
                    "Speed of vehicle {} is greater than safe speed. Safe velocity "
                    "clipping applied.\n"
                    "=====================================".format(self.veh_id))

        return v_safe

    def brake_distance(self, speed, max_deaccel, sim_step):
        """Return the distance needed to come to a full stop if braking as hard as possible. We assume the delay is a time_step.

        Parameters
        ----------
        speed : float
            ego speed
        max_deaccel : float
            maximum deaccel of the vehicle
        delay : float
            the delay before an action is executed
        sim_step : float
            size of simulation step

        Returns
        -------
        float
            the distance required to stop
        """

        # how much we can reduce the speed in each timestep
        speedReduction = max_deaccel * sim_step
        # how many steps to get the speed to zero
        steps_to_zero = int(speed / speedReduction)
        return sim_step * (steps_to_zero * speed - speedReduction * steps_to_zero * (steps_to_zero + 1) / 2) + \
            speed * sim_step

    def maximum_safe_stop_speed(self, brake_distance, speed, sim_step):
        """Compute the maximum speed that you can travel at and guarantee no collision.

        Parameters
        ----------
        brake_distance : float
            total distance the vehicle has before it must be at a full stop
        speed : float
            current vehicle speed
        sim_step : float
            simulation step size in seconds

        Returns
        -------
        v_safe : float
            maximum speed that can be travelled at without crashing
        """
        v_safe = self.maximum_safe_stop_speed_euler(brake_distance, sim_step)
        return v_safe

    def maximum_safe_stop_speed_euler(self, brake_distance, sim_step):
        """Compute the maximum speed that you can travel at and guarantee no collision for euler integration.

        Parameters
        ----------
        brake_distance : float
            total distance the vehicle has before it must be at a full stop
        sim_step : float
            simulation step size in seconds

        Returns
        -------
        v_safe : float
            maximum speed that can be travelled at without crashing
        """
        if brake_distance <= 0:
            return 0.0

        speed_reduction = self.max_deaccel * sim_step

        s = sim_step
        t = sim_step

        # h = the distance that would be covered if it were possible to stop
        # exactly after gap and decelerate with max_deaccel every simulation step
        # h = 0.5 * n * (n-1) * b * s + n * b * t (solve for n)
        # n = ((1.0/2.0) - ((t + (pow(((s*s) + (4.0*((s*((2.0*h/b) - t)) + (t*t)))), (1.0/2.0))*sign/2.0))/s))
        sqrt_quantity = math.sqrt(
            ((s * s) + (4.0 * ((s * (2.0 * brake_distance / speed_reduction - t)) + (t * t))))) * -0.5
        n = math.floor(.5 - ((t + sqrt_quantity) / s))
        h = 0.5 * n * (n - 1) * speed_reduction * s + n * speed_reduction * t
        assert(h <= brake_distance + 1e-6)
        # compute the additional speed that must be used during deceleration to fix
        # the discrepancy between g and h
        r = (brake_distance - h) / (n * s + t)
        x = n * speed_reduction + r
        assert(x >= 0)
        return x
