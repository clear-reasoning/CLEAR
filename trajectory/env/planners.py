"""Implementations of different speed planners."""
import numpy as np
import pandas as pd
from copy import deepcopy
import scipy.interpolate as spi
import numpy as np
import pandas as pd
import json
import scipy.optimize as opt
from scipy.interpolate import interp1d
from scipy.integrate import quad, odeint
import matplotlib.pyplot as plt
import bisect
import time as timeprofile
import torch


# =========================================================================== #
#                           Abstract speed planner                            #
# =========================================================================== #


class SpeedPlanner(object):

    def get_target(self, x_av, x_seg, v_seg):
        """Return the target speed and headway for every AV.
        Parameters
        ----------
        x_av : np.ndarray
            the position of every automated vehicle (in meters)
        x_seg : np.ndarray
            the position of every segment (in meters)
        v_seg : np.ndarray
            approximated speed at every segment (in m/s)
        Returns
        -------
        list of float
            target speed for every AV
        list of bool
            max headway, used to define the gap setting
        """
        raise NotImplementedError


# =========================================================================== #
#                             Zhe's speed planner                             #
# =========================================================================== #


class Zhe(SpeedPlanner):

    def __init__(self):
        # For uniform smoothing, acts as the smoothing window.
        self.width = 2000.

        # previously stored v(x) values
        self._prev_v_points = None

        # the traffic state estimation data provided in the previous step. If
        # this matches the current estimation, then no need to recompute.
        self.prev_v_seg = None

    def get_target(self, x_av, x_seg, v_seg):
        """See parent class."""
        # positions where we will compute segment speeds across
        #
        # Note that we need to be right before the last segment to compute a
        # non-NaN target speed value.
        x_points = np.linspace(x_seg[0], x_seg[-1]-1, 100)

        if np.all(self.prev_v_seg == v_seg):
            # Nothing has changed. Use old data.
            v_points = self._prev_v_points
        else:
            # Remember next segment.
            self.prev_v_seg = v_seg

            # Compute target speed at every discrete point.
            v_points = [self._uniform(x, x_seg, v_seg) for x in x_points]

            # Memorize this value to avoid redundancies in computation.
            self._prev_v_points = v_points

        # Compute the target speeds at the positions of the AVs.
        v_target = np.interp(x_av, x_points, v_points)

        # Compute the target headways.
        h_target = [True for _ in range(len(x_av))]

        return v_target, h_target

    def _uniform(self, x0, x, z):
        """Perform uniform smoothing across a window of fixed width."""
        # Collect relevant traffic-state info.
        # - ix0: segments in front of the ego vehicle
        # - ix1: final segment, or clipped to estimate congested speeds
        ix0 = next(i for i in range(len(x)) if x[i] >= x0)
        try:
            ix1 = next(j for j in range(ix0 + 2, len(x))
                       if x[j] >= (x0 + self.width)
                       or z[j] - np.mean(z[ix0:j]) > 15) + 1
        except StopIteration:
            ix1 = len(x)

        x = np.array(deepcopy(x[ix0-1:ix1]))
        z = np.array(deepcopy(z[ix0-1:ix1]))

        # Return default speed if behind the target.
        if len(x) == 0:
            return 30.

        # Replace endpoints with a cutoff
        z[0] = z[0] + (z[1]-z[0]) * (x0-x[0]) / (x[1]-x[0])
        x[0] = x0

        if x[-1] > x0 + self.width:
            z[-1] = z[-2] + \
                (z[-1]-z[-2]) * (x0+self.width-x[-2]) / (x[-1]-x[-2])
            x[-1] = x0 + self.width

        area = 0.5 * (z[1:]+z[:-1]) * (x[1:]-x[:-1])
        actual_width = x[-1] - x[0]

        return sum(area) / actual_width


# =========================================================================== #
#                            Arwa's speed planner                             #
# =========================================================================== #


class Arwa(SpeedPlanner):

    def __init__(self):
        # segment discretization provided by solver
        self.x_points = None

        # previously stored v(x) values
        self._prev_v_points = None

        # the traffic state estimation data provided in the previous step. If
        # this matches the current estimation, then no need to recompute.
        self.prev_v_seg = None

    def optimal_control(self, segments, speed):
        time, xl, vl = self.traj_from_speed_profile(segments, speed)
        [sim_time, U_star, X_star, V_star] = self.optimize_traj(time, xl, vl)
        [opt_segments, opt_speed] = self.speed_profile_from_traj(
            sim_time, V_star, xl[0])
        return opt_segments, opt_speed

    ########################################################
    ################# Data conversion ######################
    ########################################################

    def traj_from_speed_profile(self, segments, speed):
        v_of_x = interp1d(segments, speed, kind='linear')
        time = np.zeros_like(segments)
        time[0] = 0
        for i, _ in enumerate(segments[:-1]):
            time[i + 1] = \
                quad(lambda x: 1 / v_of_x(x), segments[i], segments[i + 1])[0]
        return np.cumsum(time), segments, speed

    def speed_profile_from_traj(self, time, speed, x0):
        v_of_t = interp1d(time, speed, "linear", fill_value='extrapolate')
        F = lambda x, t: v_of_t(t)
        position = odeint(F, x0, time)
        return position, speed

    ########################################################
    ############## Optimize Trajectory #####################
    ########################################################

    def optimize_traj(self, time, xl, vl):
        if xl[0] < 0:
            xl = xl + np.abs(xl[0])
        else:
            xl = xl - xl[0]

        ctf = 20
        sim_time = np.arange(time[0], time[-1], .1)
        x0 = [0, -200]
        v0 = [vl[0], vl[0]]

        vl = interp1d(time, vl, kind="linear")
        vl = vl(sim_time)

        xl = interp1d(time, xl, kind="linear")
        xl = xl(sim_time)

        U_0 = self.initial_guess(sim_time, vl, ctf)

        l = 4.5
        eps = 2
        gamma = 500

        lb = -3
        ub = 3

        A, b = self.lc(sim_time[0::ctf], xl[0::ctf], x0[1], v0[1], l, eps, gamma)
        cons = opt.LinearConstraint(A, ub=b, lb=-np.inf)  ## For trust_const method
        #     cons = [{"type": "ineq",
        #              "fun": lambda U: -A @ U + b}]
        # #              "jac": lambda U: -A}]

        res = opt.minimize(self.objective, U_0, args=(sim_time, x0[1], v0[1], ctf),
                           method='trust-constr', jac=self.grad,
                           bounds=(((lb, ub),) * len(U_0)),
                           callback=self.print_opt_progress,
                           # For trust_const method
                           #                  callback= lambda U: print_opt_progress_SLSQP(U, sim_time, x0[1], v0[1], ctf),
                           constraints=cons,
                           tol=1e-8,
                           options={"disp": True, "maxiter": 20})

        U_star = interp1d(sim_time[0::ctf], res.x, 'nearest',
                          fill_value='extrapolate')
        U_star = U_star(sim_time)
        X_star, V_star = self.system_solve(U_star, sim_time, x0[1], v0[1])

        return sim_time, U_star, X_star, V_star

    ########################################################
    ############## Objective function ######################
    ########################################################

    def objective(self, U, time, x0, v0, ctf):
        U_full = interp1d(time[0::ctf], U, 'nearest', fill_value='extrapolate')
        U_full = U_full(time)
        X, V = self.system_solve(U_full, time, x0, v0)

        E = self.simplified_fuel_model(V, U_full)
        z = (max(time) / len(time)) * np.sum(E)
        return z

    @staticmethod
    def system_solve(U, time, x0, v0):
        nt = len(time)
        T = (time[-1] / nt) * np.ones([nt, nt])
        T[0, :] = 0
        T = np.tril(T)
        TU = T @ U
        V = v0 * np.ones(nt) + TU
        X = x0 * np.ones(nt) + v0 * time + T @ TU
        return X, V

    @staticmethod
    def simplified_fuel_model(v, a):
        p0 = 0.014702037699570
        p1 = 0.051202817419139
        p2 = 0.001873784068826
        q0 = 0.0
        q1 = 0.018538661491842
        C0 = 0.245104874164936
        C1 = 0.003891740295745
        C2 = 0.0
        C3 = 3.300502213720286e-05
        beta0 = 0.194462963040005

        v = np.maximum(v, 0)
        fc = (C0 + C1 * v + C2 * v ** 2 + C3 * v ** 3
              + p0 * a + p1 * a * v + p2 * a * v ** 2 +
              q0 * np.maximum(a, 0) ** 2 + q1 * np.maximum(a, 0) ** 2 * v)
        return np.maximum(fc, beta0)

    ########################################################
    ############## Gradient function ######################
    ########################################################

    def grad(self, U, time, x0, v0, ctf):
        U_full = interp1d(time[0::ctf], U, 'nearest', fill_value='extrapolate')
        U_full = U_full(time)
        X, V = self.system_solve(U_full, time, x0, v0)

        p0 = 0.014702037699570
        p1 = 0.051202817419139
        p2 = 0.001873784068826
        q0 = 0.0
        q1 = 0.018538661491842

        PT = 0
        QT = 0
        _, Q = self.back_system_solve(X, V, U_full, time, PT, QT)

        dz = -1 * Q[0::ctf] + (p0 + p1 * V[0::ctf] + p2 * V[0::ctf] ** 2) + (
                    U > 0) * (2 * q0 * U + 2 * q1 * V[0::ctf] * U)
        return dz

    def back_system_solve(self, X, V, U, time, PT, QT):
        V = interp1d(time, V, 'linear', fill_value='extrapolate')
        U = interp1d(time, U, 'nearest', fill_value='extrapolate')
        F = lambda PQ, t: self._F_adjoint(PQ, 0, V(t), U(t))
        PQ = odeint(F, [PT, QT], np.flip(time))
        PQ = np.flip(PQ, 0)
        P = PQ[:, 0]
        Q = PQ[:, 1]
        return P, Q

    @staticmethod
    def _F_adjoint(PQ, x, v, a):
        C0 = 0.245104874164936
        C1 = 0.003891740295745
        C2 = 0.0
        C3 = 3.300502213720286e-5
        p0 = 0.014702037699570
        p1 = 0.051202817419139
        p2 = 0.001873784068826
        q0 = 0.0
        q1 = 0.018538661491842
        P = PQ[0]
        P_dot = 0
        Q_dot = (C1 + 2 * C2 * v + 3 * C3 * v ** 2 + (p1 + 2 * p2 * v) * a + (
                    a > 0) * q1 * a ** 2) - P
        return np.append(P_dot, Q_dot)

    ########################################################
    ############## Constraint function #####################
    ########################################################

    @staticmethod
    def lc(time, xl, x0, v0, l, eps, gamma):
        nt = len(time)
        T = (time[-1] / nt) * np.ones([nt, nt])
        T[0, :] = 0
        T = np.tril(T)
        TT = T @ T
        A = np.vstack((TT, -TT, -T))

        ub = np.concatenate((xl - x0 - v0 * time - l - eps,
                             gamma - xl + x0 + v0 * time + l,
                             v0 * np.ones(nt)))
        return A, ub

    ########################################################
    ################# Misc functions #######################
    ########################################################

    @staticmethod
    def print_opt_progress(U, state):
        print(
            f'F    {state.fun}     grad    {np.linalg.norm(state.grad)}      const violation     {state.constr_violation}')

    def print_opt_progress_SLSQP(self, U, time, x0, v0, ctf):
        fun = self.objective(U, time, x0, v0, ctf)
        print(f'F    {fun}')

    def initial_guess(self, sim_time, vl, ctf):
        U_0 = np.diff(self.moving_average(vl, 60)) / np.diff(sim_time)
        U_0 = np.append(U_0, 0)
        return U_0[0::ctf]

    @staticmethod
    def moving_average(s, w):
        ma_s = np.cumsum(s)
        ma_s[w:] = ma_s[w:] - ma_s[:-w]
        ma_s[w - 1:] = ma_s[w - 1:] / w
        ma_s[:w - 1] = ma_s[:w - 1] / np.arange(1, w)
        return ma_s

    def get_target(self, x_av, x_seg, v_seg):
        """See parent class."""

        if np.all(self.prev_v_seg == v_seg):
            # Nothing has changed. Use old data.
            v_points = self._prev_v_points
        else:
            # Remember next segment.
            self.prev_v_seg = deepcopy(v_seg)

            x_seg = np.append(x_seg, x_seg[-1] + 2000)
            v_seg = np.append(v_seg, v_seg[-1])

            # Compute target speed at every discrete point.
            self.x_points, v_points = self.optimal_control(x_seg, v_seg)
            self.x_points = self.x_points.flatten()

            x_final = self.x_points[-1]

            ix = bisect.bisect(self.x_points, x_final - 2000)

            self.x_points = self.x_points[:ix]
            v_points = v_points[:ix]

            # Memorize this value to avoid redundancies in computation.
            self._prev_v_points = v_points

        # Compute the target speeds at the positions of the AVs.
        v_target = np.interp(x_av, self.x_points, v_points)

        # Compute the target headways.
        h_target = [True for _ in range(len(x_av))]

        return v_target, h_target


# =========================================================================== #
#                             Han's speed planner                             #
# =========================================================================== #


class Han(SpeedPlanner):

    def __init__(self):
        # previously stored target profiles
        self.target_speed_profile = []
        self.max_headway_profile = []

        # previous inrix for bn identification
        self.inrix_history = []

        # the traffic state estimation data provided in the previous step. If
        # this matches the current estimation, then no need to recompute.
        self.prev_v_seg = None

    def get_target(self, x_av, x_seg, v_seg):
        """See parent class."""
        if np.all(self.prev_v_seg == v_seg):
            # Nothing has changed. Use old data.
            pass
        else:
            # Remember next segment.
            self.prev_v_seg = v_seg

            new_dx = 100  # 100 meter resolution on resampled profile
            x_range = np.arange(-20000, 21000, new_dx)

            # resample to finer spatial grid
            speed = np.interp(x_range, x_seg, v_seg)

            # apply gaussian smoothing
            gaussian_smoothed_speed = np.array([
                self._gaussian(x_range[i], x_range, np.array(speed), sigma=250)
                for i in range(len(x_range))])

            self.inrix_history.append(v_seg)
            bn_pos, stand_bn_pos, move_bn_pos = self.get_bn_loc_by_type(
                x_seg, v_seg)

            target_speed_profile, max_headway_profile = \
                self.get_target_profiles_with_equilibrium_mbn(
                    x_range,
                    gaussian_smoothed_speed,
                    stand_bn_pos=stand_bn_pos,
                    move_bn_pos=move_bn_pos,
                )

            self.target_speed_profile = target_speed_profile
            self.max_headway_profile = max_headway_profile

        # Get target speed and max headway.
        target_speed = self.get_target_by_position(
            self.target_speed_profile, x_av, float)
        max_headway = self.get_target_by_position(
            self.max_headway_profile, x_av, bool)

        return target_speed, max_headway

    @staticmethod
    def get_target_by_position(profile, pos, dtype=float):
        """Get target speed by position."""
        prop_speed = 4.2
        time_offset = 180
        if dtype == bool:
            kind = "previous"
        else:
            kind = "linear"
        interp = spi.interp1d(
            profile[0] - time_offset * prop_speed,
            profile[1],
            kind=kind,
            fill_value="extrapolate")
        return interp(pos)

    @staticmethod
    def _gaussian(x0, x, z, sigma):
        """Perform a kernel smoothing operation on future average speeds."""
        ix0 = 0
        ix1 = len(x)
        x = x[ix0:ix1]
        z = z[ix0:ix1]

        densities = (
            1 / (np.sqrt(2 * np.pi) * sigma) *
            np.exp(-np.square(x - x0) / (2 * sigma ** 2)))

        densities = densities / sum(densities)

        return sum(densities * z)

    @staticmethod
    def get_target_profiles_with_fixed_start_point(
            x,
            speed,
            speed_thresh=25,
            start_loc: float = 0,
            bn_loc: float = None,
            over_decel_rate=1.0):
        """Get target profile with constant decel."""
        if bn_loc is None:
            x_bn = x[speed < speed_thresh]
            if len(x_bn) > 0:
                bn_loc = x_bn.min()
            else:
                bn_loc = start_loc + 1
        target_speed_profile = (
            np.array([-20000, start_loc, bn_loc, 15000, 20000]),
            np.array([33, 33, speed_thresh*over_decel_rate, 33, 33]),
            np.array(1))

        headway_profile = tuple([x, [True] * len(speed), np.array(1)])

        return target_speed_profile, headway_profile

    def get_target_profiles_with_equilibrium_mbn(
            self,
            x,
            speed,
            stand_bn_pos=[],
            move_bn_pos=[],
            v_q=27,
            k_c=0.048,
            decel=-0.1,
            aggresiveness=0.2):
        target_speed_df = pd.DataFrame()
        dx = x[1] - x[0]

        if len(stand_bn_pos) > 0:
            speed_stand = deepcopy(speed)
            bn_loc = stand_bn_pos[0]

            speed_bn = speed[x < bn_loc]
            speed_bn = speed_bn[speed_bn < v_q]
            n_bn = self.get_k_of_v(speed_bn) * dx
            v_sl = speed_bn.min() * 1.2
            l_decel = (v_sl ** 2 - v_q ** 2) / (2 * decel)

            l_buffer = n_bn.sum() / k_c

            start_loc = bn_loc - l_buffer - l_decel

            next_speed = speed_stand[-1]

            if start_loc <= 0:
                start_loc = 0
                if bn_loc > l_decel:
                    l_buffer = bn_loc - l_decel

                else:
                    l_buffer = 0
                    l_decel = bn_loc
                    decel = (v_sl ** 2 - v_q ** 2) / (2 * l_decel)

            for i in range(len(x) - 1, 0, -1):
                decel_speed = speed_stand[i]
                if (x[i] >= start_loc) & (x[i] <= bn_loc - l_buffer):
                    decel_speed = np.sqrt(next_speed ** 2 - dx * decel * 2.0)

                elif (x[i] <= bn_loc) & (x[i] > bn_loc - l_buffer):
                    decel_speed = v_sl

                if decel_speed < speed_stand[i]:
                    speed_stand[i] = decel_speed
                next_speed = speed_stand[i]
            target_speed_df['stand_bn'] = speed_stand

        if len(move_bn_pos) > 0:
            for wave_loc in move_bn_pos:
                bn_loc = wave_loc
                move_speed = deepcopy(speed)
                v_wave = move_speed[x == int(wave_loc/dx)*dx]
                l_decel = (v_wave ** 2 - v_q ** 2) / (2 * decel)

                start_loc = bn_loc - l_decel

                next_speed = move_speed[-1]

                if start_loc <= 0:
                    start_loc = 0
                    l_decel = bn_loc
                    decel = (v_wave ** 2 - v_q ** 2) / (2 * l_decel)

                for i in range(len(x) - 1, 0, -1):
                    decel_speed = move_speed[i]
                    if (x[i] >= start_loc) & (x[i] <= bn_loc):
                        decel_speed = np.sqrt(
                            next_speed ** 2 - dx * decel * 2.0)

                    if decel_speed < move_speed[i]:
                        move_speed[i] = decel_speed
                    next_speed = move_speed[i]
                target_speed_df[f'{int(wave_loc)}'] = move_speed

        target_speed_df['raw'] = speed
        target_speed_df['target_raw'] = target_speed_df.apply(
            lambda x: x.min(), axis=1)

        target_speed_df['agrsv_con'] = speed * (1 - aggresiveness)
        target_speed_df['target_soft'] = target_speed_df.apply(
            lambda x: max(x['target_raw'], x['agrsv_con']), axis=1)

        target_speed_df['headway'] = target_speed_df.apply(
            lambda x: x['target_soft'] < x['raw'], axis=1)

        target_speed = np.array(target_speed_df['target_soft'].values)
        headway_profile = np.array(target_speed_df['headway'].values)

        return tuple([x, target_speed, np.array(1)]), \
            tuple([x, headway_profile, np.array(1)])

    @staticmethod
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
        return tuple(
            [x, speed, np.array(1)]), tuple([x, headway_profile, np.array(1)])

    @staticmethod
    def simlified_inverse_pems_bn_identify(inrix, v_diff=4):
        """Identify bottleneck with simplified inverse PeMS."""
        bn_inrix = deepcopy(inrix)
        for i in range(len(inrix)-1, 0, -1):
            if_bn = False
            if inrix[i] <= 60:
                if inrix[i] - inrix[i-1] > v_diff:
                    if_bn = True
                    bn_inrix[i] = False
            bn_inrix[i-1] = if_bn
        bn_inrix[0] = False
        bn_inrix[len(inrix)-1] = False

        return bn_inrix

    def get_bn_loc_by_type(self, x_seg, v_seg, prev_time=5, stand_time=4):
        bn_loc = self.simlified_inverse_pems_bn_identify(v_seg)

        # if bn_inrix.sum()>0 & len(veh.tse_log.columns) > prev_time:
        if len(self.inrix_history) > prev_time:
            prev_inrix = self.inrix_history[-prev_time:]
            prev_bn = [
                self.simlified_inverse_pems_bn_identify(i) for i in prev_inrix]
            stand_bn_loc = sum(prev_bn) > stand_time
            move_bn_loc = (bn_loc - stand_bn_loc) > 0
        else:
            move_bn_loc = bn_loc
            stand_bn_loc = bn_loc - move_bn_loc

        bn_loc = bn_loc > 0
        stand_bn_loc = stand_bn_loc > 0
        move_bn_loc = move_bn_loc > 0

        # # to test bn scenario
        # stand_bn_loc = bn_loc
        # move_bn_loc = [False for _ in stand_bn_loc]

        bn_pos = np.array(x_seg)[list(bn_loc)]
        stand_bn_pos = np.array(x_seg)[list(stand_bn_loc)]
        move_bn_pos = np.array(x_seg)[list(move_bn_loc)]

        return bn_pos, stand_bn_pos, move_bn_pos

    @staticmethod
    def get_k_of_v(v):
        """Get the density of the speed using IDM equilibrium."""
        v0 = 35
        T = 1.24
        delta = 4
        s0 = 2
        # a = 1.3
        # b = 2
        s = (s0 + T * v) / (1 - (v / v0) ** delta) ** (1 / 2)
        k = 1 / (s + 4)  # Assume average veh len is 4 meters
        return k


# =========================================================================== #
#                           Han's RL speed planner                            #
# =========================================================================== #


class Han_rl(SpeedPlanner):

    def __init__(self):
        # previously stored target profiles
        self.target_speed_profile = []
        self.max_headway_profile = []

        # previous inrix for bn identification
        self.inrix_history = []

        # the traffic state estimation data provided in the previous step. If
        # this matches the current estimation, then no need to recompute.
        self.prev_v_seg = None

        # load RL agent
        self.rl_vsl = torch.load('trajectory/sim/han_rl.pth').float()


    def get_target(self, x_av, x_seg, v_seg):
        """See parent class."""
        if np.all(self.prev_v_seg == v_seg):
            # Nothing has changed. Use old data.
            pass
        else:
            # Remember next segment.
            self.prev_v_seg = v_seg

            new_dx = 100  # 100 meter resolution on resampled profile
            x_range = np.arange(-20000, 21000, new_dx)

            # resample to finer spatial grid
            speed = np.interp(x_range, x_seg, v_seg)

            # apply gaussian smoothing
            gaussian_smoothed_speed = np.array([
                self._gaussian(x_range[i], x_range, np.array(speed), sigma=250)
                for i in range(len(x_range))])

            self.inrix_history.append(v_seg)
            bn_pos, stand_bn_pos, move_bn_pos = self.get_bn_loc_by_type(
                x_seg, v_seg)

            target_speed_profile, max_headway_profile = \
                self.get_target_profiles_with_equilibrium_mbn(
                    x_range,
                    gaussian_smoothed_speed,
                    raw_v_seg=v_seg,
                    stand_bn_pos=stand_bn_pos,
                    move_bn_pos=move_bn_pos,
                )

            self.target_speed_profile = target_speed_profile
            self.max_headway_profile = max_headway_profile

        # Get target speed and max headway.
        target_speed = self.get_target_by_position(
            self.target_speed_profile, x_av, float)
        max_headway = self.get_target_by_position(
            self.max_headway_profile, x_av, bool)

        return target_speed, max_headway

    @staticmethod
    def get_target_by_position(profile, pos, dtype=float):
        """Get target speed by position."""
        prop_speed = 4.2
        time_offset = 180
        if dtype == bool:
            kind = "previous"
        else:
            kind = "linear"
        interp = spi.interp1d(
            profile[0] - time_offset * prop_speed,
            profile[1],
            kind=kind,
            fill_value="extrapolate")
        return interp(pos)

    @staticmethod
    def _gaussian(x0, x, z, sigma):
        """Perform a kernel smoothing operation on future average speeds."""
        ix0 = 0
        ix1 = len(x)
        x = x[ix0:ix1]
        z = z[ix0:ix1]

        densities = (
            1 / (np.sqrt(2 * np.pi) * sigma) *
            np.exp(-np.square(x - x0) / (2 * sigma ** 2)))

        densities = densities / sum(densities)

        return sum(densities * z)

    @staticmethod
    def get_target_profiles_with_fixed_start_point(
            x,
            speed,
            speed_thresh=25,
            start_loc: float = 0,
            bn_loc: float = None,
            over_decel_rate=1.0):
        """Get target profile with constant decel."""
        if bn_loc is None:
            x_bn = x[speed < speed_thresh]
            if len(x_bn) > 0:
                bn_loc = x_bn.min()
            else:
                bn_loc = start_loc + 1
        target_speed_profile = (
            np.array([-20000, start_loc, bn_loc, 15000, 20000]),
            np.array([33, 33, speed_thresh*over_decel_rate, 33, 33]),
            np.array(1))

        headway_profile = tuple([x, [True] * len(speed), np.array(1)])

        return target_speed_profile, headway_profile

    def get_target_profiles_with_equilibrium_mbn(
            self,
            x, speed, raw_v_seg, stand_bn_pos=[], move_bn_pos=[], v_q=27, k_c=0.064, c_bn=2000 / 3600, over_decel_rate=1,
            decel=-0.1,
            aggresiveness=0.2):
        # headway_profile = [True] * len(speed)
        target_speed_df = pd.DataFrame()
        dx = x[1] - x[0]

        # Zhe's approach
        speed_for_avg = np.append(speed, np.ones(50) * speed[-1])
        zhe_tsp = [np.mean([speed_for_avg[i + j] for j in range(50)]) for i in range(len(speed))]

        if len(stand_bn_pos) > 0:

            speed_stand = deepcopy(speed)
            bn_loc = stand_bn_pos[0]

            speed_bn = speed[x < bn_loc]
            speed_bn = speed_bn[speed_bn < v_q]
            n_bn = self.get_k_of_v(speed_bn) * dx
            # q_bn = min(c_bn, get_k_of_v(speed_bn.min()) * speed_bn.min())
            # v_sl = q_bn / k_c * over_decel_rate
            # v_sl = self.get_v_of_k(k_c)

            action = self.rl_vsl(torch.tensor(raw_v_seg/50).float()).probs.argmax()
            v_sl = np.array((0.8 + action/10 * 0.6) * min(raw_v_seg) * 50)
            # print(v_sl)
            l_decel = (v_sl ** 2 - v_q ** 2) / (2 * decel)

            l_buffer = n_bn.sum() / k_c

            start_loc = bn_loc - l_buffer - l_decel

            next_speed = speed_stand[-1]

            if start_loc <= 0:
                start_loc = 0
                if bn_loc > l_decel:
                    l_buffer = bn_loc - l_decel

                else:
                    l_buffer = 0
                    l_decel = bn_loc
                    decel = (v_sl ** 2 - v_q ** 2) / (2 * l_decel)

            for i in range(len(x) - 1, 0, -1):
                decel_speed = speed_stand[i]
                if (x[i] >= start_loc) & (x[i] <= bn_loc - l_buffer):
                    decel_speed = np.sqrt(next_speed ** 2 - dx * decel * 2.0)

                elif (x[i] <= bn_loc) & (x[i] > bn_loc - l_buffer):
                    decel_speed = v_sl

                if decel_speed < speed_stand[i]:
                    speed_stand[i] = decel_speed
                next_speed = speed_stand[i]
            target_speed_df['stand_bn'] = speed_stand

        target_speed_df['raw'] = zhe_tsp
        target_speed_df['target_raw'] = target_speed_df.apply(lambda x: x.min(), axis=1)

        target_speed_df['agrsv_con'] = speed * (1 - aggresiveness)
        target_speed_df['agrsv_con_2'] = speed - 6.7056
        target_speed_df['target_soft'] = target_speed_df.apply(
            lambda x: max(x['target_raw'], x['agrsv_con'], x['agrsv_con_2']), axis=1)

        target_speed_df['headway'] = target_speed_df.apply(lambda x: x['target_soft'] < x['raw'], axis=1)

        target_speed = np.array(target_speed_df['target_soft'].values)
        headway_profile = np.array(target_speed_df['headway'].values)

        return tuple([x, target_speed, np.array(1)]), tuple([x, headway_profile, np.array(1)])

    @staticmethod
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
        return tuple(
            [x, speed, np.array(1)]), tuple([x, headway_profile, np.array(1)])

    @staticmethod
    def simlified_inverse_pems_bn_identify(inrix, v_diff=4):
        """Identify bottleneck with simplified inverse PeMS."""
        bn_inrix = deepcopy(inrix)
        for i in range(len(inrix)-1, 0, -1):
            if_bn = False
            if inrix[i] <= 60:
                if inrix[i] - inrix[i-1] > v_diff:
                    if_bn = True
                    bn_inrix[i] = False
            bn_inrix[i-1] = if_bn
        bn_inrix[0] = False
        bn_inrix[len(inrix)-1] = False

        return bn_inrix

    def get_bn_loc_by_type(self, x_seg, v_seg, prev_time=5, stand_time=4):
        bn_loc = self.simlified_inverse_pems_bn_identify(v_seg)

        # if bn_inrix.sum()>0 & len(veh.tse_log.columns) > prev_time:
        if len(self.inrix_history) > prev_time:
            prev_inrix = self.inrix_history[-prev_time:]
            prev_bn = [
                self.simlified_inverse_pems_bn_identify(i) for i in prev_inrix]
            stand_bn_loc = sum(prev_bn) > stand_time
            move_bn_loc = (bn_loc - stand_bn_loc) > 0
        else:
            move_bn_loc = bn_loc
            stand_bn_loc = bn_loc - move_bn_loc

        bn_loc = bn_loc > 0
        stand_bn_loc = stand_bn_loc > 0
        move_bn_loc = move_bn_loc > 0

        # # to test bn scenario
        # stand_bn_loc = bn_loc
        # move_bn_loc = [False for _ in stand_bn_loc]

        bn_pos = np.array(x_seg)[list(bn_loc)]
        stand_bn_pos = np.array(x_seg)[list(stand_bn_loc)]
        move_bn_pos = np.array(x_seg)[list(move_bn_loc)]

        return bn_pos, stand_bn_pos, move_bn_pos

    @staticmethod
    def get_k_of_v(v):
        k_max = 0.147  # veh / m
        v_max = 31.2
        return k_max * (1 - v / v_max)

    @staticmethod
    def get_v_of_k(k):
        k_max = 0.147  # veh / m
        v_max = 31.2
        return v_max * (1 - k / k_max)


class rlplanner_ppo(torch.nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256):
        super(rlplanner_ppo, self).__init__()

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(*input_dims, fc1_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(fc1_dims, fc2_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(fc2_dims, n_actions),
            torch.nn.Softmax(dim=-1)
        )

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = torch.distributions.categorical.Categorical(dist)
        return dist
