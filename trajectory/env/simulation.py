import json
import os
import bisect
from collections import defaultdict
from trajectory.env.vehicles import FSVehicle, FSWrappedRLVehicle, IDMVehicle, RLVehicle, TrajectoryVehicle
from trajectory.env.energy_models import PFM2019RAV4
from trajectory.env.utils import get_last_or
import random
import pickle
import numpy as np
import pandas as pd


class Simulation(object):
    def __init__(self,
                 timestep,
                 enable_lane_changing=True,
                 road_grade=None,
                 downstream_path=None):
        """Simulation object

        timestep: dt in seconds
        trajectory: ITERATOR yielding triples (position, speed, accel)
            that will be used for the first vehicle in the platoon (not spawned if this is None)
        downstream_path: the directory containing relevant downstream information.
            If set to None or the path does not exist, no downstream data is shared.
        """
        self.timestep = timestep
        # vehicles in order, from first in the platoon to last
        self.vehicles = []
        self.vlength = 5

        self.step_counter = 0
        self.time_counter = 0

        self.energy_model = PFM2019RAV4()

        self.data_by_vehicle = defaultdict(lambda: defaultdict(list))

        self.vids = 0

        self.enable_lane_changing = enable_lane_changing
        self.downstream_path = downstream_path

        self.n_cutins = 0
        self.n_cutouts = 0
        self.n_vehicles = []

        # Store downstream information data.
        self._prev_tse = None
        self._tse_obs, self._tse_times = self._init_tse(self.downstream_path)

        self.setup_grade_and_altitude_map(network=road_grade)

    def setup_grade_and_altitude_map(self, network='i24'):
        if network not in ['i24', 'i680']:
            if network is not None:
                print(f"Network '{network}' does not exist. Setting all road grades to 0.")
            self.road_grade_map = lambda x: 0
            self.altitude_map = lambda x: 0
            self.altitude_bounds = [0, 0]
            self.grade_bounds = [0, 0]
            return

        grade_path = os.path.abspath(
            os.path.join(__file__, f'../../../dataset/{network}_road_grade_interp.pkl'))
        with open(grade_path, 'rb') as fp:
            road_grade = pickle.load(fp)
            if network == 'i24':
                self.road_grade_map = lambda x: road_grade['road_grade_map'](x) * np.pi / 180
            if network == 'i680':
                # Need to convert to degrees
                self.road_grade_map = lambda pos: np.rad2deg(np.arctan(road_grade['road_grade_map'](pos)))
            self.grade_bounds = road_grade['bounds']

        altitude_path = os.path.abspath(
            os.path.join(__file__, f'../../../dataset/{network}_altitude_interp.pkl'))
        with open(altitude_path, 'rb') as fp:
            altitude = pickle.load(fp)
            self.altitude_map = altitude['altitude_map']
            self.altitude_bounds = altitude['bounds']

    def get_road_grade(self, veh):
        # Return road grade in degrees
        pos = self.get_data(veh, 'position')[-1]
        if pos < self.grade_bounds[0] or pos > self.grade_bounds[1]:
            return None
        return self.road_grade_map(pos)

    def get_altitude(self, veh):
        # Return altitude in m
        pos = self.get_data(veh, 'position')[-1]
        if pos < self.altitude_bounds[0] or pos > self.altitude_bounds[1]:
            return None
        return self.altitude_map(pos) / 3.2808

    def get_vehicles(self, controller=None):
        if controller is None:
            return self.vehicles
        else:
            return list(filter(lambda veh: veh.controller == controller, self.vehicles))

    def get_platoon(self, veh, k=5):
        # Return the k vehicles behind veh or max num vehicles behind veh
        platoon = []
        for _ in range(k):
            if (follower := veh.follower) is not None and follower.kind != 'av':
                platoon.append(follower)
                veh = follower
            else:
                break
        return platoon

    def add_vehicle(self, controller='idm', kind=None, tags=None, gap=20,
                    initial_speed=None, insert_at_index=None,
                    default_time_headway=1.1, **controller_kwargs):
        """Add a vehicle behind the platoon.

        controller: 'idm' or 'rl' or 'trajectory' (do not use trajectory)
        gap: spawn the vehicle that many meters behind last vehicle in platoon
        insert_at_index: if None, vehicle is appended behind the platoon, if set, set to the index
            where the vehicle should be appended (inserting at index i means the inserted
            vehicle will have i vehicles in front of it in the platoon)
        controller_kwargs: kwargs that will be passed along to the controller constructor
        """
        # get vehicle class corresponding to desired controller
        vehicle_class = {
            'idm': IDMVehicle,
            'fs': FSVehicle,
            'trajectory': TrajectoryVehicle,
            'rl': RLVehicle,
            'rl_fs': FSWrappedRLVehicle,
        }[controller]

        # get ID of leading and following cars (or None if they do not exist)
        idx_leader = len(self.vehicles) - 1 if len(self.vehicles) > 0 else None
        idx_follower = None
        if insert_at_index is not None:
            idx_leader = insert_at_index - 1 if insert_at_index > 0 else None
            idx_follower = insert_at_index if insert_at_index < len(self.vehicles) else None

        # if inputting a gap < 0, defaults to an initial constant time headway of 1.1s
        if gap < 0 and idx_leader is not None:
            leader_speed = self.vehicles[idx_leader].speed
            gap = leader_speed * default_time_headway

        # create vehicle object
        veh = vehicle_class(
            vid=self.vids,
            controller=controller,
            kind=kind,
            tags=tags,
            pos=0 if idx_leader is None else self.vehicles[idx_leader].pos - gap - self.vlength,
            speed=initial_speed if initial_speed is not None else (
                0 if idx_leader is None else self.vehicles[idx_leader].speed),
            accel=0,
            timestep=self.timestep,
            length=self.vlength,
            leader=None if idx_leader is None else self.vehicles[idx_leader],
            follower=None if idx_follower is None else self.vehicles[idx_follower],
            **controller_kwargs)

        # update new neighbors in linked list accordingly
        if idx_follower is not None:
            self.vehicles[idx_follower].leader = veh
        if idx_leader is not None:
            self.vehicles[idx_leader].follower = veh

        # add vehicle to simulation
        self.vids += 1
        if insert_at_index is None:
            self.vehicles.append(veh)
        else:
            self.vehicles.insert(insert_at_index, veh)

        return veh

    def remove_vehicle(self, idx):
        # update leader and follower pointers
        if idx > 0:
            self.vehicles[idx - 1].follower = self.vehicles[idx + 1] if idx + 1 < len(self.vehicles) else None
        if idx + 1 < len(self.vehicles):
            self.vehicles[idx + 1].leader = self.vehicles[idx - 1] if idx > 0 else None

        # delete vehicle
        self.vehicles.pop(idx)

    def run(self, num_steps=None):
        running = True
        i = 0
        while running:
            running = self.step()
            i += 1
            if num_steps is not None and i >= num_steps:
                running = False

    def handle_lane_changes(self):
        # cut-in and cut-out probabilities (between 0 and 1) per 0.1s timestep
        def cutin_proba_fn(space_gap, leader_speed): return \
            (1.9e-2 + -8.975e-4 * space_gap + 1.002e-4 * space_gap * space_gap) / 100.0 if leader_speed <= 25.0 \
            else (-5.068e-3 + 1.347e-3 * space_gap + 8.912e-6 * space_gap * space_gap) / 100.0
        def cutout_proba_fn(leader_speed): return \
            (-8.98e-3 + 8.763e-3 * leader_speed - 2.1e-4 * leader_speed * leader_speed) / 100.0
        # gap ratio (gap of inserted vehicle / gap of ego vehicle) on cut-in
        def gap_ratio_fn(): return min(max(random.gauss(mu=43.9, sigma=21.75) / 100.0, 0.0), 1.0)

        # compute ratio of gained and lost vehicles from the initial count, to balance out cut-ins and cut-outs
        n_vehicles = len(self.vehicles)
        n_vehicles_initially = n_vehicles - self.n_cutins + self.n_cutouts
        ratio_gained = (n_vehicles - n_vehicles_initially) / n_vehicles_initially
        ratio_lost = (n_vehicles_initially - n_vehicles) / n_vehicles_initially
        multiplier_coef = 10.0
        cutin_multipier = np.exp(- multiplier_coef * ratio_gained)
        cutout_multipier = np.exp(- multiplier_coef * ratio_lost)

        min_gap = 1.0  # minimum space gap for cut-ins (in meters)
        min_time_gap = 0.5  # minimum time gap (gap / speed) for cut-ins (in seconds)

        # iterate over the list of vehicles, starting from index 1 (vehicle behind leader,
        # ie second vehicle in the platoon) since we don't want to insert in front of leader
        i = 1
        while i < len(self.vehicles):
            veh = self.vehicles[i]

            # handle cut-ins: first make sure there's enough room to insert a vehicle
            if (gap := veh.get_headway()) > veh.length + 2.0 * min_gap:
                if random.random() <= cutin_proba_fn(gap, veh.leader.speed) * cutin_multipier:
                    gap_ratio = gap_ratio_fn()
                    inserted_speed = random.uniform(veh.speed, veh.leader.speed)

                    # bounds on inserted_gap to respect min_gap and min_time_gap constraints on both
                    # the inserted vehicle and the vehicle behind it (ie the vehicle it cut in front of)
                    min_insertion_gap = max(min_gap, min_time_gap * veh.speed)
                    max_insertion_gap = gap - veh.length - max(min_gap, min_time_gap * inserted_speed)

                    if min_insertion_gap < max_insertion_gap:
                        # compute gap between veh and the inserted vehicle
                        inserted_gap = min_insertion_gap + gap_ratio * (max_insertion_gap - min_insertion_gap)

                        # add vehicle in front of veh
                        self.add_vehicle(
                            controller='idm',
                            kind='human',
                            gap=inserted_gap,
                            initial_speed=inserted_speed,
                            insert_at_index=i)
                        self.n_cutins += 1

                        # increment index to skip newly inserted vehicle in loop
                        i += 1

            # handle cut-outs: first make sure we wouldn't remove an
            # AV or the trajectory leader
            if veh.leader.kind == 'human':
                if random.random() <= cutout_proba_fn(veh.leader.speed) * cutout_multipier:
                    # remove vehicle in front of veh
                    self.remove_vehicle(i - 1)
                    self.n_cutouts += 1

                    # removed a vehicle so decrement index to not skip a vehicle in loop
                    i -= 1

            # move to next vehicle in platoon
            i += 1

    @staticmethod
    def _init_tse(downstream_path):
        """Store traffic state estimation data.

        Parameters
        ----------
        downstream_path : str or None
            the path to the directory containing the relevant traffic state
            estimates. If set to None, no estimates are available.

        Returns
        -------
        dict or None
            A dictionary of the traffic state estimates for the current
            trajectory. This consists of the following terms:

            * segments: the list of the starting positions of different
              estimated segments
            * avg_speed: average speed of every segment at different time
              intervals

            This is set to None if no downstream information is available.
        list of float
            the times when traffic state estimates get updated
        """
        # If no downstream path was specific, or the data does not exist, no
        # data will be available.
        if (downstream_path is None) or not os.path.exists(downstream_path):
            return None, None

        tse = {}

        # Load segment positions.
        with open(os.path.join(downstream_path, "segments.json"), "r") as f:
            tse["segments"] = json.load(f)

        # Load available traffic-state estimation data.
        tse["avg_speed"] = np.genfromtxt(
            os.path.join(downstream_path, "speed.csv"),
            delimiter=",", skip_header=1)[:, 1:]

        # Import times when the traffic state estimate is updated.
        tse_times = sorted(list(pd.read_csv(
            os.path.join(downstream_path, "speed.csv"))["Time"]))
        tse_times = [x - tse_times[0] for x in tse_times]

        return tse, tse_times

    def _get_tse(self):
        """Return the traffic state estimates for this time step.

        Returns
        -------
        dict
            A dictionary of the traffic state estimates for the current time
            step. This consists of the following terms:

            * segments: the list of the starting positions of different
              estimated segments
            * avg_speed: average speed of every segment
        """
        # Find the index of the current observation.
        index = max(bisect.bisect(self._tse_times, self.time_counter) - 1, 0)

        # Return the traffic state estimates corresponding to this time.
        return {
            "segments": self._tse_obs["segments"],
            "avg_speed": self._tse_obs["avg_speed"][index, :],
        }

    def step(self, env):
        self.step_counter += 1
        self.time_counter += self.timestep

        # Collect macroscopic traffic state estimates.
        tse = self._get_tse() if self._tse_obs is not None else None

        if self.enable_lane_changing and self.step_counter < env.horizon:
            # Catch the edge case where lane change happens on last step and then data
            # isn't collected.
            self.handle_lane_changes()

            # if self.step_counter % 1000 == 0:
            #     print(len(self.vehicles), self.n_cutins, self.n_cutouts)
        self.n_vehicles.append(len(self.vehicles))

        return_status = True

        for veh in self.vehicles[::-1]:
            # update vehicles in reverse order assuming the controller is
            # independent of the vehicle behind you. if at some point it is,
            # then new position/speed/accel have to be calculated for every
            # vehicle before applying the changes
            return_status &= veh.step(tse=tse)

        self.collect_data()

        return return_status

    def add_data(self, veh, key, value):
        self.data_by_vehicle[veh.name][key].append(value)
        # TODO(nl) add data by time as well

    def get_data(self, veh, key):
        return self.data_by_vehicle[veh.name][key]

    def collect_data(self, vehicles=None):
        if vehicles is None:
            vehicles = self.vehicles
        for veh in vehicles:
            self.add_data(veh, 'time', round(self.time_counter, 4))
            self.add_data(veh, 'step', self.step_counter)
            self.add_data(veh, 'id', veh.name)
            self.add_data(veh, 'position', veh.pos)
            self.add_data(veh, 'speed', veh.speed)
            self.add_data(veh, 'accel', veh.accel)
            self.add_data(veh, 'headway', veh.get_headway())
            self.add_data(veh, 'leader_speed', veh.get_leader_speed())
            self.add_data(veh, 'speed_difference', None if veh.leader is None else veh.leader.speed - veh.speed)
            self.add_data(veh, 'time_to_collision', veh.get_time_to_collision())
            self.add_data(veh, 'leader_id', None if veh.leader is None else veh.leader.name)
            self.add_data(veh, 'follower_id', None if veh.follower is None else veh.follower.name)
            self.add_data(veh, 'road_grade', 0 if self.get_road_grade(veh) is None else self.get_road_grade(veh))
            self.add_data(veh, 'altitude', self.get_altitude(veh))
            self.add_data(veh, 'instant_energy_consumption',
                          self.energy_model.get_instantaneous_fuel_consumption(veh.accel_no_noise_with_failsafe, veh.speed,
                                                                               self.get_data(veh, 'road_grade')[-1]))
            self.add_data(veh,
                          'total_energy_consumption',
                          get_last_or(self.data_by_vehicle[veh.name]['total_energy_consumption'],
                                      0) + self.data_by_vehicle[veh.name]['instant_energy_consumption'][-1])
            self.add_data(veh, 'total_distance_traveled', veh.pos - self.data_by_vehicle[veh.name]['position'][0])
            self.add_data(veh, 'total_miles', self.data_by_vehicle[veh.name]['total_distance_traveled'][-1] / 1609.34)
            self.add_data(veh, 'total_gallons', self.data_by_vehicle[veh.name]
                          ['total_energy_consumption'][-1] / 3600.0 * self.timestep)
            self.add_data(veh, 'avg_mpg', self.data_by_vehicle[veh.name]['total_miles']
                          [-1] / (self.data_by_vehicle[veh.name]['total_gallons'][-1] + 1e-6))
            self.add_data(veh, 'realized_accel', (veh.prev_speed - veh.speed) / self.timestep)
            self.add_data(veh, 'target_accel_no_noise_no_failsafe', veh.accel_no_noise_no_failsafe)
            self.add_data(veh, 'target_accel_with_noise_no_failsafe', veh.accel_with_noise_no_failsafe)
            self.add_data(veh, 'target_accel_no_noise_with_failsafe', veh.accel_no_noise_with_failsafe)
            self.add_data(veh, 'vdes', veh.fs.v_des if hasattr(veh, 'fs') else -1)
