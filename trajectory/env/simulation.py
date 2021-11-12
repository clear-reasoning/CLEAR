from collections import defaultdict
from trajectory.env.vehicles import FSVehicle, FSWrappedRLVehicle, IDMVehicle, RLVehicle, TrajectoryVehicle
from trajectory.env.energy_models import PFM2019RAV4
from trajectory.env.utils import get_last_or
import random


class Simulation(object):
    def __init__(self, timestep):
        """Simulation object

        timestep: dt in seconds
        trajectory: ITERATOR yielding triples (position, speed, accel)
            that will be used for the first vehicle in the platoon (not spawned if this is None)
        """
        self.timestep = timestep
        # vehicles in order, from first in the platoon to last
        self.vehicles = []
        self.vlength = 5

        self.step_counter = 0
        self.time_counter = 0

        self.energy_model = PFM2019RAV4()

        self.data_by_time = []
        self.data_by_vehicle = defaultdict(lambda: defaultdict(list))

        self.vids = 0

    def get_vehicles(self, controller=None):
        if controller is None:
            return self.vehicles
        else:
            return list(filter(lambda veh: veh.controller == controller, self.vehicles))

    def add_vehicle(self, controller='idm', kind=None, tags=None, gap=20, initial_speed=None, insert_at_index=None, **controller_kwargs):
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
            'fs': FSVehicle
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
            gap = leader_speed * 1.1

        # create vehicle object
        veh = vehicle_class(
            vid=self.vids,
            controller=controller,
            kind=kind,
            tags=tags,
            pos=0 if idx_leader is None else self.vehicles[idx_leader].pos - gap - self.vlength,
            speed=initial_speed if initial_speed is not None else (0 if idx_leader is None else self.vehicles[idx_leader].speed),
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
        
        # add vehicle to simulation
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

    def step(self):
        self.step_counter += 1
        self.time_counter += self.timestep

        ## Lane changing

        # TODO(nl) create vehicle types fo that we can add another vehicle from that type from sim
        # without needing to know human controller or human kwargs defined in the environment
            # pos=0 if idx_leader is None else self.vehicles[idx_leader].pos - gap - self.vlength,
            # speed=initial_speed if initial_speed is not None else (0 if idx_leader is None else self.vehicles[idx_leader].speed),
        avs_idx = []
        for i, veh in enumerate(self.vehicles):
            if veh.kind == 'av':
                avs_idx.append((i, veh))

        for i, veh in avs_idx:
            if (s := veh.get_headway()) > 5:
                if veh.leader.speed <= 25:
                    cutin_proba = -8.975e-4 * s + 1.002e-4 * s * s
                else:
                    cutin_proba = 1.347e-3 * s + 8.912e-6 * s * s
                if random.random() <= cutin_proba:
                    # gap_ratio = new_veh.front_gap / av.gap_before_insert
                    gap_ratio = random.gauss(mu=43.9, sigma=21.75) 
                    # TODO(nl) handle boundaries better and make sure we're not inserting on a collision
                    gap_ratio = min(max(gap_ratio, 5), 2 * 43.9 - 5) 
                    gap_ratio = gap_ratio / 100.0

                    self.add_vehicle(
                        controller='idm', 
                        kind='human', 
                        gap=gap_ratio * s, 
                        initial_speed=(veh.speed + veh.leader.speed) / 2.0, 
                        insert_at_index=i)

                    # update AV index, used by cutout computations
                    i += 1  

            v = veh.leader.speed
            cutout_proba = 8.763e-3 * v - 2.1e-4 * v * v
            if random.random() <= cutout_proba:
                # TODO(nl) if leader is trajectory vehicle, we can't remove it
                # maybe instead shift its position to double the gap or something similar
                if veh.leader.kind != 'leader' and i != 0:
                    self.remove_vehicle(i - 1)

        ## Stepping

        for veh in self.vehicles[::-1]:
            # update vehicles in reverse order assuming the controller is
            # independant of the vehicle behind you. if at some point it is,
            # then new position/speed/accel have to be calculated for every
            # vehicle before applying the changes
            status = veh.step()
            if status is False:
                return False

        self.collect_data()

        return True

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
            self.add_data(veh, 'leader_id', None if veh.leader is None else veh.leader.name)
            self.add_data(veh, 'follower_id', None if veh.follower is None else veh.follower.name)
            self.add_data(veh, 'instant_energy_consumption', self.energy_model.get_instantaneous_fuel_consumption(veh.accel, veh.speed, 0))
            self.add_data(veh, 'total_energy_consumption', get_last_or(self.data_by_vehicle[veh.name]['total_energy_consumption'], 0) + self.data_by_vehicle[veh.name]['instant_energy_consumption'][-1])
            self.add_data(veh, 'total_distance_traveled', veh.pos - self.data_by_vehicle[veh.name]['position'][0])
            self.add_data(veh, 'total_miles', self.data_by_vehicle[veh.name]['total_distance_traveled'][-1] / 1609.34)
            self.add_data(veh, 'total_gallons', self.data_by_vehicle[veh.name]['total_energy_consumption'][-1] / 3600.0 * self.timestep)
            self.add_data(veh, 'avg_mpg', self.data_by_vehicle[veh.name]['total_miles'][-1] / (self.data_by_vehicle[veh.name]['total_gallons'][-1] + 1e-6))
            self.add_data(veh, 'realized_accel', (veh.prev_speed - veh.speed) / self.timestep)
            self.add_data(veh, 'target_accel_no_noise_no_failsafe', veh.accel_no_noise_no_failsafe)
            self.add_data(veh, 'target_accel_with_noise_no_failsafe', veh.accel_with_noise_no_failsafe)
            self.add_data(veh, 'target_accel_no_noise_with_failsafe', veh.accel_no_noise_with_failsafe)
            self.add_data(veh, 'vdes', veh.fs.v_des if hasattr(veh, 'fs') else -1)
