from collections import defaultdict
from env.vehicles import IDMVehicle, TrajectoryVehicle
from env.energy_models import PFMMidsizeSedan


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

        self.step_counter = -1
        self.time_counter = 0
        
        self.energy_model = PFMMidsizeSedan() 

        self.data_by_time = []
        self.data_by_vehicle = defaultdict(lambda: defaultdict(list)) 

        self.vids = 0


    def add_vehicle(self, controller='idm', gap=20, **controller_kwargs):
        """Add a vehicle behind the platoon.
        
        controller: 'idm' or 'rl' or 'trajectory' (do not use trajectory)
        gap: spawn the vehicle that many meters behind last vehicle in platoon
        controller_kwargs: kwargs that will be passed along to the controller constructor
        """
        vehicle_class = {
            'idm': IDMVehicle,
            'trajectory': TrajectoryVehicle,
        }[controller]

        veh = vehicle_class(
            vid=self.vids,
            name=controller,
            pos=0 if len(self.vehicles) == 0 else self.vehicles[-1].pos - gap - self.vlength,
            speed=0 if len(self.vehicles) == 0 else self.vehicles[-1].speed,
            accel=0,
            timestep=self.timestep,
            length=self.vlength,
            leader=None if len(self.vehicles) == 0 else self.vehicles[-1],
            **controller_kwargs)
        self.vids += 1

        self.vehicles.append(veh)

    def run(self, num_steps=None):
        running = True
        i = 0
        while running:
            running = self.step()
            i += 1
            if num_steps is not None and i >= num_steps:
                running = False

    def step(self):
        if self.step_counter == -1:
            self.collect_data()
        self.step_counter += 1
        self.time_counter += self.timestep

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
        self.data_by_vehicle[veh.id_name][key].append(value)

    def collect_data(self):
        for veh in self.vehicles:          
            self.add_data(veh, 'times', round(self.time_counter, 4))
            self.add_data(veh, 'steps', self.step_counter)
            self.add_data(veh, 'positions', veh.pos)
            self.add_data(veh, 'speeds', veh.speed)
            self.add_data(veh, 'accels', veh.accel)
            self.add_data(veh, 'headways', veh.get_headway())
            self.add_data(veh, 'leader_speeds', veh.get_leader_speed())
            self.add_data(veh, 'speed_differences', None if veh.leader is None else veh.leader.speed - veh.speed)
            self.add_data(veh, 'leader_ids', None if veh.leader is None else veh.leader.id_name)
            self.add_data(veh, 'instant_energy_consumptions', self.energy_model.get_instantaneous_fuel_consumption(veh.accel, veh.speed, 0))
