import json
from pathlib import Path
from args import parse_args_simulate
import importlib
import re
from env.trajectory_env import TrajectoryEnv
from env.simulation import Simulation
from data_loader import DataLoader
from visualize.plotter import Plotter
import numpy as np


# parse command line arguments
args = parse_args_simulate()

# load config file
# TODO right now run python train.py --iters 1 --no_eval to generate a quick CP
# even if not running a RL controller
cp_path = Path(args.cp_path)
with open(cp_path.parent.parent / 'configs.json', 'r') as fp:
    configs = json.load(fp)

# load AV controller
if args.av_controller.lower() == 'rl':
    # retrieve algorithm
    alg_module, alg_class = re.match("<class '(.+)\.([a-zA-Z\_]+)'>", configs['algorithm']).group(1, 2)
    assert (alg_module.split('.')[0] in ['stable_baselines3', 'algos'])
    algorithm = getattr(importlib.import_module(alg_module), alg_class)

    # load checkpoint into model
    model = algorithm.load(cp_path)

    print(f'\nLoaded model checkpoint at {cp_path}\n')
    if args.verbose:
        print(f'trained for {model.num_timesteps} timesteps')
        print(f'algorithm = {alg_module}.{alg_class}')
        print(f'observation space = {model.observation_space}')
        print(f'action space = {model.action_space}')
        print(f'policy = {model.policy_class}')
        print(f'\n{model.policy}\n')

    get_action = lambda state: model.predict(state, deterministic=True)[0]

# load trajectories for evaluation
trajectories = DataLoader().get_all_trajectories()

# TODO for now we're just evaluating on one random trajectory for debugging
for trajectory in [next(trajectories)]:
    # create simulation
    sim = Simulation(timestep=trajectory['timestep'])

    # populate simulation with a trajectory leader
    sim.add_vehicle(controller='trajectory', kind='leader',
        trajectory=zip(trajectory['positions'], trajectory['velocities'], trajectory['accelerations']))
    # an AV
    sim.add_vehicle(controller=args.av_controller, kind='av', gap=args.av_gap, **eval(args.av_kwargs))
    # and a platoon of IDMs
    for _ in range(args.n_idms):
        sim.add_vehicle(controller='idm', kind='platoon', gap=args.idms_gap, **eval(args.idms_kwargs))
    
    # run simulation for the whole trajectory
    sim.run(num_steps=10000)  # TODO remove num_steps to plot the whole trajectory

    # collect and plot results
    for veh in sim.vehicles:
        plotter = Plotter('figs/simulate')
        for k, v in sim.data_by_vehicle[veh.name].items():
            plotter.plot(v, title=k, grid=True)
        plotter.save(f'sim_output_{veh.name}', log='\t')
