from env.trajectory_env import TrajectoryEnv, DISTANCE_SCALE, SPEED_SCALE
from env.idm import IDMController
from env.TimeHeadwayFollowerStopper import TimeHeadwayFollowerStopper

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


if __name__ == '__main__':
    n_rollouts = 1
    controller = 'fs'
    metrics = defaultdict(list)

    env_config = {
        'max_accel': 1.5,
        'max_decel': 3.0,
        'horizon': 5000,
        'min_speed': 0,
        'max_speed': 40,
        'max_headway': 70,
        'whole_trajectory': True,  # ignores horizon if True
    }

    env = TrajectoryEnv(env_config)
    idm = IDMController(a=env.max_accel, b=env.max_decel)
    fs = TimeHeadwayFollowerStopper(max_accel=env.max_accel, max_deaccel=env.max_decel)

    for k in range(n_rollouts):
        if k % 10 == 0:
            print(f'Rollout {k+1}/{n_rollouts}')

        state = env.reset()
        done = False
        total_reward = 0
        total_distance = 0
        total_energy = 0

        times = [0]
        positions = {
            'leader': [env.leader_positions[env.traj_idx]],
            'av': [env.av['pos']],
            **{f'idm_{k}': [env.idm_followers[k]['pos']] for k in range(len(env.idm_followers))}
        }
        speeds = {
            'leader': [env.leader_speeds[env.traj_idx]],
            'av': [env.av['speed']],
            **{f'idm_{k}': [env.idm_followers[k]['speed']] for k in range(len(env.idm_followers))}
        }
        accels = {
            'av': [0],
            **{f'idm_{k}': [0] for k in range(len(env.idm_followers))}
        }

        i = 0
        while not done:
            if i % 1000 == 0:
                print(i)
            i += 1
            if controller == 'idm':
                accel = idm.get_accel(state[0] * SPEED_SCALE, state[1] * SPEED_SCALE, state[2] * DISTANCE_SCALE)
            elif controller == 'fs':
                fs.v_des =  np.mean(env.leader_speeds[env.traj_idx:env.traj_idx+100])  #state[1] * SPEED_SCALE
                accel = fs.get_accel(state[0] * SPEED_SCALE, state[1] * SPEED_SCALE, state[2] * DISTANCE_SCALE, env)
            else:
                raise ValueError

            state, reward, done, _ = env.step(accel)
            total_reward += reward

            times.append(times[-1] + env.time_step)

            positions['leader'].append(env.leader_positions[env.traj_idx])
            positions['av'].append(env.av['pos'])
            for k in range(len(env.idm_followers)):
                positions[f'idm_{k}'].append(env.idm_followers[k]['pos'])

            speeds['leader'].append(env.leader_speeds[env.traj_idx])
            speeds['av'].append(env.av['speed'])
            for k in range(len(env.idm_followers)):
                speeds[f'idm_{k}'].append(env.idm_followers[k]['speed'])

            accels['av'].append(env.av['last_accel'])
            for k in range(len(env.idm_followers)):
                accels[f'idm_{k}'].append(env.idm_followers[k]['last_accel'])

            for car in [env.av]: #, *env.idm_followers]:
                total_distance += car['speed']
                total_energy += env.energy_model.get_instantaneous_fuel_consumption(car['last_accel'], car['speed'], grade=0)

        metrics['rewards'].append(total_reward)
        metrics['mpg'].append(
            (total_distance / 1609.34) / (total_energy / 3600 + 1e-6)
        )

    # compute avg mpg
    avg_rwd = sum(metrics['rewards']) / len(metrics['rewards'])
    avg_mpg = sum(metrics['mpg']) / len(metrics['mpg'])

    print(f'Average reward over {n_rollouts} rollouts with {controller} controller: ', avg_rwd)
    print(f'Average mpg over {n_rollouts} rollouts with {controller} controller: ', avg_mpg)

    if True:
        plt.figure()
        for k in positions:
            plt.plot(times, positions[k], label=k)
        plt.ylabel('Position (m)')
        plt.xlabel('Time (s)')
        plt.title('Positions of leader, AV and IDMs if AV uses an IDM controller')
        plt.legend()
        plt.savefig('figs/test_env/positions.png')

        plt.figure()
        for k in speeds:
            plt.plot(times, speeds[k], label=k)
        plt.ylabel('Speed (m/s)')
        plt.xlabel('Time (s)')
        plt.title('Speeds of leader, AV and IDMs if AV uses an IDM controller')
        plt.legend()
        plt.savefig('figs/test_env/speeds.png')

        plt.figure()
        for k in accels:
            plt.plot(times, accels[k], label=k)
        plt.ylabel('Accel (m/s^2)')
        plt.xlabel('Time (s)')
        plt.title('Accels of leader, AV and IDMs if AV uses an IDM controller')
        plt.legend()
        plt.savefig('figs/test_env/accels.png')