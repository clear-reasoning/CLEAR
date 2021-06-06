from env.trajectory_env import TrajectoryEnv
from env.accel_controllers import IDMController, TimeHeadwayFollowerStopper

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
from env.failsafes import safe_velocity

PLOT_HEATMAPS = False
PLOT_PLOTS = False

if __name__ == '__main__':
    controller = 'idm'

    custom_cp_path = None # './checkpoints/01_beats_idm_03Jun21/checkpoints/iter_300_7680000steps'

    metrics = defaultdict(list)

    os.makedirs('figs/test_env/', exist_ok=True)

    env_config = {
        'max_accel': 1.5,
        'max_decel': 3.0,
        'min_speed': 0,
        'max_speed': 40,
        'max_headway': 120,
        'whole_trajectory': True,
        'use_fs': False,
    }

    env = TrajectoryEnv(env_config)
    idm = IDMController(a=env.max_accel, b=env.max_decel)
    fs = TimeHeadwayFollowerStopper(max_accel=env.max_accel, max_deaccel=env.max_decel)

    if custom_cp_path is not None:
        from stable_baselines3 import PPO
        # from stable
        from stable_baselines3.common.policies import ActorCriticPolicy
        model = PPO.load(custom_cp_path, env=env) #, custom_objects={'policy_class': ActorCriticPolicy})

    state = env.reset()

    s = env.unnormalize_state(state)
    fs.v_des =  s['leader_speed']

    dmax = 0

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
        # if i % 1000 == 0:
        #     print(i)
        i += 1
        if controller == 'idm':
            s = env.unnormalize_state(state)
            accel = idm.get_accel(s['speed'], s['leader_speed'], s['headway'], env.time_step)
        elif controller == 'fs_leader':
            s = env.unnormalize_state(state)
            dmax = max(dmax, abs(fs.v_des - s['leader_speed']))  # dmax = 1.3933480995153147
            fs.v_des = s['leader_speed']
            accel = fs.get_accel(s['speed'], s['leader_speed'], s['headway'], env.time_step)
        elif controller == 'fs_leader_every':
            s = env.unnormalize_state(state)
            if i % 30 == 0:
                fs.v_des =  s['leader_speed']
            accel = fs.get_accel(s['speed'], s['leader_speed'], s['headway'], env.time_step)
        elif controller == 'fs_leader_incremental':
            s = env.unnormalize_state(state)
            delta_max = 0.05
            fs.v_des += min(max(s['leader_speed'] - fs.v_des, -delta_max), delta_max)
            accel = fs.get_accel(s['speed'], s['leader_speed'], s['headway'], env.time_step)
        elif controller == 'fs_leader_late':
            s = env.unnormalize_state(state)
            delta_max = 1
            fs.v_des = min(max(fs.v_des, s['leader_speed'] - delta_max), s['leader_speed'] + delta_max)
            accel = fs.get_accel(s['speed'], s['leader_speed'], s['headway'], env.time_step)
        elif controller == 'fs_fixed':
            s = env.unnormalize_state(state)
            fs.v_des = 50
            accel = fs.get_accel(s['speed'], s['leader_speed'], s['headway'], env.time_step)
        elif controller == 'constant_accel':
            accel = 1.0
        elif controller == 'custom_cp_path':
            accel, _ = model.predict(state, deterministic=True)
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

    print(f'Avg rwd ', round(avg_rwd, 2))
    print(f'Avg mpg ', round(avg_mpg, 2))

    N = len(times)

    headways = {
        'leader': [-1] * N,
        'av': [positions['leader'][i] - positions['av'][i] for i in range(N)],
        'idm_0': [positions['av'][i] - positions['idm_0'][i] for i in range(N)],
        **{ f'idm_{k}': [positions[f'idm_{k-1}'][i] - positions[f'idm_{k}'][i] for i in range(N)] for k in range(1, len(env.idm_followers)) },
    }

    headways_diffs = {
        'leader': [-1] * N,
        'av': [-1] * N,
        'idm_0': [headways['av'][i] - headways['idm_0'][i] for i in range(N)],
        **{ f'idm_{k}': [headways[f'idm_{k-1}'][i] - headways[f'idm_{k}'][i] for i in range(N)] for k in range(1, len(env.idm_followers)) },
    }

    max_pos = np.max(positions['leader'])
    relative_positions = {
        k: [positions[k][i] - max_pos * i / N for i in range(N)] for k in positions
    }

    if PLOT_PLOTS:
        for k in range(12):
            start = k * (N // 12)
            end = (k + 1) * (N // 12)

            for label, data in [('position', positions),
                                ('speed', speeds),
                                ('accel', accels),
                                ('headway', headways),
                                ('headways_diff', headways_diffs),
                                ('relative_position', relative_positions),]:
                plt.figure()
                for veh in data:
                    plt.plot(times[start:end], data[veh][start:end], label=veh)
                plt.xlabel('Time (s)')
                plt.ylabel(label)
                plt.legend()
                save_path = f'figs/test_env/{controller}/{label}'
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f'{save_path}/{k}.png')
                plt.close()

    if PLOT_HEATMAPS and custom_cp_path is not None:
        # also write some accel heatmaps
        for failsafe in [True, False]:
            for ego_speed in range(5, 30, 2):
                lead_speed_diff_range = np.arange(-10, 10.5, 0.5)
                headway_range = np.arange(0, 101, 1)
                lead_speed_diffs, headways = np.meshgrid(lead_speed_diff_range, headway_range)
                accels = np.zeros_like(lead_speed_diffs)
                for i in range(lead_speed_diffs.shape[0]):
                    for j in range(lead_speed_diffs.shape[1]):
                        leader_speed = ego_speed + lead_speed_diffs[i,j]
                        headway = headways[i,j]

                        state = env.normalize_state({
                            'speed': ego_speed,
                            'leader_speed': leader_speed,
                            'headway': headway,
                        })
                        accel, _ = model.predict(state, deterministic=True)

                        if failsafe:
                            time_step = 0.1
                            max_accel = 1.5
                            max_decel = 3.0
                            v_safe = safe_velocity(ego_speed, leader_speed, headway, max_decel, time_step)
                            v_next = accel * time_step + ego_speed
                            if v_next > v_safe:
                                accel = np.clip((v_safe - ego_speed) / time_step, -np.abs(max_decel), max_accel)

                        accels[-1-i,j] = accel
                extent = np.min(lead_speed_diff_range), np.max(lead_speed_diff_range), np.min(headway_range), np.max(headway_range)
                figure = plt.figure() #figsize=(3,3))
                # figure.tight_layout()
                subplot = figure.add_subplot()
                im = subplot.imshow(accels, extent=extent, cmap=plt.cm.RdBu, interpolation='bilinear', vmin=-3, vmax=1.5)
                extent = im.get_extent()
                subplot.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/1.0)
                figure.colorbar(im, ax=subplot)
                subplot.set_xlabel('Leader speed delta (m/s)')
                subplot.set_ylabel('Headway (m)')
                # figure.tight_layout()
                failsafe_str = 'with failsafe' if failsafe else 'no failsafe'
                subplot.set_title(f'Ego speed {ego_speed}m/s ({failsafe_str})')
                save_path = f'figs/test_env/{controller}/accel_heatmaps'
                os.makedirs(save_path, exist_ok=True)
                failsafe_str = '_failsafe' if failsafe else ''
                plt.savefig(f'{save_path}/ego_speed_{ego_speed}{failsafe_str}.png')
                plt.close() 