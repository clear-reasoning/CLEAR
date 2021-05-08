from env.trajectory_env import TrajectoryEnv
from env.idm import IDMController

import matplotlib.pyplot as plt


if __name__ == '__main__':
    # plot trajectories obtained if av has an idm controller
    env = TrajectoryEnv(config={})
    idm = IDMController(a=env.max_accel, b=env.max_decel)

    state = env.reset()
    done = False
    total_reward = 0
    i = 0

    times = [0]
    positions = {
        'leader': [env.leader_positions[env.traj_idx]],
        'av': [env.av['pos']],
        **{f'idm_{k}': [env.idm_followers[k]['pos']] for k in range(len(env.idm_followers))}
    }

    while not done:
        accel = idm.get_accel(state[0]*50, state[1]*50, state[2]*100)
        state, reward, done, _ = env.step(accel)
        total_reward += reward
        i += 1

        print(env.traj_idx, state, accel)

        times.append(times[-1] + env.time_step)
        positions['leader'].append(env.leader_positions[env.traj_idx])
        positions['av'].append(env.av['pos'])
        for k in range(len(env.idm_followers)):
            positions[f'idm_{k}'].append(env.idm_followers[k]['pos'])

    print(f'Total reward: {total_reward}, total steps: {i}')

    plt.figure()
    for k in positions:
        plt.plot(times, positions[k], label=k)
    plt.ylabel('Position (m)')
    plt.xlabel('Time (s)')
    plt.title('Positions of leader, AV and IDMs if AV uses an IDM controller')
    plt.legend()
    plt.savefig('figs/test_env/positions.png')