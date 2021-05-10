from env.trajectory_env import TrajectoryEnv
from env.idm import IDMController
from env.TimeHeadwayFollowerStopper import TimeHeadwayFollowerStopper

import matplotlib.pyplot as plt


if __name__ == '__main__':
    rwds = []

    n_rollouts = 1000
    controller = 'fs'

    for _ in range(n_rollouts):

        # plot trajectories obtained if av has an idm controller
        env = TrajectoryEnv(config={
            'max_accel': 1.5,
            'max_decel': 3.0,
            'horizon': 500,
            'min_speed': 0,
            'max_speed': 40,
            'max_headway': 70,}
        )
        idm = IDMController(a=env.max_accel, b=env.max_decel)
        fs = TimeHeadwayFollowerStopper()

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
            if controller == 'idm':
                accel = idm.get_accel(state[0]*50, state[1]*50, state[2]*100)
            elif controller == 'fs':
                fs.v_des = state[1]*50
                accel = fs.get_accel(state[0]*50, state[1]*50, state[2]*100, env)
                # print(state[0]*50, state[1]*50, state[2]*100, accel)
            else:
                raise ValueError
            state, reward, done, _ = env.step(accel)
            total_reward += reward
            i += 1

            # print(env.traj_idx, state, accel)

            times.append(times[-1] + env.time_step)
            positions['leader'].append(env.leader_positions[env.traj_idx])
            positions['av'].append(env.av['pos'])
            for k in range(len(env.idm_followers)):
                positions[f'idm_{k}'].append(env.idm_followers[k]['pos'])

        # print(f'Total reward: {total_reward}, total steps: {i}')
        rwds.append(total_reward)


    print(f'Avg reward over {n_rollouts} rollouts with {controller}: ', sum(rwds) / len(rwds))
    # print(rwds)
    plt.figure()
    for k in positions:
        plt.plot(times, positions[k], label=k)
    plt.ylabel('Position (m)')
    plt.xlabel('Time (s)')
    plt.title('Positions of leader, AV and IDMs if AV uses an IDM controller')
    plt.legend()
    plt.savefig('figs/test_env/positions.png')