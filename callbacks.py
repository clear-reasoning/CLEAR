import numpy as np
import os
import math
from tqdm import tqdm
import time

import matplotlib.pyplot as plt
import os, os.path
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from collections import defaultdict

from env.trajectory_env import TrajectoryEnv, SPEED_SCALE, DISTANCE_SCALE
from env.accel_controllers import IDMController, TimeHeadwayFollowerStopper

from env.energy_models import PFMMidsizeSedan


class TensorboardCallback(BaseCallback):
    """Callback for plotting additional metrics in tensorboard."""
    def __init__(self, eval_freq, eval_at_start, eval_at_end):
        super(TensorboardCallback, self).__init__()

        self.eval_freq = eval_freq
        self.eval_at_start = eval_at_start
        self.eval_at_end = eval_at_end
        self.rollout = 0

        self.energy_model = PFMMidsizeSedan()

    def _on_rollout_start(self):
        self.rollout_info = defaultdict(list)
        self.n_states = self.training_env.observation_space.shape[0]

        self.rollout += 1

    def _on_step(self):
        # TODO can probably get all these info at the end from replay buffer
        if self.eval_freq is not None and self.rollout % self.eval_freq == 0:
            # get information about rollout for one of the actors
            try:
                observations = self.locals['obs_tensor'][0]
            except:
                observations = self.locals['new_obs'][0]
            for i in range(self.n_states):
                self.rollout_info[f'state_{i}'].append(observations[i])

            # check if we have a discrete action space
            if 'actions' in self.locals:
                valid_key = 'actions'
            else:
                valid_key = 'action'

            if len(self.locals[valid_key].shape) == 1:
                unwrap_func = lambda x: x[0]
            else:
                unwrap_func = lambda x: x[0][0]
            self.rollout_info[f'action'].append(unwrap_func(self.locals[valid_key]))
            # self.rollout_info[f'clipped_action'].append(unwrap_func(self.locals['clipped_actions']))
            # self.rollout_info[f'value'].append(self.locals['values'][0][0])
            # self.rollout_info[f'log_prob'].append(self.locals['log_probs'][0])
            if 'rewards' in self.locals:
                valid_key = 'rewards'
            else:
                valid_key = 'reward'
            self.rollout_info[f'reward'].append(self.locals[valid_key][0])
            if 'dones' in self.locals:
                valid_key = 'dones'
            else:
                valid_key = 'done'
            self.rollout_info[f'done'].append(int(self.locals[valid_key][0]))
            for k, v in self.locals['infos'][0].items():
                if k not in ['episode', 'terminal_observation']:
                    self.rollout_info[k].append(v)

        return True

    def _on_rollout_end(self):
        if self.eval_freq is not None and self.rollout % self.eval_freq == 0:
            # log rollout plots for one of the actors
            self.log_rollout_stats()

            # log and plot some metrics along the whole dataset trajectory
            # using the RL controller, as well as IDM and FS baselines
            self.log_trajectory_stats()
        
    def _on_training_start(self):
        if self.eval_at_start:
            self.log_trajectory_stats()

    def _on_training_end(self):
        if self.eval_at_end and (self.eval_freq is None or self.rollout % self.eval_freq != 0):
            self.log_trajectory_stats()

    def log_rollout_stats(self):
        state_names = self.training_env.envs[0].state_names
        state_scales = self.training_env.envs[0].state_scales
        # try:
        #     assert (len(state_names) == self.n_states and len(state_scales) == self.n_states)
        # except:
        #     import ipdb; ipdb.set_trace()
        for i in range(len(state_names)):
            self.rollout_info[state_names[i]] = [x * state_scales[i] for x in self.rollout_info[f'state_{i}']]
            del self.rollout_info[f'state_{i}']
        for key in self.rollout_info:
            figure = plt.figure()
            figure.add_subplot().plot(self.rollout_info[key])
            self.logger.record(f'rollout/{key}', Figure(figure, close=True), exclude=('stdout', 'log', 'json', 'csv'))
            plt.close()

    def log_trajectory_stats(self):
        for controller in ['rl', 'idm', 'fs_leader']:
            # create test env from config
            config = dict(self.training_env.envs[0].config)
            config['whole_trajectory'] = True
            if controller != 'rl':
                config['use_fs'] = False
                config['discrete'] = False
            test_env = TrajectoryEnv(config=config)

            # set controller
            if controller == 'rl':
                if test_env.use_discrete:
                    def get_action(state):
                        return self.model.predict(state, deterministic=True)[0]
                else:
                    def get_action(state):
                        return self.model.predict(state, deterministic=True)[0][0]
                    
            elif controller == 'idm':
                idm = IDMController(a=test_env.max_accel, b=test_env.max_decel)
                def get_action(state):
                    s = test_env.unnormalize_state(state)
                    return idm.get_accel(s['speed'], s['leader_speed'], s['headway'], test_env.time_step)

            elif controller == 'fs_leader':
                fs = TimeHeadwayFollowerStopper(max_accel=test_env.max_accel, max_deaccel=test_env.max_decel)
                def get_action(state):
                    s = test_env.unnormalize_state(state)
                    fs.v_des = s['leader_speed']
                    return fs.get_accel(s['speed'], s['leader_speed'], s['headway'], test_env.time_step)

            # execute controller on traj
            data = []
            state = test_env.reset()
            done = False
            while not done:
                try:
                    action = get_action(state)
                except:
                    import ipdb; ipdb.set_trace()
                state, reward, done, infos = test_env.step(action)
                data.append((state, action, reward, done, infos))
        
            # plot data
            data_plot = defaultdict(list)

            for state in [test_env.unnormalize_state(x[0]) for x in data]:
                for k, v in state.items():
                    data_plot[k].append(v)

            data_plot['actions'] = [x[1] for x in data]
            data_plot['rewards'] = [x[2] for x in data]
            data_plot['dones'] = [x[3] for x in data]

            for info in [x[4] for x in data]:
                for k, v in info.items():
                    data_plot[k].append(v)

            data_plot['speeds'] = {
                'av': data_plot['speed'],
                'leader': data_plot['leader_speed'],
            }

            data_plot['episode_reward'].append(data_plot['rewards'][0])
            for rwd in data_plot['rewards'][1:]:
                data_plot['episode_reward'].append(data_plot['episode_reward'][-1] + rwd)

            num_veh = len(test_env.idm_followers) + 1
            mpg = 0
            for i in range(num_veh):
                mpg += (sum(data_plot['speed_{}'.format(i)]) / 1609.34) / (sum(data_plot['energy_consumption_{}'.format(i)]) / 3600 + 1e-6)
            mpg /= num_veh
            self.logger.record(f'trajectory/{controller}_mpg', mpg)
            self.logger.record(f'trajectory/{controller}_total_reward', data_plot['episode_reward'][-1])

            del data_plot['speed'], data_plot['leader_speed']

            for key, values_lst in data_plot.items():
                figure = plt.figure()
                subplot = figure.add_subplot()
                if type(values_lst) is dict:
                    for label, values in values_lst.items():
                        subplot.plot(values, label=label)
                        subplot.legend()
                else:
                    subplot.plot(values_lst)
                self.logger.record(f'trajectory/{controller}_{key}', Figure(figure, close=True), exclude=('stdout', 'log', 'json', 'csv'))
                plt.close()

            # colormap
            ego_speed = 5
            lead_speed_range = np.arange(0, 40, 2)
            headway_range = np.arange(0, 100, 5)
            lead_speeds, headways = np.meshgrid(lead_speed_range, headway_range)
            accels = np.zeros_like(lead_speeds)
            for i in range(lead_speeds.shape[0]):
                for j in range(lead_speeds.shape[1]):
                    state = test_env.normalize_state({
                        'vdes': 10.0,
                        'speed': ego_speed,
                        'leader_speed': lead_speeds[i,j],
                        'headway': headways[i,j],
                    })
                    if test_env.extra_obs:
                        extra_obs_shape = int(test_env.observation_space.low.shape[0] / 2)
                        state = np.concatenate((state, np.zeros(extra_obs_shape)))

                    accels[-1-i,j] = get_action(state)
            extent = np.min(lead_speed_range), np.max(lead_speed_range), np.min(headway_range), np.max(headway_range)
            figure = plt.figure(figsize=(3,3))
            figure.tight_layout()
            subplot = figure.add_subplot()
            im = subplot.imshow(accels, extent=extent, cmap=plt.cm.RdBu, interpolation='bilinear', vmin=np.min(accels), vmax=np.max(accels))
            extent = im.get_extent()
            subplot.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/1.0)
            figure.colorbar(im, ax=subplot)
            subplot.set_xlabel('Leader speed (m/s)')
            subplot.set_ylabel('Headway (m)')
            figure.tight_layout()

            self.logger.record(f'trajectory/{controller}_accel_colormap', Figure(figure, close=True), exclude=('stdout', 'log', 'json', 'csv'))
            plt.close()       


class CheckpointCallback(BaseCallback):
    """Callback for saving a model every `save_freq` rollouts."""
    def __init__(self, save_freq=10, save_path='./checkpoints', save_at_end=False):
        super(CheckpointCallback, self).__init__()

        self.save_freq = save_freq
        self.save_path = save_path
        self.save_at_end = save_at_end

        self.rollout = 0

        os.makedirs(self.save_path, exist_ok=True)

    def _on_rollout_end(self):
        self.rollout += 1

        if self.save_freq is not None and self.rollout % self.save_freq == 0:
            self.write_checkpoint()

    def _on_training_end(self):
        if (self.save_freq is None or self.rollout % self.save_freq != 0) and self.save_at_end:
            self.write_checkpoint()

    def write_checkpoint(self):
        path = os.path.join(self.save_path, f'iter_{self.rollout}_{self.num_timesteps}steps')
        self.model.save(path)
        print(f'Saving model checkpoint to {path}.zip')

    def _on_step(self):
        return True


class LoggingCallback(BaseCallback):
    """Callback for logging additional information."""
    def __init__(self, log_metrics=False, grid_search_config={}):
        super(LoggingCallback, self).__init__()

        self.grid_search_config = grid_search_config
        self.log_metrics = log_metrics

    def _on_rollout_start(self):
        self.rollout_t0 = time.time()

    def _on_rollout_end(self):
        # log current training progress 
        timesteps_per_iter = self.training_env.num_envs * self.model.n_steps
        total_iters = math.ceil(self.locals['total_timesteps'] / timesteps_per_iter)
        total_timesteps_rounded = timesteps_per_iter * total_iters
        progress_fraction = self.num_timesteps / total_timesteps_rounded

        if self.log_metrics:
            self.logger.record('time/timesteps', self.num_timesteps)
            self.logger.record('time/iters', self.num_timesteps // timesteps_per_iter)
                
        self.logger.record('time/goal_timesteps', total_timesteps_rounded)
        self.logger.record('time/goal_iters', total_iters)
        self.logger.record('time/training_progress', f'{round(100 * progress_fraction, 1)}%')

        def duration_to_str(delta_t):
            """Convert a duration (in seconds) into a human-readable string."""
            delta_t = int(delta_t)
            s_out = ''
            for time_s, unit in [(86400, 'd'), (3600, 'h'), (60, 'm'), (1, 's')]:
                count = delta_t // time_s
                delta_t %= time_s
                if count > 0:
                    s_out += f'{count}{unit}'
            return s_out

        t = time.time()
        self.logger.record('time/time_since_start', duration_to_str(t - self.training_t0))
        self.logger.record('time/time_this_iter', duration_to_str(t - self.rollout_t0))
        time_left = (t - self.training_t0) / progress_fraction
        self.logger.record('time/estimated_time_left', duration_to_str(time_left))

        if self.log_metrics:
            self.print_metrics()

    def _on_training_start(self):
        self.training_t0 = time.time()

    def _on_training_end(self):
        if self.log_metrics:
            self.print_metrics()

    def print_metrics(self):
        metrics = self.logger.get_log_dict()
        gs_str = ', '.join([f'{k} = {v}' for k, v in self.grid_search_config.items()])
        print(f'\nEnd of rollout for grid search: {gs_str}')

        key2str = {}
        tag = None
        for (key, value) in sorted(metrics.items()):
            

            if isinstance(value, float):
                value_str = f'{value:<8.3g}'
            elif isinstance(value, int) or isinstance(value, str):
                value_str = str(value)
            else:
                continue  # plt figure

            if key.find('/') > 0: 
                tag = key[: key.find('/') + 1]
                key2str[tag] = ''
            if tag is not None and tag in key:
                key = str("   " + key[len(tag) :])

            key2str[key] = value_str

        key_width = max(map(len, key2str.keys()))
        val_width = max(map(len, key2str.values()))

        dashes = '-' * (key_width + val_width + 7)
        lines = [dashes]
        for key, value in key2str.items():
            key_space = ' ' * (key_width - len(key))
            val_space = ' ' * (val_width - len(value))
            lines.append(f'| {key}{key_space} | {value}{val_space} |')
        lines.append(dashes)
        print('\n'.join(lines))

    def _on_step(self):
        return True


class ProgressBarCallback(BaseCallback):
    """Callback to display a progress bar showing the progress of each rollout."""
    def __init__(self):
        super(ProgressBarCallback, self).__init__()

    def _on_rollout_start(self):
        self.n_envs = self.training_env.num_envs
        self.n_steps = self.model.n_steps

        self.pbar = tqdm(
            desc='Rollout progress',
            total=self.n_envs * self.n_steps,
            leave=True,
            unit=' env steps')

    def _on_rollout_end(self):        
        self.pbar.close()

    def _on_step(self):
        self.pbar.update(self.n_envs)
