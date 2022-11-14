"""Callbacks."""
import math
import os
import random
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import boto3
import matplotlib
import numpy as np
import pytz
from stable_baselines3.common.callbacks import BaseCallback

from trajectory.env.trajectory_env import TrajectoryEnv
from trajectory.env.utils import duration_to_str, get_first_element
from trajectory.visualize.plotter import TensorboardPlotter

matplotlib.use('agg')


class TensorboardCallback(BaseCallback):
    """Callback for plotting additional metrics in tensorboard."""

    def __init__(self, eval_freq, eval_at_end, env_config):
        super().__init__()

        self.eval_freq = eval_freq
        self.eval_at_end = eval_at_end
        self.rollout = 0
        self.env_config = env_config

    def _on_training_start(self):
        # self.env = self.training_env.envs[0]
        self.env_remote = self.training_env.remotes[0]
        
    def env_call(self, function, arg=None, expect_return=False):
        if arg is None:
            self.env_remote.send((str(function), arg))
        if expect_return:
            return self.env_remote.recv()

    def _on_training_end(self):
        if self.eval_at_end and (self.eval_freq is None or (self.rollout - 1) % self.eval_freq != 0):
            # self.log_rollout_dict('idm_eval', self.run_eval(av_controller='idm'))
            # self.log_rollout_dict('fs_eval', self.run_eval(av_controller='fs'))
            self.log_rollout_dict('rl_eval', self.run_eval(av_controller='rl_acc'), custom_plot=True)

    def _on_rollout_start(self):
        self.env_remote.send(('env_method', ('start_collecting_rollout', [], {})))
        # if expect_return:
        #     return self.env_remote.recv()
        # self.env_call('start_collecting_rollout')
        pass

    def _on_rollout_end(self):
        self.env_remote.send(('env_method', ('stop_collecting_rollout', [], {})))
        # self.env.stop_collecting_rollout()

        self.log_rollout_dict('metrics', self.get_rollout_dict(self.env_remote), plot_images=False)

        if self.eval_freq is not None and self.rollout % self.eval_freq == 0:
            # self.log_rollout_dict('idm_eval', self.run_eval(av_controller='idm'))
            # self.log_rollout_dict('fs_eval', self.run_eval(av_controller='fs'))
            self.log_rollout_dict('rl_eval', self.run_eval(av_controller='rl_acc'), custom_plot=True)

        self.rollout += 1

    def log_rollout_dict(self, base_name, rollout_dict, plot_images=True, custom_plot=False):
        """Log rollout dict."""
        if plot_images:
            plotter = TensorboardPlotter(self.logger)
            if custom_plot:
                custom_metrics = {
                    'speeds': {
                        'ego': rollout_dict['sim_data_av']['speed'],
                        'leader': rollout_dict['sim_data_av']['leader_speed'],
                        'trajectory': rollout_dict['sim_data_leader']['speed'],
                    },
                    'headway': {
                        'av_gap': rollout_dict['sim_data_av']['headway'],
                        'gap_closing_threshold': [max(120, 6 * vel)
                                                  for vel in rollout_dict['sim_data_av']['speed']],
                        'failsafe_threshold': [6 * ((this_vel + 1 + this_vel * 4 / 30) - lead_vel)
                                               for this_vel, lead_vel in zip(rollout_dict['sim_data_av']['speed'],
                                                                             rollout_dict['sim_data_av']['leader_speed'])],
                    },
                    'gaps (max 10 for readability)': {
                        'time_gap': [min(x, 10) for x in rollout_dict['sim_data_av']['time_gap']],
                        'time_to_collision': [min(x, 10) for x in rollout_dict['sim_data_av']['time_to_collision']],
                    },
                    'accels': {
                        'before_failsafe': rollout_dict['sim_data_av']['target_accel_no_noise_no_failsafe'],
                        'after_failsafe': rollout_dict['sim_data_av']['target_accel_no_noise_with_failsafe'],
                    },
                    'n_vehicles': rollout_dict['lane_changes']['n_vehicles'],
                    'lane_changes': {
                        'n_cutins': rollout_dict['lane_changes']['n_cutins'],
                        'n_cutouts': rollout_dict['lane_changes']['n_cutouts'],
                    },
                    'rewards': rollout_dict['training']['rewards'],
                }
                if 'gap_actions' and 'speed_actions' in rollout_dict['training']:
                    custom_metrics.update({
                        'speed_actions': rollout_dict['training']['speed_actions'],
                        'gap_actions': rollout_dict['training']['gap_actions'],
                        'speed_setting': rollout_dict['training']['speed_setting'],
                        'gap_setting': rollout_dict['training']['gap_setting'],
                    })
                if 'target_speed' in rollout_dict['training']:
                    custom_metrics.update({
                        'target_speed': rollout_dict['training']['target_speed'],
                    })
                for k, v in custom_metrics.items():
                    if isinstance(v, dict):
                        with plotter.subplot(title=k, grid=True, legend=True):
                            for subk, subv in v.items():
                                plotter.plot(subv, label=subk, linewidth=2.0,)
                    else:
                        plotter.plot(v, title=k, grid=True, linewidth=2.0)
                plotter.save(f'{base_name}/{base_name}_data')
            else:
                for group, metrics in rollout_dict.items():
                    for k, v in metrics.items():
                        plotter.plot(v, title=k, grid=True, linewidth=2.0)
                    plotter.save(f'{base_name}/{base_name}_{group}')

        episode_reward = np.sum(rollout_dict['training']['rewards'])
        episode_energy_reward = np.sum(rollout_dict['training']['energy_rewards'])
        episode_accel_reward = np.sum(rollout_dict['training']['accel_rewards'])
        episode_intervention_reward = np.sum(rollout_dict['training']['intervention_rewards'])
        episode_headway_reward = np.sum(rollout_dict['training']['headway_rewards'])
        episode_speed_diff_reward = np.sum(rollout_dict['training']['speed_diff_reward'])
        av_mpg = rollout_dict['sim_data_avs']['avg_mpg'][-1]
        system_mpg = rollout_dict['system']['avg_mpg'][-1]
        system_speed = rollout_dict['system']['speed'][-1]
        self.logger.record(f'{base_name}/{base_name}_episode_reward', episode_reward)
        self.logger.record(f'{base_name}/{base_name}_episode_energy_reward', episode_energy_reward)
        self.logger.record(f'{base_name}/{base_name}_episode_accel_reward', episode_accel_reward)
        self.logger.record(f'{base_name}/{base_name}_episode_intervention_reward', episode_intervention_reward)
        self.logger.record(f'{base_name}/{base_name}_episode_headway_reward', episode_headway_reward)
        self.logger.record(f'{base_name}/{base_name}_episode_speed_diff_reward', episode_speed_diff_reward)
        self.logger.record(f'{base_name}/{base_name}_av_mpg', av_mpg)
        self.logger.record(f'{base_name}/{base_name}_system_mpg', system_mpg)
        self.logger.record(f'{base_name}/{base_name}_system_speed', system_speed)
        self.logger.record(f'{base_name}/{base_name}_n_veh_start', rollout_dict['lane_changes']['n_vehicles'][0])
        self.logger.record(f'{base_name}/{base_name}_n_veh_end', rollout_dict['lane_changes']['n_vehicles'][-1])
        self.logger.record(f'{base_name}/{base_name}_n_cutins', rollout_dict['lane_changes']['n_cutins'][-1])
        self.logger.record(f'{base_name}/{base_name}_n_cutouts', rollout_dict['lane_changes']['n_cutouts'][-1])
        for i in range(1):
            platoon_mpg = rollout_dict[f'platoon_{i}']['platoon_mpg'][-1]
            self.logger.record(f'{base_name}/{base_name}_platoon_{i}_mpg', platoon_mpg)
        for penalty in ['crash', 'low_headway_penalty', 'large_headway_penalty', 'low_time_headway_penalty']:
            has_penalty = int(any(rollout_dict['custom_metrics'][penalty]))
            self.logger.record(f'{base_name}/{base_name}_has_{penalty}', has_penalty)

        for (name, array) in [
            ('reward', rollout_dict['training']['rewards']),
            ('headway', rollout_dict['sim_data_av']['headway']),
            ('speed_difference', rollout_dict['sim_data_av']['speed_difference']),
            ('instant_energy_consumption', rollout_dict['sim_data_av']['instant_energy_consumption']),
            ('speed', rollout_dict['base_state']['speed'])] + \
                [(f'platoon_{i}_speed', rollout_dict[f'platoon_{i}']['platoon_speed']) for i in range(1)]:
            self.logger.record(f'{base_name}/{base_name}_min_{name}', np.min(array))
            self.logger.record(f'{base_name}/{base_name}_max_{name}', np.max(array))
            self.logger.record(f'{base_name}/{base_name}_mean_{name}', np.mean(array))

    def get_rollout_dict(self, env_remote):
        """Get rollout dict."""

        env_remote.send(('env_method', ('get_collected_rollout', [], {})))
        collected_rollout = env_remote.recv()

        # collected_rollout = env.get_collected_rollout()

        rollout_dict = defaultdict(lambda: defaultdict(list))

        rollout_dict['training']['rewards'] = collected_rollout['rewards']
        rollout_dict['training']['energy_rewards'] = collected_rollout['energy_rewards']
        rollout_dict['training']['accel_rewards'] = collected_rollout['accel_rewards']
        rollout_dict['training']['intervention_rewards'] = collected_rollout['intervention_rewards']
        rollout_dict['training']['headway_rewards'] = collected_rollout['headway_rewards']
        rollout_dict['training']['speed_diff_reward'] = collected_rollout['speed_diff_reward']
        rollout_dict['training']['dones'] = collected_rollout['dones']
        rollout_dict['training']['actions'] = collected_rollout['actions']
        if env.output_acc:
            rollout_dict['training']['speed_actions'] = collected_rollout['speed_actions']
            rollout_dict['training']['gap_actions'] = collected_rollout['gap_actions']
            rollout_dict['training']['speed_setting'] = collected_rollout['speed_setting']
            rollout_dict['training']['gap_setting'] = collected_rollout['gap_setting']
        if env.speed_planner:
            rollout_dict['training']['target_speed'] = collected_rollout['target_speed']


        if 'metrics' in collected_rollout['infos'][0]:
            for info in collected_rollout['infos']:
                for k, v in info['metrics'].items():
                    rollout_dict['custom_metrics'][k].append(v)

        for base_state in collected_rollout['base_states']:
            for k, v in base_state.items():
                rollout_dict['base_state'][k].append(v[0])
        for base_state_vf in collected_rollout['base_states_vf']:
            for k, v in base_state_vf.items():
                rollout_dict['base_state'][f'vf_{k}'].append(v[0])

        for lane_change_info in collected_rollout['lane_changes']:
            for k, v in lane_change_info.items():
                rollout_dict['lane_changes'][k].append(v)

        for veh in env.sim.vehicles:
            if veh.kind == 'av':
                for k, v in env.sim.data_by_vehicle[veh.name].items():
                    rollout_dict['sim_data_av'][k] = v
            if veh.kind == 'av':
                for k, v in env.sim.data_by_vehicle[veh.name].items():
                    if k == 'avg_mpg':
                        rollout_dict['sim_data_avs'][k].append(v[-1])
            if veh.kind == 'leader':
                for k, v in env.sim.data_by_vehicle[veh.name].items():
                    if k == 'speed':
                        rollout_dict['sim_data_leader'][k] = v
                        
        for veh in env.sim.vehicles:
            if veh.kind == 'leader':
                rollout_dict['speed_planner']['leader_speed'] = env.sim.data_by_vehicle[veh.name]['speed']
            if veh.kind == 'av':
                rollout_dict['speed_planner']['gap'] = env.sim.data_by_vehicle[veh.name]['headway']
                rollout_dict['speed_planner']['inrix_local_speed'] = env.sim.data_by_vehicle[veh.name]['inrix_local_speed']
                rollout_dict['speed_planner']['inrix_next_speed'] = env.sim.data_by_vehicle[veh.name]['inrix_next_speed']
                rollout_dict['speed_planner']['inrix_next_next_speed'] = env.sim.data_by_vehicle[veh.name]['inrix_next_next_speed']
                rollout_dict['speed_planner']['target_speed'] = rollout_dict['base_state']['target_speed']
                rollout_dict['speed_planner']['max_headway'] = rollout_dict['base_state']['max_headway']

        for i in range(len(env.avs)):
            for platoon_state in collected_rollout[f'platoon_{i}']:
                for k, v in platoon_state.items():
                    rollout_dict[f'platoon_{i}'][k].append(v)

        for system_info in collected_rollout['system']:
            for k, v in system_info.items():
                rollout_dict['system'][k].append(v)

        return rollout_dict

    def run_eval(self, av_controller):
        """Run evaluation."""
        # set seed so that the different evaluated controllers use the same trajectory

        random.seed(self.rollout)
        np.random.seed(self.rollout)

        # create test env
        config = dict(self.env_config)
        config['whole_trajectory'] = True
        if av_controller != 'rl':
            config['use_fs'] = False
            config['discrete'] = False
        config['av_controller'] = av_controller
        if av_controller == 'idm':
            config['av_kwargs'] = 'dict(v0=45,noise=0)'
        test_env = TrajectoryEnv(config=config, _verbose=False)
        # dont reset at the end of eval rollout otherwise data gets erased
        test_env.do_not_reset_on_end_of_horizon = True

        # execute controller on traj
        state = test_env.reset()
        done = False
        test_env.start_collecting_rollout()
        while not test_env.end_of_horizon:
            if av_controller == 'rl_acc':
                action = self.model.predict(state, deterministic=True)[0]
            elif av_controller == 'rl':
                action = get_first_element(self.model.predict(state, deterministic=True))
            else:
                action = 0
            state, reward, done, infos = test_env.step(action)
        test_env.stop_collecting_rollout()

        return self.get_rollout_dict(test_env)

    def _on_step(self):
        return True


class CheckpointCallback(BaseCallback):
    """Callback for saving a model every `save_freq` rollouts."""

    def __init__(self, save_freq=100, save_path='./checkpoints', save_at_end=True, s3_bucket=None, exp_logdir=None):
        super(CheckpointCallback, self).__init__()

        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_at_end = save_at_end

        self.iter = 0

        self.s3_bucket = s3_bucket
        self.exp_logdir = Path(exp_logdir)

        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_rollout_end(self):
        self.iter += 1

        if self.save_freq is not None and self.iter % self.save_freq == 0:
            self.write_checkpoint()

    def _on_training_end(self):
        if (self.save_freq is None or self.iter % self.save_freq != 0) and self.save_at_end:
            self.write_checkpoint()

    def write_checkpoint(self):
        """Write checkpoint."""
        path = self.save_path / str(self.iter)
        self.model.save(path)
        print(f'Saved model checkpoint to {path}.zip')

        if self.s3_bucket is not None and self.exp_logdir is not None:
            s3 = boto3.resource('s3').Bucket(self.s3_bucket)
            for root, _, file_names in os.walk(self.exp_logdir):
                for file_name in file_names:
                    file_path = Path(root, file_name)
                    file_path_s3 = file_path.relative_to(self.exp_logdir.parent.parent)
                    s3.upload_file(str(file_path), str(file_path_s3))
            print(
                f'Uploaded exp logdir to s3://{self.s3_bucket}/{self.exp_logdir.relative_to(self.exp_logdir.parent.parent)}')

    def _on_step(self):
        return True


class LoggingCallback(BaseCallback):
    """Callback for logging additional information."""

    def __init__(self, log_metrics=False, grid_search_config={}):
        super(LoggingCallback, self).__init__()

        self.grid_search_config = grid_search_config
        self.log_metrics = log_metrics

        self.ep_info_buffer = deque(maxlen=100)

    def _on_rollout_start(self):
        self.logger.record('time/time_this_iter', duration_to_str(time.time() - self.iter_t0))

        self.rollout_t0 = time.time()
        self.iter_t0 = time.time()

    def _on_rollout_end(self):
        # log current training progress
        if hasattr(self.model, 'n_steps'):
            # for PPO
            timesteps_per_iter = self.training_env.num_envs * self.model.n_steps
            total_iters = math.ceil(self.locals['total_timesteps'] / timesteps_per_iter)
        else:
            # for TD3
            timesteps_per_iter = self.training_env.num_envs * self.model.train_freq.frequency
            if len(self.locals['total_timesteps']) > 0:
                total_iters = math.ceil(self.locals['total_timesteps'][0] / timesteps_per_iter)
            else:
                total_iters = 1

        total_timesteps_rounded = timesteps_per_iter * total_iters
        progress_fraction = self.num_timesteps / total_timesteps_rounded

        if self.log_metrics:
            self.logger.record('time/timesteps', self.num_timesteps)
            self.logger.record('time/iters', self.num_timesteps // timesteps_per_iter)

        self.logger.record('time/goal_timesteps', total_timesteps_rounded)
        self.logger.record('time/goal_iters', total_iters)
        self.logger.record('time/training_progress', f'{round(100 * progress_fraction, 1)}%')

        t = time.time()
        self.logger.record('time/time_since_start', duration_to_str(t - self.training_t0))
        self.logger.record('time/time_this_rollout', duration_to_str(t - self.rollout_t0))
        time_left = (t - self.training_t0) / progress_fraction
        self.logger.record('time/estimated_time_left', duration_to_str(time_left))
        self.logger.record('time/timesteps_per_second', round(self.num_timesteps / (t - self.training_t0), 1))

        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", np.mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", np.mean([ep_info["l"] for ep_info in self.ep_info_buffer]))

        if self.log_metrics:
            self.print_metrics()

    def _on_training_start(self):
        self.training_t0 = time.time()
        self.iter_t0 = time.time()

    def _on_training_end(self):
        if self.log_metrics:
            self.print_metrics()

    def print_metrics(self):
        """Print metrics."""
        metrics = self.logger.name_to_value
        gs_str = ', '.join([f'{k} = {v}' for k, v in self.grid_search_config.items()])
        print(f'\nEnd of rollout for grid search: {gs_str}')

        key2str = {}
        tag = None
        for (key, value) in sorted(metrics.items()):
            if isinstance(value, float):
                value_str = f'{value:<10.5g}'
            elif isinstance(value, int) or isinstance(value, str):
                value_str = str(value)
            else:
                continue  # plt figure

            if key.find('/') > 0:
                tag = key[: key.find('/') + 1]
                key2str[tag] = ''
            if tag is not None and tag in key:
                key = str("   " + key[len(tag):])

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
        if (infos := self.locals['infos'][0].get('episode')) is not None:
            self.ep_info_buffer.extend([infos])
        return True


class TelegramCallback(BaseCallback):
    """Callback for sending training notifications through Telegram."""

    def __init__(self, gs_path):
        super(TelegramCallback, self).__init__()
        self.gs_path = gs_path

        self.bot_token = os.environ['TELEGRAM_BOT_TOKEN']
        self.chat_id = os.environ['TELEGRAM_CHAT_ID']

        self.iter = 0
        self.last_update_time = None
        self.training_t0 = time.time()

    def send_message(self, msg):
        import telegram
        telegram.Bot(token=self.bot_token).send_message(text=msg, chat_id=self.chat_id)

    def total_time_human_readable(self):
        total_time = time.time() - self.training_t0
        total_time_hr = time.strftime("%H hours, %M minutes, %S seconds", time.gmtime(int(total_time)))
        return total_time_hr

    def _on_rollout_end(self):
        self.iter += 1
        if self.iter == 1:
            self.send_message(f'End of first iteration for {self.gs_path}')
            self.last_update_time = datetime.now(tz=pytz.UTC)
        else:
            import telegram
            try:
                bot = telegram.Bot(token=self.bot_token)
                for update in bot.get_updates(offset=-10, timeout=60, read_latency=60):
                    if update.message is not None and update.message.chat.id == int(self.chat_id):
                        if update.message.date.replace(tzinfo=pytz.UTC) > self.last_update_time.replace(tzinfo=pytz.UTC):
                            self.last_update_time = update.message.date
                            self.send_message(f'Update: iteration {self.iter} for {self.gs_path} after {self.total_time_human_readable()}.')
            except telegram.error.TimedOut:
                pass

    def _on_training_end(self):
        self.send_message(f'Training ended for {self.gs_path} after {self.total_time_human_readable()}. '
                          'Note that there may still be a few final evaluations running.')

    def _on_step(self):
        return True
