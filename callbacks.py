import numpy as np
import os
import math
from tqdm import tqdm

import matplotlib.pyplot as plt
import os, os.path
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from collections import defaultdict

from env.trajectory_env import TrajectoryEnv, SPEED_SCALE, DISTANCE_SCALE
from env.accel_controllers import IDMController, TimeHeadwayFollowerStopper

from env.energy_models import PFMMidsizeSedan


#     def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
#                        policies: Dict[str, Policy], episode: MultiAgentEpisode,
#                        env_index: int, **kwargs):
#         env = base_env.get_unwrapped()[0]

#         for controller in ['rl', 'idm']:
#             test_env = TrajectoryEnv(config=env.config)
#             test_env.whole_trajectory = True
#             state = test_env.reset()
#             done = False
#             total_reward = 0
#             total_distance = 0
#             total_energy = 0

#             actions = []
#             headways = []
#             speed_deltas = []

#             if controller == 'idm':
#                 idm = IDMController(a=env.max_accel, b=env.max_decel)
#                 test_env.use_fs = False  # for IDM only

#             while not done:
#                 if controller == 'rl':
#                     action = policies['default_policy'].compute_single_action(state, clip_actions=True, explore=False)[0][0]
#                 elif controller == 'idm':
#                     action = idm.get_accel(state[0] * SPEED_SCALE, state[1] * SPEED_SCALE, state[2] * DISTANCE_SCALE)
#                 state, reward, done, _ = test_env.step(action)

#                 actions.append(action)
#                 headways.append(state[2] * DISTANCE_SCALE)
#                 speed_deltas.append( np.abs(state[0] - state[1]) * SPEED_SCALE )

#                 total_reward += reward
#                 total_distance += test_env.av['speed']
#                 total_energy += test_env.energy_model.get_instantaneous_fuel_consumption(test_env.av['last_accel'], test_env.av['speed'], grade=0)

#             episode.custom_metrics[f"{controller}_traj_mpg"] = (total_distance / 1609.34) / (total_energy / 3600 + 1e-6)
#             episode.custom_metrics[f"{controller}_traj_reward"] = total_reward

#             episode.custom_metrics[f"{controller}_avg_abs_action"] = np.mean(np.abs(actions))
#             episode.custom_metrics[f"{controller}_std_abs_action"] = np.std(np.abs(actions))
#             episode.custom_metrics[f"{controller}_min_action"] = np.min(actions)
#             episode.custom_metrics[f"{controller}_max_action"] = np.max(actions)

#             episode.custom_metrics[f"{controller}_avg_headway"] = np.mean(headways)
#             episode.custom_metrics[f"{controller}_std_headway"] = np.std(headways)
#             episode.custom_metrics[f"{controller}_min_headway"] = np.min(headways)
#             episode.custom_metrics[f"{controller}_max_headway"] = np.max(headways)

#             episode.custom_metrics[f"{controller}_avg_speed_delta"] = np.mean(speed_deltas)
#             episode.custom_metrics[f"{controller}_std_speed_delta"] = np.std(speed_deltas)
#             episode.custom_metrics[f"{controller}_min_speed_delta"] = np.min(speed_deltas)
#             episode.custom_metrics[f"{controller}_max_speed_delta"] = np.max(speed_deltas)


# can also plot vdes
# and action/reward regularisations
# and maybe some of them simulatenously on the same plot (eg rwd/rwd regularisations, or lead speed/vdes)
# can also plot mpg etc on the rollout plots...
# actually it should be exactly the same data -> make a function that does both

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
            observations = self.locals['obs_tensor'][0]
            for i in range(self.n_states):
                self.rollout_info[f'state_{i}'].append(observations[i])
            self.rollout_info[f'action'].append(self.locals['actions'][0][0])
            self.rollout_info[f'clipped_action'].append(self.locals['clipped_actions'][0][0])
            self.rollout_info[f'value'].append(self.locals['values'][0][0])
            self.rollout_info[f'log_prob'].append(self.locals['log_probs'][0])
            self.rollout_info[f'reward'].append(self.locals['rewards'][0])
            self.rollout_info[f'done'].append(int(self.locals['dones'][0]))
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
        assert (len(state_names) == self.n_states and len(state_scales) == self.n_states)
        for i in range(self.n_states):
            self.rollout_info[state_names[i]] = [x * state_scales[i] for x in self.rollout_info[f'state_{i}']]
            del self.rollout_info[f'state_{i}']
        for key in self.rollout_info:
            figure = plt.figure()
            figure.add_subplot().plot(self.rollout_info[key])
            self.logger.record(f'rollout/{key}', Figure(figure, close=True), exclude=('stdout', 'log', 'json', 'csv'))
            plt.close()        

    def log_trajectory_stats(self):
        for controller in ['rl', 'idm', 'fs_leader']:
            test_env = TrajectoryEnv(config=self.training_env.envs[0].config)
            test_env.whole_trajectory = True

            state = test_env.reset()
            done = False

            if controller == 'idm':
                idm = IDMController(a=test_env.max_accel, b=test_env.max_decel)
                test_env.use_fs = False
            elif controller == 'fs_leader':
                fs = TimeHeadwayFollowerStopper(max_accel=test_env.max_accel, max_deaccel=test_env.max_decel)
                test_env.use_fs = False

            while not done:
                if controller == 'rl':
                    action = self.model.predict(state, deterministic=True)[0][0]
                elif controller == 'idm':
                    s = test_env.parse_state(state)
                    action = idm.get_accel(s['speed'], s['leader_speed'], s['headway'])
                elif controller == 'fs_leader':
                    s = test_env.parse_state(state)
                    fs.v_des = s['leader_speed']
                    action = fs.get_accel(s['speed'], s['leader_speed'], s['headway'], test_env.time_step)
                # if test_env.use_failsafe: action = failsafe(action)...
                
                state, reward, done, infos = test_env.step(action)


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
    def __init__(self):
        super(LoggingCallback, self).__init__()

    def _on_rollout_end(self):
        # log current training progress 
        timesteps_per_iter = self.training_env.num_envs * self.model.n_steps
        total_iters = math.ceil(self.locals['total_timesteps'] / timesteps_per_iter)
        total_timesteps_rounded = timesteps_per_iter * total_iters
        progress_percentage = round(100 * self.num_timesteps / total_timesteps_rounded, 1)

        self.logger.record('time/goal_timesteps', total_timesteps_rounded)
        self.logger.record('time/goal_iters', total_iters)
        self.logger.record('time/training_progress', progress_percentage)

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
