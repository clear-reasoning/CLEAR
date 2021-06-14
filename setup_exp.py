from datetime import datetime
import json
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import register_policy
from stable_baselines3.ppo import PPO
from stable_baselines3.td3 import TD3
import torch

from algos.ppo.policies import PopArtActorCriticPolicy, SplitActorCriticPolicy
from algos.ppo.ppo import PPO as AugmentedPPO
from algos.td3.policies import CustomTD3Policy
from callbacks import CheckpointCallback, LoggingCallback, TensorboardCallback
from env.trajectory_env import TrajectoryEnv
from env.utils import dict_to_json

register_policy("PopArtMlpPolicy", PopArtActorCriticPolicy)


def run_experiment(config):
    # create exp logdir
    exp_logdir = config['gs_logdir']
    exp_logdir.mkdir(parents=True, exist_ok=True)

    # create env config
    env_config = {
        'max_accel': 1.5,
        'max_decel': 3.0,
        'horizon': config['env_horizon'],
        'min_speed': 0,
        'max_speed': 40,
        'max_headway': config['env_max_headway'],
        'minimal_headway': 7,
        'whole_trajectory': False,
        'discrete': config['env_discrete'],
        'num_actions': config['env_num_actions'],
        'use_fs': config['use_fs'],
        'extra_obs': config['augment_vf'],
        # if we get closer then this time headway we are forced to break with maximum decel
        'minimal_time_headway': config['env_minimal_time_headway'],
        # if false, we only include the AVs mpg in the calculation
        'include_idm_mpg': config['env_include_idm_mpg'],
        'num_idm_cars': config['env_num_idm_cars'],
        'num_concat_states': config['env_num_concat_states'],
        'num_steps_per_sim': config['env_num_steps_per_sim'],
    }

    # create env
    multi_env = make_vec_env(TrajectoryEnv, n_envs=config['n_envs'], env_kwargs=dict(config=env_config))

    # create callbacks
    callbacks = []        
    if not config['no_eval']:
        callbacks.append(TensorboardCallback(
            eval_freq=config['eval_frequency'],
            eval_at_start=True,
            eval_at_end=True))
    callbacks += [
        LoggingCallback(
            grid_search_config=config['gs_config'],
            log_metrics=True),
        CheckpointCallback(
            save_path=exp_logdir / 'checkpoints',
            save_freq=config['cp_frequency'],
            save_at_end=True),
    ]
    callbacks = CallbackList(callbacks)

    # create train config
    if config['algorithm'].lower() == 'ppo':
        algorithm = AugmentedPPO if config['augment_vf'] else PPO
        policy = SplitActorCriticPolicy if config['augment_vf'] else PopArtActorCriticPolicy

        train_config = {
            'policy_kwargs': {
                'net_arch': [{
                    'pi': [config['hidden_layer_size']] * config['network_depth'],
                    'vf': [config['hidden_layer_size']] * config['network_depth'],
                }],
            },
            'learning_rate': config['lr'],
            'n_steps': config['n_steps'],
            'batch_size': config['batch_size'],
            'n_epochs': config['n_epochs'],
            'gamma': config['gamma'],
            'gae_lambda': config['gae_lambda'],
            'clip_range': 0.2,
            'clip_range_vf': None,
            'ent_coef': 0.0,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
        }
    elif config['algorithm'].lower() == 'td3':
        algorithm = TD3
        policy = CustomTD3Policy if config['augment_vf'] else 'MlpPolicy'

        train_config = {
            'gamma': 0.99,
            'learning_rate': 0.0003,
            'buffer_size': 1000000,
            'learning_starts': 100,
            'train_freq': 100,
            'gradient_steps': 100,
            'batch_size': 128,
            'tau': 0.005,
            'policy_delay': 2,
            'action_noise': None,
            'target_policy_noise': 0.2,
            'target_noise_clip': 0.5,
        }
    else:
        raise ValueError(f'Unknown algorithm: {config["algorithm"]}')

    train_config.update({
        'env': multi_env,
        'tensorboard_log': exp_logdir,
        'verbose': 0,  # 0 no output, 1 info, 2 debug
        'seed': None,  # only concerns PPO and not the environment
        'device': 'cpu',  # 'cpu', 'cuda', 'auto'
        'policy': policy,
    })

    # create learn config
    learn_config = {
        'total_timesteps': config['iters'] * config['n_steps'] * config['n_envs'],
        'callback': callbacks,
    }

    # save configs
    configs = {
        'algorithm': algorithm,
        'env_config': env_config,
        'train_config': train_config,
        'learn_config': learn_config
    }
    dict_to_json(configs, exp_logdir / 'configs.json')

    # create model and start training
    model = algorithm(**train_config)
    model.learn(**learn_config)
