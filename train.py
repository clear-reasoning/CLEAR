"""Train a policy."""
import argparse
import itertools
import multiprocessing
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import register_policy
from stable_baselines3.ppo import PPO
from stable_baselines3.td3 import TD3

import wandb
from trajectory.algos.ppo.policies import PopArtActorCriticPolicy, SplitActorCriticPolicy
from trajectory.algos.ppo.ppo import PPO as AugmentedPPO
from trajectory.algos.td3.policies import CustomTD3Policy
from trajectory.callbacks import CheckpointCallback, LoggingCallback, TensorboardCallback, TelegramCallback
from trajectory.env.trajectory_env import DEFAULT_ENV_CONFIG, TrajectoryEnv
from trajectory.env.utils import dict_to_json, partition

register_policy("PopArtMlpPolicy", PopArtActorCriticPolicy)


def parse_args_train():
    """Parse arguments for training."""
    parser = argparse.ArgumentParser(description='Train on the trajectory env.')

    # exp params
    parser.add_argument('--expname', type=str, default='test',
                        help='Name for the experiment.')
    parser.add_argument('--logdir', type=str, default='./log',
                        help='Experiment logs, checkpoints and tensorboard files '
                             'will be saved under {logdir}/{expname}_[current_time]/.')
    parser.add_argument('--n_processes', type=int, default=1,
                        help='Number of processes to run in parallel. Useful when running grid searches.'
                             'Can be more than the number of available CPUs.')
    parser.add_argument('--s3', default=False, action='store_true',
                        help='If set, experiment data will be uploaded to s3://trajectory.env/. '
                             'AWS credentials must have been set in ~/.aws in order to use this.')
    parser.add_argument('--wandb', default=False, action='store_true',
                        help='If set, log experiment data in WandB')

    parser.add_argument('--iters', type=int, default=1, nargs='+',
                        help='Number of iterations (rollouts) to train for.'
                             'Over the whole training, {iters} * {n_steps} * {n_envs} environment steps will be sampled.')
    parser.add_argument('--n_steps', type=int, default=640, nargs='+',
                        help='Number of environment steps to sample in each rollout in each environment.'
                             'This can span over less or more than the environment horizon.'
                             'Ideally should be a multiple of {batch_size}.')
    parser.add_argument('--n_envs', type=int, default=1, nargs='+',
                        help='Number of environments to run in parallel.')

    parser.add_argument('--cp_frequency', type=int, default=10,
                        help='A checkpoint of the model will be saved every {cp_frequency} iterations.'
                             'Set to None to not save no checkpoints during training.'
                             'Either way, a checkpoint will automatically be saved at the end of training.')
    parser.add_argument('--eval_frequency', type=int, default=10,
                        help='An evaluation of the model will be done and saved to tensorboard every {eval_frequency}'
                             ' iterations. Set to None to run no evaluations during training. Either way, an'
                             ' evaluation will automatically be done at the start and at the end of training.')
    parser.add_argument('--no_eval', default=False, action='store_true',
                        help='If set, no evaluation (ie. tensorboard plots) will be done.')
    parser.add_argument('--telegram', default=False, action='store_true',
                        help='If set, you will receive training updates on Telegram (need to set up token and chat id).')

    # training params
    parser.add_argument('--algorithm', type=str, default='PPO', nargs='+',
                        help='RL algorithm to train with. Available options: PPO, TD3.')

    parser.add_argument('--hidden_layer_size', type=int, default=32, nargs='+',
                        help='Hidden layer size to use for the policy and value function networks.'
                             'The networks will be composed of {network_depth} hidden layers of size {hidden_layer_size}.')
    parser.add_argument('--network_depth', type=int, default=2, nargs='+',
                        help='Number of hidden layers to use for the policy and value function networks.'
                             'The networks will be composed of {network_depth} hidden layers of size {hidden_layer_size}.')

    parser.add_argument('--lr', type=float, default=3e-4, nargs='+',
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=64, nargs='+',
                        help='Minibatch size.')
    parser.add_argument('--n_epochs', type=int, default=10, nargs='+',
                        help='Number of SGD iterations per training iteration.')
    parser.add_argument('--gamma', type=float, default=0.99, nargs='+',
                        help='Discount factor.')
    parser.add_argument('--gae_lambda', type=float, default=0.99, nargs='+',
                        help='Factor for trade-off of bias vs. variance for Generalized Advantage Estimator.')
    parser.add_argument('--ent_coef', type=float, default=0.0, nargs='+',
                        help='Entropy coefficient for the loss calculation')
    parser.add_argument('--seed', type=int, default=None, nargs='+',
                        help='PPO seed, random if not specified')
    parser.add_argument('--augment_vf', type=int, default=1, nargs='+',
                        help='If true, the value function will be augmented with some additional states.')
    parser.add_argument('--vf_include_chunk_idx', type=int, default=0, nargs='+',
                        help='If enabled (and augment_vf enabled), will pass in index of trajectory and starting index'
                             'of chunk into the value function.')
    # env params
    parser.add_argument('--traj_path', type=str,
                        default=None,
                        help='Set to train on a specific trajectory (eg dataset/data_v2_preprocessed_west/path/traj.csv).')
    parser.add_argument('--traj_dir', type=str, default=None, nargs='+',
                        help='Set to train on a specific set of trajectories (eg dataset/data_v2_preprocessed_west/).')
    parser.add_argument('--traj_curriculum', type=int, default=0, nargs='+',
                        help='If set to 1, introduce additional trajectories into training.')
    parser.add_argument('--traj_curriculum_dir', type=str, default=None, nargs='+',
                        help='If traj_curriculum, which set of trajectories to gradually introduce into training.')
    parser.add_argument('--traj_curriculum_freq', type=float, default=100,
                        help='Frequency at which to introduce trajectories into training.')

    parser.add_argument('--env_num_concat_states', type=int, default=1, nargs='+',
                        help='This many past states will be concatenated. If set to 1, it\'s just the current state. '
                             'This works only for the base states and not for the additional vf states.')
    parser.add_argument('--env_num_concat_states_large', type=int, default=0, nargs='+',
                        help='Same as --env_num_concat_states, but this concatenate states at a 1s interval instead of 0.1s. '
                             'The two commands can be used together.')
    parser.add_argument('--env_num_leader_speed_memory', type=int, default=0, nargs='+',
                        help='Number of previous leader speeds to add to state. If set to 0, no leader speed is added.')

    parser.add_argument('--env_discrete', type=int, default=0, nargs='+',
                        help='If true, the environment has a discrete action space.')
    parser.add_argument('--env_num_actions', type=int, default=50, nargs='+',
                        help='If discrete is set, the action space is discretized with this many actions')
    parser.add_argument('--env_min_accel', type=int, default=-3.0, nargs='+',
                        help='Lowest allowed acceleration')
    parser.add_argument('--env_max_accel', type=int, default=1.5, nargs='+',
                        help='Highest allowed acceleration')
    parser.add_argument('--use_fs', type=int, default=0, nargs='+',
                        help='If true, use a FollowerStopper wrapper.')
    parser.add_argument('--env_include_idm_mpg', type=int, default=0, nargs='+',
                        help='If true, the mpg is calculated averaged over the AV and the 5 IDMs behind.')
    parser.add_argument('--env_horizon', type=int, default=1000, nargs='+',
                        help='Sets the training horizon.')
    # Effective max headway is the higher value of max_headway and leader.speed * max_time_headway
    parser.add_argument('--env_max_headway', type=int, default=120, nargs='+',
                        help='Sets the maximum permitted headway')
    parser.add_argument('--env_max_time_headway', type=int, default=0, nargs='+',
                        help='Sets the maximum permitted time headway')
    parser.add_argument('--env_minimal_time_headway', type=float, default=1.0, nargs='+',
                        help='Sets the time headway below which we get penalized.')
    parser.add_argument('--env_minimal_time_to_collision', type=float, default=6.0, nargs='+',
                        help='Sets the time to collision below which we get penalized.')
    # Add arg for headway penalty
    parser.add_argument('--env_headway_penalty', type=float, default=0.0, nargs='+',
                        help='Sets the magnitude of the headway penalty (if > 0), where this coefficient'
                             'is multiplied by the time headway when headway > 10.')
    parser.add_argument('--env_accel_penalty', type=float, default=0.2, nargs='+',
                        help='Sets the magnitude of the acceleration penalty (to discourage large actions).')
    parser.add_argument('--env_intervention_penalty', type=float, default=0, nargs='+',
                        help='Factor to multiply accel_penalty to determine gap closing / failsafe penalty to'
                             'discourages use of these interventions')
    parser.add_argument('--env_include_thresholds', default=False, action='store_true',
                        help='If set, adds failsafe and gap-closing thresholds to base state.')
    parser.add_argument('--env_penalize_energy', type=int, default=1, nargs='+',
                        help='If true, penalize energy in the reward function')
    parser.add_argument('--env_platoon', type=str, default='av human*5', nargs='+',
                        help='Platoon of vehicles following the leader. Can contain either "human"s or "av"s. '
                             '"(av human*2)*2" can be used as a shortcut for "av human human av human human". '
                             'Vehicle tags can be passed with hashtags, eg "av#tag" "human#tag*3"')
    parser.add_argument('--env_human_kwargs', type=str, default='{}', nargs='+',
                        help='Dict of keyword arguments to pass to the IDM platoon cars controller.')
    parser.add_argument('--env_downstream', type=int, default=0, nargs='+',
                        help='If set, adds downstream speed information to the base state.')
    parser.add_argument('--env_downstream_num_segments', type=int, default=10, nargs='+',
                        help='If downstream is set, average speed and distance to this many segments is added to state.')
    parser.add_argument('--env_include_local_segment', default=False, action='store_true',
                        help='If downstream is set and this arg is set to 1, includes the local segment in state.')
    parser.add_argument('--env_inrix_mem', type=int, default=0, nargs='+',
                        help='If set to 1, inrix data will be included in memory.')
    parser.add_argument('--no_lc', type=int, default=0, nargs='+',
                        help='If set to 1, disables the lane-changing model.')
    parser.add_argument('--lc_prob', type=float, default=0, nargs='+',
                        help='If no_lc, can set a probability that the lane changing model is enabled for each rollout.')
    parser.add_argument('--lc_curriculum_iters', type=int, default=0, nargs='+',
                        help='If no_lc, can set number of iters after which lc model begins to kick in at probability lc_prob.')
    parser.add_argument('--road_grade', type=str, default=None,
                        help='Can be set to i24 or i680. If set, road grade will be included in the energy function.')
    parser.add_argument('--platoon_size', type=int, default=5,
                        help='Sets the size of the platoon to observe during training.')
    parser.add_argument('--env_speed_planner', type=int, default=0, nargs='+',
                        help='If set, adds speed planner information to the base state.')
    parser.add_argument('--acc_states', type=int, default=0, nargs='+',
                        help='If set, adds current ACC speed and gap settings into the state.')
    parser.add_argument('--acc_continuous', type=int, default=0, nargs='+',
                        help='If set, ACC output will be continuous (and clipped/rounded) instead of discrete.')
    parser.add_argument('--output_acc', default=False, action='store_true',
                        help='If set, outputs ACC settings rather than accel directly.')
    parser.add_argument('--action_delta', default=False, action='store_true',
                        help='If set with ACC, action space is in the form (-5, -1, 1, 5).')
    parser.add_argument('--jonny_style', default=False, action='store_true',
                        help='If set, calculates delta by...? ') 
    parser.add_argument('--speed_diff_reward_weight', type=float, default=0, nargs='+',
                        help='Weights speed diff reward') 
    parser.add_argument('--stripped_state', default=False, action='store_true',
                        help='If set, a stripped down state space without leader information will be used.')
    # add arg for leader_present
    parser.add_argument('--env_leader_present', type=int, default=0, nargs='+',
                        help='If set, state has flag for whether the leader is within a certain threshold')
    parser.add_argument('--env_leader_present_threshold', type=float, default=80, nargs='+',
                        help='If leader_present, sets headway threshold for when the leader is considered present')
    parser.add_argument('--env_dummy_states', type=int, default=0, nargs='+',
                        help='If set, adds this many dummy states to the state space.')
    parser.add_argument('--past_vels_state', type=int, default=0, nargs='+',
                        help='If set, includes this many past velocities in the state.')
    parser.add_argument('--past_accels_state', type=int, default=0, nargs='+',
                        help='If set, includes this many past accelerations in the state.')
    parser.add_argument('--no_failsafe', default=False, action='store_true',
                        help='If set, will not use the ACCWrappedRLVehicle failsafe for speed setting.')
    parser.add_argument('--no_gap_closing', default=False, action='store_true',
                        help='If set, will not use the ACCWrappedRLVehicle method for closing gaps above large thresholds.')
                    

    args = parser.parse_args()
    return args


def run_experiment(config):
    """Run experiment."""
    # create exp logdir
    gs_logdir = config['gs_logdir']
    gs_logdir.mkdir(parents=True, exist_ok=True)

    # create env config
    env_config = dict(DEFAULT_ENV_CONFIG)
    env_config.update({
        'horizon': config['env_horizon'],
        'max_headway': config['env_max_headway'],
        'max_time_headway': config['env_max_time_headway'],
        'discrete': config['env_discrete'],
        'num_actions': config['env_num_actions'],
        'min_accel': config['env_min_accel'],
        'max_accel': config['env_max_accel'],
        'use_fs': config['use_fs'],
        'augment_vf': config['augment_vf'],
        'vf_include_chunk_idx': config['vf_include_chunk_idx'],
        'minimal_time_headway': config['env_minimal_time_headway'],
        'minimal_time_to_collision': config['env_minimal_time_to_collision'],
        'headway_penalty': config['env_headway_penalty'],
        'accel_penalty': config['env_accel_penalty'],
        'intervention_penalty': config['env_intervention_penalty'],
        'include_thresholds': config['env_include_thresholds'],
        'penalize_energy': config['env_penalize_energy'],
        'include_idm_mpg': config['env_include_idm_mpg'],
        'num_concat_states': config['env_num_concat_states'],
        'num_concat_states_large': config['env_num_concat_states_large'],
        'num_leader_speed_memory': config['env_num_leader_speed_memory'],
        'platoon': config['env_platoon'],
        'human_kwargs': config['env_human_kwargs'],
        'downstream': config['env_downstream'],
        'downstream_num_segments': config['env_downstream_num_segments'],
        'include_local_segment': config['env_include_local_segment'],
        'inrix_mem': config['env_inrix_mem'],
        'lane_changing': not config['no_lc'],
        'lc_prob': config['lc_prob'],
        'lc_curriculum_steps': config['lc_curriculum_iters'] * config['n_steps'],
        'road_grade': config['road_grade'],
        'platoon_size': config['platoon_size'],
        'fixed_traj_path': config['traj_path'],
        'traj_dir': config['traj_dir'],
        'traj_curriculum': config['traj_curriculum'],
        'traj_curriculum_dir': config['traj_curriculum_dir'],
        'speed_planner': config['env_speed_planner'],
        'acc_states': config['acc_states'],
        'acc_continuous': config['acc_continuous'],
        # Convert curriculum frequency from iterations to steps
        'traj_curriculum_freq': config['traj_curriculum_freq'] * config['n_steps'],
        'output_acc': config['output_acc'],
        'action_delta': config['action_delta'],
        'jonny_style': config['jonny_style'],
        'speed_diff_reward_weight': config['speed_diff_reward_weight'],
        'stripped_state': config['stripped_state'],
        'leader_present': config['env_leader_present'],
        'leader_present_threshold': config['env_leader_present_threshold'],
        'dummy_states': config['env_dummy_states'],
        'past_vels_state': config['past_vels_state'],
        'past_accels_state': config['past_accels_state'],
        'no_failsafe': config['no_failsafe'],
        'no_gap_closing': config['no_gap_closing']
    })

    # create env
    multi_env = make_vec_env(TrajectoryEnv, n_envs=config['n_envs'], env_kwargs=dict(config=env_config))

    # create callbacks
    callbacks = []
    if not config['no_eval']:
        callbacks.append(TensorboardCallback(
            eval_freq=config['eval_frequency'],
            eval_at_end=True))
    callbacks += [
        LoggingCallback(
            grid_search_config=config['gs_config'],
            log_metrics=True),
        CheckpointCallback(
            save_path=gs_logdir / 'checkpoints',
            save_freq=config['cp_frequency'],
            save_at_end=True,
            s3_bucket='trajectory.env' if config['s3'] else None,
            exp_logdir=config['exp_logdir'], ),
    ]
    if config['telegram']:
        callbacks += [
            TelegramCallback(
                gs_path=gs_logdir,
            )
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
            'seed': config['seed'],
            'clip_range': 0.2,
            'clip_range_vf': 50,
            'ent_coef': config['ent_coef'],
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
        'tensorboard_log': gs_logdir,
        'verbose': 0,  # 0 no output, 1 info, 2 debug
        # 'seed': None,  # only concerns PPO and not the environment
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
    dict_to_json(configs, gs_logdir / 'configs.json')

    if config['wandb']:
        run = wandb.init(
            config=configs,
            group=config['exp_logdir'].name,
            reinit=True,
            sync_tensorboard=True,
            project="TrajectoryTraining"
        )

    # create model and start training
    model = algorithm(**train_config)
    model.learn(**learn_config)

    if config['wandb']:
        run.finish()


if __name__ == '__main__':
    # fix for macOS
    if platform.system() == 'Darwin':
        multiprocessing.set_start_method('spawn')

    # read command line arguments
    args = parse_args_train()

    # create exp logdir
    now = datetime.now()
    now_date = now.strftime('%d%b%y')
    now_time = now.strftime('%Hh%Mm%Ss')
    exp_logdir = Path(args.logdir, now_date, f'{args.expname}_{now_time}')
    exp_logdir.mkdir(parents=True, exist_ok=True)
    print(f'\nCreated experiment logdir at {exp_logdir}')

    # write params.json
    git_branches = subprocess.check_output(['git', 'branch']).decode('utf8')
    git_branch = next(filter(lambda s: s.startswith('*'), git_branches.split('\n')), '?')[2:]
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf8').split()[0]
    whoami = subprocess.check_output(['whoami']).decode('utf8').split()[0]

    exp_dict = {
        'full_command': 'python ' + ' '.join(sys.argv),
        'timestamp': datetime.timestamp(datetime.now()),
        'user': whoami,
        'git_branch': git_branch,
        'git_commit': git_commit,
        'n_cpus': multiprocessing.cpu_count(),
        'args': vars(args),
    }

    dict_to_json(exp_dict, exp_logdir / 'params.json')

    # parse command line args to separate grid search args from regular args
    fixed_config, gs_config = partition(
        vars(args).items(),
        pred=lambda kv: type(kv[1]) is list and len(kv[1]) > 1
    )

    # turn args that are a list of one element into just that element
    fixed_config = dict(map(
        lambda kv: (kv[0], kv[1][0]) if type(kv[1]) is list else kv,
        fixed_config))

    # compute cartesian product of grid search params
    try:
        gs_keys, gs_values = list(zip(*gs_config))
        grid_searches_raw = itertools.product(*gs_values)
        grid_searches = map(lambda gs: dict(zip(gs_keys, gs)), grid_searches_raw)
    except ValueError:
        grid_searches = [{}]

    # generate all configs
    configs = [{'gs_str': (gs_str := '_'.join([f'{k}={v}' for k, v in gs.items()])),
                'gs_logdir': exp_logdir / gs_str,
                'gs_config': gs,
                'exp_logdir': exp_logdir,
                **fixed_config,
                **gs} for gs in grid_searches]

    # print config and grid searches
    print('\nRunning experiment with the following config:\n')
    print('\n'.join([f'\t{k} = {v}' for k, v in fixed_config.items()]))
    if (n := len(configs)) > 1:
        print(f'\nwith a total of {n} grid searches across the following parameters:\n')
        print('\n'.join([f'\t{k} = {v}' for k, v in zip(gs_keys, gs_values)]))
    print()

    # save git diff to account for uncommited changes
    ps = subprocess.Popen(('git', 'diff', 'HEAD'), stdout=subprocess.PIPE)
    git_diff = subprocess.check_output(('cat'), stdin=ps.stdout).decode('utf8')
    ps.wait()
    if len(git_diff) > 0:
        with open(exp_logdir / 'git_diff.txt', 'w') as fp:
            print(git_diff, file=fp)

    # run experiments
    if len(configs) == 1:
        run_experiment(configs[0])
    else:
        # set environment variables so that pytorch threads don't fight each other when multithreading
        # this makes training **MUCH** faster
        # cf. https://discuss.pytorch.org/t/running-pytorch-models-in-different-processes/21638/2
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

        if configs[0]['wandb']:
            # Let wandb know that it is running in multiple processes to avoid errors
            wandb.require("service")
            wandb.setup()

        # run experiments in independent processes
        with multiprocessing.Pool(processes=(n := fixed_config['n_processes'])) as pool:
            print(f'Starting training with {n} parallel processes')
            pool.map(run_experiment, configs)

    print(f'\nTraining terminated\n\t{exp_logdir}')

    if args.telegram:
        import telegram
        import os
        bot_token = os.environ['TELEGRAM_BOT_TOKEN']
        chat_id = os.environ['TELEGRAM_CHAT_ID']
        telegram.Bot(token=bot_token).send_message(text=f'Training ended for {exp_logdir}.', chat_id=chat_id)
