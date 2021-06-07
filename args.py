import argparse


def parse_args_train():
    parser = argparse.ArgumentParser(description='Train on the I-24 trajectory env.')

    # exp params
    parser.add_argument('--expname', type=str, default='test',
        help='Name for the experiment.')
    parser.add_argument('--logdir', type=str, default='./log',
        help='Experiment logs, checkpoints and tensorboard files will be saved under {logdir}/{expname}_[current_time]/.')
    parser.add_argument('--n_processes', type=int, default=1,
        help='Number of processes to run in parallel. Useful when running grid searches.'
             'Can be more than the number of available CPUs.')

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
        help='An evaluation of the model will be done and saved to tensorboard every {eval_frequency} iterations.'
             'Set to None to run no evaluations during training.'
             'Either way, an evaluation will automatically be done at the start and at the end of training.')
    
    # training params
    parser.add_argument('--algorithm', type=str, default='PPO', nargs='+',
        help='RL algorithm to train with. Available options: PPO.')

    parser.add_argument('--hidden_layer_size', type=int, default=32, nargs='+',
        help='Hidden layer size to use for the policy and value function networks.'
             'The networks will be composed of {network_depth} hidden layers of size {hidden_layer_size}.')
    parser.add_argument('--network_depth', type=int, default=2, nargs='+',
        help='Number of hidden layers to use for the policy and value function networks.'
             'The networks will be composed of {network_depth} hidden layers of size {hidden_layer_size}.')
    parser.add_argument('--activation', type=str, default='tanh', nargs='+',
        help='Non-linear activation function to use. Available options: Tanh, ReLU.')
    parser.add_argument('--optimizer', type=str, default='adam', nargs='+',
        help='Optimizer algorithm to use. Available options: Adam.')

    parser.add_argument('--lr', type=float, default=3e-4, nargs='+',
        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=64, nargs='+',
        help='Minibatch size.')
    parser.add_argument('--n_epochs', type=int, default=10, nargs='+',
        help='Number of SGD iterations per training iteration.')
    parser.add_argument('--gamma', type=float, default=0.99, nargs='+',
        help='Discount factor.')
    parser.add_argument('--gae_lambda', type=float, default=0.99, nargs='+',
        help=' Factor for trade-off of bias vs. variance for Generalized Advantage Estimator.')

    parser.add_argument('--augment_vf', type=int, default=0, nargs='+',
                        help='If true, the value function will be augmented with info stored in the extra_obs'
                             'key of the info dict.')
    # env params
    parser.add_argument('--env_discrete', type=int, default=0, nargs='+',
        help='If true, the environment has a discrete action space.')
    parser.add_argument('--env_num_actions', type=int, default=7, nargs='+',
        help='If discrete is set, the action space is discretized by 1 and -1 with this many actions')
    parser.add_argument('--use_fs', type=int, default=0, nargs='+',
        help='If true, use a FollowerStopper wrapper.')

    args = parser.parse_args()
    return args

def parse_args_savio():
    parser = argparse.ArgumentParser(
        description='Run an experiment on Savio.',
        epilog=f'Example usage: python savio.py --jobname test --mail user@coolmail.com "echo hello world"')

    parser.add_argument('command', type=str, help='Command to run the experiment.')
    parser.add_argument('--jobname', type=str, default='test',
        help='Name for the job.')
    parser.add_argument('--logdir', type=str, default='slurm_logs',
        help='Logdir for experiment logs.')
    parser.add_argument('--mail', type=str, default=None,
        help='Email address where to send experiment status (started, failed, finished).'
             'Leave to None to receive no emails.')
    parser.add_argument('--partition', type=str, default='savio',
        help='Partition to run the experiment on.')
    parser.add_argument('--account', type=str, default='ac_mixedav',
        help='Account to use for running the experiment.')
    parser.add_argument('--time', type=str, default='24:00:00',
        help='Maximum running time of the experiment in hh:mm:ss format, maximum 72:00:00.')

    args = parser.parse_args()
    return args