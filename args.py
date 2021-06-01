import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train on the I-24 trajectoryt env.')

    # exp params
    parser.add_argument('--expname', type=str, default='test',
        help='Name for the experiment.')
    parser.add_argument('--logdir', type=str, default='./log',
        help='Experiment logs, checkpoints and tensorboard files will be saved under {logdir}/{expname}_[current_time]/.')

    parser.add_argument('--iters', type=int, default=1,
        help='Number of iterations (rollouts) to train for.'
             'Over the whole training, {iters} * {n_steps} * {n_envs} environment steps will be sampled.')
    parser.add_argument('--n_steps', type=int, default=640,
        help='Number of environment steps to sample in each rollout in each environment.'
             'This can span over less or more than the environment horizon.'
             'Ideally should be a multiple of {batch_size}.')
    parser.add_argument('--n_envs', type=int, default=1,
        help='Number of environments to run in parallel (can be set to the number of available CPUs).')

    parser.add_argument('--cp_frequency', type=int, default=10,
        help='A checkpoint of the model will be saved every {cp_frequency} iterations.'
             'Set to None to not save no checkpoints during training.'
             'Either way, a checkpoint will automatically be saved at the end of training.')
    parser.add_argument('--eval_frequency', type=int, default=10,
        help='An evaluation of the model will be done and saved to tensorboard every {eval_frequency} iterations.'
             'Set to None to run no evaluations during training.'
             'Either way, an evaluation will automatically be done at the start and at the end of training.')
    
    # training params
    parser.add_argument('--algorithm', type=str, default='PPO',
        help='RL algorithm to train with. Available options: PPO.')

    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[32, 32],
        help='Hidden layers to use for the policy and value function networks.')
    parser.add_argument('--activation', type=str, default='tanh',
        help='Non-linear activation function to use. Available options: Tanh, ReLU.')
    parser.add_argument('--optimizer', type=str, default='adam',
        help='Optimizer algorithm to use. Available options: Adam.')

    parser.add_argument('--lr', type=float, default=3e-4,
        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=64,
        help='Minibatch size.')
    parser.add_argument('--n_epochs', type=int, default=10,
        help='Number of SGD iterations per training iteration.')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='Discount factor.')
    parser.add_argument('--gae_lambda', type=float, default=0.99,
        help=' Factor for trade-off of bias vs. variance for Generalized Advantage Estimator.')

    # env params
    parser.add_argument('--env_discrete', action='store_true', default=False,
        help='If true, the environment has a discrete action space.')
    parser.add_argument('--env_num_actions', type=int, default=7,
                        help='If discrete is set, the action space is discretized by 1 and -1 with this many actions')

    args = parser.parse_args()
    return args