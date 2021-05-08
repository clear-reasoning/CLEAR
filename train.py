import ray

from env.trajectory_env import TrajectoryEnv
from args import parse_args

from ray.tune import grid_search
from progress_reporter import CLIReporter


if __name__ == '__main__':
    args = parse_args()

    exp_config = {
        'run_or_experiment': 'PPO',
        'name': 'trajectory_v0',
        'config': {
            'env': TrajectoryEnv,
            'env_config': {
                'max_accel': 1.5,
                'max_decel': 3.0,
                'horizon': 500,
                'min_speed': 0,
                'max_speed': 40,
                'max_headway': 70,
            },
            'num_gpus': 0,
            'model': {
                'vf_share_layers': False,
                'fcnet_hiddens': [64, 64],
            },
            'lr': 1e-4,
            'gamma': 0.9,
            'num_workers': 2, 
            'framework': 'torch',
        },
        'stop': {
            'training_iteration': args.iters,
        },
        'local_dir': './ray_results',
        'checkpoint_freq': 100,
        'checkpoint_at_end': True,
        'verbose': 1,
        'log_to_file': False,
        'restore': None,
        'progress_reporter': CLIReporter(metric_columns={
            'training_iteration': 'iter',
            'time_this_iter_s': 'time iter (s)',
            'time_total_s': 'total',
            'timesteps_total': 'ts',
            'episodes_this_iter': 'ep iter',
            'episodes_total': 'total',
            'episode_reward_mean': 'ep rwd mean',
            'episode_reward_min': 'min',
            'episode_reward_max': 'max',
            'episode_len_mean': 'ep len mean',
            'info/learner/default_policy/learner_stats/policy_loss': 'policy loss',
            'info/learner/default_policy/learner_stats/vf_loss': 'vf loss',
        }),
    }       

    ray.init()
    ray.tune.run(**exp_config)
    ray.shutdown()
