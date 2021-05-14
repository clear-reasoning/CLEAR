import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--expname', type=str, default='trajectory_env')
    parser.add_argument('--use_fs', type=int, default=0)
    parser.add_argument('--plot_trajectory', type=int, default=0)

    args = parser.parse_args()
    return args