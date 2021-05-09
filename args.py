import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--expname', type=str, default='trajectory_env')

    args = parser.parse_args()
    return args