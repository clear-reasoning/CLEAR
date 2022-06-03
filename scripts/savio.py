"""Prepare to run on Savio."""
import argparse
import subprocess
import os
import os.path
from datetime import datetime as dt


def parse_args_savio():
    """Parse args on savio."""
    parser = argparse.ArgumentParser(
        description='Run an experiment on Savio.',
        epilog='Example usage: python savio.py --jobname test --mail user@coolmail.com "echo hello world"')

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


if __name__ == '__main__':
    args = parse_args_savio()

    os.makedirs(args.logdir, exist_ok=True)
    time = dt.now().strftime('%d%b%y_%Hh%Mm%Ss')

    log_path = os.path.join(args.logdir, args.jobname) + f'_{time}.out'

    savio_script = \
        f"""#!/bin/bash

#SBATCH --job-name={args.jobname}
#SBATCH --account={args.account}
#SBATCH --time={args.time}
#SBATCH --output={log_path}
#SBATCH --mail-type=ALL
#SBATCH --mail-user={args.mail}
#SBATCH --partition={args.partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

{args.command}
"""

    script_path = log_path.replace('.out', '.sh')
    with open(script_path, "w") as f:
        f.write(savio_script)
    print(f'Savio script file written at \n\n\t{script_path}\n')
    print(f'Savio logs as well as experiment stdout/stderr will be redirected to \n\n\t{log_path}\n')

    print(f'Running `sbatch --parsable {script_path}`')
    job_id = subprocess.check_output(["sbatch", "--parsable", script_path]).decode('utf8').strip()
    print(f'Done, job is pending with id {job_id}')
