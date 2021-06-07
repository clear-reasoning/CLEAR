import subprocess
import os, os.path
from datetime import datetime as dt

from args import parse_args_savio


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
