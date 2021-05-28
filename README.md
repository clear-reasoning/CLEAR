# Trajectory Training

## Usage

Install libraries, cf. `setup.sh`

See training params with `python train.py -h`

## TODOs

- **Implement the logging trajectory data into tensorboard in TensorboardCallback**
- **Clean env and write env params in argparse**
- **Fix failsafes**
- Fix `test_env.py` which is obsolete right now
- Handle lane changes better in data_loader.py (right now just builds a trajectory from the collected leader velocities)
- Simulate lane changes by arbitrarily reducing the headway if it is large enough (and maybe sampling a different trajectory if it is about the same avg speed although that's probably overkill since controller doesn't have memory)
- Implement grid search (eg. could subprocess several model.learn with custom tb_log_names specifying each hyperparams)
- Can try other algos (TD3, SAC..) (only supports PPO for now)