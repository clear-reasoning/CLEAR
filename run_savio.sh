#!/bin/bash

# 6/12/2021
python savio.py 'python train.py --iters 3200 --n_steps 12800 --use_fs 0 \
--n_envs 1 --gae_lambda 0.99 0.95 0.9 --gamma 0.99 0.95 --lr .0003 .00003 --batch_size 2560 \
--cp_frequency 100 --augment_vf 1 --hidden_layer_size 64 --network_depth 4 --env_horizon 500 1000 \
--env_num_concat_states 1 5 --n_processes 20 --eval_frequency 100' --jobname test --mail vinitsky.eugene@gmail.com

python savio.py 'python train.py --iters 3200 --n_steps 12800 --use_fs 0 \
--n_envs 1 --gae_lambda 0.99 0.95 0.9 --gamma 0.99 0.95 --batch_size 2560 \
--cp_frequency 100 --augment_vf 1 --hidden_layer_size 64 --network_depth 4 --env_horizon 500 1000 \
--env_num_concat_states 1 5 10 --n_processes 20 --eval_frequency 100' --jobname test --mail vinitsky.eugene@gmail.com

# 15/06/2021
python savio.py "python train.py --logdir /global/scratch/$USER --s3 --expname rollo --iters 3200 --n_steps 25600 --use_fs 0 \
--n_envs 1 --gae_lambda 0.99 0.95 0.9 --gamma 0.99 0.95 --lr .0003 .00003 --batch_size 5120 \
--cp_frequency 100 --augment_vf 1 --hidden_layer_size 64 --network_depth 4 --env_horizon 500 1000 \
--env_num_concat_states 1 5 --n_processes 20 --eval_frequency 100" --jobname rollo --mail nathan.lichtle@gmail.com

python savio.py "python train.py --logdir /global/scratch/$USER --s3 --expname oslo --iters 3200 --n_steps 25600 --use_fs 0 \
--n_envs 1 --gae_lambda 0.99 0.95 0.9 --gamma 0.99 0.95 --batch_size 5120 \
--cp_frequency 100 --augment_vf 1 --hidden_layer_size 64 --network_depth 4 --env_horizon 500 1000 \
--env_num_concat_states 1 5 10 --n_processes 20 --eval_frequency 100" --jobname oslo --mail nathan.lichtle@gmail.com

# 16/06/2021 - branch modified_reward_func
python savio.py "python train.py --logdir /global/scratch/$USER --s3 --expname feca --iters 3200 --n_steps 25600 --use_fs 0 \
--n_envs 1 --gae_lambda 0.99 0.95 0.9 --gamma 0.99 0.95 --lr .0003 .00003 --batch_size 5120 \
--cp_frequency 100 --augment_vf 1 --hidden_layer_size 64 --network_depth 4 --env_horizon 500 1000 \
--env_num_concat_states 1 5 --n_processes 20 --eval_frequency 50" --jobname feca --mail nathan.lichtle@gmail.com

python savio.py "python train.py --logdir /global/scratch/$USER --s3 --expname cra --iters 3200 --n_steps 25600 --use_fs 0 \
--n_envs 1 --gae_lambda 0.99 0.95 0.9 --gamma 0.99 0.95 --batch_size 5120 \
--cp_frequency 100 --augment_vf 1 --hidden_layer_size 64 --network_depth 4 --env_horizon 500 1000 \
--env_num_concat_states 1 5 10 --n_processes 20 --eval_frequency 50" --jobname cra --mail nathan.lichtle@gmail.com
