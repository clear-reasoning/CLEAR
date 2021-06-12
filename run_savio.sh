#!/bin/bash

# 6/12/2021
python savio.py 'python train.py --iters 3200 --n_steps 12800 --use_fs 0 \
--n_envs 1 --gae_lambda 0.99 0.95 0.9 --gamma 0.99 0.95 --lr .0003 .00003 --batch_size 2560 \
--cp_frequency 100 --augment_vf 1 --hidden_layer_size 64 --network_depth 4 --horizon 500 1000 \
----env_num_concat_states 1 5' --jobname test --mail vinitsky.eugene@gmail.com