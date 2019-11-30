#!/usr/bin/env bash

python main.py \
--e_arch cnn-3-32-64-128-256-512 \
--d_arch cnn-512-256-128-64-32-3 \
--z_dim 10 \
-conditional \
--n_epochs 200 \
--pretrain 200 \
--batch_size 64 \
--N 1 \
--seed 0 \
--mode v \
--data_dir data/randact_traj_length_10_n_trials_10_n_contexts_10.npy \
--test_dir data/test_context_10.npy \
#--data_dir data/randact_traj_length_20_n_trials_50_n_contexts_150.npy \
#--test_dir data/test_context_20.npy \
