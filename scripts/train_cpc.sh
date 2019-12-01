#!/usr/bin/env bash

python main.py \
--mode c \
--z_dim 100 \
-conditional \
--n_epochs 20 \
--pretrain 0 \
--batch_size 32 \
--N 50 \
--seed 0 \
--data_dir data/randact_traj_length_20_n_trials_50_n_contexts_150_sharper.npy \
--test_dir data/test_context_20_sharper.npy \
--loadpath_v out/vae/var/vae-5-last-5 \
--prefix cpc
