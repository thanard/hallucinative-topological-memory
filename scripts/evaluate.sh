#!/usr/bin/env bash

python main.py \
--mode e \
--z_dim 100 \
-conditional \
--loadpath_v out/vae/var/vae-5-last-5 \
--loadpath_c out/cpc/var/cpc-5-last-5 \
--loadpath_a out/actor/var/actor-5-last-5