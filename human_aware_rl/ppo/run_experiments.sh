#!/usr/bin/env bash
python ppo_rllib_client.py with seeds="[1, 2, 3]" lr=5e-4 vf_loss_coeff=1e-4 num_training_iters=500 layout_name="soup_coordination" experiment_name="soup_coordination_sp_default"
