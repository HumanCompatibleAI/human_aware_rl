#!/usr/bin/env bash
# python ppo_rllib_client.py with seeds="[1, 2, 3]" lr=5e-4 vf_loss_coeff=1e-4 num_training_iters=500 layout_name="soup_coordination" experiment_name="soup_coordination_sp_default"
export RUN_ENV=local
#python ppo_rllib_client.py with seeds="[3]" lr=5e-4 vf_loss_coeff=1e-4 num_training_iters=20 layout_name="soup_coordination" experiment_name="soup_coordination_sp_dummy"
python ppo_rllib_client.py with seeds="[3]" lr=5e-4 vf_loss_coeff=1e-4 num_training_iters=20 layout_name="soup_coordination" experiment_name="soup_coordination_bc_opt_dummy" bc_model_dir="/Users/nathan/bair/human_aware_rl/human_aware_rl/data/bc_runs/soup_coord_all_100_epochs_weighted" bc_opt=True opt_path="/Users/nathan/ray_results/soup_coordination_sp_dummy_3_2021-06-01_21-49-245h245r_s/checkpoint_000020/checkpoint-20" bc_schedule="[(0, 1.0), (0, 1.0)]"
