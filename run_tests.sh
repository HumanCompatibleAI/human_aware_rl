#!/usr/bin/env bash
export RUN_ENV=local
cd ./human_aware_rl

# Create a dummy data_dir.py if the file does not already exist
[ ! -f data_dir.py ] && echo "import os; DATA_DIR = os.path.abspath('.')" >> data_dir.py

# Human data tests
cd ./human
python tests.py
cd ..

# BC tests
cd ./imitation
python behavior_cloning_tf2_test.py
cd ..

# rllib tests
cd ./rllib
python tests.py
cd ..

# PPO tests
cd ./ppo
python ppo_rllib_test.py
