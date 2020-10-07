#!/usr/bin/env bash
export RUN_ENV=local
cd ./human_aware_rl

# Create a dummy data_dir.py if the file does not already exist
[ ! -f data_dir.py ] && echo "import os; DATA_DIR = os.path.abspath('.')" >> data_dir.py

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
cd ..

# evaluator tests
cd ./evaluate
python evaluate.py -l cramped_room -n 2 -m True -b 11 21
python evaluate.py -l cramped_room -n 2 -b 11 21 -g True
cd ..