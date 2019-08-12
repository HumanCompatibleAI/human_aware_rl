# Human-Aware Reinforcement Learning

## Installation

When cloning the repository, make sure you also clone the submodules:
```
git clone --recursive git@github.com:HumanCompatibleAI/human_aware_rl.git
```

If you want to clone a specific branch with its submodules, use:
```
git clone --single-branch --branch BRANCH_NAME --recursive git@github.com:HumanCompatibleAI/human_aware_rl.git
```

It is useful to setup a conda environment with Python 3.7:
```
conda create -n harl python=3.7
conda activate harl
```

To complete the installation, run:
```
install.sh
```

Then install tensorflow (the GPU **or** non-GPU version depending on your setup):
```
pip install tensorflow==1.13.1
```

```
pip install tensorflow-gpu==1.13.1
```

## Verify Installation

To verify your installation, you can try running the following command from the inner `human_aware_rl` folder:

```
python run_tests.py
```

## Repo Structure Overview


`ppo/` (both using baselines):
- `ppo.py`: train one agent with PPO in Overcooked with other agent fixed

`pbt/` (all using baselines):
- `pbt.py`: train agents with population based training in overcooked

`imitation/`:
- `behaviour_cloning.py`:  simple script to perform BC on trajectory data using baselines
- `gail.py`: script to perform GAIL using stable-baselines

`human/`:
- `process_data.py` script to process human data in specific formats to be used by DRL algorithms
- `data_processing_utils.py` utils for the above

`experiments/`: folder with experiment scripts used to generate experimental results in the paper

`baselines_utils.py`: utility functions used for `pbt.py`
`overcooked_interactive.py`: script to play Overcooked in terminal against trained agents
`run_tests.py`: script to run all tests

# Playing with trained agents

## In terminal-graphics

To play with trained agents in the terminal, use `overcooked_interactive.py`. A sample command is:

`python overcooked_interactive.py -t bc -r simple_bc_test_seed4`

Playing requires not clicking away from the terminal window.

## With javascript graphics

This requires transferring the trained models to the [Overcooked-Demo](https://github.com/HumanCompatibleAI/overcooked-demo) code.

### Converting models to JS format

Unfortunately, converting models requires creating a new conda environment to avoid module conflicts.

Create and activate a new conda environment.

Run the base `setup.py` (from `human_aware_rl`) and then install `tensorflowjs`:
```
pip install tensorflowjs==0.8.5
```

To convert models in the right format, use the `convert_model_to_web.sh` script. Example usage:
```
convert_model_to_web.sh ppo_runs ppo_sp_simple 193
```
where 193 is the seed number of the DRL run.

### Transferring agents to Overcooked-Demo

The converted models can be found in `human_aware_rl/data/web_models/` and should be transferred to the `static/assets` folder with the same naming as the standard models.

### Playing with agents

To play with the trained agents, just follow the instructions in the [Overcooked-Demo](https://github.com/HumanCompatibleAI/overcooked-demo) README.

# Reproducing results

All DRL results can be reproduced by running the `.sh` scripts under `human_aware_rl/experiments/`.

All non-DRL results can be reproduced by running cells in `NeurIPS Experiments and Visualizations.ipynb`.
