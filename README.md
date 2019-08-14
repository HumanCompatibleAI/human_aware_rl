# Human-Aware Reinforcement Learning

This code can be used to reproduce the results in the paper [On the Utility of Learning about Humans for Human-AI Coordination](https://bit.ly/2XoYHAm). *Note that this repository uses a specific older commit of the [overcooked_ai repository](https://github.com/HumanCompatibleAI/overcooked_ai)*, and should not be expected to work with the current version of that repository.

To play the game with trained agents, you can use [Overcooked-Demo](https://github.com/HumanCompatibleAI/overcooked-demo).

For more information about the Overcooked-AI environment, check out [this](https://github.com/HumanCompatibleAI/overcooked_ai) repo.

## Installation

When cloning the repository, make sure you also clone the submodules:
```
$ git clone --recursive git@github.com:HumanCompatibleAI/human_aware_rl.git
```

If you want to clone a specific branch with its submodules, use:
```
$ git clone --single-branch --branch BRANCH_NAME --recursive git@github.com:HumanCompatibleAI/human_aware_rl.git
```

It is useful to setup a conda environment with Python 3.7:
```
$ conda create -n harl python=3.7
$ conda activate harl
```

To complete the installation, run:
```
               $ cd human_aware_rl
human_aware_rl $ ./install.sh
```

Then install tensorflow (the GPU **or** non-GPU version depending on your setup):
```
$ pip install tensorflow==1.13.1
```

```
$ pip install tensorflow-gpu==1.13.1
```

Note that using tensorflow-gpu will not enable to pass the DRL tests due to intrinsic randomness introduced by GPU computations. We recommend to first install tensorflow (non-GPU), run the tests, and then install tensorflow-gpu.

## Verify Installation

To verify your installation, you can try running the following command from the inner `human_aware_rl` folder:

```
python run_tests.py
```

Note that most of the DRL tests rely on having the exact randomness settings that were used to generate the tests.

On OSX, you may run into an error saying that Python must be installed as a framework. You can fix it by [telling Matplotlib to use a different backend](https://markhneedham.com/blog/2018/05/04/python-runtime-error-osx-matplotlib-not-installed-as-framework-mac/).

## Repo Structure Overview


`ppo/` (both using baselines):
- `ppo.py`: train one agent with PPO in Overcooked with other agent fixed

`pbt/` (all using baselines):
- `pbt.py`: train agents with population based training in overcooked

`imitation/`:
- `behaviour_cloning.py`:  simple script to perform BC on trajectory data using baselines

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

## With JavaScript graphics

This requires converting the trained models to Tensorflow JS format, and visualizing with the [overcooked-demo](https://github.com/HumanCompatibleAI/overcooked-demo) code. First install overcooked-demo and ensure it works properly.

### Converting models to JS format

Unfortunately, converting models requires creating a new conda environment to avoid module conflicts.

Create and activate a new conda environment:
```
$ conda create -n model_conversion python=3.7
$ conda activate model_conversion
```

Run the base `setup.py` (from the inner `human_aware_rl`) and then install `tensorflowjs`:
```
human_aware_rl $ cd human_aware_rl
human_aware_rl $ python setup.py develop
human_aware_rl $ pip install tensorflowjs==0.8.5
```

To convert models in the right format, use the `convert_model_to_web.sh` script. Example usage:
```
human_aware_rl $ ./convert_model_to_web.sh ppo_runs ppo_sp_simple 193
```
where 193 is the seed number of the DRL run.

### Transferring agents to Overcooked-Demo

The converted models can be found in `human_aware_rl/data/web_models/` and should be transferred to the `static/assets` folder with the same naming as the standard models.

### Playing with newly trained agents

To play with newly trained agents, just follow the instructions in the [Overcooked-Demo](https://github.com/HumanCompatibleAI/overcooked-demo) README.

# Reproducing results

All DRL results can be reproduced by running the `.sh` scripts under `human_aware_rl/experiments/`.

All non-DRL results can be reproduced by running cells in `NeurIPS Experiments and Visualizations.ipynb`.
