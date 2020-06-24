# Human-Aware Reinforcement Learning

This code can be used to reproduce the results in the paper [On the Utility of Learning about Humans for Human-AI Coordination](https://arxiv.org/abs/1910.05789). *Note that this repository uses a specific older commit of the [overcooked_ai repository](https://github.com/HumanCompatibleAI/overcooked_ai)*, and should not be expected to work with the current version of that repository.

To play the game with trained agents, you can use [Overcooked-Demo](https://github.com/HumanCompatibleAI/overcooked-demo).

For more information about the Overcooked-AI environment, check out [this](https://github.com/HumanCompatibleAI/overcooked_ai) repo.

## Installation

When cloning the repository, make sure you also clone the submodules (this implementation is linked to specific commits of the submodules, and will mostly not work with more recent ones):
```
$ git clone --recursive https://github.com/HumanCompatibleAI/human_aware_rl.git
```

If you want to clone a specific branch with its submodules, use:
```
$ git clone --single-branch --branch BRANCH_NAME --recursive https://github.com/HumanCompatibleAI/human_aware_rl.git
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

Then install tensorflow and mpi4py (the GPU **or** non-GPU version depending on your setup):
```
$ pip install tensorflow==1.13.1
$ conda install mpi4py
```

```
$ pip install tensorflow-gpu==1.13.1
$ conda install mpi4py
```

Note that using tensorflow-gpu will not enable to pass the DRL tests due to intrinsic randomness introduced by GPU computations. We recommend to first install tensorflow (non-GPU), run the tests, and then install tensorflow-gpu.

## Verify Installation

To verify your installation, you can try running the following command from the inner `human_aware_rl` folder:

```
python run_tests.py
```

Note that most of the DRL tests rely on having the exact randomness settings that were used to generate the tests (and thus will not pass on a GPU-enabled device).

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

# Rllib

Some of the agents are now compatible to be trained with rllib, using Tensorflow 2. This requires alternative installation. Make sure you are on the branch `rllib`

## Installation

Ensure that have the correct `overcooked-ai` submodule code (if you cloned directly into the `rllib` branch this should be done automatically)

```bash
$ cd overcooked_ai
overcooked_ai $ git checkout overcooked_ai_improvements
overcooked_ai $ cd ..
```

Now create a new conda environment and run the install script as before

```bash
$ conda create -n harl_rllib python=3.7
$ conda activate harl_rllib
(harl_rllib) $ ./install.sh
```

Finally, install the latest stable version of tensorflow compatible with rllib
```bash
(harl_rllib) $ pip install tensorflow
```
Or, if working with gpus, install a version of tensorflow 2.*.* and cuDNN that is compatible with the available Cuda drivers. The following example works for Cuda 10.0.0. You can verify what version of Cuda is installed by running `nvcc --version`. For a full list of driver compatibility, refer [here](https://www.tensorflow.org/install/source#gpu)
```bash
(harl_rllib) $ pip install tensorflow-gpu==2.0.0
(harl_rllib) $ conda install -c anaconda cudnn=7.6.0
```

Your virtual environment should now be configured to run the rllib training code. Verify it by running the following command 

```bash
python -c "from ray import rllib"
./run_tests.sh
```

## Testing

If set-up was successful, all unit tests and local reproducibility tests should pass. They can be run as follows

### PPO Tests
Highest level integration tests that combine self play, bc training, and ppo_bc training
```bash
$ cd human_aware_rl/ppo
human_aware_rl/ppo $ python ppo_rllib_test.py
```

### BC Tests
All tests involving creation, training, and saving of bc models. No dependency on rllib
```bash
$ cd imitation
imitation $ python behavior_cloning_tf2_test.py
```

### Rllib Tests
Tests rllib environments and models, as well as various utility functions. Does not actually test rllib training
```bash
$ cd rllib
rllib $ python tests.py
```

You should see all tests passing. 

Note: the tests are broken up into separate files because they rely on different tensorflow execution states (i.e. the bc tests run tf in eager mode, while rllib requires tensorflow to be running symbollically). Going forward, it would probably be best to standardize the tensorflow execution state, or re-write the code such that it is robust to execution state.

## Rllib code overview

`ppo/`:
- `ppo_rllib.py`: Primary module where code for training a PPO agent resides. This includes an rllib compatible wrapper on `OvercookedEnv`, utilities for converting rllib `Policy` classes to Overcooked `Agent`s, as well as utility functions and callbacks
- `ppo_rllib_test.py` Reproducibility tests for local sanity checks
- `ppo_rllib_client.py` Driver code for configuing and launching the training of an agent. More details about usage below

`imitation/`:
- `behavior_cloning_tf2.py`:  Module for training, saving, and loading a BC model
- `behavior_cloning_tf2_test.py`: Contains basic reproducibility tests as well as unit tests for the various components of the bc module.

## Usage

Before proceeding, it is important to note that there are two primary groups of hyperparameter defaults, `local` and `production`. Which is selected is controlled by the `RUN_ENV` environment variable, which defaults to `production`. In order to use local hyperparameters, run
```bash
$ export RUN_ENV=local
```

Training of agents is done through the `ppo_rllib_client.py` script. It has the following usage:

```bash
 ppo_rllib_client.py [with [<param_0>=<argument_0>] ... ]
```

For example, the following snippet trains a self play ppo agent on seed 1, 2, and 3, with learning rate `1e-3`, on the `"cramped_room"` layout for `5` iterations without using any gpus. The rest of the parameters are left to their defaults
```
(harl_rllib) ppo $ python ppo_rllib_client.py with seeds="[1, 2, 3] lr=1e-3 layout_name=cramped_room num_training_iters=5 num_gpus=0 experiment_name="my_agent"
```

For a complete list of all hyperparameters as well as their local and production defaults, refer to the `my_config` section of  `ppo_rllib_client.py`


Training results and checkpoints are stored in a directory called `~/ray_results/my_agent_<seed>_<timestamp>`. You can visualize the results using tensorboard
```bash
(harl_rllib) $ cd ~/ray_results
(harl_rllib) ray_results $ tensorboard --logdir .
```

## Troubleshooting

### Tensorflow
Many tensorflow errors are caused by the tensorflow state of execution. For example, if you get an error similar to 

```
ValueError: Could not find matching function to call loaded from the SavedModel. Got:
  Positional arguments (1 total):
    * Tensor("inputs:0", shape=(1, 62), dtype=float64)
  Keyword arguments: {}
```

or

```
NotImplementedError: Cannot convert a symbolic Tensor (model_1/logits/BiasAdd:0) to a numpy array.
```

or

```
TypeError: Variable is unhashable. Instead, use tensor.ref() as the key.
```

It is likely because the code you are running relies on tensorflow executing symbolically (or eagerly) and it is executing eagerly (or symbolically)

This can be fixed by either changing the order of imports. This is because `import tensorflow as tf` sets eager execution to true, while any `rllib` import disables eager execution. Once the execution state has been set, it cannot be changed. For example, if you require eager execution, make sure `import tensorflow as tf` comes BEFORE `from ray import rllib` and vise versa.


## 'human_aware_rl.data_dir' not found
If you encounter 
```
ModuleNotFoundError: No module named 'human_aware_rl.data_dir'
```

, please run 

```
./run_tests.sh
``` 

to initiate those variables

