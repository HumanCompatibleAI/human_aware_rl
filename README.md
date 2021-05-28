# Human-Aware Reinforcement Learning

This code is based on the work in [On the Utility of Learning about Humans for Human-AI Coordination](https://arxiv.org/abs/1910.05789). 

# Contents

To play the game with trained agents, you can use [Overcooked-Demo](https://github.com/HumanCompatibleAI/overcooked-demo).

For more information about the Overcooked-AI environment, check out [this](https://github.com/HumanCompatibleAI/overcooked_ai) repo.

* [Installation](#installation)
* [Testing](#testing)
* [Repo Structure Overview](#repo-structure-overview)
* [Usage](#usage)
* [Troubleshooting](#troubleshooting)
* [Playing With Agents](#playing-with-agents)
* [Reproducing Results](#repoducing-results)
* [Human Data](./human_aware_rl/static/human_data/README.md)

# Installation

When cloning the repository, make sure you also clone the submodules (this implementation is linked to specific commits of the submodules, and will mostly not work with more recent ones):
```
$ git clone --recursive https://github.com/HumanCompatibleAI/human_aware_rl.git
```

If you want to clone a specific branch with its submodules, use:
```
$ git clone --single-branch --branch BRANCH_NAME --recursive https://github.com/HumanCompatibleAI/human_aware_rl.git
```


## CUDA 10.0 Installation on Ubuntu 18.04
For Ubuntu 18.04, follow the direction [here](https://www.pugetsystems.com/labs/hpc/How-To-Install-CUDA-10-together-with-9-2-on-Ubuntu-18-04-with-support-for-NVIDIA-20XX-Turing-GPUs-1236/)

The only difference being the very last step. 

Instead of running 

```bash
$ sudo apt-get install cuda
```

Please run 
```bash
$ sudo apt-get install cuda-libraries-10-0
$ sudo apt-get install cuda-10-0
```

## Conda Environment Setup

Create a new conda environment and run the install script as before

[Optional Conda Installation for 18.04](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-18-04)

```bash
$ conda create -n harl_rllib python=3.7
$ conda activate harl_rllib
(harl_rllib) $ ./install.sh
```

Finally, install the latest stable version of tensorflow compatible with rllib
```bash
(harl_rllib) $ pip install tensorflow==2.0.2
```
Or, if working with gpus, install a version of tensorflow 2.*.* and cuDNN that is compatible with the available Cuda drivers. The following example works for Cuda 10.0.0. You can verify what version of Cuda is installed by running `nvcc --version`. For a full list of driver compatibility, refer [here](https://www.tensorflow.org/install/source#gpu)
```bash
(harl_rllib) $ pip install tensorflow-gpu==2.0.0
(harl_rllib) $ conda install -c anaconda cudnn=7.6.0
```

Your virtual environment should now be configured to run the rllib training code. Verify it by running the following command 

```bash
(harl_rllib) $ python -c "from ray import rllib"
```

Note: if you ever get an import error, please first check if you activated the conda env

# Testing

If set-up was successful, all unit tests and local reproducibility tests should pass. They can be run as follows

You can run all the tests with 
```bash
(harl_rllib) $ ./run_tests.sh
```

## PPO Tests
Highest level integration tests that combine self play, bc training, and ppo_bc training
```bash
(harl_rllib) $ cd human_aware_rl/ppo
(harl_rllib) human_aware_rl/ppo $ python ppo_rllib_test.py
```

## BC Tests
All tests involving creation, training, and saving of bc models. No dependency on rllib
```bash
(harl_rllib) $ cd imitation
(harl_rllib) imitation $ python behavior_cloning_tf2_test.py
```

## Rllib Tests
Tests rllib environments and models, as well as various utility functions. Does not actually test rllib training
```bash
(harl_rllib) $ cd rllib
(harl_rllib) rllib $ python tests.py
```

You should see all tests passing. 

Note: the tests are broken up into separate files because they rely on different tensorflow execution states (i.e. the bc tests run tf in eager mode, while rllib requires tensorflow to be running symbollically). Going forward, it would probably be best to standardize the tensorflow execution state, or re-write the code such that it is robust to execution state.

# Repo Structure Overview

`ppo/`:
- `ppo_rllib.py`: Primary module where code for training a PPO agent resides. This includes an rllib compatible wrapper on `OvercookedEnv`, utilities for converting rllib `Policy` classes to Overcooked `Agent`s, as well as utility functions and callbacks
- `ppo_rllib_client.py` Driver code for configuing and launching the training of an agent. More details about usage below
- `ppo_rllib_from_params_client.py`: train one agent with PPO in Overcooked with variable-MDPs 
- `ppo_rllib_test.py` Reproducibility tests for local sanity checks

`rllib/`:
- `rllib.py`: rllib agent and training utils that utilize Overcooked APIs
- `utils.py`: utils for the above
- `tests.py`: preliminary tests for the above

`imitation/`:
- `behavior_cloning_tf2.py`:  Module for training, saving, and loading a BC model
- `behavior_cloning_tf2_test.py`: Contains basic reproducibility tests as well as unit tests for the various components of the bc module.

`human/`:
- `process_data.py` script to process human data in specific formats to be used by DRL algorithms
- `data_processing_utils.py` utils for the above

`utils.py`: utils for the repo

# Usage

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


# Troubleshooting

## Tensorflow
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

# Reproducing Results

The specific results in that paper were obtained using code that is no longer in the master branch. If you are interested in reproducing results, please check out [this](https://github.com/HumanCompatibleAI/human_aware_rl/tree/neurips2019) and follow the install instructions there.
