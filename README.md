# Human-Robot Coordination

## Installation

When cloning the repository, make sure you also clone the submodules.

```
git clone --recursive git@github.com:HumanCompatibleAI/human_aware_rl.git
```

If you want to clone a specific branch with its submodules, use:
```
git clone --single-branch --branch BRANCH_NAME --recursive git@github.com:HumanCompatibleAI/human_aware_rl.git
```

It is useful to setup a conda environment with Python 3.7:

```
conda create -n hrc python=3.7
conda activate hrc
```

To complete the installation, run:

```
install.sh
```

Then install tensorflow (the GPU or non-GPU version depending on your setup):
```
pip install tensorflow==1.13.1
```

```
pip install tensorflow-gpu==1.13.1
```

## Verify Installation

To verify your installation, you can try running the following command from the inner `human_aware_rl` folder:

```
python run_tests.py -f
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