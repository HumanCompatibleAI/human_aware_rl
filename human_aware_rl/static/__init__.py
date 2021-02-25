import os

_curr_directory = os.path.dirname(os.path.abspath(__file__))

HUMAN_DATA_PATH = os.path.join(_curr_directory, "human_data", "clean_train_trials.pkl")

bc_data_dir = os.path.join(_curr_directory, "testing_data", "bc")
BC_EXPECTED_DATA_PATH = os.path.join(bc_data_dir, "expected.pickle")
JSON_TRAJECTORY_DATA_PATH = os.path.join(bc_data_dir, "cramped_room_example_trajectory.json")
PICKLE_TRAJECTORY_DATA_PATH = os.path.join(bc_data_dir, "cramped_room_example_trajectory.pickle")


ppo_data_dir = os.path.join(_curr_directory, "testing_data", "ppo")
PPO_EXPECTED_DATA_PATH = os.path.join(ppo_data_dir, "expected.pickle")
AGENTS_SCHEDULE_PATH = os.path.join(ppo_data_dir, "example_agent_schedule.json")
NON_ML_AGENTS_PARAMS_PATH = os.path.join(ppo_data_dir, "non_ml_agents_params.txt")
FEATURIZE_FNS_PATH = os.path.join(ppo_data_dir, "example_featurize_fns.json")
OBS_SPACES_PATH = os.path.join(ppo_data_dir, "example_obs_spaces.json")
MLAM_PARAMS_JSON_PATH = os.path.join(ppo_data_dir, "example_mlam_params.json")
MLAM_PARAMS_TXT_PATH = os.path.join(ppo_data_dir, "example_mlam_params.txt")