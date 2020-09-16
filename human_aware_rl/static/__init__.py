import os

_curr_directory = os.path.dirname(os.path.abspath(__file__))

HUMAN_DATA_PATH = os.path.join(_curr_directory, "human_data", "clean_train_trials.pkl")
BC_EXPECTED_DATA_PATH = os.path.join(_curr_directory, "testing_data", "bc", "expected.pickle")
PPO_EXPECTED_DATA_PATH = os.path.join(_curr_directory, "testing_data", "ppo", "expected.pickle")
