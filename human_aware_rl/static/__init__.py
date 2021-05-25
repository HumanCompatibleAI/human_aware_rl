import os

_curr_directory = os.path.dirname(os.path.abspath(__file__))

# Root dir where all hunan data is located
HUMAN_DATA_DIR = os.path.join(_curr_directory, "human_data")

# Paths to pre-processed data
CLEAN_HUMAN_DATA_DIR = os.path.join(HUMAN_DATA_DIR, "cleaned")
CLEAN_HUMAN_DATA_ALL = os.path.join(CLEAN_HUMAN_DATA_DIR, "2020_hh_trials_all.pickle")
CLEAN_HUMAN_DATA_TRAIN = os.path.join(CLEAN_HUMAN_DATA_DIR, "2020_hh_trials_train.pickle")
CLEAN_HUMAN_DATA_TEST = os.path.join(CLEAN_HUMAN_DATA_DIR, "2020_hh_trials_test.pickle")

# Paths to raw data
RAW_HUMAN_DATA_PATH = os.path.join(HUMAN_DATA_DIR, 'raw', '2020_hh_trials.csv')

# Expected values for reproducibility unit tests
BC_EXPECTED_DATA_PATH = os.path.join(_curr_directory, "testing_data", "bc", "expected.pickle")
PPO_EXPECTED_DATA_PATH = os.path.join(_curr_directory, "testing_data", "ppo", "expected.pickle")

# Human data tests (smaller datasets for more efficient sanity checks)
DUMMY_HUMAN_DATA_DIR = os.path.join(HUMAN_DATA_DIR, "dummy")
DUMMY_CLEAN_HUMAN_DATA_PATH = os.path.join(DUMMY_HUMAN_DATA_DIR, "dummy_hh_trials.pickle")
DUMMY_RAW_HUMAN_DATA_PATH = os.path.join(DUMMY_HUMAN_DATA_DIR, "dummy_hh_trials.csv")
