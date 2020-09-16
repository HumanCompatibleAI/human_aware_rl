import os

_curr_directory = os.path.dirname(os.path.abspath(__file__))

HUMAN_DATA_PATH = os.path.join(_curr_directory, "data", "clean_train_trials.pkl")