
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, Recipe
import pandas as pd
import numpy as np
import json, pickle, os
from human_aware_rl.static import *
from human_aware_rl.human.process_dataframes import csv_to_df_pickle

Recipe.configure({})

raw_dummy_data = pd.read_csv(RAW_2019_HUMAN_DATA)

def forward_port_state(state):
    if type(state) is str:
        state = json.loads(state)
    if "players" in state:
        for player in state['players']:
            if not 'held_object' in player:
                player['held_object'] = None
    if "objects" in state:
        state['objects'] = list(state['objects'].values())

    return json.dumps(OvercookedState.from_dict(state).to_dict())

def forward_port_state_row(row):
    return forward_port_state(row['state'])


raw_dummy_data['state'] = raw_dummy_data.apply(forward_port_state_row, axis=1)
raw_dummy_data.to_csv(RAW_2019_HUMAN_DATA)

csv_to_df_pickle(RAW_2019_HUMAN_DATA, CLEAN_HUMAN_DATA_DIR, '2019_hh_trials', perform_train_test_split=True)
