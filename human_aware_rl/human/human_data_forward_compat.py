from human_aware_rl.human.data_processing_utils import AI_ID
import pandas as pd
import numpy as np
import os, argparse

"""
Script for converting legacy-schema human data to current schema.

Note: This script, and working with the raw CSV files in general, should only be done by advanced users.
It is recommended that most users work with the pre-processed pickle files in /human_aware_rl/data/cleaned.
See docs for more info
"""


OLD_SCHEMA = set(['Unnamed: 0', 'Unnamed: 0.1', 'cur_gameloop', 'datetime', 'is_leader', 'joint_action', 'layout', 
              'layout_name', 'next_state', 'reward', 'round_num', 'round_type', 'score', 'state', 'time_elapsed', 
              'time_left', 'is_wait', 'completed', 'run', 'workerid_num'])

NEW_SCHEMA = set(['state', 'joint_action', 'reward', 'time_left', 'score', 'time_elapsed', 'cur_gameloop', 'layout', 
              'layout_name', 'trial_id', 'player_0_id', 'player_1_id', 'player_0_is_human', 'player_1_is_human'])

def write_csv(data, output_file_path):
    if os.path.exists(output_file_path):
        raise FileExistsError("File {} already exists, aborting to avoid overwriting".format(output_file_path))
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data.to_csv(output_file_path, index=False)


def main(input_file, output_file, is_human_ai=False):
    print("Loading data from {}...".format(input_file))
    data = pd.read_csv(input_file, header=0)
    print("Success!")


    print("Updating schema...")
    assert (set(data.columns) == OLD_SCHEMA), "Input data has unexected schema"
    
    data['trial_id'] = (data['layout_name'] != data['layout_name'].shift(1)).astype(int).cumsum() - 1
    data['pairing_id'] = (data['workerid_num'] != data['workerid_num'].shift(1)).astype(int).cumsum()


    if not is_human_ai:
        data['player_0_is_human'] = True
        data['player_1_is_human'] = True
        data['player_0_id'] = str(data['pairing_id'] * 2)
        data['player_1_id'] = str(data['pairing_id'] * 2 + 1)
    else:
        data['player_0_is_human'] = True
        data['player_1_is_human'] = False
        data['player_0_id'] = str(data['pairing_id'])
        data['player_1_id'] = AI_ID

    columns_to_drop = (OLD_SCHEMA - NEW_SCHEMA).union(set(['pairing_id']))

    data = data.drop(columns=columns_to_drop)

    assert (set(data.columns == NEW_SCHEMA)), "Output data has misformed schema"

    print("Success!")

    print("Writing data to {}...".format(output_file))
    write_csv(data, output_file)
    print("Success!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', type=str, required=True, help='path to old-schema data')
    parser.add_argument('--output_file', '-o', type=str, required=True, help='path to save new-schema data')
    parser.add_argument('--is_human_ai', '-ai', action='store_true', help='Provide this flag if data from human-AI games')

    args = vars(parser.parse_args())
    main(**args)



    

