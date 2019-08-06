import json
import random
import numpy as np
import pandas as pd
from collections import defaultdict

from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.utils import mean_and_std_err

from human_aware_rl.human.data_processing_utils import convert_joint_df_trajs_to_overcooked_single, extract_df_for_worker_on_layout, df_traj_to_python_joint_traj


#############################
# DATAFRAME TO TRAJECTORIES #
#############################

def get_trajs_from_data(data_path, train_mdps, ordered_trajs, human_ai_trajs):
    """
    Converts and returns trajectories from dataframe at `data_path` to overcooked trajectories.
    """
    print("Loading data from {}".format(data_path))

    main_trials = pd.read_pickle(data_path)
    all_workers = list(main_trials['workerid_num'].unique())

    trajs = convert_joint_df_trajs_to_overcooked_single(
        main_trials,
        all_workers,
        train_mdps,
        ordered_pairs=ordered_trajs,
        human_ai_trajs=human_ai_trajs
    )
    
    return trajs

def get_overcooked_traj_for_worker_layout(main_trials, worker_id, layout_name, complete_traj=True):
    """
    Extract trajectory for specific worker-layout pair and then return trajectory data 
    in standard format, plus some metadata
    """
    one_traj_df = extract_df_for_worker_on_layout(main_trials, worker_id, layout_name)
    trajectory, metadata = df_traj_to_python_joint_traj(one_traj_df, complete_traj)
    
    if trajectory is None:
        print("Layout {} is missing from worker {}".format(layout_name, worker_id))

    return trajectory, metadata

def save_npz_file(trajs, output_filename):
    AgentEvaluator.save_traj_in_stable_baselines_format(trajs, output_filename)


############################
## TRAJ DISPLAY FUNCTIONS ##
############################

def interactive_from_traj_df(df_traj):
    python_traj, _ = df_traj_to_python_joint_traj(df_traj)
    AgentEvaluator.interactive_from_traj(python_traj, traj_idx=0)
    
def display_interactive_by_workerid(main_trials, worker_id, limit=None):
    print("Displaying main trials for worker", worker_id)
    worker_trials = main_trials[main_trials['workerid_num'] == worker_id]
    count = 0
    for (r, layout_name), rtrials in worker_trials.groupby(['round_num', 'layout_name']):
        interactive_from_traj_df(rtrials)
        count += 1
        if limit is not None and count >= limit:
            return
        
def display_interactive_by_layout(main_trials, layout_name, limit=None):
    print("Displaying main trials for layout", layout_name)
    layout_trials = main_trials[main_trials['layout_name'] == layout_name]
    count = 0
    for wid, wtrials in layout_trials.groupby('workerid_num'):
        print("Worker: ", wid)
        for (r, layout_name), rtrials in wtrials.groupby(['round_num', 'layout_name']):
            interactive_from_traj_df(rtrials)
            count += 1
            if limit is not None and count >= limit:
                return


############################
# DATAFRAME PRE-PROCESSING #
############################

# General utility functions

def remove_rounds_with_low_rewards(trials, min_rew_fn, clip_400):
    layouts = np.unique(trials['layout_name'])
    
    cleaned_layout_dfs = []
    for layout in layouts:
        # Gettings trials for curr layout
        curr_layout_trials = trials[trials['layout_name'] == layout]

        # Discarding ones with low reward (corresponding to rew that just one agent could have gotten on their own)
        curr_layout_trials = curr_layout_trials[curr_layout_trials['reward_norm_total'] >= min_rew_fn(layout, clip_400)]
        cleaned_layout_dfs.append(curr_layout_trials)

    return pd.concat(cleaned_layout_dfs)

def get_trials_scenario_and_worker_rews(trials):
    scenario_rews = defaultdict(list)
    worker_rews = defaultdict(list)
    for wid, wtrials in trials.groupby('workerid_num'):
        for (r, layout_name), rtrials in wtrials.groupby(['round_num', 'layout_name']):
            tot_rew = sum(rtrials.reward_norm)
            scenario_rews[layout_name].append(tot_rew)
            worker_rews[wid].append(tot_rew)
    return dict(scenario_rews), dict(worker_rews)

def get_dict_stats(d):
    new_d = d.copy()
    for k, v in d.items():
        new_d[k] = {
            'mean': np.mean(v), 
            'standard_error': np.std(v) / np.sqrt(len(v)),
            'max': np.max(v),
            'n': len(v)
        }
    return new_d

def train_test_split(trials, print_stats=False):
    cleaned_trials_dict = defaultdict(dict)
    
    layouts = np.unique(trials['layout_name'])
    for layout in layouts:
        # Gettings trials for curr layout
        curr_layout_trials = trials[trials['layout_name'] == layout]

        # Get all worker ids for the layout
        curr_layout_workers = list(curr_layout_trials.drop_duplicates('workerid_num')['workerid_num'])

        # Split workers into train and test sets
        random.shuffle(curr_layout_workers)
        mid_idx = int(np.ceil(len(curr_layout_workers) / 2))
        train_workers, test_workers = curr_layout_workers[:mid_idx], curr_layout_workers[mid_idx:]

        # Get corresponding trials
        layout_train = curr_layout_trials[curr_layout_trials['workerid_num'].isin(train_workers)]
        layout_test = curr_layout_trials[curr_layout_trials['workerid_num'].isin(test_workers)]

        train_dset_avg_rew = int(np.mean(layout_train['reward_norm_total']))
        test_dset_avg_rew = int(np.mean(layout_test['reward_norm_total']))
        
        if print_stats:
            print(layout, len(train_workers), train_dset_avg_rew, len(test_workers), test_dset_avg_rew)

        if train_dset_avg_rew > test_dset_avg_rew or test_dset_avg_rew - train_dset_avg_rew > 30:
            return None

        cleaned_trials_dict[layout]["train"] = layout_train
        cleaned_trials_dict[layout]["test"] = layout_test
    return cleaned_trials_dict

def remove_worker(trials, workerid_num):
    return trials[trials["workerid_num"] != workerid_num]

def remove_worker_on_map(trials, workerid_num, layout):
    to_remove = (trials['workerid_num'] == workerid_num) & (trials['layout_name'] == layout)
    to_keep = ~to_remove
    assert to_remove.sum() > 0
    return trials[to_keep]


## Human-human dataframe processing functions

def format_hh_trials_df(trials, clip_400):
    """Get trials for layouts in standard format for data exploration, with right reward and length information"""
    layouts = np.unique(trials['layout_name'])
    print("Layouts found", layouts)
    
    main_trials = trials[trials['is_leader']]

    # Change all non-zero rewards to 20 (in web interface were == 5)
    main_trials['reward_norm'] = np.where(main_trials['reward'] != 0.0, main_trials['reward'] * 4.0, 0.0)

    if clip_400:
        main_trials = main_trials[main_trials["cur_gameloop"] <= 400]
        
    # Add game length for each round
    main_trials = main_trials.join(main_trials.groupby(['workerid_num', 'round_num', 'layout_name'])['cur_gameloop'].count(), on=['workerid_num', 'round_num', 'layout_name'], rsuffix='_total')

    # Calculate total reward for each round
    main_trials = main_trials.join(main_trials.groupby(['workerid_num', 'round_num', 'layout_name'])['reward_norm'].sum(), on=['workerid_num', 'round_num', 'layout_name'], rsuffix='_total')

    return main_trials

def min_hh_rew_per_scenario(layout, clip_400):
    if layout == "cramped_room":
        min_rew = 220
    elif layout == "asymmetric_advantages":
        min_rew = 280
    elif layout == "coordination_ring":
        min_rew = 150
    elif layout == "random0":
        min_rew = 160
    elif layout == "random3":
        min_rew = 180
        
    if clip_400:
        min_rew /= 3

    return min_rew

## Human-AI dataframe data processing functions

def trial_type_by_unique_id_dict(trial_questions_df):
    trial_type_dict = {}
    unique_ids = trial_questions_df['workerid'].unique()
    for unique_id in unique_ids:
        person_data = trial_questions_df[trial_questions_df['workerid'] == unique_id]
        model_type, player_index = person_data['MODEL_TYPE'].iloc[0], int(person_data['PLAYER_INDEX'].iloc[0])
        trial_type_dict[unique_id] = (model_type, player_index)
    return trial_type_dict

def format_hai_trials_df(full_hai_trials, trial_type_dict):
    d = []
    for row_idx, row in full_hai_trials.iterrows():
        event_data = json.loads(row['event'])
        uniqueid = row['workerid']
        event_data['workerid'] = uniqueid
        event_data['model_type'], event_data['player_index'] = trial_type_dict[uniqueid]
        d.append(event_data)
    trials = pd.DataFrame(d)
    print("Unpacked event field")
    
    main_trials = trials[trials['round_type'] == "main"]
    main_trials['reward_norm'] = np.where(main_trials['reward'] != 0.0, 20.0, 0.0)
    
    # Add game length for each round
    main_trials = main_trials.join(main_trials.groupby(['participant_id', 'round_num', 'layout_name'])['cur_gameloop'].count(), on=['participant_id', 'round_num', 'layout_name'], rsuffix='_total')
    
    # Calculate total reward for each round
    main_trials = main_trials.join(main_trials.groupby(['participant_id', 'round_num', 'layout_name'])['reward_norm'].sum(), on=['participant_id', 'round_num', 'layout_name'], rsuffix='_total')
    return main_trials

def is_human_idx_1(trial_df, workerid_num):
    return (trial_df.groupby('workerid_num')['player_index'].sum() > 0)[workerid_num]

def add_means_and_stds_from_df(data, main_trials, algo_name):
    """Calculate means and SEs for each layout, and add them to the data dictionary under algo name `algo`"""
    layouts = ['asymmetric_advantages', 'coordination_ring', 'cramped_room', 'random0', 'random3']
    for layout in layouts:
        layout_trials = main_trials[main_trials['layout_name'] == layout]
        
        idx_1_workers = []
        idx_0_workers = []
        for worker_id in layout_trials['workerid_num'].unique():

            if is_human_idx_1(layout_trials, worker_id):
                idx_1_workers.append(worker_id)
            else:
                idx_0_workers.append(worker_id)
    
        idx_0_trials = layout_trials[layout_trials['workerid_num'].isin(idx_0_workers)]
        data[layout][algo_name + "_0"] = mean_and_std_err(idx_0_trials.groupby('workerid_num')['reward_norm_total'].mean())
        
        idx_1_trials = layout_trials[layout_trials['workerid_num'].isin(idx_1_workers)]
        data[layout][algo_name + "_1"] = mean_and_std_err(idx_1_trials.groupby('workerid_num')['reward_norm_total'].mean())
