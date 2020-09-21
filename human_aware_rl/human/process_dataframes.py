import random, json, copy
import numpy as np
import pandas as pd
from collections import defaultdict

from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.utils import mean_and_std_err

from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.human.data_processing_utils import convert_joint_df_trajs_to_overcooked_single, \
    extract_df_for_worker_on_layout, df_traj_to_python_joint_traj


######################
# HIGH LEVEL METHODS #
######################

def get_human_human_trajectories(layouts, dataset_type, processed=False):
    """Get human-human trajectories"""
    assert dataset_type in ["train", "test"]
    # please double check this
    from human_aware_rl.imitation.behavior_cloning_tf2 import DEFAULT_BC_PARAMS

    expert_data = {}
    for layout in layouts:
        print(layout)
        bc_params = copy.deepcopy(DEFAULT_BC_PARAMS)
        bc_params["data_params"]['train_mdps'] = [layout]
        bc_params["data_params"]['data_path'] = DATA_DIR + "human/anonymized/clean_{}_trials.pkl".format(dataset_type)
        bc_params["mdp_params"]['layout_name'] = layout
        bc_params["mdp_params"]['start_order_list'] = None
        expert_data[layout] = get_trajs_from_data(**bc_params["data_params"], silent=True, processed=processed)[0]

    return expert_data


#############################
# DATAFRAME TO TRAJECTORIES #
#############################

def get_trajs_from_data(data_path, train_mdps, ordered_trajs, processed, silent=False):
    """
    Converts and returns trajectories from dataframe at `data_path` to overcooked trajectories.
    """
    print("Loading data from {}".format(data_path))

    main_trials = pd.read_pickle(data_path)

    trajs, info = convert_joint_df_trajs_to_overcooked_single(
        main_trials,
        train_mdps,
        ordered_pairs=ordered_trajs,
        processed=processed,
        silent=silent
    )
    return trajs, info


############################
## TRAJ DISPLAY FUNCTIONS ##
############################

def interactive_from_traj_df(df_traj):
    python_traj = df_traj_to_python_joint_traj(df_traj)
    AgentEvaluator.interactive_from_traj(python_traj, traj_idx=0)


def display_interactive_by_workerid(main_trials, worker_id, limit=None):
    print("Displaying main trials for worker", worker_id)
    worker_trials = main_trials[main_trials['player_0_id'] == worker_id | main_trials['player_1_id'] == worker_id]
    count = 0
    for _, rtrials in worker_trials.groupby(['trial_id']):
        interactive_from_traj_df(rtrials)
        count += 1
        if limit is not None and count >= limit:
            return


def display_interactive_by_layout(main_trials, layout_name, limit=None):
    print("Displaying main trials for layout", layout_name)
    layout_trials = main_trials[main_trials['layout_name'] == layout_name]
    count = 0
    for wid, wtrials in layout_trials.groupby('player_0_id'):
        print("Worker: ", wid)
        for _, rtrials in wtrials.groupby(['trial_id']):
            interactive_from_traj_df(rtrials)
            count += 1
            if limit is not None and count >= limit:
                return
    for wid, wtrials in layout_trials.groupby('player_1_id'):
        print("Worker: ", wid)
        for _, rtrials in wtrials.groupby(['trial_id']):
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
        curr_layout_trials = curr_layout_trials[curr_layout_trials['score_total'] >= min_rew_fn(layout, clip_400)]
        cleaned_layout_dfs.append(curr_layout_trials)

    return pd.concat(cleaned_layout_dfs)


def get_trials_scenario_and_worker_rews(trials):
    scenario_rews = defaultdict(list)
    worker_rews = defaultdict(list)
    for _, trial in trials.groupby('trial_id'):
        datapoint = trial.iloc[0]
        layout = datapoint['layout_name']
        player_0, player_1 = datapoint['player_0_id'], datapoint['player_1_id']
        tot_rew = datapoint['score_total']
        scenario_rews[layout].append(tot_rew)
        worker_rews[player_0].append(tot_rew)
        worker_rews[player_1].append(tot_rew)
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

def train_test_split(trials, train_size=0.7, print_stats=False):
    cleaned_trials_dict = defaultdict(dict)

    layouts = np.unique(trials['layout_name'])
    for layout in layouts:
        # Gettings trials for curr layout
        curr_layout_trials = trials[trials['layout_name'] == layout]

        # Get all trial ids for the layout
        curr_trial_ids = np.unique(curr_layout_trials['trial_id'])

        # Split trials into train and test sets
        random.shuffle(curr_trial_ids)
        mid_idx = int(np.ceil(len(curr_trial_ids) * train_size))
        train_trials, test_trials = curr_trial_ids[:mid_idx], curr_trial_ids[mid_idx:]
        assert len(train_trials) > 0 and len(test_trials) > 0, "Cannot have empty split"

        # Get corresponding trials
        layout_train = curr_layout_trials[curr_layout_trials['trial_id'].isin(train_trials)]
        layout_test = curr_layout_trials[curr_layout_trials['trial_id'].isin(test_trials)]

        train_dset_avg_rew = int(np.mean(layout_train['score_total']))
        test_dset_avg_rew = int(np.mean(layout_test['score_total']))

        if print_stats:
            print(
                "Layout: {}\nNum Train Trajs: {}\nTrain Traj Average Rew: {}\nNum Test Trajs: {}\nTest Traj Average Rew: {}".format(
                    layout, len(train_trials), train_dset_avg_rew, len(test_trials), test_dset_avg_rew,
                ))

        # if train_dset_avg_rew > test_dset_avg_rew or test_dset_avg_rew - train_dset_avg_rew > 30:
        #     return None

        cleaned_trials_dict[layout]["train"] = layout_train
        cleaned_trials_dict[layout]["test"] = layout_test
    return cleaned_trials_dict


def remove_worker(trials, worker_id):
    return trials[trials["player_0_id"] != worker_id & trials['player_1_id'] != worker_id]


def remove_worker_on_map(trials, workerid_num, layout):
    to_remove = ((trials['player_0_id'] == workerid_num) | (trials['player_1_id'] == workerid_num)) & (
                trials['layout_name'] == layout)
    to_keep = ~to_remove
    assert to_remove.sum() > 0
    return trials[to_keep]


## Human-human dataframe processing functions

def format_hh_trials_df(trials, clip_400):
    """Get trials for layouts in standard format for data exploration, with right reward and length information"""
    layouts = np.unique(trials['layout_name'])
    print("Layouts found", layouts)

    if clip_400:
        trials = trials[trials["cur_gameloop"] <= 400]
    # Add game length for each round
    trials = trials.join(trials.groupby(['trial_id'])['cur_gameloop'].count(), on=['trial_id'], rsuffix='_total')

    # Calculate total reward for each round
    trials = trials.join(trials.groupby(['trial_id'])['score'].max(), on=['trial_id'], rsuffix='_total')

    return trials

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
    main_trials = main_trials.join(
        main_trials.groupby(['participant_id', 'round_num', 'layout_name'])['cur_gameloop'].count(),
        on=['participant_id', 'round_num', 'layout_name'], rsuffix='_total')

    # Calculate total reward for each round
    main_trials = main_trials.join(
        main_trials.groupby(['participant_id', 'round_num', 'layout_name'])['reward_norm'].sum(),
        on=['participant_id', 'round_num', 'layout_name'], rsuffix='_total')
    return main_trials


def add_means_and_stds_from_df(data, main_trials, algo_name):
    """Calculate means and SEs for each layout, and add them to the data dictionary under algo name `algo`"""
    layouts = ['asymmetric_advantages', 'coordination_ring', 'cramped_room', 'random0', 'random3']
    for layout in layouts:
        layout_trials = main_trials[main_trials['layout_name'] == layout]

        idx_1_workers = []
        idx_0_workers = []
        for worker_id in layout_trials['player_0_id'].unique():

            if layout_trials[layout_trials['player_0_id'] == worker_id]['player_0_is_human'][0]:
                idx_0_workers.append(worker_id)

        for worker_id in layout_trials['player_1_id'].unique():

            if layout_trials[layout_trials['player_1_id'] == worker_id]['player_1_is_human'][0]:
                idx_1_workers.append(worker_id)

        idx_0_trials = layout_trials[layout_trials['player_0_id'].isin(idx_0_workers)]
        data[layout][algo_name + "_0"] = mean_and_std_err(idx_0_trials.groupby('player_0_id')['score_total'].mean())

        idx_1_trials = layout_trials[layout_trials['plaer_1_id'].isin(idx_1_workers)]
        data[layout][algo_name + "_1"] = mean_and_std_err(idx_1_trials.groupby('plaer_1_id')['score_total'].mean())
