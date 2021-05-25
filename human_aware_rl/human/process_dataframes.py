import random, json, copy, os
import numpy as np
import pandas as pd
from collections import defaultdict

from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.utils import mean_and_std_err

from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.human.data_processing_utils import convert_joint_df_trajs_to_overcooked_single, df_traj_to_python_joint_traj, is_button_press, is_interact


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

def csv_to_df_pickle(csv_path, out_dir, out_file_prefix, button_presses_threshold=0.25):
    """
    High level function that converts raw CSV data into well formatted and cleaned pickled pandas dataframes.

    Arguments:
        - csv_path (str): Full path to human csv data
        - out_dir(str): Full path to directory where cleaned data will be saved
        - out_file_prefix(str): common prefix for all saved files
        - button_presses_threshold (float): minimum button presses per timestep over rollout required to 
            keep entire game

    After running, the following files are created

        /{out_dir}
            - {out_file_prefix}_all.pickle
            - {out_file_prefix}_train.pickle
            - {out_file_prefix}_test.pickle

    Returns:
        - clean_trials (pd.DataFrame): Dataframe containing _all_ cleaned and formatted transitions
    """
    all_trials = pd.read_csv(csv_path)
    all_trials = format_trials_df(all_trials, False)
    def filter_func(row):
        return row['button_presses_per_timstep'] >= button_presses_threshold
    clean_trials = filter_trials(all_trials, filter_func)
    cleaned_trials_dict = train_test_split(clean_trials)
    
    layouts = np.unique(clean_trials['layout_name'])
    train_trials = pd.concat([cleaned_trials_dict[layout]["train"] for layout in layouts])
    test_trials = pd.concat([cleaned_trials_dict[layout]["test"] for layout in layouts])
    clean_trials = pd.concat([train_trials, test_trials])

    full_outfile_prefix = os.path.join(out_dir, out_file_prefix)
    clean_trials.to_pickle(full_outfile_prefix + "_all.pickle")
    train_trials.to_pickle(full_outfile_prefix + "_train.pickle")
    test_trials.to_pickle(full_outfile_prefix + "_test.pickle")

    return clean_trials


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

def filter_trials(trials, filter):
    """
    Prune games based on user-defined fileter function

    Note: 'filter' must accept a single row as input and whether the entire trial should be kept
    based on its first row
    """
    trial_ids = np.unique(trials['trial_id'])

    cleaned_trial_dfs = []
    for trial_id in trial_ids:
        curr_trial = trials[trials['trial_id'] == trial_id]

        # Discard entire trials based on filter function applied to first row
        element = curr_trial.iloc[0]
        keep = filter(element)
        if keep:
            cleaned_trial_dfs.append(curr_trial)

    return pd.concat(cleaned_trial_dfs)

def filter_transitions(trials, filter):
    """
    Prune games based on user-defined fileter function

    Note: 'filter' must accept a pandas Series as input and return a Series of booleans
    where the ith boolean is True if the ith entry should be kept
    """
    trial_ids = np.unique(trials['trial_id'])

    cleaned_trial_dfs = []
    for trial_id in trial_ids:
        curr_trial = trials[trials['trial_id'] == trial_id]

        # Discard entire trials based on filter function applied to first row
        keep = filter(curr_trial)
        curr_trial_kept = curr_trial[keep]
        cleaned_trial_dfs.append(curr_trial_kept)

    return pd.concat(cleaned_trial_dfs)


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

def _add_interactivity_metrics(trials):
    # this method is non-destructive
    trials = trials.copy()

    # whether any human INTERACT actions were performed
    is_interact_row = lambda row : int(np.sum(np.array([row['player_0_is_human'], row['player_1_is_human']]) * is_interact(row['joint_action'])) > 0)
    # Whehter any human keyboard stroked were performed
    is_button_press_row = lambda row : int(np.sum(np.array([row['player_0_is_human'], row['player_1_is_human']]) * is_button_press(row['joint_action'])) > 0)

    # temp column to split trajectories on INTERACTs
    trials['interact'] = trials.apply(is_interact_row, axis=1).cumsum()
    trials['dummy'] = 1

    # Temp column indicating whether current timestep required a keyboard press
    trials['button_press'] = trials.apply(is_button_press_row, axis=1)

    # Add 'button_press_total' column to each game indicating total number of keyboard strokes
    trials = trials.join(trials.groupby(['trial_id'])['button_press'].sum(), on=['trial_id'], rsuffix='_total')

    # Count number of timesteps elapsed since last human INTERACT action
    trials['timesteps_since_interact'] = trials.groupby(['interact'])['dummy'].cumsum() - 1

    # Drop temp columns
    trials = trials.drop(columns=['interact', 'dummy'])

    return trials

def format_trials_df(trials, clip_400):
    """Get trials for layouts in standard format for data exploration, cumulative reward and length information + interactivity metrics"""
    layouts = np.unique(trials['layout_name'])
    print("Layouts found", layouts)

    if clip_400:
        trials = trials[trials["cur_gameloop"] <= 400]

    # Add game length for each round
    trials = trials.join(trials.groupby(['trial_id'])['cur_gameloop'].count(), on=['trial_id'], rsuffix='_total')

    # Calculate total reward for each round
    trials = trials.join(trials.groupby(['trial_id'])['score'].max(), on=['trial_id'], rsuffix='_total')

    # Add interactivity metadata
    trials = _add_interactivity_metrics(trials)
    trials['button_presses_per_timstep'] = trials['button_press_total'] / trials['cur_gameloop_total']


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
pd.DataFrame(data={'a' : [0, 0, 1, 1, 1], 'b' : [1, 2, 3, 2, 1]})