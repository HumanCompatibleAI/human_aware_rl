import json
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, ObjectState, PlayerState, OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv


####################
# CONVERSION UTILS #
####################

# Currently Amazon Turk data has different names for some of the layouts
PYTHON_LAYOUT_NAME_TO_JS_NAME = {
    "unident_s": "asymmetric_advantages",
    "simple": "cramped_room",
    "random1": "coordination_ring",
    "random0": "random0",
    "random3": "random3"
}

JS_LAYOUT_NAME_TO_PYTHON_NAME = {v:k for k, v in PYTHON_LAYOUT_NAME_TO_JS_NAME.items()}

def json_action_to_python_action(action):
    if type(action) is list:
        action = tuple(action)
    elif type(str) and action == 'INTERACT':
        action = 'interact'
    assert action in Action.ALL_ACTIONS
    return action

def json_joint_action_to_python_action(json_joint_action):
    """Port format from javascript to python version of Overcooked"""
    if type(json_joint_action) is str:
        json_joint_action = eval(json_joint_action)
    return tuple(json_action_to_python_action(a) for a in json_joint_action)

def json_state_to_python_state(mdp, df_state):
    """Convert from a df cell format of a state to an Overcooked State"""
    if type(df_state) is str:
        df_state = eval(df_state)

    player_0 = df_state['players'][0]
    player_1 = df_state['players'][1]
    
    pos_0, or_0 = tuple(player_0['position']), tuple(player_0['orientation'])
    pos_1, or_1 = tuple(player_1['position']), tuple(player_1['orientation'])
    obj_0, obj_1 = None, None
    
    if 'held_object' in player_0.keys():
        obj_0 = json_obj_to_python_obj_state(mdp, player_0['held_object'])

    if 'held_object' in player_1.keys():
        obj_1 = json_obj_to_python_obj_state(mdp, player_1['held_object'])

    player_state_0 = PlayerState(pos_0, or_0, obj_0)
    player_state_1 = PlayerState(pos_1, or_1, obj_1)

    world_objects = {}
    for obj in df_state['objects'].values():
        object_state = json_obj_to_python_obj_state(mdp, obj)
        world_objects[object_state.position] = object_state

    assert not df_state["pot_explosion"]
    overcooked_state = OvercookedState(players=(player_state_0, player_state_1),
                                      objects=world_objects,
                                      order_list=None)
    return overcooked_state

def json_obj_to_python_obj_state(mdp, df_object):
    """Translates from a df cell format of a state to an Overcooked State"""
    obj_pos = tuple(df_object['position'])
    if 'state' in df_object.keys():
        soup_type, num_items, cook_time = tuple(df_object['state'])

        # Fix differing dynamics from Amazon Turk version
        if cook_time > mdp.soup_cooking_time:
            cook_time = mdp.soup_cooking_time

        obj_state = (soup_type, num_items, cook_time)
    else:
        obj_state = None
    return ObjectState(df_object['name'], obj_pos, obj_state)

def extract_df_for_worker_on_layout(main_trials, worker_id, layout_name):
    """Extract trajectory for a specific layout and worker pair from main_trials df"""
    worker_trajs_df = main_trials[main_trials['workerid_num'] == worker_id]
    layout_name = PYTHON_LAYOUT_NAME_TO_JS_NAME[layout_name]
    worker_layout_traj_df = worker_trajs_df[worker_trajs_df['layout_name'] == layout_name]
    return worker_layout_traj_df

def df_traj_to_python_joint_traj(traj_df, complete_traj=True):
    if len(traj_df) == 0:
        return None

    datapoint = traj_df.iloc[0]
    python_layout_name = JS_LAYOUT_NAME_TO_PYTHON_NAME[datapoint['layout_name']]
    agent_evaluator = AgentEvaluator(
        mdp_params={"layout_name": python_layout_name}, 
        env_params={"horizon": 1250}
    )
    mdp = agent_evaluator.env.mdp
    env = agent_evaluator.env

    overcooked_states = [json_state_to_python_state(mdp, s) for s in traj_df.state]
    overcooked_actions = [json_joint_action_to_python_action(joint_action) for joint_action in traj_df.joint_action]
    overcooked_rewards = list(traj_df.reward_norm)

    assert sum(overcooked_rewards) == datapoint.reward_norm_total, "Rewards didn't sum up to cumulative rewards. Probably trajectory df is corrupted / not complete"

    trajectories = {
        "ep_observations": [overcooked_states],
        "ep_actions": [overcooked_actions],
        "ep_rewards": [overcooked_rewards], # Individual (dense) reward values

        "ep_dones": [[False] * len(overcooked_states)], # Individual done values

        "ep_returns": [sum(overcooked_rewards)], # Sum of dense rewards across each episode
        "ep_returns_sparse": [sum(overcooked_rewards)], # Sum of sparse rewards across each episode
        "ep_lengths": [len(overcooked_states)], # Lengths of each episode
        "mdp_params": [mdp.mdp_params],
        "env_params": [env.env_params]
    }
    trajectories = {k: np.array(v) if k != "ep_actions" else v for k, v in trajectories.items() }

    if complete_traj:
        agent_evaluator.check_trajectories(trajectories)

    traj_metadata = {
        'worker_id': datapoint['workerid_num'],
        'round_num': datapoint['round_num'],
        'mdp': agent_evaluator.env.mdp
    }
    return trajectories, traj_metadata

def convert_joint_df_trajs_to_overcooked_single(main_trials, worker_ids, layout_names, ordered_pairs=True, human_ai_trajs=False):
    """
    Takes in a dataframe `main_trials` containing joint trajectories, and extract trajectories of workers `worker_ids` 
    on layouts `layout_names`, with specific options.
    """

    single_agent_trajectories = {
        # With shape (n_timesteps, game_len), where game_len might vary across games:
        "ep_observations": [],
        "ep_actions": [],
        "ep_rewards": [], # Individual reward values
        "ep_dones": [], # Individual done values,

        # With shape (n_episodes, ):
        "ep_returns": [], # Sum of rewards across each episode
        "ep_lengths": [], # Lengths of each episode
        "ep_agent_idxs": [], # Agent index for current episode
        "mdp_params": [],
        "env_params": []
    }

    for worker_id, layout_name in itertools.product(worker_ids, layout_names):
        # Get an single game
        one_traj_df = extract_df_for_worker_on_layout(main_trials, worker_id, layout_name)

        if len(one_traj_df) == 0:
            print("Layout {} is missing from worker {}".format(layout_name, worker_id))
            continue

        # Get python trajectory data and information on which player(s) was/were human
        joint_traj_data, traj_metadata = df_traj_to_python_joint_traj(one_traj_df, complete_traj=ordered_pairs)

        human_idx = [get_human_player_index_for_df(one_traj_df)] if human_ai_trajs else [0, 1]

        # Convert joint trajectories to single agent trajectories, appending recovered info to the `trajectories` dict
        joint_state_trajectory_to_single(single_agent_trajectories, joint_traj_data, traj_metadata, human_idx, processed=(not human_ai_trajs))

    return single_agent_trajectories

def get_human_player_index_for_df(one_traj_df):
    """Determines which player index had a human player"""
    assert len(one_traj_df['workerid_num'].unique()) == 1
    return (one_traj_df.groupby('workerid_num')['player_index'].sum() > 0).iloc[0]

def joint_state_trajectory_to_single(trajectories, joint_traj_data, traj_metadata, player_indices_to_convert=None, processed=True):
    """
    Take a joint trajectory and split it into two single-agent trajectories, adding data to the `trajectories` dictionary

    player_indices_to_convert: which player indexes' trajs we should return
    """
    from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS

    mdp = traj_metadata['mdp']
    mlp = MediumLevelPlanner.from_pickle_or_compute(
            mdp=mdp,
            mlp_params=NO_COUNTERS_PARAMS,
            force_compute=False
    )

    assert len(joint_traj_data['ep_observations']) == 1, "This method only takes in one trajectory"
    states, joint_actions = joint_traj_data['ep_observations'][0], joint_traj_data['ep_actions'][0]
    rewards, length = joint_traj_data['ep_rewards'][0], joint_traj_data['ep_lengths'][0]

    # Getting trajectory for each agent
    for agent_idx in player_indices_to_convert:

        ep_obs, ep_acts, ep_dones = [], [], []
        for i in range(len(states)):
            state, action = states[i], joint_actions[i][agent_idx]
            
            if processed:
                # Pre-processing (default is state featurization)
                action = np.array([Action.ACTION_TO_INDEX[action]]).astype(int)

                # NOTE: Could parallelize a bit more if slow
                # state = mdp.preprocess_observation(state)[agent_idx]
                state = mdp.featurize_state(state, mlp)[agent_idx]

            ep_obs.append(state)
            ep_acts.append(action)
            ep_dones.append(False)

        if len(ep_obs) == 0:
            worker_id, layout_name = traj_metadata['workerid_num'], mdp.layout_name
            print("{} on layout {} had an empty traj?. Excluding from dataset.".format(worker_id, layout_name))

        ep_dones[-1] = True

        trajectories["ep_observations"].append(ep_obs)
        trajectories["ep_actions"].append(ep_acts)
        trajectories["ep_dones"].append(ep_dones)
        trajectories["ep_rewards"].append(rewards)
        trajectories["ep_returns"].append(sum(rewards))
        trajectories["ep_lengths"].append(length)
        trajectories["ep_agent_idxs"].append(agent_idx)
        trajectories["mdp_params"].append(mdp.mdp_params)
        trajectories["env_params"].append({})
