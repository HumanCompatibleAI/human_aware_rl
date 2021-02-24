import json
import numpy as np

from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, ObjectState, PlayerState
from human_aware_rl.imitation.default_bc_params import DEFAULT_DATA_PARAMS, DEFAULT_BC_PARAMS
from human_aware_rl.rllib.utils import get_encoding_function

####################
# CONVERSION UTILS #
####################

def json_action_to_python_action(action):
    if type(action) is list:
        action = tuple(action)
    assert action in Action.ALL_ACTIONS
    return action


def json_joint_action_to_python_action(json_joint_action):
    """Port format from javascript to python version of Overcooked"""
    if type(json_joint_action) is str:
        json_joint_action = json.loads(json_joint_action)
    return tuple(json_action_to_python_action(a) for a in json_joint_action)


def json_state_to_python_state(df_state):
    """Convert from a df cell format of a state to an Overcooked State"""
    if type(df_state) is str:
        df_state = json.loads(df_state)

    return OvercookedState.from_dict(df_state)


def extract_df_for_worker_on_layout(main_trials, worker_id, layout_name):
    """Extract trajectory for a specific layout and worker pair from main_trials df"""
    worker_trajs_df = main_trials[main_trials['workerid_num'] == worker_id]
    worker_layout_traj_df = worker_trajs_df[worker_trajs_df['layout_name'] == layout_name]
    return worker_layout_traj_df


def df_traj_to_python_joint_traj(traj_df, complete_traj=True):
    if len(traj_df) == 0:
        return None

    datapoint = traj_df.iloc[0]
    layout_name = datapoint['layout_name']
    agent_evaluator = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout_name},
        env_params={"horizon": 1250}  # Defining the horizon of the mdp of origin of the trajectories
    )
    mdp = agent_evaluator.env.mdp
    env = agent_evaluator.env

    overcooked_states = [json_state_to_python_state(s) for s in traj_df.state]
    overcooked_actions = [json_joint_action_to_python_action(joint_action) for joint_action in traj_df.joint_action]
    overcooked_rewards = list(traj_df.reward)

    assert sum(
        overcooked_rewards) == datapoint.score_total, "Rewards didn't sum up to cumulative rewards. Probably trajectory df is corrupted / not complete"

    trajectories = {
        "ep_observations": [overcooked_states],
        "ep_actions": [overcooked_actions],
        "ep_rewards": [overcooked_rewards],  # Individual (dense) reward values
        "ep_dones": [[False] * len(overcooked_states)],  # Individual done values
        "ep_infos": [{}] * len(overcooked_states),

        "ep_returns": [sum(overcooked_rewards)],  # Sum of dense rewards across each episode
        "ep_lengths": [len(overcooked_states)],  # Lengths of each episode
        "mdp_params": [mdp.mdp_params],
        "env_params": [env.env_params],
        "metadatas": {
            'player_0_id': [datapoint['player_0_id']],
            'player_1_id': [datapoint['player_1_id']],
            'env': [agent_evaluator.env]
        }
    }
    trajectories = {k: np.array(v) if k not in ["ep_actions", "metadatas"] else v for k, v in trajectories.items()}

    if complete_traj:
        agent_evaluator.check_trajectories(trajectories)
    return trajectories


def convert_joint_df_trajs_to_overcooked_single(main_trials, layout_names, ordered_pairs=True, processed=False,
                                                silent=False, include_orders=DEFAULT_BC_PARAMS["predict_orders"],
                                                state_processing_function=DEFAULT_DATA_PARAMS["state_processing_function"], 
                                                action_processing_function=DEFAULT_DATA_PARAMS["action_processing_function"], 
                                                orders_processing_function=DEFAULT_DATA_PARAMS["orders_processing_function"]):
    """
    Takes in a dataframe `main_trials` containing joint trajectories, and extract trajectories of workers `worker_ids`
    on layouts `layout_names`, with specific options.
    """

    single_agent_trajectories = {
        # With shape (n_timesteps, game_len), where game_len might vary across games:
        "ep_observations": [],
        "ep_actions": [],
        "ep_rewards": [],  # Individual reward values
        "ep_dones": [],  # Individual done values
        "ep_infos": [],

        # With shape (n_episodes, ):
        "ep_returns": [],  # Sum of rewards across each episode
        "ep_lengths": [],  # Lengths of each episode
        "mdp_params": [],
        "env_params": [],
        "metadatas": {"ep_agent_idxs": []}  # Agent index for current episode
    }
    if include_orders:
        single_agent_trajectories["ep_orders"] = []
    human_indices = []
    num_trials_for_layout = {}
    for layout_name in layout_names:
        trial_ids = np.unique(main_trials[main_trials['layout_name'] == layout_name]['trial_id'])
        num_trials_for_layout[layout_name] = len(trial_ids)
        for trial_id in trial_ids:
            # Get an single game
            one_traj_df = main_trials[main_trials['trial_id'] == trial_id]

            # Get python trajectory data and information on which player(s) was/were human
            joint_traj_data = df_traj_to_python_joint_traj(one_traj_df, complete_traj=ordered_pairs)

            human_idx = get_human_player_index_for_df(one_traj_df)
            human_indices.append(human_idx)

            # Convert joint trajectories to single agent trajectories, appending recovered info to the `trajectories` dict
            joint_state_trajectory_to_single(single_agent_trajectories, joint_traj_data, human_idx, processed=processed,
                                             silent=silent, include_orders=include_orders,
                                             state_processing_function=state_processing_function,
                                             action_processing_function=action_processing_function,
                                             orders_processing_function=orders_processing_function)

    if not silent: print("Number of trajectories processed for each layout: {}".format(num_trials_for_layout))
    return single_agent_trajectories, human_indices


def get_human_player_index_for_df(one_traj_df):
    """Determines which player index had a human player"""
    human_player_indices = []
    assert len(one_traj_df['player_0_id'].unique()) == 1
    assert len(one_traj_df['player_1_id'].unique()) == 1
    datapoint = one_traj_df.iloc[0]
    if datapoint['player_0_is_human']:
        human_player_indices.append(0)
    if datapoint['player_1_is_human']:
        human_player_indices.append(1)

    return human_player_indices

def process_trajs_from_json_obj(trajectories, processed, agent_idxs,
                                include_orders=DEFAULT_BC_PARAMS["predict_orders"],
                                state_processing_function=DEFAULT_DATA_PARAMS["state_processing_function"], 
                                action_processing_function=DEFAULT_DATA_PARAMS["action_processing_function"], 
                                orders_processing_function=DEFAULT_DATA_PARAMS["orders_processing_function"]):
    if processed:
        state_processing_function = get_encoding_function(state_processing_function,
            mdp_params=trajectories["mdp_params"][0], env_params=trajectories["env_params"][0])
        action_processing_function = get_encoding_function(action_processing_function,
            mdp_params=trajectories["mdp_params"][0], env_params=trajectories["env_params"][0])
        orders_processing_function = get_encoding_function(orders_processing_function,
            mdp_params=trajectories["mdp_params"][0], env_params=trajectories["env_params"][0])
    else:
        # identity functions 
        state_processing_function = lambda state: [state]*(max(agent_idxs)+1)
        action_processing_function = lambda action: action
        orders_processing_function = lambda state: [state.orders_list]*(max(agent_idxs)+1)
    
    all_observations = []
    all_actions = []
    all_rewards = []
    all_orders = []
    for states, actions, rewards in zip(trajectories["ep_states"], trajectories["ep_actions"], trajectories["ep_rewards"]):
        for agent_idx in agent_idxs:
            single_agent_episode_observations = []
            single_agent_episode_actions = []
            single_agent_episode_rewards = []
            single_agent_episode_orders = []
            for state, action, reward in zip(states, actions, rewards):
                single_agent_episode_observations.append(state_processing_function(state)[agent_idx])
                single_agent_episode_actions.append(action_processing_function(action)[agent_idx])
                single_agent_episode_rewards.append([reward])
                if include_orders: single_agent_episode_orders.append(orders_processing_function(state)[agent_idx])
            all_observations.append(single_agent_episode_observations)
            all_actions.append(single_agent_episode_actions)
            all_rewards.append(single_agent_episode_rewards)
            if include_orders: all_orders.append(single_agent_episode_orders)

    if not trajectories.get("metadatas"):
        trajectories["metadatas"] = {}
    trajectories["metadatas"]["ep_agent_idxs"] = agent_idxs

    trajectories["ep_observations"] = all_observations
    trajectories["ep_actions"] = all_actions
    trajectories["ep_rewards"] = all_rewards
    if include_orders: trajectories["ep_orders"] = all_orders
    
    return trajectories, agent_idxs


def joint_state_trajectory_to_single(trajectories, joint_traj_data, player_indices_to_convert,
                                    processed=True, silent=False, include_orders=DEFAULT_BC_PARAMS["predict_orders"],
                                    state_processing_function=DEFAULT_DATA_PARAMS["state_processing_function"], 
                                    action_processing_function=DEFAULT_DATA_PARAMS["action_processing_function"], 
                                    orders_processing_function=DEFAULT_DATA_PARAMS["orders_processing_function"]
                                    ):
    """
    Take a joint trajectory and split it into two single-agent trajectories, adding data to the `trajectories` dictionary
    player_indices_to_convert: which player indexes' trajs we should return
    """
    env = joint_traj_data['metadatas']['env'][0]

    assert len(joint_traj_data['ep_observations']) == 1, "This method only takes in one trajectory"
    states, joint_actions = joint_traj_data['ep_observations'][0], joint_traj_data['ep_actions'][0]

    rewards, length = joint_traj_data['ep_rewards'][0], joint_traj_data['ep_lengths'][0]

    if processed:
        state_processing_function = get_encoding_function(state_processing_function, env=env)
        action_processing_function = get_encoding_function(action_processing_function, env=env)
        orders_processing_function = get_encoding_function(orders_processing_function, env=env)
    else:
        # identity functions 
        state_processing_function = lambda state: [state]*(max(player_indices_to_convert)+1)
        action_processing_function = lambda action: action
        orders_processing_function = lambda state: [state.orders_list]*(max(player_indices_to_convert)+1)
    
    # Getting trajectory for each agent
    for agent_idx in player_indices_to_convert:
        ep_obs, ep_acts, ep_dones, ep_orders = [], [], [], []
        for i in range(len(states)):
            state, joint_action = states[i], joint_actions[i]

            action = joint_action[agent_idx]
            ep_obs.append(state_processing_function(state)[agent_idx])
            ep_acts.append(action_processing_function(joint_action)[agent_idx])
            ep_dones.append(False)
            if include_orders: ep_orders.append(orders_processing_function(state)[agent_idx])
        
        ep_dones[-1] = True

        trajectories["ep_observations"].append(ep_obs)
        trajectories["ep_actions"].append(ep_acts)
        trajectories["ep_rewards"].append(rewards)
        trajectories["ep_dones"].append(ep_dones)
        trajectories["ep_infos"].append([{}] * len(rewards))
        trajectories["ep_returns"].append(sum(rewards))
        trajectories["ep_lengths"].append(length)
        trajectories["mdp_params"].append(env.mdp.mdp_params)
        trajectories["env_params"].append({})
        trajectories["metadatas"]["ep_agent_idxs"].append(agent_idx)
        if include_orders: trajectories["ep_orders"].append(ep_orders)