from human_aware_rl.imitation.behavior_cloning_tf2 import BehaviorCloningAgent
from human_aware_rl.rllib.rllib import load_agent, get_base_ae
from overcooked_ai_py.agents.benchmarking import *
from overcooked_ai_py.agents.agent import RandomAgent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
import json, argparse
import numpy as np


ALL_AGENTS = ['bc', 'ppo_bc', 'ppo_bc_opt', 'opt', 'rnd', 'opt_1', 'opt_2', 'bc_opt']
PPO_BC_OPT_PATH = '/Users/nathan/bair/human_aware_rl/human_aware_rl/data/ppo_bc_opt_runs/ppo_bc_opt_prelim/checkpoint-1667'
PPO_BC_PATH = '/Users/nathan/bair/human_aware_rl/human_aware_rl/data/ppo_bc_runs/ppo_bc_prelim/checkpoint-1667'
PPO_SP_PATH = '/Users/nathan/bair/human_aware_rl/human_aware_rl/data/ppo_sp_runs/soup_coord_hotfix/checkpoint-1200'
NEW_PPO_SP_PATH = '/Users/nathan/bair/human_aware_rl/human_aware_rl/data/ppo_sp_runs/upgraded_ray_960_return/checkpoint-1200'
OTHER_PPO_SP_PATH = '/Users/nathan/bair/human_aware_rl/human_aware_rl/data/ppo_sp_runs/upgraded_ray_915_return/checkpoint-1200'
BC_TEST_MODEL_PATH = '/Users/nathan/bair/human_aware_rl/human_aware_rl/data/bc_runs/soup_coord_test_balanced_100_epochs_off_dist_weighted_True'
BC_TRAIN_MODEL_PATH = '/Users/nathan/bair/human_aware_rl/human_aware_rl/data/bc_runs/soup_coord_train_balanced_100_epochs_off_dist_weighted_True'
OFF_DIST_STATE_PATH = './off_dist_state.json'

def eval(agent_1_type, agent_2_type, num_games, off_dist_start):
    assert agent_1_type in ALL_AGENTS
    assert agent_2_type in ALL_AGENTS

    print("Evaluating {} + {}".format(agent_1_type, agent_2_type))
    pair = load_pair_by_type(agent_1_type, agent_2_type)
    results = rollout(pair, num_games, off_dist_start)
    analyze(results)

def load_agent_by_type(agent_type):
    print("Loading {} agent".format(agent_type))
    if agent_type == 'opt':
        return load_agent(PPO_SP_PATH)
    elif agent_type == 'opt_1':
        return load_agent(NEW_PPO_SP_PATH)
    elif agent_type == 'opt_2':
        return load_agent(OTHER_PPO_SP_PATH)
    elif agent_type == 'bc':
        return BehaviorCloningAgent.from_model_dir(BC_TEST_MODEL_PATH)
    elif agent_type == 'bc_opt':
        bc_opt_trainer_params_to_override = {
            "model_dir" : BC_TRAIN_MODEL_PATH,
            "opt_path" : PPO_SP_PATH
        }
        return load_agent(PPO_BC_OPT_PATH, policy_id='bc_opt', trainer_params_to_override=bc_opt_trainer_params_to_override)
    elif agent_type == 'ppo_bc':
        bc_trainer_params_to_override = {
            "model_dir" : BC_TRAIN_MODEL_PATH
        }
        return load_agent(PPO_BC_PATH, policy_id='ppo', trainer_params_to_override=bc_trainer_params_to_override)
    elif agent_type == 'ppo_bc_opt':
        bc_opt_trainer_params_to_override = {
            "model_dir" : BC_TRAIN_MODEL_PATH,
            "opt_path" : PPO_SP_PATH
        }
        return load_agent(PPO_BC_OPT_PATH, policy_id='ppo', trainer_params_to_override=bc_opt_trainer_params_to_override)
    elif agent_type == 'rnd':
        return RandomAgent(all_actions=True)

def load_pair_by_type(type_1, type_2):
    # BC must load second to avoid having graph overriden by rllib loading routine
    if type_1 == 'bc':
        agent_2 = load_agent_by_type(type_2)
        agent_1 = load_agent_by_type(type_1)
    else:
        agent_1 = load_agent_by_type(type_1)
        agent_2 = load_agent_by_type(type_2)
    pair = AgentPair(agent_1, agent_2)
    pair.reset()
    return pair

def rollout(pair, num_games, off_dist_start):
    mdp_params = {
        "layout_name" : 'soup_coordination'
    }
    env_params = {
        "horizon" : 400
    }

    eval_params = {
        "num_games" : num_games,
        "metadata_fn" : metadata_fn,
        "start_state_fn" : off_dist_start_state_fn if off_dist_start else None
    }
    ae = get_base_ae(mdp_params, env_params)
    results = ae.evaluate_agent_pair(pair,**eval_params)
    return results

def analyze(results):
    off_dist_rewards = results['metadatas']['off_dist_reward']
    off_dist_percentage = results['metadatas']['off_dist_percentage']
    sparse_rewards = results['ep_returns']

    off_dist_percentage = np.sort(off_dist_percentage)
    N = len(off_dist_percentage)
    percentiles = [0, .25, .5, .75, 1]
    for percentile in percentiles:
        idx = min(N-1, int(N*percentile))
        print("Off dist {}-percentile: {}".format(percentile, off_dist_percentage[idx]))
    print("Correlation coeff between sparse reward and off distribution percentage", np.corrcoef(sparse_rewards, off_dist_percentage)[1,0])

def metadata_fn(rollout):
    transitions = rollout[0]
    rewards, infos = transitions[:, 2], transitions[:, 4]
    off_dist = np.array([info['is_off_distribution'] for info in infos])
    off_dist_reward = np.sum(rewards * off_dist)
    off_dist_percentage = np.sum(off_dist) / len(off_dist)
    return { "off_dist_reward" : off_dist_reward, "off_dist_percentage" : off_dist_percentage}

def off_dist_start_state_fn():
    with open(OFF_DIST_STATE_PATH, 'r') as f:
        state = OvercookedState.from_dict(json.load(f))
    return state

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_1_type', '-a1', default='bc', type=str, choices=ALL_AGENTS)
    parser.add_argument('--agent_2_type', '-a2', default='bc', type=str, choices=ALL_AGENTS)
    parser.add_argument('--num_games', '-n', default=50, type=int)
    parser.add_argument('--off_dist_start', '-ood', action='store_true')
    args = vars(parser.parse_args())
    eval(**args)
    os.path.join('hello', 'world)')
