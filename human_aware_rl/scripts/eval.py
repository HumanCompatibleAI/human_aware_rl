from human_aware_rl.imitation.behavior_cloning_tf2 import BehaviorCloningAgent
from human_aware_rl.rllib.rllib import get_base_ae, PPOAgent
from overcooked_ai_py.agents.benchmarking import *
from overcooked_ai_py.agents.agent import RandomAgent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
import json, argparse
import numpy as np


ALL_AGENTS = ['bc_test', 'ppo_bc', 'ppo_bc_opt', 'opt_fsp', 'rnd', 'opt_robust', 'opt_overfit', 'opt_overfit_1', 'bc_opt', 'bc_train']
ALL_LAYOUTS = ['soup_coordination', 'asymmetric_advantages_tomato', 'counter_circuit']

PATH_MAP = { layout : { agent_type : None for agent_type in ALL_AGENTS } for layout in ALL_LAYOUTS }

PATH_MAP['soup_coordination']['bc_train'] = '/Users/nathan/bair/overcooked/human_aware_rl/human_aware_rl/data/bc_runs/soup_coordination/soup_coord_train_balanced_75_epochs_off_dist_weighted_False'
PATH_MAP['soup_coordination']['bc_test'] = '/Users/nathan/bair/overcooked/human_aware_rl/human_aware_rl/data/bc_runs/soup_coordination/soup_coord_test_balanced_75_epochs_off_dist_weighted_False'
PATH_MAP['soup_coordination']['ppo_bc'] = '/Users/nathan/bair/overcooked/human_aware_rl/human_aware_rl/data/ppo_bc_runs/soup_coordination/weighted_bc_2/checkpoint-1667'
PATH_MAP['soup_coordination']['ppo_bc_opt'] = '/Users/nathan/bair/overcooked/human_aware_rl/human_aware_rl/data/ppo_bc_opt_runs/soup_coordination/bc_opt_unweighted_robost_1/checkpoint-1667'
PATH_MAP['soup_coordination']['opt_fsp'] = '/Users/nathan/bair/overcooked/human_aware_rl/human_aware_rl/data/ppo_fsp_runs/soup_coordination/fsp_50_N_1_K_prelim/checkpoint-1200'
PATH_MAP['soup_coordination']['opt_robust'] = '/Users/nathan/bair/overcooked/human_aware_rl/human_aware_rl/data/ppo_sp_runs/soup_coordination/forward_port_hotfix/checkpoint-1200'
PATH_MAP['soup_coordination']['opt_overfit'] = '/Users/nathan/bair/overcooked/human_aware_rl/human_aware_rl/data/ppo_sp_runs/soup_coordination/forward_port_upgraded_ray_960_return/checkpoint-1200'
PATH_MAP['soup_coordination']['opt_overfit_1'] = '/Users/nathan/bair/overcooked/human_aware_rl/human_aware_rl/data/ppo_sp_runs/soup_coordination/upgraded_ray_915_return/checkpoint-1200'
PATH_MAP['soup_coordination']['bc_opt'] = '/Users/nathan/bair/overcooked/human_aware_rl/human_aware_rl/data/ppo_bc_opt_runs/soup_coordination/bc_opt_unweighted_robost_1/checkpoint-1667'

PATH_MAP['asymmetric_advantages_tomato']['bc_train'] = '/Users/nathan/bair/overcooked/human_aware_rl/human_aware_rl/data/bc_runs/asymmetric_advantages_tomato/train_balanced_50_epochs_128_hidden_size'
PATH_MAP['asymmetric_advantages_tomato']['bc_test'] = '/Users/nathan/bair/overcooked/human_aware_rl/human_aware_rl/data/bc_runs/asymmetric_advantages_tomato/test_balanced_50_epochs_128_hidden_size'
PATH_MAP['asymmetric_advantages_tomato']['ppo_bc'] = '/Users/nathan/bair/overcooked/human_aware_rl/human_aware_rl/data/ppo_bc_runs/asymmetric_advantages_tomato/prelim_ppo_idx_0/checkpoint-1667'
PATH_MAP['asymmetric_advantages_tomato']['ppo_bc_opt'] = '/Users/nathan/bair/overcooked/human_aware_rl/human_aware_rl/data/ppo_bc_opt_runs/asymmetric_advantages_tomato/ppo_bc_opt_prelim/checkpoint-1667'
PATH_MAP['asymmetric_advantages_tomato']['bc_opt'] = '/Users/nathan/bair/overcooked/human_aware_rl/human_aware_rl/data/ppo_bc_opt_runs/asymmetric_advantages_tomato/ppo_bc_opt_prelim/checkpoint-1667'
PATH_MAP['asymmetric_advantages_tomato']['opt_fsp'] = '/Users/nathan/bair/overcooked/human_aware_rl/human_aware_rl/data/ppo_fsp_runs/asymmeric_advantages_tomato/lr_7e-4_batch_6e4_N_per_check_41/checkpoint-833'

OFF_DIST_STATE_PATHS = {
    'soup_coordination' : './off_dist_state.json',
    'asymmetric_advantages_tomato' : None
}

def eval(agent_0_type, agent_1_type, layout, num_games, off_dist_start):
    assert agent_0_type in ALL_AGENTS
    assert agent_1_type in ALL_AGENTS
    assert layout in ALL_LAYOUTS

    print("Evaluating {} + {} on {}".format(agent_0_type, agent_1_type, layout))
    pair = load_pair_by_type(agent_0_type, agent_1_type, layout)
    results = rollout(pair, layout, num_games, off_dist_start)
    analyze(results)

def load_agent_by_type(agent_type, layout):
    print("Loading {} agent".format(agent_type))
    if agent_type == 'rnd':
        return RandomAgent(all_actions=True)
    elif agent_type.startswith('opt'):
        return PPOAgent.from_trainer_path(PATH_MAP[layout][agent_type])
    elif agent_type == 'bc_train' or agent_type == 'bc_test':
        return BehaviorCloningAgent.from_model_dir(PATH_MAP[layout][agent_type], use_predict=False)
    elif agent_type == 'bc_opt' or agent_type == 'ppo_bc' or agent_type == 'ppo_bc_opt':
        bc_opt_trainer_params_to_override = {
            "model_dir" : PATH_MAP[layout]['bc_train'],
            "opt_path" : PATH_MAP[layout]['opt_fsp']
        }
        return PPOAgent.from_trainer_path(PATH_MAP[layout][agent_type], agent_type=agent_type, trainer_params_to_override=bc_opt_trainer_params_to_override)

def load_pair_by_type(type_0, type_1, layout):
    # BC must load second to avoid having graph overriden by rllib loading routine
    if type_0.startswith('bc'):
        agent_1 = load_agent_by_type(type_1, layout)
        agent_0 = load_agent_by_type(type_0, layout)
    else:
        agent_0 = load_agent_by_type(type_0, layout)
        agent_1 = load_agent_by_type(type_1, layout)
    pair = AgentPair(agent_0, agent_1)
    pair.reset()
    return pair

def rollout(pair, layout, num_games, off_dist_start):
    mdp_params = {
        "layout_name" : layout
    }
    env_params = {
        "horizon" : 400
    }

    eval_params = {
        "num_games" : num_games,
        "metadata_fn" : metadata_fn,
        "start_state_fn" : get_off_dist_start_state_fn(layout) if off_dist_start else None
    }
    ae = get_base_ae(mdp_params, env_params)
    results = ae.evaluate_agent_pair(pair,**eval_params)
    return results

def analyze(results):
    off_dist_rewards = results['metadatas']['off_dist_reward']
    off_dist_percentage = results['metadatas']['off_dist_percentage']
    sparse_rewards = results['ep_returns']

    off_dist_percentage_sorted = np.sort(off_dist_percentage)
    N = len(off_dist_percentage)
    percentiles = [0, .25, .5, .75, 1]
    for percentile in percentiles:
        idx = min(N-1, int(N*percentile))
        print("Off dist {}-percentile: {}".format(percentile, off_dist_percentage_sorted[idx]))
    print("Correlation coeff between sparse reward and off distribution percentage {0:.4f}".format(np.corrcoef(sparse_rewards, off_dist_percentage)[1,0]))

def metadata_fn(rollout):
    transitions = rollout[0]
    rewards, infos = transitions[:, 2], transitions[:, 4]
    off_dist = np.array([info['is_off_distribution'] for info in infos])
    off_dist_reward = np.sum(rewards * off_dist)
    off_dist_percentage = np.sum(off_dist) / len(off_dist)
    return { "off_dist_reward" : off_dist_reward, "off_dist_percentage" : off_dist_percentage}

def get_off_dist_start_state_fn(layout):
    def off_dist_start_state_fn():
        state_path = OFF_DIST_STATE_PATHS[layout]
        if not state_path:
            raise ValueError("Off distribution starts for {} not supported!".format(layout))
        with open(state_path, 'r') as f:
            state = OvercookedState.from_dict(json.load(f))
        return state
    return off_dist_start_state_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_0_type', '-a0', default='bc_test', type=str, choices=ALL_AGENTS)
    parser.add_argument('--agent_1_type', '-a1', default='bc_test', type=str, choices=ALL_AGENTS)
    parser.add_argument('--layout', '-l', default='soup_coordination', type=str, choices=ALL_LAYOUTS)
    parser.add_argument('--num_games', '-n', default=50, type=int)
    parser.add_argument('--off_dist_start', '-ood', action='store_true')
    args = vars(parser.parse_args())
    eval(**args)
