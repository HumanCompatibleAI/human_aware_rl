import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.utils import load_dict_from_txt

from human_aware_rl.human.data_processing_utils import PYTHON_LAYOUT_NAME_TO_JS_NAME
from human_aware_rl.human.process_dataframes import get_trajs_from_data
from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved
from human_aware_rl.ppo.ppo import get_ppo_agent
from human_aware_rl.pbt.pbt import PBT_DATA_DIR
from human_aware_rl.baselines_utils import get_pbt_agent_from_config
from human_aware_rl.utils import accuracy, reset_tf, cross_entropy


def get_trajs_losses_for_model(trajs, agent, eps=None):
    """
    Compute the cumulative cross entropy loss and % accuracy for predictions of 
    next-timestep actions of the agent that generated `trajs` (under assumptions that other agent is `agent`).
    """
    losses, accuracies = [], []
    for j in tqdm.trange(len(trajs['ep_observations'])):
        obs, acts, agent_idx = trajs['ep_observations'][j], trajs['ep_actions'][j], trajs['ep_agent_idxs'][j]
        agent.reset()
        agent.set_agent_index(agent_idx)
        probs = []
        for i in range(len(obs)):
            ob = obs[i]
            p = agent.action(ob)
            probs.append(p)
        probs = np.array(probs)
        actss = np.array([Action.ACTION_TO_INDEX[act] for act in acts])
        losses.append(cross_entropy(probs, actss, eps))
        accuracies.append(accuracy(probs, actss))
        agent.reset()
    return np.array(losses), np.array(accuracies)

def evaluate_layout_loss_for_bc_models(best_bc_model_paths, layout_name, trajs, eps):
    # TODO Check this isn't stochastic
    layout_losses = defaultdict(dict)
    model_name = best_bc_model_paths["train"][layout_name]
    bc_train, _ = get_bc_agent_from_saved(model_name=model_name)
    
    model_name = best_bc_model_paths["test"][layout_name]
    bc_test, _ = get_bc_agent_from_saved(model_name=model_name)
    
    bc_agents = {"train": bc_train, "test": bc_test}
    for agent_type, bc_agent in bc_agents.items():
        bc_agent.action_probs = True
        bc_agent.stochastic = False
        bc_agent.will_unblock_if_stuck = False
        
        losses, accuracies = get_trajs_losses_for_model(trajs, bc_agent, eps)
        layout_losses[agent_type]['losses'] = losses
        layout_losses[agent_type]['accuracies'] = accuracies
    return layout_losses

def evaluate_layout_loss_for_ppo_models(ppo_path, layout_name, trajs, eps, seeds):
    layout_losses = defaultdict(dict)
    for seed in seeds:
        reset_tf()
        agent_ppo, bc_params = get_ppo_agent(ppo_path, seed, best=True)
        agent_ppo.action_probs = True
        agent_ppo.set_mdp(OvercookedGridworld.from_layout_name(**bc_params["mdp_params"]))
        
        losses, accuracies = get_trajs_losses_for_model(trajs, agent_ppo, eps)
        layout_losses["{}_seed{}".format(layout_name, seed)]['losses'] = losses
        layout_losses["{}_seed{}".format(layout_name, seed)]['accuracies'] = accuracies
    return layout_losses

def evaluate_layout_loss_for_pbt_models(pbt_model_paths, layout_name, trajs, eps, seeds, best=True):
    layout_losses = defaultdict(dict)
    
    pbt_save_dir = PBT_DATA_DIR + pbt_model_paths[layout_name] + "/"
    pbt_config = load_dict_from_txt(pbt_save_dir + "config")
    
    for seed in seeds:
        reset_tf()
        agent_pbt = get_pbt_agent_from_config(pbt_save_dir, pbt_config["sim_threads"], seed=seed, agent_idx=0, best=best)
        agent_pbt.action_probs = True
        agent_pbt.set_mdp(OvercookedGridworld.from_layout_name(**pbt_config["mdp_params"]))
        
        losses, accuracies = get_trajs_losses_for_model(trajs, agent_pbt, eps)
        layout_losses["{}_seed{}".format(layout_name, seed)]['losses'] = losses
        layout_losses["{}_seed{}".format(layout_name, seed)]['accuracies'] = accuracies
    return layout_losses