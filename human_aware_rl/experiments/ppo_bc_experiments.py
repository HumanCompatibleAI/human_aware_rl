import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import defaultdict

from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.utils import save_pickle

from human_aware_rl.utils import reset_tf, set_global_seed, prepare_nested_default_dict_for_pickle, common_keys_equal
from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved
from human_aware_rl.ppo.ppo import get_ppo_agent, plot_ppo_run, PPO_DATA_DIR


def plot_runs_training_curves(ppo_bc_model_paths, seeds, single=False, show=False, save=False):
    # Plot PPO BC models
    for run_type, type_dict in ppo_bc_model_paths.items():
        print(run_type)
        for layout, layout_model_path in type_dict.items():
            print(layout)
            plt.figure(figsize=(8,5))
            plot_ppo_run(layout_model_path, sparse=True, print_config=False, single=single, seeds=seeds[run_type])
            plt.xlabel("Environment timesteps")
            plt.ylabel("Mean episode reward")
            if save: plt.savefig("rew_ppo_bc_{}_{}".format(run_type, layout), bbox_inches='tight')
            if show: plt.show()

def evaluate_ppo_and_bc_models_for_layout(layout, num_rounds, bc_model_paths, ppo_bc_model_paths, seeds, best=False, display=False):
    assert len(seeds["bc_train"]) == len(seeds["bc_test"])
    ppo_bc_performance = defaultdict(lambda: defaultdict(list))

    agent_bc_test, bc_params = get_bc_agent_from_saved(bc_model_paths['test'][layout])
    ppo_bc_train_path = ppo_bc_model_paths['bc_train'][layout]
    ppo_bc_test_path = ppo_bc_model_paths['bc_test'][layout]
    evaluator = AgentEvaluator(mdp_params=bc_params["mdp_params"], env_params=bc_params["env_params"])
    
    for seed_idx in range(len(seeds["bc_train"])):
        agent_ppo_bc_train, ppo_config = get_ppo_agent(ppo_bc_train_path, seeds["bc_train"][seed_idx], best=best)
        assert common_keys_equal(bc_params["mdp_params"], ppo_config["mdp_params"])

        # For curiosity, how well does agent do with itself?
        # ppo_and_ppo = evaluator.evaluate_agent_pair(AgentPair(agent_ppo_bc_train, agent_ppo_bc_train), num_games=max(int(num_rounds/2), 1), display=display)
        # avg_ppo_and_ppo = np.mean(ppo_and_ppo['ep_returns'])
        # ppo_bc_performance[layout]["PPO_BC_train+PPO_BC_train"].append(avg_ppo_and_ppo)

        # How well it generalizes to new agent in simulation?
        ppo_and_bc = evaluator.evaluate_agent_pair(AgentPair(agent_ppo_bc_train, agent_bc_test), num_games=num_rounds, display=display)
        avg_ppo_and_bc = np.mean(ppo_and_bc['ep_returns'])
        ppo_bc_performance[layout]["PPO_BC_train+BC_test_0"].append(avg_ppo_and_bc)

        bc_and_ppo = evaluator.evaluate_agent_pair(AgentPair(agent_bc_test, agent_ppo_bc_train), num_games=num_rounds, display=display)
        avg_bc_and_ppo = np.mean(bc_and_ppo['ep_returns'])
        ppo_bc_performance[layout]["PPO_BC_train+BC_test_1"].append(avg_bc_and_ppo)
        
        # How well could we do if we knew true model BC_test?
        agent_ppo_bc_test, ppo_config = get_ppo_agent(ppo_bc_test_path, seeds["bc_test"][seed_idx], best=best)
        assert common_keys_equal(bc_params["mdp_params"], ppo_config["mdp_params"])
        
        ppo_and_bc = evaluator.evaluate_agent_pair(AgentPair(agent_ppo_bc_test, agent_bc_test), num_games=num_rounds, display=display)
        avg_ppo_and_bc = np.mean(ppo_and_bc['ep_returns'])
        ppo_bc_performance[layout]["PPO_BC_test+BC_test_0"].append(avg_ppo_and_bc)

        bc_and_ppo = evaluator.evaluate_agent_pair(AgentPair(agent_bc_test, agent_ppo_bc_test), num_games=num_rounds, display=display)
        avg_bc_and_ppo = np.mean(bc_and_ppo['ep_returns'])
        ppo_bc_performance[layout]["PPO_BC_test+BC_test_1"].append(avg_bc_and_ppo)
    
    return ppo_bc_performance

def evaluate_all_ppo_bc_models(ppo_bc_model_paths, best_bc_model_paths, num_rounds, seeds, best):
    layouts = list(ppo_bc_model_paths['bc_train'].keys())
    ppo_bc_performance = {}
    for layout in layouts:
        print(layout)
        layout_eval = evaluate_ppo_and_bc_models_for_layout(layout, num_rounds, best_bc_model_paths, ppo_bc_model_paths, seeds=seeds, best=best)
        ppo_bc_performance.update(dict(layout_eval))
    return ppo_bc_performance

def run_all_ppo_bc_experiments(best_bc_model_paths):
    reset_tf()

    seeds = {
        "bc_train": [9456, 1887, 5578, 5987,  516],
        "bc_test": [2888, 7424, 7360, 4467,  184]
    }

    ppo_bc_model_paths = {
        'bc_train': {
            "simple": "ppo_bc_train_simple",
            "unident_s": "ppo_bc_train_unident_s",
            "random1": "ppo_bc_train_random1",
            "random0": "ppo_bc_train_random0",
            "random3": "ppo_bc_train_random3"
        },
        'bc_test':{
            "simple": "ppo_bc_test_simple",
            "unident_s": "ppo_bc_test_unident_s",
            "random1": "ppo_bc_test_random1",
            "random0": "ppo_bc_test_random0",
            "random3": "ppo_bc_test_random3"
        }
    }

    plot_runs_training_curves(ppo_bc_model_paths, seeds, save=True)

    set_global_seed(248)
    num_rounds = 100
    ppo_bc_performance = evaluate_all_ppo_bc_models(ppo_bc_model_paths, best_bc_model_paths, num_rounds, seeds, best=True)
    ppo_bc_performance = prepare_nested_default_dict_for_pickle(ppo_bc_performance)
    save_pickle(ppo_bc_performance, PPO_DATA_DIR + "ppo_bc_models_performance")
