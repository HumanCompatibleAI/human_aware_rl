import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from overcooked_ai_py.utils import save_pickle, load_pickle
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator

from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved
from human_aware_rl.utils import reset_tf, set_global_seed, prepare_nested_default_dict_for_pickle
from human_aware_rl.ppo.ppo import get_ppo_agent, plot_ppo_run, PPO_DATA_DIR


def plot_ppo_sp_training_curves(ppo_sp_model_paths, seeds, single=False, show=False, save=False):
    for layout, model_path in ppo_sp_model_paths.items():
        plt.figure(figsize=(8,5))
        plot_ppo_run(model_path, sparse=True, limit=None, print_config=False, single=single, seeds=seeds)
        plt.title(layout.split("_")[0])
        plt.xlabel("Environment timesteps")
        plt.ylabel("Mean episode reward")
        if save: plt.savefig("rew_ppo_sp_" + layout, bbox_inches='tight')
        if show: plt.show()

def evaluate_sp_ppo_and_bc(layout, ppo_sp_path, bc_test_path, num_rounds, seeds, best=False, display=False):
    sp_ppo_performance = defaultdict(lambda: defaultdict(list))

    agent_bc_test, bc_params = get_bc_agent_from_saved(bc_test_path)
    del bc_params["data_params"]
    del bc_params["mdp_fn_params"]
    evaluator = AgentEvaluator(**bc_params)

    for seed in seeds:
        agent_ppo, _ = get_ppo_agent(ppo_sp_path, seed, best=best)

        ppo_and_ppo = evaluator.evaluate_agent_pair(AgentPair(agent_ppo, agent_ppo, allow_duplicate_agents=True), num_games=num_rounds, display=display)
        avg_ppo_and_ppo = np.mean(ppo_and_ppo['ep_returns'])
        sp_ppo_performance[layout]["PPO_SP+PPO_SP"].append(avg_ppo_and_ppo)

        # Evaluate with BC test
        ppo_and_bc = evaluator.evaluate_agent_pair(AgentPair(agent_ppo, agent_bc_test), num_games=num_rounds, display=display)
        avg_ppo_and_bc = np.mean(ppo_and_bc['ep_returns'])
        sp_ppo_performance[layout]["PPO_SP+BC_test_0"].append(avg_ppo_and_bc)

        bc_and_ppo = evaluator.evaluate_agent_pair(AgentPair(agent_bc_test, agent_ppo), num_games=num_rounds, display=display)
        avg_bc_and_ppo = np.mean(bc_and_ppo['ep_returns'])
        sp_ppo_performance[layout]["PPO_SP+BC_test_1"].append(avg_bc_and_ppo)

    return sp_ppo_performance

def evaluate_all_sp_ppo_models(ppo_sp_model_paths, bc_test_model_paths, num_rounds, seeds, best):
    ppo_sp_performance = {}
    for layout in ppo_sp_model_paths.keys():
        print(layout)
        layout_eval = evaluate_sp_ppo_and_bc(layout, ppo_sp_model_paths[layout], bc_test_model_paths[layout], num_rounds, seeds, best)
        ppo_sp_performance.update(dict(layout_eval))
    return prepare_nested_default_dict_for_pickle(ppo_sp_performance)

def run_all_ppo_sp_experiments(best_bc_model_paths):
    reset_tf()

    seeds = [2229, 7649, 7225, 9807,  386]

    ppo_sp_model_paths = {
        "simple": "ppo_sp_simple",
        "unident_s": "ppo_sp_unident_s",
        "random1": "ppo_sp_random1",
        "random0": "ppo_sp_random0",
        "random3": "ppo_sp_random3"
    }

    plot_ppo_sp_training_curves(ppo_sp_model_paths, seeds, save=True)

    set_global_seed(124)
    num_rounds = 100
    ppo_sp_performance = evaluate_all_sp_ppo_models(ppo_sp_model_paths, best_bc_model_paths['test'], num_rounds, seeds, best=True)
    save_pickle(ppo_sp_performance, PPO_DATA_DIR + "ppo_sp_models_performance")

if __name__ == "__main__":
    best_bc_model_paths = load_pickle("data/bc_runs/best_bc_model_paths")
    run_all_ppo_sp_experiments(best_bc_model_paths)
