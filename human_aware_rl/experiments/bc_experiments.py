import copy
import numpy as np

from overcooked_ai_py.utils import save_pickle, mean_and_std_err
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import AgentPair

from human_aware_rl.utils import reset_tf, set_global_seed, common_keys_equal
from human_aware_rl.imitation.behavioural_cloning import train_bc_agent, eval_with_benchmarking_from_saved, BC_SAVE_DIR, plot_bc_run, DEFAULT_BC_PARAMS, get_bc_agent_from_saved


# Path for dict containing the best bc models paths
BEST_BC_MODELS_PATH = BC_SAVE_DIR + "best_bc_model_paths"
BC_MODELS_EVALUATION_PATH = BC_SAVE_DIR + "bc_models_all_evaluations"

def train_bc_agent_from_hh_data(layout_name, agent_name, num_epochs, lr, adam_eps, model):
    """Trains a BC agent from human human data (model can either be `train` or `test`, which is trained
    on two different subsets of the data)."""

    bc_params = copy.deepcopy(DEFAULT_BC_PARAMS)
    bc_params["data_params"]['train_mdps'] = [layout_name]
    bc_params["data_params"]['data_path'] = "data/human/clean_{}_trials.pkl".format(model)
    bc_params["mdp_params"]['layout_name'] = layout_name
    bc_params["mdp_params"]['start_order_list'] = None

    model_save_dir = layout_name + "_" + agent_name + "/"
    return train_bc_agent(model_save_dir, bc_params, num_epochs=num_epochs, lr=lr, adam_eps=adam_eps)

def train_bc_models(all_params, seeds):
    """Train len(seeds) num of models for each layout"""
    for params in all_params:
        for seed_idx, seed in enumerate(seeds):
            set_global_seed(seed)
            model = train_bc_agent_from_hh_data(agent_name="bc_train_seed{}".format(seed_idx), model='train', **params)
            plot_bc_run(model.bc_info, params['num_epochs'])
            model = train_bc_agent_from_hh_data(agent_name="bc_test_seed{}".format(seed_idx), model='test', **params)
            plot_bc_run(model.bc_info, params['num_epochs'])
            reset_tf()

def evaluate_all_bc_models(all_params, num_rounds, num_seeds):
    """Evaluate all trained models"""
    bc_models_evaluation = {}
    for params in all_params:
        layout_name = params["layout_name"]
        
        print(layout_name)
        bc_models_evaluation[layout_name] = { "train": {}, "test": {} }

        for seed_idx in range(num_seeds):
            eval_trajs = eval_with_benchmarking_from_saved(num_rounds, layout_name + "_bc_train_seed{}".format(seed_idx))
            bc_models_evaluation[layout_name]["train"][seed_idx] = np.mean(eval_trajs['ep_returns'])
            
            eval_trajs = eval_with_benchmarking_from_saved(num_rounds, layout_name + "_bc_test_seed{}".format(seed_idx))
            bc_models_evaluation[layout_name]["test"][seed_idx] = np.mean(eval_trajs['ep_returns'])

    return bc_models_evaluation

def evaluate_bc_models(bc_model_paths, num_rounds):
    """
    Evaluate BC models passed in over `num_rounds` rounds
    """
    best_bc_models_performance = {}

    # Evaluate best
    for layout_name in bc_model_paths['train'].keys():
        print(layout_name)
        best_bc_models_performance[layout_name] = {}
        
        eval_trajs = eval_with_benchmarking_from_saved(num_rounds, bc_model_paths['train'][layout_name])
        best_bc_models_performance[layout_name]["BC_train+BC_train"] = mean_and_std_err(eval_trajs['ep_returns'])
        
        eval_trajs = eval_with_benchmarking_from_saved(num_rounds, bc_model_paths['test'][layout_name])
        best_bc_models_performance[layout_name]["BC_test+BC_test"] = mean_and_std_err(eval_trajs['ep_returns'])

        bc_train, bc_params_train = get_bc_agent_from_saved(bc_model_paths['train'][layout_name])
        bc_test, bc_params_test = get_bc_agent_from_saved(bc_model_paths['test'][layout_name])
        del bc_params_train["data_params"]
        del bc_params_test["data_params"]
        assert common_keys_equal(bc_params_train, bc_params_test)
        ae = AgentEvaluator(mdp_params=bc_params_train["mdp_params"], env_params=bc_params_train["env_params"])
        
        train_and_test = ae.evaluate_agent_pair(AgentPair(bc_train, bc_test), num_games=num_rounds)
        best_bc_models_performance[layout_name]["BC_train+BC_test_0"] = mean_and_std_err(train_and_test['ep_returns'])

        test_and_train = ae.evaluate_agent_pair(AgentPair(bc_test, bc_train), num_games=num_rounds)
        best_bc_models_performance[layout_name]["BC_train+BC_test_1"] = mean_and_std_err(test_and_train['ep_returns'])
    
    return best_bc_models_performance

def run_all_bc_experiments():
    # Train BC models
    seeds = [5415, 2652, 6440, 1965, 6647]
    num_seeds = len(seeds)

    params_unident = {"layout_name": "unident_s", "num_epochs": 120, "lr": 1e-3, "adam_eps":1e-8}
    params_simple = {"layout_name": "simple", "num_epochs": 100, "lr": 1e-3, "adam_eps":1e-8}
    params_random1 = {"layout_name": "random1", "num_epochs": 120, "lr": 1e-3, "adam_eps":1e-8}
    params_random0 = {"layout_name": "random0", "num_epochs": 90, "lr": 1e-3, "adam_eps":1e-8}
    params_random3 = {"layout_name": "random3", "num_epochs": 110, "lr": 1e-3, "adam_eps":1e-8}

    all_params = [params_simple, params_random1, params_unident, params_random0, params_random3]
    train_bc_models(all_params, seeds)

    # Evaluate BC models
    set_global_seed(64)

    num_rounds = 100
    bc_models_evaluation = evaluate_all_bc_models(all_params, num_rounds, num_seeds)
    save_pickle(bc_models_evaluation, BC_MODELS_EVALUATION_PATH)
    print("All BC models evaluation: ", bc_models_evaluation)

    # These models have been manually selected to more or less match in performance,
    # (test BC model should be a bit better than the train BC model)
    selected_models = {
        "simple": [0, 1],
        "unident_s": [0, 0],
        "random1": [4, 2],
        "random0": [2, 1],
        "random3": [3, 3]
    }

    final_bc_model_paths = { "train": {}, "test": {} }
    for layout_name, seed_indices in selected_models.items():
        train_idx, test_idx = seed_indices
        final_bc_model_paths["train"][layout_name] = "{}_bc_train_seed{}".format(layout_name, train_idx)
        final_bc_model_paths["test"][layout_name] = "{}_bc_test_seed{}".format(layout_name, test_idx)

    best_bc_models_performance = evaluate_bc_models(final_bc_model_paths, num_rounds)
    save_pickle(best_bc_models_performance, BC_SAVE_DIR + "best_bc_models_performance")
    

# Automatic selection of best BC models. Caused imbalances that made interpretation of results more difficult, 
# better to select manually non-best ones.

# def select_bc_models(bc_models_evaluation, num_rounds, num_seeds):
#     best_bc_model_paths = { "train": {}, "test": {} }

#     for layout_name, layout_eval_dict in bc_models_evaluation.items():
#         for model_type, seed_eval_dict in layout_eval_dict.items():
#             best_seed = np.argmax([seed_eval_dict[i] for i in range(num_seeds)])
#             best_bc_model_paths[model_type][layout_name] = "{}_bc_{}_seed{}".format(layout_name, model_type, best_seed)

#     save_pickle(best_bc_model_paths, BEST_BC_MODELS_PATH)
#     return best_bc_model_paths