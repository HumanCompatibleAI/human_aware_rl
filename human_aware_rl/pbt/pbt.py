import os, gym, time, sys, random, itertools
import numpy as np
import tensorflow as tf
from collections import defaultdict
from memory_profiler import profile
from tensorflow.saved_model import simple_save

from sacred import Experiment
from sacred.observers import FileStorageObserver

PBT_DATA_DIR = "data/pbt_runs/"

ex = Experiment('PBT')
ex.observers.append(FileStorageObserver.create(PBT_DATA_DIR + "pbt_exps"))


from overcooked_ai_py.utils import profile, load_pickle, save_pickle, save_dict_to_file, load_dict_from_file
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import AgentPair

from human_aware_rl.utils import create_dir_if_not_exists, delete_dir_if_exists, reset_tf, set_global_seed
from human_aware_rl.baselines_utils import create_model, get_vectorized_gym_env, update_model, get_agent_from_model, save_baselines_model, overwrite_model, load_baselines_model, LinearAnnealer


class PBTAgent(object):
    """An agent that can be saved and loaded and all and the main data it contains is the self.model
    
    Goal is to be able to pass in save_locations or PBTAgents to workers that will load such agents
    and train them together.
    """
    
    def __init__(self, agent_name, start_params, start_logs=None, model=None, gym_env=None):
        self.params = start_params
        self.logs = start_logs if start_logs is not None else {
            "agent_name": agent_name,
            "avg_rew_per_step": [],
            "params_hist": defaultdict(list),
            "num_ppo_runs": 0,
            "reward_shaping": []
        }
        with tf.device('/device:GPU:{}'.format(self.params["GPU_ID"])):
            self.model = model if model is not None else create_model(gym_env, agent_name, **start_params)

    @property
    def num_ppo_runs(self):
        return self.logs["num_ppo_runs"]
    
    @property
    def agent_name(self):
        return self.logs["agent_name"]

    def get_agent(self):
        return get_agent_from_model(self.model, self.params["sim_threads"])

    def update(self, gym_env):
        with tf.device('/device:GPU:{}'.format(self.params["GPU_ID"])):
            train_info = update_model(gym_env, self.model, **self.params)

            for k, v in train_info.items():
                if k not in self.logs.keys():
                    self.logs[k] = []
                self.logs[k].extend(v)
            self.logs["num_ppo_runs"] += 1

    def update_avg_rew_per_step_logs(self, avg_rew_per_step_stats):
        self.logs["avg_rew_per_step"] = avg_rew_per_step_stats

    def save(self, save_folder):
        """Save agent model, logs, and parameters"""
        create_dir_if_not_exists(save_folder)
        save_baselines_model(self.model, save_folder)
        save_dict_to_file(dict(self.logs), save_folder + "logs")
        save_dict_to_file(self.params, save_folder + "params")

    @staticmethod
    def from_dir(load_folder):
        logs = load_dict_from_file(load_folder + "logs")
        agent_name = logs["agent_name"]
        params = load_dict_from_file(load_folder + "params")
        model = load_baselines_model(load_folder, agent_name)
        return PBTAgent(agent_name, params, start_logs=logs, model=model)

    @staticmethod
    def update_from_files(file0, file1, gym_env, save_dir):
        pbt_agent0 = PBTAgent.from_dir(file0)
        pbt_agent1 = PBTAgent.from_dir(file1)
        gym_env.other_agent = pbt_agent1
        pbt_agent0.update(gym_env)
        return pbt_agent0

    def save_predictor(self, save_folder):
        """Saves easy-to-load simple_save tensorflow predictor for agent"""
        simple_save(
            tf.get_default_session(),
            save_folder,
            inputs={"obs": self.model.act_model.X},
            outputs={
                "action": self.model.act_model.action, 
                "value": self.model.act_model.vf,
                "action_probs": self.model.act_model.action_probs
            }
        )

    def update_pbt_iter_logs(self):
        for k, v in self.params.items():
            self.logs["params_hist"][k].append(v)
        self.logs["params_hist"] = dict(self.logs["params_hist"])

    def explore_from(self, best_training_agent):
        overwrite_model(best_training_agent.model, self.model)
        self.logs["num_ppo_runs"] = best_training_agent.num_ppo_runs
        self.params = self.mutate_params(best_training_agent.params)

    def mutate_params(self, params_to_mutate):
        params_to_mutate = params_to_mutate.copy()
        for k in self.params["HYPERPARAMS_TO_MUTATE"]:
            if np.random.random() < params_to_mutate["RESAMPLE_PROB"]:
                mutation = np.random.choice(self.params["MUTATION_FACTORS"])
                
                if k == "LAM": 
                    # Move eps/2 in either direction
                    eps = min(
                        (1 - params_to_mutate[k]) / 2,      # If lam is > 0.5, avoid going over 1
                        params_to_mutate[k] / 2             # If lam is < 0.5, avoid going under 0
                    )
                    rnd_direction = (-1)**np.random.randint(2) 
                    mutation = rnd_direction * eps
                    params_to_mutate[k] = params_to_mutate[k] + mutation
                elif type(params_to_mutate[k]) is int:
                    params_to_mutate[k] = max(int(params_to_mutate[k] * mutation), 1)
                else:
                    params_to_mutate[k] = params_to_mutate[k] * mutation
                    
                print("Mutated {} by a factor of {}".format(k, mutation))

        print("Old params", self.params)
        print("New params", params_to_mutate)
        return params_to_mutate

@ex.config
def my_config():

    ##################
    # GENERAL PARAMS #
    ##################

    TIMESTAMP_DIR = True
    EX_NAME = "undefined_name"

    if TIMESTAMP_DIR:
        SAVE_DIR = PBT_DATA_DIR + time.strftime('%Y_%m_%d-%H_%M_%S_') + EX_NAME + "/"
    else:
        SAVE_DIR = PBT_DATA_DIR + EX_NAME + "/"

    print("Saving data to ", SAVE_DIR)

    RUN_TYPE = "pbt"

    # Reduce parameters to be able to run locally to test for simple bugs
    LOCAL_TESTING = False

    # GPU id to use
    GPU_ID = 1

    # List of seeds to run
    SEEDS = [0]

    # Number of parallel environments used for simulating rollouts
    sim_threads = 50 if not LOCAL_TESTING else 2

    ##############
    # PBT PARAMS #
    ##############

    TOTAL_STEPS_PER_AGENT = 1.5e7 if not LOCAL_TESTING else 1e4

    POPULATION_SIZE = 4

    ITER_PER_SELECTION = POPULATION_SIZE**2 # How many pairings and model training updates before the worst model is overwritten

    RESAMPLE_PROB = 0.33
    MUTATION_FACTORS = [0.75, 1.25]
    HYPERPARAMS_TO_MUTATE = ["LAM", "CLIPPING", "LR", "STEPS_PER_UPDATE", "ENTROPY", "VF_COEF"]

    NUM_SELECTION_GAMES = 10 if not LOCAL_TESTING else 2

    ##############
    # PPO PARAMS #
    ##############

    # Total environment timesteps for the PPO run
    PPO_RUN_TOT_TIMESTEPS = 40000 if not LOCAL_TESTING else 1000
    NUM_PBT_ITER = int(TOTAL_STEPS_PER_AGENT * POPULATION_SIZE // (ITER_PER_SELECTION * PPO_RUN_TOT_TIMESTEPS))

    # How many environment timesteps will be simulated (across all environments)
    # for one set of gradient updates. Is divided equally across environments
    TOTAL_BATCH_SIZE = 20000 if not LOCAL_TESTING else 1000

    # Number of minibatches we divide up each batch into before
    # performing gradient steps
    MINIBATCHES = 5 if not LOCAL_TESTING else 1

    BATCH_SIZE = TOTAL_BATCH_SIZE // sim_threads

    # Number of gradient steps to perform on each mini-batch
    STEPS_PER_UPDATE = 8 if not LOCAL_TESTING else 1

    # Learning rate
    LR = 5e-3

    # Entropy bonus coefficient
    ENTROPY = 0.5

    # Value function coefficient
    VF_COEF = 0.1

    # Gamma discounting factor
    GAMMA = 0.99

    # Lambda advantage discounting factor
    LAM = 0.98

    # Max gradient norm
    MAX_GRAD_NORM = 0.1

    # PPO clipping factor
    CLIPPING = 0.05

    # 0 is default value that does no annealing
    REW_SHAPING_HORIZON = 0

    ##################
    # NETWORK PARAMS #
    ##################

    # Network type used
    NETWORK_TYPE = "conv_and_mlp"

    # Network params
    NUM_HIDDEN_LAYERS = 3
    SIZE_HIDDEN_LAYERS = 64
    NUM_FILTERS = 25
    NUM_CONV_LAYERS = 3


    ##################
    # MDP/ENV PARAMS #
    ##################

    # Mdp params
    layout_name = None
    start_order_list = None

    rew_shaping_params = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0.015,
        "POT_DISTANCE_REW": 0.03,
        "SOUP_DISTANCE_REW": 0.1,
    }
    
    # Env params
    horizon = 400


    #########
    # OTHER #
    #########

    # For non fixed MDPs
    mdp_generation_params = {
        "padded_mdp_shape": (11, 7),
        "mdp_shape_fn": ([5, 11], [5, 7]),
        "prop_empty_fn": [0.6, 1],
        "prop_feats_fn": [0, 0.6]
    }

    # Approximate info stats
    GRAD_UPDATES_PER_AGENT = STEPS_PER_UPDATE * MINIBATCHES * (PPO_RUN_TOT_TIMESTEPS // TOTAL_BATCH_SIZE) * ITER_PER_SELECTION * NUM_PBT_ITER // POPULATION_SIZE

    print("Total steps per agent", TOTAL_STEPS_PER_AGENT)
    print("Grad updates per agent", GRAD_UPDATES_PER_AGENT)

    params = {
        "LOCAL_TESTING": LOCAL_TESTING,
        "RUN_TYPE": RUN_TYPE,
        "EX_NAME": EX_NAME,
        "SAVE_DIR": SAVE_DIR,
        "GPU_ID": GPU_ID,
        "mdp_params": {
            "layout_name": layout_name,
            "start_order_list": start_order_list,
            "rew_shaping_params": rew_shaping_params
        },
        "env_params": {
            "horizon": horizon
        },
        "PPO_RUN_TOT_TIMESTEPS": PPO_RUN_TOT_TIMESTEPS,
        "NUM_PBT_ITER": NUM_PBT_ITER,
        "ITER_PER_SELECTION": ITER_PER_SELECTION,
        "POPULATION_SIZE": POPULATION_SIZE,
        "RESAMPLE_PROB": RESAMPLE_PROB,
        "MUTATION_FACTORS": MUTATION_FACTORS,
        "mdp_generation_params": mdp_generation_params, # NOTE: currently not used
        "HYPERPARAMS_TO_MUTATE": HYPERPARAMS_TO_MUTATE,
        "REW_SHAPING_HORIZON": REW_SHAPING_HORIZON,
        "ENTROPY": ENTROPY,
        "GAMMA": GAMMA,
        "sim_threads": sim_threads,
        "TOTAL_BATCH_SIZE": TOTAL_BATCH_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,
        "LR": LR,
        "VF_COEF": VF_COEF,
        "STEPS_PER_UPDATE": STEPS_PER_UPDATE,
        "MINIBATCHES": MINIBATCHES,
        "CLIPPING": CLIPPING,
        "LAM": LAM,
        "NETWORK_TYPE": NETWORK_TYPE,
        "NUM_HIDDEN_LAYERS": NUM_HIDDEN_LAYERS,
        "SIZE_HIDDEN_LAYERS": SIZE_HIDDEN_LAYERS,
        "NUM_FILTERS": NUM_FILTERS,
        "NUM_CONV_LAYERS": NUM_CONV_LAYERS,
        "SEEDS": SEEDS,
        "NUM_SELECTION_GAMES": NUM_SELECTION_GAMES,
        "total_steps_per_agent": TOTAL_STEPS_PER_AGENT,
        "grad_updates_per_agent": GRAD_UPDATES_PER_AGENT
    }

@ex.named_config
def fixed_mdp():
    LOCAL_TESTING = False
    # fixed_mdp = True
    layout_name = "simple"

    sim_threads = 30 if not LOCAL_TESTING else 2
    PPO_RUN_TOT_TIMESTEPS = 36000 if not LOCAL_TESTING else 1000
    TOTAL_BATCH_SIZE = 12000 if not LOCAL_TESTING else 1000

    STEPS_PER_UPDATE = 5
    MINIBATCHES = 6 if not LOCAL_TESTING else 2

    LR = 5e-4

@ex.named_config
def fixed_mdp_rnd_init():
    # NOTE: Deprecated
    LOCAL_TESTING = False
    fixed_mdp = True
    layout_name = "scenario2"

    sim_threads = 10 if LOCAL_TESTING else 50
    PPO_RUN_TOT_TIMESTEPS = 24000
    TOTAL_BATCH_SIZE = 8000

    STEPS_PER_UPDATE = 4
    MINIBATCHES = 4

    # RND_OBJS = True
    # RND_POS = True

    LR = 5e-4

@ex.named_config
def padded_all_scenario():
    # NOTE: Deprecated
    LOCAL_TESTING = False
    fixed_mdp = ["scenario2", "simple", "schelling_s", "unident_s"]
    PADDED_MDP_SHAPE = (10, 5)

    sim_threads = 10 if LOCAL_TESTING else 60
    PPO_RUN_TOT_TIMESTEPS = 40000 if not LOCAL_TESTING else 1000
    TOTAL_BATCH_SIZE = 20000 if not LOCAL_TESTING else 1000

    STEPS_PER_UPDATE = 8
    MINIBATCHES = 4

    # RND_OBJS = False
    # RND_POS = True

    LR = 5e-4
    REW_SHAPING_HORIZON = 1e7

def pbt_one_run(params, seed):
    # Iterating noptepochs over same batch data but shuffled differently
    # dividing each batch in `nminibatches` and doing a gradient step for each one
    create_dir_if_not_exists(params["SAVE_DIR"])
    save_dict_to_file(params, params["SAVE_DIR"] + "config")

    #######
    # pbt #
    #######

    mdp = OvercookedGridworld.from_layout_name(**params["mdp_params"])
    overcooked_env = OvercookedEnv(mdp, **params["env_params"])

    print("Sample training environments:")
    for _ in range(5):
        overcooked_env.reset()
        print(overcooked_env)

    gym_env = get_vectorized_gym_env(
        overcooked_env, 'Overcooked-v0', featurize_fn=lambda x: mdp.lossless_state_encoding(x), **params
    )
    gym_env.update_reward_shaping_param(1.0) # Start reward shaping from 1

    annealer = LinearAnnealer(horizon=params["REW_SHAPING_HORIZON"])

    # ppo_expert_model = load_model("data/expert_agent/", "agent0", actual_agent_name="agent0")
    # pbt_expert_model = load_model("data/expert_agent/", "agent2", actual_agent_name="agent2")

    # AGENT POPULATION INITIALIZATION
    population_size = params["POPULATION_SIZE"]
    pbt_population = []
    pbt_agent_names = ['agent' + str(i) for i in range(population_size)]
    for agent_name in pbt_agent_names:
        agent = PBTAgent(agent_name, params, gym_env=gym_env)

        # overwrite_model(ppo_expert_model, model)
        
        pbt_population.append(agent)

    print("Initialized agent models")

    all_pairs = []
    for i in range(population_size):
        for j in range(i + 1, population_size):
            all_pairs.append((i, j))

    # MAIN LOOP

    def pbt_training():
        best_sparse_rew_avg = [-np.Inf] * population_size

        for pbt_iter in range(1, params["NUM_PBT_ITER"] + 1):
            print("\n\n\nPBT ITERATION NUM {}".format(pbt_iter))

            # TRAINING PHASE
            assert params["ITER_PER_SELECTION"] == population_size ** 2
            pairs_to_train = list(itertools.product(range(population_size), range(population_size)))

            for sel_iter in range(params["ITER_PER_SELECTION"]):
                # Randomly select agents to be trained
                pair_idx = np.random.choice(len(pairs_to_train))
                idx0, idx1 = pairs_to_train.pop(pair_idx)
                pbt_agent0, pbt_agent1 = pbt_population[idx0], pbt_population[idx1]

                # Training agent 1, leaving agent 0 fixed
                print("Training agent {} ({}) with agent {} ({}) fixed (pbt #{}/{}, sel #{}/{})".format(
                    idx1, pbt_agent1.num_ppo_runs, 
                    idx0, pbt_agent0.num_ppo_runs, 
                    pbt_iter, params["NUM_PBT_ITER"], sel_iter, params["ITER_PER_SELECTION"])
                )

                agent_env_steps = pbt_agent1.num_ppo_runs * params["PPO_RUN_TOT_TIMESTEPS"]
                reward_shaping_param = annealer.param_value(agent_env_steps)
                print("Current reward shaping:", reward_shaping_param, "\t Save_dir", params["SAVE_DIR"])
                pbt_agent1.logs["reward_shaping"].append(reward_shaping_param)
                gym_env.update_reward_shaping_param(reward_shaping_param)

                gym_env.other_agent = pbt_agent0.get_agent()
                pbt_agent1.update(gym_env)

                save_folder = params["SAVE_DIR"] + pbt_agent1.agent_name + '/'
                pbt_agent1.save(save_folder)

                agent_pair = AgentPair(pbt_agent0.get_agent(), pbt_agent1.get_agent())
                overcooked_env.get_rollouts(agent_pair, num_games=1, final_state=True, reward_shaping=reward_shaping_param)

            assert len(pairs_to_train) == 0

            # SELECTION PHASE
            # Overwrite worst agent with best model's agent, according to 
            # a proxy for generalization performance (avg dense reward across population)
            print("\nSELECTION PHASE\n")

            # Dictionary with average returns for each agent when matched with each other agent
            avg_ep_returns_dict = defaultdict(list)
            avg_ep_returns_sparse_dict = defaultdict(list)

            for i, pbt_agent in enumerate(pbt_population):
                # Saving each agent model at the end of the pbt iteration
                pbt_agent.update_pbt_iter_logs()

                if pbt_iter == params["NUM_PBT_ITER"]:
                    save_folder = params["SAVE_DIR"] + pbt_agent.agent_name + '/'
                    pbt_agent.save_predictor(save_folder + "pbt_iter{}/".format(pbt_iter))
                    pbt_agent.save(save_folder + "pbt_iter{}/".format(pbt_iter))

                for j in range(i, population_size):
                    # Pairs each agent with all other agents including itself in assessing generalization performance
                    print("Evaluating agent {} and {}".format(i, j))
                    pbt_agent_other = pbt_population[j]

                    agent_pair = AgentPair(pbt_agent.get_agent(), pbt_agent_other.get_agent())
                    trajs = overcooked_env.get_rollouts(agent_pair, params["NUM_SELECTION_GAMES"], reward_shaping=reward_shaping_param)
                    dense_rews, sparse_rews, lens = trajs["ep_returns"], trajs["ep_returns_sparse"], trajs["ep_lengths"]
                    rew_per_step = np.sum(dense_rews) / np.sum(lens)
                    avg_ep_returns_dict[i].append(rew_per_step)
                    avg_ep_returns_sparse_dict[i].append(sparse_rews)
                    if j != i:
                        avg_ep_returns_dict[j].append(rew_per_step)
                        avg_ep_returns_sparse_dict[j].append(sparse_rews)

            print("AVG ep rewards dict", avg_ep_returns_dict)

            for i, pbt_agent in enumerate(pbt_population):
                pbt_agent.update_avg_rew_per_step_logs(avg_ep_returns_dict[i])

                avg_sparse_rew = np.mean(avg_ep_returns_sparse_dict[i])
                if avg_sparse_rew > best_sparse_rew_avg[i]:
                    best_sparse_rew_avg[i] = avg_sparse_rew
                    agent_name = pbt_agent.agent_name
                    print("New best avg sparse rews {} for agent {}, saving...".format(best_sparse_rew_avg, agent_name))
                    best_save_folder = params["SAVE_DIR"] + agent_name + '/best/'
                    delete_dir_if_exists(best_save_folder, verbose=True)
                    pbt_agent.save_predictor(best_save_folder)
                    pbt_agent.save(best_save_folder)

            # Get best and worst agents when averageing rew per step across all agents
            best_agent_idx = max(avg_ep_returns_dict, key=lambda key: np.mean(avg_ep_returns_dict[key]))
            worst_agent_idx = min(avg_ep_returns_dict, key=lambda key: np.mean(avg_ep_returns_dict[key]))

            # MUTATION PHASE

            pbt_population[worst_agent_idx].explore_from(pbt_population[best_agent_idx])
            print("Overwrote worst model {} ({} rew) with best model {} ({} rew)"
                    .format(worst_agent_idx, avg_ep_returns_dict[worst_agent_idx],
                            best_agent_idx, avg_ep_returns_dict[best_agent_idx]))

            best_agent = pbt_population[best_agent_idx].get_agent()
            best_agent_copy = pbt_population[best_agent_idx].get_agent()
            agent_pair = AgentPair(best_agent, best_agent_copy)
            overcooked_env.get_rollouts(agent_pair, num_games=1, final_state=True, display=True, reward_shaping=reward_shaping_param)

    pbt_training()
    reset_tf()
    print(params["SAVE_DIR"])

@ex.automain
def run_pbt(params):
    create_dir_if_not_exists(params["SAVE_DIR"])
    save_dict_to_file(params, params["SAVE_DIR"] + "config")
    for seed in params["SEEDS"]:
        set_global_seed(seed)
        curr_seed_params = params.copy()
        curr_seed_params["SAVE_DIR"] += "seed_{}/".format(seed)
        pbt_one_run(curr_seed_params, seed)
