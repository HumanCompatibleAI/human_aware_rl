import gym, time, os, seaborn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from memory_profiler import profile

from sacred import Experiment
from sacred.observers import FileStorageObserver
from tensorflow.saved_model import simple_save

PPO_DATA_DIR = 'data/ppo_runs/'

ex = Experiment('PPO')
ex.observers.append(FileStorageObserver.create(PPO_DATA_DIR + 'ppo_exp'))

from overcooked_ai_py.utils import load_pickle, save_pickle, load_dict_from_file, profile
from overcooked_ai_py.agents.agent import RandomAgent, GreedyHumanModel, AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS, MediumLevelPlanner
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from human_aware_rl.baselines_utils import get_vectorized_gym_env, create_model, update_model, save_baselines_model, load_baselines_model, get_agent_from_saved_model
from human_aware_rl.utils import create_dir_if_not_exists, reset_tf, delete_dir_if_exists, set_global_seed
from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved, DEFAULT_ENV_PARAMS, BC_SAVE_DIR
from human_aware_rl.experiments.bc_experiments import BEST_BC_MODELS_PATH


# PARAMS
@ex.config
def my_config():
    
    ##################
    # GENERAL PARAMS #
    ##################

    TIMESTAMP_DIR = True
    EX_NAME = "undefined_name"

    if TIMESTAMP_DIR:
        SAVE_DIR = PPO_DATA_DIR + time.strftime('%Y_%m_%d-%H_%M_%S_') + EX_NAME + "/"
    else:
        SAVE_DIR = PPO_DATA_DIR + EX_NAME + "/"

    print("Saving data to ", SAVE_DIR)

    RUN_TYPE = "ppo"

    # Reduce parameters to be able to run locally to test for simple bugs
    LOCAL_TESTING = False

    # Choice among: bc_train, bc_test, sp, hm, rnd
    OTHER_AGENT_TYPE = "bc_train"

    # Human model params, only relevant if OTHER_AGENT_TYPE is "hm"
    HM_PARAMS = [True, 0.3]

    # GPU id to use
    GPU_ID = 1

    # List of seeds to run
    SEEDS = [0]

    # Number of parallel environments used for simulating rollouts
    sim_threads = 30 if not LOCAL_TESTING else 2

    # Threshold for sparse reward before saving the best model
    SAVE_BEST_THRESH = 50

    # Every `VIZ_FREQUENCY` gradient steps, display the first 100 steps of a rollout of the agents
    VIZ_FREQUENCY = 50 if not LOCAL_TESTING else 10

    ##############
    # PPO PARAMS #
    ##############

    # Total environment timesteps for the PPO run
    PPO_RUN_TOT_TIMESTEPS = 5e6 if not LOCAL_TESTING else 10000

    # How many environment timesteps will be simulated (across all environments)
    # for one set of gradient updates. Is divided equally across environments
    TOTAL_BATCH_SIZE = 12000 if not LOCAL_TESTING else 800

    # Number of minibatches we divide up each batch into before
    # performing gradient steps
    MINIBATCHES = 6 if not LOCAL_TESTING else 1

    # Calculating `batch size` as defined in baselines
    BATCH_SIZE = TOTAL_BATCH_SIZE // sim_threads

    # Number of gradient steps to perform on each mini-batch
    STEPS_PER_UPDATE = 8 if not LOCAL_TESTING else 1

    # Learning rate
    LR = 1e-3

    # Factor by which to reduce learning rate over training
    LR_ANNEALING = 1 

    # Entropy bonus coefficient
    ENTROPY = 0.1

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

    # None is default value that does no schedule whatsoever
    # [x, y] defines the beginning of non-self-play trajectories
    SELF_PLAY_HORIZON = None

    # 0 is default value that does no annealing
    REW_SHAPING_HORIZON = 0 

    # Whether mixing of self play policies
    # happens on a trajectory or on a single-timestep level
    # Recommended to keep to true
    TRAJECTORY_SELF_PLAY = True


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
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    }
    
    # Env params
    horizon = 400

    # For non fixed MDPs
    mdp_generation_params = {
        "padded_mdp_shape": (11, 7),
        "mdp_shape_fn": ([5, 11], [5, 7]),
        "prop_empty_fn": [0.6, 1],
        "prop_feats_fn": [0, 0.6]
    }

    # Approximate info
    GRAD_UPDATES_PER_AGENT = STEPS_PER_UPDATE * MINIBATCHES * (PPO_RUN_TOT_TIMESTEPS // TOTAL_BATCH_SIZE)
    print("Grad updates per agent", GRAD_UPDATES_PER_AGENT)

    params = {
        "RUN_TYPE": RUN_TYPE,
        "SEEDS": SEEDS,
        "LOCAL_TESTING": LOCAL_TESTING,
        "EX_NAME": EX_NAME,
        "SAVE_DIR": SAVE_DIR,
        "GPU_ID": GPU_ID,
        "PPO_RUN_TOT_TIMESTEPS": PPO_RUN_TOT_TIMESTEPS,
        "mdp_params": {
            "layout_name": layout_name,
            "start_order_list": start_order_list,
            "rew_shaping_params": rew_shaping_params
        },
        "env_params": {
            "horizon": horizon
        },
        "mdp_generation_params": mdp_generation_params,
        "ENTROPY": ENTROPY,
        "GAMMA": GAMMA,
        "sim_threads": sim_threads,
        "TOTAL_BATCH_SIZE": TOTAL_BATCH_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,
        "LR": LR,
        "LR_ANNEALING": LR_ANNEALING,
        "VF_COEF": VF_COEF,
        "STEPS_PER_UPDATE": STEPS_PER_UPDATE,
        "MINIBATCHES": MINIBATCHES,
        "CLIPPING": CLIPPING,
        "LAM": LAM,
        "SELF_PLAY_HORIZON": SELF_PLAY_HORIZON,
        "REW_SHAPING_HORIZON": REW_SHAPING_HORIZON,
        "OTHER_AGENT_TYPE": OTHER_AGENT_TYPE,
        "HM_PARAMS": HM_PARAMS,
        "NUM_HIDDEN_LAYERS": NUM_HIDDEN_LAYERS,
        "SIZE_HIDDEN_LAYERS": SIZE_HIDDEN_LAYERS,
        "NUM_FILTERS": NUM_FILTERS,
        "NUM_CONV_LAYERS": NUM_CONV_LAYERS,
        "NETWORK_TYPE": NETWORK_TYPE,
        "SAVE_BEST_THRESH": SAVE_BEST_THRESH,
        "TRAJECTORY_SELF_PLAY": TRAJECTORY_SELF_PLAY,
        "VIZ_FREQUENCY": VIZ_FREQUENCY,
        "grad_updates_per_agent": GRAD_UPDATES_PER_AGENT
    }

def save_ppo_model(model, save_folder):
    delete_dir_if_exists(save_folder, verbose=True)
    simple_save(
        tf.get_default_session(),
        save_folder,
        inputs={"obs": model.act_model.X},
        outputs={
            "action": model.act_model.action, 
            "value": model.act_model.vf,
            "action_probs": model.act_model.action_probs
        }
    )

def configure_other_agent(params, gym_env, mlp, mdp):
    if params["OTHER_AGENT_TYPE"] == "hm":
        hl_br, hl_temp, ll_br, ll_temp = params["HM_PARAMS"]
        agent = GreedyHumanModel(mlp, hl_boltzmann_rational=hl_br, hl_temp=hl_temp, ll_boltzmann_rational=ll_br, ll_temp=ll_temp)
        gym_env.use_action_method = True

    elif params["OTHER_AGENT_TYPE"][:2] == "bc":
        best_bc_model_paths = load_pickle(BEST_BC_MODELS_PATH)
        if params["OTHER_AGENT_TYPE"] == "bc_train":
            bc_model_path = best_bc_model_paths["train"][mdp.layout_name]
        elif params["OTHER_AGENT_TYPE"] == "bc_test":
            bc_model_path = best_bc_model_paths["test"][mdp.layout_name]
        else:
            raise ValueError("Other agent type must be bc train or bc test")

        print("LOADING BC MODEL FROM: {}".format(bc_model_path))
        agent, bc_params = get_bc_agent_from_saved(bc_model_path)
        gym_env.use_action_method = True
        # Make sure environment params are the same in PPO as in the BC model
        for k, v in bc_params["env_params"].items():
            assert v == params["env_params"][k], "{} did not match. env_params: {} \t PPO params: {}".format(k, v, params[k])
        for k, v in bc_params["mdp_params"].items():
            assert v == params["mdp_params"][k], "{} did not match. mdp_params: {} \t PPO params: {}".format(k, v, params[k])

    elif params["OTHER_AGENT_TYPE"] == "rnd":
        agent = RandomAgent()

    elif params["OTHER_AGENT_TYPE"] == "sp":
        gym_env.self_play_randomization = 1

    else:
        raise ValueError("unknown type of agent to match with")
        
    if not params["OTHER_AGENT_TYPE"] == "sp":
        assert mlp.mdp == mdp
        agent.set_mdp(mdp)
        gym_env.other_agent = agent

def load_training_data(run_name, seeds=None):
    run_dir = PPO_DATA_DIR + run_name + "/"
    config = load_pickle(run_dir + "config")

    # To add backwards compatibility
    if seeds is None:
        if "NUM_SEEDS" in config.keys():
            seeds = list(range(min(config["NUM_SEEDS"], 5)))
        else:
            seeds = config["SEEDS"]

    train_infos = []
    for seed in seeds:
        train_info = load_pickle(run_dir + "seed{}/training_info".format(seed))
        train_infos.append(train_info)

    return train_infos, config

def get_ppo_agent(save_dir, seed=0, best=False):
    save_dir = PPO_DATA_DIR + save_dir + '/seed{}'.format(seed)
    config = load_pickle(save_dir + '/config')
    if best:
        agent = get_agent_from_saved_model(save_dir + "/best", config["sim_threads"])
    else:
        agent = get_agent_from_saved_model(save_dir + "/ppo_agent", config["sim_threads"])
    return agent, config

def match_ppo_with_other_agent(save_dir, other_agent, n=1, display=False):
    agent, agent_eval = get_ppo_agent(save_dir)
    ap0 = AgentPair(agent, other_agent)
    agent_eval.evaluate_agent_pair(ap0, display=display, num_games=n)

    # Sketch switch
    ap1 = AgentPair(other_agent, agent)
    agent_eval.evaluate_agent_pair(ap1, display=display, num_games=n)

def plot_ppo_run(name, sparse=False, limit=None, print_config=False, seeds=None, single=False):
    from collections import defaultdict
    train_infos, config = load_training_data(name, seeds)
    
    if print_config:
        print(config)
    
    if limit is None:
        limit = config["PPO_RUN_TOT_TIMESTEPS"]
    
    num_datapoints = len(train_infos[0]['eprewmean'])
    
    prop_data = limit / config["PPO_RUN_TOT_TIMESTEPS"]
    ciel_data_idx = int(num_datapoints * prop_data)

    datas = []
    for seed_num, info in enumerate(train_infos):
        info['xs'] = config["TOTAL_BATCH_SIZE"] * np.array(range(1, ciel_data_idx + 1))
        if single:
            plt.plot(info['xs'], info["ep_sparse_rew_mean"][:ciel_data_idx], alpha=1, label="Sparse{}".format(seed_num))
        datas.append(info["ep_sparse_rew_mean"][:ciel_data_idx])
    if not single:
        seaborn.tsplot(time=info['xs'], data=datas)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    if single:
        plt.legend()

@ex.automain
# @profile
def ppo_run(params):

    create_dir_if_not_exists(params["SAVE_DIR"])
    save_pickle(params, params["SAVE_DIR"] + "config")

    #############
    # PPO SETUP #
    #############

    train_infos = []

    for seed in params["SEEDS"]:
        reset_tf()
        set_global_seed(seed)

        curr_seed_dir = params["SAVE_DIR"] + "seed" + str(seed) + "/"
        create_dir_if_not_exists(curr_seed_dir)

        save_pickle(params, curr_seed_dir + "config")

        print("Creating env with params", params)
        # Configure mdp
        
        mdp = OvercookedGridworld.from_layout_name(**params["mdp_params"])
        env = OvercookedEnv(mdp, **params["env_params"])
        mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=True) 

        # Configure gym env
        gym_env = get_vectorized_gym_env(
            env, 'Overcooked-v0', featurize_fn=lambda x: mdp.lossless_state_encoding(x), **params
        )
        gym_env.self_play_randomization = 0 if params["SELF_PLAY_HORIZON"] is None else 1
        gym_env.trajectory_sp = params["TRAJECTORY_SELF_PLAY"]
        gym_env.update_reward_shaping_param(1 if params["mdp_params"]["rew_shaping_params"] != 0 else 0)

        configure_other_agent(params, gym_env, mlp, mdp)

        # Create model
        with tf.device('/device:GPU:{}'.format(params["GPU_ID"])):
            model = create_model(gym_env, "ppo_agent", **params)

        # Train model
        params["CURR_SEED"] = seed
        train_info = update_model(gym_env, model, **params)
        
        # Save model
        save_ppo_model(model, curr_seed_dir + model.agent_name)
        print("Saved training info at", curr_seed_dir + "training_info")
        save_pickle(train_info, curr_seed_dir + "training_info")
        train_infos.append(train_info)
    
    return train_infos
