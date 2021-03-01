# All imports except rllib
import argparse, os, sys, copy
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import StayAgent, RandomAgent, GreedyHumanModel
from human_aware_rl.utils import load_dict_from_unknown_filetype
import numpy as np
# import whole agent module so any agent can be supplied in non_ml_agents_params_file
from overcooked_ai_py.agents import agent
# environment variable that tells us whether this code is running on the server or not
LOCAL_TESTING = os.getenv('RUN_ENV', 'production') == 'local'

# Sacred setup (must be before rllib imports)
from sacred import Experiment
ex_fp = Experiment("PPO RLLib From Params")

# Necessary work-around to make sacred pickling compatible with rllib
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Slack notification configuration
from sacred.observers import SlackObserver
if os.path.exists('slack.json') and not LOCAL_TESTING:
    slack_obs = SlackObserver.from_config('slack.json')
    ex_fp.observers.append(slack_obs)

    # Necessary for capturing stdout in multiprocessing setting
    SETTINGS.CAPTURE_MODE = 'sys'

# rllib and rllib-dependent imports
# Note: tensorflow and tensorflow dependent imports must also come after rllib imports
# This is because rllib disables eager execution. Otherwise, it must be manually disabled
import ray
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo.ppo import PPOTrainer
from human_aware_rl.ppo.ppo_rllib import RllibPPOModel, RllibLSTMPPOModel
from human_aware_rl.rllib.rllib import OvercookedMultiAgent, save_trainer, gen_trainer_from_params, DictObsSpacePPOTFPolicy
from human_aware_rl.imitation.behavior_cloning_tf2 import BehaviorCloningPolicy, BC_SAVE_DIR


###################### Temp Documentation #######################
#   run the following command in order to train a PPO self-play #
#   agent with the static parameters listed in my_config        #
#                                                               #
#   python ppo_rllib_client.py                                  #
#                                                               #
#   In order to view the results of training, run the following #
#   command                                                     #
#                                                               #
#   tensorboard --log-dir ~/ray_results/                        #
#                                                               #
#################################################################

# Dummy wrapper to pass rllib type checks
def _env_creator(env_config):
    # Re-import required here to work with serialization
    from human_aware_rl.rllib.rllib import OvercookedMultiAgent 
    return OvercookedMultiAgent.from_config(env_config)

# Variables that were previously hardcoded in multiple places
CUSTOM_MODEL_ID = "MyPPOModel"
INNER_SHAPE = (5, 4)
PROP_EMPTY = 0.95
PROP_FEATS =  0.1

def get_mdp_default_gen_params(rew_shaping=True):
    mdp_default_gen_params = {
        "inner_shape": INNER_SHAPE,
        "prop_empty": PROP_EMPTY,
        "prop_feats": PROP_FEATS,
        "start_all_orders": [
            {"ingredients": ["onion", "onion", "onion"]}
        ],
        "display": False
    }
    if rew_shaping:
        mdp_default_gen_params["rew_shaping_params"] = {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5,
            "DISH_DISP_DISTANCE_REW": 0,
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0,
            "TOMATO_COUNTER_PICKUP_REWARD": 0,
            "ONION_COUNTER_PICKUP_REWARD": 0,
            "TOMATO_DISPENSER_PICKUP_REWARD": 0,
            "ONION_DISPENSER_PICKUP_REWARD": 0
        }
    return mdp_default_gen_params
    
def naive_params_schedule_fn(outside_information):
    """
    In this preliminary version, the outside information is ignored
    """
    # Rewards the agent will receive for intermediate actions
    return get_mdp_default_gen_params()

@ex_fp.config
def my_config():
    ### PPO model params ###

    # Base model params
    NUM_HIDDEN_LAYERS = 3
    SIZE_HIDDEN_LAYERS = 64
    NUM_FILTERS = 25
    NUM_CONV_LAYERS = 3
    # whether to use recurrence in ppo model
    use_lstm = False
    D2RL = False
    # LSTM memory cell size (only used if use_lstm=True)
    CELL_SIZE = 256



    ### Other agents params ###
    # path to pickled policy model for behavior cloning
    bc_model_dir = os.path.join(BC_SAVE_DIR, "default")

    # Whether bc agents should return action logit argmax or sample
    bc_stochastic = True
    
    # config for non machine learning based agent (rule based)
    # agent_init_kwargs_variables is used by fill_init_kwargs inside OvercookedMultiAgent.create_non_ml_agent to fill selected fields with env/mdp attributes
    # if you want to change these easiest way to do it is to supply .txt file with whole variable that can be evaluated into this dict
    non_ml_agents_params = {
        "StayAgent": {
            "config": {
                "agent_cls": StayAgent
            }
        },
        "RandomAgentInteracting": {
             "config": {
                "agent_cls": RandomAgent,
                "agent_init_kwargs": {
                    "all_actions": True
                }
             }
        },
        "RandomAgent": {
             "config": {
                "agent_cls": RandomAgent,
                "agent_init_kwargs": {
                    "all_actions": False
                }
             }
        },
        "GreedyHumanModel": {
            "config": {
                "agent_cls": GreedyHumanModel,
                "agent_init_kwargs": {},
                "agent_init_kwargs_variables": {
                    "mlam": "env.mlam"
                }
            }
        }
    }
    non_ml_agents_params_file = None
    if non_ml_agents_params_file:
        with open(non_ml_agents_params_file, "r") as f:
            non_ml_agents_params = eval(f.read())



    ### Training Params ###

    num_workers = 20 if not LOCAL_TESTING else 2

    # list of all random seeds to use for experiments, used to reproduce results
    seeds = [0]

    # Placeholder for random for current trial
    seed = None

    # Number of gpus the central driver should use
    num_gpus = 0 if LOCAL_TESTING else 1

    # How many environment timesteps will be simulated (across all environments)
    # for one set of gradient updates. Is divided equally across environments
    # train_batch_size = 40000 if not LOCAL_TESTING else 800
    train_batch_size = 100000 if not LOCAL_TESTING else 800

    # size of minibatches we divide up each batch into before
    # performing gradient steps
    # sgd_minibatch_size = 10000 if not LOCAL_TESTING else 800
    sgd_minibatch_size = 25000 if not LOCAL_TESTING else 800

    # Rollout length
    rollout_fragment_length = 400

    # Stepsize of SGD.
    lr = 5e-3

    # Learning rate schedule.
    lr_schedule = None

    # If specified, clip the global norm of gradients by this amount
    grad_clip = 0.1

    # Discount factor
    gamma = 0.99

    # Exponential decay factor for GAE (how much weight to put on monte carlo samples)
    # Reference: https://arxiv.org/pdf/1506.02438.pdf
    lmbda = 0.98

    # Whether the value function shares layers with the policy model
    vf_share_layers = True

    # How much the loss of the value network is weighted in overall loss
    vf_loss_coeff = 1e-4

    # Entropy bonus coefficient, will anneal linearly from _start to _end over _horizon steps
    entropy_coeff_start = 0.02
    entropy_coeff_end = 0.00005
    entropy_coeff_horizon = 3e5

    # Initial coefficient for KL divergence.
    kl_coeff = 0.2

    # PPO clipping factor
    clip_param = 0.05

    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    num_sgd_iter = 8 if not LOCAL_TESTING else 1

    # Whether tensorflow should execute eagerly or not
    eager = False

    # Number of training iterations to run
    num_training_iters = 400 if not LOCAL_TESTING else 2
    
    # How many trainind iterations (calls to trainer.train()) to run before saving model checkpoint
    save_freq = 250

    # Whether all PPO agents should share the same policy network
    shared_policy = True



    ### Evaluation params ###
    
    # Agents used in evaluation
    evaluation_agents = ["ppo", "ppo"]

    # How many timesteps should be in an evaluation episode
    evaluation_ep_length = 400

    # Number of games to simulation each evaluation
    evaluation_num_games = 2

    # Whether to display rollouts in evaluation
    evaluation_display = True

    # Where to store replay txt files
    evaluation_replay_store_dir = None

    # How many training iterations to run between each evaluation
    evaluation_interval = 50 if not LOCAL_TESTING else 1

    ### Environment Params ###

    outer_shape = (5, 4)
    
    # The number of MDP in the env.mdp_lst
    num_mdp = 1
    # num_mdp = np.inf  # for infinite mdp

    # Max episode length
    horizon = 400

    # used when one of the agents uses mlam
    # NOTE: as layouts are generated there is no point in modifying counter_goals/drop/pickup
    mlam_params = {
        'start_orientations': False,
        'wait_allowed': False,
        'counter_goals': [],
        'counter_drop': [],
        'counter_pickup': [],
        'same_motion_goals': True
    }

    # Whether dense reward should come from potential function or not
    use_phi = True

    # Constant by which shaped rewards are multiplied by when calculating total reward
    reward_shaping_factor = 1.0

    # Linearly anneal the reward shaping factor such that it reaches zero after this number of timesteps
    reward_shaping_horizon = 1e6

    # bc_factor represents that ppo agent gets paired with a bc agent for any episode
    # schedule for bc_factor is represented by a list of points (t_i, v_i) where v_i represents the 
    # value of bc_factor at timestep t_i. Values are linearly interpolated between points
    # The default listed below represents bc_factor=0 for all timesteps
    bc_schedule = None

    # agents_schedule is list of dicts where key "agents" is list of dicts representing probability of having agent 
    #   of given type for every player at given timestep (with key "timestep") with linear interpolation in between the timesteps
    #   example dict: {"timestep": 10, "agents": [{"ppo":1}, {"ppo":0.3, "bc": 0.7}]}
    # you cannot change agents_schedule directly by running file from command line: python ppo_rllib_client.py with agents_schedule=...,
    #   use agents_schedule_file instead
    agents_schedule_file = None
    if bc_schedule is not None:
        agents_schedule = OvercookedMultiAgent.bc_schedule_to_agents_schedule(bc_schedule)
    elif agents_schedule_file:
        agents_schedule = load_dict_from_unknown_filetype(agents_schedule_file)
    else:
        agents_schedule = OvercookedMultiAgent.self_play_schedule
    agents_schedule_file = None

    # if agents starting positions should be shuffled in training and evaluation
    shuffle_agents = True

    # map type of ml based agent (ppo or bc) to state encoding functions and observation spaces
    # state encoding functions and observation spaces can be encoded in form of string if they are attributes/properties of mdp with 
    # format: "mdp.property_name" i.e.
    # {"ppo": "mdp.lossless_state_encoding_gym_space", 
    # "bc": "mdp.lossless_state_encoding_gym_space"}
    # use dicts for dict state spaces (only supported keys are: "observations" (encoded observation, used by default) 
    # and "auxillary_info" (1d vector containing info about anything else i.e. orders)) i.e.
    # {"ppo": {"observations": "mdp.lossless_state_encoding_gym_space", 
    #          "auxillary_info": "mdp.multi_hot_orders_encoding_gym_space"},
    #  "bc": "mdp.lossless_state_encoding_gym_space"}
    featurize_fns = copy.deepcopy(OvercookedMultiAgent.default_featurize_fns)
    featurize_fns_file = None
    if featurize_fns_file:
        featurize_fns = load_dict_from_unknown_filetype(featurize_fns_file)

    observation_spaces = copy.deepcopy(OvercookedMultiAgent.default_observation_spaces)
    observation_spaces_file = None
    if observation_spaces_file:
        observation_spaces = load_dict_from_unknown_filetype(observation_spaces_file)



    # Where to log the ray dashboard stats
    temp_dir = os.path.join(os.path.abspath(os.sep), "tmp", "ray_tmp") if not LOCAL_TESTING else None

    # Where to store model checkpoints and training stats
    results_dir = os.path.join(os.path.abspath('.'), 'results_client_temp')
    
    params_str = "nw=%d_vf=%f_es=%f_en=%f_kl=%f_outer_shape=%d_%d--inner_shape=%d_%d--prop_empty=%f--prop_feats=%f" % (
        num_workers,
        vf_loss_coeff,
        entropy_coeff_start,
        entropy_coeff_end,
        kl_coeff,
        outer_shape[0],
        outer_shape[1],
        INNER_SHAPE[0],
        INNER_SHAPE[1],
        PROP_EMPTY,
        PROP_FEATS
    )

    # Name of directory to store training results in (stored in ~/ray_results/<experiment_name>)
    experiment_name = "{0}_{1}".format("PPO_fp_", params_str)


    ppo_agent_params = {
            "policy_cls": DictObsSpacePPOTFPolicy,
            "config": {
                "model": {
                    # To be passed into rl-lib model/custom_options config
                    "custom_options": {
                        "NUM_HIDDEN_LAYERS": NUM_HIDDEN_LAYERS,
                        "SIZE_HIDDEN_LAYERS": SIZE_HIDDEN_LAYERS,
                        "NUM_FILTERS": NUM_FILTERS,
                        "NUM_CONV_LAYERS": NUM_CONV_LAYERS,      
                        "use_lstm": use_lstm,
                        "CELL_SIZE": CELL_SIZE,
                        "D2RL": D2RL
                    },
                    "custom_model": CUSTOM_MODEL_ID
                }
            }
    }

    bc_agent_params = {
        "policy_cls": BehaviorCloningPolicy,
        "config": {
            "model_dir": bc_model_dir,
            "stochastic": bc_stochastic,
            "eager": eager
        }
    }

    ml_agent_params = {
        "ppo": ppo_agent_params,
        "bc": bc_agent_params
    }

    # to be passed into the rllib.PPOTrainer class
    training_params = {
        "num_workers": num_workers,
        "seed": seed,
        "num_gpus": num_gpus,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "rollout_fragment_length": rollout_fragment_length,
        "lr": lr,
        "lr_schedule": lr_schedule,
        "grad_clip": grad_clip,
        "gamma": gamma,
        "lambda": lmbda,
        "vf_share_layers": vf_share_layers,
        "vf_loss_coeff": vf_loss_coeff,
        "entropy_coeff_schedule": [(0, entropy_coeff_start), (entropy_coeff_horizon, entropy_coeff_end)],
        "kl_coeff": kl_coeff,
        "clip_param": clip_param,
        "num_sgd_iter": num_sgd_iter,
        "evaluation_interval": evaluation_interval,
        "eager": eager
    }

    # To be passed into AgentEvaluator constructor and _evaluate function
    evaluation_params = {
        "agents": evaluation_agents,
        "ep_length": evaluation_ep_length,
        "num_games": evaluation_num_games,
        "display": evaluation_display,
        "store_dir": evaluation_replay_store_dir,
        "non_ml_agents_params": non_ml_agents_params,
        "display_phi": True
    }

    environment_params = {
        # To be passed into OvercookedGridWorld constructor
        "outer_shape": outer_shape,
        "mdp_params_schedule_fn": naive_params_schedule_fn,
        # To be passed into OvercookedEnv constructor
        "env_params": {
            "horizon": horizon,
            "num_mdp": num_mdp,
            "initial_info": {}
        },

        # evaluation mdp params
        "eval_mdp_params": get_mdp_default_gen_params(rew_shaping=False),

        #"eval_mdp_params" :{
        #    "layout_name": "cramped_room"
        #},
        # To be passed into OvercookedMultiAgent constructor
        "multi_agent_params": {
            "use_phi": use_phi,
            "reward_shaping_factor": reward_shaping_factor,
            "reward_shaping_horizon": reward_shaping_horizon,
            "agents_schedule": agents_schedule,
            "shuffle_agents": shuffle_agents,
            "featurize_fns": featurize_fns,
            "observation_spaces": observation_spaces,
            "non_ml_agents_params": non_ml_agents_params
        }
    }

    ray_params = {
        "custom_model_id": CUSTOM_MODEL_ID,
        "custom_model_cls": RllibLSTMPPOModel if use_lstm else RllibPPOModel,
        "temp_dir": temp_dir,
        "env_creator": _env_creator
    }

    params = {
        "agent_params": ml_agent_params,
        "training_params": training_params,
        "evaluation_params": evaluation_params,
        "environment_params": environment_params,
        "ray_params": ray_params,
        "shared_policy": shared_policy,
        "num_training_iters": num_training_iters,
        "experiment_name": experiment_name,
        "save_every": save_freq,
        "seeds": seeds,
        "results_dir": results_dir,
    }

# Dummy wrapper to pass rllib type checks
def _env_creater(env_config):
    return OvercookedMultiAgent.from_config(env_config)


def run(params):
    # Retrieve the tune.Trainable object that is used for the experiment
    trainer = gen_trainer_from_params(params)

    # Object to store training results in
    result = {}

    # Training loop
    for i in range(params['num_training_iters']):
        print("Starting training iteration", i)
        result = trainer.train()

        if i % params['save_every'] == 0:
            save_path = save_trainer(trainer, params)
            print("saved trainer at", save_path)

    # Save the state of the experiment at end
    save_path = save_trainer(trainer, params)
    print("saved trainer at", save_path)

    return result


@ex_fp.automain
def main(params):
    # All ray environment set-up
    ray.init(temp_dir=params["ray_params"]['temp_dir'])
    register_env("overcooked_multi_agent", _env_creater)
    ModelCatalog.register_custom_model(CUSTOM_MODEL_ID, params["ray_params"]["custom_model_cls"])

    # List of each random seed to run
    seeds = params['seeds']
    del params['seeds']

    # List to store results dicts (to be passed to sacred slack observer)
    results = []

    # Train an agent to completion for each random seed specified
    for seed in seeds:
        # Override the seed
        params['training_params']['seed'] = seed

        # Do the thing
        result = run(params)
        results.append(result)

    # Return value gets sent to our slack observer for notification
    average_sparse_reward = np.mean([res['custom_metrics']['sparse_reward_mean'] for res in results])
    average_episode_reward = np.mean([res['episode_reward_mean'] for res in results])
    return { "average_sparse_reward": average_sparse_reward, "average_total_reward": average_episode_reward }