# All imports except rllib
import os
import numpy as np

# environment variable that tells us whether this code is running on the server or not
LOCAL_TESTING = os.getenv('RUN_ENV', 'production') == 'local'

# Sacred setup (must be before rllib imports)
from sacred import Experiment
ex = Experiment("PPO RLLib")

# Necessary work-around to make sacred pickling compatible with rllib
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Slack notification configuration
from sacred.observers import SlackObserver
if os.path.exists('slack.json') and not LOCAL_TESTING:
    slack_obs = SlackObserver.from_config('slack.json')
    ex.observers.append(slack_obs)

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
from human_aware_rl.ppo.ppo_rllib import RllibPPOModel
from human_aware_rl.rllib.rllib import OvercookedMultiAgent, save_trainer, gen_trainer_from_params
from human_aware_rl.rllib.meta_policies import EnsemblePolicy
from human_aware_rl.imitation.behavior_cloning_tf2 import BehaviorCloningPolicy, BC_SAVE_DIR, BernoulliBCSelfPlayOPTPolicy, OffDistCounterBCOPT


###################### Temp Documentation #######################
#   run the following command in order to train a PPO self-play #
#   agent with the static parameters listed in my_config        #
#                                                               #
#   python ppo_rllib_client.py                                  #
#                                                               #
#   In order to view the results of training, run the following #
#   command                                                     #
#                                                               #x
#   tensorboard --log-dir ~/ray_results/                        #
#                                                               #
#################################################################

# Dummy wrapper to pass rllib type checks
def _env_creator(env_config):
    # Re-import required here to work with serialization
    from human_aware_rl.rllib.rllib import OvercookedMultiAgent 
    return OvercookedMultiAgent.from_config(env_config)

BC_OPT_CLS_MAP = {
    'bernoulli' : BernoulliBCSelfPlayOPTPolicy,
    'counters' : OffDistCounterBCOPT
}

@ex.config
def my_config():
    ### Model params ###

    # whether to use recurrence in ppo model
    use_lstm = False

    # Base model params
    NUM_HIDDEN_LAYERS = 3
    SIZE_HIDDEN_LAYERS = 64
    NUM_FILTERS = 25
    NUM_CONV_LAYERS = 3

    # LSTM memory cell size (only used if use_lstm=True)
    CELL_SIZE = 256

    # whether to use D2RL https://arxiv.org/pdf/2010.09163.pdf (concatenation the result of last conv layer to each hidden layer); works only when use_lstm is False
    D2RL = False
    ### Training Params ###

    num_workers = 30 if not LOCAL_TESTING else 2

    # list of all random seeds to use for experiments, used to reproduce results
    seeds = [0]

    # Placeholder for random for current trial
    seed = None

    # Number of gpus the central driver should use
    num_gpus = 0 if LOCAL_TESTING else 1

    # How many environment timesteps will be simulated (across all environments)
    # for one set of gradient updates. Is divided equally across environments
    train_batch_size = 12000 if not LOCAL_TESTING else 800

    # size of minibatches we divide up each batch into before
    # performing gradient steps
    sgd_minibatch_size = 2000 if not LOCAL_TESTING else 800

    # Rollout length
    rollout_fragment_length = 400
    
    # Whether all PPO agents should share the same policy network
    shared_policy = True

    # Number of training iterations to run
    num_training_iters = 420 if not LOCAL_TESTING else 2

    # Stepsize of SGD.
    lr = 5e-5

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
    entropy_coeff_start = 0.2
    entropy_coeff_end = 0.1
    entropy_coeff_horizon = 3e5

    # Fully specified entropy coefficient schedule (w/ linear imputation between points)
    # Overrides other entropy coeff params if present
    entropy_coeff_schedule = None

    # Initial coefficient for KL divergence.
    kl_coeff = 0.2

    # PPO clipping factor
    clip_param = 0.05

    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    num_sgd_iter = 8 if not LOCAL_TESTING else 1

    # How many trainind iterations (calls to trainer.train()) to run before saving model checkpoint
    save_freq = 25

    # How many training iterations to run between each evaluation
    evaluation_interval = 50 if not LOCAL_TESTING else 1

    # How many timesteps should be in an evaluation episode
    evaluation_ep_length = 400

    # Number of games to simulation each evaluation
    evaluation_num_games = 1

    # Whether we should perform an evaluation rollout with a random agent
    evaluation_rnd_eval = False

    # Whether to display rollouts in evaluation
    evaluation_display = False

    # Where to log the ray dashboard stats
    temp_dir = os.path.join(os.path.abspath(os.sep), "tmp", "ray_tmp")

    # Where to store model checkpoints and training stats
    results_dir = DEFAULT_RESULTS_DIR

    # Whether tensorflow should execute eagerly or not
    eager = False

    # Whether to log training progress and debugging info
    verbose = True

    # Whether to log by-agent timestep level events to tensorboard
    log_timestep_events = False


    ### Self-Play Params ###

    # Whether (if True) PPO agent should train against ensemble of past copies of itself
    ficticious_self_play = False

    # Number of environment timesteps after which we save a PPO checkpoint for the ensemble
    # Note: only applicable if ficticious_self_play=True
    training_iters_per_ensemble_checkpoint = 25

    # Number of training iterations after which we resample from our self-play buffer
    training_iters_per_ensemble_sample = 5

    # Maximum number of policy checkpoints we keep loaded in memory
    max_policies_in_memory = 5


    ### BC Params ###
    # path to pickled policy model for behavior cloning
    bc_model_dir = os.path.join(BC_SAVE_DIR, "default")

    # Whether bc agents should return action logit argmax or sample
    bc_stochastic = True

    # Whether bc agent should bc optimal off-distribution
    bc_opt = False

    # Rllib.Policy subclass to wrap BC_OPT policy in
    bc_opt_cls_key = 'counters'

    # Path to serialized pre-trained OPT agent
    opt_path = os.path.join(os.path.abspath("~"), 'ray_results', 'my_experiment')



    ### Environment Params ###
    # Which overcooked level to use
    layout_name = "cramped_room"

    # all_layout_names = '_'.join(layout_names)

    # Rewards the agent will receive for intermediate actions
    rew_shaping_params = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    }

    potential_constants = {
        'max_delivery_steps' : 10,
        'max_pickup_steps' : 20,
        'pot_onion_steps' : 10,
        'pot_tomato_steps' : 10
    }

    # Whether dense reward should include potential function or not
    use_potential_shaping = True

    # Whether the dense reward should include vanilla reward shaping or not
    use_reward_shaping = False

    # Max episode length
    horizon = 400

    # Constant by vanilla reward shaping is multiplied by before adding to total reward
    reward_shaping_factor = 1.0

    # Linearly anneal the reward shaping factor such that it reaches zero after this number of timesteps
    reward_shaping_horizon = 3e6

    # Fully specified reward shaping schedule; overrides other shaping coeff params if present
    reward_shaping_schedule = None

    # Potential shaping coefficient
    potential_shaping_factor = 1.0

    # Linearly anneal the potential shaping factor such that it reaches zero after this number of timesteps
    potential_shaping_horizon = float('inf')

    # Fully specified potential shaping schedule; overrides other potential shaping coeff params if present
    potential_shaping_schedule = None

    # bc_factor represents that ppo agent gets paired with a bc agent for any episode
    # schedule for bc_factor is represented by a list of points (t_i, v_i) where v_i represents the 
    # value of bc_factor at timestep t_i. Values are linearly interpolated between points
    # The default listed below represents bc_factor=0 for all timesteps
    bc_schedule = OvercookedMultiAgent.self_play_bc_schedule

    # What index (0 or 1) the PPO agent should occupy for all episodes. -1 implies uniform sampling over possible indices
    ppo_idx = -1

    # Name of directory to store training results in (stored in ~/ray_results/<experiment_name>)

    params_str = str(use_potential_shaping) + "_nw=%d_vf=%f_es=%f_en=%f_kl=%f" % (
        num_workers,
        vf_loss_coeff,
        entropy_coeff_start,
        entropy_coeff_end,
        kl_coeff
    )

    experiment_name = "{0}_{1}_{2}".format("PPO", layout_name, params_str)


    # To be passed into rl-lib model/custom_options config
    model_params = {
        "use_lstm" : use_lstm,
        "vf_share_layers" : vf_share_layers,
        "NUM_HIDDEN_LAYERS" : NUM_HIDDEN_LAYERS,
        "SIZE_HIDDEN_LAYERS" : SIZE_HIDDEN_LAYERS,
        "NUM_FILTERS" : NUM_FILTERS,
        "NUM_CONV_LAYERS" : NUM_CONV_LAYERS,
        "CELL_SIZE" : CELL_SIZE,
        "D2RL": D2RL
    }

    # to be passed into the rllib.PPOTrainer class
    training_params = {
        "num_workers" : num_workers,
        "train_batch_size" : train_batch_size,
        "sgd_minibatch_size" : sgd_minibatch_size,
        "rollout_fragment_length" : rollout_fragment_length,
        "num_sgd_iter" : num_sgd_iter,
        "lr" : lr,
        "lr_schedule" : lr_schedule,
        "grad_clip" : grad_clip,
        "gamma" : gamma,
        "lambda" : lmbda,
        "vf_loss_coeff" : vf_loss_coeff,
        "kl_coeff" : kl_coeff,
        "clip_param" : clip_param,
        "num_gpus" : num_gpus,
        "seed" : seed,
        "evaluation_interval" : evaluation_interval,
        "entropy_coeff_schedule" : entropy_coeff_schedule if entropy_coeff_schedule else [(0, entropy_coeff_start), (entropy_coeff_horizon, entropy_coeff_end)],
        "eager_tracing" : eager,
        "log_level" : "WARN" if verbose else "ERROR",
    }

    # To be passed into AgentEvaluator constructor and _evaluate function
    evaluation_params = {
        "rnd_eval" : evaluation_rnd_eval,
        "ep_length" : evaluation_ep_length,
        "num_games" : evaluation_num_games,
        "display" : evaluation_display
    }


    environment_params = {
        # To be passed into OvercookedGridWorld constructor

        "mdp_params" : {
            "layout_name": layout_name,
            "rew_shaping_params": rew_shaping_params
        },
        # To be passed into OvercookedEnv constructor
        "env_params" : {
            "horizon" : horizon
        },

        # To be passed into OvercookedMultiAgent constructor
        "multi_agent_params" : {
            "gamma" : gamma,
            "reward_shaping_schedule" : reward_shaping_schedule if reward_shaping_schedule else [(0, reward_shaping_factor), (reward_shaping_horizon, 0)],
            "potential_shaping_schedule" : potential_shaping_schedule if potential_shaping_schedule else [(0, potential_shaping_factor), (potential_shaping_horizon, 0)],
            "use_potential_shaping" : use_potential_shaping,
            "use_reward_shaping" : use_reward_shaping,
            "bc_schedule" : bc_schedule,
            "potential_constants" : potential_constants,
            "bc_opt" : bc_opt,
            "ficticious_self_play" : ficticious_self_play,
            "ppo_idx" : ppo_idx
        }
    }

    ray_params = {
        "custom_model_id" : "MyPPOModel",
        "custom_model_cls" : None if model_params['use_lstm'] else RllibPPOModel,
        "temp_dir" : temp_dir,
        "env_creator" : _env_creator
    }

    self_play_params = {
        "ficticious_self_play" : ficticious_self_play,
        "training_iters_per_ensemble_checkpoint" : training_iters_per_ensemble_checkpoint,
        "training_iters_per_ensemble_sample" : training_iters_per_ensemble_sample

    }

    bc_params = {
        "cls" : BehaviorCloningPolicy,
        "config" : {
            "model_dir" : bc_model_dir,
            "stochastic" : bc_stochastic,
            "eager" : eager
        }
    }

    bc_opt_params = {
        "cls" : BC_OPT_CLS_MAP[bc_opt_cls_key],
        "config" : {
            "on_dist_config" : {
                "model_dir" : bc_model_dir,
                "stochastic" : bc_stochastic,
                "eager" : eager
            },
            "off_dist_config" : {
                "opt_path" : opt_path,
                "policy_id" : "ppo"
            }
        }
    }

    ppo_params = {
        "cls" : None,
        "config" : {
            "model" : {
                "custom_model_config" : model_params,
                
                "custom_model" : ray_params['custom_model_id']
            }
        }
    }

    ensemble_ppo_params = {
        "cls" : EnsemblePolicy,
        "config" : {
            "max_policies_in_memory" : max_policies_in_memory
        }
    }

    policy_params = {
        "ppo" : ppo_params,
        "bc" : bc_params,
        "bc_opt" : bc_opt_params,
        "ensemble_ppo" : ensemble_ppo_params
    }


    # Whether to return pointer to trainer in memory in results dict. Useful for debugging
    return_trainer = False

    params = {
        "model_params" : model_params,
        "training_params" : training_params,
        "environment_params" : environment_params,
        "policy_params" : policy_params,
        "shared_policy" : shared_policy,
        "num_training_iters" : num_training_iters,
        "evaluation_params" : evaluation_params,
        "self_play_params" : self_play_params,
        "experiment_name" : experiment_name,
        "save_every" : save_freq,
        "seeds" : seeds,
        "results_dir" : results_dir,
        "ray_params" : ray_params,
        "return_trainer" : return_trainer,
        "verbose" : verbose,
        "log_timestep_events" : log_timestep_events
    }


def run(params):
    # Retrieve the tune.Trainable object that is used for the experiment
    trainer = gen_trainer_from_params(params)

    # Object to store training results in
    result = {}

    # Params that dictate how we organize self-play
    self_play_params = params['self_play_params']

    # Training loop
    for i in range(params['num_training_iters']):
        if params['verbose']:
            print("Starting training iteration", i)
        result = trainer.train()

        save_path = None
        if i % params['save_every'] == 0:
            save_path = save_trainer(trainer, params)
            if params['verbose']:
                print("saved trainer at", save_path)

        if self_play_params['ficticious_self_play'] and i % self_play_params['training_iters_per_ensemble_checkpoint'] == 0:
            if params['verbose']:
                print("Adding checkpoint to ficticious self-play buffer")
            if not save_path:
                save_path = save_trainer(trainer, params)
                if params['verbose']:
                    print("saved trainer at", save_path)
            
            def add_to_ensemble(policy, pid):
                if pid != 'ensemble_ppo':
                    return False
                policy.add_base_policy(save_path)
                return True
            trainer.workers.foreach_policy(
                add_to_ensemble
            )

        if self_play_params['ficticious_self_play'] and i % self_play_params['training_iters_per_ensemble_sample'] == 0:
            if params['verbose']:
                print("Resampling from ficticious self-play buffer")
            def sample_from_buffer(policy, pid):
                if pid != 'ensemble_ppo':
                    return False
                policy.sample_policy()
                return True
            trainer.workers.foreach_policy(
                sample_from_buffer
            )

    # Save the state of the experiment at end
    save_path = save_trainer(trainer, params)
    result['save_path'] = save_path

    # Store pointer to trainer in memeory if requested (useful for testing)
    result['trainer'] = trainer if params['return_trainer'] else None

    if params['verbose']:
        print("saved trainer at", save_path)

    return result


@ex.automain
def main(params):
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
    save_paths = [res['save_path'] for res in results]
    trainers = [res['trainer'] for res in results]
    results =  { "average_sparse_reward" : average_sparse_reward, "average_total_reward" : average_episode_reward, "save_paths" : save_paths, "trainers" : trainers }
    return results
