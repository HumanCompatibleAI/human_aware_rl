import gym
import time
import numpy as np
import tensorflow as tf

from overcooked_ai_py.mdp.actions import Direction, Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator
from overcooked_ai_py.agents.agent import AgentFromPolicy, AgentPair
from overcooked_ai_py.utils import load_pickle, save_pickle, load_dict_from_file

from human_aware_rl.utils import create_dir_if_not_exists, num_tf_params, get_max_iter

from baselines.ppo2.ppo2 import learn
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.models import register
        

class RewardShapingEnv(VecEnvWrapper):
    """
    Wrapper for the Baselines vectorized environment, which
    modifies the reward obtained to be a combination of intrinsic
    (dense, shaped) and extrinsic (sparse, from environment) reward"""

    def __init__(self, env, reward_shaping_factor=0.0):
        super().__init__(env)
        self.reward_shaping_factor = reward_shaping_factor
        self.env_name = "Overcooked-v0"

        ### Set various attributes to false, than will then be overwritten by various methods

        # Whether we want to query the actual action method from the agent class, 
        # or we use direct_action. Might change things if there is post-processing 
        # of actions returned, as in the Human Model
        self.use_action_method = False

        # Fraction of self-play actions/trajectories (depending on value of self.trajectory_sp)
        self.self_play_randomization = 0.0
        
        # Whether SP randomization should be done on a trajectory level
        self.trajectory_sp = False

        # Whether the model is supposed to output the joint action for all agents (centralized policy)
        # Joint action models are currently deprecated.
        self.joint_action_model = False

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rew, done, infos = self.venv.step_wait()
        # replace rew with shaped rew
        for env_num in range(self.num_envs):
            dense_reward = infos[env_num]['shaped_r']
            rew = list(rew)
            shaped_rew = rew[env_num] + float(dense_reward) * self.reward_shaping_factor
            rew[env_num] = shaped_rew

            if done[env_num]:
                # Log both sparse and dense rewards for episode
                sparse_ep_rew = infos[env_num]['episode']['ep_sparse_r']
                dense_ep_rew = infos[env_num]['episode']['ep_shaped_r']
                infos[env_num]['episode']['r'] = sparse_ep_rew + dense_ep_rew * self.reward_shaping_factor

        return obs, rew, done, infos

    def update_reward_shaping_param(self, reward_shaping_factor):
        """Takes in what fraction of the run we are at, and determines the reward shaping coefficient"""
        self.reward_shaping_factor = reward_shaping_factor

class LinearAnnealer():
    """Anneals a parameter from 1 to 0 over the course of training,
    over a specified horizon."""

    def __init__(self, horizon):
        self.horizon = horizon

    def param_value(self, timestep):
        if self.horizon == 0:
            return 0
        curr_value = max(1 - (timestep / self.horizon), 0)
        assert 0 <= curr_value <= 1
        return curr_value

class DummyEnv(object):
    """
    Class used to save number of envs, observation space and action 
    space data, when loading and saving baselines models
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        pass

########################
# UTILS AND HELPER FNS #
########################

@register("conv_and_mlp")
def conv_network_fn(**kwargs):
    """Used to register custom network type used by Baselines for Overcooked"""

    if "network_kwargs" in kwargs.keys():
        params = kwargs["network_kwargs"]
    else:
        params = kwargs

    num_hidden_layers = params["NUM_HIDDEN_LAYERS"]
    size_hidden_layers = params["SIZE_HIDDEN_LAYERS"]
    num_filters = params["NUM_FILTERS"]
    num_convs = params["NUM_CONV_LAYERS"]

    def network_fn(X):
        print(X.shape)
        
        conv_out = tf.layers.conv2d(
            inputs=X, 
            filters=num_filters, 
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_initial"
        )

        for i in range(0, num_convs - 1):
            padding = "same" if i < num_convs - 2 else "valid"
            conv_out = tf.layers.conv2d(
                inputs=conv_out,
                filters=num_filters,
                kernel_size=[3, 3],
                padding=padding,
                activation=tf.nn.leaky_relu,
                name="conv_{}".format(i)
            )
        
        out = tf.layers.flatten(conv_out)
        for _ in range(num_hidden_layers):
            out = tf.layers.dense(out, size_hidden_layers, activation=tf.nn.leaky_relu)
        
        print("Last layer conv network output shape", out.shape)
        
        # NOTE: not sure if not supposed to add linear layer. I think it is though, 
        # as things work and similar to code in baseline/models.py? Maybe double check later.

        # To check how many parameters uncomment next line
        # num_tf_params()
        return out
    return network_fn

def get_vectorized_gym_env(base_env, gym_env_name, featurize_fn=None, **kwargs):
    """
    Create a one-player overcooked gym environment in which the other player is fixed (embedded in the environment)
    
    base_env: A OvercookedEnv instance (fixed or variable map)
    sim_threads: number of threads used during simulation, that corresponds to the number of parallel
                 environments used
    """
    def gym_env_fn():
        gym_env = gym.make(gym_env_name)
        if kwargs["RUN_TYPE"] == "joint_ppo":
            # If doing joint training, action space will be different (^2 compared to single agent training)
            gym_env.custom_init(base_env, joint_actions=True, featurize_fn=featurize_fn, baselines=True)    
        else:
            gym_env.custom_init(base_env, featurize_fn=featurize_fn, baselines=True)
        return gym_env
    vectorized_gym_env = RewardShapingEnv(SubprocVecEnv([gym_env_fn] * kwargs["sim_threads"]))
    return vectorized_gym_env

def get_pbt_agent_from_config(save_dir, sim_threads, seed, agent_idx=0, best=False):
    agent_folder = save_dir + 'seed_{}/agent{}'.format(seed, agent_idx)
    if best:
        agent_to_load_path = agent_folder  + "/best"
    else:
        agent_to_load_path = agent_folder  + "/pbt_iter" + str(get_max_iter(agent_folder))
    agent = get_agent_from_saved_model(agent_to_load_path, sim_threads)
    return agent

def get_agent_from_saved_model(save_dir, sim_threads):
    """Get Agent corresponding to a saved model"""
    # NOTE: Could remove dependency on sim_threads if get the sim_threads from config or dummy env
    state_policy, processed_obs_policy = get_model_policy_from_saved_model(save_dir, sim_threads)
    return AgentFromPolicy(state_policy, processed_obs_policy)

def get_agent_from_model(model, sim_threads, is_joint_action=False):
    """Get Agent corresponding to a loaded model"""
    state_policy, processed_obs_policy = get_model_policy_from_model(model, sim_threads, is_joint_action=is_joint_action)
    return AgentFromPolicy(state_policy, processed_obs_policy)

def get_model_policy_from_saved_model(save_dir, sim_threads):
    """Get a policy function from a saved model"""
    predictor = tf.contrib.predictor.from_saved_model(save_dir)
    step_fn = lambda obs: predictor({"obs": obs})["action_probs"]
    return get_model_policy(step_fn, sim_threads)

def get_model_policy_from_model(model, sim_threads, is_joint_action=False):
    def step_fn(obs):
        action_probs = model.act_model.step(obs, return_action_probs=True)
        return action_probs
    return get_model_policy(step_fn, sim_threads, is_joint_action=is_joint_action)

def get_model_policy(step_fn, sim_threads, is_joint_action=False):
    """
    Returns the policy function `p(s, index)` from a saved model at `save_dir`.
    
    step_fn: a function that takes in observations and returns the corresponding
             action probabilities of the agent
    """
    def encoded_state_policy(observations, stochastic=True, return_action_probs=False):
        """Takes in SIM_THREADS many losslessly encoded states and returns corresponding actions"""
        action_probs_n = step_fn(observations)

        if return_action_probs:
            return action_probs_n
        
        if stochastic:
            action_idxs = [np.random.choice(len(Action.ALL_ACTIONS), p=action_probs) for action_probs in action_probs_n]
        else:
            action_idxs = [np.argmax(action_probs) for action_probs in action_probs_n]

        return np.array(action_idxs)

    def state_policy(mdp_state, mdp, agent_index, stochastic=True, return_action_probs=False):
        """Takes in a Overcooked state object and returns the corresponding action"""
        obs = mdp.lossless_state_encoding(mdp_state)[agent_index]
        padded_obs = np.array([obs] + [np.zeros(obs.shape)] * (sim_threads - 1))
        action_probs = step_fn(padded_obs)[0] # Discards all padding predictions

        if return_action_probs:
            return action_probs

        if stochastic:
            action_idx = np.random.choice(len(action_probs), p=action_probs)
        else:
            action_idx = np.argmax(action_probs)

        if is_joint_action:
            # NOTE: Probably will break for this case, untested
            action_idxs = Action.INDEX_TO_ACTION_INDEX_PAIRS[action_idx]
            joint_action = [Action.INDEX_TO_ACTION[i] for i in action_idxs]
            return joint_action

        return Action.INDEX_TO_ACTION[action_idx]

    return state_policy, encoded_state_policy

def create_model(env, agent_name, use_pretrained_weights=False, **kwargs):
    """Creates a model and saves it at a location
    
    env: a dummy environment that is used to determine observation and action spaces
    agent_name: the scope under which the weights of the agent are saved
    """
    model, _ = learn(
        network=kwargs["NETWORK_TYPE"], 
        env=env, 
        total_timesteps=1, 
        save_interval=0, 
        nsteps=kwargs["BATCH_SIZE"],
        nminibatches=kwargs["MINIBATCHES"],
        noptepochs=kwargs["STEPS_PER_UPDATE"],
        scope=agent_name,
        network_kwargs=kwargs
    )
    model.agent_name = agent_name
    model.dummy_env = env
    return model

def save_baselines_model(model, save_dir):
    """
    Saves Model (from baselines) into `path/model` file, 
    and saves the tensorflow graph in the `path` directory
    
    NOTE: Overwrites previously saved models at the location
    """
    create_dir_if_not_exists(save_dir)
    model.save(save_dir + "/model")
    # We save the dummy env so that one doesn't 
    # have to pass in an actual env to load the model later, 
    # as the only information taken from the env are these parameters
    # at test time (if no training happens)
    dummy_env = DummyEnv(
        model.dummy_env.num_envs, 
        model.dummy_env.observation_space, 
        model.dummy_env.action_space
    )
    save_pickle(dummy_env, save_dir + "/dummy_env")

def load_baselines_model(save_dir, agent_name, config):
    """
    NOTE: Before using load it might be necessary to clear the tensorflow graph
    if there are already other variables defined
    """
    dummy_env = load_pickle(save_dir + "/dummy_env")
    model, _ = learn(
        network='conv_and_mlp',
        env=dummy_env,
        total_timesteps=0,
        load_path=save_dir + "/model",
        scope=agent_name,
        network_kwargs=config
    )
    model.dummy_env = dummy_env
    return model

def update_model(env, model, **kwargs):
    """
    Train agent defined by a model using the specified environment.

    The idea is that one can update model on a different environment than the one 
    that was used to create the model (vs a different agent for example, where the
    agent is embedded within the environment)
    """
    def model_fn(**kwargs):
        return model

    updated_model, run_info = learn(
        network=kwargs["NETWORK_TYPE"], 
        env=env, 
        total_timesteps=kwargs["PPO_RUN_TOT_TIMESTEPS"],
        nsteps=kwargs["BATCH_SIZE"],
        ent_coef=kwargs["ENTROPY"], 
        lr=kwargs["LR"], 
        vf_coef=kwargs["VF_COEF"],
        max_grad_norm=kwargs["MAX_GRAD_NORM"], 
        gamma=kwargs["GAMMA"], 
        lam=kwargs["LAM"],
        nminibatches=kwargs["MINIBATCHES"],
        noptepochs=kwargs["STEPS_PER_UPDATE"],
        cliprange=kwargs["CLIPPING"],
        model_fn=model_fn,
        save_interval=0, 
        log_interval=1,
        network_kwargs=kwargs
    )
    return run_info

def overwrite_model(model_from, model_to):
    model_from_vars = tf.trainable_variables(model_from.scope)
    model_to_vars = tf.trainable_variables(model_to.scope)
    overwrite_variables(model_from_vars, model_to_vars)

def overwrite_variables(variables_to_copy, variables_to_overwrite):
    sess = tf.get_default_session()
    restores = []
    assert len(variables_to_copy) == len(variables_to_overwrite), 'number of variables loaded mismatches len(variables)'
    for d, v in zip(variables_to_copy, variables_to_overwrite):
        restores.append(v.assign(d))
    sess.run(restores)
    

############################
#### DEPRECATED METHODS ####
############################

def get_model_value_fn(model, sim_threads, debug=False):
    """Returns the estimated value function `V(s, index)` from a saved model at `save_dir`."""
    print(model)
    def value_fn(mdp_state, mdp, agent_index):
        obs = mdp.lossless_state_encoding(mdp_state, debug=debug)[agent_index]
        padded_obs = np.array([obs] + [np.zeros(obs.shape)] * (sim_threads - 1))
        a, v, state, neglogp = model.act_model.step(padded_obs)
        return v[0]
    return value_fn

def get_model_value_fn_policy(model, sim_threads, boltzmann_rationality=1):
    """Returns a policy based on the value function approximation of the model"""
    v_fn = get_model_value_fn(model, sim_threads)

    def v_policy(mdp_state, mdp, agent_index):
        # Array in which idx corresponds to action with same idx encoding
        successor_vals = []

        for a in Action.INDEX_TO_ACTION:
            joint_action = (a, Direction.STAY) if agent_index == 0 else (Direction.STAY, a)
            s_prime = mdp.get_state_transition(mdp_state, joint_action)[0][0][0]
            s_prime_val = v_fn(s_prime, mdp, agent_index)

            successor_vals.append(s_prime_val)

        numerator = boltzmann_rationality * np.exp(successor_vals)
        normalizer = sum(numerator)

        num_actions = len(Action.INDEX_TO_ACTION)

        if normalizer != 0:
            probability_distribution = numerator / normalizer
        else:
            probability_distribution = np.ones(num_actions) / num_actions
        
        action_idx_array = list(range(num_actions))
        sampled_action_idx = np.random.choice(action_idx_array, p=probability_distribution)
        return Action.INDEX_TO_ACTION[sampled_action_idx]

    return v_policy

def get_boltzmann_rational_agent_from_model(model, sim_threads, boltzmann_rationality):
    p = get_model_value_fn_policy(model, sim_threads, boltzmann_rationality=boltzmann_rationality)
    trained_agent = AgentFromPolicy(p, None)
    return trained_agent

