from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, EVENT_TYPES
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import Agent, AgentPair
from ray.rllib import policy
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from human_aware_rl.rllib.policies import UniformPolicy
from human_aware_rl.rllib.utils import get_base_env, softmax, get_base_ae, get_required_arguments, iterable_equal
from human_aware_rl.utils import recursive_dict_update
from datetime import datetime
import tempfile
import gym
import numpy as np
import os, copy, dill
import ray
import logging

action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
obs_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")


class RlLibAgent(Agent):
    """ 
    Class for wrapping a trained RLLib Policy object into an Overcooked compatible Agent
    """
    def __init__(self, policy, agent_index, featurize_fn, stochastic=True):
        """
        Arguments:
            policy (rllib.Policy): Rllib policy we wish to wrap in the 'agent' API
            agent_index (int): {0, 1}, determines whether to parse the 0th or first players observation
            featurize_fn (f OvercookedState -> (np.array, np.array)): Converts OvercookedState objects to encoded
                numpy arrays that will be passed to forward pass of agent's `policy`. 
            stochastic (bool): Whether the `action` method should argmx (if False) or sample from logits (if True)
        """
        self.policy = policy
        self.agent_index = agent_index
        self.featurize = featurize_fn
        self.stochastic = stochastic

    def reset(self):
        # Get initial rnn states and add batch dimension to each
        if hasattr(self.policy, "model") and hasattr(self.policy.model, 'get_initial_state'):
            self.rnn_state = [np.expand_dims(state, axis=0) for state in self.policy.model.get_initial_state()]
        elif hasattr(self.policy, "get_initial_state"):
            self.rnn_state = [np.expand_dims(state, axis=0) for state in self.policy.get_initial_state()]
        else:
            self.rnn_state = []
        return self

    def action_probabilities(self, state):
        """
        Arguments:
            - state (Overcooked_mdp.OvercookedState) object encoding the global view of the environment
        returns:
            - Normalized action probabilities determined by self.policy
        """
        # Preprocess the environment state
        obs = self.featurize(state)
        my_obs = obs[self.agent_index]

        # Compute non-normalized log probabilities from the underlying model
        logits = self.policy.compute_actions(np.array([my_obs]), state_batches=self.rnn_state)[2]['action_dist_inputs']

        # Softmax in numpy to convert logits to normalized probabilities
        return softmax(logits)

    def action(self, state):
        """
        Arguments: 
            - state (Overcooked_mdp.OvercookedState) object encoding the global view of the environment
        returns: 
            - the argmax action for a single observation state
            - action_info (dict) that stores action probabilities under 'action_probs' key
        """
        # Preprocess the environment state
        obs = self.featurize(state)
        my_obs = obs[self.agent_index]

        # Use Rllib.Policy class to compute action argmax and action logits
        [argmax_action_idx], rnn_state, info = self.policy.compute_actions(np.array([my_obs]), state_batches=self.rnn_state)
        argmax_action =  Action.INDEX_TO_ACTION[argmax_action_idx]
        
        # Softmax in numpy to convert logits to normalized probabilities
        logits = np.squeeze(info['action_dist_inputs'])
        action_probabilities = softmax(logits)

        # Sample action space according to logits
        sampled_action_idx = np.random.choice(len(action_probabilities), p=action_probabilities)
        sampled_action = Action.INDEX_TO_ACTION[sampled_action_idx]

        agent_action_info = {'action_probs' : action_probabilities, **info}
        self.rnn_state = rnn_state

        # To argmax or not to argmax; that is the question
        agent_action = sampled_action if self.stochastic else argmax_action

        return agent_action, agent_action_info


class OvercookedMultiAgent(MultiAgentEnv):
    """
    Class used to wrap OvercookedEnv in an Rllib compatible multi-agent environment
    """

    # List of all agent types currently supported
    supported_agents = ['ppo', 'bc', 'bc_opt', 'ensemble_ppo']

    # Default bc_schedule, includes no bc agent at any time
    bc_schedule = self_play_bc_schedule = zero_schedule = [(0, 0), (float('inf'), 0)]

    # Default environment params used for creation
    DEFAULT_CONFIG = {
        # To be passed into OvercookedGridWorld constructor
        "mdp_params" : {
            "layout_name" : "cramped_room",
            "rew_shaping_params" : {}
        },
        # To be passed into OvercookedEnv constructor
        "env_params" : {
            "horizon" : 400
        },
        # To be passed into OvercookedMultiAgent constructor
        "multi_agent_params" : {
            "bc_opt" : False,
            "use_reward_shaping" : False,
            "reward_shaping_schedule" : zero_schedule,
            "use_potential_shaping" : False,
            "potential_shaping_schedule" : zero_schedule,
            "bc_schedule" : self_play_bc_schedule,
            "gamma" : 0.99,
            "potential_constants" : {},
            "ficticious_self_play" : False
        }
    }

    def __init__(self, base_env, use_reward_shaping=False, reward_shaping_schedule=None,
                            use_potential_shaping=False, potential_shaping_schedule=None,
                            bc_schedule=None, gamma=0.99, potential_constants={},
                            bc_opt=False, ficticious_self_play=False, **kwargs):
        """
        base_env: OvercookedEnv
        reward_shaping_factor (float): Coefficient multiplied by dense reward before adding to sparse reward to determine shaped reward
        reward_shaping_horizon (int): Timestep by which the reward_shaping_factor reaches zero through linear annealing
        bc_schedule (list[tuple]): List of (t_i, v_i) pairs where v_i represents the value of bc_factor at timestep t_i
            with linear interpolation in between the t_i
        use_potential_shaping (bool): Whether to use 'shaped_r_by_agent' or 'phi_s_prime' - 'phi_s' to determine dense reward
        bc_opt (bool): Whether the BC agent (if present) is pure BC or BC_OPT meta-agent
        ficticious_self_play (bool): If True, other agent obs key titled 'ensemble_ppo'
        """
        if use_reward_shaping and not reward_shaping_schedule:
            raise ValueError("must specify `reward_shaping_schedule` if `use_reward_shaping` is true")
        if use_potential_shaping and not potential_shaping_schedule:
            raise ValueError("Must specify `potential_shaping_scheduld` if `use_potnetial_shaping` is True")

        if bc_schedule:
            self.bc_schedule = bc_schedule
        
        self.reward_shaping_schedule = reward_shaping_schedule if use_reward_shaping else self.zero_schedule
        self.potential_shaping_schedule = potential_shaping_schedule if use_potential_shaping else self.zero_schedule

        self._validate_schedule(self.bc_schedule)
        self._validate_schedule(self.reward_shaping_schedule)
        self._validate_schedule(self.potential_shaping_schedule)
        self.base_env = base_env
        self.gamma = gamma
        self.potential_constants = potential_constants
        self.use_potential_shaping = use_potential_shaping
        self.use_reward_shaping = use_reward_shaping
        self.bc_opt = bc_opt
        self.ficticious_self_play = ficticious_self_play

        # since we are not passing featurize_fn in as an argument, we create it here and check its validity
        self.featurize_fn_map = {
            "ensemble_ppo" : lambda state: self.base_env.lossless_state_encoding_mdp(state),
            "ppo": lambda state: self.base_env.lossless_state_encoding_mdp(state),
            "bc": lambda state: self.base_env.featurize_state_mdp(state),
            "bc_opt" : lambda state: OvercookedMultiAgent.bc_opt_featurize_fn(self.base_env, state)
        }
        self._validate_featurize_fns(self.featurize_fn_map)
        self._setup_observation_space()
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        self.anneal_bc_factor(0)
        self.anneal_potential_shaping_factor(0)
        self.anneal_reward_shaping_factor(0)
        self.reset()

    @classmethod
    def from_base_env_params(cls, mdp_params, env_params={"horizon" : 400}, multi_agent_params={}):
        base_env = get_base_env(mdp_params, env_params)
        return cls(base_env, **multi_agent_params)

    @staticmethod
    def bc_opt_featurize_fn(base_env, state):
        on_dist_obs = base_env.featurize_state_mdp(state)
        off_dist_obs = base_env.lossless_state_encoding_mdp(state)
        p0_obs = { "on_dist" : on_dist_obs[0], "off_dist" : off_dist_obs[0] }
        p1_obs = { "on_dist" : on_dist_obs[1], "off_dist" : off_dist_obs[1] }

        return p0_obs, p1_obs
    
    def _validate_featurize_fns(self, mapping):
        assert 'ppo' in mapping, "At least one ppo agent must be specified"
        for k, v in mapping.items():
            assert k in self.supported_agents, "Unsuported agent type in featurize mapping {0}".format(k)
            assert callable(v), "Featurize_fn values must be functions"
            assert len(get_required_arguments(v)) == 1, "Featurize_fn value must accept exactly one argument"
    
    def _validate_schedule(self, schedule):
        timesteps = [p[0] for p in schedule]
        values = [p[1] for p in schedule]

        assert len(schedule) >= 2, "Need at least 2 points to linearly interpolate schedule"
        assert schedule[0][0] == 0, "Schedule must start at timestep 0"
        assert all([t >=0 for t in timesteps]), "All timesteps in schedule must be non-negative"
        assert all([v >=0 and v <= 1 for v in values]), "All values in schedule must be between 0 and 1"
        assert sorted(timesteps) == timesteps, "Timesteps must be in increasing order in schedule"

        # To ensure we flatline after passing last timestep
        if (schedule[-1][0] < float('inf')):
            schedule.append((float('inf'), schedule[-1][1]))

    def _setup_observation_space(self):
        dummy_state = self.base_env.mdp.get_standard_start_state()

        #ppo observation
        featurize_fn_ppo = lambda state: self.base_env.lossless_state_encoding_mdp(state)
        obs_shape = featurize_fn_ppo(dummy_state)[0].shape
        high = np.ones(obs_shape) * float("inf")
        low = np.ones(obs_shape) * 0
        self.ppo_observation_space = gym.spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)

        # bc observation
        featurize_fn_bc = lambda state: self.base_env.featurize_state_mdp(state)
        obs_shape = featurize_fn_bc(dummy_state)[0].shape
        high = np.ones(obs_shape) * 100
        low = np.ones(obs_shape) * -100
        self.bc_observation_space = gym.spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)

        # bc_opt observation
        self.bc_opt_observation_space = gym.spaces.Dict({"on_dist" : self.bc_observation_space, "off_dist" : self.ppo_observation_space})

    def _get_featurize_fn(self, agent_id):
        agent_type = '_'.join(agent_id.split('_')[:-1])
        if agent_type not in self.supported_agents:
            raise ValueError("Unsupported agent type {}".format(agent_type))
        return self.featurize_fn_map[agent_type]

    def _get_obs(self, state):
        ob_p0 = self._get_featurize_fn(self.curr_agents[0])(state)[0]
        ob_p1 = self._get_featurize_fn(self.curr_agents[1])(state)[1]

        return ob_p0, ob_p1

    def _populate_agents(self):
        # Always include at least one ppo agent (i.e. bc_sp not supported for simplicity)
        agents = ['ppo']

        # Coin flip to determine whether other agent should be ppo or bc/bc_opt
        include_bc = np.random.uniform() < self.bc_factor
        if include_bc:
            other_agent = 'bc_opt' if self.bc_opt else 'bc'
        else:
            other_agent = 'ensemble_ppo' if self.ficticious_self_play else 'ppo'
        agents.append(other_agent)

        # Randomize starting indices
        np.random.shuffle(agents)

        # Ensure agent names are unique
        agents[0] = agents[0] + '_0'
        agents[1] = agents[1] + '_1'
        
        return agents

    def _anneal(self, start_v, curr_t, end_t, end_v=0, start_t=0):
        if end_t == 0:
            # No annealing if horizon is zero
            return start_v
        else:
            off_t = curr_t - start_t
            # Calculate the new value based on linear annealing formula
            fraction = max(1 - float(off_t) / (end_t - start_t), 0)
            return fraction * start_v + (1 - fraction) * end_v

    def _anneal_from_schedule(self, timestep, schedule):
        p_0 = schedule[0]
        p_1 = schedule[1]
        i = 2
        while timestep > p_1[0] and i < len(schedule):
            p_0 = p_1
            p_1 = schedule[i]
            i += 1
        start_t, start_v = p_0
        end_t, end_v = p_1
        new_factor = self._anneal(start_v, timestep, end_t, end_v, start_t)
        return new_factor


    def step(self, action_dict):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format
        
        returns:
            observation (dict): formatted to be standard input for self.agent_idx's policy
            rewards (dict): by-agent timestep reward
            dones (dict): by-agent done flags
            infos (dict): by-agent info dictionaries
        """
        action = [action_dict[self.curr_agents[0]], action_dict[self.curr_agents[1]]]
        assert all(self.action_space.contains(a) for a in action), "%r (%s) invalid"%(action, type(action))
        joint_action = [Action.INDEX_TO_ACTION[a] for a in action]
        next_state, sparse_reward, done, info = self.base_env.step(joint_action)
        ob_p0, ob_p1 = self._get_obs(next_state)
        phi_s_prime = self.base_env.potential(gamma=self.gamma, potential_constants=self.potential_constants)

        potential_reward = dense_reward = (0, 0)

        if self.use_potential_shaping:
            potential = self.gamma * phi_s_prime - self.phi_s
            potential_reward = (potential, potential)
            self.phi_s = phi_s_prime
        if self.use_reward_shaping:
            dense_reward = info["shaped_r_by_agent"]

        shaped_reward_p0 = sparse_reward + self.reward_shaping_factor * dense_reward[0] + self.potential_shaping_factor * potential_reward[0]
        shaped_reward_p1 = sparse_reward + self.reward_shaping_factor * dense_reward[1] + self.potential_shaping_factor * potential_reward[1]
        
        obs = { self.curr_agents[0]: ob_p0, self.curr_agents[1]: ob_p1 }
        rewards = { self.curr_agents[0]: shaped_reward_p0, self.curr_agents[1]: shaped_reward_p1 }
        dones = { self.curr_agents[0]: done, self.curr_agents[1]: done, "__all__": done }
        infos = { self.curr_agents[0]: info, self.curr_agents[1]: info }
        return obs, rewards, dones, infos

    def reset(self, regen_mdp=True):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to 
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset(regen_mdp)
        self.phi_s = self.base_env.potential(gamma=self.gamma, potential_constants=self.potential_constants)
        self.curr_agents = self._populate_agents()
        ob_p0, ob_p1 = self._get_obs(self.base_env.state)
        return {self.curr_agents[0]: ob_p0, self.curr_agents[1]: ob_p1}

    def anneal_all_factors(self, timesteps):
        """
        Anneal all environment coefficients (reward shaping, potential shaping, bc) according to the provided schedule
        in __init__ and the current tiemstep

        Args:
            timesteps (int): Current number of total environment timesteps so far in training
        """
        self.anneal_bc_factor(timesteps)
        self.anneal_potential_shaping_factor(timesteps)
        self.anneal_reward_shaping_factor(timesteps)
    
    def anneal_reward_shaping_factor(self, timesteps):
        """
        Set the current reward shaping factor such that we anneal linearly until self.reward_shaping_horizon
        timesteps, given that we are currently at timestep "timesteps"
        """
        new_factor = self._anneal_from_schedule(timesteps, self.reward_shaping_schedule)
        self.set_reward_shaping_factor(new_factor)

    def anneal_bc_factor(self, timesteps):
        """
        Set the current bc factor such that we anneal linearly until self.bc_factor_horizon
        timesteps, given that we are currently at timestep "timesteps"
        """

        new_factor = self._anneal_from_schedule(timesteps, self.bc_schedule)
        self.set_bc_factor(new_factor)

    def anneal_potential_shaping_factor(self, timesteps):
        """
        Set the current potential shaping factor such that we anneal linearly according to self.potential_shaping_schedule
        timesteps, given that we are currently at timestep "timesteps"
        """

        new_factor = self._anneal_from_schedule(timesteps, self.potential_shaping_schedule)
        self.set_potential_shaping_factor(new_factor)

    def set_reward_shaping_factor(self, factor):
        self.reward_shaping_factor = factor

    def set_bc_factor(self, factor):
        self.bc_factor = factor

    def set_potential_shaping_factor(self, factor):
        self.potential_shaping_factor = factor

    def seed(self, seed):
        """
        set global random seed to make environment deterministic
        """
        # Our environment is already deterministic
        pass
    
    @classmethod
    def from_config(cls, env_config):
        """
        Factory method for generating environments in style with rllib guidlines

        env_config (dict):  Must contain keys 'mdp_params', 'env_params' and 'multi_agent_params', the last of which
                            gets fed into the OvercookedMultiAgent constuctor

        Returns:
            OvercookedMultiAgent instance specified by env_config params
        """
        assert env_config and "env_params" in env_config and "multi_agent_params" in env_config
        assert "mdp_params" in env_config or "mdp_params_schedule_fn" in env_config, \
            "either a fixed set of mdp params or a schedule function needs to be given"
        # "layout_name" and "rew_shaping_params"
        if "mdp_params" in env_config:
            mdp_params = env_config["mdp_params"]
            outer_shape = None
            mdp_params_schedule_fn = None
        elif "mdp_params_schedule_fn" in env_config:
            mdp_params = None
            outer_shape = env_config["outer_shape"]
            mdp_params_schedule_fn = env_config["mdp_params_schedule_fn"]

        # "start_state_fn" and "horizon"
        env_params = env_config["env_params"]
        # "reward_shaping_factor"
        multi_agent_params = env_config["multi_agent_params"]
        base_ae = get_base_ae(mdp_params, env_params, outer_shape, mdp_params_schedule_fn)
        base_env = base_ae.env

        return cls(base_env, **multi_agent_params)



##################
# Training Utils #
##################

class TrainingCallbacks(DefaultCallbacks):
    def __init__(self, params):
        self.params = params
    
    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        pass

    def on_episode_step(self, worker, base_env, episode, **kwargs):
        pass

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        """
        Used in order to add custom metrics to our tensorboard data

        sparse_reward (int) - total reward from deliveries agent earned this episode
        shaped_reward (int) - total reward shaping reward the agent earned this episode
        """
        # Get rllib.OvercookedMultiAgentEnv refernce from rllib wraper
        env = base_env.get_unwrapped()[0]
        # Both agents share the same info so it doesn't matter whose we use, just use 0th agent's
        info_dict = episode.last_info_for(env.curr_agents[0])

        ep_info = info_dict["episode"]
        game_stats = ep_info["ep_game_stats"]

        # List of episode stats we'd like to collect by agent
        stats_to_collect = EVENT_TYPES

        # Parse info dicts generated by OvercookedEnv
        tot_sparse_reward = ep_info["ep_sparse_r"]
        tot_shaped_reward = ep_info["ep_shaped_r"]
        tot_off_dist_sparse_reward = ep_info["ep_off_dist_sparse_r"]
        tot_off_dist_shaped_reward = ep_info["ep_off_dist_shaped_r"]


        # Store metrics where they will be visible to rllib for tensorboard logging
        episode.custom_metrics["sparse_reward"] = tot_sparse_reward
        episode.custom_metrics["shaped_reward"] = tot_shaped_reward
        episode.custom_metrics["off_dist_sparse_reward"] = tot_off_dist_sparse_reward
        episode.custom_metrics["off_dist_shaped_reward"] = tot_off_dist_shaped_reward

        # Store per-agent game stats to rllib info dicts
        for stat in stats_to_collect:
            stats = game_stats[stat]
            episode.custom_metrics[stat + "_agent_0"] = len(stats[0])
            episode.custom_metrics[stat + "_agent_1"] = len(stats[1])

        # Log environment coefficients
        episode.custom_metrics["reward_shaping_factor"] = env.reward_shaping_factor
        episode.custom_metrics["potential_shaping_factor"] = env.potential_shaping_factor
        episode.custom_metrics["bc_factor"] = env.bc_factor

    def on_sample_end(self, worker, samples, **kwargs):
        pass

    # Executes at the end of a call to Trainer.train, we'll update environment params (like annealing shaped rewards)
    def on_train_result(self, trainer, result, **kwargs):
        timestep = result['timesteps_total']

        # Anneal the reward shaping coefficient, bc factor, and potential shaping factor based on environment paremeters and current timestep
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.anneal_all_factors(timestep)))

    def on_postprocess_trajectory(self, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
        pass

def get_rllib_eval_function(eval_params, eval_mdp_params, env_params, outer_shape, agent_0_policy_str='ppo', agent_1_policy_str='ppo', verbose=False):
    """
    Used to "curry" rllib evaluation function by wrapping additional parameters needed in a local scope, and returning a
    function with rllib custom_evaluation_function compatible signature

    eval_params (dict): Contains 'num_games' (int), 'display' (bool), and 'ep_length' (int)
    mdp_params (dict): Used to create underlying OvercookedMDP (see that class for configuration)
    env_params (dict): Used to create underlying OvercookedEnv (see that class for configuration)
    outer_shape (list): a list of 2 item specifying the outer shape of the evaluation layout
    agent_0_policy_str (str): Key associated with the rllib policy object used to select actions (must be either 'ppo', 'bc', 'bc_opt', or 'ensemble_ppo')
    agent_1_policy_str (str): Key associated with the rllib policy object used to select actions (must be either 'ppo', 'bc', 'bc_opt', or 'ensemble_ppo')
    Returns:
        _evaluate (func): Runs an evaluation specified by the curried params, ignores the rllib parameter 'evaluation_workers'
    """

    def _evaluate(trainer, evaluation_workers):
        if verbose:
            print("Computing rollout of current trained policy")

        # Get the corresponding rllib policy objects for each policy string name
        agent_0_policy = trainer.get_policy(agent_0_policy_str)
        agent_1_policy = trainer.get_policy(agent_1_policy_str)

        # Random policy where actions are sampled uniformly at random
        rnd_policy = UniformPolicy(agent_0_policy.observation_space, agent_0_policy.action_space, {})
        rnd_feat_fn = lambda state : (state, state)

        # Conditionally assign featuriation fns
        rllib_env = OvercookedMultiAgent.from_base_env_params(eval_mdp_params, env_params)
        agent_0_feat_fn = rllib_env.featurize_fn_map[agent_0_policy_str]
        agent_1_feat_fn = rllib_env.featurize_fn_map[agent_1_policy_str]

        # Compute the evauation rollout. Note this doesn't use the rllib passed in evaluation_workers, so this 
        # computation all happens on the CPU. Could change this if evaluation becomes a bottleneck
        sp_results_0 = evaluate(eval_params, eval_mdp_params, outer_shape, agent_0_policy, agent_1_policy, agent_0_feat_fn, agent_1_feat_fn, verbose=verbose)
        sp_results_1 = evaluate(eval_params, eval_mdp_params, outer_shape, agent_1_policy, agent_0_policy, agent_1_feat_fn, agent_0_feat_fn, verbose=verbose)
        rnd_results_0 = evaluate(eval_params, eval_mdp_params, outer_shape, agent_0_policy, rnd_policy, agent_0_feat_fn, rnd_feat_fn, verbose=verbose)
        rnd_results_1 = evaluate(eval_params, eval_mdp_params, outer_shape, rnd_policy, agent_0_policy, rnd_feat_fn, agent_0_feat_fn, verbose=verbose)

        # Log any metrics we care about for rllib tensorboard visualization
        metrics = {}
        metrics['average_sparse_reward_{}_{}'.format(agent_0_policy_str, agent_1_policy_str)] = np.mean(sp_results_0['ep_returns'])
        metrics['average_sparse_reward_{}_{}'.format(agent_1_policy_str, agent_0_policy_str)] = np.mean(sp_results_1['ep_returns'])
        metrics['average_sparse_reward_{}_rnd'.format(agent_0_policy_str)] = np.mean(rnd_results_0['ep_returns'])
        metrics['average_sparse_reward_rnd_{}'.format(agent_0_policy_str)] = np.mean(rnd_results_1['ep_returns'])
        return metrics

    return _evaluate


def evaluate(eval_params, mdp_params, outer_shape, agent_0_policy, agent_1_policy, agent_0_featurize_fn=None, agent_1_featurize_fn=None, verbose=False):
    """
    Used to visualize rollouts of trained policies

    eval_params (dict): Contains configurations such as the rollout length, number of games, and whether to display rollouts
    mdp_params (dict): OvercookedMDP compatible configuration used to create environment used for evaluation
    outer_shape (list): a list of 2 item specifying the outer shape of the evaluation layout
    agent_0_policy (rllib.Policy): Policy instance used to map states to action logits for agent 0
    agent_1_policy (rllib.Policy): Policy instance used to map states to action logits for agent 1
    agent_0_featurize_fn (func): Used to preprocess states for agent 0, defaults to identity if 'None'
    agent_1_featurize_fn (func): Used to preprocess states for agent 1, defaults to identity if 'None'
    """
    evaluator = get_base_ae(mdp_params, {"horizon" : eval_params['ep_length'], "num_mdp":1}, outer_shape)

    # Override pre-processing functions with defaults if necessary
    agent_0_featurize_fn = agent_0_featurize_fn if agent_0_featurize_fn else lambda state : (state, state)
    agent_1_featurize_fn = agent_1_featurize_fn if agent_1_featurize_fn else lambda state : (state, state)

    # Wrap rllib policies in overcooked agents to be compatible with Evaluator code
    agent0 = RlLibAgent(agent_0_policy, agent_index=0, featurize_fn=agent_0_featurize_fn)
    agent1 = RlLibAgent(agent_1_policy, agent_index=1, featurize_fn=agent_1_featurize_fn)

    # Compute rollouts
    if 'store_dir' not in eval_params:
        eval_params['store_dir'] = None
    if 'display_phi' not in eval_params:
        eval_params['display_phi'] = False
    results = evaluator.evaluate_agent_pair(AgentPair(agent0, agent1),
                                            num_games=eval_params['num_games'],
                                            display=eval_params['display'],
                                            dir=eval_params['store_dir'],
                                            display_phi=eval_params['display_phi'],
                                            info=verbose)

    return results


###########################
# rllib.Trainer functions #
###########################


def gen_trainer_from_params(params):
    # All ray environment set-up
    if not ray.is_initialized():
        init_params = {
            "ignore_reinit_error" : True,
            "_temp_dir" : params['ray_params']['temp_dir'],
            "log_to_driver" : params['verbose'],
            "logging_level" : logging.INFO if params['verbose'] else logging.CRITICAL
        }
        ray.init(**init_params)
    register_env("overcooked_multi_agent", params['ray_params']['env_creator'])
    ModelCatalog.register_custom_model(params['ray_params']['custom_model_id'], params['ray_params']['custom_model_cls'])

    # Parse params
    training_params = params['training_params']
    environment_params = params['environment_params']
    evaluation_params = params['evaluation_params']
    # bc_params = params['bc_params']
    # bc_opt_params = params['bc_opt_params']
    # ppo_params = params['ppo_params']
    # ensemble_ppo_params = params['ensemble_ppo_params']
    policy_params = params['policy_params']
    multi_agent_params = params['environment_params']['multi_agent_params']

    env = OvercookedMultiAgent.from_config(environment_params)

    # Returns a properly formatted policy tuple to be passed into ppotrainer config
    def gen_policy(policy_type="ppo"):
        # supported policy types thus far
        assert policy_type in OvercookedMultiAgent.supported_agents

        curr_policy_params = policy_params[policy_type]
        policy_cls = curr_policy_params['cls']
        policy_config = curr_policy_params['config']

        policy_observation_space = None
        if policy_type == 'ppo' or policy_type == 'ensemble_ppo':
            policy_observation_space = env.ppo_observation_space
        elif policy_type == 'bc':
            policy_observation_space = env.bc_observation_space
        else:
            policy_observation_space = env.bc_opt_observation_space
        return (policy_cls, policy_observation_space, env.action_space, policy_config)

    # Rllib compatible way of setting the directory we store agent checkpoints in
    logdir_prefix = params['logdir_prefix'] if 'logdir_prefix' in params else "{0}_{1}_{2}".format(params["experiment_name"], params['training_params']['seed'], timestr)
    def custom_logger_creator(config):
        """
        Creates a Unified logger that stores results in <params['results_dir']>/<params["experiment_name"]>_<seed>_<timestamp>
        """
        results_dir = params['results_dir']
        if not os.path.exists(results_dir):
            try:
                os.makedirs(results_dir)
            except Exception as e:
                print("error creating custom logging dir: {}. Falling back to default logdir {}".format(str(e), DEFAULT_RESULTS_DIR))
                results_dir = DEFAULT_RESULTS_DIR
        logdir = params['logdir'] if 'logdir' in params else tempfile.mkdtemp(
            prefix=logdir_prefix, dir=results_dir)
        logger = UnifiedLogger(config, logdir, loggers=None)
        return logger

    # Create rllib compatible multi-agent config based on params
    multi_agent_config = {}
    all_policies = set(['ppo'])
    other_policy = 'ensemble_ppo' if multi_agent_params['ficticious_self_play'] else 'ppo'

    # Whether both agents should be learned
    self_play = iterable_equal(multi_agent_params['bc_schedule'], OvercookedMultiAgent.self_play_bc_schedule)
    if not self_play:
        other_policy = 'bc_opt' if multi_agent_params['bc_opt'] else 'bc' 
    all_policies.add(other_policy)

    multi_agent_config['policies'] = { policy : gen_policy(policy) for policy in all_policies }

    def select_policy(agent_id):
        agent_type = '_'.join(agent_id.split('_')[:-1])
        assert agent_type in OvercookedMultiAgent.supported_agents
        return agent_type

    multi_agent_config['policy_mapping_fn'] = select_policy
    multi_agent_config['policies_to_train'] = ['ppo']

    if "outer_shape" not in environment_params:
        environment_params["outer_shape"] = None

    if "mdp_params" in environment_params:
        environment_params["eval_mdp_params"] = environment_params["mdp_params"]
    trainer = PPOTrainer(env="overcooked_multi_agent", config={
        "multiagent": multi_agent_config,
        "callbacks" : lambda : TrainingCallbacks(params),
        "custom_eval_function" : get_rllib_eval_function(evaluation_params, environment_params['eval_mdp_params'], environment_params['env_params'],
                                        environment_params["outer_shape"], 'ppo', other_policy,
                                        verbose=params['verbose']),
        "env_config" : environment_params,
        **training_params
    }, logger_creator=custom_logger_creator)
    return trainer



### Serialization ###


def save_trainer(trainer, params, path=None):
    """
    Saves a serialized trainer checkpoint at `path`. If none provided, the default path is
    ~/ray_results/<experiment_results_dir>/checkpoint_<i>/checkpoint-<i>

    Note that `params` should follow the same schema as the dict passed into `gen_trainer_from_params`
    """
    # Save trainer
    save_path = trainer.save(path)

    # Save params used to create trainer in /path/to/checkpoint_dir/config.pkl
    config = copy.deepcopy(params)
    config_path = os.path.join(os.path.dirname(save_path), "config.pkl")

    # Note that we use dill (not pickle) here because it supports function serialization
    with open(config_path, "wb") as f:
        dill.dump(config, f)
    return save_path

def load_trainer(save_path, **params_to_override):
    """
    Returns a ray compatible trainer object that was previously saved at `save_path` by a call to `save_trainer`
    Note that `save_path` is the full path to the checkpoint FILE, not the checkpoint directory
    """
    # Ensure tf is executing in graph mode
    import tensorflow as tf
    if tf.executing_eagerly():
        tf.compat.v1.disable_eager_execution()
    # Read in params used to create trainer
    checkpoint_dir = os.path.dirname(save_path)
    experiment_dir = os.path.dirname(checkpoint_dir)
    config_path = os.path.join(checkpoint_dir, "config.pkl")
    with open(config_path, "rb") as f:
        # We use dill (instead of pickle) here because we must deserialize functions
        config = dill.load(f)
    
    # Override this param to lower overhead in trainer creation
    config['training_params']['num_workers'] = 0
    config['logdir'] = experiment_dir

    for param, val in params_to_override.items():
        updated = recursive_dict_update(config, param, val)
        if not updated:
            print("WARNING, no value for specified bc argument {} found in schema. Adding as top level parameter".format(param))

    # Get un-trained trainer object with proper config
    trainer = gen_trainer_from_params(config)

    # Load weights into dummy object
    trainer.restore(save_path)
    return trainer

def get_agent_from_trainer(trainer, policy_id="ppo", agent_kwargs={"agent_index" : 0}):
    policy = trainer.get_policy(policy_id)
    dummy_env = trainer.env_creator(trainer.config['env_config'])
    featurize_fn = dummy_env.featurize_fn_map[policy_id]
    agent = RlLibAgent(policy, featurize_fn=featurize_fn, **agent_kwargs)
    return agent

def get_agent_pair_from_trainer(trainer, policy_id_0='ppo', policy_id_1='ppo', agent_1_kwargs={}, agent_2_kwargs={}):
    agent0 = get_agent_from_trainer(trainer, policy_id=policy_id_0, agent_kwargs=agent_1_kwargs)
    agent1 = get_agent_from_trainer(trainer, policy_id=policy_id_1, agent_kwargs=agent_2_kwargs)
    return AgentPair(agent0, agent1)


def load_agent_pair(save_path, policy_id_0='ppo', policy_id_1='ppo', agent_1_kwargs={"agent_index" : 0}, agent_2_kwargs={"agent_index" : 1}, trainer_params_to_override={}):
    """
    Returns an Overcooked AgentPair object that has as player 0 and player 1 policies with 
    ID policy_id_0 and policy_id_1, respectively
    """
    trainer = load_trainer(save_path, **trainer_params_to_override)
    return get_agent_pair_from_trainer(trainer, policy_id_0, policy_id_1, agent_1_kwargs, agent_2_kwargs)

def load_agent(save_path, policy_id='ppo', agent_kwargs={"agent_index" : 0}, trainer_params_to_override={}):
    """
    Returns an RllibAgent (compatible with the Overcooked Agent API) from the `save_path` to a previously
    serialized trainer object created with `save_trainer`

    The trainer can have multiple independent policies, so extract the one with ID `policy_id` to wrap in
    an RllibAgent

    Agent index indicates whether the agent is player zero or player one (or player n in the general case)
    as the featurization is not symmetric for both players
    """
    trainer = load_trainer(save_path, **trainer_params_to_override)
    return get_agent_from_trainer(trainer, policy_id=policy_id, agent_kwargs=agent_kwargs)


