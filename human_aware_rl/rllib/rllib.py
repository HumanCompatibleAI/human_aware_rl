from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import Agent, AgentPair
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from human_aware_rl.rllib.utils import softmax, get_base_env, get_base_env_lst, get_mlp, get_mlp_lst, get_required_arguments, iterable_equal
from datetime import datetime
import tensorflow as tf
import inspect
import ray
import tempfile
import gym
import numpy as np
import os, pickle, copy
import random

action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
obs_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")


class RlLibAgent(Agent):
    """ 
    Class for wrapping a trained RLLib Policy object into an Overcooked compatible Agent
    """
    def __init__(self, policy, agent_index, featurize_fn):
        self.policy = policy
        self.agent_index = agent_index
        self.featurize = featurize_fn

    def reset(self):
        # Get initial rnn states and add batch dimension to each
        if hasattr(self.policy.model, 'get_initial_state'):
            self.rnn_state = [np.expand_dims(state, axis=0) for state in self.policy.model.get_initial_state()]
        elif hasattr(self.policy, "get_initial_state"):
            self.rnn_state = [np.expand_dims(state, axis=0) for state in self.policy.get_initial_state()]
        else:
            self.rnn_state = []

    def action_probabilities(self, state):
        """
        Arguments:
            - state (Overcooked_mdp.OvercookedState) object encoding the global view of the environment
        returns:
            - Normalized action probabilities determined by self.policy
        """
        # Preprocess the environment state
        obs = self.featurize(state, debug=False)
        my_obs = obs[self.agent_index]

        # Compute non-normalized log probabilities from the underlying model
        logits = self.policy.compute_actions(np.array([my_obs]), self.rnn_state)[2]['action_dist_inputs']

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

        # Use Rllib.Policy class to compute action argmax and action probabilities
        [action_idx], rnn_state, info = self.policy.compute_actions(np.array([my_obs]), self.rnn_state)
        agent_action =  Action.INDEX_TO_ACTION[action_idx]
        
        # Softmax in numpy to convert logits to normalized probabilities
        logits = info['action_dist_inputs']
        action_probabilities = softmax(logits)

        agent_action_info = {'action_probs' : action_probabilities}
        self.rnn_state = rnn_state

        return agent_action, agent_action_info


class OvercookedMultiAgent(MultiAgentEnv):
    """
    Class used to wrap OvercookedEnv in an Rllib compatible multi-agent environment
    """

    # List of all agent types currently supported
    supported_agents = ['ppo', 'bc']

    # Default bc_schedule, includes no bc agent at any time
    bc_schedule = self_play_bc_schedule = [(0, 0), (float('inf'), 0)]

    # Default environment params used for creation
    DEFAULT_CONFIG = {
        # To be passed into OvercookedGridWorld constructor
        "mdp_params_lst" : [{
            "layout_name" : "cramped_room",
            "rew_shaping_params" : {}
        }],
        # To be passed into OvercookedEnv constructor
        "env_params" : {
            "horizon" : 400
        },
        # To be passed into OvercookedMultiAgent constructor
        "multi_agent_params" : {
            "reward_shaping_factor" : 0.0,
            "reward_shaping_horizon" : 0,
            "bc_schedule" : self_play_bc_schedule,
            "use_phi" : True
        }
    }

    def __init__(self, base_env_lst, featurize_fn_lst, reward_shaping_factor=0.0, reward_shaping_horizon=0,
                            bc_schedule=None, use_phi=True):
        """
        base_env_lst (list): a list of OvercookedEnv
        featurize_fn (list): a list of dictionaries mapping agent names to featurization functions of type state -> list(np.array)
        reward_shaping_factor (float): Coefficient multiplied by dense reward before adding to sparse reward to determine shaped reward
        reward_shaping_horizon (int): Timestep by which the reward_shaping_factor reaches zero through linear annealing
        bc_schedule (list[tuple]): List of (t_i, v_i) pairs where v_i represents the value of bc_factor at timestep t_i
            with linear interpolation in between the t_i
        use_phi (bool): Whether to use 'shaped_r_by_agent' or 'phi_s_prime' - 'phi_s' to determine dense reward
        """
        if bc_schedule:
            self.bc_schedule = bc_schedule
        for featurize_fn in featurize_fn_lst:
            self._validate_featurize_fns(featurize_fn)
        self._validate_schedule(self.bc_schedule)
        self.base_env_lst = base_env_lst
        self.featurize_fn_map_lst = copy.deepcopy(featurize_fn_lst)
        self._initial_reward_shaping_factor = reward_shaping_factor
        self.reward_shaping_factor = reward_shaping_factor
        self.reward_shaping_horizon = reward_shaping_horizon
        self.use_phi = use_phi
        self._setup_observation_space()
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        self.anneal_bc_factor(0)
        self.num_env = len(base_env_lst)
        # initalized to -1 because increment by 1 at reset
        self.num_visit = [-1] * self.num_env
        # intialized to true so everything is resetted
        self.altered = [True] * self.num_env
        # the variable that keep track of which base env are we currently on
        self.cur_env_idx = -1
        self.reset()
    
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
        dummy_env_idx = 0
        dummy_state = self.base_env_lst[dummy_env_idx].mdp.get_standard_start_state()
        if 'ppo' in self.featurize_fn_map_lst[dummy_env_idx]:
            featurize_fn = self.featurize_fn_map_lst[dummy_env_idx]['ppo']
            obs_shape = featurize_fn(dummy_state)[0].shape
            high = np.ones(obs_shape) * max(self.base_env_lst[0].mdp.soup_cooking_time, self.base_env_lst[0].mdp.num_items_for_soup, 5)
            self.ppo_observation_space = gym.spaces.Box(high * 0, high, dtype=np.float32)
        if 'bc' in self.featurize_fn_map_lst[dummy_env_idx]:
            featurize_fn = self.featurize_fn_map_lst[dummy_env_idx]['bc']
            obs_shape = featurize_fn(dummy_state)[0].shape
            high = np.ones(obs_shape) * 10
            low = np.ones(obs_shape) * -10
            # Verify this
            self.bc_observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def _get_featurize_fn(self, agent_id):
        if agent_id.startswith('ppo'):
            return self.featurize_fn_map_lst[self.cur_env_idx]['ppo']
        if agent_id.startswith('bc'):
            return self.featurize_fn_map_lst[self.cur_env_idx]['bc']
        raise ValueError("Unsupported agent type {0}".format(agent_id))

    def _get_obs(self, state):
        ob_p0 = self._get_featurize_fn(self.curr_agents[0])(state)[0]
        ob_p1 = self._get_featurize_fn(self.curr_agents[1])(state)[1]

        return ob_p0, ob_p1

    def _populate_agents(self):
        # Always include at least one ppo agent (i.e. bc_sp not supported for simplicity)
        agents = ['ppo']

        # Coin flip to determine whether other agent should be ppo or bc
        other_agent = 'bc' if np.random.uniform() < self.bc_factor else 'ppo'
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


    def step(self, action_dict):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format
        
        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        action = [action_dict[self.curr_agents[0]], action_dict[self.curr_agents[1]]]
        assert all(self.action_space.contains(a) for a in action), "%r (%s) invalid"%(action, type(action))
        joint_action = [Action.INDEX_TO_ACTION[a] for a in action]
        # take a step in the current base environment
        next_state, sparse_reward, done, info = self.base_env_lst[self.cur_env_idx].step(joint_action)
        ob_p0, ob_p1 = self._get_obs(next_state)

        if self.use_phi:
            potential = info['phi_s_prime'] - info['phi_s']
            dense_reward = (potential, potential)
        else:
            dense_reward = info["shaped_r_by_agent"]

        shaped_reward_p0 = sparse_reward + self.reward_shaping_factor * dense_reward[0]
        shaped_reward_p1 = sparse_reward + self.reward_shaping_factor * dense_reward[1]
        
        obs = { self.curr_agents[0]: ob_p0, self.curr_agents[1]: ob_p1 }
        rewards = { self.curr_agents[0]: shaped_reward_p0, self.curr_agents[1]: shaped_reward_p1 }
        dones = { self.curr_agents[0]: done, self.curr_agents[1]: done, "__all__": done }
        infos = { self.curr_agents[0]: info, self.curr_agents[1]: info }
        return obs, rewards, dones, infos

    def reset(self):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to 
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        # FIXME: only fix what is changed

        for i in range(self.num_env):
            if self.altered[i]:
                self.base_env_lst[i].reset()
                self.altered[i] = False
                self.num_visit[i] += 1
        self.curr_agents = self._populate_agents()
        # set the enviroment to be explored next to be the env visited the least
        self.cur_env_idx = self.num_visit.index(min(self.num_visit))
        self.altered[self.cur_env_idx] = True
        print("Resetting env", self.cur_env_idx, ",total number of visits for each env", self.num_visit)
        ob_p0, ob_p1 = self._get_obs(self.base_env_lst[self.cur_env_idx].state)
        return { self.curr_agents[0] : ob_p0, self.curr_agents[1] : ob_p1 }
    
    def anneal_reward_shaping_factor(self, timesteps):
        """
        Set the current reward shaping factor such that we anneal linearly until self.reward_shaping_horizon
        timesteps, given that we are currently at timestep "timesteps"
        """
        new_factor = self._anneal(self._initial_reward_shaping_factor, timesteps, self.reward_shaping_horizon)
        self.set_reward_shaping_factor(new_factor)

    def anneal_bc_factor(self, timesteps):
        """
        Set the current bc factor such that we anneal linearly until self.bc_factor_horizon
        timesteps, given that we are currently at timestep "timesteps"
        """
        p_0 = self.bc_schedule[0]
        p_1 = self.bc_schedule[1]
        i = 2
        while timesteps > p_1[0] and i < len(self.bc_schedule):
            p_0 = p_1
            p_1 = self.bc_schedule[i]
            i += 1
        start_t, start_v = p_0
        end_t, end_v = p_1
        new_factor = self._anneal(start_v, timesteps, end_t, end_v, start_t)
        self.set_bc_factor(new_factor)

    def set_reward_shaping_factor(self, factor):
        self.reward_shaping_factor = factor

    def set_bc_factor(self, factor):
        self.bc_factor = factor

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

        env_config (dict):  Must contain keys 'mdp_params_lst', 'env_params' and 'multi_agent_params', the last of which
                            gets fed into the OvercookedMultiAgent constuctor

        Returns:
            OvercookedMultiAgent instance specified by env_config params
        """
        assert env_config and "mdp_params_lst" in env_config and "env_params" in env_config and "multi_agent_params" in env_config
        # "layout_name" and "rew_shaping_params"
        mdp_params_lst = env_config["mdp_params_lst"]

        # "start_state_fn" and "horizon"
        env_params = env_config["env_params"]
        # "reward_shaping_factor"
        multi_agent_params = env_config["multi_agent_params"]

        base_env_lst = get_base_env_lst(mdp_params_lst, env_params)
        mlp_lst = get_mlp_lst(mdp_params_lst, env_params)

        featurize_fn_map_lst = []
        for i in range(len(mdp_params_lst)):

            ppo_featurize_fn_i = base_env_lst[i].mdp.lossless_state_encoding
            bc_featurize_fn_i = lambda state : base_env_lst[i].mdp.featurize_state(state, mlp_lst[i])

            featurize_fn_map_i = {
                'ppo' : ppo_featurize_fn_i,
                'bc' : bc_featurize_fn_i
            }
            featurize_fn_map_lst.append(featurize_fn_map_i)

        return cls(base_env_lst, featurize_fn_map_lst, **multi_agent_params)



##################
# Training Utils #
##################

class TrainingCallbacks(DefaultCallbacks):
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
        stats_to_collect = ["onion_pickup", "useful_onion_pickup", "onion_drop", "useful_onion_drop", 
            "potting_onion", "dish_pickup", "useful_dish_pickup", "dish_drop", "useful_dish_drop", "soup_pickup", "soup_delivery"]

        # Parse info dicts generated by OvercookedEnv
        tot_sparse_reward = ep_info["ep_sparse_r"]
        tot_shaped_reward = ep_info["ep_shaped_r"]


        # Store metrics where they will be visible to rllib for tensorboard logging
        episode.custom_metrics["sparse_reward"] = tot_sparse_reward
        episode.custom_metrics["shaped_reward"] = tot_shaped_reward

        # Store per-agent game stats to rllib info dicts
        for stat in stats_to_collect:
            stats = game_stats[stat]
            episode.custom_metrics[stat + "_agent_0"] = len(stats[0])
            episode.custom_metrics[stat + "_agent_1"] = len(stats[1])

    def on_sample_end(self, worker, samples, **kwargs):
        pass

    # Executes at the end of a call to Trainer.train, we'll update environment params (like annealing shaped rewards)
    def on_train_result(self, trainer, result, **kwargs):
        # Anneal the reward shaping coefficient based on environment paremeters and current timestep
        timestep = result['timesteps_total']
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.anneal_reward_shaping_factor(timestep)))

        # Anneal the bc factor based on environment paremeters and current timestep
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.anneal_bc_factor(timestep)))

    def on_postprocess_trajectory(self, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
        pass


def get_rllib_eval_function(eval_params, mdp_params_lst, env_params, agent_0_policy_str='ppo', agent_1_policy_str='ppo'):

    """
    Used to "curry" rllib evaluation function by wrapping additional parameters needed in a local scope, and returning a 
    function with rllib custom_evaluation_function compatible signature
    
    eval_params (dict): Contains 'num_games' (int), 'display' (bool), and 'ep_length' (int)
    mdp_params_lst (list): a list of {"layout": layout_name, "rew_shaping_params": reward_shaping_params}
            could be used to create underlying OvercookedMDP (see that class for configuration)
    env_params (dict): Used to create underlying OvercookedEnv (see that class for configuration)
    agent_0_policy_str (str): Key associated with the rllib policy object used to select actions (must be either 'ppo' or 'bc')
    agent_1_policy_str (str): Key associated with the rllib policy object used to select actions (must be either 'ppo' or 'bc')

    Note: Agent policies are shuffled each time, so agent_0_policy_str and agent_1_policy_str are symmetric

    Returns:
        _evaluate (func): Runs an evaluation specified by the curried params, ignores the rllib parameter 'evaluation_workers'
    """

    def _evaluate(trainer, evaluation_workers):
        print("Computing rollout of current trained policy")
        # first, pick a layout among layout_names to assemble the mdp_params for a single layout
        mdp_params_i = random.choice(mdp_params_lst)
        print("randomly choose", mdp_params_i["layout_name"], "this time")

        # Randomize starting indices
        policies = [agent_0_policy_str, agent_1_policy_str]
        np.random.shuffle(policies)
        agent_0_policy, agent_1_policy = policies

        # Get the corresponding rllib policy objects for each policy string name
        agent_0_policy = trainer.get_policy(agent_0_policy)
        agent_1_policy = trainer.get_policy(agent_1_policy)

        agent_0_feat_fn = agent_1_feat_fn = None
        if 'bc' in policies:
            base_env = get_base_env(mdp_params_i, env_params)
            mlp = get_mlp(mdp_params_i, env_params)
            bc_featurize_fn = lambda state : base_env.mdp.featurize_state(state, mlp)
            if policies[0] == 'bc':
                agent_0_feat_fn = bc_featurize_fn
            if policies[1] == 'bc':
                agent_1_feat_fn = bc_featurize_fn

        # Compute the evauation rollout. Note this doesn't use the rllib passed in evaluation_workers, so this 
        # computation all happens on the CPU. Could change this if evaluation becomes a bottleneck
        results = evaluate(eval_params, mdp_params_i, agent_0_policy, agent_1_policy, agent_0_feat_fn, agent_1_feat_fn)

        # Log any metrics we care about for rllib tensorboard visualization
        metrics = {}
        metrics['average_sparse_reward'] = np.mean(results['ep_returns'])
        return metrics

    return _evaluate


def evaluate(eval_params, mdp_params, agent_0_policy, agent_1_policy, agent_0_featurize_fn=None, agent_1_featurize_fn=None):
    """
    Used to visualize rollouts of trained policies

    eval_params (dict): Contains configurations such as the rollout length, number of games, and whether to display rollouts
    mdp_params (dict): OvercookedMDP compatible configuration used to create environment used for evaluation
    agent_0_policy (rllib.Policy): Policy instance used to map states to action logits for agent 0
    agent_1_policy (rllib.Policy): Policy instance used to map states to action logits for agent 1
    agent_0_featurize_fn (func): Used to preprocess states for agent 0, defaults to lossless_state_encoding if 'None'
    agent_1_featurize_fn (func): Used to preprocess states for agent 1, defaults to lossless_state_encoding if 'None'
    """
    evaluator = AgentEvaluator([mdp_params], {"horizon" : eval_params['ep_length']})

    # Override pre-processing functions with defaults if necessary
    agent_0_featurize_fn = agent_0_featurize_fn if agent_0_featurize_fn else evaluator.env_lst[0].mdp.lossless_state_encoding
    agent_1_featurize_fn = agent_1_featurize_fn if agent_1_featurize_fn else evaluator.env_lst[0].mdp.lossless_state_encoding

    # Wrap rllib policies in overcooked agents to be compatible with Evaluator code
    agent0 = RlLibAgent(agent_0_policy, agent_index=0, featurize_fn=agent_0_featurize_fn)
    agent1 = RlLibAgent(agent_1_policy, agent_index=1, featurize_fn=agent_1_featurize_fn)

    # Compute rollouts
    results = evaluator.evaluate_agent_pair(AgentPair(agent0, agent1), num_games=eval_params['num_games'], display=eval_params['display'])

    return results


###########################
# rllib.Trainer functions #
###########################


def gen_trainer_from_params(params):
    # Parse params
    model_params = params['model_params']
    training_params = params['training_params']
    environment_params = params['environment_params']
    evaluation_params = params['evaluation_params']
    bc_params = params['bc_params']
    multi_agent_params = params['environment_params']['multi_agent_params']

    # dummy env to be used as container for data like action and observation spaces
    mdp_params_lst = environment_params["mdp_params_lst"]
    mdp_params_i = random.choice(mdp_params_lst)
    print("randomly choose", mdp_params_i["layout_name"], "this time")

    env = OvercookedMultiAgent.from_config(environment_params)
    environment_params.pop("mdp_params", None)

    # Returns a properly formatted policy tuple to be passed into ppotrainer config
    def gen_policy(policy_type="ppo"):
        # supported policy types thus far
        assert policy_type in ["ppo", "bc"]

        if policy_type == "ppo":
            config = {
                "model" : {
                    "custom_options" : model_params,
                    
                    "custom_model" : "MyPPOModel"
                }
            }
            return (None, env.ppo_observation_space, env.action_space, config)
        elif policy_type == "bc":
            bc_cls = bc_params['bc_policy_cls']
            bc_config = bc_params['bc_config']
            return (bc_cls, env.bc_observation_space, env.action_space, bc_config)

    # Rllib compatible way of setting the directory we store agent checkpoints in
    logdir_prefix = "{0}_{1}_{2}".format(params["experiment_name"], params['training_params']['seed'], timestr)
    def custom_logger_creator(config):
                """Creates a Unified logger that stores results in <params['results_dir']>/<params["experiment_name"]>_<seed>_<timestamp>
                """
                if not os.path.exists(params['results_dir']):
                    os.makedirs(params['results_dir'])
                logdir = tempfile.mkdtemp(
                    prefix=logdir_prefix, dir=params['results_dir'])
                return UnifiedLogger(config, logdir, loggers=None)

    # Create rllib compatible multi-agent config based on params
    multi_agent_config = {}
    all_policies = ['ppo']

    # Whether both agents should be learned
    self_play = iterable_equal(multi_agent_params['bc_schedule'], OvercookedMultiAgent.self_play_bc_schedule)
    if not self_play:
        all_policies.append('bc')

    multi_agent_config['policies'] = { policy : gen_policy(policy) for policy in all_policies }

    def select_policy(agent_id):
        if agent_id.startswith('ppo'):
            return 'ppo'
        if agent_id.startswith('bc'):
            return 'bc'
    multi_agent_config['policy_mapping_fn'] = select_policy
    multi_agent_config['policies_to_train'] = 'ppo'

    trainer = PPOTrainer(env="overcooked_multi_agent", config={
        "multiagent": multi_agent_config,
        "callbacks" : TrainingCallbacks,
        "custom_eval_function" : get_rllib_eval_function(evaluation_params, environment_params['mdp_params_lst'], environment_params['env_params'],
                                        'ppo', 'ppo' if self_play else 'bc'),
        "env_config" : environment_params,
        "eager" : False,
        **training_params
    }, logger_creator=custom_logger_creator)
    return trainer


def save_trainer(trainer, params, path=None):
    config = trainer.get_config()
    save_path = trainer.save(path)
    save_dir = os.path.dirname(save_path)
    config_path = os.path.join(save_dir, "config.pkl")
    config = copy.deepcopy(params)
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    return save_path


# TODO: Make trainer loading not depent on featurize_fn so client code can be cleaner (i.e. parse the 
#   environment from the pickled trainer and grab the lossless_state_encoding from there)

def load_trainer(save_path):
    save_dir = os.path.dirname(save_path)
    config_path = os.path.join(save_dir, "config.pkl")
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    # Override this param to lower overhead in trainer creation
    config['training_params']['num_workers'] = 0
    trainer = gen_trainer_from_params(config)
    trainer.restore(save_path)
    return trainer

def get_agent_pair_from_trainer(trainer, featurize_fn):
    central_policy = trainer.get_policy('agent')
    agent0 = RlLibAgent(central_policy, agent_index=0, featurize_fn=featurize_fn)
    agent1 = RlLibAgent(central_policy, agent_index=0, featurize_fn=featurize_fn)
    return AgentPair(agent0, agent1)


def load_agent_pair(save_path, featurize_fn):
    trainer = load_trainer(save_path)
    return get_agent_pair_from_trainer(trainer, featurize_fn)


