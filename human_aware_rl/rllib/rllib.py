from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, EVENT_TYPES
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import Agent, AgentPair
from human_aware_rl.imitation import default_bc_params
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import Policy as RllibPolicy
from ray.rllib.agents.ppo.ppo import PPOTFPolicy
from ray.rllib.utils.tf_run_builder import TFRunBuilder
from human_aware_rl.rllib.utils import softmax, get_base_ae, get_required_arguments, \
    get_encoding_function, get_gym_space, get_overcooked_obj_attr
from datetime import datetime
import gym
import numpy as np
import os, copy, dill, random, tempfile
import ray


action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
obs_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    
class DictObsSpacePPOTFPolicy(PPOTFPolicy):
    """
    Object used to run policy with dict observation spaces after training
    """     
    def compute_actions(
            self,
            obs_batch,
            state_batches = None,
            prev_action_batch = None,
            prev_reward_batch = None,
            info_batch = None,
            episodes = None,
            explore = None,
            timestep = None,
            **kwargs):
        """
        method used in PPOTFPolicy but edited to handle dict inputs at runtime  (it is handled 
            at training by existing rllib code, but not for using already trained model)
        """
        explore = explore if explore is not None else self.config["explore"]
        timestep = timestep if timestep is not None else self.global_timestep

        builder = TFRunBuilder(self._sess, "compute_actions")
        if type(obs_batch) is dict:
            some_batch_data = list(obs_batch.values())[0]
            obs_batch_len = len(some_batch_data) if isinstance(some_batch_data, list) \
                else some_batch_data.shape[0]
            flattened_obs = []
            for k in self.observation_space.original_space.spaces.keys():
                if k in obs_batch:
                    obs = np.array(obs_batch[k])
                    flattened_obs.append(obs.reshape(obs_batch_len, np.prod(obs.shape[1:])))
            obs_batch = np.concatenate(flattened_obs, axis=-1)       
        else:
            obs_batch_len = len(obs_batch) if isinstance(obs_batch, list) \
                else obs_batch.shape[0]
            obs_batch = np.array(obs_batch)

        to_fetch = self._build_compute_actions(
            builder,
            obs_batch=obs_batch,
            state_batches=state_batches,
            prev_action_batch=prev_action_batch,
            prev_reward_batch=prev_reward_batch,
            explore=explore,
            timestep=timestep)

        # Execute session run to get action (and other fetches).
        fetched = builder.get(to_fetch)

        # Update our global timestep by the batch size.
        self.global_timestep += obs_batch_len
        return fetched

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
        if type(obs) is dict:
            my_obs = {k: np.array([o[self.agent_index]]) for k,o in obs.items()}
            logits = self.policy.compute_actions(my_obs, self.rnn_state)[2]['action_dist_inputs']
        else:
            my_obs = np.array([obs[self.agent_index]])
            # Compute non-normalized log probabilities from the underlying model
            logits = self.policy.compute_actions(my_obs, self.rnn_state)[2]['action_dist_inputs']

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

        # Use Rllib.Policy class to compute action argmax and action probabilities
        if type(obs) is dict:
            my_obs = {k: np.array([o[self.agent_index]]) for k,o in obs.items()}
            [action_idx], rnn_state, info = self.policy.compute_actions(my_obs, self.rnn_state)
        else:
            my_obs = obs[self.agent_index]
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

    # Default self play ppo schedule
    agents_schedule = self_play_schedule = self_play_bc_schedule = [
        {"timestep": 0, "agents": [{"ppo": 1}, {"ppo": 1}]},
        {"timestep": float('inf'), "agents": [{"ppo": 1}, {"ppo": 1}]}
    ]

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
            "reward_shaping_factor" : 0.0,
            "reward_shaping_horizon" : 0,
            "bc_schedule": None,
            "agents_schedule": agents_schedule,
            "use_phi" : True,
            "shuffle_agents": True
        }
    }

    ml_agents = ["ppo", "bc"]
    default_featurize_fns = {
        "ppo": "env.lossless_state_encoding_mdp",
        "bc": default_bc_params.DEFAULT_DATA_PARAMS["state_processing_function"]
        }
    default_observation_spaces = {
        "ppo": "mdp.lossless_state_encoding_gym_space",
        "bc":  "mdp.lossless_state_encoding_gym_space"
    }
    def __init__(self, base_env, reward_shaping_factor=0.0, reward_shaping_horizon=0,
                            bc_schedule=None, agents_schedule=None, use_phi=True, non_ml_agents_params=None, shuffle_agents=True,
                            featurize_fns={}, observation_spaces={}):
        """
        base_env: OvercookedEnv
        reward_shaping_factor (float): Coefficient multiplied by dense reward before adding to sparse reward to determine shaped reward
        reward_shaping_horizon (int): Timestep by which the reward_shaping_factor reaches zero through linear annealing
        agents_schedule (list[dict]):  List of dicts where key "agents" is list of dicts representing probability of having agent 
            of given type for every player at given timestep (with key "timestep") with linear interpolation in between the timesteps
            example dict: {"timestep": 10, "agents": [{"ppo":1}, {"ppo":0.3, "bc": 0.7}]}
        bc_schedule (list[tuple]): List of (t_i, v_i) pairs where v_i represents the value of bc_factor at timestep t_i
                with linear interpolation in between the t_i; can be used instead of agents_schedule
        use_phi (bool): Whether to use 'shaped_r_by_agent' or 'phi_s_prime' - 'phi_s' to determine dense reward
        non_ml_agents_params (dict): Params used to initialize non_ml_agents
        """
        self.non_ml_agents_params = non_ml_agents_params
        self._initial_reward_shaping_factor = reward_shaping_factor
        self.reward_shaping_factor = reward_shaping_factor
        self.reward_shaping_horizon = reward_shaping_horizon
        self.use_phi = use_phi
        self.bc_schedule = bc_schedule

        if bc_schedule:
            self.agents_schedule = OvercookedMultiAgent.bc_schedule_to_agents_schedule(bc_schedule)
            print(f"converted bc schedule to agents schedule {bc_schedule} -> {self.agents_schedule}")
        else:
            self.agents_schedule = agents_schedule

        self._validate_schedule(self.agents_schedule)
        self.base_env = base_env
        self._setup_featurize_fns(featurize_fns)
        self._setup_observation_space(observation_spaces)
        assert set(self.featurize_fns.keys()) == set(self.observation_spaces.keys())
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        self.shuffle_agents = shuffle_agents
        self.anneal_agents_schedule(0)
        self.reset()


    @staticmethod
    def bc_schedule_to_agents_schedule(bc_schedule):
        '''
        detect and changes possible old bc_schedule to new, more flexible format
        old format:
            bc_schedule (list[tuple]): List of (t_i, v_i) pairs where v_i represents the value of bc_factor at timestep t_i
                with linear interpolation in between the t_i
        new format:
            bc_schedule (list[dict]):  List of dicts where key "agents" is list of dicts representing probability of having agent 
                of given type for every player at given timestep (with key "timestep") with linear interpolation in between the timesteps
                example dict: {"timestep": 10, "agents": [{"ppo":1}, {"ppo":0.3, "bc": 0.7}]}
        '''
        return [{
                    "timestep": t_i,
                    "agents": [{"ppo": 1}, {"ppo": 1-float(bc_proba), "bc": bc_proba}]
                }
                for t_i, bc_proba in bc_schedule]
    
    @staticmethod
    def agent_id_to_agent_name(agent_id):
        # delete agent idx (characters after last "_" chart)
        assert OvercookedMultiAgent.is_valid_agent_id(agent_id)
        return "_".join(agent_id.split("_")[:-1])

    @staticmethod
    def is_valid_agent_id(agent_id):
        return len(agent_id.split("_")) > 1
        
    @staticmethod
    def agent_id_to_agent_idx(agent_id):
        assert OvercookedMultiAgent.is_valid_agent_id(agent_id)
        return int(agent_id.split("_")[-1])

    @staticmethod
    def is_ml_agent(agent_id_or_name):
        is_ml_agent_name = agent_id_or_name in OvercookedMultiAgent.ml_agents
        is_ml_agent_id = OvercookedMultiAgent.is_valid_agent_id(agent_id_or_name) \
            and OvercookedMultiAgent.agent_id_to_agent_name(agent_id_or_name) in OvercookedMultiAgent.ml_agents
        return is_ml_agent_name or is_ml_agent_id
    
    def _validate_schedule(self, schedule):
        timesteps = [p["timestep"] for p in schedule]
        values = [[list(player_probas.values()) for player_probas in p["agents"]] for p in schedule]
        assert all([len(p) == 2 for p in schedule]), "All points needs to have only 2 keys: timesteps, agents"
        # assert all([len(timestep_values) == 2 for timestep_values in values]), "Only 2 player setting is currently supported"
        assert len(schedule) >= 2, "Need at least 2 points to linearly interpolate schedule"
        assert schedule[0]["timestep"] == 0, "Schedule must start at timestep 0"
        assert all([t >=0 for t in timesteps]), "All timesteps in schedule must be non-negative"
        assert all([v >=0 and v <= 1 for timestep_values in values for agent_values in timestep_values
            for v in agent_values]), "All values in schedule must be between 0 and 1"
        assert all([sum(agent_values) == 1 for timestep_values in values for agent_values in timestep_values]), \
            "All probabilities needs to sum to 1"
        assert sorted(timesteps) == timesteps, "Timesteps must be in increasing order in schedule"

        # To ensure we flatline after passing last timestep
        if (schedule[-1]["timestep"] < float('inf')):
            schedule.append({"timestep": float('inf'), "agents":schedule[-1]["agents"]})
        
    def _setup_observation_space(self, observation_spaces=None):
        if observation_spaces is None:
            observation_spaces = self.observation_spaces

        observation_spaces = copy.deepcopy(observation_spaces)
        for agent_name, obs_space in OvercookedMultiAgent.default_observation_spaces.items():
            if agent_name not in observation_spaces:
                observation_spaces[agent_name] = obs_space

        self.observation_spaces = {agent_name: get_gym_space(obs_space, env=self.base_env)
            for agent_name, obs_space in observation_spaces.items()}

    def _setup_featurize_fns(self, featurize_fns=None):
        if featurize_fns is None:
            featurize_fns = self.featurize_fns
        featurize_fns = copy.deepcopy(featurize_fns)
        for agent_name, fn in OvercookedMultiAgent.default_featurize_fns.items():
            if agent_name not in featurize_fns:
                featurize_fns[agent_name] = fn
        self.featurize_fns = {agent_name: get_encoding_function(fn, env=self.base_env)
            for agent_name, fn in featurize_fns.items()}

    def _get_featurize_fn(self, agent_id):
        agent_name = OvercookedMultiAgent.agent_id_to_agent_name(agent_id)
        fn = self.featurize_fns.get(agent_name)
        if fn is None:
            raise ValueError(f"Unsupported agent {agent_id} of type {agent_name}")
        else:
            return fn

    def _featurize_ob_for_agent(self, state, agent_id):
        result = self._get_featurize_fn(agent_id)(state)
        agent_idx = OvercookedMultiAgent.agent_id_to_agent_idx(agent_id)
        if type(result) is dict:
            return {k: v[agent_idx] for k,v in result.items()}
        else:
            return result[agent_idx]
        
    def _get_featurized_obs(self, state):
        return {agent_id: self._featurize_ob_for_agent(state, agent_id)
            for agent_id in self.current_ml_agents_ids}

    def _create_agents_ids(self, agent_names):
        # Adds agent_idx to end of agent names, this ensures agent names are unique and lets get agent idx from its name
        return ["%s_%i" % (agent_name, i) for i, agent_name in enumerate(agent_names)]

    def _populate_agents(self):
        agents = [random.choices(list(player_probas.keys()), weights=player_probas.values(), k=1)[0]
            for player_probas in self.agents_probas]
        if self.shuffle_agents:
            np.random.shuffle(agents)
        
        return self._create_agents_ids(agents)

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
        action_dict: {agent_id: agent_action}
        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """

        joint_action = [None for agent in self.current_agents_ids]
        # set actions from ml agents
        for agent_id, a in action_dict.items():
            assert self.action_space.contains(a), "%r (%s) invalid"%(a, type(a))
            joint_action[OvercookedMultiAgent.agent_id_to_agent_idx(agent_id)] = Action.INDEX_TO_ACTION[a]
        # set actions from non ml agents (seen by rllib as just part of the env)
        for agent_id in self.current_non_ml_agents_ids:
            joint_action[OvercookedMultiAgent.agent_id_to_agent_idx(agent_id)] = \
                self.non_ml_agents_objects[agent_id].action(self.base_env.state)[0]

        if self.use_phi:
            next_state, sparse_reward, done, info = self.base_env.step(joint_action, display_phi=True)
            potential = info['phi_s_prime'] - info['phi_s']
            dense_reward = tuple(potential for agent_id in self.current_agents_ids)
        else:
            next_state, sparse_reward, done, info = self.base_env.step(joint_action, display_phi=False)
            dense_reward = info["shaped_r_by_agent"]

        featurized_obs = self._get_featurized_obs(next_state)
        rewards = { agent_id: sparse_reward + 
        self.reward_shaping_factor * dense_reward[OvercookedMultiAgent.agent_id_to_agent_idx(agent_id)]
            for agent_id in self.current_ml_agents_ids}
        dones = { agent_id: done for agent_id in self.current_ml_agents_ids + ["__all__"]}
        infos = { agent_id: info for agent_id in self.current_ml_agents_ids}
        return featurized_obs, rewards, dones, infos

    def reset(self, regen_mdp=True):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to 
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset(regen_mdp)
        self.current_agents_ids = self._populate_agents()
        self.current_ml_agents_ids = [agent_id for agent_id in self.current_agents_ids
            if OvercookedMultiAgent.is_ml_agent(agent_id)]
        assert len(self.current_ml_agents_ids) > 0, "needs to have at least one ml based agent, \
             use normal overcooked_ai functionality instead to run 2 non_ml_based agents"
        self.current_non_ml_agents_ids = [agent_id for agent_id in self.current_agents_ids
            if not OvercookedMultiAgent.is_ml_agent(agent_id)]
        self.non_ml_agents_objects = self.create_non_ml_agents(self.current_non_ml_agents_ids)
        return self._get_featurized_obs(self.base_env.state)

    def create_non_ml_agents(self, agents_ids):
        non_ml_agents_objects = {}
        for agent_id in agents_ids:
            agent_name = OvercookedMultiAgent.agent_id_to_agent_name(agent_id)
            agent_obj = OvercookedMultiAgent.create_non_ml_agent(agent_name,
                self.non_ml_agents_params, self.base_env)
            agent_obj.set_agent_index(OvercookedMultiAgent.agent_id_to_agent_idx(agent_id))
            non_ml_agents_objects[agent_id] = agent_obj
        return non_ml_agents_objects
    
    @staticmethod
    def create_non_ml_agent(agent_name, non_ml_agents_params, base_env):
        # static method so it can be used outside of the object in rllib eval
        def fill_init_kwargs(kwargs, kwargs_variables):
            kwargs = copy.deepcopy(kwargs)
            
            kwargs_variables = copy.deepcopy(kwargs_variables)
            for k, v_name in kwargs_variables.items():
                filled_v = get_overcooked_obj_attr(v_name, env=base_env)
                if filled_v == v_name:
                    raise ValueError("Unsupported init_kwarg {0}".format(v))
                else:
                    kwargs[k] = filled_v
            return kwargs
        assert not OvercookedMultiAgent.is_ml_agent(agent_name)
        agent_config = non_ml_agents_params[agent_name]["config"]
        agent_cls = agent_config["agent_cls"]
        init_kwargs = fill_init_kwargs(agent_config.get("agent_init_kwargs", {}),
            agent_config.get("agent_init_kwargs_variables", {}))
        return agent_cls(**init_kwargs)


    def anneal_reward_shaping_factor(self, timesteps):
        """
        Set the current reward shaping factor such that we anneal linearly until self.reward_shaping_horizon
        timesteps, given that we are currently at timestep "timesteps"
        """
        self.reward_shaping_factor = self._anneal(self._initial_reward_shaping_factor, 
            timesteps, self.reward_shaping_horizon)


    def anneal_agents_schedule(self, timesteps):
        """
        Set the agents_probas that we anneal linearly based on self.agents_schedule
        """
        p_0, p_1 = self._find_agents_schedule_points_around_timestep(timesteps)
        start_t = p_0["timestep"]
        end_t = p_1["timestep"]

        self.agents_probas = [{ k: self._anneal(p_0["agents"][i].get(k, 0), timesteps, end_t, p_1["agents"][i].get(k, 0), start_t) 
            for k in OvercookedMultiAgent.agents_from_schedule([p_0, p_1]) }
            for i in range(len(p_0["agents"]))]

    @staticmethod
    def agents_from_schedule(schedule):
        return list(set([k for p in schedule
            for player_probas in p["agents"]
            for k in player_probas.keys()]))

    def _find_agents_schedule_points_around_timestep(self, timestep):
        p_0 = self.agents_schedule[0]
        p_1 = self.agents_schedule[1]
        i = 2
        while timestep > p_1["timestep"] and i < len(self.agents_schedule):
            p_0 = p_1
            p_1 = self.agents_schedule[i]
            i += 1
        
        return p_0, p_1
        
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
        # Get rllib.OvercookedMultiAgentEnv reference from rllib wrapper
        env = base_env.get_unwrapped()[0]
        # Both agents share the same info so it doesn't matter whose we use, just use 0th agent's
        info_dict = episode.last_info_for(env.current_ml_agents_ids[0])

        ep_info = info_dict["episode"]
        game_stats = ep_info["ep_game_stats"]

        # List of episode stats we'd like to collect by agent
        stats_to_collect = EVENT_TYPES

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
                lambda env: env.anneal_agents_schedule(timestep)))

    def on_postprocess_trajectory(self, worker, episode, agent_id, policy_id,
            policies, postprocessed_batch, original_batches, **kwargs):
        pass


def get_rllib_eval_function(eval_params, eval_mdp_params, env_params, outer_shape, featurize_fns_map, shuffle=True):
    """
    Used to "curry" rllib evaluation function by wrapping additional parameters needed in a local scope, and returning a
    function with rllib custom_evaluation_function compatible signature

    eval_params (dict): Contains 'num_games' (int), 'display' (bool), 'ep_length' (int) and 'agents' (list[str])
    eval_mdp_params (dict): Used to create underlying OvercookedMDP (see that class for configuration)
    env_params (dict): Used to create underlying OvercookedEnv (see that class for configuration)
    outer_shape (list): a list of 2 item specifying the outer shape of the evaluation layout
    featurize_fns_map (dict): Used to featurize/encode observations; maps from policy name to function or overcooked method
    shuffle(bool): If True then shuffles agent starting positions
    Returns:
        _evaluate (func): Runs an evaluation specified by the curried params, ignores the rllib parameter 'evaluation_workers'
    """

    def _evaluate(trainer, evaluation_workers):
        print("Computing rollout of current trained policy")
        assert len(eval_params["agents"]) == 2, "currently only evaluation for 2 agents is supported"
        # Randomize starting indices

        policies_names = copy.deepcopy(eval_params["agents"])
        if shuffle:
            np.random.shuffle(policies_names)
        # # Get the corresponding rllib policy objects for each policy string name or create Overcooked Agent object
        policies = []
        base_ae = get_base_ae(eval_mdp_params, env_params, outer_shape)
        base_env = base_ae.env
        for policy_name in policies_names:
            if OvercookedMultiAgent.is_ml_agent(policy_name):
                policies.append(trainer.get_policy(policy_name))
            else:
                policies.append(policy_name)
        ppo_featurization = get_encoding_function(featurize_fns_map["ppo"], env=base_env)
        featurize_fns = [ppo_featurization] * len(policies)
        if 'bc' in policies_names:
            bc_featurization = get_encoding_function(featurize_fns_map["bc"], env=base_env)
            for i, policy_name in enumerate(policies_names):
                if policy_name == 'bc':
                    featurize_fns[i] = bc_featurization
        
        # Compute the evaluation rollout. Note this doesn't use the rllib passed in evaluation_workers, so this 
        # computation all happens on the CPU. Could change this if evaluation becomes a bottleneck
        results = evaluate(eval_params, eval_mdp_params, outer_shape, policies, featurize_fns)

        # Log any metrics we care about for rllib tensorboard visualization
        metrics = {}
        metrics['average_sparse_reward'] = np.mean(results['ep_returns'])
        return metrics

    return _evaluate


def evaluate(eval_params, mdp_params, outer_shape, policies, featurize_fns):
    """
    Used to visualize rollouts of trained policies

    eval_params (dict): Contains configurations such as the rollout length, number of games, and whether to display rollouts
    mdp_params (dict): OvercookedMDP compatible configuration used to create environment used for evaluation
    outer_shape (list): a list of 2 item specifying the outer shape of the evaluation layout
    policies (list(rllib.Policy or str(non_ml_agent_name))): Policy instances used to map states to action logits for agents or non ml agent name
    featurize_fns(list(func)): Used to preprocess states for agents defaults to lossless_state_encoding if 'None';
        used only when policy inside policies param with_fns
    """
    assert len(policies) == len(featurize_fns), "featurize_fns needs to have same length as policies"
    evaluator = get_base_ae(mdp_params, {"horizon" : eval_params['ep_length'], "num_mdp":1, "mlam_params": eval_params.get("mlam_params")}, outer_shape)

    agents = []
    # Wrap rllib policies in overcooked agents to be compatible with Evaluator code
    for i, policy, featurize_fn in zip(range(len(policies)), policies, featurize_fns):
        if isinstance(policy, RllibPolicy):
            agent = RlLibAgent(policy, agent_index=i, 
                featurize_fn=featurize_fn or evaluator.env.lossless_state_encoding_mdp)
        else:
            agent = OvercookedMultiAgent.create_non_ml_agent(policy, eval_params["non_ml_agents_params"], evaluator.env)
            agent.set_agent_index(i)
        agents.append(agent)
   
    # Compute rollouts
    if 'store_dir' not in eval_params:
        eval_params['store_dir'] = None
    if 'display_phi' not in eval_params:
        eval_params['display_phi'] = False
    
    results = evaluator.evaluate_agent_pair(AgentPair(*agents),
                                            num_games=eval_params['num_games'],
                                            display=eval_params['display'],
                                            dir=eval_params['store_dir'],
                                            display_phi=eval_params['display_phi'],
                                            native_eval=True)
    return results


###########################
# rllib.Trainer functions #
###########################


def gen_trainer_from_params(params):
    # All ray environment set-up
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_webui=False, temp_dir=params['ray_params']['temp_dir'])
    register_env("overcooked_multi_agent", params['ray_params']['env_creator'])
    ModelCatalog.register_custom_model(params['ray_params']['custom_model_id'], params['ray_params']['custom_model_cls'])

    # Parse params
    training_params = params['training_params']
    environment_params = params['environment_params']
    evaluation_params = params['evaluation_params']
    multi_agent_params = params['environment_params']['multi_agent_params']
    agent_params = params["agent_params"] # only ml based agents

    env = OvercookedMultiAgent.from_config(environment_params)

    # Returns a properly formatted policy tuple to be passed into ppotrainer config
    def gen_policy(policy_type="ppo"):
        return (
            agent_params[policy_type].get("policy_cls"),
            env.observation_spaces[policy_type],
            env.action_space,
            agent_params[policy_type]["config"]
            )

    # Rllib compatible way of setting the directory we store agent checkpoints in
    logdir_prefix = "{0}_{1}_{2}".format(params["experiment_name"], params['training_params']['seed'], timestr)
    def custom_logger_creator(config):
        """Creates a Unified logger that stores results in <params['results_dir']>/<params["experiment_name"]>_<seed>_<timestamp>
        """
        results_dir = params['results_dir']
        if not os.path.exists(results_dir):
            try:
                os.makedirs(results_dir)
            except Exception as e:
                print("error creating custom logging dir. Falling back to default logdir {}".format(DEFAULT_RESULTS_DIR))
                results_dir = DEFAULT_RESULTS_DIR
        logdir = tempfile.mkdtemp(
            prefix=logdir_prefix, dir=results_dir)
        logger = UnifiedLogger(config, logdir, loggers=None)
        return logger

    if "outer_shape" not in environment_params:
        environment_params["outer_shape"] = None

    if "mdp_params" in environment_params:
        environment_params["eval_mdp_params"] = environment_params["mdp_params"]
    
    # Create rllib compatible multi-agent config based on params
    multi_agent_config = {}

    if multi_agent_params.get('bc_schedule'):
        agents_schedule = OvercookedMultiAgent.bc_schedule_to_agents_schedule(multi_agent_params['bc_schedule'])
    else:
        agents_schedule = multi_agent_params['agents_schedule']
    all_policies = OvercookedMultiAgent.agents_from_schedule(agents_schedule)
    ml_policies = [p for p in all_policies if OvercookedMultiAgent.is_ml_agent(p)]

    multi_agent_config['policies'] = { policy : gen_policy(policy) for policy in ml_policies }
    
    def select_policy(agent_id):
        return OvercookedMultiAgent.agent_id_to_agent_name(agent_id)

    multi_agent_config['policy_mapping_fn'] = select_policy
    multi_agent_config['policies_to_train'] = 'ppo'

    eval_function = get_rllib_eval_function(evaluation_params, environment_params['eval_mdp_params'],
        environment_params['env_params'], environment_params["outer_shape"], multi_agent_params["featurize_fns"], shuffle=multi_agent_params["shuffle_agents"],
        )

    trainer = PPOTrainer(env="overcooked_multi_agent", config={
        "multiagent": multi_agent_config,
        "callbacks" : TrainingCallbacks,
        "custom_eval_function" : eval_function,
        "env_config" : environment_params,
        "eager" : False,
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

def load_trainer(save_path):
    """
    Returns a ray compatible trainer object that was previously saved at `save_path` by a call to `save_trainer`
    Note that `save_path` is the full path to the checkpoint FILE, not the checkpoint directory
    """
    # Read in params used to create trainer
    config_path = os.path.join(os.path.dirname(save_path), "config.pkl")
    with open(config_path, "rb") as f:
        # We use dill (instead of pickle) here because we must deserialize functions
        config = dill.load(f)
    
    # Override this param to lower overhead in trainer creation
    config['training_params']['num_workers'] = 0

    # Get un-trained trainer object with proper config
    trainer = gen_trainer_from_params(config)

    # Load weights into dummy object
    trainer.restore(save_path)
    return trainer

def get_agent_from_trainer(trainer, policy_id="ppo", agent_index=0):
    policy = trainer.get_policy(policy_id)
    dummy_env = trainer.env_creator(trainer.config['env_config'])
    featurize_fn = dummy_env.featurize_fns[policy_id]
    agent = RlLibAgent(policy, agent_index, featurize_fn=featurize_fn)
    return agent

def get_agent_pair_from_trainer(trainer, policy_id_0='ppo', policy_id_1='ppo'):
    agent0 = get_agent_from_trainer(trainer, policy_id=policy_id_0)
    agent1 = get_agent_from_trainer(trainer, policy_id=policy_id_1)
    return AgentPair(agent0, agent1)


def load_agent_pair(save_path, policy_id_0='ppo', policy_id_1='ppo'):
    """
    Returns an Overcooked AgentPair object that has as player 0 and player 1 policies with 
    ID policy_id_0 and policy_id_1, respectively
    """
    trainer = load_trainer(save_path)
    return get_agent_pair_from_trainer(trainer, policy_id_0, policy_id_1)

def load_agent(save_path, policy_id='ppo', agent_index=0):
    """
    Returns an RllibAgent (compatible with the Overcooked Agent API) from the `save_path` to a previously
    serialized trainer object created with `save_trainer`

    The trainer can have multiple independent policies, so extract the one with ID `policy_id` to wrap in
    an RllibAgent

    Agent index indicates whether the agent is player zero or player one (or player n in the general case)
    as the featurization is not symmetric for both players
    """
    trainer = load_trainer(save_path)
    return get_agent_from_trainer(trainer, policy_id=policy_id, agent_index=agent_index)


