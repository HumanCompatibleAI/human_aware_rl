import gym
import tqdm, copy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from stable_baselines import GAIL
from stable_baselines.gail import ExpertDataset

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, DEFAULT_ENV_PARAMS
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import AgentFromPolicy, AgentPair
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
from overcooked_ai_py.utils import save_pickle, load_pickle

from human_aware_rl.baselines_utils import create_dir_if_not_exists
from human_aware_rl.human.process_dataframes import save_npz_file, get_trajs_from_data

BC_SAVE_DIR = "data/bc_runs/"

DEFAULT_DATA_PARAMS = {
    "train_mdps": ["simple"],
    "ordered_trajs": True,
    "human_ai_trajs": False,
    "data_path": "data/human/anonymized/clean_train_trials.pkl"
}

DEFAULT_BC_PARAMS = {
    "data_params": DEFAULT_DATA_PARAMS,
    "mdp_params": {}, # Nothing to overwrite defaults
    "env_params": DEFAULT_ENV_PARAMS,
    "mdp_fn_params": {}
}

def init_gym_env(bc_params):
    env_setup_params = copy.deepcopy(bc_params)
    del env_setup_params["data_params"] # Not necessary for setting up env
    mdp = OvercookedGridworld.from_layout_name(**bc_params["mdp_params"])
    env = OvercookedEnv(mdp, **bc_params["env_params"])
    gym_env = gym.make("Overcooked-v0")
    
    mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)
    gym_env.custom_init(env, featurize_fn=lambda x: mdp.featurize_state(x, mlp))
    return gym_env

def train_bc_agent(model_save_dir, bc_params, num_epochs=1000, lr=1e-4, adam_eps=1e-8):
    # Extract necessary expert data and save in right format
    expert_trajs = get_trajs_from_data(**bc_params["data_params"])
    
    # Load the expert dataset
    save_npz_file(expert_trajs, "temp.npz")
    dataset = ExpertDataset(expert_path="temp.npz", verbose=1, train_fraction=0.85)
    assert dataset is not None
    assert dataset.train_loader is not None
    return bc_from_dataset_and_params(dataset, bc_params, model_save_dir, num_epochs, lr, adam_eps)

def bc_from_dataset_and_params(dataset, bc_params, model_save_dir, num_epochs, lr, adam_eps):
    # Setup env
    gym_env = init_gym_env(bc_params)

    # Train and save model
    create_dir_if_not_exists(BC_SAVE_DIR + model_save_dir)

    model = GAIL("MlpPolicy", gym_env, dataset, verbose=1)
    model.pretrain(dataset, n_epochs=num_epochs, learning_rate=lr, adam_epsilon=adam_eps, save_dir=BC_SAVE_DIR + model_save_dir)

    save_bc_model(model_save_dir, model, bc_params)
    return model

def save_bc_model(model_save_dir, model, bc_params):
    print("Saved BC model at", BC_SAVE_DIR + model_save_dir)
    print(model_save_dir)
    model.save(BC_SAVE_DIR + model_save_dir + "model")
    bc_metadata = {
        "bc_params": bc_params,
        "train_info": model.bc_info
    }
    save_pickle(bc_metadata, BC_SAVE_DIR + model_save_dir + "bc_metadata")

def get_bc_agent_from_saved(model_name, no_waits=False):
    model, bc_params = load_bc_model_from_path(model_name)
    return get_bc_agent_from_model(model, bc_params, no_waits), bc_params

def get_bc_agent_from_model(model, bc_params, no_waits=False):
    mdp = OvercookedGridworld.from_layout_name(**bc_params["mdp_params"])
    mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)
    
    def encoded_state_policy(observations, include_waits=True, stochastic=False):
        action_probs_n = model.action_probability(observations)

        if not include_waits:
            action_probs = ImitationAgentFromPolicy.remove_indices_and_renormalize(action_probs_n, [Action.ACTION_TO_INDEX[Direction.STAY]])
        
        if stochastic:
            return [np.random.choice(len(action_probs[i]), p=action_probs[i]) for i in range(len(action_probs))]
        return action_probs_n

    def state_policy(mdp_states, agent_indices, include_waits, stochastic=False):
        # encode_fn = lambda s: mdp.preprocess_observation(s)
        encode_fn = lambda s: mdp.featurize_state(s, mlp)

        obs = []
        for agent_idx, s in zip(agent_indices, mdp_states):
            ob = encode_fn(s)[agent_idx]
            obs.append(ob)
        obs = np.array(obs)
        action_probs = encoded_state_policy(obs, include_waits, stochastic)
        return action_probs

    return ImitationAgentFromPolicy(state_policy, encoded_state_policy, no_waits=no_waits, mlp=mlp)

def eval_with_benchmarking_from_model(n_games, model, bc_params, no_waits, display=False):
    bc_params = copy.deepcopy(bc_params)
    a0 = get_bc_agent_from_model(model, bc_params, no_waits)
    a1 = get_bc_agent_from_model(model, bc_params, no_waits)
    del bc_params["data_params"], bc_params["mdp_fn_params"]
    a_eval = AgentEvaluator(**bc_params)
    ap = AgentPair(a0, a1)
    trajectories = a_eval.evaluate_agent_pair(ap, num_games=n_games, display=display)
    return trajectories

def eval_with_benchmarking_from_saved(n_games, model_name, no_waits=False, display=False):
    model, bc_params = load_bc_model_from_path(model_name)
    return eval_with_benchmarking_from_model(n_games, model, bc_params, no_waits, display=display)

def load_bc_model_from_path(model_name):
    # NOTE: The lowest loss and highest accuracy models 
    # were also saved, can be found in the same dir with
    # special suffixes.
    bc_metadata = load_pickle(BC_SAVE_DIR + model_name + "/bc_metadata")
    bc_params = bc_metadata["bc_params"]
    model = GAIL.load(BC_SAVE_DIR + model_name + "/model")
    return model, bc_params

def plot_bc_run(run_info, num_epochs):
    xs = range(0, num_epochs, max(int(num_epochs/10), 1))
    plt.plot(xs, run_info['train_losses'], label="train loss")
    plt.plot(xs, run_info['val_losses'], label="val loss")
    plt.plot(xs, run_info['val_accuracies'], label="val accuracy")
    plt.legend()
    plt.show()


class ImitationAgentFromPolicy(AgentFromPolicy):
    """Behavior cloning agent interface"""

    def __init__(self, state_policy, direct_policy, mlp=None, stochastic=True, no_waits=False, stuck_time=3):
        super().__init__(state_policy, direct_policy)
        # How many turns in same position to be considered 'stuck'
        self.stuck_time = stuck_time
        self.history_length = stuck_time + 1
        self.stochastic = stochastic
        self.action_probs = False
        self.no_waits = no_waits
        self.will_unblock_if_stuck = False if stuck_time == 0 else True
        self.mlp = mlp
        self.reset()

    def action(self, state):
        return self.actions(state)

    def actions(self, states, agent_indices=None):
        """
        The standard action function call, that takes in a Overcooked state
        and returns the corresponding action.

        Requires having set self.agent_index and self.mdp
        """
        if agent_indices is None:
            assert isinstance(states, OvercookedState)
            # Chose to overwrite agent index, set it as current agent index. Useful for Vectorized environments
            agent_indices = [self.agent_index]
            states = [states]
        
        # Actually now state is a list of states
        assert len(states) > 0

        all_actions = self.multi_action(states, agent_indices)

        if len(agent_indices) > 1:
            return all_actions
        return all_actions[0]

    def multi_action(self, states, agent_indices):
        try:
            action_probs_n = list(self.state_policy(states, agent_indices, not self.no_waits))
        except AttributeError:
            raise AttributeError("Need to set the agent_index or mdp of the Agent before using it")

        all_actions = []
        for parallel_agent_idx, curr_agent_action_probs in enumerate(action_probs_n):
            curr_agent_idx = agent_indices[parallel_agent_idx]
            curr_agent_state = states[parallel_agent_idx]
            self.set_agent_index(curr_agent_idx)
            
            # Removing wait action
            if self.no_waits:
                curr_agent_action_probs = self.remove_indices_and_renormalize(curr_agent_action_probs, [Action.ACTION_TO_INDEX[Direction.STAY]])

            if self.will_unblock_if_stuck:
                curr_agent_action_probs = self.unblock_if_stuck(curr_agent_state, curr_agent_action_probs)

            if self.stochastic:
                action_idx = np.random.choice(len(curr_agent_action_probs), p=curr_agent_action_probs)
            else:
                action_idx = np.argmax(curr_agent_action_probs)
            curr_agent_action = Action.INDEX_TO_ACTION[action_idx]
            self.add_to_history(curr_agent_state, curr_agent_action)

            if self.action_probs:
                all_actions.append(curr_agent_action_probs)
            else:
                all_actions.append(curr_agent_action)
        return all_actions

    def unblock_if_stuck(self, state, action_probs):
        """Get final action for a single state, given the action probabilities
        returned by the model and the current agent index.
        NOTE: works under the invariance assumption that self.agent_idx is already set
        correctly for the specific parallel agent we are computing unstuck for"""
        stuck, last_actions = self.is_stuck(state)
        if stuck:
            assert any([a not in last_actions for a in Direction.ALL_DIRECTIONS]), last_actions
            last_action_idxes = [Action.ACTION_TO_INDEX[a] for a in last_actions]
            action_probs = self.remove_indices_and_renormalize(action_probs, last_action_idxes)
        return action_probs

    def is_stuck(self, state):
        if None in self.history[self.agent_index]:
            return False, []
        
        last_states = [s_a[0] for s_a in self.history[self.agent_index][-self.stuck_time:]]
        last_actions = [s_a[1] for s_a in self.history[self.agent_index][-self.stuck_time:]]
        player_states = [s.players[self.agent_index] for s in last_states]
        pos_and_ors = [p.pos_and_or for p in player_states] + [state.players[self.agent_index].pos_and_or]
        if self.checkEqual(pos_and_ors):
            return True, last_actions
        return False, []

    @staticmethod
    def remove_indices_and_renormalize(probs, indices):
        if len(np.array(probs).shape) > 1:
            probs = np.array(probs)
            for row_idx, row in enumerate(indices):
                for idx in indices:
                    probs[row_idx][idx] = 0
            norm_probs =  probs.T / np.sum(probs, axis=1)
            return norm_probs.T
        else:
            for idx in indices:
                probs[idx] = 0
            return probs / sum(probs)

    def checkEqual(self, iterator):
        first_pos_and_or = iterator[0]
        for curr_pos_and_or in iterator:
            if curr_pos_and_or[0] != first_pos_and_or[0] or curr_pos_and_or[1] != first_pos_and_or[1]:
                return False
        return True

    def add_to_history(self, state, action):
        assert len(self.history[self.agent_index]) == self.history_length
        self.history[self.agent_index].append((state, action))
        self.history[self.agent_index] = self.history[self.agent_index][1:]

    def reset(self):
        # Matrix of histories, where each index/row corresponds to a specific agent
        self.history = defaultdict(lambda: [None] * self.history_length)


##########
# EXTRAS #
##########

def stable_baselines_predict_fn(model, observation):
    a_probs = model.action_probability(observation)
    action_idx = np.random.choice(len(a_probs), p=a_probs)
    return action_idx

def eval_with_standard_baselines(n_games, model_name, display=False):
    """Method to evaluate agent performance with stable-baselines infrastructure,
    just to make sure everything is compatible and integrating correctly."""
    bc_metadata = load_pickle(BC_SAVE_DIR + model_name + "/bc_metadata")
    bc_params = bc_metadata["bc_params"]
    model = GAIL.load(BC_SAVE_DIR + model_name + "/model")

    gym_env = init_gym_env(bc_params)

    tot_rew = 0
    for i in tqdm.trange(n_games):
        obs, _ = gym_env.reset()
        done = False
        while not done:
            ob0, ob1 = obs
            a0 = stable_baselines_predict_fn(model, ob0)
            a1 = stable_baselines_predict_fn(model, ob1)
            joint_action = (a0, a1)
            (obs, _), rewards, done, info = gym_env.step(joint_action)
            tot_rew += rewards

    print("avg reward", tot_rew / n_games)
    return tot_rew / n_games

def symmetric_bc(model_savename, bc_params, num_epochs=1000, lr=1e-4, adam_eps=1e-8):
    """DEPRECATED: Trains two BC models from the same data. Splits data 50-50 and uses each subset as training data for
    one model and validation for the other."""
    expert_trajs = get_trajs_from_data(bc_params["data_params"])

    save_npz_file(expert_trajs, "temp")
    train_dataset = ExpertDataset(expert_path="temp", verbose=1, train_fraction=0.5)
    train_indices = train_dataset.train_loader.original_indices
    val_indices = train_dataset.val_loader.original_indices

    # Train BC model
    train_model_save_dir = model_savename + "_train/"
    bc_from_dataset_and_params(train_dataset, bc_params, train_model_save_dir, num_epochs, lr, adam_eps)

    # Switching testing and validation datasets (somewhat hacky)
    indices_split = (val_indices, train_indices)
    test_dataset = ExpertDataset(expert_path="temp", verbose=1, train_fraction=0.5, indices_split=indices_split)

    # Test BC model
    test_model_save_dir = model_savename + "_test/"
    bc_from_dataset_and_params(test_dataset, bc_params, test_model_save_dir, num_epochs, lr, adam_eps)
