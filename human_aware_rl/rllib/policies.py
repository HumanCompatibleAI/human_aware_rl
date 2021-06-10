from ray.rllib.agents import trainer
from ray.rllib.policy import Policy as RllibPolicy
from ray.rllib.agents.ppo import PPOTFPolicy
import ray
import numpy as np

from human_aware_rl.rllib.rllib import load_trainer
from human_aware_rl.rllib.utils import softmax

class StaticPolicy(RllibPolicy):
    """
    Rllib policy class that has no weights to update. Overrides update functions to prevent rllib from breaking
    """

    def __init__(self, *args, **kwargs):
        super(StaticPolicy, self).__init__(*args, **kwargs)

    def get_weights(self):
        """
        No-op to keep rllib from breaking, won't be necessary in future rllib releases
        """
        pass

    def set_weights(self, weights):
        """
        No-op to keep rllib from breaking
        """
        pass


    def learn_on_batch(self, samples):
        """
        Static policy requires no learning
        """
        return {}

    def get_initial_state(self):
        """
        Returns the initial hidden and memory states for the model if it is recursive

        Note, this shadows the rllib.Model.get_initial_state function

        Also note, either this function or self.model.get_initial_state (if it exists) must be called at 
        start of an episode
        """
        return []

class UniformPolicy(StaticPolicy):

    def __init__(self, observation_space, action_space, config):
        super(UniformPolicy, self).__init__(observation_space, action_space, config)
        n = self.action_space.n
        self.logits = np.ones(n) * (1/n)

    def compute_actions(self, obs_batch, *args, **kwargs):
        N = len(obs_batch)
        actions = [self.action_space.sample() for _ in range(N)]
        infos = { "action_dist_inputs" : self.logits }
        return actions, [], infos

class ConstantPolicy(StaticPolicy):

    def __init__(self, observation_space, action_space, config):
        super(ConstantPolicy, self).__init__(observation_space, action_space, config)
        self.logits = config.get('logits', np.ones(self.action_space.n) / self.action_space.n)

    def compute_actions(self, obs_batch, *args, **kwargs):
        infos = { "action_dist_inputs" : self.logits }
        actions =  [np.random.choice(self.action_space.n, p=softmax(self.logits))]
        states = []
        return actions, states, infos

class EnsemblePolicy(StaticPolicy):
    """
    Meta-Policy composed of ensemble of previously trained policies
    """

    def __init__(self, observation_space, action_space, config):
        super(EnsemblePolicy, self).__init__(observation_space, action_space, config)
        initial_buff_size = 5
        self._curr_id_cnt = 0
        self._base_policies = [None] * initial_buff_size
        self._base_policy_loaders = [(None, {})] * initial_buff_size
        self._base_policy_metadata = [{}] * initial_buff_size
        self._loaded = [False] * initial_buff_size
        self.num_base_policies = 0
        self.max_policies_in_memory = config.get('max_policies_in_memory', 5)
        self._initial_policy = UniformPolicy(self.observation_space, self.action_space, {})
        self.curr_policy = None
        self.curr_policy_idx = -1
        self.add_base_policy(loader_fn=lambda : self._initial_policy)

    @property
    def num_loaded(self):
        return np.sum(np.array(self._loaded).astype(int))
    @property
    def base_policies(self):
        return [self.get_policy(i) for i in range(self.num_base_policies)]

    @staticmethod
    def default_load_fn(trainer_path, policy_id):
        return load_trainer(trainer_path).get_policy(policy_id)

    def get_rnd_policy(self):
        return self.get_policy(0)

    def get_policy(self, idx):
        if self._loaded[idx]:
            return self._base_policies[idx]
        
        if self.num_loaded == self.max_policies_in_memory:
            evicted = self.evict_base_policy()
            if not evicted:
                raise MemoryError("Failed to successfully evict policy to free space")
        
        return self.load_base_policy(idx)

    def add_base_policy(self, trainer_path=None, policy_id='ppo', loader_fn=None, loader_kwargs={}, metadata={}):
        assert trainer_path or loader_fn, "must either specify trainer path or provide custom load function"
        if not loader_fn:
            loader_fn = self.default_load_fn
            loader_kwargs = {"trainer_path" : trainer_path, "policy_id" : policy_id}
        if self.num_base_policies == len(self._loaded):
            self._expand()
        self._base_policy_loaders[self.num_base_policies] = (loader_fn, loader_kwargs)
        self._base_policy_metadata[self.num_base_policies] = metadata
        self._loaded[self.num_base_policies] = False
        self.num_base_policies += 1
        return True

    def load_base_policy(self, idx):
        if self._loaded[idx]:
            self._base_policies[idx]

        self._base_policies[idx] = self._load(idx)
        self._loaded[idx] = True
        return self._base_policies[idx]

    def evict_base_policy(self):
        candidates = list(np.argwhere(self._loaded[1:]).flatten())
        if self.curr_policy_idx in candidates:
            candidates.remove(self.curr_policy_idx)
        if not candidates:
            return False
        evict_idx = np.random.choice(candidates)
        self._loaded[evict_idx] = False
        self._base_policies[evict_idx] = None
        return True
    
    def compute_actions(self, obs_batch, *args, **kwargs):
        if not self.curr_policy:
            self.sample_policy()
        return self.curr_policy.compute_actions(obs_batch, *args, **kwargs)

    def sample_policy(self, *args, **kwargs):
        self.curr_policy_idx = np.random.randint(low=0, high=self.num_base_policies)
        self.curr_policy = self.get_policy(self.curr_policy_idx)
        return self.curr_policy
        
    def _load(self, idx):
        loader_fn, loader_kwargs = self._base_policy_loaders[idx]
        return loader_fn(**loader_kwargs)

    def _expand(self):
        N = len(self._loaded)
        self._loaded = self._loaded + [False] * N
        self._base_policies = self._base_policies + [None] * N
        self._base_policy_loaders = self._base_policy_loaders + [(None, {})] * N
        self._base_policy_metadata = self._base_policy_metadata + [{}] * N

