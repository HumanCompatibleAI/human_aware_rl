from human_aware_rl.rllib.policies import StaticPolicy, UniformPolicy
from human_aware_rl.rllib.rllib import load_trainer
from ray.rllib.models.modelv2 import restore_original_dimensions
import numpy as np

"""
Home of all higher-order user-defined Rllib policies. All classes here subclass `StaticPolicy` (see human_aware_rl.rllib.policies)

Notably, these policies all contain multiple sub-instances of rllib policies, and stitch together these base policies to form a meta-policy.
See class-specific docstrings for more info.
"""

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

class AbstractOffDistrubutionPolicy(StaticPolicy):

    """
    Abstract class for a OOD policy. At a high level, this can be viewed as a meta agent that
    has both an "on distribution" policy, and an "off distriubtion" policy that 
    is invoked whenever `self._of_distribution` is true

    Methods:
        compute_actions:            overrides base RllibPolicy's compute_actions
        
    Abstract Methods
        _on_distribution_init:      Used to initialize on distribution policy. Must return a RllibPolicy type
        _off_distribution_init:     Used to initialize off distribution policy. Must return a RllibPolicy type
        _off_distribution:           Invoked to determine whether state is on/off distribution. Returns boolean mask

    Instances:
        observation_space (gym.spaces.Dict):    Information about observation space. Contains keys "off_dist" and "on_dist"
        action_space (gym.space.Discrete):      Information about action space
        on_distribution_policy (RllibPolicy):   Queried for on distribution actions
        off_distribution_policy (RllibPolicy):  Queried for off-distribution actions
    """

    def __init__(self, observation_space, action_space, config):
        """
        RLLib compatible constructor for initializing a OOD model

        observation_space (gym.Space)           Shape of the featurized observations
        action_space (gym.space)                Shape of the action space (len(Action.All_ACTIONS),)
        config (dict)                           Dictionary of relavant policy params
            - on_dist_config (dict):                Parameters passed into the "on distribution" policy
            - off_dist_config (dict):               Parameters passed into the "off distrubtion" policy
        """
        super(AbstractOffDistrubutionPolicy, self).__init__(observation_space, action_space, config)
        self.on_distrubtion_policy = self._on_dist_init(config['on_dist_config'])
        self.off_distrubtion_policy = self._off_dist_init(config['off_dist_config'])
        
    def _on_dist_init(self, config):
        """
        Initializes on-distribution policy. Must return an RllibPolicy instance
        """
        raise NotImplementedError("Must subclass and override this method")

    def _off_dist_init(self, config):
        """
        Initializes off-distribution policy. Must return an RllibPolicy instance
        """
        raise NotImplementedError("Must subclass and override this method")


    def _off_distribution(self, obs_batch, *args, **kwargs):
        """
        Determine whether given states are on/off distribution

        Arguments:
            - obs_batch (np.array) (N, *obs_space.shape): Array of observations. 
                    Must also be able to handle single, non-batched observation

        Returns
            - mask (np.array) (N,): Array of booleans, True if off distribution, false otherwise
        """
        raise NotImplementedError("Must subclass and override this method")

    def parse_observations(self, obs_batch):
        # Parse out the on/off-distribution actions
        if type(obs_batch) == dict:
            # We received non-batched, non-flattened dictionary observation
            off_dist_obs, on_dist_obs = obs_batch['off_dist'], obs_batch['on_dist']
        if hasattr(obs_batch, '__iter__') and type(obs_batch[0]) == dict:
            # We received batched, non-flattened dictionary observation
            off_dist_obs, on_dist_obs = [obs['off_dist'] for obs in obs_batch], [obs['on_dist'] for obs in obs_batch]
        else:
            # We received batched/unbatched flattened dictionary observation (this is done by default on the Rllib backened
            # with no public API to disable...). We thus "unpack" the observation to restore it to its original dictionary config
            restored_batch = restore_original_dimensions(obs_batch, self.observation_space, tensorlib=np)
            off_dist_obs, on_dist_obs = restored_batch['off_dist'], restored_batch['on_dist']

        return off_dist_obs, on_dist_obs

    def compute_actions(self, obs_batch, *args, **kwargs):
        off_dist_mask = self._off_distribution(obs_batch, *args, **kwargs)
        return self._compute_actions(obs_batch, off_dist_mask)

    def _compute_actions(self, obs_batch, off_dist_mask):
        """
        Note: Both off- and on-distribution policies are queried for every timestep. There is no lazy computation
        """
        off_dist_obs, on_dist_obs = self.parse_observations(obs_batch)

        # On distrubion forward
        on_dist_actions, on_dist_logits = self._on_dist_compute_actions(on_dist_obs)

        # Off distribution forward
        off_dist_actions, off_dist_logits = self._off_dist_compute_actions(off_dist_obs)

        # Batched ternary switch based on previously computed masks
        actions, logits = np.where(off_dist_mask, off_dist_actions, on_dist_actions), np.where(off_dist_mask, off_dist_logits.T, on_dist_logits.T).T

        return actions, [], { "action_dist_inputs" : logits, "off_dist_mask" : off_dist_mask }

    def _on_dist_compute_actions(self, on_dist_obs_batch, **kwargs):
        actions, _, infos = self.on_distrubtion_policy.compute_actions(on_dist_obs_batch)
        logits = infos['action_dist_inputs']
        return actions, logits

    def _off_dist_compute_actions(self, off_dist_obs_batch, **kwargs):
        actions, _, infos = self.off_distrubtion_policy.compute_actions(off_dist_obs_batch)
        logits = infos['action_dist_inputs']
        return actions, logits
  