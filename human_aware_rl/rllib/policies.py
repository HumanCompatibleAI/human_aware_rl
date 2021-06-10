from human_aware_rl.rllib.utils import softmax
from ray.rllib.policy import Policy as RllibPolicy
import numpy as np
import ray

"""
Home of all 'base' user-defined rllib policies. These are all subclasses of 'rllib.policy.Policy' class

Notably, the `StaticPolicy` class allows for wrapping pre-trained models (BC, OPT, etc) in an rllib-compatible
API for GPU accelerated training
"""

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
        actions = np.array([self.action_space.sample() for _ in range(N)])
        logits = np.array([self.logits for _ in range(N)])
        infos = { "action_dist_inputs" : logits }
        return actions, [], infos

class ConstantPolicy(StaticPolicy):

    def __init__(self, observation_space, action_space, config):
        super(ConstantPolicy, self).__init__(observation_space, action_space, config)
        self.logits = config.get('logits', np.ones(self.action_space.n) / self.action_space.n)
        self.stochastic = config.get('stochastic', True)

    def compute_actions(self, obs_batch, *args, **kwargs):
        N = len(obs_batch)
        infos = { "action_dist_inputs" : self.logits }
        if self.stochastic:
            actions =  np.array([np.random.choice(self.action_space.n, p=softmax(self.logits)) for _ in range(N)])
        else:
            actions = np.array([np.argmax(self.logits) for _ in range(N)])
        states = []
        return actions, states, infos

