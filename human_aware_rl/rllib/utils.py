from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
import numpy as np
import inspect

def softmax(logits):
    e_x = np.exp(logits.T - np.max(logits))
    return (e_x / np.sum(e_x, axis=0)).T

def get_base_env(mdp_params, env_params):
    mdp_fn = lambda: OvercookedGridworld.from_layout_name(**mdp_params)
    base_env = OvercookedEnv(mdp_fn, **env_params)
    return base_env

def get_mlp(env):
    return MediumLevelPlanner.from_pickle_or_compute(env.mdp, NO_COUNTERS_PARAMS, force_compute=False)

# Returns the required arguments as inspect.Parameter objects in a list
def get_required_arguments(fn):
    required = []
    params = inspect.signature(fn).parameters.values()
    for param in params:
        if param.default == inspect.Parameter.empty and param.kind == param.POSITIONAL_OR_KEYWORD:
            required.append(param)
    return required

def iterable_equal(a, b):
    if hasattr(a, '__iter__') != hasattr(b, '__iter__'):
        return False
    if not hasattr(a, '__iter__'):
        return a == b

    if len(a) != len(b):
        return False

    for elem_a, elem_b in zip(a, b):
        if not iterable_equal(elem_a, elem_b):
            return False

    return True