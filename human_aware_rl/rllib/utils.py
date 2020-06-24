from overcooked_ai_py.agents.benchmarking import AgentEvaluator
import numpy as np
import inspect

def softmax(logits):
    e_x = np.exp(logits.T - np.max(logits))
    return (e_x / np.sum(e_x, axis=0)).T

def get_base_env(mdp_params, env_params):
    # modified to ensure backwards compatibility
    ae = AgentEvaluator(mdp_params_lst=[mdp_params], env_params=env_params)
    return ae.env_lst[0]

def get_base_env_lst(mdp_params_lst, env_params):
    """
    mdp_params_lst (list): a list of mdp_params
    """
    ae = AgentEvaluator(mdp_params_lst=mdp_params_lst, env_params=env_params)
    return ae.env_lst

def get_mlp(mdp_params, env_params):
    # modified to ensure backwards compatibility
    ae = AgentEvaluator(mdp_params_lst=[mdp_params], env_params=env_params)
    return ae.mlp_lst[0]

def get_mlp_lst(mdp_params_lst, env_params):
    """
    mdp_params_lst (list): a list of mdp_params
    """
    ae = AgentEvaluator(mdp_params_lst=mdp_params_lst, env_params=env_params)
    return ae.mlp_lst

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