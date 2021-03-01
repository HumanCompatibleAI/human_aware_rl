from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.utils import only_valid_named_args
import gym
import numpy as np
import inspect

def softmax(logits):
    e_x = np.exp(logits.T - np.max(logits))
    return (e_x / np.sum(e_x, axis=0)).T

def sigmoid(logits):
    return 1 / (1 + np.exp(-logits))

def get_base_env(mdp_params, env_params, outer_shape=None, mdp_params_schedule_fn=None):
    ae = get_base_ae(mdp_params, env_params, outer_shape, mdp_params_schedule_fn)
    return ae.env

def get_base_mlam(mdp_params, env_params, outer_shape=None, mdp_params_schedule_fn=None):
    ae = get_base_ae(mdp_params, env_params, outer_shape, mdp_params_schedule_fn)
    return ae.mlam

def get_base_ae(mdp_params, env_params, outer_shape=None, mdp_params_schedule_fn=None):
    """
    mdp_params: one set of fixed mdp parameter used by the environment
    env_params: env parameters (horizon, etc)
    outer_shape: outer shape of the environment
    mdp_params_schedule_fn: the schedule for varying mdp params

    return: the base agent evaluator
    """
    assert mdp_params == None or mdp_params_schedule_fn == None, "either of the two has to be null"
    if type(mdp_params) == dict and "layout_name" in mdp_params:
        ae = AgentEvaluator.from_layout_name(mdp_params=mdp_params, env_params=env_params)
    elif 'num_mdp' in env_params:
        if np.isinf(env_params['num_mdp']):
            ae = AgentEvaluator.from_mdp_params_infinite(mdp_params=mdp_params, env_params=env_params,
                                                         outer_shape=outer_shape, mdp_params_schedule_fn=mdp_params_schedule_fn)
        else:
            ae = AgentEvaluator.from_mdp_params_finite(mdp_params=mdp_params, env_params=env_params,
                                                         outer_shape=outer_shape, mdp_params_schedule_fn=mdp_params_schedule_fn)
    else:
        # should not reach this case
        raise NotImplementedError()
    return ae

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

    if not hasattr(a, '__iter__') or isinstance(a, str):
        return a == b

    if len(a) != len(b):
        return False

    if isinstance(a, dict):
        a = a.items()
    if isinstance(b, dict):
        b = b.items()
    
    for elem_a, elem_b in zip(a, b):
        if not iterable_equal(elem_a, elem_b):
            return False

    return True


def get_overcooked_obj_attr(attr, env=None, mdp=None, env_params=None, mdp_params=None):
    """
    returns overcooked object attribute based on its name; used mostly to get state processing (encoding) functions and gym spaces
    when receives string parse it to get attribute; format is "env"/"mdp" + "." + method name i.e "env.lossless_state_encoding_mdp"
    also support dicts (where replaces strings in values with object attributes)
    when receives method/function returns original method; this obviously does not work this way if attr is str/dict
    """
    attr_type = type(attr)
    if attr_type is str:
        name = attr
        [obj_name, attr_name] = name.split(".")
        if obj_name == "mdp":
            if not mdp:
                if env:
                    mdp = env.mdp
                else:
                    mdp = OvercookedGridworld(**mdp_params)
            attr = getattr(mdp, attr_name)
        elif obj_name == "env":
            if not env:
                if not mdp:
                    mdp = OvercookedGridworld(**mdp_params)
                env_params = only_valid_named_args(env_params, OvercookedEnv.from_mdp) 
                env = OvercookedEnv.from_mdp(mdp, **env_params)
            attr = getattr(env, attr_name)
        # not tested or used anywhere yet
        # elif obj_name in kwargs:
        #     attr = getattr(kwargs[obj_name], attr_name)
        else:
            raise ValueError("Unsupported obj attr string "+name)
    elif attr_type is dict:
        attr = {k: get_overcooked_obj_attr(v, env=env, mdp=mdp, env_params=env_params, 
            mdp_params=mdp_params) for k, v in attr.items()}
    # not tested or used anywhere yet
    # elif attr_type in [list, tuple]:
    #     attr = attr_type(get_overcooked_obj_attr(elem, env=env, mdp=mdp, env_params=env_params, 
    #         mdp_params=mdp_params) for elem in attr)
    return attr

def get_encoding_function(function, env=None, mdp=None, env_params=None, mdp_params=None):
    """
    returns processing function from overcooked object based on supplied name
    when receives string parse it to get object attribute; format is "env"/"mdp" + "." + method name i.e "env.lossless_state_encoding_mdp"
    also support dicts
    when receives method/function returns original method; this obviously does not work this way if attr is str/dict
    """
    function = get_overcooked_obj_attr(function, env=env, mdp=mdp, env_params=env_params, 
            mdp_params=mdp_params)

    if type(function) is dict:
        fn_dict = function
        def result_fn(*args, **kwargs):
            return {k: f(*args, **kwargs) for k, f in fn_dict.items()}
        function = result_fn
    # not tested or used anywhere yet
    # elif type(function) in [list, tuple]:
    #     fn_iterable = function
    #     def result_fn(*args, **kwargs):
    #         return type(function)(f(*args, **kwargs) for f in fn_iterable.items())
    #     function = result_fn
    return function

def get_gym_space(space, env=None, mdp=None, env_params=None, mdp_params=None):
    """
    returns gyn observation space from overcooked object (currently only in mdps) based on supplied name
    when receives string parse it to get object attribute; format is "env"/"mdp" + "." + method name i.e "mdp.lossless_state_encoding_gym_space"
    also support Dict observation spaces (supply dict of strings then)
    when receives method/function returns original method; this obviously does not work this way if attr is str/dict
    """
    space = get_overcooked_obj_attr(space, env=env, mdp=mdp, env_params=env_params, 
            mdp_params=mdp_params)
    if type(space) is dict:
        space = gym.spaces.Dict(space)
    # not tested or used anywhere yet
    # elif type(space) in [list, tuple]:
    #     space = gym.spaces.Tuple(tuple(space))
    return space
