from overcooked_ai_py.agents.benchmarking import AgentEvaluator
import numpy as np
import inspect, os, shutil, glob

def softmax(logits):
    e_x = np.exp(logits.T - np.max(logits))
    return (e_x / np.sum(e_x, axis=0)).T

def get_base_env(mdp_params, env_params, outer_shape=None, mdp_params_schedule_fn=None):
    ae = get_base_ae(mdp_params, env_params, outer_shape, mdp_params_schedule_fn)
    return ae.env

def get_base_mlam(mdp_params, env_params, outer_shape=None, mdp_params_schedule_fn=None):
    ae = get_base_ae(mdp_params, env_params, outer_shape, mdp_params_schedule_fn)
    return ae.mlam

def get_base_ae(mdp_params, env_params, outer_shape=None, mdp_params_schedule_fn=None):
    """
    mdp_params: one set of fixed mdp parameter used by the enviroment
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
    if not hasattr(a, '__iter__'):
        return a == b

    if len(a) != len(b):
        return False

    for elem_a, elem_b in zip(a, b):
        if not iterable_equal(elem_a, elem_b):
            return False

    return True

def move_ppo_agent(old_dir, new_dir):
    """
    ### Summary
    Move a serialized PPO trainer, preserving our default directory schema

    Arguments:
        - old_dir (str): Path to previously trained PPO trainer
        - new_dir (str): Where to copy files into our new schema. This directory will be created
            if it doesn't exist

    Before executing, the directory structure should be of the following form:

    /old_dir
        checkpoint*
        checkpoint*.tune-metadata
        config.pickle

    After executing, the following directory structure will exit

    /new_dir
        /agent
            agent
            agent.tune-metadata
            config.pickle

    TODO: Make this function idepotent
    """
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    agent_dir = os.path.join(new_dir, 'agent')
    shutil.copytree(old_dir, agent_dir)
    checkpoint_files = glob.glob(os.path.join(agent_dir, 'checkpoint*'))
    for checkpoint_file in checkpoint_files:
        path, extension = os.path.splitext(checkpoint_file)
        new_file_name = os.path.join(os.path.dirname(path), 'agent' + extension)
        os.rename(checkpoint_file, new_file_name)