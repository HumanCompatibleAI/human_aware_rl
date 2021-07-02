from human_aware_rl.scripts.eval import PATH_MAP
from human_aware_rl.rllib.rllib import PPOAgent
from human_aware_rl.imitation.behavior_cloning_tf2 import BehaviorCloningAgent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
import argparse

SAVE_PATH_TEMPLATE = '/Users/nathan/bair/overcooked/overcooked-demo/server/static/assets/agents/{}/{}'

def main(layout, agent_type, out_name):
    out_name = out_name if out_name else agent_type
    save_path = SAVE_PATH_TEMPLATE.format(layout, out_name)
    load_path = PATH_MAP[layout][agent_type]
    

    if agent_type.startswith('opt'):
        agent_type = 'ppo'
    print("Loading agent from {}...".format(load_path))
    agent = load_agent(load_path, agent_type, layout)
    print("Success!")

    print("Saving agent at {}...".format(save_path))
    agent.save(save_path)    
    print("Success!")


def load_agent(load_path, agent_type, layout):
    if agent_type.startswith('bc'):
        return BehaviorCloningAgent.from_model_dir(load_path)
    if agent_type.startswith('opt'):
        agent_type = 'ppo'
    params_to_override = {
        "model_dir" : PATH_MAP[layout]['bc_train'],
        "opt_path" : PATH_MAP[layout]['opt_fsp']
    }
    return PPOAgent.from_trainer_path(load_path, agent_type=agent_type, trainer_params_to_override=params_to_override)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', '-l', default='soup_coordination', type=str)
    parser.add_argument('--agent_type', '-a', default='opt_fsp', type=str)
    parser.add_argument('--out_name', '-o', default=None, type=str)

    args = vars(parser.parse_args())
    main(**args)

