from human_aware_rl.scripts.eval import PATH_MAP
from human_aware_rl.rllib.rllib import PPOAgent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
import argparse

SAVE_PATH_TEMPLATE = '/Users/nathan/bair/overcooked/overcooked-demo/server/static/assets/agents/{}/{}'

def main(layout, agent_type, out_name):
    out_name = out_name if out_name else agent_type
    save_path = SAVE_PATH_TEMPLATE.format(layout, out_name)
    load_path = PATH_MAP[layout][agent_type]
    params_to_override = {
        "model_dir" : PATH_MAP[layout]['bc_train'],
        "opt_path" : PATH_MAP[layout]['opt_fsp']
    }

    if agent_type.startswith('opt'):
        agent_type = 'ppo'
    print("Loading agent from {}...".format(load_path))
    ppo_bc_opt_agent = PPOAgent.from_trainer_path(load_path, agent_type=agent_type, trainer_params_to_override=params_to_override)
    print("Success!")

    print("Saving trainer at {}...".format(save_path))
    ppo_bc_opt_agent.save(save_path)    
    print("Success!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', '-l', default='soup_coordination', type=str)
    parser.add_argument('--agent_type', '-a', default='opt_fsp', type=str)
    parser.add_argument('--out_name', '-o', default=None, type=str)

    args = vars(parser.parse_args())
    main(**args)

