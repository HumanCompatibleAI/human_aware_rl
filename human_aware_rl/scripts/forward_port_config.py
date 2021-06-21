import os, dill, copy
from human_aware_rl.rllib.rllib import load_agent
from human_aware_rl.rllib.meta_policies import EnsemblePolicy

def forward_port_config(config):
    config = copy.deepcopy(config)
    if 'policy_params' in config:
        return config
    model_params = config['model_params']
    ray_params = config['ray_params']
    bc_params = config['bc_params']
    bc_opt_params = config['bc_opt_params']
    del config['bc_params']
    del config['bc_opt_params']
    
    new_bc_params = {
        "cls" : bc_params['bc_policy_cls'],
        "config" : bc_params["bc_config"]
    }
    new_bc_opt_params = {
        "cls" : bc_opt_params['bc_opt_policy_cls'],
        "config" : bc_opt_params['bc_opt_config']
    }
    new_ppo_params = {
        "cls" : None,
        "config" : {
            "model" : {
                "custom_model_config" : model_params,
                
                "custom_model" : ray_params['custom_model_id']
            }
        }
    }
    new_ensemble_ppo_params = {
        "cls" : EnsemblePolicy,
        "config" : {
            "max_policies_in_memory" : 5
        }
    }
    self_play_params = {
        "ficticious_self_play" : False,
        "training_iters_per_ensemble_checkpoint" : 25,
        "training_iters_per_ensemble_sample" : 5
    }
    policy_params = {
        "bc" : new_bc_params,
        "bc_opt" : new_bc_opt_params,
        "ppo" : new_ppo_params,
        "ensemble_ppo" : new_ensemble_ppo_params
    }
    config['policy_params'] = policy_params
    config['self_play_params'] = self_play_params
    config['return_trainer'] = False
    config['environment_params']['multi_agent_params']['ficticious_self_play'] = False
    return config
    
def forward_port_checkpoint(save_path):
    config_path = os.path.join(os.path.dirname(save_path), 'config.pkl')
    with open(config_path, 'rb') as f:
        config = dill.load(f)
    config = forward_port_config(config)
    with open(config_path, 'wb') as f:
        dill.dump(config, f)

if __name__ == '__main__':
    import sys
    args = sys.argv
    assert len(args) == 2, "Improper arguments: Please specify checkpoint path"

    path = args[1]
    assert os.path.exists(path), "Path {} does not exist!".format(path)
    
    # Update the config schema
    forward_port_checkpoint(path)

    # Ensure agent loading works to verify we properly forward ported
    load_agent(path)