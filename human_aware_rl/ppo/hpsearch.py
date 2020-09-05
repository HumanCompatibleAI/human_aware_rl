from human_aware_rl.ppo.ppo_rllib_client import ex
from human_aware_rl.ppo.ppo_rllib_from_params_client import ex_fp
import os, ray

temp_results_dir = os.path.join(os.path.abspath('.'), 'results_og_client_temp')

################################################



ex_fp.run(config_updates={
    "num_workers": 1,
    "train_batch_size": 2000,
    "sgd_minibatch_size": 1000,
    "num_training_iters": 10,
    "evaluation_interval": 2,
    "use_phi": False,
    "entropy_coeff_start": 0.0002,
    "entropy_coeff_end": 0.00005,
    "lr": 7e-4,
    "seeds": [0],
    "outer_shape": (5, 4)
})
ray.shutdown()


"""



ex_fp.run(config_updates={
    "num_mdp": 1,
    "num_workers": 16,
    "train_batch_size": 128000,
    "sgd_minibatch_size": 80000,
    "num_training_iters": 200,
    "evaluation_interval": 100,
    "use_phi": False,
    "entropy_coeff_start": 0.2,
    "entropy_coeff_end": 0.0005,
    "lr": 7e-4,
    "seeds": [0],
    "outer_shape": (5, 4)
})
ray.shutdown()


ex_fp.run(config_updates={
    "num_mdp": 1,
    "num_workers": 16,
    "train_batch_size": 128000,
    "sgd_minibatch_size": 80000,
    "num_training_iters": 200,
    "evaluation_interval": 100,
    "use_phi": False,
    "entropy_coeff_start": 1,
    "entropy_coeff_end": 0.0005,
    "lr": 7e-4,
    "seeds": [0],
    "outer_shape": (5, 4)
})
ray.shutdown()


ex_fp.run(config_updates={
    "num_mdp": 2,
    "num_workers": 16,
    "train_batch_size": 128000,
    "sgd_minibatch_size": 80000,
    "num_training_iters": 200,
    "evaluation_interval": 100,
    "use_phi": False,
    "entropy_coeff_start": 0.2,
    "entropy_coeff_end": 0.0005,
    "lr": 7e-4,
    "seeds": [0],
    "outer_shape": (5, 4)
})
ray.shutdown()


ex_fp.run(config_updates={
    "num_mdp": 4,
    "num_workers": 16,
    "train_batch_size": 128000,
    "sgd_minibatch_size": 80000,
    "num_training_iters": 200,
    "evaluation_interval": 100,
    "use_phi": False,
    "entropy_coeff_start": 0.2,
    "entropy_coeff_end": 0.0005,
    "lr": 7e-4,
    "seeds": [0],
    "outer_shape": (5, 4)
})
ray.shutdown()

"""