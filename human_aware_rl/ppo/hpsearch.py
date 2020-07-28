from human_aware_rl.ppo.ppo_rllib_client import ex
import os, ray, shutil

temp_results_dir = os.path.join(os.path.abspath('.'), 'results_og_client_temp')

############################

ex.run(config_updates={
    "layout_name": "cramped_room_tomato_simple",
    "num_training_iters": 1000,
    "evaluation_interval": 100,
    "use_phi": True,
    "entropy_coeff_start": 0.2,
    "entropy_coeff_end": 0.0005,
    "lr": 5e-3
})
ray.shutdown()



ex.run(config_updates={
    "layout_name": "cramped_room_tomato_simple",
    "num_training_iters": 1000,
    "evaluation_interval": 100,
    "use_phi": False,
    "entropy_coeff_start": 0.2,
    "entropy_coeff_end": 0.0005,
    "lr": 5e-3
})
ray.shutdown()
##################################


ex.run(config_updates={
    "layout_name": "cramped_room_tomato_simple",
    "train_batch_size": 12800,
    "sgd_minibatch_size": 5000,
    "num_training_iters": 500,
    "evaluation_interval": 50,
    "use_phi": True,
    "entropy_coeff_start": 0.2,
    "entropy_coeff_end": 0.0005,
    "lr": 5e-3
})
ray.shutdown()



ex.run(config_updates={
    "layout_name": "cramped_room_tomato_simple",
    "train_batch_size": 12800,
    "sgd_minibatch_size": 5000,
    "num_training_iters": 500,
    "evaluation_interval": 50,
    "use_phi": False,
    "entropy_coeff_start": 0.2,
    "entropy_coeff_end": 0.0005,
    "lr": 5e-3
})
ray.shutdown()

