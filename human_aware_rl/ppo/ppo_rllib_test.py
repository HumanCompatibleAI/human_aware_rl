import unittest, os, shutil, pickle, ray, random
os.environ['RUN_ENV'] = 'local'
from human_aware_rl.ppo.ppo_rllib_client import ex
from human_aware_rl.imitation.behavior_cloning_tf2 import get_default_bc_params, train_bc_model
from human_aware_rl.static import PPO_EXPECTED_DATA_PATH
from human_aware_rl.data_dir import DATA_DIR
import tensorflow as tf
import numpy as np

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

class TestPPORllib(unittest.TestCase):

    def setUp(self):
        set_global_seed(0)
        self.temp_results_dir = os.path.join(os.path.abspath('.'), 'results_temp')
        self.temp_model_dir = os.path.join(os.path.abspath('.'), 'model_temp')

        if not os.path.exists(self.temp_model_dir):
            os.makedirs(self.temp_model_dir)

        if not os.path.exists(self.temp_results_dir):
            os.makedirs(self.temp_results_dir)

        with open(PPO_EXPECTED_DATA_PATH, 'rb') as f:
            self.expected = pickle.load(f)

    def tearDown(self):
        shutil.rmtree(self.temp_results_dir)
        shutil.rmtree(self.temp_model_dir)
        ray.shutdown()

    def test_ppo_sp(self):
        results = ex.run(config_updates={"results_dir" : self.temp_results_dir}).result
        self.assertDictEqual(results, self.expected['test_ppo_sp'])

    def test_ppo_bc(self):
        # Train bc model
        model_dir = self.temp_model_dir
        bc_params = get_default_bc_params()
        bc_params['training_params']['epochs'] = 10
        train_bc_model(model_dir, bc_params)

        # Train rllib model
        results = ex.run(config_updates={"results_dir" : self.temp_results_dir, "bc_schedule" : [(0.0, 0.0), (8e3, 1.0)], "num_training_iters" : 10, "bc_model_dir" : model_dir}).result
        self.assertDictEqual(results, self.expected['test_ppo_bc'])


if __name__ == '__main__':
    unittest.main()
        


