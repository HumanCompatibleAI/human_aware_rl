import unittest, os, shutil, pickle, ray, random, argparse, sys
os.environ['RUN_ENV'] = 'local'
from human_aware_rl.ppo.ppo_rllib_client import ex
from human_aware_rl.imitation.behavior_cloning_tf2 import get_default_bc_params, train_bc_model
from human_aware_rl.static import PPO_EXPECTED_DATA_PATH
from human_aware_rl.data_dir import DATA_DIR
import tensorflow as tf
import numpy as np

# Note: using the same seed across architectures can still result in differing values
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)

class TestPPORllib(unittest.TestCase):

    """
    Unittests for rllib PPO training loop

    compute_pickle (bool):      Whether the results of this test should be stored as the expected values for future tests
    strict (bool):              Whether the results of this test should be compared against expected values for exact match

    Note, this test always performs a basic sanity check to verify some learning is happening, even if the `strict` param is false
    """

    def __init__(self, test_name, compute_pickle, strict):
        super(TestPPORllib, self).__init__(test_name)
        self.compute_pickle = compute_pickle
        self.strict = strict
    
    def setUp(self):
        set_global_seed(0)

        # Temporary disk space to store logging results from tests
        self.temp_results_dir = os.path.join(os.path.abspath('.'), 'results_temp')
        self.temp_model_dir = os.path.join(os.path.abspath('.'), 'model_temp')

        # Make all necessary directories
        if not os.path.exists(self.temp_model_dir):
            os.makedirs(self.temp_model_dir)

        if not os.path.exists(self.temp_results_dir):
            os.makedirs(self.temp_results_dir)

        # Load in expected values (this is an empty dict if compute_pickle=True)
        with open(PPO_EXPECTED_DATA_PATH, 'rb') as f:
            self.expected = pickle.load(f)

    def tearDown(self):
        # Write results of this test to disk for future reproducibility tests
        # Note: This causes unit tests to have a side effect (generally frowned upon) and only works because 
        #   unittest is single threaded. If tests were run concurrently this could result in a race condition!
        if self.compute_pickle:
            with open(PPO_EXPECTED_DATA_PATH, 'wb') as f:
                pickle.dump(self.expected, f)
        
        # Cleanup 
        shutil.rmtree(self.temp_results_dir)
        shutil.rmtree(self.temp_model_dir)
        ray.shutdown()

    def test_ppo_sp(self):
        # Train a self play agent for 20 iterations
        results = ex.run(config_updates={"results_dir" : self.temp_results_dir, "num_training_iters" : 20, "evaluation_interval" : 5, "entropy_coeff_start" : 0.0, "entropy_coeff_end" : 0.0}).result

        # Sanity check (make sure it begins to learn to receive dense reward)
        self.assertGreaterEqual(results['average_total_reward'], 10.0)

        if self.compute_pickle:
            self.expected['test_ppo_sp'] = results
        
        # Reproducibility test
        if self.strict:
            self.assertDictEqual(results, self.expected['test_ppo_sp'])

    def test_ppo_bc(self):
        # Train bc model
        model_dir = self.temp_model_dir
        bc_params = get_default_bc_params()
        bc_params['training_params']['epochs'] = 10
        train_bc_model(model_dir, bc_params)

        # Train rllib model
        results = ex.run(config_updates={"results_dir" : self.temp_results_dir, "bc_schedule" : [(0.0, 0.0), (8e3, 1.0)], "num_training_iters" : 20, "bc_model_dir" : model_dir, "evaluation_interval" : 5}).result

        # Sanity check
        self.assertGreaterEqual(results['average_total_reward'], 20.0)

        if self.compute_pickle:
            self.expected['test_ppo_bc'] = results

        # Reproducibility test
        if self.strict:
            self.assertDictEqual(results, self.expected['test_ppo_bc'])

def _clear_pickle():
    # Write an empty dictionary to our static "expected" results location
    with open(PPO_EXPECTED_DATA_PATH, 'wb') as f:
        pickle.dump({}, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute-pickle', '-cp', action="store_true")
    parser.add_argument('--strict', '-s', action="store_true")

    args = parser.parse_args()

    assert not (args.compute_pickle and args.strict), "Cannot compute pickle and run strict reproducibility tests at same time"

    if args.compute_pickle:
        _clear_pickle()

    suite = unittest.TestSuite()
    suite.addTest(TestPPORllib('test_ppo_sp', args.compute_pickle, args.strict))
    # suite.addTest(TestPPORllib('test_ppo_bc', args.compute_pickle, args.strict))
    success = unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
    sys.exit(not success)
        


