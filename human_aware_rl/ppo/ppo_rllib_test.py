import unittest, os, shutil, pickle, ray, random, argparse, sys, glob
os.environ['RUN_ENV'] = 'local'
from human_aware_rl.ppo.ppo_rllib_client import ex
from human_aware_rl.ppo.ppo_rllib_from_params_client import ex_fp
from human_aware_rl.imitation.behavior_cloning_tf2 import get_default_bc_params, train_bc_model
from human_aware_rl.static import PPO_EXPECTED_DATA_PATH, AGENTS_SCHEDULE_PATH, NON_ML_AGENTS_PARAMS_PATH, \
    FEATURIZE_FNS_PATH, OBS_SPACES_PATH, MLAM_PARAMS_JSON_PATH, MLAM_PARAMS_TXT_PATH
from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.rllib.rllib import load_agent, load_agent_pair
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
import tensorflow as tf
import numpy as np


# Note: using the same seed across architectures can still result in differing values
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)

greedy_human_model_schedule = [{'timestep': 0, 'agents': [{'ppo': 1}, {'GreedyHumanModel': 1}]},
                               {'timestep': 1, 'agents': [{'ppo': 1}, {'GreedyHumanModel': 1}]}]

class TestPPORllib(unittest.TestCase):

    """
    Unittests for rllib PPO training loop

    compute_pickle (bool):      Whether the results of this test should be stored as the expected values for future tests
    strict (bool):              Whether the results of this test should be compared against expected values for exact match
    min_performance (int):      Minimum sparse reward that must be achieved during training for test to count as "success"

    Note, this test always performs a basic sanity check to verify some learning is happening, even if the `strict` param is false
    """

    def __init__(self, test_name, compute_pickle, strict, min_performance, **kwargs):
        super(TestPPORllib, self).__init__(test_name)
        self.compute_pickle = compute_pickle
        self.strict = strict
        self.min_performance = min_performance
    
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
    
    def _compare_results_with_expected(self, key, results, min_performance_key="average_total_reward"):
        # Sanity check (make sure it begins to learn to receive dense reward)
        self.assertGreaterEqual(results[min_performance_key], self.min_performance)
        if self.compute_pickle:
            self.expected[key] = results
        # Reproducibility test
        if self.strict:
            if self.expected.get(key):
                self.assertDictEqual(results, self.expected[key])
            else:
                print(f"no key found in expected pickle: {key}")
                self.assertTrue(self.expected.get(key))
    
    def test_save_load(self):
        # Train a quick self play agent for 2 iterations
        ex.run(
            config_updates={
                # Please feel free to modify the parameters below
                "results_dir": self.temp_results_dir,
                "experiment_name" : "save_load_test",
                "layout_name" : "cramped_room",
                "num_workers": 1,
                "train_batch_size": 800,
                "sgd_minibatch_size": 800,
                "num_training_iters": 2,
                "evaluation_interval": 10,
                "entropy_coeff_start": 0.0,
                "entropy_coeff_end": 0.0,
                "use_phi": False,
                "evaluation_display": False
            }
        )

        # Kill all ray processes to ensure loading works in a vaccuum
        ray.shutdown()

        # Where the agent is stored (this is kind of hardcoded, would like for it to be more easily obtainable)
        load_path = os.path.join(glob.glob(os.path.join(self.temp_results_dir, "save_load_test*"))[0], 'checkpoint_2', 'checkpoint-2')

        # Load a dummy state
        mdp = OvercookedGridworld.from_layout_name("cramped_room")
        state = mdp.get_standard_start_state()

        # Ensure simple single-agent loading works
        agent_0 = load_agent(load_path)
        agent_0.reset()

        agent_1 = load_agent(load_path)
        agent_1.reset()

        # Ensure forward pass of policy network still works
        _, _ = agent_0.action(state)
        _, _ = agent_1.action(state)

        # Now let's load an agent pair and evaluate it
        agent_pair = load_agent_pair(load_path)
        ae = AgentEvaluator.from_layout_name(mdp_params={"layout_name" : "cramped_room"}, env_params={"horizon" : 400})

        # We assume no runtime errors => success, no performance consistency check for now
        ae.evaluate_agent_pair(agent_pair, 1)

    def test_ppo_sp_no_phi(self):
        # Train a self play agent for 30 iterations
        results = ex.run(
            config_updates={
                # Please feel free to modify the parameters below
                "results_dir": self.temp_results_dir,
                "num_workers": 2,
                "train_batch_size": 1600,
                "sgd_minibatch_size": 800,
                "num_training_iters": 30,
                "evaluation_interval": 10,
                "entropy_coeff_start": 0.0,
                "entropy_coeff_end": 0.0,
                "use_phi": False,
                "evaluation_display": False
            }
        ).result
        self._compare_results_with_expected('test_ppo_sp_no_phi', results)

    def test_ppo_sp_yes_phi(self):
        # Train a self play agent for 30 iterations
        results = ex.run(
            config_updates={
                # Please feel free to modify the parameters below
                "results_dir": self.temp_results_dir,
                "num_workers": 2,
                "train_batch_size": 1600,
                "sgd_minibatch_size": 800,
                "num_training_iters": 30,
                "evaluation_interval": 10,
                "entropy_coeff_start": 0.0,
                "entropy_coeff_end": 0.0,
                "use_phi": True,
                "evaluation_display": False
            }
        ).result
        self._compare_results_with_expected('test_ppo_sp_yes_phi', results)

    def test_ppo_fp_sp_no_phi(self):
        # Train a self play agent for 30 iterations
        results = ex_fp.run(
            config_updates={
                "results_dir": self.temp_results_dir,
                "num_workers": 1,
                "train_batch_size": 1600,
                "sgd_minibatch_size": 800,
                "num_training_iters": 30,
                "evaluation_interval": 10,
                "use_phi": False,
                "entropy_coeff_start": 0.0002,
                "entropy_coeff_end": 0.00005,
                "lr": 7e-4,
                "seeds": [0],
                "outer_shape": (5, 4),
                "evaluation_display": False
            }
        ).result
        self._compare_results_with_expected('test_ppo_fp_sp_no_phi', results)

    def test_ppo_d2rl(self):
        # Train a self play agent for 30 iterations
        results = ex_fp.run(
            config_updates={
                "results_dir": self.temp_results_dir,
                "num_workers": 1,
                "train_batch_size": 1600,
                "sgd_minibatch_size": 800,
                "num_training_iters": 30,
                "evaluation_interval": 10,
                "use_phi": False,
                "entropy_coeff_start": 0.0002,
                "entropy_coeff_end": 0.00005,
                "lr": 7e-4,
                "seeds": [0],
                "outer_shape": (5, 4),
                "evaluation_display": False,
                "D2RL": True
            }
        ).result
        self._compare_results_with_expected('test_ppo_d2rl', results)

    def test_ppo_fp_sp_yes_phi(self):
        # Train a self play agent for 30 iterations
        results = ex_fp.run(
            config_updates={
                "results_dir": self.temp_results_dir,
                "num_workers": 1,
                "train_batch_size": 1600,
                "sgd_minibatch_size": 800,
                "num_training_iters": 30,
                "evaluation_interval": 10,
                "use_phi": True,
                "entropy_coeff_start": 0.0002,
                "entropy_coeff_end": 0.00005,
                "lr": 7e-4,
                "seeds": [0],
                "outer_shape": (5, 4),
                "evaluation_display": False
            }
        ).result
        self._compare_results_with_expected('test_ppo_fp_sp_yes_phi', results)

    def test_ppo_non_ml_agents(self):
        results = ex.run(
            config_updates={
                # Please feel free to modify the parameters below
                "results_dir": self.temp_results_dir,
                "num_workers": 2,
                "train_batch_size": 1600,
                "sgd_minibatch_size": 800,
                "num_training_iters": 30,
                "evaluation_interval": 10,
                "entropy_coeff_start": 0.0,
                "entropy_coeff_end": 0.0,
                "use_phi": False,
                "evaluation_display": False,
                "evaluation_agents": ["ppo", "GreedyHumanModel"],
                "agents_schedule": greedy_human_model_schedule
            }
        ).result
        self._compare_results_with_expected('test_ppo_non_ml_agents', results)

    def test_ppo_dict_obs_spaces(self, lstm=False):
        results = ex.run(
            config_updates={
                # Please feel free to modify the parameters below
                "results_dir": self.temp_results_dir,
                "num_workers": 2,
                "train_batch_size": 1600,
                "sgd_minibatch_size": 800,
                "num_training_iters": 30,
                "evaluation_interval": 10,
                "entropy_coeff_start": 0.0,
                "entropy_coeff_end": 0.0,
                "use_phi": False,
                "evaluation_display": False,
                "featurize_fns_file": FEATURIZE_FNS_PATH,
                "observation_spaces_file": OBS_SPACES_PATH,
                "use_lstm": lstm
            }
        ).result
        key_name = "test_ppo_dict_obs_spaces"
        if lstm: key_name = "lstm_" + key_name
        self._compare_results_with_expected(key_name, results)

    def test_change_featurization_func(self):
        results = ex.run(
            config_updates={
                # Please feel free to modify the parameters below
                "results_dir": self.temp_results_dir,
                "num_workers": 2,
                "train_batch_size": 1600,
                "sgd_minibatch_size": 800,
                "num_training_iters": 20,
                "evaluation_interval": 10,
                "entropy_coeff_start": 0.0,
                "entropy_coeff_end": 0.0,
                "use_phi": False,
                "evaluation_display": False,
                "featurize_fns": {"ppo": "env.featurize_state_mdp", "bc": "env.featurize_state_mdp"},
                "observation_spaces": {"ppo": "mdp.featurize_state_gym_space", "bc": "mdp.featurize_state_gym_space"},
                "NUM_CONV_LAYERS": 0
            }
        ).result
        self._compare_results_with_expected('test_change_featurization_func', results)

    def test_load_data_from_files(self):
        # check first for loaded mlam params from json file
        config_updates = {
                # Please feel free to modify the parameters below
                "results_dir": self.temp_results_dir,
                "num_workers": 2,
                "train_batch_size": 1600,
                "sgd_minibatch_size": 800,
                "num_training_iters": 20,
                "evaluation_interval": 10,
                "entropy_coeff_start": 0.0,
                "entropy_coeff_end": 0.0,
                "use_phi": False,
                "evaluation_display": False,
                # name "greedy-human" is specified in file at NON_ML_AGENTS_PARAMS_PATH
                "evaluation_agents": ["ppo", "greedy-model"], 
                "agents_schedule_file": AGENTS_SCHEDULE_PATH,
                "non_ml_agents_params_file": NON_ML_AGENTS_PARAMS_PATH,
                "featurize_fns_file": FEATURIZE_FNS_PATH,
                "observation_spaces_file": OBS_SPACES_PATH,
                "mlam_params_file": MLAM_PARAMS_JSON_PATH,
            }
        results = ex.run(config_updates=config_updates).result

        self._compare_results_with_expected('test_load_data_from_files', results)
        # check for for loaded mlam params from txt file (by doing eval on string)
        config_updates["mlam_params_file"] = MLAM_PARAMS_TXT_PATH
        set_global_seed(0)
        results = ex.run(config_updates=config_updates).result
        self._compare_results_with_expected('test_load_data_from_files', results)

    def test_ppo_bc(self, lstm=False):
        # Train bc model
        model_dir = self.temp_model_dir
        bc_params = get_default_bc_params()
        bc_params['training_params']['epochs'] = 10
        train_bc_model(model_dir, bc_params)
    
        # Train rllib model
        results = ex.run(config_updates={
            "results_dir" : self.temp_results_dir,
            "bc_schedule" : [(0.0, 0.0), (8e3, 1.0)],
            "num_training_iters" : 20,
            "bc_model_dir" : model_dir,
            "evaluation_interval" : 5,
            "evaluation_agents": ["ppo", "bc"],
            "use_lstm": lstm
            }
        ).result
        key_name = 'test_ppo_bc'
        if lstm: key_name = "lstm_" + key_name
        self._compare_results_with_expected(key_name, results)

    
    def test_lstm_ppo_dict_obs_spaces(self):
        self.test_ppo_dict_obs_spaces(lstm=True)

    def test_lstm_ppo_bc(self):
        self.test_ppo_bc(lstm=True)

def _clear_pickle():
    # Write an empty dictionary to our static "expected" results location
    with open(PPO_EXPECTED_DATA_PATH, 'wb') as f:
        pickle.dump({}, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute-pickle', '-cp', action="store_true")
    parser.add_argument('--strict', '-s', action="store_true")
    parser.add_argument('--min_performance', '-mp', default=5)
    parser.add_argument('--run-lstm-tests', action="store_true")

    args = vars(parser.parse_args())

    assert not (args['compute_pickle'] and args['strict']), "Cannot compute pickle and run strict reproducibility tests at same time"
    if args['compute_pickle']:
        _clear_pickle()

    suite = unittest.TestSuite()
    suite.addTest(TestPPORllib('test_save_load', **args))
    suite.addTest(TestPPORllib('test_ppo_sp_no_phi', **args))
    suite.addTest(TestPPORllib('test_ppo_sp_yes_phi', **args))
    suite.addTest(TestPPORllib('test_ppo_fp_sp_no_phi', **args))
    suite.addTest(TestPPORllib('test_ppo_d2rl', **args))
    suite.addTest(TestPPORllib('test_ppo_fp_sp_yes_phi', **args))
    suite.addTest(TestPPORllib('test_ppo_non_ml_agents', **args))
    suite.addTest(TestPPORllib('test_ppo_dict_obs_spaces', **args))
    suite.addTest(TestPPORllib('test_change_featurization_func', **args))
    suite.addTest(TestPPORllib('test_load_data_from_files', **args))
    suite.addTest(TestPPORllib('test_ppo_bc', **args))


    if args['run_lstm_tests']:
        suite.addTest(TestPPORllib('test_lstm_ppo_dict_obs_spaces', **args))
        suite.addTest(TestPPORllib('test_lstm_ppo_bc', **args))
    
    success = unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
    sys.exit(not success)
