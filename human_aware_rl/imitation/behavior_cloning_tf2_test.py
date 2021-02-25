import unittest, os, shutil, copy, pickle, random, argparse, sys
import numpy as np
import tensorflow as tf
from human_aware_rl.imitation.behavior_cloning_tf2 import BC_SAVE_DIR, get_default_bc_params, train_bc_model, \
    build_bc_model, save_bc_model, load_bc_model, evaluate_bc_model, _get_observation_shape, _get_orders_shape, \
    evaluate_bc_model_metrics, BehaviorCloningPolicy, load_data
from human_aware_rl.human.process_dataframes import get_trajs_from_data
from human_aware_rl.static import BC_EXPECTED_DATA_PATH, JSON_TRAJECTORY_DATA_PATH, PICKLE_TRAJECTORY_DATA_PATH

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def setup_bc_params(bc_params):
    # sets up shapes and figure out if use CNN
    bc_params['observation_shape'] = _get_observation_shape(bc_params)
    bc_params['orders_shape'] = _get_orders_shape(bc_params)
    bc_params["cnn_params"]["use_cnn"] = bc_params["data_params"]["state_processing_function"] != 'env.featurize_state_mdp'


def get_dummy_input(bc_params):
    #NOTE: can add caching if too slow, see: https://stackoverflow.com/questions/6358481/using-functools-lru-cache-with-dictionary-arguments
    # no processing orders as they won't be used anyway
    processed_trajs, _ = get_trajs_from_data(**bc_params["data_params"], include_orders=False, silent=True)
    return np.vstack(processed_trajs["ep_observations"])[:1, :]

class TestBCTraining(unittest.TestCase):

    """
    Unittests for behavior cloning training and utilities

    compute_pickle (bool):      Whether the results of this test should be stored as the expected values for future tests
    strict (bool):              Whether the results of this test should be compared against expected values for exact match
    min_performance (int):      Minimum reward achieved in BC-BC rollout after training to consider training successfull

    multiple_encodings (bool):  Whether tests should be made using multiple encodings
    Note, this test always performs a basic sanity check to verify some learning is happening, even if the `strict` param is false
    """

    def __init__(self, test_name, compute_pickle, strict, multiple_encodings, min_performance, **kwargs):
        super(TestBCTraining, self).__init__(test_name)
        self.compute_pickle = compute_pickle
        self.strict = strict
        self.min_performance = min_performance
        self.multiple_encodings = multiple_encodings

    def _assert_almost_equal_iterables(self, iterable1, iterable2):
        self.assertEqual(type(iterable1), type(iterable2))
        if isinstance(iterable1, dict):
            self.assertSetEqual(set(iterable1.keys()), set(iterable2.keys()))
            for k in iterable1.keys():
                self._assert_almost_equal_iterables(iterable1[k], iterable2[k])
        elif isinstance(iterable1, list) or isinstance(iterable1, tuple):
            self.assertEqual(len(iterable1), len(iterable2))
            for elem1, elem2 in zip(iterable1, iterable2):
                self._assert_almost_equal_iterables(elem1, elem2)
        elif isinstance(iterable1, np.ndarray) or isinstance(iterable1, tf.Tensor):
            self.assertTrue(np.allclose(iterable1, iterable2))
        else:
            self.assertAlmostEqual(iterable1, iterable2)

    def _compare_iterable_with_expected(self, key, compared_value):
        if self.compute_pickle:
            self.expected[key] = compared_value
        if self.strict:
            if self.expected.get(key) is None:
                print(f"no key found in expected pickle: {key}")
                self.assertIsNotNone(self.expected.get(key))
            else:
                self._assert_almost_equal_iterables(compared_value, self.expected[key])

    def setUp(self):
        set_global_seed(0)
        default_bc_params = get_default_bc_params()
        default_bc_params["mdp_params"]["layout_name"] = "cramped_room"
        default_bc_params["training_params"]["epochs"] = 1

        bc_params_lossless_with_orders = copy.deepcopy(default_bc_params)
        bc_params_lossless_with_orders["data_params"]["state_processing_function"] = "env.lossless_state_encoding_mdp"
        bc_params_lossless_with_orders["predict_orders"] = True
        setup_bc_params(bc_params_lossless_with_orders)
        self.bc_params_to_test = {"lossless_encoding_and_orders": bc_params_lossless_with_orders}

        if self.multiple_encodings:
            bc_params_lossless_encoding = copy.deepcopy(default_bc_params)
            bc_params_lossless_encoding["data_params"]["state_processing_function"] = "env.lossless_state_encoding_mdp"
            bc_params_lossless_encoding["predict_orders"] = False
            setup_bc_params(bc_params_lossless_encoding)
            self.bc_params_to_test["lossless_encoding"] = bc_params_lossless_encoding
            
            bc_params_featurization_state = copy.deepcopy(default_bc_params)
            bc_params_featurization_state["data_params"]["state_processing_function"] = "env.featurize_state_mdp"
            bc_params_featurization_state["predict_orders"] = False
            setup_bc_params(bc_params_featurization_state)
            self.bc_params_to_test["featurization_state"] = bc_params_featurization_state

        self.model_dir = os.path.join(BC_SAVE_DIR, "test_model")

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.initial_states = [np.zeros((1, default_bc_params['cell_size'])), np.zeros((1, default_bc_params['cell_size']))]
        with open(BC_EXPECTED_DATA_PATH, "rb") as f:
            self.expected = pickle.load(f)

    def tearDown(self):
        if self.compute_pickle:
            with open(BC_EXPECTED_DATA_PATH, 'wb') as f:
                pickle.dump(self.expected, f)

        shutil.rmtree(self.model_dir)

    def test_load_data_from_json(self, lstm=False):
        #NOTE: there is not corresponding method for pickled df data as it is used by default in other tests
        expected_single_agent_data_shapes = [(200, 5, 4, 26), (), (200, 1), (200, 1)]
        expected_double_agent_data_shapes = [(400, 5, 4, 26), (), (400, 1), (400, 1)]
        expected_single_agent_lstm_data_shapes = [(2, 100, 5, 4, 26), (2,), (2, 100, 1), (2, 100, 1)]
        expected_double_agent_lstm_data_shapes = [(4, 100, 5, 4, 26), (4,), (4, 100, 1), (4, 100, 1)]

        bc_params = self.bc_params_to_test["lossless_encoding_and_orders"]
        bc_params["use_lstm"] = lstm
        data_params = bc_params["data_params"]
        data_params["from_dataframe"] = False

        data_params["data_path"] = JSON_TRAJECTORY_DATA_PATH
        data_params["agent_idxs"] = [0]
        data = load_data(bc_params, False)
        key_name = "test_load_data_from_json_single_agent"
        if lstm:
            key_name = "lstm_"+key_name
            expected_data_shapes = expected_single_agent_lstm_data_shapes
        else:
            expected_data_shapes = expected_single_agent_data_shapes
        self.assertListEqual([np.array(d).shape for d in data], expected_data_shapes)
        self._compare_iterable_with_expected(key_name, data)

        data_params["agent_idxs"] = [0,1]
        data = load_data(bc_params, False)
        key_name = "test_load_data_from_json"
        if lstm:
            key_name = "lstm_"+key_name
            expected_data_shapes = expected_double_agent_lstm_data_shapes
        else:
            expected_data_shapes = expected_double_agent_data_shapes
        self.assertListEqual([np.array(d).shape for d in data], expected_data_shapes)
        self._compare_iterable_with_expected(key_name, data)

        data_params["data_path"] = PICKLE_TRAJECTORY_DATA_PATH
        data = load_data(bc_params, False)
        #NOTE: same key as previous check - result should be the same
        key_name = "test_load_data_from_json"
        if lstm:
            key_name = "lstm_"+key_name
            expected_data_shapes = expected_double_agent_lstm_data_shapes
        else:
            expected_data_shapes = expected_double_agent_data_shapes
        self.assertListEqual([np.array(d).shape for d in data], expected_data_shapes)
        self._compare_iterable_with_expected(key_name, data)

    def test_model_construction(self, lstm=False):
        for name, bc_params in self.bc_params_to_test.items():
            bc_params["use_lstm"] = lstm
            model = build_bc_model(**bc_params)
            key_name = f"test_model_construction_{name}"
            if lstm: key_name = "lstm_" + key_name
            self._compare_iterable_with_expected(key_name, self._model_forward(model, bc_params=bc_params, lstm=lstm))


    def test_save_and_load(self, lstm=False):
        for name, bc_params in self.bc_params_to_test.items():
            bc_params["use_lstm"] = lstm
            model = build_bc_model(**bc_params)
            save_bc_model(self.model_dir, model, bc_params)
            loaded_model, loaded_params = load_bc_model(self.model_dir)
            self.assertDictEqual(bc_params, loaded_params)
            dummy_input = get_dummy_input(bc_params)

            self._assert_almost_equal_iterables(
                self._model_action_prediction(model, dummy_input, lstm=lstm),
                self._model_action_prediction(loaded_model, dummy_input, lstm=lstm)
            )


    def test_training(self, lstm=False):       
        for name, bc_params in self.bc_params_to_test.items():
            bc_params["use_lstm"] = lstm
            model = train_bc_model(self.model_dir, bc_params)
            key_name = f'test_training_{name}'
            if lstm: key_name = "lstm_" + key_name
            self._compare_iterable_with_expected(key_name, self._model_forward(model, bc_params=bc_params, lstm=lstm))

    def test_agent_evaluation(self, lstm=False):
        for name, bc_params in self.bc_params_to_test.items():
            bc_params["use_lstm"] = lstm
            bc_params["training_params"]["epochs"] = 1 if lstm else 20
            model = train_bc_model(self.model_dir, bc_params)
            results = evaluate_bc_model(model, bc_params)
            key_name = f'test_agent_evaluation_{name}'
            if lstm: key_name = "lstm_" + key_name
            # Sanity Check
            self.assertGreaterEqual(results, self.min_performance)
            self._compare_iterable_with_expected(key_name, results)

    def test_agent_metrics_evaluation(self, lstm=False):
        for name, bc_params in self.bc_params_to_test.items():
            bc_params["use_lstm"] = lstm
            bc_params["training_params"]["epochs"] = 1 if lstm else 20
            model = train_bc_model(self.model_dir, bc_params)
            metrics = evaluate_bc_model_metrics(model, bc_params)

            key_name = f'test_agent_metrics_evaluation_{name}'
            if lstm: key_name = "lstm_" + key_name
            self._compare_iterable_with_expected(key_name, metrics)
    
    def test_behavior_cloning_policy(self, lstm=False):
        for name, bc_params in self.bc_params_to_test.items():
            bc_params["use_lstm"] = lstm
            model = train_bc_model(self.model_dir, bc_params)
            policy = BehaviorCloningPolicy.from_model(model, bc_params, stochastic=False)
            key_name = f'test_behavior_cloning_policy_actions_{name}'
            if lstm: key_name = "lstm_" + key_name
            observations = get_dummy_input(bc_params)
            actions_output = policy.compute_actions(observations, self.initial_states)
            self._compare_iterable_with_expected(key_name, actions_output)

            if bc_params["predict_orders"]:
                key_name = f'test_behavior_cloning_policy_orders_{name}'
                if lstm: key_name = "lstm_" + key_name
                orders_output = policy.predict_orders(observations, self.initial_states)
                self._compare_iterable_with_expected(key_name, orders_output)
            else:
                self.assertRaises(AssertionError, policy.predict_orders, observations, self.initial_states)
            
    def test_lstm_construction(self):
        self.test_model_construction(lstm=True)

    def test_lstm_save_and_load(self):
        self.test_save_and_load(lstm=True)
        
    def test_lstm_training(self):
        self.test_training(lstm=True)

    def test_lstm_evaluation(self):
        self.test_agent_evaluation(lstm=True)

    def test_lstm_agent_metrics_evaluation(self):
        self.test_agent_metrics_evaluation(lstm=True)

    def test_lstm_behavior_cloning_policy(self):
        self.test_behavior_cloning_policy(lstm=True)

    def test_lstm_load_data_from_json(self):
        self.test_load_data_from_json(lstm=True)

    def _model_forward(self, model, obs_batch=None, lstm=False, states=None, bc_params=None):
        if obs_batch is None:
            assert bc_params is not None
            obs_batch = get_dummy_input(bc_params)
        if lstm:
            return self._lstm_forward(model, obs_batch, states)
        else:
            return model(obs_batch)

    def _lstm_forward(self, model, obs_batch, states=None):
        obs_batch = np.expand_dims(obs_batch, 1)
        seq_lens = np.ones(len(obs_batch))
        states_batch = states if states else self.initial_states
        model_out = model.predict([obs_batch, seq_lens] + states_batch)
        predicted_orders = len(model_out) > 3
        if predicted_orders:
            logits, states, orders_logits = model_out[0], model_out[1:3], model_out[3]
            orders_logits = orders_logits.reshape((orders_logits.shape[0], -1))
        else:
            logits, states = model_out[0], model_out[1:]
            orders_logits = None
        logits = logits.reshape((logits.shape[0], -1))

        return logits, states, orders_logits

    def _model_action_prediction(self, *args, **kwargs):
        model_out = self._model_forward(*args, **kwargs)
        if type(model_out) is list:
            model_out = model_out[0]
        return model_out

def _clear_pickle():
    with open(BC_EXPECTED_DATA_PATH, 'wb') as f:
        pickle.dump({}, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute-pickle', '-cp', action="store_true")
    parser.add_argument('--strict', '-s', action="store_true")
    parser.add_argument('--min-performance', '-mp', default=0)
    parser.add_argument('--run-lstm-tests', action="store_true")
    parser.add_argument('--multiple-encodings', action="store_true")

    args = vars(parser.parse_args())
    tf_version = tf.__version__

    assert not (args['compute_pickle'] and args['strict']), "Cannot compute pickle and run strict reproducibility tests at same time"

    if args['compute_pickle']:
        _clear_pickle()

    suite = unittest.TestSuite()
    suite.addTest(TestBCTraining('test_model_construction', **args))
    suite.addTest(TestBCTraining('test_save_and_load', **args))
    suite.addTest(TestBCTraining('test_training', **args))
    suite.addTest(TestBCTraining('test_agent_evaluation', **args))
    suite.addTest(TestBCTraining('test_agent_metrics_evaluation', **args))
    suite.addTest(TestBCTraining('test_behavior_cloning_policy', **args))
    suite.addTest(TestBCTraining('test_load_data_from_json', **args))

    # LSTM tests break on older versions of tensorflow so be careful with this
    if args['run_lstm_tests']:
        suite.addTest(TestBCTraining('test_lstm_construction', **args))
        suite.addTest(TestBCTraining('test_lstm_save_and_load', **args))
        suite.addTest(TestBCTraining('test_lstm_training', **args))
        suite.addTest(TestBCTraining('test_lstm_evaluation', **args))
        suite.addTest(TestBCTraining('test_lstm_agent_metrics_evaluation', **args))
        suite.addTest(TestBCTraining('test_lstm_behavior_cloning_policy', **args))
        suite.addTest(TestBCTraining('test_lstm_load_data_from_json', **args))

    success = unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
    sys.exit(not success)
