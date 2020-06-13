import unittest, os, shutil, copy, pickle, random
import numpy as np
import tensorflow as tf
from human_aware_rl.imitation.behavior_cloning_tf2 import BC_SAVE_DIR, get_default_bc_params, train_bc_model, build_bc_model, save_bc_model, load_bc_model, evaluate_bc_model
from human_aware_rl.human.process_dataframes import get_trajs_from_data
from human_aware_rl.static import BC_EXPECTED_DATA_PATH

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

class TestBCTraining(unittest.TestCase):
    
    def setUp(self):
        set_global_seed(0)
        self.bc_params = get_default_bc_params()
        self.bc_params["mdp_params"]["layout_name"] = "cramped_room"
        self.bc_params["training_params"]["epochs"] = 1
        self.model_dir = os.path.join(BC_SAVE_DIR, "test_model")

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        processed_trajs, _ = get_trajs_from_data(**self.bc_params["data_params"], silent=True)
        self.dummy_input = np.vstack(processed_trajs["ep_observations"])[:1, :]
        with open(BC_EXPECTED_DATA_PATH, "rb") as f:
            self.expected = pickle.load(f)

    def tearDown(self):
        shutil.rmtree(self.model_dir)

    def test_model_construction(self):
        model = build_bc_model(**self.bc_params)
        self.assertTrue(np.allclose(model(self.dummy_input), self.expected["test_model_construction"]))

    def test_save_and_load(self):
        model = build_bc_model(**self.bc_params)
        save_bc_model(self.model_dir, model, self.bc_params)
        loaded_model, loaded_params = load_bc_model(self.model_dir)
        self.assertDictEqual(self.bc_params, loaded_params)
        self.assertTrue(np.allclose(model(self.dummy_input), loaded_model(self.dummy_input)))


    def test_training(self):        
        model = train_bc_model(self.model_dir, self.bc_params)
        self.assertTrue(np.allclose(model(self.dummy_input), self.expected["test_training"]))

    def test_agent_evaluation(self):
        model = train_bc_model(self.model_dir, self.bc_params)
        results = evaluate_bc_model(model, self.bc_params)
        self.assertAlmostEqual(results, self.expected['test_agent_evaluation'])




if __name__ == '__main__':
    unittest.main()