import unittest

from overcooked_ai_py.utils import load_pickle, save_pickle

from human_aware_rl.utils import reset_tf
from human_aware_rl.ppo.ppo import ex as ex_ppo
from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved
from human_aware_rl.experiments.bc_experiments import BEST_BC_MODELS_PATH

class TestPPO(unittest.TestCase):

    def setUp(self):
        reset_tf()

    def test_running_ppo_sp(self):
        reset_tf()

        run = ex_ppo.run(config_updates={'LOCAL_TESTING': True, 'layout_name': 'simple', 'OTHER_AGENT_TYPE': 'sp'})
        # Just making sure seeding is working correctly and not changing actual outputs
        train_info = run.result[0]

        # Uncomment to make current output standard output to check against
        # save_pickle(train_info, 'data/testing/ppo_sp_train_info')

        expected_sp_dict = load_pickle('data/testing/ppo_sp_train_info')
        for k, v in train_info.items():
            for found_item, expected_item in zip(v, expected_sp_dict[k]):
                self.assertAlmostEqual(found_item, expected_item, places=5)

    def test_running_ppo_bc_train(self):
        # Check model exists and has right params
        layout_name = 'simple'
        best_bc_model_paths = load_pickle(BEST_BC_MODELS_PATH)
        bc_model_path = best_bc_model_paths["train"][layout_name]

        print("LOADING BC MODEL FROM: {}".format(bc_model_path))
        _, bc_params = get_bc_agent_from_saved(bc_model_path)

        expected_bc_params = {'data_params': {'train_mdps': ['simple'], 'ordered_trajs': True, 'human_ai_trajs': False, 'data_path': 'data/human/clean_train_trials.pkl'}, 'mdp_params': {'layout_name': 'simple', 'start_order_list': None}, 'env_params': {'horizon': 400}, 'mdp_fn_params': {}}
        self.assertDictEqual(expected_bc_params, bc_params)

        # Run twice with same seed and compare output dicts. Did not do as above because additional dependency on the human model

        reset_tf()
        run = ex_ppo.run(config_updates={'LOCAL_TESTING': True, 'layout_name': layout_name, 'OTHER_AGENT_TYPE': 'bc_train', 'SEEDS': [10]})
        train_info0 = run.result[0]

        reset_tf()
        run = ex_ppo.run(config_updates={'LOCAL_TESTING': True, 'layout_name': layout_name, 'OTHER_AGENT_TYPE': 'bc_train', 'SEEDS': [10]})
        train_info1 = run.result[0]

        self.assertDictEqual(train_info0, train_info1)

        # Uncomment to make current output standard output to check against
        # save_pickle(train_info1, 'data/testing/ppo_bc_train_info')

        expected_dict = load_pickle('data/testing/ppo_bc_train_info')
        for k, v in train_info1.items():
            for found_item, expected_item in zip(v, expected_dict[k]):
                self.assertAlmostEqual(found_item, expected_item, places=5)

if __name__ == '__main__':
    unittest.main()