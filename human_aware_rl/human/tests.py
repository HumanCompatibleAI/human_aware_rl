import unittest, os, shutil
import numpy as np
import pickle, copy
from numpy.testing._private.utils import assert_raises
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentPair, GreedyHumanModel
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from human_aware_rl.utils import equal_dicts

from human_aware_rl.static import *
from human_aware_rl.human.process_dataframes import csv_to_df_pickle, get_trajs_from_data

class TestProcessDataFrames(unittest.TestCase):

    temp_data_dir = 'this_is_a_temp'

    base_csv_to_df_params = {
        "csv_path" : DUMMY_RAW_HUMAN_DATA_PATH,
        "out_file_prefix" : 'unittest',
        "train_test_split" : False
    }

    base_get_trajs_from_data_params = {
        "processed" : False,
        "ordered_trajs" : False,
        "silent" : True,
        "train_mdps" : ['cramped_room']
    }

    def setUp(self):
        if not os.path.exists(self.temp_data_dir):
            os.makedirs(self.temp_data_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_data_dir)

    def test_csv_to_df_pickle(self):
        # Try various button thresholds (hand-picked to lie between different values for dummy data games)
        button_thresholds = [0.2, 0.6, 0.7]
        lengths = []
        for threshold in button_thresholds:
            # dummy dataset is too small to partion so we set train_test_split=False
            params = copy.deepcopy(self.base_csv_to_df_params)
            params['out_dir'] = self.temp_data_dir
            params['button_presses_threshold'] = threshold
            data = csv_to_df_pickle(**params)
            lengths.append(len(data))
        
        # Filtered data size should be monotonically decreasing wrt button_threshold
        for i in range(len(lengths) - 1):
            self.assertGreaterEqual(lengths[i], lengths[i+1])

        # Picking a threshold that's suficiently high discards all data, should result in value error
        params = copy.deepcopy(self.base_csv_to_df_params)
        params['out_dir'] = self.temp_data_dir
        params['button_presses_threshold'] = 0.8
        self.assertRaises(ValueError, csv_to_df_pickle, out_dir=self.temp_data_dir, button_presses_threshold=0.8, **self.base_csv_to_df_params)

    def test_get_trajs_from_data(self):
        # Ensure this runs without breaking (Note: can't ensure MDP dynamics atm due to MLAM changes and variable MDPs)
        params = copy.deepcopy(self.base_get_trajs_from_data_params)
        params['data_path'] = DUMMY_CLEAN_HUMAN_DATA_PATH
        trajectories = get_trajs_from_data(**params)

    def test_get_trajs_from_data_tomato(self):
        # Ensure we can properly featurize states with updated dynamics and updated objects (i.e tomatoes)
        params = copy.deepcopy(self.base_get_trajs_from_data_params)
        params['train_mdps'] = ['inverse_marshmallow_experiment']
        params['data_path'] = os.path.join(DUMMY_HUMAN_DATA_DIR, 'dummy_hh_trials_tomato_all.pickle')
        trajectories = get_trajs_from_data(**params)

    def test_csv_to_df_to_trajs_integration(self):
        # Ensure the output of 'csv_to_df_pickle' works as valid input to 'get_trajs_from_data'
        params = copy.deepcopy(self.base_csv_to_df_params)
        params['out_dir'] = self.temp_data_dir
        _ = csv_to_df_pickle(**params)

        params = copy.deepcopy(self.base_get_trajs_from_data_params)
        params['data_path'] = os.path.join(self.temp_data_dir, 'unittest_all.pickle')
        params['train_mdps'] = ['inverse_marshmallow_experiment']
        _ = get_trajs_from_data(**params)

class TestHumanDataConversion(unittest.TestCase):

    horizon = 400
    DATA_TYPE = "train"
    layout_name = "coordination_ring"

    def _equal_pickle_and_env_state_dict(self, pickle_state_dict, env_state_dict):
        return equal_dicts(pickle_state_dict, env_state_dict, ['timestep'])

    def setUp(self):
        self.base_mdp = OvercookedGridworld.from_layout_name(self.layout_name)
        self.mlam = MediumLevelActionManager.from_pickle_or_compute(self.base_mdp, NO_COUNTERS_PARAMS,
                                                                    force_compute=True)
        self.env = OvercookedEnv.from_mdp(self.base_mdp, horizon=self.horizon)
        self.starting_state_dict = self.base_mdp.get_standard_start_state().to_dict()
        pickle_in = open("human_data_%s_state_dict_and_action_inserted.pkl" % DATA_TYPE, "rb")
        self.human_data = pickle.load(pickle_in)[self.layout_name]
        print("loaded data of length", len(self.human_data))

    def test_state(self):
        print(self.starting_state_dict)
        idx = 0
        for state_dict, joint_action in self.human_data[:100]:
            if state_dict.items() == self.starting_state_dict.items():
                self.env.reset()
            else:
                if not self.equal_pickle_and_env_state_dict(state_dict, self.env.state.to_dict()):
                    print('s_{t-1}')
                    print(self.base_mdp.state_string(OvercookedState.from_dict(self.human_data[idx-1][0])))
                    print('a_{t-1}')
                    print(self.human_data[idx - 1][1])
                    print("------------------>")

                    print("s_t: pickle")
                    print(self.base_mdp.state_string(OvercookedState.from_dict(self.human_data[idx][0])))
                    print("s_t: env")
                    print(self.base_mdp.state_string(self.env.state))

                    print("s_t dict: pickle")
                    print(self.human_data[idx][0])
                    print("s_t dict: env")
                    print(self.env.state.to_dict())

                    print("=================")
                    print("=================")
                    raise NotImplementedError()
            self.env.step(joint_action=joint_action)
            idx += 1
        print("++++++++++++++++++++++++++++++")
        print("%s is completely good" % self.layout_name)
        print("++++++++++++++++++++++++++++++")

if __name__ == '__main__':
    unittest.main()