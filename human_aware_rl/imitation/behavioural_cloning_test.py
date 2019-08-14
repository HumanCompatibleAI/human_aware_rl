import unittest, copy
import numpy as np
from human_aware_rl.utils import reset_tf, set_global_seed
from human_aware_rl.imitation.behavioural_cloning import train_bc_agent, DEFAULT_BC_PARAMS, eval_with_benchmarking_from_saved, eval_with_standard_baselines

class TestBCTraining(unittest.TestCase):
    
    def setUp(self):
        reset_tf()
        set_global_seed(0)

    def test_running_and_evaluating_bc(self):
        model_save_dir = "bc_testing/"
        
        bc_params = copy.deepcopy(DEFAULT_BC_PARAMS)
        bc_params["mdp_params"]["layout_name"] = "simple"
        train_bc_agent(model_save_dir, bc_params, num_epochs=1)

        model_name = "bc_testing"
        eval_with_benchmarking_from_saved(1, model_name)

        # TODO: Have reproducibility test


if __name__ == '__main__':
    unittest.main()