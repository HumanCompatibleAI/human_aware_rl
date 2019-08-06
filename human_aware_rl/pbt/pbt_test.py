import unittest
from hr_coordination.utils import reset_tf

class TestPBT(unittest.TestCase):

    def setUp(self):
        reset_tf()

    def test_running_ppo(self):
        from human_aware_rl.pbt.pbt import ex as ex_pbt
        ex_pbt.run(named_configs=['fixed_mdp'], config_updates={'LOCAL_TESTING': True})
        # TODO: Add some metric for reproducibility?

if __name__ == '__main__':
    unittest.main()