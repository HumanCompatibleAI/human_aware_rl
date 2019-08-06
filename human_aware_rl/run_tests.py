import unittest
import overcooked_ai_py.run_tests_fast

from hr_coordination.imitation.behavioural_cloning_test import TestBCTraining
from hr_coordination.ppo.ppo_test import TestPPO
from hr_coordination.pbt.pbt_test import TestPBT

if __name__ == '__main__':
    unittest.main()