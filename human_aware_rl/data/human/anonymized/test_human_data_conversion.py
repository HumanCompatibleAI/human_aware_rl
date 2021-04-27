import unittest, os
import numpy as np
import pickle
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentPair, GreedyHumanModel
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS

horizon = 400

DATA_TYPE = "train"

layout_name = "coordination_ring"



def equal_dicts(d1, d2, ignore_keys):
    ignored = set(ignore_keys)
    for k1, v1 in d1.items():
        if k1 not in ignored and (k1 not in d2 or d2[k1] != v1):
            if k1 not in d2:
                print("d2 missing", k1)
            else:
                if k1 == "objects":
                    print("object difference")
                    for o1 in d1[k1]:
                        print(o1)
                    print("----")
                    for o2 in d2[k1]:
                        print(o2)
                else:
                    print("different at ", k1, "one is ", d2[k1], "one is ", v1)
            return False
    for k2, v2 in d2.items():
        if k2 not in ignored and k2 not in d1:
            print("d1 missing", k2)
            return False
    return True


def equal_pickle_and_env_state_dict(pickle_state_dict, env_state_dict):
    return equal_dicts(pickle_state_dict, env_state_dict, ['timestep'])


class TestLayout(unittest.TestCase):
    def setUp(self):
        self.base_mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.mlam = MediumLevelActionManager.from_pickle_or_compute(self.base_mdp, NO_COUNTERS_PARAMS,
                                                                    force_compute=True)
        self.env = OvercookedEnv.from_mdp(self.base_mdp, horizon=horizon)
        self.starting_state_dict = self.base_mdp.get_standard_start_state().to_dict()
        pickle_in = open("human_data_%s_state_dict_and_action_inserted.pkl" % DATA_TYPE, "rb")
        self.human_data = pickle.load(pickle_in)[layout_name]
        print("loaded data of length", len(self.human_data))

    def test_state(self):
        print(self.starting_state_dict)
        idx = 0
        for state_dict, joint_action in self.human_data[:100]:
            if state_dict.items() == self.starting_state_dict.items():
                self.env.reset()
            else:
                if not equal_pickle_and_env_state_dict(state_dict, self.env.state.to_dict()):
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
        print("%s is completely good" % layout_name)
        print("++++++++++++++++++++++++++++++")



if __name__ == '__main__':
    unittest.main()
