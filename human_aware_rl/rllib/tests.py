from human_aware_rl.rllib.rllib import OvercookedMultiAgent
from human_aware_rl.rllib.utils import softmax, sigmoid, get_required_arguments, iterable_equal, get_encoding_function, get_gym_space
from math import isclose
import unittest, copy, gym
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

class RllibEnvTest(unittest.TestCase):

    def setUp(self):
        self.params = copy.deepcopy(OvercookedMultiAgent.DEFAULT_CONFIG)
        self.timesteps = [0, 10, 100, 500, 1000, 1500, 2000, 2500]

    def tearDown(self):
        pass

    def _assert_dicts_almost_equal(self, first, second, places=7):
        self.assertEqual(len(first), len(second))
        for k, v in first.items():
            v2 = second[k]
            if isinstance(v, list) or isinstance(v2, list):
                self._assert_lists_almost_equal(self, v, v2, places=places)
            else:
                self.assertAlmostEqual(v, v2, places=places)
        

    def _assert_lists_almost_equal(self, first, second, places=7):
        self.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            if isinstance(a, dict) or isinstance(b, dict):
                self._assert_dicts_almost_equal(a, b, places=places)
            elif isinstance(a, list) or isinstance(b, list):
                self._assert_lists_almost_equal(a, b, places=places)
            else:
                self.assertAlmostEqual(a, b, places=places)

    def _test_bc_schedule(self, bc_schedule, expected_agents_probas):
        self.params['multi_agent_params']['bc_schedule'] = bc_schedule

        env = OvercookedMultiAgent.from_config(self.params)
        actual_agents_probas = []

        for t in self.timesteps:
            env.anneal_agents_schedule(t)
            actual_agents_probas.append(env.agents_probas)
        self._assert_lists_almost_equal(expected_agents_probas, actual_agents_probas)


    def _test_bc_creation_proportion(self, env, factor, trials=10000):
        env.agents_probas = [{"ppo": 1, "bc": 0}, {"ppo": 1-factor, "bc": factor}]
        tot_bc = 0
        for _ in range(trials):
            env.reset(regen_mdp=False)
            num_bc = sum(map(lambda agent: int(agent.startswith('bc')), env.current_agents_ids))
            self.assertLessEqual(num_bc, 1)
            tot_bc += num_bc
        actual_factor = tot_bc / trials
        self.assertAlmostEqual(actual_factor, factor, places=1)


    def test_env_creation(self):
        # Valid creation
        env = OvercookedMultiAgent.from_config(self.params)
        for param, expected in self.params['multi_agent_params'].items():
            self.assertEqual(expected, getattr(env, param))

        # Invalid bc_schedules
        invalid_schedules = [[(-1, 0.0), (1.0, 1e5)], [(0.0, 0.0), (10, 1),  (5, 0.5)], [(0, 0), (5, 1), (10, 1.5)]]
        for sched in invalid_schedules:
            self.params['multi_agent_params']['bc_schedule'] = sched
            self.assertRaises(AssertionError, OvercookedMultiAgent.from_config, self.params)

    def test_reward_shaping_annealing(self):
        self.params['multi_agent_params']['reward_shaping_factor'] = 1
        self.params['multi_agent_params']['reward_shaping_horizon'] = 1e3

        expected_rew_factors = [1, 990/1e3, 900/1e3, 500/1e3, 0.0, 0.0, 0.0, 0.0]
        actual_rew_factors = []

        env = OvercookedMultiAgent.from_config(self.params)

        for t in self.timesteps:
            env.anneal_reward_shaping_factor(t)
            actual_rew_factors.append(env.reward_shaping_factor)

        self._assert_lists_almost_equal(expected_rew_factors, actual_rew_factors)

    def test_bc_annealing(self):
        # Test no annealing
        self_play_ppo_bc_schedule = [(0, 0.0), (float("inf"), 0)]
        self_play_ppo_expected_agents_probas = [[{"ppo": 1, "bc": 0}, {"ppo": 1, "bc": 0}] for t in self.timesteps]
        self._test_bc_schedule(self_play_ppo_bc_schedule, self_play_ppo_expected_agents_probas)

        # Test annealing
        anneal_bc_schedule = [(0, 0.0), (1e3, 1.0), (2e3, 0.0)]
        expected_bc_factors = [0.0, 10/1e3, 100/1e3, 500/1e3, 1.0, 500/1e3, 0.0, 0.0]
        expected_agents_probas = [[{"ppo": 1, "bc": 0}, {"ppo": 1-factor, "bc": factor}] for factor in expected_bc_factors]
        self._test_bc_schedule(anneal_bc_schedule, expected_agents_probas)

    def test_agent_creation(self):
        env = OvercookedMultiAgent.from_config(self.params)
        obs = env.reset()

        # Check that we have the right number of agents with valid names
        self.assertEqual(len(env.current_agents_ids), 2)
        self.assertListEqual(list(obs.keys()), env.current_agents_ids)

        # Ensure that bc agents are created 'factor' percentage of the time
        bc_factors = [0.0, 0.1, 0.5, 0.9, 1.0]
        for factor in bc_factors:
            self._test_bc_creation_proportion(env, factor)


class RllibUtilsTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_softmax(self):
        logits = np.array([[0.1, 0.1, 0.1],
                           [-0.1, 0.0, 0.1],
                           [0.5, -1.2, 3.2],
                           [-1.6, -2.0, -1.5]])
        expected = np.array([[0.33333333, 0.33333333, 0.33333333],
                             [0.30060961, 0.33222499, 0.3671654 ],
                             [0.06225714, 0.01137335, 0.92636951],
                             [0.36029662, 0.24151404, 0.39818934]])

        actual = softmax(logits)

        self.assertTrue(np.allclose(expected, actual))

    def test_sigmoid(self):
        logits = np.array([[-4.0, -3.0, -2.0],
                           [-1.0, 0.0, 1.0],
                           [2.0, 3.0, 4.0]])
                           
        expected = np.array([[0.01798621, 0.04742587, 0.11920292],
                             [0.26894142, 0.5       , 0.73105858],
                             [0.88079708, 0.95257413, 0.98201379]])

        actual = sigmoid(logits)

        self.assertTrue(np.allclose(expected, actual))

    def test_iterable_equal(self):
        a = [(1,), (1, 2)]
        b = ([1], [1, 2])

        self.assertTrue(iterable_equal(a, b))

        a = [(1, 2), (1)]
        b = [(1,), (1, 2)]

        self.assertFalse(iterable_equal(a, b))

    def test_get_required_arguments(self):
        
        def foo1(a):
            pass
        def foo2(a, b):
            pass
        def foo3(a, b, c):
            pass
        def foo4(a, b, c='bar'):
            pass
        def foo5(a, b='bar', d='baz', **kwargs):
            pass

        fns = [foo1, foo2, foo3, foo4, foo5]
        expected = [1, 2, 3, 2, 1]

        for fn, expected in zip(fns, expected):
            self.assertEqual(expected, len(get_required_arguments(fn)))

    def test_get_encoding_function(self):
        mdp = OvercookedGridworld.from_layout_name("cramped_room")
        mdp_params = mdp.mdp_params
        env_params = {"horizon": 100}
        env = OvercookedEnv.from_mdp(mdp, **env_params)
        state = mdp.get_standard_start_state()
        example_encoding_fns_names = ["mdp.multi_hot_orders_encoding", "env.featurize_state_mdp", "env.lossless_state_encoding_mdp"]
        example_encoding_fns = [mdp.multi_hot_orders_encoding, env.featurize_state_mdp, env.lossless_state_encoding_mdp]
        for encoding_fn_name, encoding_fn in zip(example_encoding_fns_names, example_encoding_fns):
            encoding_fn_from_name = get_encoding_function(encoding_fn_name, env=env)
            self.assertEqual(encoding_fn_from_name, encoding_fn)
            if encoding_fn_name.split(".")[0] == "mdp":
                encoding_fn_from_name = get_encoding_function(encoding_fn_name, mdp=mdp)
                self.assertEqual(encoding_fn_from_name, encoding_fn)
                encoding_fn_from_name = get_encoding_function(encoding_fn_name, mdp_params=mdp_params)
                # compare names as new instance of mdp is created
                self.assertEqual(encoding_fn_from_name.__name__, encoding_fn.__name__)
            else:
                encoding_fn_from_name = get_encoding_function(encoding_fn_name, env_params=env_params, mdp_params=mdp_params)
                # compare names as new instance of env is created
                self.assertEqual(encoding_fn_from_name.__name__, encoding_fn.__name__) 
        
        expected_encoded_state_dict = {str(i): fn(state) for i, fn in enumerate(example_encoding_fns)}
        actual_encoded_state_dict = get_encoding_function({str(i): fn_name for i, fn_name in enumerate(example_encoding_fns_names)}, env=env)(state)
        self.assertEqual(expected_encoded_state_dict.keys(), actual_encoded_state_dict.keys())
        for k in expected_encoded_state_dict.keys():
            self.assertTrue(np.array_equal(expected_encoded_state_dict[k], actual_encoded_state_dict[k]))

    def test_get_gym_space(self):
        mdp = OvercookedGridworld.from_layout_name("cramped_room")
        mdp_params = mdp.mdp_params
        env_params = {"horizon": 100}
        env = OvercookedEnv.from_mdp(mdp, **env_params)
        example_gym_space_names = ["mdp.multi_hot_orders_encoding_gym_space", "mdp.featurize_state_gym_space", 
            "mdp.lossless_state_encoding_gym_space"]
        example_gym_spaces = [mdp.multi_hot_orders_encoding_gym_space, mdp.featurize_state_gym_space, 
            mdp.lossless_state_encoding_gym_space]
        for space_name, space in zip(example_gym_space_names, example_gym_spaces):
            space_from_name = get_gym_space(space_name, env=env)
            self.assertEqual(space_from_name, space)
            if space_name.split(".")[0] == "mdp":
                space_from_name = get_gym_space(space_name, mdp=mdp)
                self.assertEqual(space_from_name, space)
                space_from_name = get_gym_space(space_name, mdp_params=mdp_params)
                self.assertEqual(space_from_name, space)
            else:
                space_from_name = get_gym_space(space_name, env_params=env_params, mdp_params=mdp_params)
                self.assertEqual(space_from_name, space)
        expected_space = gym.spaces.Dict({str(i): space for i,space in enumerate(example_gym_spaces)})
        actual_space = get_gym_space({str(i): space_name for i, space_name in enumerate(example_gym_space_names)}, env=env)
        self.assertEqual(expected_space, actual_space)

if __name__ == '__main__':
    unittest.main()