from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from human_aware_rl.rllib.rllib import OvercookedMultiAgent, RlLibAgent, load_agent, load_agent_pair
from human_aware_rl.rllib.utils import get_base_ae, softmax, get_required_arguments, iterable_equal, get_base_env
from human_aware_rl.static import RLLIB_TRAINER_PATH, TESTING_DATA_DIR
from human_aware_rl.rllib.policies import ConstantPolicy
from human_aware_rl.rllib.meta_policies import EnsemblePolicy
from numpy.lib.stride_tricks import DummyArray
from overcooked_ai_py.mdp.actions import Action
from scipy.stats import norm
import unittest, copy, pickle, os, ray, logging
import numpy as np

from human_aware_rl.utils import set_global_seed

class RllibEnvTest(unittest.TestCase):

    def setUp(self):
        self.params = copy.deepcopy(OvercookedMultiAgent.DEFAULT_CONFIG)
        self.timesteps = [0, 10, 100, 500, 1000, 1500, 2000, 2500]

    def tearDown(self):
        pass

    def _assert_lists_almost_equal(self, first, second, places=7):
        for a, b in zip(first, second):
            self.assertAlmostEqual(a, b, places=places)

    def _test_bc_schedule(self, bc_schedule, expected_bc_factors):
        self.params['multi_agent_params']['bc_schedule'] = bc_schedule
        env = OvercookedMultiAgent.from_config(self.params)
        actual_bc_factors = []

        for t in self.timesteps:
            env.anneal_bc_factor(t)
            actual_bc_factors.append(env.bc_factor)

        self._assert_lists_almost_equal(expected_bc_factors, actual_bc_factors)

    def _test_bc_creation_proportion(self, env, factor, trials=10000):
        env.bc_factor = factor
        tot_bc = 0
        for _ in range(trials):
            env.reset(regen_mdp=False)
            num_bc = sum(map(lambda agent : int(agent.startswith('bc')), env.curr_agents))
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
        self.params['multi_agent_params']['use_reward_shaping'] = True
        self.params['multi_agent_params']['reward_shaping_schedule'] = [(0, 1), (1e3, 0)]

        expected_rew_factors = [1, 990/1e3, 900/1e3, 500/1e3, 0.0, 0.0, 0.0, 0.0]
        actual_rew_factors = []

        env = OvercookedMultiAgent.from_config(self.params)

        for t in self.timesteps:
            env.anneal_reward_shaping_factor(t)
            actual_rew_factors.append(env.reward_shaping_factor)

        self._assert_lists_almost_equal(expected_rew_factors, actual_rew_factors)

    def test_bc_annealing(self):
        # Test no annealing
        self._test_bc_schedule(OvercookedMultiAgent.self_play_bc_schedule, [0.0]*len(self.timesteps))

        # Test annealing
        anneal_bc_schedule = [(0, 0.0), (1e3, 1.0), (2e3, 0.0)]
        expected_bc_factors = [0.0, 10/1e3, 100/1e3, 500/1e3, 1.0, 500/1e3, 0.0, 0.0]
        self._test_bc_schedule(anneal_bc_schedule, expected_bc_factors)

    def test_agent_creation(self):
        env = OvercookedMultiAgent.from_config(self.params)
        obs = env.reset()

        # Check that we have the right number of agents with valid names
        self.assertEqual(len(env.curr_agents), 2)
        self.assertListEqual(list(obs.keys()), env.curr_agents)

        # Ensure that bc agents are created 'factor' percentage of the time
        bc_factors = [0.0, 0.1, 0.5, 0.9, 1.0]
        for factor in bc_factors:
            self._test_bc_creation_proportion(env, factor)

    def test_ppo_idx(self):
        # Create default PPO_BC environment
        params = copy.deepcopy(self.params)
        params['multi_agent_params']['bc_schedule'] = OvercookedMultiAgent.pure_bc_schedule
        env = OvercookedMultiAgent.from_config(params)

        # Run 'trials' resets, ensuring we get ~50% agent 0 ~50% agent 1
        confidence = 0.999 # % of the time this test will pass according to CLT
        cnt_0 = 0
        trials = 1000
        std = np.sqrt(0.25 / trials)
        expected_bound = std * norm.ppf(confidence)
        for _ in range(trials):
            obs = env.reset()

            # Ensure we have one of the two possible sets of correct keys
            self.assertTrue('ppo_0' in obs.keys() or 'ppo_1' in obs.keys())
            self.assertTrue('bc_0' in obs.keys() or 'bc_1' in obs.keys())

            # Keep track of how many times ppo was agent 0
            if 'ppo_0' in obs.keys():
                cnt_0 += 1

        # Ensure we followed a 50/50 bernouilli coin flip on agent idx assignment
        self.assertLess(abs(cnt_0/trials - 0.5), expected_bound)

        # Now create env with ppo idx 0
        params['multi_agent_params']['ppo_idx'] = 0
        env = OvercookedMultiAgent.from_config(params)
        for _ in range(trials):
            obs = env.reset()

            # There is only one option for keys
            expected_keys = set(['ppo_0', 'bc_1'])
            actual_keys = set(obs.keys())
            self.assertEqual(expected_keys, actual_keys)

        params['multi_agent_params']['ppo_idx'] = 1
        env = OvercookedMultiAgent.from_config(params)

        for _ in range(trials):
            obs = env.reset()

            expected_keys = set(['ppo_1', 'bc_0'])
            actual_keys = set(obs.keys())
            self.assertEqual(expected_keys, actual_keys)


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

class RllibAgentTest(unittest.TestCase):

    def setUp(self):
        set_global_seed(0)
        logits = np.log(np.arange(len(Action.ALL_ACTIONS)) + 1)
        self.base_env = get_base_env({'layout_name' : 'cramped_room'}, {"horizon" : 4e3})
        rllib_env = OvercookedMultiAgent(self.base_env)
        self.dummy_policy = ConstantPolicy(rllib_env.ppo_observation_space, rllib_env.action_space, { "logits" : logits, "stochastic" : False })

    def assertArrayAlmostEqual(self, arr_1, arr_2, **kwargs):
        arr_1 = np.array(arr_1)
        arr_2 = np.array(arr_2)
        self.assertTrue(np.allclose(arr_1, arr_2, **kwargs))

    def test_stochastic(self):
        dummy_feat_fn = lambda state : (state, state)
        rnd_agent = RlLibAgent(self.dummy_policy, agent_index=0, featurize_fn=dummy_feat_fn, stochastic=True).reset()
        deterministic_agent_1 = RlLibAgent(self.dummy_policy, agent_index=1, featurize_fn=dummy_feat_fn, stochastic=False).reset()
        deterministic_agent_2 = RlLibAgent(self.dummy_policy, agent_index=1, featurize_fn=dummy_feat_fn, stochastic=False).reset()

        rnd_actions = []

        state = dummy_state = self.base_env.reset()
        done = False
        while not done:
            rnd_action, _ = rnd_agent.action(state)
            det_action_1, _ = deterministic_agent_1.action(state)
            det_action_2, _ = deterministic_agent_2.action(state)
            rnd_actions.append(Action.ACTION_TO_INDEX[rnd_action])
            self.assertEqual(det_action_1, det_action_2)

            state, _, done, _ = self.base_env.step((rnd_action, det_action_1))

        empi_action_probs = np.unique(rnd_actions, return_counts=True)[1] / len(rnd_actions)
        actual_action_probs = softmax(self.dummy_policy.logits)
        calculated_action_probs = rnd_agent.action_probabilities(dummy_state)

        self.assertArrayAlmostEqual(empi_action_probs, actual_action_probs, atol=0.02)
        self.assertArrayAlmostEqual(actual_action_probs, calculated_action_probs, atol=0.02)

class RllibPoliciesTest(unittest.TestCase):

    def setUp(self):
        set_global_seed(0)
        self.base_env = get_base_env({'layout_name' : 'cramped_room'}, {"horizon" : 4e3})
        self.rllib_env = OvercookedMultiAgent(self.base_env, ficticious_self_play=True)
        self.default_policy_kwargs = {
            "observation_space" : self.rllib_env.ppo_observation_space,
            "action_space" : self.rllib_env.action_space,
            "config" : {}
        }

    def assertArrayAlmostEqual(self, arr_1, arr_2, **kwargs):
        arr_1 = np.array(arr_1)
        arr_2 = np.array(arr_2)
        self.assertTrue(np.allclose(arr_1, arr_2, **kwargs), "Expected: {}\nActual: {}".format(arr_2, arr_1))

    def _run_policy(self, policy):
        obs = self.rllib_env.reset()
        done = False
        ensemble_key = [key for key in obs.keys() if key.startswith('ensemble_ppo')][0]
        rnd_actions = []
        while not done:
            [action], _, _ = policy.compute_actions([obs[ensemble_key]])
            rnd_actions.append(action)
            obs, _, done, _ = self.rllib_env.step({ key : action for key in obs.keys()})
            done = done['__all__']

        empi_action_probs = np.unique(rnd_actions, return_counts=True)[1] / len(rnd_actions)
        return empi_action_probs

    def test_ensemble_policy(self):
        ensemble_policy = EnsemblePolicy(**self.default_policy_kwargs)

        # Ensure we have rnd agent before any additional agents are provided
        actual_action_probs = self._run_policy(ensemble_policy)
        expected_action_probs = np.ones(self.rllib_env.action_space.n) / self.rllib_env.action_space.n
        self.assertArrayAlmostEqual(actual_action_probs, expected_action_probs, atol=0.02)

        # Add some new additional base policies and ensure that they can be cycled through
        dummy_base_policies = [ensemble_policy.get_rnd_policy()]
        for i in range(1, 6):
            kwargs = self.default_policy_kwargs.copy()
            logits = np.random.random(self.rllib_env.action_space.n)
            logits = logits / np.sum(logits)
            kwargs['config']['logits'] = logits
            dummy_base_policies.append(ConstantPolicy(**kwargs))
            ensemble_policy.add_base_policy(loader_fn=lambda i : dummy_base_policies[i], loader_kwargs={"i" : i})

        # Run the policy again and ensure RND policy still 'active'
        actual_action_probs = self._run_policy(ensemble_policy)
        expected_action_probs = np.ones(self.rllib_env.action_space.n) / self.rllib_env.action_space.n
        self.assertArrayAlmostEqual(actual_action_probs, expected_action_probs, atol=0.02)

        # Ensure all loaders work properly
        for expected_policy, loaded_policy in zip(dummy_base_policies, ensemble_policy.base_policies):
            self.assertIs(expected_policy, loaded_policy)

        # Re-sample a couple of times and ensure we are running the right policy
        for _ in range(5):
            ensemble_policy.sample_policy()
            self.assertIs(dummy_base_policies[ensemble_policy.curr_policy_idx], ensemble_policy.curr_policy)
            actual_action_probs = self._run_policy(ensemble_policy)
            expected_action_probs = softmax(dummy_base_policies[ensemble_policy.curr_policy_idx].logits)
            self.assertArrayAlmostEqual(actual_action_probs, expected_action_probs, atol=0.02)

        # Add some more dummy policies and repeat the above experiment
        for i in range(6, 12):
            kwargs = self.default_policy_kwargs.copy()
            logits = np.random.random(self.rllib_env.action_space.n)
            logits = logits / np.sum(logits)
            kwargs['config']['logits'] = logits
            dummy_base_policies.append(ConstantPolicy(**kwargs))
            ensemble_policy.add_base_policy(loader_fn=lambda i : dummy_base_policies[i], loader_kwargs={"i" : i})

        # Ensure all loaders work properly
        for expected_policy, loaded_policy in zip(dummy_base_policies, ensemble_policy.base_policies):
            self.assertIs(expected_policy, loaded_policy)
        
        # Ensure we're still using the correct policy
        for _ in range(5):
            ensemble_policy.sample_policy()
            actual_action_probs = self._run_policy(ensemble_policy)
            expected_action_probs = softmax(dummy_base_policies[ensemble_policy.curr_policy_idx].logits)
            self.assertArrayAlmostEqual(actual_action_probs, expected_action_probs, atol=0.02)

class RllibSerializationTest(unittest.TestCase):

    def setUp(self):
        set_global_seed(0)
        ray.init(log_to_driver=False, logging_level=logging.CRITICAL)
        mdp = OvercookedGridworld.from_layout_name("cramped_room")
        self.dummy_state = mdp.get_standard_start_state()
        self.mdp_params = {"layout_name" : "cramped_room"}
        self.env_params = {"horizon" : 400}
        self.expected_path = os.path.join(TESTING_DATA_DIR, 'rllib', 'expected.pickle')

    def tearDown(self):
        ray.shutdown()

    def test_serialization_backwards_compat(self):
        # Load agents from fixed pre-trained rllib trainer to ensure backwards compatibility
        agent_0 = load_agent(RLLIB_TRAINER_PATH, 'ppo', trainer_params_to_override={'log_level' : 'ERROR'})
        agent_0.reset()

        agent_1 = load_agent(RLLIB_TRAINER_PATH, 'ppo', trainer_params_to_override={'log_level' : 'ERROR'})
        agent_1.reset()

        # Ensure forward pass of policy network still works
        _, _ = agent_0.action(self.dummy_state)
        _, _ = agent_1.action(self.dummy_state)

        # Load agent pair for full rollout
        pair = load_agent_pair(RLLIB_TRAINER_PATH, trainer_params_to_override={"log_level" : 'ERROR'})
        ae = get_base_ae(self.mdp_params, self.env_params)
        actual_trajectory = ae.evaluate_agent_pair(pair, 1, info=False)['ep_states']
        
        # Ensure trajectory matches expected trajecotry
        with open(self.expected_path, 'rb') as f:
            expected = pickle.load(f)
        
        expected_trajectory = expected['test_serialization_backwards_compat']
        self.assertTrue(np.array_equal(expected_trajectory, actual_trajectory))

        

if __name__ == '__main__':
    unittest.main()