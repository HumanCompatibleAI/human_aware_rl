import os, pickle, copy
from tensorflow import keras
import tensorflow as tf
import numpy as np
from human_aware_rl.human.process_dataframes import get_trajs_from_data
from human_aware_rl.static import HUMAN_DATA_PATH
from human_aware_rl.rllib.rllib import RlLibAgent, softmax, evaluate, get_base_env, get_mlp
from human_aware_rl.data_dir import DATA_DIR
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS
from ray.rllib.policy import Policy as RllibPolicy

#################
# Configuration #
#################

BC_SAVE_DIR = os.path.join(DATA_DIR, "bc_runs")

DEFAULT_DATA_PARAMS = {
    "train_mdps": ["cramped_room"],
    "ordered_trajs": False,
    "human_ai_trajs": False,
    "processed" : True,
    "data_path": HUMAN_DATA_PATH
}

DEFAULT_MLP_PARAMS = {
    # Number of fully connected layers to use in our network
    "num_layers" : 2,
    # Each int represents a layer of that hidden size
    "net_arch" : [64, 64]
}

DEFAULT_TRAINING_PARAMS = {
    "epochs" : 100,
    "validation_split" : 0.15,
    "batch_size" : 64,
    "learning_rate" : 1e-3,
    "use_class_weights" : False
}

DEFAULT_EVALUATION_PARAMS = {
    "ep_length" : 400,
    "num_games" : 1,
    "display" : False
}

DEFAULT_BC_PARAMS = {
    "data_params": DEFAULT_DATA_PARAMS,
    "mdp_params": {'layout_name': "cramped_room"},
    "env_params": DEFAULT_ENV_PARAMS,
    "mdp_fn_params": {},
    "mlp_params" : DEFAULT_MLP_PARAMS,
    "training_params" : DEFAULT_TRAINING_PARAMS,
    "evaluation_params" : DEFAULT_EVALUATION_PARAMS,
    "action_shape" :  (len(Action.ALL_ACTIONS), )
}

# Boolean indicating whether all param dependencies have been loaded. Used to prevent re-loading unceccesarily
_params_initalized = False

def _get_base_env(bc_params):
    return get_base_env(bc_params['mdp_params'], bc_params['env_params'])

def _get_mlp(bc_params):
    return get_mlp(bc_params['mdp_params'], bc_params['env_params'])

def _get_observation_shape(bc_params):
    """
    Helper function for creating a dummy environment from "mdp_params" and "env_params" specified
    in bc_params and returning the shape of the observation space
    """
    base_env = _get_base_env(bc_params)
    mlp = _get_mlp(bc_params)
    dummy_state = base_env.mdp.get_standard_start_state()
    obs_shape = base_env.mdp.featurize_state(dummy_state, mlp)[0].shape
    return obs_shape

# For lazing loading the default params. Prevents loading on every import of this module 
def get_default_bc_params():
    global _params_initalized, DEFAULT_BC_PARAMS
    if not _params_initalized:
        DEFAULT_BC_PARAMS['observation_shape'] = _get_observation_shape(DEFAULT_BC_PARAMS)
        _params_initalized = False
    return copy.deepcopy(DEFAULT_BC_PARAMS)




##############
# Model code #
##############

def build_bc_model(observation_shape, action_shape, mlp_params, **kwargs):
    ## Inputs
    inputs = keras.Input(shape=observation_shape, name="Overcooked_observation")
    x = inputs

    ## Build fully connected layers
    assert len(mlp_params["net_arch"]) == mlp_params["num_layers"], "Invalid Fully Connected params"

    for i in range(mlp_params["num_layers"]):
        units = mlp_params["net_arch"][i]
        x = keras.layers.Dense(units, activation="relu", name="fc_{0}".format(i))(x)

    ## output layer
    logits = keras.layers.Dense(action_shape[0], name="logits")(x)

    return keras.Model(inputs=inputs, outputs=logits)

def train_bc_model(model_dir, bc_params, verbose=False):
    processed_trajs, _ = get_trajs_from_data(**bc_params["data_params"], silent=not verbose)
    inputs, targets = processed_trajs["ep_observations"], processed_trajs["ep_actions"]
    inputs = np.vstack(inputs)
    targets = np.vstack(targets)

    training_params = bc_params["training_params"]

    
    if training_params['use_class_weights']:
        # Get class counts, and use these to compute balanced class weights
        classes, counts = np.unique(targets.flatten(), return_counts=True)
        weights = sum(counts) / counts
        class_weights = dict(zip(classes, weights))
    else:
        # Default is uniform class weights
        class_weights = None

    # Retrieve un-initialized keras model
    model = build_bc_model(**bc_params)

    # Initialize the model
    model.compile(optimizer=keras.optimizers.Adam(training_params["learning_rate"]),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["sparse_categorical_accuracy"])


    # Customize our training loop with callbacks
    callbacks = [
        # Early terminate training if loss doesn't improve for "patience" epochs
        keras.callbacks.EarlyStopping(
            monitor="loss", patience=20
        ),
        # Reduce lr by "factor" after "patience" epochs of no improvement in loss
        keras.callbacks.ReduceLROnPlateau(
            monitor="loss", patience=3, factor=0.1
        ),
        # Log all metrics model was compiled with to tensorboard every epoch
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, "logs"),
            write_graph=False
        ),
        # Save checkpoints of the models at the end of every epoch (saving only the best one so far)
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "checkpoints"),
            monitor="loss",
            save_best_only=True
        )
    ]

    # Actually train our model
    model.fit(inputs, targets, callbacks=callbacks, batch_size=training_params["batch_size"], 
                epochs=training_params['epochs'], validation_split=training_params["validation_split"],
                class_weight=class_weights,
                verbose=2 if verbose else 0)

    # Save the model
    save_bc_model(model_dir, model, bc_params)

    return model
    


def save_bc_model(model_dir, model, bc_params):
    """
    Saves the specified model under the directory model_dir. This creates three items

        assets/         stores information essential to reconstructing the context and tf graph
        variables/      stores the model's trainable weights
        saved_model.pd  the saved state of the model object

    Additionally, saves a pickled dictionary containing all the parameters used to construct this model
    at model_dir/metadata.pickle
    """   
    print("Saving bc model at ", model_dir)
    model.save(model_dir, save_format='tf')
    with open(os.path.join(model_dir, "metadata.pickle"), 'wb') as f:
        pickle.dump(bc_params, f)


def load_bc_model(model_dir):
    """
    Returns the model instance (including all compilation data like optimizer state) and a dictionary of parameters
    used to create the model
    """
    print("Loading bc model from ", model_dir)
    model = keras.models.load_model(model_dir)
    with open(os.path.join(model_dir, "metadata.pickle"), "rb") as f:
        bc_params = pickle.load(f)
    return model, bc_params

def evaluate_bc_model(model, bc_params):
    """
    Creates an AgentPair object containing two instances of BC Agents, whose policies are specified by `model`. Runs
    a rollout using AgentEvaluator class in an environment specified by bc_params

    Arguments

        - model (tf.keras.Model)        A function that maps featurized overcooked states to action logits
        - bc_params (dict)              Specifies the environemnt in which to evaluate the agent (i.e. layout, reward_shaping_param)
                                            as well as the configuration for the rollout (rollout_length)

    Returns

        - reward (int)                  Total sparse reward achieved by AgentPair during rollout
    """
    evaluation_params = bc_params['evaluation_params']
    mdp_params = bc_params['mdp_params']

    # Get reference to state encoding function used by bc agents, with compatible signature
    base_env = _get_base_env(bc_params)
    mlp = _get_mlp(bc_params)
    def featurize_fn(state):
        return base_env.mdp.featurize_state(state, mlp)

    # Wrap Keras models in rllib policies
    agent_0_policy = BehaviorCloningPolicy.from_model(model, bc_params, stochastic=True)
    agent_1_policy = BehaviorCloningPolicy.from_model(model, bc_params, stochastic=True)

    # Compute the results of the rollout(s)
    results = evaluate(evaluation_params, mdp_params, agent_0_policy, agent_1_policy, featurize_fn, featurize_fn)

    # Compute the average sparse return obtained in each rollout
    reward = np.mean(results['ep_returns'])
    return reward



################
# Rllib Policy #
################

class BehaviorCloningPolicy(RllibPolicy):

    def __init__(self, observation_space, action_space, config):
        """
        RLLib compatible constructor for initializing a behavior cloning model

        observation_space (gym.Space|tuple)     Shape of the featurized observations
        action_space (gym.space|tuple)          Shape of the action space (len(Action.All_ACTIONS),)
        config (dict)                           Dictionary of relavant bc params
            - model_dir (str)                   Path to pickled keras.Model used to map observations to action logits
            - stochastic (bool)                 Whether action should return logit argmax or sample over distribution
            - bc_model (keras.Model)            Pointer to loaded policy model. Overrides model_dir
            - bc_params (dict)                  Dictionary of parameters used to train model. Required if "model" is present
        """
        super(BehaviorCloningPolicy, self).__init__(observation_space, action_space, config)

        

        if 'bc_model' in config and config['bc_model']:
            assert 'bc_params' in config, "must specify params in addition to model"
            assert type(config['bc_model']) == keras.Model, "model must be of type keras.Model"
            self._graph = None
            model, bc_params = config['bc_model'], config['bc_params']
        else:
            assert 'model_dir' in config, "must specify model directory if model not specified"
            # Construct a private graph unique to this instance to run forward pass (so it doesn't conflict with rllib graphs)
            self._graph = tf.Graph()
            self._sess = tf.compat.v1.Session(graph=self._graph)
            with self._sess.as_default():
                    model, bc_params = load_bc_model(config['model_dir'])

        self._setup_shapes()

        # Basic check to make sure model dimensions match
        assert self.observation_shape == bc_params['observation_shape']
        assert self.action_shape == bc_params['action_shape']

        self.model = model
        self.stochastic = config['stochastic']

    def _setup_shapes(self):
        # This is here to make the class compatible with both tuples or gym.Space objs for the spaces
        # Note: action_space = (len(Action.ALL_ACTIONS,)) is technically NOT the action space shape, which would be () since actions are scalars
        self.observation_shape = self.observation_space if type(self.observation_space) == tuple else self.observation_space.shape
        self.action_shape = self.action_space if type(self.action_space) == tuple else (self.action_space.n,)

        

    @classmethod
    def from_model_dir(cls, model_dir, stochastic=True):
        model, bc_params = load_bc_model(model_dir)
        config = {
            "bc_model" : model,
            "bc_params" : bc_params,
            "stochastic" : stochastic
        }
        return cls(bc_params['observation_shape'], bc_params['action_shape'], config)

    @classmethod
    def from_model(cls, model, bc_params, stochastic=True):
        config = {
            "bc_model" : model,
            "bc_params" : bc_params,
            "stochastic" : stochastic
        }
        return cls(bc_params["observation_shape"], bc_params["action_shape"], config)

    def compute_actions(self, obs_batch, 
                        state_batches=None, 
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        """
        Computes sampled actions for each of the corresponding OvercookedEnv states in obs_batch

        Args:
            obs_batch (np.array): batch of pre-process (lossless state encoded) observations

        Returns:
            actions (list|np.array): batch of output actions shape [BATCH_SIZE, ACTION_SHAPE]
            state_outs (list): only necessary for rnn hidden states
            infos (dict): dictionary of extra feature batches { "action_dist_inputs" : [BATCH_SIZE, ...] }
        """
        # Cast to np.array if list (no-op if already np.array)        
        obs_batch = np.array(obs_batch)

        # Run the model
        if self._graph:
            with self._sess.as_default():
                action_logits = self.model.predict(obs_batch)
        else:
            action_logits = self.model.predict(obs_batch)

        # Softmax in numpy to convert logits to probabilities
        action_probs = softmax(action_logits)
        if self.stochastic:
            # Sample according to action_probs for each row in the output
            actions = np.array([np.random.choice(self.action_shape[0], p=action_probs[i]) for i in range(len(action_probs))])
        else:
            actions = np.argmax(action_logits, axis=1)

        return actions,  [], { "action_dist_inputs" : action_logits }

    def get_weights(self):
        """
        No-op to keep rllib from breaking, won't be necessary in future rllib releases
        """
        pass

    def set_weights(self, weights):
        """
        No-op to keep rllib from breaking
        """
        pass


    def learn_on_batch(self, samples):
        """
        Static policy requires no learning
        """
        return {}


if __name__ == "__main__":
    params = get_default_bc_params()
    model = train_bc_model(os.path.join(BC_SAVE_DIR, 'default'), params, verbose=True)
    # Evaluate our model's performance in a rollout
    evaluate_bc_model(model, params)