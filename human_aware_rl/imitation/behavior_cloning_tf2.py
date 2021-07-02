import os, pickle, copy
from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.compat.v1.keras.backend import set_session, get_session
from human_aware_rl.human.process_dataframes import get_trajs_from_data
from human_aware_rl.static import HUMAN_DATA_PATH
from human_aware_rl.rllib.rllib import RlLibAgent, softmax, evaluate, get_base_ae
from human_aware_rl.data_dir import DATA_DIR
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from ray.rllib.policy import Policy as RllibPolicy

#################
# Configuration #
#################

BC_SAVE_DIR = os.path.join(DATA_DIR, "bc_runs")

DEFAULT_DATA_PARAMS = {
    "train_mdps": ["cramped_room"],
    "ordered_trajs": False,
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
    "slice_freq": 10,
    "seed": 0,
    "validation_split" : 0.15,
    "batch_size" : 64,
    "learning_rate" : 5e-4,
    "use_class_weights" : False,
    "single_traj" : False
}

DEFAULT_EVALUATION_PARAMS = {
    "ep_length" : 400,
    "num_games" : 10,
    "display" : False
}

DEFAULT_BC_PARAMS = {
    "lossless_feature": False,
    "eager" : True,
    "use_lstm" : False,
    "cell_size" : 256,
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

def _get_base_ae(bc_params):
    return get_base_ae(bc_params['mdp_params'], bc_params['env_params'])

def _get_observation_shape(bc_params):
    """
    Helper function for creating a dummy environment from "mdp_params" and "env_params" specified
    in bc_params and returning the shape of the observation space
    """
    base_ae = _get_base_ae(bc_params)
    base_env = base_ae.env
    dummy_state = base_env.mdp.get_standard_start_state()
    if bc_params["lossless_feature"]:
        obs_shape = base_env.lossless_state_encoding_mdp(dummy_state)[0].shape
    else:
        obs_shape = base_env.featurize_state_mdp(dummy_state)[0].shape
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

class LstmStateResetCallback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        self.model.reset_states()


def _pad(sequences, maxlen=None, default=0):
    if not maxlen:
        maxlen = max([len(seq) for seq in sequences])
    for seq in sequences:
        pad_len = maxlen - len(seq)
        seq.extend([default]*pad_len)
    return sequences

def load_data(bc_params, verbose):
    processed_trajs, _ = get_trajs_from_data(**bc_params["data_params"], silent=not verbose)
    inputs, targets = processed_trajs["ep_observations"], processed_trajs["ep_actions"]

    if bc_params['use_lstm']:
        seq_lens = np.array([len(seq) for seq in inputs])
        seq_padded = _pad(inputs, default=np.zeros((len(inputs[0][0],))))
        targets_padded = _pad(targets, default=np.zeros(1))
        seq_t = np.dstack(seq_padded).transpose((2, 0, 1))
        targets_t = np.dstack(targets_padded).transpose((2, 0, 1))
        return seq_t, seq_lens, targets_t
    else:
        return np.vstack(inputs), None, np.vstack(targets)

def build_bc_model(use_lstm=True, eager=False, lossless_feature=False, **kwargs):
    if not eager:
        tf.compat.v1.disable_eager_execution()

    if lossless_feature:
        return _build_lossless_model(**kwargs)
    else:
        if use_lstm:
            return _build_lstm_model(**kwargs)
        else:
            return _build_model(**kwargs)

def initialized_bc_model(model_dir, bc_params):
    model = build_bc_model(**bc_params)
    save_bc_model(model_dir, model, bc_params)
    return model

def train_bc_model(model_dir, bc_params, verbose=False, preprocessed_data=None):
    # The preprocessed_data is added to allow direct feed of data into the bc model
    if preprocessed_data:
        inputs = preprocessed_data['inputs']
        seq_lens = preprocessed_data['seq_lens']
        targets = preprocessed_data['targets']
        # whether to use test dataset
        test_eval = True
        inputs_test = preprocessed_data['inputs_test']
        seq_lens_test = preprocessed_data['seq_lens_test']
        targets_test = preprocessed_data['targets_test']
    else:
        inputs, seq_lens, targets = load_data(bc_params, verbose)
        test_eval = False
        inputs_test, seq_lens_test, targets_test = None, None, None

    training_params = bc_params["training_params"]


    if training_params['use_class_weights']:
        # Get class counts, and use these to compute balanced class weights
        classes, counts = np.unique(targets.flatten(), return_counts=True)
        weights = sum(counts) / counts
        class_weights = dict(zip(classes, weights))
    else:
        # Default is uniform class weights
        class_weights = None

    if "seed" in training_params and training_params["seed"] != 0:
        seed = training_params["seed"]
        print("USING SPECIFIC SEED", seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
    else:
        print("DOES NOT USE CUSTOM SEED")

    # Retrieve un-initialized keras model
    model = build_bc_model(**bc_params, max_seq_len=np.max(seq_lens))

    # Initialize the model
    # Note: have to use lists for multi-output model support and not dicts because of tensorlfow 2.0.0 bug
    if bc_params['use_lstm']:
        loss = [keras.losses.SparseCategoricalCrossentropy(from_logits=True), None, None]
        metrics = [["sparse_categorical_accuracy"], [], []]
    else:
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ["sparse_categorical_accuracy"]
    model.compile(optimizer=keras.optimizers.Adam(training_params["learning_rate"]),
                  loss=loss,
                  metrics=metrics)


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

    ## Actually train our model

    # Create input dict for both models
    N = inputs.shape[0]
    inputs = { "Overcooked_observation" : inputs }
    targets = { "logits" : targets }

    # Inputs unique to lstm model
    if bc_params['use_lstm']:
        inputs['seq_in'] = seq_lens
        inputs['hidden_in'] = np.zeros((N, bc_params['cell_size']))
        inputs['memory_in'] = np.zeros((N, bc_params['cell_size']))

    # Batch size doesn't include time dimension (seq_len) so it should be smaller for rnn model
    batch_size = 1 if bc_params['use_lstm'] else training_params['batch_size']
    best_eval_score_so_far = 0

    eval_scores = []
    eval_scores_ses = []
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    initial_train_loss, initial_train_accuracy = model.evaluate(inputs, targets, verbose=0)
    train_losses.append(initial_train_loss)
    train_accuracies.append(initial_train_accuracy)

    initial_eval_score, initial_eval_score_se = evaluate_bc_model(model, bc_params)
    eval_scores.append(initial_eval_score)
    eval_scores_ses.append(initial_eval_score_se)

    if test_eval:
        initial_test_loss, initial_test_accuracy = model.evaluate(inputs_test, targets_test, verbose=0)
        test_losses.append(initial_test_loss)
        test_accuracies.append(initial_test_accuracy)

    # save the initial model so we always have something
    save_bc_model(model_dir, model, bc_params)

    for i in range(training_params['epochs']//training_params['slice_freq']):
        print("slice", i)
        print("EVAL_SCORES", eval_scores)
        print("EVAL_SCORES_SES", eval_scores_ses)
        print("TRAIN_LOSSES", train_losses)
        print("TRAIN_ACCURACIES", train_accuracies)
        print("TEST_LOSSES", test_losses)
        print("TEST_ACCURACIES", test_accuracies)
        train_history = model.fit(inputs, targets, callbacks=callbacks, batch_size=batch_size,
                    epochs=training_params['slice_freq'], validation_split=training_params["validation_split"],
                    class_weight=class_weights,
                    verbose=2 if verbose else 0)

        train_losses.append(train_history.history['loss'][-1])
        train_accuracies.append(train_history.history['sparse_categorical_accuracy'][-1])

        # calculate losses on the test set if using test_eval
        if test_eval:
            test_res= model.evaluate(inputs_test, targets_test, verbose=0)
            test_losses.append(test_res[0])
            test_accuracies.append(test_res[1])

        # run some evaluation games to get validation rewards
        eval_score, eval_score_se = evaluate_bc_model(model, bc_params)
        # save the best bc model
        if eval_score > best_eval_score_so_far:
            # Save the model
            print("new best:", eval_score)
            save_bc_model(model_dir, model, bc_params)
        best_eval_score_so_far = max(eval_score, best_eval_score_so_far)
        eval_scores.append(eval_score)
        eval_scores_ses.append(eval_score_se)


    print("FINAL EVAL_SCORES", eval_scores)
    print("FINAL EVAL_SCORES_SES", eval_scores_ses)
    print("FINAL TRAIN_LOSSES", train_losses)
    print("FINAL TRAIN_ACCURACIES", train_accuracies)
    print("FINAL TEST_LOSSES", test_losses)
    print("FINAL TEST_ACCURACIES", test_accuracies)

    training_stats = {
        "seed": seed,
        "slice_freq": training_params['slice_freq'],
        "val_reward": eval_scores,
        "val_reward_se": eval_scores_ses,
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies
    }

    # save the eval scores as a pickle file
    with open(model_dir + '/training_stats.pickle', 'wb') as handle:
        pickle.dump(training_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    best_model = load_bc_model(model_dir)
    return best_model
    


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
    model = keras.models.load_model(model_dir, custom_objects={ 'tf' : tf })
    with open(os.path.join(model_dir, "metadata.pickle"), "rb") as f:
        bc_params = pickle.load(f)
    return model, bc_params

def load_bc_agent(model_dir, agent_index, featurize_fn=None):
    model, bc_params = load_bc_model(model_dir)
    policy = BehaviorCloningPolicy.from_model(model, bc_params, stochastic=True)
    if featurize_fn is None:
        layout_name = bc_params["mdp_params"]["layout_name"]
        dummy_mdp = OvercookedGridworld.from_layout_name(layout_name)
        mlam = MediumLevelActionManager.from_pickle_or_compute(dummy_mdp, NO_COUNTERS_PARAMS, force_compute=True)
        featurize_fn = lambda state: dummy_mdp.featurize_state(state, mlam)

    bc_agent = RlLibAgent(policy, agent_index=agent_index, featurize_fn=featurize_fn)
    return bc_agent


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
    lossless_feature = bc_params['lossless_feature']

    # Get reference to state encoding function used by bc agents, with compatible signature
    base_ae = _get_base_ae(bc_params)
    base_env = base_ae.env
    def featurize_fn(state):
        if lossless_feature:
            o1, o2 = base_env.lossless_state_encoding_mdp(state)
            return [np.cast['float16'](o1), np.cast['float16'](o2)]
        else:
            return base_env.featurize_state_mdp(state)

    # Wrap Keras models in rllib policies
    agent_0_policy = BehaviorCloningPolicy.from_model(model, bc_params, stochastic=True)
    agent_1_policy = BehaviorCloningPolicy.from_model(model, bc_params, stochastic=True)

    # Compute the results of the rollout(s)
    results = evaluate(eval_params=evaluation_params,
                       mdp_params=mdp_params,
                       outer_shape=None,
                       agent_0_policy=agent_0_policy,
                       agent_1_policy=agent_1_policy,
                       agent_0_featurize_fn=featurize_fn,
                       agent_1_featurize_fn=featurize_fn)

    # Compute the average sparse return obtained in each rollout
    reward = np.mean(results['ep_returns'])
    std = np.std(results['ep_returns'])/np.sqrt(len(results['ep_returns']))
    return (reward, std)

def _build_model(observation_shape, action_shape, mlp_params, **kwargs):
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

DEFAULT_LOSSLESS_BC_PARAMS = {
    "NUM_HIDDEN_LAYERS": 3,
    "SIZE_HIDDEN_LAYERS": 64,
    "NUM_FILTERS": 25,
    "NUM_CONV_LAYERS": 3
}

# largely the same as ppo_rllib. #TODO: merge the two function
# If you are getting an cuDNN error, look into https://github.com/tensorflow/tensorflow/issues/24828
# the best solution I found was https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-742469380
def _build_lossless_model(observation_shape, action_shape, **kwargs):
    ## Inputs
    num_hidden_layers = DEFAULT_LOSSLESS_BC_PARAMS['NUM_HIDDEN_LAYERS']
    size_hidden_layers = DEFAULT_LOSSLESS_BC_PARAMS['SIZE_HIDDEN_LAYERS']
    num_filters = DEFAULT_LOSSLESS_BC_PARAMS['NUM_FILTERS']
    num_convs = DEFAULT_LOSSLESS_BC_PARAMS['NUM_CONV_LAYERS']

    ## Create graph of custom network. It will under a shared tf scope such that all agents
    ## use the same model
    inputs = keras.Input(shape=observation_shape, name="Overcooked_observation") # matching this function's signature
    out = inputs

    # Apply initial conv layer with a larger kenel (why?)
    if num_convs > 0:
        out = keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.leaky_relu,
            name='conv_initial'
        )(out)

    # Apply remaining conv layers, if any
    for i in range(0, num_convs - 1):
        padding = 'same' if i < num_convs - 2 else 'valid'
        out = keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=[3, 3],
            padding=padding,
            activation=tf.nn.leaky_relu,
            name='conv_{}'.format(i)
        )(out)

    # Apply dense hidden layers, if any
    out = keras.layers.Flatten()(out)
    for i in range(num_hidden_layers):
        out = keras.layers.Dense(
            size_hidden_layers,
            name=f'dense_{i}',
        )(out)
        out = keras.layers.LeakyReLU()(out)

    # Linear last layer for action distribution logits
    layer_out = keras.layers.Dense(
        action_shape[0], # matching this function's signature
        name="logits",  # matching this function's signature
    )(out)

    # Linear last layer for value function branch of model, never used here
    value_out = keras.layers.Dense(1, name='value_head')(out)

    base_model = keras.Model(inputs=inputs, outputs=layer_out)
    return base_model

# # temporarily deprecated due to uncentainty about compatibility
# def _build_lstm_model(observation_shape, action_shape, mlp_params, cell_size, max_seq_len=20, **kwargs):
#     ## Inputs
#     obs_in = keras.Input(shape=(None, *observation_shape), name="Overcooked_observation")
#     seq_in = keras.Input(shape=(), name="seq_in", dtype=tf.int32)
#     h_in = keras.Input(shape=(cell_size,), name="hidden_in")
#     c_in = keras.Input(shape=(cell_size,), name="memory_in")
#     x = obs_in
#
#     ## Build fully connected layers
#     assert len(mlp_params["net_arch"]) == mlp_params["num_layers"], "Invalid Fully Connected params"
#
#     for i in range(mlp_params["num_layers"]):
#         units = mlp_params["net_arch"][i]
#         x = keras.layers.TimeDistributed(keras.layers.Dense(units, activation="relu", name="fc_{0}".format(i)))(x)
#
#     mask = keras.layers.Lambda(lambda x : tf.sequence_mask(x, maxlen=max_seq_len))(seq_in)
#
#     ## LSTM layer
#     lstm_out, h_out, c_out = keras.layers.LSTM(cell_size, return_sequences=True, return_state=True, stateful=False, name="lstm")(
#         inputs=x,
#         mask=mask,
#         initial_state=[h_in, c_in]
#     )
#
#     ## output layer
#     logits = keras.layers.TimeDistributed(keras.layers.Dense(action_shape[0]), name="logits")(lstm_out)
#
#     return keras.Model(inputs=[obs_in, seq_in, h_in, c_in], outputs=[logits, h_out, c_out])



################
# Rllib Policy #
################

class NullContextManager:
    """
    No-op context manager that does nothing
    """
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass

class TfContextManager:
    """
    Properly sets the execution graph and session of the keras backend given a "session" object as input

    Used for isolating tf execution in graph mode. Do not use with eager models or with eager mode on
    """
    def __init__(self, session):
        self.session = session
    def __enter__(self):
        self.ctx = self.session.graph.as_default()
        self.ctx.__enter__()
        set_session(self.session)
    def __exit__(self, *args):
        self.ctx.__exit__(*args)

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
            - eager (bool)                      Whether the model should run in eager (or graph) mode. Overrides bc_params['eager'] if present
        """
        super(BehaviorCloningPolicy, self).__init__(observation_space, action_space, config)

        if 'bc_model' in config and config['bc_model']:
            assert 'bc_params' in config, "must specify params in addition to model"
            assert issubclass(type(config['bc_model']), keras.Model), "model must be of type keras.Model"
            model, bc_params = config['bc_model'], config['bc_params']
        else:
            assert 'model_dir' in config, "must specify model directory if model not specified"
            model, bc_params = load_bc_model(config['model_dir'])
        
        # Save the session that the model was loaded into so it is available at inference time if necessary
        self._sess = get_session()
        self._setup_shapes()

        # Basic check to make sure model dimensions match
        assert self.observation_shape == bc_params['observation_shape']
        assert self.action_shape == bc_params['action_shape']

        self.model = model
        self.stochastic = config['stochastic']
        self.use_lstm = bc_params['use_lstm']
        self.cell_size = bc_params['cell_size']
        self.eager = config['eager'] if 'eager' in config else bc_params['eager']
        self.context = self._create_execution_context()

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
        with self.context:
            action_logits, states = self._forward(obs_batch, state_batches)
        
        # Softmax in numpy to convert logits to probabilities
        action_probs = softmax(action_logits)
        if self.stochastic:
            # Sample according to action_probs for each row in the output
            actions = np.array([np.random.choice(self.action_shape[0], p=action_probs[i]) for i in range(len(action_probs))])
        else:
            actions = np.argmax(action_logits, axis=1)

        return actions,  states, { "action_dist_inputs" : action_logits }

    def get_initial_state(self):
        """
        Returns the initial hidden and memory states for the model if it is recursive

        Note, this shadows the rllib.Model.get_initial_state function, but had to be added here as
        keras does not allow mixins in custom model classes

        Also note, either this function or self.model.get_initial_state (if it exists) must be called at 
        start of an episode
        """
        if self.use_lstm:
            return [np.zeros(self.cell_size,), np.zeros(self.cell_size,)]
        return []


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

    def _forward(self, obs_batch, state_batches):
        if self.use_lstm:
            obs_batch = np.expand_dims(obs_batch, 1)
            seq_lens = np.ones(len(obs_batch))
            model_out = self.model.predict([obs_batch, seq_lens] + state_batches)
            logits, states = model_out[0], model_out[1:]
            logits = logits.reshape((logits.shape[0], -1))
            return logits, states
        else:
            return self.model.predict(obs_batch), []

    def _create_execution_context(self):
        """
        Creates a private execution context for the model 

        Necessary if using with rllib in order to isolate this policy model from others
        """
        if self.eager:
            return NullContextManager()
        return TfContextManager(self._sess)


if __name__ == "__main__":
    params = get_default_bc_params()
    model = train_bc_model(os.path.join(BC_SAVE_DIR, 'default'), params, verbose=True)
    # Evaluate our model's performance in a rollout
    evaluate_bc_model(model, params)
