import os, pickle, copy
from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.compat.v1.keras.backend import set_session, get_session
from human_aware_rl.human.process_dataframes import get_trajs_from_data

from human_aware_rl.rllib.rllib import RlLibAgent, evaluate
from human_aware_rl.rllib.utils import softmax, sigmoid, get_base_ae, get_encoding_function
from ray.rllib.policy import Policy as RllibPolicy
from human_aware_rl.imitation.default_bc_params import BC_SAVE_DIR, DEFAULT_DATA_PARAMS, \
    DEFAULT_MLP_PARAMS, DEFAULT_TRAINING_PARAMS, DEFAULT_EVALUATION_PARAMS, DEFAULT_BC_PARAMS, \
    DEFAULT_CNN_PARAMS

OBS_INPUT_NAME = "Overcooked_observation"
EP_LEN_INPUT_NAME = "seq_in"
HIDDEN_INPUT_NAME = "hidden_in"
MEMORY_INPUT_NAME = "memory_in"
ACTION_OUTPUT_NAME = "logits"
ORDERS_OUTPUT_NAME = "orders_logits"

# Boolean indicating whether all param dependencies have been loaded. Used to prevent re-loading unnecessarily
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
    encoding_f = get_encoding_function(bc_params["data_params"]["state_processing_function"], env=base_env)
    obs_shape = encoding_f(dummy_state)[0].shape
    return obs_shape

def _get_orders_shape(bc_params):
    """
    Helper function for creating a dummy environment from "mdp_params" and "env_params" specified
    in bc_params and returning the shape of the order space
    NOTE: does work when output logit layer shape is same as orders shape (does not work for sparse encodings)
    """
    base_ae = _get_base_ae(bc_params)
    base_env = base_ae.env
    dummy_state = base_env.mdp.get_standard_start_state()
    
    encoding_f = get_encoding_function(bc_params["data_params"]["orders_processing_function"], env=base_env)
    orders_shape = encoding_f(dummy_state)[0].shape
    return orders_shape

# For lazing loading the default params. Prevents loading on every import of this module 
def get_default_bc_params():
    global _params_initalized, DEFAULT_BC_PARAMS
    if not _params_initalized:
        DEFAULT_BC_PARAMS['observation_shape'] = _get_observation_shape(DEFAULT_BC_PARAMS)
        DEFAULT_BC_PARAMS['orders_shape'] = _get_orders_shape(DEFAULT_BC_PARAMS)
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
    predict_orders = bc_params.get("predict_orders", DEFAULT_BC_PARAMS["predict_orders"])
    
    processed_trajs, _ = get_trajs_from_data(**bc_params["data_params"], silent=not verbose, include_orders=predict_orders)
    observations, actions = processed_trajs["ep_observations"], processed_trajs["ep_actions"],
    if predict_orders:
        orders = processed_trajs["ep_orders"]
    else:
        orders = None
    
    if bc_params['use_lstm']:
        ep_lens = np.array([len(ob) for ob in observations])
        observations = _pad(observations, default=np.zeros(bc_params['observation_shape']))
        observations = np.moveaxis(np.stack(np.moveaxis(observations, 0, 2)), 2, 0)

        actions = _pad(actions, default=np.zeros(1))
        actions = np.dstack(actions).transpose((2, 0, 1))
    
        if predict_orders:
            orders = _pad(orders, default=np.zeros(bc_params["orders_shape"]))
            orders = np.moveaxis(np.stack(np.moveaxis(orders, 0, 2)), 2, 0)
    else:
        observations = np.vstack(observations)
        ep_lens = None # not used in non recurrent networks
        actions = np.vstack(actions)
        if predict_orders: orders = np.vstack(orders)
    
    return observations, ep_lens, actions, orders
    

def build_bc_model(use_lstm=True, eager=False, **kwargs):
    if not eager:
        tf.compat.v1.disable_eager_execution()
    if use_lstm:
        return _build_lstm_model(**kwargs)
    else:
        return _build_model(**kwargs)

def _get_loss(loss_or_name, **kwargs):
    if type(loss_or_name) is str:
        loss = getattr(tf.keras.losses, loss_or_name)(**kwargs)
    else:
        loss = loss_or_name
    return loss

def _bc_model_inputs(bc_params, obs, ep_lens=None):
    N = obs.shape[0]
    inputs = { OBS_INPUT_NAME: obs }
    # Inputs unique to lstm model
    if bc_params['use_lstm']:
        assert ep_lens is not None
        inputs[EP_LEN_INPUT_NAME] = ep_lens
        inputs[HIDDEN_INPUT_NAME] = np.zeros((N, bc_params['cell_size']))
        inputs[MEMORY_INPUT_NAME] = np.zeros((N, bc_params['cell_size']))
    return inputs

def _bc_model_targets(bc_params, acts=None, orders=None):
    targets = {}
    predict_orders = bc_params.get("predict_orders", DEFAULT_BC_PARAMS["predict_orders"])
    targets[ACTION_OUTPUT_NAME] = acts

    if predict_orders:
        assert orders is not None
        targets[ORDERS_OUTPUT_NAME] = orders
    return targets
    
def train_bc_model(model_dir, bc_params, verbose=False, loaded_data=None):
    if loaded_data is None:
        loaded_data = load_data(bc_params, verbose)
    (obs, ep_lens, acts, orders) = loaded_data
    #NOTE: some pointing to default params is to maintain backward compatibility (older version of BC params did not included many of the variables)
    predict_orders = bc_params.get("predict_orders", DEFAULT_BC_PARAMS["predict_orders"])


    training_params = bc_params["training_params"]
    eval_params = bc_params["evaluation_params"]
    if training_params['use_class_weights']:
        # Get class counts, and use these to compute balanced class weights
        classes, counts = np.unique(acts.flatten(), return_counts=True)
        weights = sum(counts) / counts
        class_weights = dict(zip(classes, weights))
    else:
        # Default is uniform class weights
        class_weights = None

    # Retrieve un-initialized keras model
    model = build_bc_model(**bc_params, max_seq_len=np.max(ep_lens))
    # Initialize the model

    metrics = {}
    losses = {}
    loss_weights = {}
    metrics[ACTION_OUTPUT_NAME] = eval_params.get("actions_metrics", DEFAULT_EVALUATION_PARAMS["actions_metrics"])
    losses[ACTION_OUTPUT_NAME] =  _get_loss(training_params.get("actions_loss", DEFAULT_TRAINING_PARAMS["actions_loss"]), from_logits=True)
    loss_weights[ACTION_OUTPUT_NAME] = training_params.get("actions_loss_coeff", 1.0)

    if predict_orders:
        metrics[ORDERS_OUTPUT_NAME] = eval_params["orders_metrics"]
        losses[ORDERS_OUTPUT_NAME] =  _get_loss(training_params["orders_loss"], from_logits=True)
        loss_weights[ORDERS_OUTPUT_NAME] = training_params.get("orders_loss_coeff", 1.0)
        
    model.compile(optimizer=keras.optimizers.Adam(training_params["learning_rate"]),
                  loss=losses,
                  metrics=metrics,
                  loss_weights=loss_weights)

    # Customize our training loop with callbacks
    callbacks = [
        # Early terminate training if loss doesn't improve for "patextend([]
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
    inputs = _bc_model_inputs(bc_params, obs, ep_lens)
    targets = _bc_model_targets(bc_params, acts, orders)
    
    # Batch size doesn't include time dimension (seq_len) so it should be smaller for rnn model
    batch_size = 1 if bc_params['use_lstm'] else training_params['batch_size']
    model.fit(inputs, targets, callbacks=callbacks, batch_size=batch_size, 
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
    model = keras.models.load_model(model_dir, custom_objects={ 'tf' : tf })
    with open(os.path.join(model_dir, "metadata.pickle"), "rb") as f:
        bc_params = pickle.load(f)
    return model, bc_params

def evaluate_bc_model_metrics(model, bc_params, use_training_data=False, use_validation_data=True, data=None):
    # evaluate BC model on its metrics i.e. accuracy
    if data is None:
        data = load_data(bc_params, verbose=False)

    (obs, ep_lens, acts, orders) = data
    training_params = bc_params["training_params"]
    training_samples_num = int((1-training_params["validation_split"])*len(obs))
    
    if use_validation_data and use_training_data:
        pass
    elif use_validation_data and not use_training_data:
        obs = obs[training_samples_num:]
        if ep_lens is not None: ep_lens = ep_lens[training_samples_num:]
        if acts is not None: acts = acts[training_samples_num:]
        if orders is not None: orders = orders[training_samples_num:]
    elif use_training_data and not use_validation_data:
        obs = obs[:training_samples_num]
        if ep_lens is not None: ep_lens = ep_lens[:training_samples_num]
        if acts is not None: acts = acts[:training_samples_num]
        if orders is not None: orders = orders[:training_samples_num]
    else:
        raise ValueError('At least one of the variables "use_training_data" or "use_validation_data" needs to be set to True')
    inputs = _bc_model_inputs(bc_params, obs, ep_lens)
    targets = _bc_model_targets(bc_params, acts, orders)
    
    # Batch size doesn't include time dimension (seq_len) so it should be smaller for rnn model
    batch_size = 1 if bc_params['use_lstm'] else training_params['batch_size']
    # NOTE: 2 lines below has same effect as return_dict model.evaluate(inputs, targets, batch_size=batch_size, return_dict=True),
    #  but it is not compatible with older tf versions
    result = model.evaluate(inputs, targets, batch_size=batch_size)
    return {metric_name: result[i] for i, metric_name in enumerate(model.metrics_names)}

def evaluate_bc_model(model, bc_params):
    """
    Creates an AgentPair object containing two instances of BC Agents, whose policies are specified by `model`. Runs
    a rollout using AgentEvaluator class in an environment specified by bc_params

    Arguments

        - model (tf.keras.Model)        A function that maps featurized overcooked states to action logits
        - bc_params (dict)              Specifies the environment in which to evaluate the agent (i.e. layout, reward_shaping_param)
                                            as well as the configuration for the rollout (rollout_length)

    Returns

        - reward (int)                  Total sparse reward achieved by AgentPair during rollout
    """
    evaluation_params = bc_params['evaluation_params']
    mdp_params = bc_params['mdp_params']

    # Get reference to state encoding function used by bc agents, with compatible signature
    base_ae = _get_base_ae(bc_params)
    base_env = base_ae.env
    featurize_fn = get_encoding_function(bc_params["data_params"]["state_processing_function"], env=base_env)

    # Wrap Keras models in rllib policies
    policies = [BehaviorCloningPolicy.from_model(model, bc_params, stochastic=True), 
        BehaviorCloningPolicy.from_model(model, bc_params, stochastic=True)]
    featurize_fns = [featurize_fn for p in policies]
    # Compute the results of the rollout(s)
    results = evaluate(eval_params=evaluation_params, 
                       mdp_params=mdp_params, 
                       outer_shape=None,
                       policies=policies, 
                       featurize_fns=featurize_fns)

    # Compute the average sparse return obtained in each rollout
    reward = np.mean(results['ep_returns'])
    return reward

def _build_model(observation_shape, action_shape, mlp_params, cnn_params=DEFAULT_CNN_PARAMS, **kwargs):
    ## Inputs
    obs_in = keras.Input(shape=observation_shape, name=OBS_INPUT_NAME)
    x = obs_in
    if cnn_params.get("use_cnn"):
        num_filters = cnn_params["num_filters"]
        num_convs = cnn_params["num_conv_layers"]
        if num_convs > 0:
            x = keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.leaky_relu,
                name="conv_initial"
            )(x)
        # Apply remaining conv layers, if any
        for i in range(0, num_convs-1):
            padding = "same" if i < num_convs - 2 else "valid"
            x = tf.keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[3, 3],
                padding=padding,
                activation=tf.nn.leaky_relu,
                name="conv_{}".format(i)
            )(x)
    
        # Apply dense hidden layers, if any
        x = tf.keras.layers.Flatten()(x)

    ## Build fully connected layers
    assert len(mlp_params["net_arch"]) == mlp_params["num_layers"], "Invalid Fully Connected params"

    for i in range(mlp_params["num_layers"]):
        units = mlp_params["net_arch"][i]
        x = keras.layers.Dense(units, activation="relu", name="fc_{0}".format(i))(x)

    ## output layer
    logits = keras.layers.Dense(action_shape[0], name=ACTION_OUTPUT_NAME)(x)
    outputs = [logits]
    if kwargs.get("predict_orders", DEFAULT_BC_PARAMS["predict_orders"]):
        orders_shape = kwargs["orders_shape"]
        orders_logits = keras.layers.Dense(orders_shape[0], name=ORDERS_OUTPUT_NAME)(x)
        outputs.append(orders_logits)
    
    return keras.Model(inputs=[obs_in], outputs=outputs)

def _build_lstm_model(observation_shape, action_shape, mlp_params, cell_size, max_seq_len=20, cnn_params=DEFAULT_CNN_PARAMS, **kwargs):
    ## Inputs
    obs_in = keras.Input(shape=(None, *observation_shape), name=OBS_INPUT_NAME)
    seq_in = keras.Input(shape=(), name=EP_LEN_INPUT_NAME, dtype=tf.int32)
    h_in = keras.Input(shape=(cell_size,), name=HIDDEN_INPUT_NAME)
    c_in = keras.Input(shape=(cell_size,), name=MEMORY_INPUT_NAME)
    x = obs_in
    if cnn_params.get("use_cnn", DEFAULT_CNN_PARAMS["use_cnn"]): 
        num_filters = cnn_params["num_filters"]
        num_convs = cnn_params["num_conv_layers"]
        if num_convs > 0:
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.leaky_relu,
                name="conv_initial"
            ))(x)

        # Apply remaining conv layers, if any
        for i in range(0, num_convs-1):
            padding = "same" if i < num_convs - 2 else "valid"
            out = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[3, 3],
                padding=padding,
                activation=tf.nn.leaky_relu,
                name="conv_{}".format(i)
            ))(x)
        
        # Flatten spatial features
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

    ## Build fully connected layers
    assert len(mlp_params["net_arch"]) == mlp_params["num_layers"], "Invalid Fully Connected params"

    for i in range(mlp_params["num_layers"]):
        units = mlp_params["net_arch"][i]
        x = keras.layers.TimeDistributed(keras.layers.Dense(units, activation="relu", name="fc_{0}".format(i)))(x)

    mask = keras.layers.Lambda(lambda x : tf.sequence_mask(x, maxlen=max_seq_len))(seq_in)

    ## LSTM layer
    lstm_out, h_out, c_out = keras.layers.LSTM(cell_size, return_sequences=True, return_state=True, stateful=False, name="lstm")(
        inputs=x,
        mask=mask,
        initial_state=[h_in, c_in]
    )

    ## output layer
    logits = keras.layers.TimeDistributed(keras.layers.Dense(action_shape[0]), name=ACTION_OUTPUT_NAME)(lstm_out)
    outputs = [logits, h_out, c_out]
    if kwargs.get("predict_orders", DEFAULT_BC_PARAMS["predict_orders"]):
        orders_shape = kwargs["orders_shape"]
        orders_logits = keras.layers.TimeDistributed(keras.layers.Dense(orders_shape[0]), name=ORDERS_OUTPUT_NAME)(lstm_out)
        outputs.append(orders_logits)

    return keras.Model(inputs=[obs_in, seq_in, h_in, c_in], outputs=outputs)



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
        config (dict)                           Dictionary of relevant bc params
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
        self.model_predicts_orders = bc_params.get("predict_orders", DEFAULT_BC_PARAMS["predict_orders"])
        self.eager = config['eager'] if 'eager' in config else bc_params['eager']
        self.context = self._create_execution_context()
        self.bc_params = bc_params

    def _setup_shapes(self):
        # This is here to make the class compatible with both tuples or gym.Space objs for the spaces
        # Note: action_space = (len(Action.ALL_ACTIONS,)) is technically NOT the action space shape, which would be () since actions are scalars
        self.observation_shape = self.observation_space if type(self.observation_space) == tuple else self.observation_space.shape
        self.action_shape = self.action_space if type(self.action_space) == tuple else (self.action_space.n,)

    @classmethod
    def from_model_dir(cls, model_dir, stochastic=True, **kwargs):
        model, bc_params = load_bc_model(model_dir)
        config = {
            "bc_model" : model,
            "bc_params" : bc_params,
            "stochastic" : stochastic,
            **kwargs
        }
        return cls(bc_params['observation_shape'], bc_params['action_shape'], config)

    @classmethod
    def from_model(cls, model, bc_params, stochastic=True, **kwargs):
        config = {
            "bc_model" : model,
            "bc_params" : bc_params,
            "stochastic" : stochastic,
            **kwargs
        }
        return cls(bc_params["observation_shape"], bc_params["action_shape"], config)
    
    def predict_orders(self, obs_batch, state_batches=None):
        # predict probabilities what orders are ready to fullfill based on the past
        # assumes sigmoid function from logits(for multi-hot encoding)
        assert self.model_predicts_orders
        obs_batch = np.array(obs_batch)
        with self.context:
            action_logits, states, orders_logits = self._forward(obs_batch, state_batches)
        orders_probs = sigmoid(orders_logits)
        return orders_probs
    
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
            action_logits, states, orders_logits = self._forward(obs_batch, state_batches)
        
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
        multiple_outs = self.use_lstm or self.model_predicts_orders

        if self.use_lstm:
            obs_batch = np.expand_dims(obs_batch, 1)
            seq_lens = np.ones(len(obs_batch))
            model_out = self.model.predict(
                {
                    OBS_INPUT_NAME:obs_batch,
                    EP_LEN_INPUT_NAME:seq_lens,
                    HIDDEN_INPUT_NAME:state_batches[0],
                    MEMORY_INPUT_NAME:state_batches[1]
                })
            logits = model_out[0]
            logits = logits.reshape((logits.shape[0], -1))
            states = model_out[1:3]
        else:
            model_out = self.model.predict({OBS_INPUT_NAME:obs_batch})
            if multiple_outs:
                logits = model_out[0]
            else:
                logits = model_out
            states = []
            
        if self.model_predicts_orders:
            orders_logits = model_out[-1]
            if self.use_lstm:
                orders_logits = orders_logits.reshape((orders_logits.shape[0], -1))
        else:
            orders_logits = []

        return logits, states, orders_logits


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
