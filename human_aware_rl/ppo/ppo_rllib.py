from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
import numpy as np
import tensorflow as tf
import gym

AUXILLARY_INFO_NAME = "auxillary_info"
OBSERVATIONS_NAME = "observations"

class RllibPPOModel(TFModelV2):
    """
    Model that will map environment states to action probabilities. Will be shared across agents
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):

        super(RllibPPOModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        # params we got to pass in from the call to "run"
        custom_params = model_config["custom_options"]


        ## Parse custom network params
        num_hidden_layers = custom_params["NUM_HIDDEN_LAYERS"]
        size_hidden_layers = custom_params["SIZE_HIDDEN_LAYERS"]
        num_filters = custom_params["NUM_FILTERS"]
        num_convs = custom_params["NUM_CONV_LAYERS"]
        d2rl = custom_params["D2RL"]
        assert type(d2rl) == bool
        
        if hasattr(obs_space, "original_space"):
            obs_space = obs_space.original_space

        auxillary_info_input_layer = None
        if isinstance(obs_space, gym.spaces.Dict):
            auxillary_info = obs_space.spaces.get(AUXILLARY_INFO_NAME)
            obs_space = obs_space.spaces[OBSERVATIONS_NAME]
            if auxillary_info is not None:
                auxillary_info_input_layer = tf.keras.Input(shape=auxillary_info.shape, name=AUXILLARY_INFO_NAME)
        
        obs_input_layer = tf.keras.Input(shape=obs_space.shape, name=OBSERVATIONS_NAME)
        out = obs_input_layer


        ## Create graph of custom network. It will under a shared tf scope such that all agents
        ## use the same model

        self.inputs = [obs_input_layer]

        if auxillary_info_input_layer is not None:
            self.inputs.append(auxillary_info_input_layer)
        # Apply initial conv layer with a larger kernel (why?)
        if num_convs > 0:
            out = tf.keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.leaky_relu,
                name="conv_initial"
            )(out)

        # Apply remaining conv layers, if any
        for i in range(0, num_convs-1):
            padding = "same" if i < num_convs - 2 else "valid"
            out = tf.keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[3, 3],
                padding=padding,
                activation=tf.nn.leaky_relu,
                name="conv_{}".format(i)
            )(out)
        
        # Apply dense hidden layers, if any
        out = tf.keras.layers.Flatten()(out)

        if auxillary_info_input_layer is not None:
            flat_auxillary_info = tf.keras.layers.Flatten()(auxillary_info_input_layer)
            out = tf.keras.layers.Concatenate()([out, flat_auxillary_info])

        conv_out = out
        for i in range(num_hidden_layers):
            if i > 0 and d2rl:
                out = tf.keras.layers.Concatenate()([out, conv_out])
            out = tf.keras.layers.Dense(size_hidden_layers)(out)
            out = tf.keras.layers.LeakyReLU()(out)

        # Linear last layer for action distribution logits
        layer_out = tf.keras.layers.Dense(self.num_outputs)(out)

        # Linear last layer for value function branch of model
        value_out = tf.keras.layers.Dense(1)(out)

        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.register_variables(self.base_model.variables)


    def forward(self, input_dict, state=None, seq_lens=None):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class RllibLSTMPPOModel(RecurrentTFModelV2):
    """
    Model that will map encoded environment observations to action logits
                            (optional) auxillary_info       
                                       ||                |_______|
                                       \/            /-> | value |
             ___________     _________     ________ /    |_______|  
    state -> | conv_net | -> | fc_net | -> | lstm | 
             |__________|    |________|    |______| \\    |_______________|
                                           /    \\   \\-> | action_logits |
                                          h_in   c_in     |_______________|
                                         
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super(RllibLSTMPPOModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        # params we passed in from rllib client
        custom_params = model_config["custom_options"]

        ## Parse custom network params
        num_hidden_layers = custom_params["NUM_HIDDEN_LAYERS"]
        size_hidden_layers = custom_params["SIZE_HIDDEN_LAYERS"]
        num_filters = custom_params["NUM_FILTERS"]
        num_convs = custom_params["NUM_CONV_LAYERS"]
        cell_size = custom_params["CELL_SIZE"]
        
        ### Create graph of the model ###
        # take obs_space shape before splitting obs_space into obs_space and auxillary_info as flattened_obs_dim still works here if obs_space is Dict
        flattened_obs_dim = np.prod(obs_space.shape)
        # Need an extra batch dimension (None) for time dimension
        flattened_obs_inputs = tf.keras.Input(shape=(None, flattened_obs_dim), name="input")
        
        if hasattr(obs_space, "original_space"):
            original_obs_space = obs_space.original_space
        else:
            original_obs_space = obs_space
        
        if isinstance(original_obs_space, gym.spaces.Dict):
            auxillary_info_space = original_obs_space.spaces.get(AUXILLARY_INFO_NAME, None)
            obs_space = original_obs_space.spaces[OBSERVATIONS_NAME]
        else:
            auxillary_info_space = None
            obs_space = original_obs_space

        if auxillary_info_space is None:
            obs_input = flattened_obs_inputs
            auxillary_info_input = None
        else:
            def divide_input_into_obs_and_auxillary_info(flattened_obs_inputs, auxillary_info_space, original_obs_space):
                is_auxillary_info_first = list(original_obs_space.spaces.keys())[0] == AUXILLARY_INFO_NAME
                auxillary_info_size = np.prod(auxillary_info_space.shape)

                if is_auxillary_info_first:
                    first_slices_elem_size = auxillary_info_size
                else:
                    first_slices_elem_size = flattened_obs_dim - auxillary_info_size
                
                sliced_data = tf.keras.layers.Lambda(lambda x: (x[:,:,:first_slices_elem_size], x[:,:,first_slices_elem_size:]))(flattened_obs_inputs)

                if is_auxillary_info_first:
                    (auxillary_info_input, obs_input) = sliced_data 
                else:
                    (obs_input, auxillary_info_input) = sliced_data
                    
                return obs_input, auxillary_info_input
                
            obs_input, auxillary_info_input = divide_input_into_obs_and_auxillary_info(flattened_obs_inputs, auxillary_info_space, original_obs_space)

        lstm_h_in = tf.keras.Input(shape=(cell_size,), name="h_in")
        lstm_c_in = tf.keras.Input(shape=(cell_size,), name="c_in")
        seq_in = tf.keras.Input(shape=(), name="seq_in", dtype=tf.int32)
        inputs = [flattened_obs_inputs, seq_in, lstm_h_in, lstm_c_in]

        # Restore initial observation shape
        obs_input = tf.keras.layers.Reshape(target_shape=(-1, *obs_space.shape))(obs_input)
        out = obs_input

        ## Initial "vision" network

        # Apply initial conv layer with a larger kenel (why?)
        if num_convs > 0:
            out = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.leaky_relu,
                name="conv_initial"
            ))(out)

        # Apply remaining conv layers, if any
        for i in range(0, num_convs-1):
            padding = "same" if i < num_convs - 2 else "valid"
            out = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[3, 3],
                padding=padding,
                activation=tf.nn.leaky_relu,
                name="conv_{}".format(i)
            ))(out)
        
        # Flatten spatial features
        out = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(out)
        
        if auxillary_info_input is not None:
            # NOTE: auxillary_info_input is already flat
            out = tf.keras.layers.Concatenate()([out, auxillary_info_input])

        # Apply dense hidden layers, if any
        for i in range(num_hidden_layers):
            out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
                units=size_hidden_layers, 
                activation=tf.nn.leaky_relu, 
                name="fc_{0}".format(i)
            ))(out)

        ## LSTM network
        lstm_out, h_out, c_out = tf.keras.layers.LSTM(cell_size, return_sequences=True, return_state=True, name="lstm")(
            inputs=out,
            mask=tf.sequence_mask(seq_in),
            initial_state=[lstm_h_in, lstm_c_in]
        )

        # Linear last layer for action distribution logits
        layer_out = tf.keras.layers.Dense(self.num_outputs, name="logits")(lstm_out)

        # Linear last layer for value function branch of model
        value_out = tf.keras.layers.Dense(1, name="values")(lstm_out)

        self.cell_size = cell_size
        self.base_model = tf.keras.Model(
            inputs=inputs,
            outputs=[layer_out, value_out, h_out, c_out]
        )
        self.register_variables(self.base_model.variables)


    def forward_rnn(self, inputs, state, seq_lens):
        """
        Run the forward pass of the model

        Arguments:
            inputs: np.array of shape [BATCH, T, obs_shape]
            state:  list of np.arrays [h_in, c_in] each of shape [BATCH, self.cell_size]
            seq_lens: np.array of shape [BATCH] where the ith element is the length of the ith sequence

        Output:
            model_out: tensor of shape [BATCH, T, self.num_outputs] representing action logits
            state: list of tensors [h_out, c_out] each of shape [BATCH, self.cell_size]
        """
        model_out, self._value_out, h_out, c_out = self.base_model([inputs, seq_lens, state])

        return model_out, [h_out, c_out]

    def value_function(self):
        """
        Returns a tensor of shape [BATCH * T] representing the value function for the most recent forward pass
        """
        return tf.reshape(self._value_out, [-1])

    def get_initial_state(self):
        """
        Returns the initial hidden state for the LSTM
        """
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]