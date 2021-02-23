import os
from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.static import HUMAN_DATA_PATH
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS
from overcooked_ai_py.mdp.actions import Action

BC_SAVE_DIR = os.path.join(DATA_DIR, "bc_runs")

DEFAULT_DATA_PARAMS = {
    "train_mdps": ["cramped_room"],
    "ordered_trajs": False,
    "processed" : True,
    "data_path": HUMAN_DATA_PATH, 
    "orders_processing_function": "mdp.multi_hot_orders_encoding",
    "action_processing_function": "mdp.sparse_categorical_joint_action_encoding",
    # "env.featurize_state_mdp" is possible for layouts with only onions
    "state_processing_function": "env.lossless_state_encoding_mdp", 
    "from_dataframe": True
}

DEFAULT_MLP_PARAMS = {
    # Number of fully connected layers to use in our network
    "num_layers" : 2,
    # Each int represents a layer of that hidden size
    "net_arch" : [64, 64],
}

DEFAULT_CNN_PARAMS = {
    "num_filters": 25,
    "num_conv_layers": 3,
    # change to False when using env.featurize_state_mdp
    "use_cnn": True
}

DEFAULT_TRAINING_PARAMS = {
    "epochs" : 100,
    "validation_split" : 0.15,
    "batch_size" : 64,
    "learning_rate" : 1e-3,
    "use_class_weights" : False,
    "actions_loss_coeff": 1.0,
    "orders_loss_coeff": 1.0,
    "orders_loss": "BinaryCrossentropy",
    "actions_loss": "SparseCategoricalCrossentropy"
}

DEFAULT_EVALUATION_PARAMS = {
    "ep_length" : 400,
    "num_games" : 1,
    "display" : False,
    "orders_metrics": "binary_accuracy",
    "actions_metrics": "sparse_categorical_accuracy"
}

DEFAULT_BC_PARAMS = {
    "eager" : True,
    "use_lstm" : False,
    "cell_size" : 256,
    "data_params": DEFAULT_DATA_PARAMS,
    "mdp_params": {'layout_name': "cramped_room"},
    "env_params": DEFAULT_ENV_PARAMS,
    "mdp_fn_params": {},
    "mlp_params" : DEFAULT_MLP_PARAMS,
    "cnn_params": DEFAULT_CNN_PARAMS,
    "training_params" : DEFAULT_TRAINING_PARAMS,
    "evaluation_params" : DEFAULT_EVALUATION_PARAMS,
    "action_shape" :  (len(Action.ALL_ACTIONS), ),
    # turning it to True results in predicting both next action and ordered recipes 
    "predict_orders": False
    }

