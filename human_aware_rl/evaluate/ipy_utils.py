from ipywidgets import Layout, Box, IntText, Textarea, Dropdown, Label, Text
from ray.tune.result import DEFAULT_RESULTS_DIR


# the default form item layout
# Feel free to declear your own form_item_layout
DEFAULT_FORM_ITEM_LAYOUT = Layout(
    display='flex',
    flex_flow='row',
    justify_content='space-between'
)

DEFAULT_FORM_OUTER_LAYOUT = Layout(
    display='flex',
    flex_flow='column',
    border='solid 2px',
    align_items='stretch',
    width='50%'
)


def parse_str_to_lst(str_lst_org):
    if str_lst_org != "":
        return str_lst_org.replace(" ", "").split(",")
    else:
        return []


################################
########## IPYTWIDGET ##########
################################

##### evaluation params Boxes #####

def layout_name_box(default_value="cramped_room", form_item_layout=DEFAULT_FORM_ITEM_LAYOUT):
    """
    Return a Box that takes in layout_name as a string
    """
    return Box(
        [Label(value='layout_name'),
         Text(
             value=default_value,
             placeholder='type the layout_name',
             disabled=False
         )],
        layout=form_item_layout
    )


def num_games_box(default_value=40, form_item_layout=DEFAULT_FORM_ITEM_LAYOUT):
    """
    Return a Box that takes in num_game as an integer
    """
    return Box(
        [Label(value='num_games'),
         IntText(
             value=default_value
         )],
        layout=form_item_layout
    )


def mixed_play_box(default_value="True", form_item_layout=DEFAULT_FORM_ITEM_LAYOUT):
    """
    Return a Box that takes in mixed_play as a string-boolean
    """
    return Box(
        [Label(value='mixed_play'),
         Dropdown(
             value=default_value,
             options=['True', 'False']
         )],
        layout=form_item_layout
    )


def ppo_sp_baseline_lst_box(default_value="11, 21, 31, 41", form_item_layout=DEFAULT_FORM_ITEM_LAYOUT):
    """
    Return a Box that takes in ppo_sp_baseline_lst as a string-list
    If empty, no baseline will be used
    """
    return Box(
        [Label(value='ppo_sp_baseline_lst'),
         Textarea(
             value=default_value,
             placeholder='Enter the list of seed of PPO SP baseline, separated by comma',
         )],
        layout=form_item_layout
    )


def include_greedy_human_box(default_value="True", form_item_layout=DEFAULT_FORM_ITEM_LAYOUT):
    """
    Return a Box that takes in include_greedy_human as a string-boolean
    """
    return Box(
        [Label(value='include_greedy_human'),
         Dropdown(
             value=default_value,
             options=['True', 'False']
         )],
        layout=form_item_layout
    )


def custom_ai_cp_path_lst_box(form_item_layout=DEFAULT_FORM_ITEM_LAYOUT):
    """
    Return a Box that takes in a string-list of ai checkpoint paths, separated by comma
    """
    return Box(
        [Label(value='custom_ai_cp_path_lst'),
         Textarea(
             placeholder='Enter relative path from DEFAULT_RESULTS_DIR, separated by comma',
         )],
        layout=form_item_layout
    )


def custom_h_cp_path_lst_box(form_item_layout=DEFAULT_FORM_ITEM_LAYOUT):
    """
    Return a Box that takes in a string-list of h checkpoint paths, separated by comma
    """
    return Box(
        [Label(value='custom_h_cp_path_lst'),
         Textarea(
             placeholder='Enter relative path from DEFAULT_RESULTS_DIR, separated by comma',
         )],
        layout=form_item_layout
    )

def custom_bc_cp_path_lst_box(form_item_layout=DEFAULT_FORM_ITEM_LAYOUT):
    """
    Return a Box that takes in a string-list of behavior cloning checkpoint paths, separated by comma
    """
    return Box(
        [Label(value='custom_bc_cp_path_lst'),
         Textarea(
             placeholder='Enter relative path from bc_runs directory, separated by comma',
         )],
        layout=form_item_layout
    )

##### visualization params Boxes #####

def load_path_box(default_value, form_item_layout=DEFAULT_FORM_ITEM_LAYOUT):
    """
    Return a Box that takes in load_path as a string
    """
    return Box(
        [Label(value='load_path'),
         Text(
             value=default_value,
             placeholder='Enter the load path. Default can be changed at the top',
             disabled=False
         )],
        layout=form_item_layout
    )


def data_timestamp_box(form_item_layout=DEFAULT_FORM_ITEM_LAYOUT):
    """
    Return a Box that takes in data_timestamp as a string
    """
    return Box(
        [Label(value='data_timestamp'),
         Text(
             placeholder='Enter the data_timestamp returned by the run',
             disabled=False
         )],
        layout=form_item_layout
    )


def custom_ai_cp_display_lst_box(form_item_layout=DEFAULT_FORM_ITEM_LAYOUT):
    """
    Return a Box that takes in a string-list of ai checkpoint display, separated by comma
    """
    return Box(
        [Label(value='custom_ai_cp_display_lst'),
         Textarea(
             placeholder='Enter custom display for ai agent, separated by comma',
         )], layout=form_item_layout
    )


def custom_h_cp_display_lst_box(form_item_layout=DEFAULT_FORM_ITEM_LAYOUT):
    """
    Return a Box that takes in a string-list of h checkpoint display, separated by comma
    """
    return Box(
        [Label(value='custom_h_cp_display_lst'),
         Textarea(
             placeholder='Enter display name for h cp agent, separated by comma',
         )], layout=form_item_layout
    )

def custom_bc_cp_display_lst_box(form_item_layout=DEFAULT_FORM_ITEM_LAYOUT):
    """
    Return a Box that takes in a string-list of bc checkpoint display, separated by comma
    """
    return Box(
        [Label(value='custom_bc_cp_display_lst'),
         Textarea(
             placeholder='Enter display name for bc cp agent, separated by comma',
         )], layout=form_item_layout
    )


def save_fig_box(default_value="True", form_item_layout=DEFAULT_FORM_ITEM_LAYOUT):
    """
    Return a Box that takes in mixed_play as a string-boolean
    """
    return Box(
        [Label(value='save_fig'),
         Dropdown(
             value=default_value,
             options=['True', 'False']
         )],
        layout=form_item_layout
    )

def parse_ppo_sp_baseline_lst_org(ppo_sp_baseline_lst_org):
    ppo_sp_baseline_lst = parse_str_to_lst(ppo_sp_baseline_lst_org)
    if ppo_sp_baseline_lst:
        ppo_sp_baseline_lst_str = " ".join(ppo_sp_baseline_lst)
        ppo_sp_baseline_lst_arg = "-b %s" % ppo_sp_baseline_lst_str
    else:
        ppo_sp_baseline_lst_arg = ""
    return ppo_sp_baseline_lst, ppo_sp_baseline_lst_arg

def parse_custom_ai_cp_path_lst_org(custom_ai_cp_path_lst_org):
    custom_ai_cp_path_lst = parse_str_to_lst(custom_ai_cp_path_lst_org)
    if custom_ai_cp_path_lst:
        custom_ai_cp_path_lst = [DEFAULT_RESULTS_DIR + "/" + path for path in custom_ai_cp_path_lst]
        custom_ai_cp_path_lst_str = " ".join(custom_ai_cp_path_lst)
        custom_ai_cp_path_lst_arg = "-aic %s" % custom_ai_cp_path_lst_str
    else:
        # we don't want to pass in the -aic flag unless there are actally more than 1 custom checkpoint
        custom_ai_cp_path_lst_arg = ""
    return custom_ai_cp_path_lst, custom_ai_cp_path_lst_arg

def parse_custom_h_cp_path_lst_org(custom_h_cp_path_lst_org):
    custom_h_cp_path_lst = parse_str_to_lst(custom_h_cp_path_lst_org)
    if custom_h_cp_path_lst:
        custom_h_cp_path_lst = [DEFAULT_RESULTS_DIR + "/" + path for path in custom_h_cp_path_lst]
        custom_h_cp_path_lst_str = " ".join(custom_h_cp_path_lst)
        custom_h_cp_path_lst_arg = "-hc %s" % custom_h_cp_path_lst_str
    else:
        # we don't want to pass in the -hc flag unless there are actally more than 1 custom checkpoint
        custom_h_cp_path_lst_arg = ""
    return custom_h_cp_path_lst, custom_h_cp_path_lst_arg

def parse_custom_bc_cp_path_lst_org(custom_bc_cp_path_lst_org):
    custom_bc_cp_path_lst = parse_str_to_lst(custom_bc_cp_path_lst_org)
    if custom_bc_cp_path_lst:
        custom_bc_cp_path_lst = ["bc_runs/" + path for path in custom_bc_cp_path_lst]
        custom_bc_cp_path_lst_str = " ".join(custom_bc_cp_path_lst)
        custom_bc_cp_path_lst_arg = "-bcc %s" % custom_bc_cp_path_lst_str
    else:
        # we don't want to pass in the -hc flag unless there are actally more than 1 custom checkpoint
        custom_bc_cp_path_lst_arg = ""
    return custom_bc_cp_path_lst, custom_bc_cp_path_lst_arg

################################
########### OPTION 1 ###########
################################

def generate_eval_form_option_1(default_PPO_baseline_lst="11, 21, 31, 41"):
    """
    Generate a form (Box object in ipywidget) to take in user input
    """
    form_items = [
        layout_name_box("cramped_room"),
        num_games_box(40),
        mixed_play_box("True"),
        ppo_sp_baseline_lst_box(default_PPO_baseline_lst),
        custom_ai_cp_path_lst_box(),
        Label(value='Notes: '),
        Label(value='1. cp stands for checkpoint; '),

    ]

    form = Box(form_items, layout=DEFAULT_FORM_OUTER_LAYOUT)
    return form


def generate_eval_command_option_1(f):
    """
    Arguments:
        f (Box): form generated by generate_eval_form_option_1
    print the eval command used for evaluation, and
    return the dictionary used in the evaluation
    """
    # load all variables from its respective fields
    layout_name = f.children[0].children[1].value
    num_games = f.children[1].children[1].value
    mixed_play = f.children[2].children[1].value

    ppo_sp_baseline_lst_org = f.children[3].children[1].value
    ppo_sp_baseline_lst, ppo_sp_baseline_lst_arg = parse_ppo_sp_baseline_lst_org(ppo_sp_baseline_lst_org)

    custom_ai_cp_path_lst_org = f.children[4].children[1].value
    custom_ai_cp_path_lst, custom_ai_cp_path_lst_arg = parse_custom_ai_cp_path_lst_org(custom_ai_cp_path_lst_org)

    eval_params = {
        "layout_name": layout_name,
        "num_games": num_games,
        "mixed_play": mixed_play == "True",
        "ppo_sp_baseline_lst": ppo_sp_baseline_lst,
        "custom_ai_cp_path_lst": custom_ai_cp_path_lst
    }
    print("Evaluation params for option 1", eval_params)
    print("=========")
    print("Please run the following commands:")
    print("---------")
    print("conda activate harl_rllib")
    print("python evaluate.py -l %s -n %d -m %s %s %s" % (layout_name, num_games, mixed_play, ppo_sp_baseline_lst_arg, custom_ai_cp_path_lst_arg))

    return eval_params


def generate_visual_form_option_1(LOAD_PATH, SAVE_FIG="True"):
    """
    Generate a form (Box object in ipywidget) to take in user input for visualization
    """
    form_items = [
        load_path_box(LOAD_PATH),
        data_timestamp_box(),
        custom_ai_cp_display_lst_box(),
        save_fig_box(SAVE_FIG),
        Label(value='Notes: '),
        Label(value='1. cp stands for checkpoint; ')

    ]

    form = Box(form_items, layout=DEFAULT_FORM_OUTER_LAYOUT)
    return form


def generate_visual_params_option_1(f):
    # load all variables from its respective fields
    load_path = f.children[0].children[1].value
    data_timestamp = f.children[1].children[1].value

    # parse the original string into lists
    custom_ai_cp_display_lst_org = f.children[2].children[1].value
    custom_ai_cp_display_lst = parse_str_to_lst(custom_ai_cp_display_lst_org)
    save_fig = f.children[3].children[1].value == "True"
    visual_params = {
        "load_path": load_path,
        "data_timestamp": data_timestamp,
        "custom_ai_cp_display_lst": custom_ai_cp_display_lst,
        "save_fig": save_fig
    }
    return visual_params


################################
########### OPTION 2 ###########
################################

def generate_eval_form_option_2(default_PPO_baseline_lst="11, 21, 31, 41"):
    """
    Generate a form (Box object in ipywidget) to take in user input
    """
    form_items = [
        layout_name_box("cramped_room"),
        num_games_box(40),
        ppo_sp_baseline_lst_box(default_PPO_baseline_lst),
        include_greedy_human_box("True"),
        custom_ai_cp_path_lst_box(),
        custom_h_cp_path_lst_box(),

        Label(value='Notes: '),
        Label(value='1. cp stands for checkpoint; '),
    ]

    form = Box(form_items, layout=DEFAULT_FORM_OUTER_LAYOUT)
    return form


def generate_eval_command_option_2(f):
    """
    Arguments:
        f (Box): form generated by generate_eval_form_option_2
    print the eval command used for evaluation, and
    return the dictionary used in the evaluation
    """
    # load all variables from its respective fields
    layout_name = f.children[0].children[1].value
    num_games = f.children[1].children[1].value
    ppo_sp_baseline_lst_org = f.children[2].children[1].value
    ppo_sp_baseline_lst, ppo_sp_baseline_lst_arg = parse_ppo_sp_baseline_lst_org(ppo_sp_baseline_lst_org)
    include_greedy_human = f.children[3].children[1].value

    # processing custom ai path argument
    custom_ai_cp_path_lst_org = f.children[4].children[1].value
    custom_ai_cp_path_lst, custom_ai_cp_path_lst_arg = parse_custom_ai_cp_path_lst_org(custom_ai_cp_path_lst_org)

    # processing custom human model path argument
    custom_h_cp_path_lst_org = f.children[5].children[1].value
    custom_h_cp_path_lst, custom_h_cp_path_lst_arg = parse_custom_h_cp_path_lst_org(custom_h_cp_path_lst_org)

    eval_params = {
        "layout_name": layout_name,
        "num_games": num_games,
        "ppo_sp_baseline_lst": ppo_sp_baseline_lst,
        "include_greedy_human": include_greedy_human == "True",
        "custom_ai_cp_path_lst": custom_ai_cp_path_lst,
        "custom_h_cp_path_lst": custom_h_cp_path_lst
    }
    print("Evaluation params for option 1", eval_params)
    print("=========")
    print("Please run the following commands:")
    print("---------")
    print("conda activate harl_rllib")
    print("python evaluate.py -l %s -n %d %s -g %s %s %s" % (layout_name, num_games, ppo_sp_baseline_lst_arg, include_greedy_human, custom_ai_cp_path_lst_arg, custom_h_cp_path_lst_arg))

    return eval_params

def generate_visual_form_option_2(LOAD_PATH, SAVE_FIG="True"):
    """
    Generate a form (Box object in ipywidget) to take in user input for visualization
    """
    form_items = [
        load_path_box(LOAD_PATH),
        data_timestamp_box(),
        custom_ai_cp_display_lst_box(),
        custom_h_cp_display_lst_box(),
        save_fig_box(SAVE_FIG),
        Label(value='Notes: '),
        Label(value='1. cp stands for checkpoint; ')

    ]

    form = Box(form_items, layout=DEFAULT_FORM_OUTER_LAYOUT)
    return form


def generate_visual_params_option_2(f):
    # load all variables from its respective fields
    load_path = f.children[0].children[1].value
    data_timestamp = f.children[1].children[1].value
    # process custom ai checkpoint display
    custom_ai_cp_display_lst_org = f.children[2].children[1].value
    custom_ai_cp_display_lst = parse_str_to_lst(custom_ai_cp_display_lst_org)

    # process custom h checkpoint display
    custom_h_cp_display_lst_org = f.children[3].children[1].value
    custom_h_cp_display_lst = parse_str_to_lst(custom_h_cp_display_lst_org)

    save_fig = f.children[4].children[1].value == "True"
    visual_params = {
        "load_path": load_path,
        "data_timestamp": data_timestamp,
        "custom_ai_cp_display_lst": custom_ai_cp_display_lst,
        "custom_h_cp_display_lst": custom_h_cp_display_lst,
        "save_fig": save_fig
    }
    return visual_params
