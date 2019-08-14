from collections import defaultdict

def get_algorithm_color(alg):
    opt_baseline_col = "#eaeaea"
    ours_col = '#F79646'# orange #'#f79646'
    other_col = '#4BACC6' # thiel
    other_other_col = "#2d6777"
    human_baseline_col = "#aeaeae"#"#c1c1c1"

    if alg == 'CP+CP' or alg == 'PPO_SP+PPO_SP' or alg == 'PBT+PBT' or alg == "H+H" or alg == "avg_bc_train+bc_train":
        return opt_baseline_col
    elif alg in ["CP+BC_test", 'PPO_SP+BC_test_0', 'PPO_SP+BC_test_1', 'ppo_sp_no_advers_0', 'ppo_sp_no_advers_1', 'ppo_sp_base_0', 'ppo_sp_base_1']:
        return other_other_col #'#35ce47'#'#0000cc'
    elif alg in ['pbt_no_advers_0', 'pbt_no_advers_1', 'pbt_base_0', 'pbt_base_1', 'PBT+BC_0', 'PBT+BC_1']:
        return other_col #'#35ce47'#'#0000cc'
    elif alg in ["BC_train+BC_test_0", "BC_train+BC_test_1"]:
        return human_baseline_col #"#3884c9"
    elif alg in ["P+BC_test", 'PPO_BC_train+BC_test_0', 'PPO_BC_train+BC_test_1', 'ppo_bc_no_advers_0', 'ppo_bc_no_advers_1', 'ppo_bc_base_0', 'ppo_bc_base_1']:
        return ours_col #'#35ce47'#'#00cc00'
    else:
        raise ValueError(alg, "not recognized")

def get_texture(alg):
    if alg in ['PBT+PBT', 'avg_bc_train+bc_train']:
        return '\\\\'
    elif alg == 'CP+CP':
        return '\\\\'
    elif alg[-1:] == '1': #'-', '+', 'x', '\\', '*', 'o', 'O', '.'
        return '/'
    else:
        return ''
    
def get_alpha(alg):
    if alg == ["BC_train+BC_test_0", "BC_train+BC_test_1"]:
        return 0.3
    else:
        return 1

def switch_indices(idx0, idx1, lst):
    lst = list(lst)
    lst[idx1], lst[idx0] = lst[idx0], lst[idx1]
    return lst

def means_and_stds_by_algo(full_data):
    mean_by_algo = defaultdict(list)
    std_by_algo = defaultdict(list)
    for layout, layout_algo_dict in full_data.items():
        for k in layout_algo_dict.keys():
            if type(layout_algo_dict[k]) is list or type(layout_algo_dict[k]) is tuple:
                mean, std = layout_algo_dict[k]
            else:
                mean, std = layout_algo_dict[k], 0
            mean_by_algo[k].append(mean)
            std_by_algo[k].append(std)
    return mean_by_algo, std_by_algo

def y_lim(hist_type):
    if hist_type == "cp":
        return 600
    elif hist_type in ["humanai", "humanai_base"]:
        return 200
    return 255

def graph_title(hist_type):
    if hist_type == "humanai":
        return "Performance with real humans"
    elif hist_type == "humanai_base":
        return "Performance with real humans (unfiltered)"
    return 'Performance with human proxy model'
    