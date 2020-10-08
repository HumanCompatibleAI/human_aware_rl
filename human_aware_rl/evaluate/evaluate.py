from human_aware_rl.evaluate.eval_utils import from_params_stats_ppo_agent_pair_lst_self_play, \
    from_params_stats_ppo_agent_pair_lst_mixed_play, from_params_stats_ppo_agent_pair_lst_human_play
from ray.tune.result import DEFAULT_RESULTS_DIR

BASELINE_DIR = "../data/ppo_sp_baseline_checkpoints"

def get_checkpoint_path_from_experiment(experiment_name, checkpoint_num):
    """
    Please look under DEFAULT_RESULTS_DIR (which usually is ~/ray_results)
    Arguments:
        experiment_name (str): the name of the experiment
        checkpoint_num (int): the place of checkpoint
    Return:
        (str): checkpoint path for EXPERIMENT_NAME at CHECKPOINT_NUM
    """
    return DEFAULT_RESULTS_DIR + "/%s/checkpoint_%s/checkpoint-%s" % (experiment_name, checkpoint_num, checkpoint_num)


def get_baseline_checkpoint_path_from_experiment(experiment_name, checkpoint_num):
    """
    Please look under BASELINE_DIR (which usually is ../data/ppo_sp_baseline_checkpoints)
    Arguments:
        experiment_name (str): the name of the experiment
        checkpoint_num (int): the place of checkpoint
    Return:
        (str): checkpoint path for baseline EXPERIMENT_NAME at CHECKPOINT_NUM
    """
    return BASELINE_DIR + "/%s/checkpoint_%s/checkpoint-%s" % (experiment_name, checkpoint_num, checkpoint_num)


def get_baseline_checkpoint_path_for_layout(layout_name, ppo_sp_baseline_lst):
    """
    Arguments:
        layout_name (str): name of the layout being evaluated
        ppo_sp_baseline_lst (list of string): list of integer seeds for the baseline in use
    Return:
        (list of str): list of checkpoint path for the baselines
    """
    ppo_sp_baseline_seed_lst = [int(baseline_seed) for baseline_seed in ppo_sp_baseline_lst]
    # this is assuming using 11, 21, 31, 41 as seeds. Need to modify if using another set of random seeds
    for seed in ppo_sp_baseline_seed_lst:
        assert seed in {11, 21, 31, 41}
    ppo_sp_baseline_index_lst = [seed // 10 - 1 for seed in ppo_sp_baseline_seed_lst]
    ppo_sp_baseline_exp_names = layout_name_to_exp_names_dict[layout_name]
    ppo_sp_baseline_exp_names_selected = [ppo_sp_baseline_exp_names[idx] for idx in ppo_sp_baseline_index_lst]
    return [get_baseline_checkpoint_path_from_experiment(exp_name, 500) for exp_name in ppo_sp_baseline_exp_names_selected]


def eval_from_layout_name(layout_name, ai_checkpoint_path_lst, h_checkpoint_path_lst=[], num_games=40, outer_shape=None, mixed_play=False, include_greedy_human=False):
    """
    Arguments:
        layout_name (str) the name of layouts
        num_games (int): the number of games we are evaluating
    """
    mdp_gen_param = {"layout_name": layout_name}
    human_ai_evaluation = len(h_checkpoint_path_lst) > 0 or include_greedy_human

    if human_ai_evaluation:
        print("HUMAN-AI PLAY")
        return from_params_stats_ppo_agent_pair_lst_human_play(
            mdp_gen_param,
            include_greedy_human,
            ai_checkpoint_path_lst,
            h_checkpoint_path_lst,
            outer_shape=outer_shape,
            num_layouts=1,
            num_games=num_games
        )

    else:
        if mixed_play:
            print("MIXED PLAY")
            return from_params_stats_ppo_agent_pair_lst_mixed_play(
                mdp_gen_param,
                ai_checkpoint_path_lst + h_checkpoint_path_lst,
                outer_shape=outer_shape,
                num_layouts=1,
                num_games=num_games
            )
        else:
            print("SELF PLAY")
            return from_params_stats_ppo_agent_pair_lst_self_play(
                mdp_gen_param,
                ai_checkpoint_path_lst + h_checkpoint_path_lst,
                outer_shape=outer_shape,
                num_layouts=1,
                num_games=num_games
            )


layout_name_to_exp_names_dict = {
    "cramped_room":
        [
            "PPO_cramped_room_11",
            "PPO_cramped_room_21",
            "PPO_cramped_room_31",
            "PPO_cramped_room_41"
         ],

    "asymmetric_advantages":
        [
            "PPO_asymmetric_advantages_11",
            "PPO_asymmetric_advantages_21",
            "PPO_asymmetric_advantages_31",
            "PPO_asymmetric_advantages_41"
        ],

    "coordination_ring":
        [
            "PPO_coordination_ring_11",
            "PPO_coordination_ring_21",
            "PPO_coordination_ring_31",
            "PPO_coordination_ring_41"
        ],

    "forced_coordination":
        [
            "PPO_forced_coordination_11",
            "PPO_forced_coordination_21",
            "PPO_forced_coordination_31",
            "PPO_forced_coordination_41"
        ],

    "counter_circuit_o_1order":
        [
            "PPO_counter_circuit_o_1order_11",
            "PPO_counter_circuit_o_1order_21",
            "PPO_counter_circuit_o_1order_31",
            "PPO_counter_circuit_o_1order_41"
        ],
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout_name', '-l', type=str, default="")
    parser.add_argument('--num_games', '-n', type=int, default=1)
    parser.add_argument('--mixed_play', '-m', type=str, default="False")
    parser.add_argument('--ppo_sp_baseline_lst', '-b', nargs='+', default=[])
    parser.add_argument('--include_greedy_human', '-g', type=str, default="False")
    parser.add_argument('--num_layouts', '-t', type=int, default=1)
    parser.add_argument('--custom_ai_cp_path_lst', '-aic', nargs='+', default=[])
    parser.add_argument('--custom_h_cp_path_lst', '-hc', nargs='+', default=[])

    args = parser.parse_args()
    num_games = args.num_games
    mixed_play = args.mixed_play == "True"
    ppo_sp_baseline_lst = args.ppo_sp_baseline_lst
    include_greedy_human = args.include_greedy_human == "True"
    custom_ai_cp_path_lst = args.custom_ai_cp_path_lst
    custom_h_cp_path_lst = args.custom_h_cp_path_lst

    print("custom ai checkpoint path:", args.custom_ai_cp_path_lst)
    print("custom h checkpoint path:", args.custom_h_cp_path_lst)

    if args.layout_name != "":
        layout_name = args.layout_name
        print("Single layout evaluation on", layout_name)
        # populate ai_checkpoint_path_lst
        ai_checkpoint_path_lst = get_baseline_checkpoint_path_for_layout(layout_name, ppo_sp_baseline_lst) + custom_ai_cp_path_lst
        h_checkpoint_path_lst = custom_h_cp_path_lst
        data_timestamp = eval_from_layout_name(
            layout_name,
            ai_checkpoint_path_lst,
            h_checkpoint_path_lst,
            num_games=num_games,
            mixed_play=mixed_play,
            include_greedy_human=include_greedy_human
        )
        print("=================")
        print(layout_name, "data_timestamp:", data_timestamp)

    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
