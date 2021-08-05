from human_aware_rl_adaption.tom.tom_policy import ToMPolicy, GreedyPolicy

DEFAULT_TOM_PARAMS = {
    "tom_policy_cls": ToMPolicy,
    "tom_config": {
        "stochastic": False,
        "tom_attributes": {
            "prob_random_action": 0,
            "compliance": 0.5,
            "teamwork": 0.8,
            "retain_goals": 0.8,
            "wrong_decisions": 0.02,
            "prob_thinking_not_moving": 0.2,
            "path_teamwork": 0.8,
            "rationality_coefficient": 3,
            "prob_pausing": 0.5,
            "use_OLD_ml_action": False,
            "prob_greedy": 0,
            "prob_obs_other": 0,
            "look_ahead_steps": 4
        }
    }

}


DEFAULT_GREEDY_PARAMS = {
    "greedy_policy_cls": GreedyPolicy,
    "greedy_config": {
        "greedy_attributes": {
            "hl_temp": 1,
            "ll_temp": 1,
            "prob_wait": 0.5
        }
    }
}