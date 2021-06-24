from human_aware_rl.human.process_dataframes import get_human_human_trajectories
from human_aware_rl.static import CLEAN_2020_HUMAN_DATA_ALL, LAYOUTS_WITH_DATA_2020
from overcooked_ai_py.agents.agent import AgentPair, RandomAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
import numpy as np
import argparse, itertools

ALL_AGENTS = ['true_human', 'rnd']

def main(layout_name, max_ood_counters, agent_type, percentiles, find_best):
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    all_counter_positions = mdp.get_counter_locations()
    print("Gathering trajectories...")
    rnd_trajectories = trajectories = None
    trajectories = get_trajectories(layout_name, agent_type)
    if find_best:
        assert agent_type == 'true_human'
        rnd_trajectories = get_trajectories(layout_name, 'rnd')
    print("Done!")

    ood_counters_combinations = [[]]
    for i in range(1, max_ood_counters+1):
        ood_counters_combinations.extend(itertools.combinations(all_counter_positions, i))
    
    best_score = 0
    best_combo = None
    for ood_counters in ood_counters_combinations:
        print("Analyzing OOD counter={}...".format(ood_counters))
        human_ood_percentage = analyze_ood_percentages(layout_name, trajectories, ood_counters, percentiles)
        if find_best:
            rnd_ood_percentage = analyze_ood_percentages(layout_name, rnd_trajectories, ood_counters, percentiles)
            score = rnd_ood_percentage - human_ood_percentage
            if score > best_score:
                best_score = score
                best_combo = ood_counters

    if find_best:
        print("Best OOD counters found={}".format(best_combo))
        print("score={}".format(best_score))

def analyze_ood_percentages(layout_name, trajectories, ood_counters=[], percentiles=True):
    if ood_counters:
        OvercookedGridworld.set_ood_counters(layout_name, ood_counters)
    mdp = OvercookedGridworld.from_layout_name(layout_name)

    traj_lens = trajectories['ep_lengths']
    traj_states = trajectories['ep_states']
    trajs_ood_counts = []

    for traj in traj_states:
        ood_cnt = 0
        for state in traj:
            if mdp.is_off_distribution(state):
                ood_cnt += 1
        trajs_ood_counts.append(ood_cnt)

    ood_percentages = np.array(trajs_ood_counts) / np.array(traj_lens)
    gloabl_ood_percentage = sum(trajs_ood_counts) / sum(traj_lens)

    print("Results for OOD Counters={}".format(ood_counters))
    print("Global OOD%: {}\n".format(gloabl_ood_percentage))

    if percentiles:
        off_dist_percentage_sorted = np.sort(ood_percentages)
        N = len(off_dist_percentage_sorted)
        percentiles = [0, .25, .5, .75, 1]
        for percentile in percentiles:
            idx = min(N-1, int(N*percentile))
            print("Off dist {}-percentile: {}".format(percentile, off_dist_percentage_sorted[idx]))
    print("")
    return gloabl_ood_percentage

def get_trajectories(layout_name, agent_type):
    if agent_type == 'true_human':
        return get_human_human_trajectories([layout_name], data_path=CLEAN_2020_HUMAN_DATA_ALL, featurize_states=False)
    elif agent_type == 'rnd':
        agent = RandomAgent(all_actions=True)
        ae = AgentEvaluator.from_layout_name({"layout_name" : layout_name}, {'horizon' : 400})
        return ae.evaluate_agent_pair(AgentPair(agent, agent, allow_duplicate_agents=True), num_games=50)
    else:
        raise NotImplementedError() 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout_name', '-l', type=str, required=True, choices=LAYOUTS_WITH_DATA_2020)
    parser.add_argument('--max_ood_counters', '-n', type=int, default=0)
    parser.add_argument('--agent_type', '-a', type=str, default='true_human', choices=ALL_AGENTS)
    parser.add_argument('--percentiles', '-p', action='store_true')
    parser.add_argument('--find_best', '-b', action='store_true')

    args = vars(parser.parse_args())
    main(**args)