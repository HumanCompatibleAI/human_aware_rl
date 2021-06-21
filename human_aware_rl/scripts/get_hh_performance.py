from collections import defaultdict
from human_aware_rl.static import HUMAN_DATA_DIR, LAYOUTS_WITH_DATA_2020
import numpy as np
import os, pickle

CLEAN_AND_BALANCED_DIR = os.path.join(HUMAN_DATA_DIR, 'cleaned_and_balanced')

scores = defaultdict(dict)
datasets = ['train', 'test']
for dataset in datasets:
    path = os.path.join(CLEAN_AND_BALANCED_DIR, '2020_hh_trials_balanced_rew_50_50_split_{}.pickle'.format(dataset))
    with open(path, 'rb') as f:
        data = pickle.load(f)
    for layout in LAYOUTS_WITH_DATA_2020:
        curr_layout_data = data[data['layout_name'] == layout]
        layout_scores = np.unique(curr_layout_data['score_total'])
        mean, std = np.mean(layout_scores), np.std(layout_scores)

        scores[layout][dataset] = (mean, std)

for layout in LAYOUTS_WITH_DATA_2020:
    print("Layout:", layout)
    for dataset in datasets:
        mean, std = scores[layout][dataset]
        print("H-H {} performance mean: {:.2f}".format(dataset, mean))
        print("H-H {} performance std: {:.2f}".format(dataset, std))
        print("")