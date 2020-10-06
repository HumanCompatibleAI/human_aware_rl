import numpy as np
import matplotlib.pyplot as plt
from os import makedirs


# Where to save the figures
FIGURE_DIR = "figures/"

makedirs(FIGURE_DIR, exist_ok=True)

def plot_mean_std(mean, std, labels, checkpoint, outer_shape, title_info, figure_dir=FIGURE_DIR, save_fig=False):
    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots()

    rects = ax.bar(x, np.round(mean, decimals=2), width, label='sparse reward', yerr = std)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("mean spare reward")
    ax.set_title(title_info + " " + outer_shape + ' mean spare reward by PPO agent at checkpoint ' + checkpoint)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects)
    fig.set_size_inches(10, 6)
    if save_fig:
        plt.savefig(figure_dir + title_info + "_" + outer_shape + "_" + checkpoint + "_sparse_rew.png")
    plt.show()


def plot_matrix(agent_0, agent_1, data, data_label, layout_name, figure_dir=FIGURE_DIR, save_fig=False):
    fig, ax = plt.subplots()
    ax.imshow(data, vmin=0, vmax=np.max(data))
    print(agent_0)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(agent_1)))
    ax.set_yticks(np.arange(len(agent_0)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(agent_1)
    ax.xaxis.tick_top()
    ax.set_yticklabels(agent_0)

    # Loop over data dimensions and create text annotations.
    for i in range(len(agent_0)):
        for j in range(len(agent_1)):
            text = ax.text(j, i, data[i, j],
                           ha="center", va="center", color="w")

    ax.set_title(data_label + " for SP pairs: " + layout_name)
    fig.set_size_inches(4.5, 4.5)

    if save_fig:
        plt.savefig(figure_dir + layout_name + "_" + data_label + "_matrix.png")
    plt.show()


def plot_single(mean, std, labels, layout_name, figure_dir=FIGURE_DIR, save_fig=False):
    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots()

    rects = ax.bar(x, np.round(mean, decimals=2), width, yerr = std)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("mean scores")
    ax.set_title('agent self play: ' + layout_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects)
    fig.set_size_inches(5, 5)
    if save_fig:
        plt.savefig(figure_dir + layout_name + "_sp.png")
    plt.show()

def plot_for_all_pairs(sparse_rew, layout_name, bins, figure_dir=FIGURE_DIR, save_fig=False):
    NUM_COLORS = len(bins)
    fig, axis = plt.subplots()
    _, _, patches = axis.hist(sparse_rew, bins, weights=np.ones(len(sparse_rew)) / len(sparse_rew))
    coloring_patches(patches, NUM_COLORS - 1)
    fig.suptitle("Score distribution for all pairs: %s" % layout_name)
    fig.set_size_inches(5, 5)
    if save_fig:
        fig.savefig(figure_dir + layout_name + "_score_all_pair_hist.png")
    plt.show()

def plot_matrix_aug(agent_0, agent_1, data, data_label, layout_name, display_title, figure_dir=FIGURE_DIR, save_fig=False):
    fig, ax = plt.subplots()
    im = ax.imshow(data, vmin=0, vmax=np.max(data))

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(agent_1)))
    ax.set_yticks(np.arange(len(agent_0)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(agent_1)
    ax.xaxis.tick_top()
    ax.set_yticklabels(agent_0)

    # Loop over data dimensions and create text annotations.
    for i in range(len(agent_0)):
        for j in range(len(agent_1)):
            text = ax.text(j, i, data[i, j],
                           ha="center", va="center", color="w")

    ax.set_title(data_label + " for [human]-[ai] pairs: " + layout_name)

    fig.set_size_inches(9, 4.5)

    if save_fig:
        plt.savefig(figure_dir + layout_name + "_" + data_label + "ai-human_" + display_title + "_matrix.png")
    plt.show()

def plot_sparse_rew_over_agent_pairs(sparse_rew_original, pairing_names, layout_info, bins, figure_dir=FIGURE_DIR, save_fig=False):
    """
    Arguments:
        sparse_rew_original (2d numpy array): sparse reward for pair i is in sparse_rew_original[i]
        pairing_names (list of string): display labels
        layout_info (dictionary): contains information for the title of graph
        bins (list): the list of bins being used in the plot
        figure_dir (string): save directory
    """
    # make sure all the pairs are here
    assert sparse_rew_original.shape[0] == len(pairing_names)
    # select the rows with poor sparse reward
    sparse_rew = sparse_rew_original.flatten()
    # populate all but last column of distribution matrix with the subset_layouts
    distribution_mtx = np.zeros((len(bins), len(sparse_rew_original) + 1))
    for i in range(len(sparse_rew_original)):
        sparse_rew_original_i = sparse_rew_original[i]
        for j in range(len(bins)):
            b = bins[j]
            distribution_mtx[j][i] = len(np.where(sparse_rew_original_i < b)[0]) / len(sparse_rew_original_i)
    # the last column of distribution matrix is the mean
    for j in range(len(bins)):
        b = bins[j]
        distribution_mtx[j][-1] = len(np.where(sparse_rew < b)[0]) / len(sparse_rew)
    # print(distribution_mtx)

    ind = np.arange(len(sparse_rew_original) + 1) + .15  # the x locations for the groups
    width = 0.35  # the width of the bars
    # plotting, using the set_pro_cycle in pyplot
    fig, ax = plt.subplots()
    cm = plt.get_cmap('gist_rainbow')
    NUM_COLORS = len(bins)
    ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
    for j in reversed(range(len(bins))):
        sparse_j = distribution_mtx[j]
        ax.bar(ind, sparse_j, width)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Proportion of games')
    if "layout_name" in layout_info:
        display_info = layout_info["layout_name"]
    else:
        display_info = ""
        # TODO
        raise NotImplementedError()

    ax.set_title('Score distribution for each pair: %s' % display_info)
    fig.set_size_inches(5, 5)
    ax.set_xticks(ind)
    ax.set_xticklabels(pairing_names + ["all\npairs"])
    if save_fig:
        plt.savefig(figure_dir + display_info + "_score_each_pair_hist" + ".png")
    plt.show()

def coloring_patches(patches, NUM_COLORS):
    cm = plt.get_cmap('gist_rainbow')
    color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)][::-1]
    for i in range(len(patches)):
        patches[i].set_facecolor(color[i])
