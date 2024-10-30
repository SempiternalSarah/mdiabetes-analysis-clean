import matplotlib.pyplot as plt
import numpy as np
from utils.replay import ReplayDB

# helper functions to load ai training data from each week

def AIAnalytics(w):
    rep = ReplayDB().week(w)
    cluster = rep.replay("clusters").load()
    debug = rep.replay("debug").load()
    return (cluster, debug)
    
def plot_loss(w):
    # plot a line graph of the loss curve for one week
    cluster, debug = AIAnalytics(w)
    fig, ax = plt.subplots(figsize=(6,4))
    for i, l in enumerate(debug['loss']):
        ax.plot(l)
    ax.set_title(f"Loss week {w}")
    
def plot_cluster_counts(w):
    # plot the total number of members in the cluster 
    # against the number of transitions/responses collected from each
    cluster, debug = AIAnalytics(w)
    c = np.unique(cluster, return_counts=True)
    c_counts = {c[0][i]: c[1][i] for i in range(c[0].size)}
    t_counts = debug['metrics']['cluster_t_counts']
    fig, ax = plt.subplots(ncols=2, figsize=(6,4))
    ymax = -1
    for i, k in enumerate(c_counts.keys()):
        ax[0].bar(i, c_counts[k])
        ax[1].bar(i, t_counts[k])
        if c_counts[k] > ymax:
            ymax = c_counts[k]
    ax[0].set_title(f"Cluster Counts week {w}")
    ax[1].set_title(f"Transition Count week {w}")
    ax[0].set_ylim(0, ymax)
    ax[1].set_ylim(0, ymax)
