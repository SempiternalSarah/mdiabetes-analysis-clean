import matplotlib.pyplot as plt
import sys
import numpy as np
from utils.content import MessageHandler
from utils.replay import ReplayDB

path_pre = ""
MsgH = MessageHandler(path_prepend=path_pre)
rep = ReplayDB(path_pre=path_pre).replay("actions")

class WeeklyMessageHistogram:
    
    def __init__(self, ):
        self.actions = rep
        
    def rhist(self, minw=None, maxw=None):
        # running/stacked histogram of each week
        if maxw is None:
            maxw = self.actions.maxweek()
        if minw is None:
            minw = self.actions.minweek()
        rhist = np.zeros((maxw+1-minw,57))
        for i, w in enumerate(range(minw, maxw+1)):
            rhist[i] = self.hist(w)
        return rhist
    
    def rdist(self, minw=None, maxw=None):
        # running/stacked distribution of each week
        rhist = self.rhist(minw, maxw).astype(float)
        rdist = rhist.copy()
        for i in range(rhist.shape[0]):
            rdist[i] /= np.sum(rhist[i])
        return rdist
    
    def shist(self, minw=None, maxw=None):
        # cumulative/sum of histograms of each week
        return self.rhist(minw, maxw).sum(0)
    
    def hist(self, w):
        # histogram of one week
        msgs = []
        act = self.actions.week(w).load()[:,1].numpy()
        for a in act:
            msgs.extend(MsgH.mid_lookup(a))
        msgs = np.array(msgs)
        msgs = np.append(msgs, np.array([0,56]))
        v, c = np.unique(msgs, return_counts=True)
        hist = np.zeros((1,57))
        for i in range(len(v)):
            hist[0][v[i]-1] += c[i]
        hist[0,[0,56]] -= 1
        return hist
    
# bar chart one weekly histogram
def bar_hist(w=None):
    if w is None:
        w = ReplayDB().replay("outfiles").maxweek()
    wh = WeeklyMessageHistogram()
    fig, ax = plt.subplots(figsize=(5,4))
    h = wh.hist(w)
    x = np.array(list(range(57)))
    ax.bar(x, h[0])
    ax.set_title(f"Message Histogram week {w}")
    
# bar chart the sum of histograms from minweek-->maxweek
# aka total/cumulative histogram of messages sent
def bar_sum_hist(*minmaxw):
    n = len(minmaxw)
    colors = "bgry"
    fig, ax = plt.subplots(figsize=(5,4))
    if len(minmaxw) == 0:
        minmaxw = [[None, None]]
    width = 1.0 / len(minmaxw)
    width = max(width, .9)
    for i, (minw, maxw) in enumerate(minmaxw):
        if minw is None:
            minw = rep.minweek()
        if maxw is None:
            maxw = rep.maxweek()
        wh = WeeklyMessageHistogram()
        h = wh.shist(minw, maxw)
        x = np.array(list(range(57)))
        ax.bar(x+(i*width), h, color=colors[i], width=width)
    ax.set_title(f"Cumulative Message Histogram week {minw} to {maxw}")
    
# create a heatmap of the distribution of messages each week
def heatmap_running_hist(minw=None, maxw=None, cmap="magma"):
    if minw is None:
        minw = rep.minweek()
    if maxw is None:
        maxw = rep.maxweek()
    fig, ax = plt.subplots(figsize=(5,4))
    wh = WeeklyMessageHistogram()
    h = wh.rdist(minw, maxw)
    ax.imshow(h, cmap=cmap, aspect="auto")
    ax.set_title(f"Heatmap of Message Distribution week {minw} to {maxw}")