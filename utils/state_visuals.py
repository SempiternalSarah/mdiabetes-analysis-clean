import numpy as np
import os
import matplotlib.pyplot as plt
from utils.state_data import StateData

sd = StateData()

elem_map = {
    1: "Healthy Food Intake",
    2: "Unhealthy Food/Tobacco/\nAlcohol Intake",
    3: "Fitness/Activiy Level",
    4: "Cause Knowledge",
    5: "Complication Knowledge"
}

def sub_adj(**kw):
    cf = {"wspace": 0.2, "hspace": 0.2}
    cf.update(**kw)
    plt.subplots_adjust(**cf)
    
def ticks(*aa, x=True, y=True, **kw):
    cf = {"rotation": 0}
    for a in aa:
        if x:
            a.tick_params(axis="x", **cf)
        if y:
            a.tick_params(axis="y", **cf)
            
def label(ax, i, l, fi=None, **cf):
    cf = {"rotation": 0, "va": "center_baseline",
          "labelpad": 5, "fontsize": 12,
          "ha": "right",}
    if fi is None:
        fi = i
    ax[fi].set_ylabel(l, **cf)  
    
def elem_label(ax, i, fi=None, **cf):
    if fi is None:
        fi = i
    label(ax, i, elem_map[i+1], fi, **cf) 
    
def title_to_file(t):
    t = t.replace(" ", "-").replace("\n", "-").replace("%", "")
    return "img/" + t + ".png"
    
def save(f, t):
    os.makedirs("img", exist_ok=True)
    f.savefig(title_to_file(t), facecolor="w", bbox_inches="tight")
    plt.clf()
    plt.close("all")

def plot_state_change(data, title="State Element Change Histogram", **kw):
    selems = sd.calc_state_elem_change(data)
    L = len(selems)
    fig, ax = plt.subplots(nrows=L, ncols=2, figsize=(10,10))
    cf = {
        "range": (1,3),
        "bins": "auto",
        "alpha": 0.5
    }
    cf.update(**kw)
    for i, selem in enumerate(selems):
        ax[i][0].hist(selem['start'], label="Start", **cf)
        ax[i][0].hist(selem['end'], label="End", **cf)
        ax[i][1].hist(selem['end']-selem['start'], range=(-2,2))
        elem_label(ax[i], i, fi=0)
        if i == 0:
            ax[i][0].legend(loc="best")
        if i < (L-1):
            ax[i][0].get_xaxis().set_visible(False)
            ax[i][1].get_xaxis().set_visible(False)
        if i == (L-1):
            ax[i][0].set_xlabel("State Element Value")
            ax[i][1].set_xlabel("Change in State Element Value")
        ticks(ax[i][0])
    ax[0][0].set_title("Start vs. End")
    ax[0][1].set_title("Change")
    sub_adj()
    fig.suptitle(title, fontsize=14, y=.975)
    save(fig, title)
    
def plot_state_elem_running_change(data, title="State Elem Running Change", path=None):
    selems = sd.calc_state_elem_change(data)
    L = len(selems)
    fig, ax = plt.subplots(nrows=5, figsize=(5,12))
    x = np.arange(selems[0]['start'].shape[0])
    for i, selem in enumerate(selems):
        diff = selem['start']-selem['end']
        diff = np.sort(diff)[::-1]
        C = np.array(["r"] * diff.shape[0])
        imp = diff > 0
        dec = diff < 0
        imp_perc = imp.sum() / imp.shape[0]
        dec_perc = dec.sum() / dec.shape[0]
        C[imp] = "r"
        C[dec] = "b"
        imp_stop = np.where(imp==True)[0][-1]
        dec_start = np.where(dec==True)[0][0]
        ax[i].axvline(imp_stop+.4, alpha=0.8, ymin=0.5, linestyle="--", color="r", label="Improvement Stops")
        ax[i].axvline(dec_start-.4, alpha=0.8, ymax=0.5, linestyle="--", color="b", label="Deterioration Starts")
        ax[i].text(int(imp.sum()*.85), -.7, f"{imp_perc*100:.1f}%")
        ax[i].text((~dec).sum(), .5, f"{dec_perc*100:.1f}%")
        ax[i].bar(x, diff, color=C)
        ax[i].set_ylim((-2,2))
        elem_label(ax, i)
        ax[i].axhline(0, alpha=0.8, linestyle="--", color="k")
        if i < (L-1):
            ax[i].get_xaxis().set_visible(False)
        if i == 0:
            ax[i].legend(loc="lower left")
        if i == (L-1):
            ax[i].set_xlabel("Participant ID")
        ticks(ax[i])
        
    sub_adj()
    ax[0].set_title(title, fontsize=14)
    save(fig, title)
    
def plot_response_counts(data, title="Response Counts", sample=1):
    rc = sd.active_responders(sample, data)
    fig, ax = plt.subplots(ncols=3, figsize=(15,4))
    counts = rc['counts']
    counts = [x.sum() for x in counts]
    counts = np.sort(counts)[::-1]
    x = np.arange(len(counts))
    ax[0].bar(x, counts)
    ax[0].set_title(title, fontsize=14)
    ax[0].set_yticks(np.arange(0, np.max(counts)+2, 2))
    ax[0].set_yticklabels(np.arange(0, np.max(counts)+2, 2))
    ax[0].set_xticks(np.arange(0, counts.shape[0]+75, 75))
    ax[0].set_xticklabels(np.arange(0, counts.shape[0]+75, 75))
    hstop = (counts >= np.max(counts)//2).sum()
    hstop_perc = (hstop/counts.shape[0]) * 100
    ax[0].axvline(hstop, linestyle="-", color="k", label="Respond to 1/2")
    ax[0].text(hstop+.5, np.max(counts)-1, f"{hstop_perc:.1f}%")
    rstop = (counts > 0).sum()
    rstop_perc = (rstop/counts.shape[0]) * 100
    ax[0].axvline(rstop, linestyle="--", color="k", label="Respond to at least 1")
    ax[0].legend(loc="upper right")
    ax[0].text(rstop+.5, np.max(counts)//2, f"{rstop_perc:.1f}%")
    ax[0].set_title("Sorted by Participant")
    ax[0].set_xlabel("# Participants")
    ax[0].set_ylabel("# Responses")
    ax[1].hist(counts)
    ax[1].set_title("Histogram for all Participants")
    ax[1].set_xlabel("# Responses")
    ax[1].set_ylabel("# Participants")
    cx, cy, ly = [], [], []
    for m in range(np.max([c.shape[0] for c in rc['counts']])):
        cx.append(m+1)
        cym = 0
        for c in rc['counts']:
            if len(c) > m and np.all(c > 0):
                cym += 1
        cy.append(cym/len(rc['counts']))
    ax[2].bar(cx, cy)
    ax[2].set_title("Consistent Responders by Opt-In Time")
    ax[2].set_xlabel("Weeks Since Opt-In")
    ax[2].set_ylabel("# Participants")
    fig.suptitle(title, fontsize=14, y=1.05)
    save(fig, title)
    
def _plot_active_participants(data, sample=1, title_add=""):
    title_end = f"\nTop {sample*100:.1f}% responders"
    maketitle = lambda x: x + title_add + title_end
    rc = sd.active_responders(sample, data)
    plot_state_elem_running_change(
        data.loc[rc['ids']],
        title=maketitle("State Element Running Change"),
    )
    plot_state_change(
        data.loc[rc['ids']],
        title=maketitle("Start vs End State Value Distribution"),
    )
    
def plot_active_participants(data, samp):
    plot_response_counts(data, title="Breakdown of Response Counts For all Participants")
    for _samp in samp:
        _plot_active_participants(data, _samp)
        print(_samp)
        