import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
from utils.content import MessageHandler, QuestionHandler
from utils.replay import ReplayDB

class StateData:
    
    def __init__(self, path_pre="", storage_dir="prod", detailed=False):
        # path_pre: relative path to the mdiabetes/ folder
        # storage_dir: which directory from local_storage/ to read from 
        #                       (we have [prod]uction and [simul]ation)
        self.rep = ReplayDB(path_pre + "local_storage/" + storage_dir)
        self.msgh = MessageHandler(path_prepend=path_pre, detailed=detailed)
        if (detailed):
            self.qsnh = QuestionHandler(path_prepend=path_pre, map="detailed_question_state_element_map.json")
        else:
            self.qsnh = QuestionHandler(path_prepend=path_pre)
        self.path_pre = path_pre
        self.storage_dir = storage_dir
                
    def buildby(self, by, minw=4, maxw=7, data=None, **kw):
        out = {}
        if data is None:
            data = self.build(minw, maxw, **kw)
        vals = data[by].unique()
        for v in vals:
            sub = data[data[by] == v].copy()
            out[v] = sub
        return out
    
    def build(self, minw=4, maxw=7):
        # {pid: [[week_idx, state, action], ...], ...}
        data = []
        cols = ["week", "pid", "state", "cluster", "action_sids", \
                "msg_ids", "pmsg_sids", "paction_sids", "pmsg_ids", "qids", "response"]
        for w in range(minw, maxw+1):
            states, clusters, ids, actsids, \
                msg_ids, pmsgsids, pactsids, pmsg_ids, questions, responses = self.weekly_state_data(w)
            for i in range(len(responses)):
                st, clt, pid = states[i], clusters[i], ids[i]
                st = st.tolist()[:5]
                clt = clt.item()
                pid = pid.item()
                qids = [], []
                pids, p_sids = [], []
                if i < len(questions):
                    qids = questions[i]
                try:
                    resp = responses[i]
                    for j in range(len(resp)):
                        if pd.isna(resp[j]):
                            resp[j] = 0
                        resp[j] = int(resp[j])
                except:
                    resp = [-1,-1]
                row = [w, pid, st, clt, actsids[i], msg_ids[i], pmsgsids[i], pactsids[i], \
                           pmsg_ids[i], qids, resp]
                data.append(row)
        data = pd.DataFrame(data, columns=cols)
        data = data.sort_values(by="week", ascending=True)
        return data
    
    def weekly_state_data(self, w):
        states = self.rep.replay("states").week(w).load()
        actions = self.rep.replay("actions").week(w).load()
        pactions = self.rep.replay("actions").week(w-1).load()
        clust = self.rep.replay("clusters").week(w).load()
        ids = self.rep.replay("ids").week(w).load()
        resp = self.rep.replay("responses").week(w).load()
        k = states.shape[0]
        action_sids = []
        msg_ids = []
        paction_sids = []
        pmsg_ids = []
        questions = []
        responses = []
        pmsg_sids = []
        for c, pid in enumerate(ids):
            r = resp[resp['ID']==pid.item()]
            if r.shape[0] == 0:
                continue
            msg_ids.append(self.msgh.mid_lookup(actions[c,1]))
            pmsg_ids.append(self.msgh.mid_lookup(pactions[c,1]))
            action_sids.append(self.msgh.sid_lookup(actions[c,1]))
            qrow = [r['Q1_ID'].item(), r['Q2_ID'].item()]
            paction_sids.append(self.qsnh.sid_lookup(qrow))
            pmsg_sids.append(self.msgh.sid_lookup(pactions[c,1]))
            questions.append(qrow)
            rrow = [r['Q1_response'].item(), r['Q2_response'].item()]
            responses.append(rrow)
        return states, clust, ids, action_sids, msg_ids, pmsg_sids,\
                    paction_sids, pmsg_ids, questions, responses
    
    def analyze(self, data):
        out = {}
        for k in data.keys():
            out[k] = self._analyze(data[k])
        return pd.DataFrame(out).T
    
    def _analyze(self, data):
        def statesmat(data):
            mat = np.zeros((data.shape[0],5))
            st = data["state"].values
            for i, s in enumerate(st):
                mat[i] = s
            return mat
        def count_resp(data):
            c = []
            for r in data["response"].values:
                ci = 0
                for ri in r:
                    if ri not in [1,2,3]:
                        continue
                    ci += 1
                c.append(ci)
            return np.array(c)
        mat = statesmat(data)
        running_change = np.zeros((mat.shape[0]-1,mat.shape[1]))
        weeks = []
        for i in range(mat.shape[0]-1):
            running_change[i] = mat[i+1]-mat[i]
            weeks.append(data["week"].iloc[i])
        response_count = count_resp(data)
        return {"running_change": running_change,
                "response_count": np.array(response_count),
                "mat": mat,
                "week": weeks,}
    
    def rank_by(self, data, col, fn):
        series = data[col].map(fn)
        series = series.sort_values(ascending=False)
        return data.loc[series.index]
    
        # find top samp% of responders
    def active_responders(self, _samp, data):
        responders = self.rank_by(data, "response_count", np.sum).index
        print(responders.shape[0])
        samp = int(responders.shape[0] * _samp)
        m = data.iloc[0]['running_change'].shape[1]
        responders_sample = responders[:samp]
        changes = np.zeros((samp,m))
        ids = []
        start = np.zeros((samp, m))
        end = start.copy()
        rcs = []
        for i, rs in enumerate(responders_sample):
            resp = data.loc[rs]
            ids.append(rs)
            start[i,:] = resp["mat"][0]
            end[i,:] = resp["mat"][-1]
            rcs.append(resp["response_count"])
        return {"samp": _samp, "ids": ids, "start": start, "end": end, "counts": rcs}
    
    def calc_state_elem_change(self, data):
        data = data['mat']
        start = data.map(lambda x: x[0,:])
        end = data.map(lambda x: x[-1,:])
        out = [] # [[start1, end1], [start2, end2]]
        L = len(start.iloc[0])
        for i in range(L):
            si = start.map(lambda x: x[i])
            ei = end.map(lambda x: x[i])
            out.append({"si": i+1, "start": si.values, "end": ei.values})
        return out


    
