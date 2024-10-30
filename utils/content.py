import pandas as pd
import itertools as itt
import numpy as np
import torch
import json
import os
from collections import OrderedDict

class MessageHandler:
    # handle action space/messages and organizing by state element

    def __init__(self, path='arogya_content/mDiabetes-content-final.xlsx',
            core_timeline_path='arogya_content/core_message_timeline_map.csv',
            sheet='mDiabetes Content-AI',
            path_prepend='',
            detailed=False):
        self.detailed = detailed
        self.path = path_prepend + path
        self.core_timeline_path = path_prepend + core_timeline_path
        self.sheet = sheet
        self.action_space = self.create_action_space()
        self.N = self.action_space.shape[0]
        self.core_timeline_map = self.read_timeline_map()

    def read_and_process(self):
        mat = pd.read_excel(self.path, sheet_name=self.sheet)
        mat.columns = mat.iloc[0]
        mat.drop(0, axis=0, inplace=True)
        if self.detailed:
            mat = mat[['Sl. No', 'Core', 'DetailedStateElementID']]
            mat.rename({'DetailedStateElementID': 'StateElementID'}, axis=1, inplace=True)
        else:
            mat = mat[['Sl. No', 'Core', 'StateElementID']]
        mat.rename({'Sl. No': 'ID'}, axis=1, inplace=True)
        mat = mat.astype(int)
        return mat

    def create_action_space(self):
        mat = self.read_and_process()
        iscore = lambda x: mat.loc[x]['Core']
        elemid = lambda x: mat.loc[x]['StateElementID']
        action_space = {}
        for i, (m1, m2) in enumerate(itt.combinations(mat['ID'], 2)):
            m1_core = iscore(m1)
            m2_core = iscore(m2)
            m1_elemid, m2_elemid = elemid(m1), elemid(m2)
            row = {"M1_ID": m1, "M2_ID": m2, 
                    "M1_CORE": m1_core, "M2_CORE": m2_core,
                    "M1_StateElementID": m1_elemid,
                    "M2_StateElementID": m2_elemid,
                }
            action_space[i] = row
        return pd.DataFrame(action_space).T

    def random_core_actions(self, n):
        core = self.action_space[(self.action_space['M1_CORE'] == True) & \
                (self.action_space['M2_CORE'] == True)]
        replace = False
        if n > len(core):
            replace = True
        core = core.sample(n=n, replace=replace)
        return core.index.tolist()

    def scheduled_core_actions(self, timeline):
        # use the timeline to find which messages to send
        timeline = timeline[:,1].long().tolist()
        actions = []
        def find(msgs):
            a = self.action_space
            row = a[(a['M1_ID'] == msgs[0]) & (a['M2_ID'] == msgs[1])]
            return row.index[0]
        for tl in timeline:
            msgs = self.core_timeline_map[tl]
            try:
                act = find(msgs)
            except:
                act = find(msgs[::-1])
            actions.append(act)
        return actions

    def messages_from_action(self, action_id):
        if isinstance(action_id, torch.Tensor):
            action_id = action_id.item()
        row = self.action_space.loc[action_id]
        messages = (row['M1_ID'], row['M2_ID'])
        elems = (row['M1_StateElementID'], row['M2_StateElementID'])
        return (messages, elems)

    def mid_lookup(self, action_id):
        return self.messages_from_action(action_id)[0]

    def sid_lookup(self, action_id):
        return self.messages_from_action(action_id)[1]

    def duplicate_sid(self, action_id):
        sid = self.sid_lookup(action_id)
        return sid[0] == sid[1]

    def read_timeline_map(self):
        with open(self.core_timeline_path, 'r') as fp:
            tl = fp.read().splitlines()[1:]
        tlmap = {}
        for t in tl:
            t = t.partition(',')
            k = int(t[0])
            v = [int(_) for _ in t[-1].split(',')]
            tlmap[k] = v
        return tlmap

class QuestionHandler:
    # handles access to the question bank

    def __init__(self, path='arogya_content/mDiabetes-content.xlsx',
            sheet_name='mDiabetes Questions-AI(English)',
            path_prepend='',
            map="question_state_element_map.json"):
        self.path_pre = path_prepend
        self.path = path_prepend + path
        self.sheet_name = sheet_name
        self.map = map
        self.N = 0
        self.question_map = self.create_question_map()

    def create_question_map(self):
        # read in and create in memory the dictionary to map
        # from state element IDs --> question IDs
        questions = pd.read_excel(self.path, sheet_name=self.sheet_name)
        questions = questions[["ID"]]
        questions["ID"] = questions["ID"].astype(int)
        self.N = len(questions)
        with open(self.path_pre+self.map, 'r') as fp:
            qmap = json.loads(fp.read())
        question_map = {int(k): [] for k in qmap.keys()}
        for k, v in qmap.items():
            for qid in v:
                question_map[int(k)].append(qid)
        return question_map
    
    def sid_lookup(self, qs):
        sids = []
        for q in qs:
            for k in self.question_map.keys():
                if q in self.question_map[k]:
                    sids.append(k)
                    break
        return sids

    def random_questions(self, state_elems):
        qs1 = self.question_map[state_elems[0]]
        qs2 = self.question_map[state_elems[1]]
        q1 = np.random.choice(qs1)
        q2 = q1
        while q2 == q1:
            q2 = np.random.choice(qs2)
        return q1, q2

class Questionnaire:
    # logic for reading in an individual questionnaire and calculating states

    def __init__(self, pref, lang, map, endline=False): 
        self.pref = pref
        self.lang = lang
        self.spath = map
        self.smap = self.read_state_map()
        if endline:
            self.path = "arogya_content/GAI4SG-Endline-Responses.xlsx"
        else:
            self.path = os.path.join("arogya_content", f"{self.pref}_baseline_questionnaires", f"mDiabetes-baseline-{self.lang}.xlsx")
        try:
            self.mat = pd.read_excel(self.path)
        except:
            self.mat = None
        self.preprocess()

    def read_state_map(self):
        # read the predefined json state map which determines how questions
        # and answers should be converted to a state vector
        with open(self.spath, "r") as fp:
            smap = json.loads(fp.read())
            smap['dynamic'] = OrderedDict(smap['dynamic'])
            smap['fixed'] = OrderedDict(smap['fixed'])
            return OrderedDict(smap)

    def compute_states(self):
        # store the ID and calculate the state of participants
        whatsapps, states, elemlist = [], [], []
        if self.mat is None:
            return whatsapps, states, elemlist
        for i in range(self.mat.shape[0]):
            whatsapp = str(self.mat.iloc[i]['18'])
            if whatsapp is None or whatsapp == '':
                continue
            whatsapp = int(whatsapp)
            st, elemlist = self.compute_participant_state(i)
            whatsapps.append(whatsapp)
            states.append(st)
        return whatsapps, states, elemlist

    def compute_participant_state(self, i):
        # perform the logic defined in the state map
        participant = self.mat.iloc[i]
        participant_state = []
        state_elems = []
        if self.mat is None:
            return participant_state
        for skey in ['dynamic', 'fixed']:
            for state_elem, block in self.smap[skey].items():
                val, count = 0, 0
                for method, column, low, medium, high in block:
                    if column not in participant:
                        continue
                    entry = participant[column]
                    if isinstance(entry, float) and np.isnan(entry):
                        count += 1
                        continue
                    if method == "match":
                        participant_entry = entry.partition(" ")[0]
                    elif method == "count":
                        participant_entry = len(entry.split(","))
                    elif method == "mult":
                        participant_entry = len(entry.split(","))
                        # workaround to consider multiple select question as multiple questions
                        val += (3*participant_entry)
                        # high[0] contains the total number of questions
                        count += high[0] - 1
                        participant_entry = "Ignored"
                    if participant_entry in low:
                        val += 1
                    elif participant_entry in medium:
                        val += 2
                    elif participant_entry in high:
                        val += 3
                    count += 1
                participant_state.append(val/count if count > 0 else 0)
                if (skey == 'dynamic'):
                    state_elems.append(state_elem)
        return participant_state, state_elems

    def get_SID_translation_list(self, questions):
        sidlist = []
        for question in questions:
            toBreak = False
            for id, (state_elem, block) in enumerate(self.smap['dynamic'].items()):
                # don't repeatedly look for same question
                if (toBreak):
                    toBreak = False
                    break
                for method, column, low, medium, high in block:
                    if question == column:
                        sidlist.append(id + 1)
                        toBreak = True
                        break
                        
        return sidlist


    def preprocess(self):
        # clean up the raw questionnaires
        if self.mat is None:
            return
        def colid(c):
            cid = ""
            try:
                idx = c.index("[")
                part = c[idx+1:].partition(".")
                cid = part[0] + part[1] + part[2].split(" ")[0]
            except:
                cid = c.partition('.')[0]
            return cid
        cols = self.mat.columns.tolist()
        for i in range(len(cols)):
            if cols[i] == 'Timestamp':
                cols[i] = "0"
            cols[i] = colid(cols[i])
        self.mat.columns = cols
        if self.pref == "pilot":
            self.mat.drop([0,1], axis=0, inplace=True)
        self.mat.dropna(subset=['18'], inplace=True)
        self.mat['18'] = self.mat['18'].astype('int64')
        self.mat.drop_duplicates(subset=['18'], inplace=True)

class StatesHandler:
    # handles multiple sets of states (from questionnaires) in one

    def __init__(self, pref="preprod", langs=['english', 'hindi', 'kannada'], map="map.json", endline=False):
        self.pref = pref
        self.langs = langs
        self.qhs = [Questionnaire(self.pref, l, map=map, endline=endline) for l in self.langs]
        self.state_max = 3
        self.N_elem = None

    def get_SID_translation_list(self, questions):
        return self.qhs[2].get_SID_translation_list(questions)
        
    def compute_states(self):
        # compute the states for all questionnaire groups
        # and merge into one 
        whatsapps, states = [], []
        for qh in self.qhs:
            wa, st, statelist = qh.compute_states()
            for i in range(len(wa)):
                if wa[i] in whatsapps:
                    continue
                whatsapps.append(wa[i])
                states.append(st[i])
        if len(states) > 0:
            self.N_elem = len(states[0])
        whatsapps = torch.tensor(whatsapps).long()
        states = torch.tensor(states)
        return whatsapps, states, statelist


# if __name__ != "__main__":
    # StatesH = StatesHandler()
    # MessagesH = MessageHandler()
    # QuestionsH = QuestionHandler()

if __name__ == "__main__":
    import sys
    states = StatesHandler(sys.argv[1])
    m = states.qhs[-1].mat
    print(m.columns)
    w, s = states.compute_states()
    print(w)
    print(s)
    print(s.shape)
