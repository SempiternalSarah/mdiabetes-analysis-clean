import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OrdinalEncoder
from sklearn.mixture import GaussianMixture
import sklearn.cluster
import matplotlib.pyplot as plt
from utils.state_data import StateData
from utils.content import StatesHandler, QuestionHandler, MessageHandler
from torch import save, load
import torch
import json

# class to manage loading and encoding behavioral data
class BehaviorData:
    
    def __init__(self, 
                 minw=2, maxw=31, 
                 include_pid=False, include_state=True, 
                 active_samp=None, 
                 window=3,
                 train_perc=.8,
                 expanded_states=True,
                 top_respond_perc=1.0,
                 full_questionnaire=False,
                 insert_predictions=False,
                 num_weeks_history=1,
                 one_hot_response_features=True,
                 response_feature_noise=.05,
                 max_state_week=1,
                 split_model_features=True,
                 split_weekly_questions=False,
                 category_specific_history = False,
                 no_response_class = False,
                 regression = False,
                 num_clusters = 3,
                 cluster_by = None,
                 cluster_method = "Kmeans",
                 only_rnr = False,
                 predictStates = False):
        # minw, maxw: min and max weeks to collect behavior from
        # include_pid: should the participant id be a feature to the model
        # include_state: should the participant state be a feature
        self.oneHotResponseFeatures = one_hot_response_features
        # whether to use feature modifications to replace non responses with predicted responses
        self.insert_predictions = insert_predictions
        # whether to use each question in the questionnaire as features
        self.full_questionnaire = full_questionnaire
        # what percent of participants to use (taken from the top responders)
        self.top_respond_perc = top_respond_perc
        # min and max weeks to load data from 
        # (2 and 31 for the whole dataset, other values probably break things now)
        self.minw, self.maxw = minw, maxw
        # whether to use participant ID as a feature
        # not used
        self.include_pid = include_pid
        # whether to use new (expanded) state mappings
        # default to true
        self.expanded_states = expanded_states
        # number of weeks of history (question/message/response) values to include
        self.num_weeks_history = min(num_weeks_history, 23)
        # whether to include state values as features at all
        # default to true
        self.include_state = include_state
        # how many weeks to include non-zero state features
        self.max_state_week = max_state_week
        # not used
        self.active_samp = active_samp if active_samp is not None else 1
        # not used
        self.window = window
        # clustering info
        self.cluster_by = cluster_by
        self.num_clusters = num_clusters
        self.cluster_method = cluster_method
        # insert noise to response features
        self.responseFeatureNoise = response_feature_noise
        # calculate file name for storing/loading data
        # whether to zero out state features (off at first, experiment.py will turn on)
        self.zeroStateFeatures = False
        # whether to have one question per week
        self.split_weekly_questions = split_weekly_questions
        # whether to consider only questions in the same category for history features
        self.category_specific_history = category_specific_history

        # adds extra features denoting the category of each question
        # (consumption, exercise, knowledge)
        self.split_model_features = split_model_features
        self.no_response_class = no_response_class
        self.regression = regression
        self.only_rnr = only_rnr
        self.predictStates = predictStates

        self.fname = f"./saved_data/{self.minw}-{self.maxw}{self.include_pid}{self.include_state}{self.max_state_week}{self.expanded_states}{self.full_questionnaire}{self.num_weeks_history}{self.category_specific_history}{self.oneHotResponseFeatures}{self.top_respond_perc}{self.split_model_features}{self.split_weekly_questions}{self.no_response_class}{self.regression}{self.only_rnr}{self.num_clusters}{self.cluster_by}{self.cluster_method}{self.predictStates}.pickle"

        # print(self.include_state, self.expanded_states, self.full_questionnaire)

        # data saved - we can just load it
        if os.path.exists(self.fname):
            self.load()
        else:
            self.data = self.build()
            self.features, self.labels, self.featureList = self.encode(self.data)
            self.save()

        # calculate mask to zero out state values if later desired
        self.stateZeroMask = torch.where(torch.tensor(self.featureList == "state"), 0, 1)

        # print(self.featureList)
        # find index of first response value so we don't have to compute this again
        # used to replace non responses with the model's prediction
        for idx, feature in enumerate(self.featureList):
            if ("response" in feature):
                self.responseIdx = idx
                break
        

        self.splitData(train_perc)

        # set up our response modifications
        # these will be used to replace non responses with the model's prediction
        self.responseMods = {}
        for idx in self.train:
            self.responseMods[idx] = np.zeros_like(self.chunkedFeatures[idx])
        for idx in self.test:
            self.responseMods[idx] = np.zeros_like(self.chunkedFeatures[idx])


        
    # splits data into test and training
    def splitData(self, train_perc):
        numParticipants = len(self.nzindices)
        numTrainParticipants = int(train_perc * numParticipants)
        self.train = np.random.choice(numParticipants, numTrainParticipants, replace=False)
        self.test = np.array([idx for idx in range(numParticipants) if idx not in self.train])
    
        self.chunkedFeatures = torch.tensor_split(self.features, self.nzindices)
        self.chunkedLabels = torch.tensor_split(self.labels, self.nzindices)


    # get features for a participant
    def get_features(self, idx):
        if (self.insert_predictions):
            toReturn = self.chunkedFeatures[idx] + self.responseMods[idx]
        else:
            toReturn = self.chunkedFeatures[idx].clone()
        if (self.responseFeatureNoise > 0):
            toReturn = self.add_feature_noise(toReturn, idx)
        if (self.zeroStateFeatures):
            toReturn = self.stateZeroMask * toReturn
        return toReturn
    
    def add_feature_noise(self, data, indx):
        # in this case, response features are the hard predicted classes
        # adding noise doesn't make much sense so return without doing anything
        if (not self.oneHotResponseFeatures):
            return data

        if (self.split_weekly_questions):
            numQs = 1
            k = self.dimensions[1]
        else:
            numQs = 2
            k = self.dimensions[1] // 2

        # iterate through each week (row of this participants data)
        for i, weekRow in enumerate(self.chunkedFeatures[indx]):
            # iterate through the responses of weeks before this one
            for j in range(self.num_weeks_history - 1):
                # get index of this week's response to q1
                idx = self.responseIdx + (k * numQs * j)
                for offset in range(numQs):
                    curIdx = idx + k*offset
                    replace = data[i][curIdx:curIdx + k]
                    if torch.sum(replace) > 1.01:
                        print("Some problem", replace)
                    # no feature here yet - may be replaced by predictions later
                    if (not replace.sum() > 0):
                        continue
                    replace += torch.normal(mean=torch.zeros(k), std=self.responseFeatureNoise * torch.ones(k))
                    # re-normalize labels to ensure no < 0 and that sum = 1
                    replace[replace < 0] = 0
                    # divide each row by sum of that row
                    replace /= replace.sum()
                    # replace row with normalized
                    data[i][curIdx:curIdx + k] = replace
        return data

    # set feature modifications for all participants
    def set_feature_response_mods(self, indx, preds):
        # do nothing if we're not inserting predictions
        # modifications will remain 0
        if (not self.insert_predictions):
            return
        if (self.split_weekly_questions):
            numQs = 1
            k = self.dimensions[1]
        else:
            numQs = 2
            k = self.dimensions[1] // 2
        # set up our feature modifications
        mods = np.zeros_like(self.chunkedFeatures[indx].numpy())
        # iterate through each week (row of this participants data)
        for i, weekRow in enumerate(self.chunkedFeatures[indx]):
            # iterate through the responses of weeks before this one
            for j in range(self.num_weeks_history - 1):
                # weeks before start should be 0
                if (i - j) < 0:
                    break
                # get index of this week's response to q1
                if (self.oneHotResponseFeatures):
                    idx = self.responseIdx + (numQs * k * j)
                    for offset in range(numQs):
                        curIdx = idx + k*offset
                        if weekRow[curIdx] == -1:
                            mods[i][curIdx:curIdx + k] = 1 + preds[j][offset*k:(k*offset)+k]
                else:
                    idx = self.responseIdx + (numQs * j)
                    for offset in range(numQs):
                        if weekRow[idx + offset] == -1:
                            # first question
                            if (offset == 0):
                                # calculate most likely predicted class and save to use as the feature
                                mods[i][idx + offset] = 2 + np.argmax(preds[j][0:k])
                                # need to add 2 (feature itself is -1, argmax is 0 if pred class is 1)
                            else:
                                mods[i][idx + offset] = 2 + np.argmax(preds[j][k:])
        self.responseMods[indx] = mods
                    


    # load state information from the baseline questionnaire
    def load_questionnaire_states(self, endline=False):
        if (self.full_questionnaire):
            sh = StatesHandler(map="map_individual.json", endline=endline)
            if (self.expanded_states):
                shForIds = StatesHandler(map="map_detailed.json", endline=endline)
            else:
                shForIds = StatesHandler(map="map.json", endline=endline)
        elif (self.expanded_states):
            sh = StatesHandler(map="map_detailed.json", endline=endline)
        else:
            sh = StatesHandler(map="map.json", endline=endline)
        whatsapps, states, qlist = sh.compute_states()
        states = states.numpy()


        # load question state IDs
        if(self.full_questionnaire):
            if (self.expanded_states):
                maxSVal = 17
            else:
                maxSVal = 5
            def _padded_binary(a, b):
                # helper function to binary encode a and 
                # pad it to be the length of encoded b
                a, b = int(a), int(b)
                l = len(format(b,"b"))
                a = format(a,f"0{l}b")
                return np.array([int(_) for _ in a])
            finalStates = []
            statelist = shForIds.get_SID_translation_list(qlist)
            # append dynamic question values along with their state IDs
            for idx, state in enumerate(statelist):
                finalStates.append(states[:, idx])
                # print(_padded_binary(state, maxSVal))
                idVals = _padded_binary(state, maxSVal)
                for idVal in idVals:
                    finalStates.append(np.repeat(idVal, states.shape[0]))
            # append remaining (static demographic information) question values
            for idx in range(len(statelist), states.shape[1]):
                finalStates.append(states[:, idx])
            states = np.array(finalStates).transpose()


        def modify_whatsapp(x):
            # helper function to parse the whatsapp numbers
            x = str(x)
            x = x[len(x)-10:]
            return int(x)
        participantIDs = torch.tensor(np.loadtxt("arogya_content/all_ai_participants.csv", delimiter=",", skiprows=1, dtype="int64"))
        participantIDs[:, 1].apply_(modify_whatsapp)
        start_weeks = self.get_participant_start_weeks()
        
        # filter responses to only include ones in the AI participant set
        isect, idIdxs, stateIdxs = np.intersect1d(participantIDs[:, 1], whatsapps, return_indices=True)
        # combine the glific IDs with the states into a dictionary and return
        return dict(zip(participantIDs[idIdxs, 0].numpy(), states[stateIdxs])), start_weeks

    def get_participant_start_weeks(self):
        # get start week for each participant
        startWeeks = {}
        for week in range(2, 32):
            pids = np.loadtxt(f"./local_storage/prod/responses/participant_responses_week_{week}.csv", delimiter=",", skiprows=1, dtype="str")
            pids = pids[:, 0].astype("int64")
            for pid in pids:
                if pid not in startWeeks:
                    startWeeks[pid] = week
        return startWeeks

    
    def filter_top_responders(self, df: pd.DataFrame):
        # get WEEKLY response count for each participant
        df["response_count"] = df.apply(lambda row: np.count_nonzero(row["response"]), axis=1)
        # sum response count for each participant
        counts = df.groupby("pid")['response_count'].sum().sort_values()
        # find cutoff value (participants with fewer total responses are removed)
        cutoff_idx = int((len(counts)) * (1 - self.top_respond_perc))
        # print(counts)
        print(counts[cutoff_idx:])
        # print(cutoff_idx, len(counts))
        # select top responders
        df = df[df["pid"].isin(counts[cutoff_idx:].keys())]
        # save response counts for later use
        self.counts = counts

        return df

    def get_weekly_response_rates(self, raw=True, anyr=True):
        # sum response count for each week
        if (raw):
            elem = "weekraw"
        else:
            elem = "week"
        if (anyr):
            def aResponse(row):
                if np.count_nonzero(row["response"]) > 0:
                    return 1
                else:
                    return 0
            self.data["nzero_response"] = self.data.apply(aResponse, axis=1)
            counts = self.data.groupby(elem)['nzero_response'].sum()
            totals = self.data.groupby(elem)['nzero_response'].count()
        else:
            counts = self.data.groupby(elem)['response_count'].sum()
            totals = 2 * self.data.groupby(elem)['response_count'].count()
        return (counts/totals).values
    
    def assign_cluster_features(self, d: pd.DataFrame):
        if self.cluster_by is None:
            return d
        # keep reference to original dataframe to modify and return
        df = d
        # grab the last week row for each participant
        if (self.cluster_by == "History" or self.cluster_by == "All"):
            week = d['week'].max()
        else:
            week = 0
        d = d[d['week'] == week]
        # make copy just to be safe
        d = d.copy()

        if (self.cluster_by == "History"):
            cols = [col for col in d.columns if "_last_" in col]
            vals = d[cols].to_numpy()
        elif (self.cluster_by == "Initial"):
            vals = np.array([val for val in d["state"]])
        elif (self.cluster_by == "All"):
            cols = [col for col in d.columns if "_last_" in col]
            vals = d[cols].to_numpy()
            # print(vals.shape)
            vals2 = np.array([val for val in d["state"]])
            # print(vals2.shape)
            vals = np.concatenate([vals, vals2], axis=1)
            # print(vals.shape)
        elif (self.cluster_by == "Demographics"):
            vals = np.array([val for val in d["state"]])
            vals = vals[:, -5:]
            # print(vals)
        else:
            print("INVALID CLUSTER OPTION")
            exit(-1)
        # unpack any tuples inside the array
        unpacked = []
        for row in vals:
            unpackedRow = []
            for entry in row:
                if (type(entry) is tuple):
                    for item in entry:
                        unpackedRow.append(item)
                else:
                    unpackedRow.append(entry)
            unpacked.append(unpackedRow)
        vals = np.array(unpacked)

        # normalize each column to have a max of 1
        vals = vals / vals.max(axis=1)[:,None]

        # form clusters
        if (self.cluster_method == "Kmeans"):
            kmeans = sklearn.cluster.KMeans(n_clusters=self.num_clusters).fit(vals)
            labels = kmeans.labels_
        elif (self.cluster_method == "Gaussian"):
            gm = GaussianMixture(n_components=self.num_clusters).fit(vals)
            labels = gm.predict(vals)
        elif (self.cluster_method == "Spectral"):
            labels = sklearn.cluster.SpectralClustering(n_clusters=self.num_clusters).fit_predict(vals)
        # print(labels)
        # add cluster number to every row for each participant
        lookup = dict(zip(d["pid"].to_numpy(), labels))
        def setCluster(row):
            return lookup[row["pid"]]
        df["cluster"] = df.apply(setCluster, axis=1)
        # print (df["cluster"])
        return df


    def build(self):
        # call StateData and build our initial unencoded dataset
        sd = StateData(detailed=self.expanded_states)
        d = sd.build(minw=self.minw, maxw=self.maxw)
        enc = OrdinalEncoder().fit_transform
        # load dictionary of pids to states
        init_states, start_weeks = self.load_questionnaire_states()
        final_states, temp = self.load_questionnaire_states(endline=True)
        # filter out rows that aren't supposed to be here
        # unsure where they come from but they aren't listed in the all_ai_participants.csv
        d = d[d["pid"].isin(list(dict.keys(init_states)))]

        d["weekraw"] = d["week"]
        # adjust week values per participant (their first week should be 0, last 23)
        def adjustWeek(row):
            w = row["week"] - start_weeks[row["pid"]]
            return w
        d["week"] = d.apply(adjustWeek, axis=1)

        # participants are in for 24 weeks (??)
        # first response from final group of participants seen week 8
        # last responses received week 31
        # THIS SEEMS CORRECT FROM THE DATA BUT POSSIBLE IM MISSING SOMETHING

        # filter out rows after participant completed study
        d = d[d["week"] < 24]

        # change the computed states to be the initial questionnaire states instead
        def replaceState(row):
            if (row['week'] >= self.max_state_week):
                return np.zeros_like(init_states[row["pid"]])
            else:
                return init_states[row["pid"]]
        d["state"] = d.apply(replaceState, axis=1)
        def fillFinalState(row):
            if row["pid"] in final_states:
                return final_states[row["pid"]]
            else:
                return np.NaN
        d["finalState"] = d.apply(fillFinalState, axis=1)
        if (self.predictStates):
            d = d.dropna(subset=["finalState"])

        # rescale pid values in case we want to use them as features
        d["pidFeat"] = enc(d["pid"].values.reshape(-1,1)).astype(int)
        # sort by week
        d = d.sort_values(by="week")
        # sort by participant, keeping the week sort
        # result is sorted by participants
        # and participant rows are sorted in increasing week order
        d = d.sort_values(by="pidFeat", kind="stable")

        # select top responders based on parameter passed in constructor
        d = self.filter_top_responders(d)
        # insert feature for question categories
        if self.split_model_features:
            with open("detailed_question_state_element_map.json", 'r') as fp:
                qmap = json.loads(fp.read())
            qCatDict = {}
            for key in qmap.keys():
                for elem in qmap[key]:
                    if int(key) < 9:
                        qCatDict[elem] = 0
                    elif int(key) < 13:
                        qCatDict[elem] = 1
                    else:
                        qCatDict[elem] = 2
            def get_category_num(row):
                toReturn = []
                for entry in row["qids"]:
                    toReturn.append(qCatDict[entry])
                return tuple(toReturn)

            d["qcats"] = d.apply(get_category_num, axis=1)


        # check if we need to build up the full sequence
        # method to insert a week's element into all other rows
        def construct_week_elem(row, weekOffset, elem):
            weekno = row['week'] - weekOffset
            if weekno >= 0 and ((weekOffset > 0) or (elem != "response" and weekOffset == 0)):
                if (self.category_specific_history):
                    temp = d[d["pid"] == row['pid']]
                    temp = temp[temp["week"] <= row["week"]]

                    vals = []
                    for idx, qcat in enumerate(row['qcats']):
                        temp2 = temp[temp["qcats"].str[0] == qcat]
                        entryNo = len(temp2.index) - weekOffset - 1
                        # if we have enough history for this category
                        if (entryNo >= 0):
                            entry = temp2.iloc[entryNo][elem][idx]
                            if (elem == "response" and entry == 0):
                                vals.append(-1)
                            else:
                                vals.append(entry)
                        # otherwise, use recent of any category
                        else:
                            vals.append(temp.iloc[len(temp.index) - weekOffset - 1][elem][idx])
                    return tuple(vals)
                else:
                    # use full knowledge of the past
                    temp = d[d["pid"] == row['pid']]
                    temp = temp[temp["week"] == weekno]
                    # set non response to -1 to distinguish from undefined (before the study) response
                    if (elem == "response" and 0 in temp.iloc[0][elem]):
                        vals = []
                        for val in temp.iloc[0][elem]:
                            if (val == 0):
                                vals.append(-1)
                            else:
                                vals.append(val)
                        return tuple(vals)
                    else:
                        return tuple(temp.iloc[0][elem])
            else:
                # don't use knowledge of the future
                return (0, 0)
        # go through and insert msg/question/response for all weeks as appropriate
        for elem in ["pmsg_sids", "paction_sids", "pmsg_ids", "qids", "response"]:
            # print(elem)
            for week in range(self.num_weeks_history + 1):
                d[f"{elem}_last_{week}"] = d.apply(lambda row: construct_week_elem(row, week, elem), axis=1, result_type='reduce')

        # record splits between different participants for later
        # basically we want to easily extract individual participant data
        # indices will REMAIN THE SAME for the encoded pytorch tensor
        # nonzero() returns a 1 element tuple, unpack, take 0th entry off (it's 0)
        # convert to list for later use
        if (self.predictStates):
            d = d[d["week"] == 23]
        self.nzindices = d["pid"].diff().to_numpy().nonzero()[0][1:].tolist()
        # if we're splitting the rows into 2, double split values
        if (self.split_weekly_questions):
            for x in range(len(self.nzindices)):
                self.nzindices[x] = self.nzindices[x] * 2
        d = self.assign_cluster_features(d)
        

        return d
    
    def encode(self, data: pd.DataFrame):
        # encode the row locations of data
        # data: pd.DataFrame
        X, Y = [], []
        for idx, row in data.iterrows():
            x1, x2, y1, y2, featureList = self.encode_row(row)
            X.append(x1)
            Y.append(y1)
            # returns not none x2 if data is one question per row
            if x2 is not None:
                X.append(x2)
                Y.append(y2)
        # for x in X:
        #     print(x.shape)

        X = np.stack(X)
        Y = np.stack(Y)
        # print(X.shape, Y.shape, len(featureList))
        return torch.tensor(X).float(), torch.tensor(Y).float(), np.array(featureList)
                
    def encode_row(self, row):
        # here we take a row from the main behavior dataset and 
        # encode all of the features for our model
        # Features:
        #  - participant ID                    (enumeration of participants) (NOT USED NORMALLY)
        #  - dynamic state elements            (real values between (1,3)
        #  - action id which prompted question (enumerated 1-5,  binary encoded)
        #  - message ids of action             (enumerated 1-57, binary encoded)
        #  - question ids to predict repsonse  (enumerated 1-32, binary encoded
        def _padded_binary(a, b):
            # helper function to binary encode a and 
            # pad it to be the length of encoded b
            a, b = int(a), int(b)
            l = len(format(b,"b"))
            a = format(a,f"0{l}b")
            return np.array([int(_) for _ in a])
        def _onehot(a, l):
            vec = np.zeros(l)
            vec[a] = 1
            return vec

        featureList = []

        elems = []
        # max value for each (state elem, message id, question id) for padding
        ls = []
        if (self.expanded_states):
            maxSVal = 17
        else:
            maxSVal = 5
        for week in range(self.num_weeks_history):
            for elem in ["pmsg_sids", "paction_sids", "pmsg_ids", "qids"]:
                elems.append(f"{elem}_last_{week}")
            ls += [maxSVal,maxSVal,57,32]
        feats_to_enc = np.array(row[elems].values)
        
        if self.include_pid:
            X = np.array([row["pidFeat"]])
            featureList.append("pidFeat")
        else:
            X = np.array([])
        if self.include_state:
            X = np.append(X, row["state"])
            featureList += ["state"] * len(row["state"])

        def onehot_response(a, l):
            # unchanged if future/unknown response
            vec = np.zeros(l)
            if (a > 0):
                # 1 0 0 for class 1
                # 0 1 0 for class 2
                # 0 0 1 for class 3
                vec[a - 1] = 1
            elif (a < 0):
                # -1 -1 -1 for non response
                vec -= 1
            return vec

        for week in range(1, self.num_weeks_history):
            if (self.oneHotResponseFeatures):
                for r in row[f"response_last_{week}"]:
                    if (self.only_rnr):
                        length = 2
                        encoding = onehot_response(min(r, 1), 2)
                    elif (self.no_response_class):
                        length = 4
                        encoding = onehot_response(r, 4)
                    else:
                        length = 3
                        encoding = onehot_response(r, 3)
                    X = np.append(X, encoding)
                featureList += [f"response_last_{week}_q1"] * length
                featureList += [f"response_last_{week}_q2"] * length
            else:
                X = np.append(X, row[f"response_last_{week}"])
                featureList += [f"response_last_{week}_q1", f"response_last_{week}_q2"]

        for j in range(len(feats_to_enc)):
            for k in range(len(feats_to_enc[j])):
                # encode the feature and add it to our feat vector
                if (feats_to_enc[j][k] > ls[j]):
                    print("ERROR!!!!!!")
                bin_feat = _padded_binary(feats_to_enc[j][k],ls[j])
                # print(len(bin_feat), feats_to_enc[j][k], elems[j], ls[j])
                X = np.append(X, bin_feat)
                featureList += [f"{elems[j]}_q{k+1}"] * len(bin_feat)

        if self.cluster_by != None:
            bin_feat = _padded_binary(row["cluster"], self.num_clusters)
            X = np.append(X, bin_feat)
            featureList.append("ClusterNum")
        if self.split_model_features:
            for idx, qid in enumerate(row["qcats"]):
                bin_feat = _padded_binary(qid, 3)
                X = np.append(X, bin_feat)
                featureList += [f"q{idx+1}_cat"] * len(bin_feat)

        # final states are labels
        if (self.predictStates):
            X1 = X
            X2 = None
            Y2 = None

            Y1 = np.array(row["finalState"])[0:5]
        # responses are the labels
        else:
            Y1 = np.array([])
            # go in and split data into 2 rows if desired
            if self.split_weekly_questions:
                Y2 = np.array([])
                # fill both labels
                for i,r in enumerate(row["response"]):
                    if (self.regression):
                        if (i == 0):
                            Y1 = np.append(Y1, r)
                        else:
                            Y2 = np.append(Y2, r)
                    elif (self.only_rnr):
                        if (i == 0):
                            Y1 = np.append(Y1, _onehot(min(1, r), 2))
                        else:
                            Y2 = np.append(Y2, _onehot(min(1, r), 2))
                    else:
                        if (i == 0):
                            Y1 = np.append(Y1, _onehot(r,4))
                        else:
                            Y2 = np.append(Y2, _onehot(r,4))
                # now split rows
                X1 = np.array([])
                X2 = np.array([])
                featureListFinal = []
                for idx, name in enumerate(featureList):
                    if "q1" in name:
                        X1 = np.append(X1, X[idx])
                        featureListFinal.append(name)
                    elif "q2" in name:
                        X2 = np.append(X2, X[idx])
                    # both rows get all non question specific features
                    else:
                        X1 = np.append(X1, X[idx])
                        X2 = np.append(X2, X[idx])
                        featureListFinal.append(name)
                featureList = featureListFinal
            else:
                X1 = X
                X2 = None
                Y2 = None
                for i,r in enumerate(row["response"]):
                    if self.regression:
                        Y1 = np.append(Y1, r)
                    elif (self.only_rnr):
                        Y1 = np.append(Y1, _onehot(min(1, r), 2))
                    else:
                        Y1 = np.append(Y1, _onehot(r,4))
        
        return X1, X2, Y1, Y2, featureList
    
    def save(self):
        out = {"data": self.data, "features": self.features, "labels":self.labels, "featureList": self.featureList, "nzIndices": self.nzindices}
        save(out, self.fname)
        
    def load(self):
        d = load(self.fname)
        self.features = d["features"]
        self.data = d["data"]
        self.labels = d["labels"]
        self.featureList = d["featureList"]
        self.nzindices = d["nzIndices"]
        
    @property
    def dimensions(self):
        # helper to get the x and y input dimensions
        if (self.regression or self.only_rnr):
            lshape = self.labels.shape[1]
        elif (not self.no_response_class):
            if (self.split_weekly_questions):
                lshape = self.labels.shape[1] - 1
            else:
                lshape = self.labels.shape[1] - 2
        else:
            lshape = self.labels.shape[1]

        return self.features.shape[1], lshape
        
