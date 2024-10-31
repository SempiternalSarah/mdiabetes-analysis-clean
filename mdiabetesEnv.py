import torch
import numpy as np
from utils.behavior_data import BehaviorData
from models.BasicNN import BasicNN
import json

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
def _onehot_response(a, l):
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



class DiabetesEnv():
    def __init__(self, startingStates, eqmodel, rewardStateDecay, endQPred, cuda = False, numHist = 3):
        self.endQPred = endQPred
        self.eqmodel = eqmodel
        self.startingStates = np.stack(startingStates, axis=0)
        self.rewardStateDecay = rewardStateDecay
        self.numHist = numHist
        self.cuda = cuda
        self.model = torch.load("trainedDiabetesPred.pt")
        if cuda:
            self.model = self.model.cuda()
        self.model.eval()
        self.currentStartedState = None
        self.maxMsgId = 57
        self.maxQId = 32
        self.maxSId = 17

        self.perCategoryRewards = np.zeros([self.maxSId], dtype=float)

        with open("detailed_question_state_element_map.json", 'r') as fp:
            qmap = json.loads(fp.read())
            qCatDict = {}
            for key in qmap.keys():
                for elem in qmap[key]:
                    qCatDict[elem] = int(key)
            self.qmap = qCatDict
        self.qmap[0] = 0
        mmap = np.loadtxt("detailed_message_map.csv", delimiter=',', dtype=int)
        self.mmap = dict(mmap)
        self.mmap[0] = 0

    def reset(self):
        if np.random.random() > .5:
            # start with random real data 
            idx = np.random.choice(self.startingStates.shape[0])
            toReturn = self.startingStates[idx]
        else:
            # randomly select values for each state element
            # choose from existing values in dataset
            mask = np.zeros_like(self.startingStates, dtype=bool)
            x = np.random.choice(mask.shape[0], mask.shape[1])
            y = np.arange(mask.shape[1])
            mask[x, y] = 1
            toReturn = self.startingStates[mask]
        self.currentStartedState = toReturn
        self.rewardState = toReturn
        self.perCategoryRewards = np.zeros([self.maxSId], dtype=float)
        self.lastQs = [np.zeros([2], dtype=int) for x in range(self.numHist)]
        self.lastMs = [np.zeros([2], dtype=int) for x in range(self.numHist)]
        self.lastRs = [np.zeros([2], dtype=int) for x in range(self.numHist - 1)]
        self.numEpochs = 0
        return toReturn

    def step(self, action):
        action = (action + 1) / 2
        msgs = np.array(action[0:2] *self.maxMsgId, dtype=float)
        msgs = np.ceil(msgs).astype(int)
        if msgs.min() < 0 or msgs.max() >self.maxMsgId:
            print("MSG OUT OF BOUNDS!!!!")
            print(action)
        qs = np.array(action[2:4] *self.maxQId, dtype=float)
        qs = np.ceil(qs).astype(int)
        if qs.min() < 0 or qs.max() >self.maxQId:
            print("Q OUT OF BOUNDS!!!!")
            print(action)
        self.lastMs = [msgs] + self.lastMs
        self.lastQs = [qs] + self.lastQs
        rows = self.encode_new_rows()
        observation = np.zeros_like(self.currentStartedState)
        knowns = np.zeros_like(self.currentStartedState, dtype=bool)
        observation[self.maxSId - 1:] = self.currentStartedState[self.maxSId - 1:]
        knowns[self.maxSId - 1:] = 1
        with torch.no_grad():
            anses = []
            for idx, row in enumerate(rows):
                row = torch.tensor(row).float()
                if (self.cuda):
                    row = row.cuda()
                answer, temp = self.model(row)
                answer = torch.argmax(answer)
                if np.random.rand() > 0.5:
                    anses.append(answer.numpy())
                    knowns[self.qmap[self.lastQs[0][idx]]] = True
                    # penalize choosing the same category twice
                    if observation[self.qmap[self.lastQs[0][idx]]] != 0:
                        observation[self.qmap[self.lastQs[0][idx]]] = min(observation[self.qmap[self.lastQs[0][idx]]], answer)
                    else:
                        observation[self.qmap[self.lastQs[0][idx]]] = answer
                else:
                    anses.append(-1)
                # print(observation)
                # print(knowns)
            self.lastRs = [np.array(anses)] + self.lastRs
        self.numEpochs += 1
        self.rewardState[~knowns] = self.rewardState[~knowns] * self.rewardStateDecay

        if self.numEpochs > 24:
            feats = self.encode_final_statepred_feats()
            with torch.no_grad():
                eqpred = self.eqmodel(feats).numpy()
            predictedImprovement = eqpred
            if self.endQPred:
                self.perCategoryRewards = predictedImprovement
            else:
                self.perCategoryRewards[knowns[:self.maxSId]] += observation[:self.maxSId][knowns[:self.maxSId]] - self.rewardState[:self.maxSId][knowns[:self.maxSId]] 
            reward = np.sum(self.perCategoryRewards)
        else:
            if not self.endQPred:
                self.perCategoryRewards[knowns[:self.maxSId]] += observation[:self.maxSId][knowns[:self.maxSId]] - self.rewardState[:self.maxSId][knowns[:self.maxSId]] 
            reward = 0
            predictedImprovement = None
        self.rewardState[knowns] = observation[knowns]

        return observation, knowns, reward, self.numEpochs > 24, predictedImprovement
        
    def encode_final_statepred_feats(self):
        feats = []
        for x in range(len(self.lastRs) - 1, self.numHist, -1):
            toReturn = np.array(self.currentStartedState)
            for qid in self.lastQs[x]:
                toReturn = np.append(_padded_binary(qid,self.maxQId), toReturn)
            for mid in self.lastMs[x]:
                toReturn = np.append(_padded_binary(mid,self.maxMsgId), toReturn)
            for res in self.lastRs[x]:
                toReturn = np.append(_onehot_response(res, 3), toReturn)
            feats.append(toReturn)
        feats = np.stack(feats)
        feats = torch.tensor(feats).float()
        return feats

    def encode_new_rows(self):
        toReturn = [None, None]
        for y in [0, 1]:
            toReturn[y] = self.currentStartedState
            for x in range(self.numHist - 1):
                toReturn[y] = np.append(_onehot_response(self.lastRs[x][y], 3), toReturn[y])
            for x in range(self.numHist):
                toReturn[y] = np.append(toReturn[y], _padded_binary(self.mmap[self.lastMs[x][y]],self.maxSId))
                toReturn[y] = np.append(toReturn[y], _padded_binary(self.qmap[self.lastQs[x][y]],self.maxSId))
                toReturn[y] = np.append(toReturn[y], _padded_binary(self.lastMs[x][y],self.maxMsgId))
                toReturn[y] = np.append(toReturn[y], _padded_binary(self.lastQs[x][y],self.maxQId))
        return toReturn

    

