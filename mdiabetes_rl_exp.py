import argparse
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import os
import time
from utils.behavior_data import BehaviorData
from utils.content import StatesHandler
from mdiabetesEnv import DiabetesEnv


def toBool(x):
    return (str(x).lower() in ['true', '1', 't'])

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=2, help="random seed")
parser.add_argument("--actlr", type=float, default=3e-5, help="Actor learning rate")
parser.add_argument("--statelr", type=float, default=1e-4, help="State pred learning rate")
parser.add_argument("--context", type=int, default=5, help="No. timesteps history for state prediction")
parser.add_argument("--hiddenSize", type=int, default=128, help="Hidden layer size in models")
parser.add_argument("--qlr", type=float, default=3e-5, help="critic learning rate")
parser.add_argument("--alpha_lr", type=float, default=3e-4, help="alpha learning rate")
parser.add_argument("--alpha", type=float, default=0.1, help="Entropy/regularization")
parser.add_argument("--gamma", type=float, default=0.5, help="the discount factor gamma")
parser.add_argument("--rewardStateDecay", type=float, default=0.9, help="weekly multiplier for state baseline used for rewards")
parser.add_argument("--numSteps", type=int, default=10000, help="iterations of overall algorithm")
parser.add_argument("--startLearning", type=int, default=10000, help="no. samples taken before training occurs")
parser.add_argument("--bufferSize", type=int, default=100000, help="no. samples in buffer")
parser.add_argument("--train_batches", type=int, default=100, help="no. batches per timestep")
parser.add_argument("--envSteps", type=int, default=100, help="no. environment steps per timestep")
parser.add_argument("--logging", type=toBool, default=False, help="Log data to disk")
parser.add_argument("--keepRealData", type=toBool, default=True, help="Keep all real world data in the replay buffer (do not replace it)")
parser.add_argument("--cuda", type=toBool, default=False, help="Use GPU for neural nets")
parser.add_argument("--endQPred", type=toBool, default=True, help="Use prediction of end questionnare to compute episodic rewards")
parser.add_argument("--statepred", type=toBool, default=False, help="Use state predictive model (our method)")
parser.add_argument("--statemodel", type=str, default="lstm", help="Architecture to use for state predictive model")
parser.add_argument("--numBreaks", type=int, default=4, help="Minibatches for state pred training")



args = parser.parse_args()

# overwrite
if args.cuda:
    args.cuda = torch.cuda.is_available()
    print(f"Using Cuda? {args.cuda}")
torch.random.manual_seed(args.seed)
np.random.seed(args.seed)

# print(args.actlr, args.qlr, args.bufferSize, args.startLearning


class Trajectory:
    def __init__(self, observations, actions, rewards, dones, knownses):
        # print(observations[0])
        self.obs = np.stack(observations, axis=0)
        # print(self.obs.max(), self.obs.min()).
        self.actions = np.stack(actions, axis=0)
        self.dones = np.stack(dones, axis=0)
        self.knownses = np.stack(knownses, axis=0)
        self.rewards = np.stack(rewards, axis=0)
        # print("_________IN BUFFER________________", self.obs[0], self.actions[0], self.obs[1], self.rewards[0], sep="\n")
        if len(self.obs) != len(self.actions) + 1 != len(self.dones) != len(self.knownses) != len(self.rewards) + 1:
            print("ERROR IN BUFFER STORAGE")

        self.length = len(actions)

    def getElement(self, idx):
        # print(idx, self.length)
        return {
            'obs':self.obs[idx], 
            'actions': self.actions[idx], 
            'nextobs': self.obs[idx + 1], 
            'dones': self.dones[idx + 1], 
            'knowns': self.knownses[idx], 
            'nextknowns': self.knownses[idx + 1], 
            'rewards': self.rewards[idx]}
    
    def sample(self, size):
        idxs = np.random.choice(self.length, size, replace = size>self.length)
        # if size > self.length:
        #     mult = 1
        # else:
        #     mult = 1
        return {
            'obs': self.obs[idxs], 
            'actions': self.actions[idxs], 
            'nextobs': self.obs[idxs + 1], 
            'dones': self.dones[idxs + 1], 
            'knowns': self.knownses[idxs], 
            'nextknowns': self.knownses[idxs + 1], 
            'rewards': [np.mean(self.rewards)]}
            # 'rewards': self.rewards}
    
    def retrieveStateFeatures(self, idx):
        mindx = max(idx - args.context, 0)
        obs = self.obs[mindx:idx + 1]
        acts = self.actions[mindx:idx + 1]
        labels = np.copy(self.obs[idx + 1]) - self.obs[idx]
        feats = np.concatenate([obs, acts], axis=1)
        knowns = self.knownses[idx]
        return feats, labels, knowns



class Buffer:
    def __init__(self, numElements):
        self.n = numElements
        self.count = 0
        self.els = []
        self.splits = []
        self.cutoff = 0

    def setCutoff(self):
        self.cutoff = len(self.els)

    def addElement(self, obs, act, rew, dones, knownses):
        e = Trajectory(obs, act, rew, dones, knownses)
        self.els.append(e)
        self.count += e.length

        if len(self.splits) > 0:
            self.splits.append(self.splits[-1] + len(act))
        else: 
            self.splits.append(len(act))
        if self.count > self.n:
            temp = self.els.pop(self.cutoff)
            self.splits.pop(self.cutoff)
            self.count -= temp.length
            tempSplits = self.splits[self.cutoff:]
            tempSplits = list(map(lambda x: x - temp.length, tempSplits))
            self.splits[self.cutoff:] = tempSplits

    def sample(self, size):
        idxs = np.random.choice(self.count, size, replace = size>self.count)
        idxs = sorted(idxs)
        i = 0
        toReturn = []
        for idx in idxs:
            while (idx >= self.splits[i]):
                i += 1
            if i == 0:
                offset = 0
            else:
                offset = self.splits[i - 1]
            if (idx - offset) < 0:
                print("ERRORR!!!!!!! Buffer index offset wrong")
            toReturn.append(self.els[i].getElement(idx - offset))
        return toReturn
    
    def sampleSubSeqs(self, subLen, numSubs):
        idxs = np.random.choice(len(self.els), numSubs, replace = numSubs>len(self.els))
        toReturn = {}
        # print(idxs, "!!")
        for i in idxs:
            temp = self.els[i].sample(subLen)
            for key in temp.keys():
                if key in toReturn:
                    # print("HAS", i)
                    # print(toReturn[key].shape)
                    toReturn[key].append(temp[key])
                else:
                    # print("HASNOT", i)
                    toReturn[key] = [temp[key]]
                    # print(toReturn[key])
        return toReturn
    
    def sampleForStatePred(self, size):
        idxs = np.random.choice(self.count, size, replace = size>self.count)
        idxs = sorted(idxs)
        i = 0
        returnFeats, returnLabs, returnKnowns = [], [], []
        for idx in idxs:
            while (idx >= self.splits[i]):
                i += 1
            if i == 0:
                offset = 0
            else:
                offset = self.splits[i - 1]
            if (idx - offset) < 0:
                print("ERRORR!!!!!!! Buffer index offset wrong")
            tfeat, tlab, tknown = self.els[i].retrieveStateFeatures(idx - offset)
            returnFeats.append(torch.tensor(tfeat))
            returnLabs.append(tlab)
            returnKnowns.append(tknown)
        lengths = [len(feat) for feat in returnFeats]
        # shape is (sequence, batch, features)
        return torch.nn.utils.rnn.pad_sequence(returnFeats).float(), np.stack(returnLabs, axis=0), np.stack(returnKnowns, axis=0), lengths


def layer_init(layer, bias_const=0.1):
    nn.init.xavier_uniform_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# LSTM used for predicting the state values during the RL process
class StateLSTMNetwork(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(obs_shape[0] + action_shape[0], self.hidden_size)
        self.outlayer = nn.Linear(self.hidden_size, obs_shape[0])
    
    def forward(self, x):
        if isinstance(x, torch.Tensor) and x.dim() < 2:
            x.unsqueeze_(0)
        output, (H,C) = self.lstm(x)
        # only use last prediction to keep context length consistent
        if isinstance(output, torch.nn.utils.rnn.PackedSequence):
            output = torch.nn.utils.rnn.pad_packed_sequence(output)[0]
        output = output[-1]
        output = F.relu(output)
        output = self.outlayer(output)
        return output
    
# lstm network for predicting end questionnaire values
class EndQLSTMNetwork(nn.Module):
    def __init__(self, in_shape, out_shape, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(in_shape, self.hidden_size)
        self.outlayer = nn.Linear(self.hidden_size, out_shape)
    
    def forward(self, x):
        if isinstance(x, torch.Tensor) and x.dim() < 2:
            x.unsqueeze_(0)
        output, (H,C) = self.lstm(x)
        # only use last prediction to keep context length consistent
        if isinstance(output, torch.nn.utils.rnn.PackedSequence):
            output = torch.nn.utils.rnn.pad_packed_sequence(output)[0]
            output = output[-1, :, :]
        else:
            output = output[-1, :]
        output = F.relu(output)
        output = self.outlayer(output)
        output = 2 * F.tanh(output)
        return output

# simple neural network for predicting states (UNUSED in final experiments)
class StateNNNetwork(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.l1 = layer_init(nn.Linear(obs_shape[0] + action_shape[0], self.hidden_size))
        self.l2 = layer_init(nn.Linear(self.hidden_size, self.hidden_size))
        self.outlayer = layer_init(nn.Linear(self.hidden_size, obs_shape[0]))
    
    def forward(self, x):
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x = torch.nn.utils.rnn.pad_packed_sequence(x)[0]
        print(x.shape)
        if x.dim() == 3:
            x = x.reshape([x.shape[1], x.shape[0] * x.shape[2]])
        print(x.shape)
        output = self.l1(x)
        output = F.relu(output)
        output = self.l2(output)
        output = F.relu(output)
        output = self.outlayer(output)
        return output

# soft Q network for SAC
class SoftQNetwork(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        self.input_shape = obs_shape[0] + action_shape[0]
        self.fc1 = layer_init(nn.Linear(self.input_shape, args.hiddenSize))
        self.fc2 = layer_init(nn.Linear(args.hiddenSize, args.hiddenSize))
        self.fc_q = layer_init(nn.Linear(args.hiddenSize, 1))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_vals = self.fc_q(x)
        return q_vals

# actor network for SAC
class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        self.fc1 = layer_init(nn.Linear(self.obs_shape[0], args.hiddenSize))
        self.fc2 = layer_init(nn.Linear(args.hiddenSize, args.hiddenSize))
        self.fc_mean_logdev = layer_init(nn.Linear(args.hiddenSize, 2*self.action_shape[0]))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        temp = self.fc_mean_logdev(x)
        mean = temp[:, :self.action_shape[0]]
        logdev = temp[:, self.action_shape[0]:]
        logdev = torch.clamp(logdev, -20, 2)

        return mean, logdev
    
    def only_action(self, x, exploration=True):
        with torch.no_grad():
            mean, logdev = self(x)
            dev = torch.exp(logdev)
            if (exploration):
                policy_dist = Normal(loc=mean, scale=dev)
                samp = policy_dist.sample()
            else:
                samp = mean
            if (args.cuda):
                samp = samp.cuda()
            action = torch.tanh(samp)
            return action

    def get_action(self, x, epsilon=1e-6, debug=False, exploration=False):
        mean, logdev = self(x)
        if (debug):
            print(logdev)
            print(x.shape, mean.shape, logdev.shape)
        dev = torch.max(logdev.exp(), .01*torch.ones_like(logdev))
        policy_dist = Normal(loc=mean, scale=dev)
        if (exploration):
            samp = policy_dist.rsample()
        else:
            samp = mean
        if (args.cuda):
            samp = samp.cuda()
        action = torch.tanh(samp)
        logprob = policy_dist.log_prob(samp)
        logprob -= torch.log((1 - action.pow(2)) + epsilon)
        logprob = logprob.sum(1, keepdim=True)
        # print(samp.shape, action.shape, logprob.shape)
        # exit()
        return action, logdev, logprob

# simple network for reward redistribution (parameters matching RRD paper)
class RRDModel(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        self.input_shape = 2 * obs_shape[0] + action_shape[0]
        self.fc1 = layer_init(nn.Linear(self.input_shape, args.hiddenSize))
        self.fc2 = layer_init(nn.Linear(args.hiddenSize, args.hiddenSize))
        self.fc3 = layer_init(nn.Linear(args.hiddenSize, 1))
    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out

# get current model "belief" of participant state given the raw observation, 
# and the "knowns" mask indicating which components are visible to the model
def getStateBelief(observations, knowns, acts=None, statepred: torch.nn.Module = None):
    if args.statepred:
        obsfeat = torch.tensor(np.concatenate([observations, acts], axis=-1)).float()
        obsfeat = obsfeat.unsqueeze_(1)
        if args.cuda:
            obsfeat = obsfeat.cuda()
        toReturn = statepred(obsfeat).squeeze()
        if args.cuda:
            toReturn = toReturn.cpu().detach().numpy()
        else:
            toReturn = toReturn.detach().numpy()
        return toReturn + observations[-1]
    observations = observations[-1]
    if len(observations.shape) > 2:
        print(observations.shape)
        print("OBS SHAPE INCORRECT!!!!")
    else:
        latest = observations[0]
        for x in range(1, len(observations)):
            # update latest knowledge
            latest = (knowns[x] * observations[x]) + ((1 - knowns[x]) * latest)
            # update observation with latest knowledge
            observations[x] = latest
    return observations

# set random seeds for reproducible results
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# LOAD BEHAVIOR DATA
data_kw={"minw": 2,
            "maxw": 31,
            "include_state": True,
            "include_pid": False,
            "expanded_states": True,
            "top_respond_perc": .5,
            "full_questionnaire": False,
            "num_weeks_history": 3,
            "insert_predictions": True,
            "split_model_features": False,
            "split_weekly_questions": True,
            "category_specific_history": False,
            "max_state_week": 500,
            "regression": False,
            "no_response_class": False,
            "only_rnr": False,
            "cluster_by": None
            }

bd = BehaviorData(**data_kw)

# load participant states values from both baseline and endline questionnaires
def load_questionnaire_states(endline=False):
        sh = StatesHandler(map="map_detailed.json", endline=endline)
        whatsapps, states, slist = sh.compute_states()
        def modify_whatsapp(x):
            # helper function to parse the whatsapp numbers
            x = str(x)
            x = x[len(x)-10:]
            return int(x)
        participantIDs = torch.tensor(np.loadtxt("arogya_content/all_ai_participants.csv", delimiter=",", skiprows=1, dtype="int64"))
        participantIDs[:, 1].apply_(modify_whatsapp)
        
        # filter responses to only include ones in the AI participant set
        isect, idIdxs, stateIdxs = np.intersect1d(participantIDs[:, 1], whatsapps, return_indices=True)
        return dict(zip(participantIDs[idIdxs, 0].numpy(), states[stateIdxs].numpy()))

# initialize info based on questionnaires
pre = load_questionnaire_states(False)
post = load_questionnaire_states(True)
maxMsgId = 57
maxQId = 32
maxSId = 17
rewsDiff = {}
# labels for the end questionnaire predictive task
# computed based on the difference between endline questionnaire values and baseline questionnaire values
endStateLabel = {}
for gid in pre.keys():
    if gid in post.keys():
        rewsDiff[gid] = np.sum(post[gid] - pre[gid])
        endStateLabel[gid] = post[gid][0:maxSId] - pre[gid][0:maxSId]

# transform participant data for use in predicting end states from action/response sequence
# this is done on a row-by-row (week-by-week) basis
def extractEndQPredFeats(row):
    toReturn = np.array(row['state'])
    def _padded_binary(a, b):
        # helper function to binary encode a and 
        # pad it to be the length of encoded b
        a, b = int(a), int(b)
        l = len(format(b,"b"))
        a = format(a,f"0{l}b")
        return np.array([int(_) for _ in a])
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
    
    for qid in row['qids']:
        toReturn = np.append(toReturn, _padded_binary(qid, maxQId))
    for mid in row['pmsg_ids']:
        toReturn = np.append(toReturn, _padded_binary(mid, maxMsgId))
    for res in row['response']:
        toReturn = np.append(toReturn, _onehot_response(res, 3))
    return toReturn

    
buff = Buffer(args.bufferSize)
# add real data to replay buffer
state = None
startingStates = []
endStatePredFeatures = {}
# go through each row (rows are pre-sorted by week and participant)
# we are translating the table information into the replay buffer format
for idx, row in bd.data.iterrows():
    if row['pid'] not in rewsDiff.keys():
        continue
    if row['week'] == 0:
        # if state is None, this is the first participant
        # otherwise, we've moved to next participant
        # and need to add the previous one's trajectory to replay buffer
        if state is not None:
            # remove last action as is irrelevant
            acts.pop()
            rewards = np.zeros_like(dones)
            rewards[-1] = rewsDiff[row['pid']]
            buff.addElement(obs, acts, rewards, dones, knownses)
        # initialize info based on this first week's data
        obs = []
        knownses = []
        acts = []
        dones = []
        endStatePredFeatures[row['pid']] = []
        state = np.array(row['state'])
        startingStates.append(state)
        obs.append(state)
        knownses.append(np.ones_like(state))
    else:
        # not the first week for this participant 
        # copy state, then update individual components based on responses
        state = np.copy(state)
        state[row['paction_sids'][0] - 1] = max(row['response'][0], 0)
        state[row['paction_sids'][1] - 1] = max(row['response'][1], 0)
        # update "knowns" indicating which components are observed
        # ie mark state categories that participant responded to this week
        knowns = np.zeros_like(state, dtype=bool)
        knowns[maxSId - 1:] = 1
        if row['response'][0] > -1:
            knowns[row['paction_sids'][0] - 1] = 1
        if row['response'][1] > -1:
            knowns[row['paction_sids'][1] - 1] = 1
        # append state belief and knowns to the lists
        obs.append(state)
        knownses.append(knowns)
    # build participant feature set for predicting end questionnaire results
    endStatePredFeatures[row['pid']].append(extractEndQPredFeats(row))
    # rescale action info as if it was output from actor network (between -1 and 1)
    action = np.append(np.array(row['pmsg_ids'])/maxMsgId, np.array(row['qids'])/maxQId)
    action = (action * 2) - 1
    acts.append(action)
    dones.append(0)

# build data for end questionnaire state predictive task
eqfeats = []
eqlabs = []
# ensure all participants here have weekly data AND questionnaire data
for k in endStateLabel.keys():
    if k not in endStatePredFeatures.keys():
        continue
    eqfeats.append(np.stack(endStatePredFeatures[k]))
    eqlabs.append(endStateLabel[k])

# stack participants together and tensorize the features
eqfeats = np.stack(eqfeats)
eqfeats = torch.tensor(eqfeats).float()
# reshape to match expected LSTM input (sequence length, batch dimension, input size)
eqfeats = eqfeats.reshape([eqfeats.shape[1], eqfeats.shape[0], eqfeats.shape[2]])
# tensorize labels (end questionnaire participant state values)
eqlabs = np.stack(eqlabs)
eqlabs = torch.tensor(eqlabs).float()
# initialize model and optimizer
eqmodel = EndQLSTMNetwork(eqfeats.shape[-1], eqlabs.shape[-1])
eqopt = optim.Adam(list(eqmodel.parameters()), 0.003)
lfn = torch.nn.MSELoss()
# train end state predictive model
# 500 epochs seems about right to converge without overfitting
for b in range(501):
    eqopt.zero_grad()
    preds = eqmodel(eqfeats)
    eqloss = lfn(preds, eqlabs)
    if b % 50 == 0:
        print(eqloss.item())
    eqloss.backward()
    eqopt.step()
# print(eqfeats.shape, eqlabs.shape)

# ensure buffer keeps all real data
if args.keepRealData:
    buff.setCutoff()

# set up test and train environments
env = DiabetesEnv(startingStates, eqmodel, args.rewardStateDecay, args.endQPred)
testenv = DiabetesEnv(startingStates, eqmodel, args.rewardStateDecay, args.endQPred)

# initialize all networks required for the reinforcement learning process
action_shape = [4, 1]
obs_shape = [22, 1]
actor = Actor(obs_shape, action_shape)
qf1 = SoftQNetwork(obs_shape, action_shape)
qf2 = SoftQNetwork(obs_shape, action_shape)
qf1_target = SoftQNetwork(obs_shape, action_shape)
qf2_target = SoftQNetwork(obs_shape, action_shape)
rrder = RRDModel(obs_shape, action_shape)
if args.cuda:
    actor = actor.cuda()
    qf1 = qf1.cuda()
    qf1_target = qf1_target.cuda()
    qf2 = qf2.cuda()
    qf2_target = qf2_target.cuda()
    rrder = rrder.cuda()
# this experiment uses state prediction - initialize LSTM
if (args.statepred):
    if args.statemodel == "lstm":
        # exit()
        statepred = StateLSTMNetwork(obs_shape, action_shape)
        statepred_target = StateLSTMNetwork(obs_shape, action_shape)
        statepred_target.load_state_dict(statepred.state_dict())
    if args.cuda:
        statepred = statepred.cuda()
        statepred_target = statepred_target.cuda()
else:
    statepred = None
    statepred_target = None
# disable gradient tracking for target networks
for p1, p2 in zip(qf1_target.parameters(), qf2_target.parameters()):
    p1.requires_grad = False
    p2.requires_grad = False
# set up optimizer for alpha if we are learning it, otherwise, use static value
if (args.alpha_lr > 0):
    logalpha = torch.tensor(0).float()
    logalpha.requires_grad = True
    alpha_opt = optim.Adam([logalpha], lr=args.alpha_lr)
    alpha = torch.exp(logalpha)
else: 
    alpha = args.alpha


# ensure target network and training network alignment
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())
# optimizers for each component of the RL model
q_opt = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.qlr)
act_opt = optim.Adam(list(actor.parameters()), lr=args.actlr)
rrd_opt = optim.Adam(list(rrder.parameters()), lr=3e-4)
if args.statepred:
    stateopt = optim.Adam(list(statepred.parameters()), lr=args.statelr)

# evaluate the policy (using on-policy actions)
# for reporting purposes
def evaluatePolicy(numRollouts=100):
    totalReward = 0
    totalLoss = 0
    obstrajs = []
    acttrajs = []
    percatlist = []
    numImproved = 0
    for x in range(numRollouts):
        episodicReward = 0
        episodicLoss = 0
        done = False
        testobs = testenv.reset()
        knowns = np.ones_like(testobs)
        obsbuff = [testobs]
        actbuff = []
        while not done:
            obsfeat = torch.tensor(testobs).float().unsqueeze(0)
            if args.cuda:
                obsfeat = obsfeat.cuda()
            testaction = actor.only_action(obsfeat, exploration=False)
            if args.cuda:
                testaction = testaction.detach().cpu().numpy().squeeze()
            else:
                testaction = testaction.detach().numpy().squeeze()
            nextobsUn, knowns, testreward, done, percat = testenv.step(testaction)
            actbuff.append(testaction)
            start = min(args.context, len(obsbuff))
            nextObsPred = getStateBelief(obsbuff[-start:], knowns, actbuff[-start:], statepred)
            testobs = (knowns * nextobsUn) + ((1 - knowns) * (nextObsPred))
            testObsIdxs = (1 - knowns) != 0
            # don't track timesteps where num hidden is 0 (first observation)
            if testObsIdxs.sum() > 0:
                # print(nextobsUn[testObsIdxs])
                testLoss = (nextobsUn[testObsIdxs] - nextObsPred[testObsIdxs]) ** 2
                testLoss = testLoss.mean().item()
                episodicLoss += testLoss
            obsbuff.append(testobs)
            episodicReward += testreward
        # append final per category rewards to list
        percatlist.append(percat)
        obstraj = np.stack(obsbuff, axis=0)
        acttraj = np.stack(actbuff, axis=0)
        acttraj = (acttraj + 1) / 2
        acttraj[:, 0:2] = acttraj[:, 0:2] * testenv.maxMsgId
        acttraj[:, 2:4] = acttraj[:, 2:4] * testenv.maxQId
        acttraj = np.ceil(acttraj).astype(int)
        totalLoss += episodicLoss / len(actbuff)
        if np.sum(percat) > 0:
            numImproved += 1
        totalReward += episodicReward
        obstrajs.append(obstraj)
        acttrajs.append(acttraj)
    # translate final category reward list to % of participants that improved
    partimp = np.stack(percatlist, axis=0)
    partimp = np.where(partimp > 0, 1, 0)
    partimp = np.mean(partimp, axis=0)
    # print(numImproved, partimp)
    return totalReward / numRollouts, totalLoss / numRollouts, np.stack(obstrajs, axis=0), np.stack(acttrajs, axis=0), partimp, numImproved/numRollouts

# set up directories to save information over the course of training
statestr = ''
if args.statepred:
    statestr = args.statemodel
    statestr += '/'
# avoid file system access collisions by only having one seed try to create folders
if (args.logging and args.seed == 1):
    if not os.path.exists(f"./saved_mdiabetes_rl/{statestr}rewards"):
        os.makedirs(f"./saved_mdiabetes_rl/{statestr}rewards")
    if not os.path.exists(f"./saved_mdiabetes_rl/{statestr}actloss"):
        os.makedirs(f"./saved_mdiabetes_rl/{statestr}actloss")
    if not os.path.exists(f"./saved_mdiabetes_rl/{statestr}qfloss"):
        os.makedirs(f"./saved_mdiabetes_rl/{statestr}qfloss")
    if not os.path.exists(f"./saved_mdiabetes_rl/{statestr}rrdloss"):
        os.makedirs(f"./saved_mdiabetes_rl/{statestr}rrdloss")
    if not os.path.exists(f"./saved_mdiabetes_rl/{statestr}statepredloss"):
        os.makedirs(f"./saved_mdiabetes_rl/{statestr}statepredloss")
    if not os.path.exists(f"./saved_mdiabetes_rl/{statestr}statepredtestloss"):
        os.makedirs(f"./saved_mdiabetes_rl/{statestr}statepredtestloss")
    if not os.path.exists(f"./saved_mdiabetes_rl/{statestr}trajlog"):
        os.makedirs(f"./saved_mdiabetes_rl/{statestr}trajlog")

# empty lists to track information over training process
rewardList = []
catImproveList = []
testStateLossList = []
qflosslist = []
actlosslist = []
rrdlosslist = []
statelosslist = []

# initialize fields for current trajectory
observation = env.reset()
obs = [observation]
acts = []
dones = [0]
knowns = np.ones_like(observation)
knownses = [knowns]
rewards = []
numSteps = 0
hiddenLeft = 0

# mark losses as None for easier checking when printing results
actloss = None
qfloss = None

# keep track of total runtime
starttime = time.time()

# overall loop for entire training process
for step in range(int(args.numSteps)):
    # loop for sampling steps from the environment
    for envStep in range(args.envSteps):
        with torch.no_grad():
            obsfeat = torch.tensor(observation).float().unsqueeze(0)
            if args.cuda:
                obsfeat = obsfeat.cuda()
            if (envStep % 100) == 55 and step % 10 == 1:
                # print(obsfeat.max())
                None
            action = actor.only_action(obsfeat)
            if args.cuda:
                action = action.cpu().detach().numpy().squeeze()
            else:
                action = action.detach().numpy().squeeze()
            nextObs, knowns, reward, done, percat = env.step(action)
            acts.append(action)
            start = min(args.context, len(obs))
            # compute state belief using raw observation, previous observation(s), and "knowns" mask
            nextObsPred = getStateBelief(obs[-start:], knowns, acts[-start:], statepred_target)
            observation = (knowns * nextObs) + ((1 - knowns) * (nextObsPred))
            knownses.append(knowns)
            rewards.append(reward)
            obs.append(observation)
            # trajectory has reached end state 
            # record results in replay buffer, and reset environment
            if done:
                # as this domain exclusively uses timestep termination,
                # we always indicate that this is NOT a real terminal state
                # in SAC, REAL terminal states indicate that the agent's actions
                # led to the termination of the trajectory (eg robot breaking)
                dones.append(0)
                buff.addElement(obs, acts, rewards, dones, knownses)
                numSteps += len(acts)
                # init
                observation = env.reset()
                obs = [observation]
                acts = []
                dones = [0]
                knowns = np.ones_like(observation)
                knownses = [knowns]
                rewards = []
                hiddenLeft = 0
            else:
                dones.append(0)
    # check if it is time to start training the RL models
    if (numSteps) >= args.startLearning:
        # train state prediction model
        if args.statepred:
            bsize = (args.train_batches * 256) // args.numBreaks
            # loop to train each batch
            for x in range(args.numBreaks):
                # ----------------------
                # train state prediction network
                # ----------------------

                feats, labels, mask, lengths = buff.sampleForStatePred(bsize)
                feats = torch.nn.utils.rnn.pack_padded_sequence(feats, lengths, enforce_sorted=False)
                labels = torch.tensor(labels).float()
                mask = torch.tensor(mask).bool()
                mask[maxSId - 1:] = 0
                if args.cuda:
                    feats = feats.cuda()
                    labels = labels.cuda()
                    mask = mask.cuda()
                preds = statepred(feats)

                # EXCLUDE STATIC (demographic) STATE FEATURES FROM LOSS!!
                preds = preds[:, :maxSId]
                labels = labels[:, :maxSId]
                mask = mask[:, maxSId]
                
                # exclude unobserved components from loss
                preds = preds[mask]
                labels = labels[mask]
                # compute MSE
                statePredLoss = (preds - labels) ** 2
                statePredLoss = statePredLoss.mean()
                stateopt.zero_grad()
                statePredLoss.backward()
                # clip gradient values to smooth out training process
                # probably not necessary in this domain
                nn.utils.clip_grad_value_(statepred.parameters(), 1.0)
                stateopt.step()
        # train reward redistribution, actor, and critic networks
        for trainstep in range(args.train_batches):
            # 1 update here for every environment step

            # ----------------------
            # train RRD network
            # ----------------------

            # sample from replay buffer, extract individual components
            samples = buff.sampleSubSeqs(64, 4)
            obSamp = np.array(samples['obs'])
            actSamp = np.array(samples['actions'])
            if len(actSamp.shape) < len(obSamp.shape):
                actSamp = np.expand_dims(actSamp, -1)
            ob2Samp = np.array(samples['nextobs'])
            rewSamp = np.array(samples['rewards'])
            knownSamp = np.array(samples['knowns'])
            known2Samp = np.array(samples['nextknowns'])

            # create features for the reward redistribution model
            feats = torch.tensor(np.concatenate((obSamp, actSamp, obSamp - ob2Samp), axis=-1)).float()
            if args.cuda:
                feats = feats.cuda()
            rHat = rrder(feats).squeeze(-1)
            # sum predicted rewards, per episode
            episodicSums = torch.mean(rHat, dim=-1, keepdim=True)
            rewTens = torch.tensor(rewSamp).float()
            if args.cuda:
                rewTens = rewTens.cuda()
            rrdloss = F.mse_loss(episodicSums, rewTens)
            rrdloss.backward()
            rrd_opt.step()

            # ----------------------
            # train Q-value networks
            # ----------------------

            samples = buff.sample(256)

            obSamp = np.array([samp['obs'] for samp in samples])
            actSamp = np.array([samp['actions'] for samp in samples])
            if len(actSamp.shape) < len(obSamp.shape):
                actSamp = np.expand_dims(actSamp, -1)
            nextObsSamp = np.array([samp['nextobs'] for samp in samples])
            donesSamp = torch.tensor([samp['dones'] for samp in samples]).float().unsqueeze(-1)
            if args.cuda:
                donesSamp = donesSamp.cuda()
            knownSamp = np.array([samp['knowns'] for samp in samples])
            nextKnownSamp = np.array([samp['nextknowns'] for samp in samples])

            # no gradients required for this part
            with torch.no_grad():
                # set up features for reward redistribution network
                feats = torch.tensor(np.concatenate((obSamp, actSamp, obSamp - nextObsSamp), axis=-1)).float()
                # features for actor network
                obsfeats = torch.tensor(nextObsSamp).float()
                if(args.cuda):
                    feats = feats.cuda()
                    obsfeats = obsfeats.cuda()
                rHat = rrder(feats)
                # sample new action at current observation based on current policy (with exploration)
                nextActions, logdev, logprob = actor.get_action(obsfeats, debug=False, exploration=True)
                nextFeats = torch.concat([obsfeats, nextActions], dim=-1).float()
                if args.cuda:
                    nextFeats = nextFeats.cuda()
                # use target networks to calculate the target q value for this new action
                qtarg1 = qf1_target(nextFeats)
                qtarg2 = qf2_target(nextFeats)
                qtargmin = torch.min(qtarg1, qtarg2) - (alpha * logprob)
                qtarget = rHat + (args.gamma * (1 - donesSamp) * qtargmin)

            # this part requires gradient

            # create features for q value networks
            feats = np.concatenate((obSamp, actSamp), axis=-1)
            feats1 = torch.tensor(feats).float()
            feats2 = torch.tensor(feats).float()
            if args.cuda:
                feats1 = feats1.cuda()
                feats2 = feats2.cuda()
            # compute q values using each network
            qf1vals = qf1(feats1)
            qf2vals = qf2(feats2)
            # compute losses
            qf1loss = F.mse_loss(qf1vals, qtarget)
            qf2loss = F.mse_loss(qf2vals, qtarget)
            # combine losses to allow for single backwards pass
            qfloss = qf1loss + qf2loss
            q_opt.zero_grad()
            qfloss.backward()
            q_opt.step()

            # ----------------------
            # train actor network
            # ----------------------

            # create features from observation
            obsfeat = torch.tensor(obSamp).float()
            if args.cuda:
                obsfeat = obsfeat.cuda()
            # sample action using exploration
            actSamp, logdev, logprob = actor.get_action(obsfeat, debug=False, exploration=True)
            # create features for computing Q value of the new action
            feats = torch.concat((obsfeat, actSamp), dim=-1).float()
            if args.cuda:
                feats = feats.cuda()

            # do NOT compute gradient w.r.t. q value networks
            # however, we NEED the gradient to go THROUGH these networks
            # to update the actor network
            for p1, p2 in zip(qf1.parameters(), qf2.parameters()):
                p1.requires_grad = False
                p2.requires_grad = False
            # compute q values from networks
            qf1vals = qf1(feats)
            qf2vals = qf2(feats)
            # calculate actor loss based on alpha, logprob of the action, and the q value
            actloss = torch.mean((alpha * logprob) - torch.min(qf1vals, qf2vals))
            act_opt.zero_grad()
            actloss.backward()
            act_opt.step()
            # re-enable gradients
            for p1, p2 in zip(qf1.parameters(), qf2.parameters()):
                p1.requires_grad = True
                p2.requires_grad = True

            # ----------------------
            # update alpha, if appropriate
            # ----------------------

            if (args.alpha_lr > 0):
                alpha_opt.zero_grad()
                with torch.no_grad():
                    multiplier = logprob - np.prod(action_shape)
                aloss = -1.0*(torch.exp(logalpha) * multiplier).mean()
                aloss.backward()
                alpha_opt.step()
                alpha = torch.exp(logalpha)

            # smoothly update target networks at each iteration of the training algorithm
            with torch.no_grad():
                for qt1w, qf1w in zip(qf1_target.parameters(), qf1.parameters()):
                    qt1w.data.copy_(0.995 * qt1w.data + (.005 * qf1w.data))
                for qt2w, qf2w in zip(qf2_target.parameters(), qf2.parameters()):
                    qt2w.data.copy_(0.995 * qt2w.data + (.005 * qf2w.data))

                if args.statepred:
                    for spt, sp in zip(statepred_target.parameters(), statepred.parameters()):
                        spt.data.copy_(0.995 * spt.data + (.005 * sp.data))

        # evaluate current policy and record results every 50 outer steps
        if (step % 50) == 0 or (step == args.numSteps - 1):
            with torch.no_grad():
                testrew, testloss, obstraj, acttraj, partImp, totalImproved = evaluatePolicy()
                rewardList.append(testrew)
            # print detailed information to stdout if training has started
            if not actloss is None:
                if args.statepred:
                    print(f"Steps: {step * args.envSteps}, Time: {time.time() - starttime:.3f}s, Test rewards: {rewardList[-1]:.3f}, Actor loss: {actloss.item():.3e}, Q loss: {qfloss.item():.3e}, RRD loss: {rrdloss.item():.3e}, Alpha: {alpha:.3e}, StateLoss: {statePredLoss.item():.3e}, Test StateLoss: {testloss:.3e}")
                    # if (args.cuda):
                    #     print(f"GPU: {torch.cuda.max_memory_allocated(device=None)}")
                else:
                    print(f"Steps: {step * args.envSteps}, Time: {time.time() - starttime:.3f}s, Test rewards: {rewardList[-1]:.3f}, Actor loss: {actloss.item():.3e}, Q loss: {qfloss.item():.3e}, RRD loss: {rrdloss.item():.3e}, Alpha: {alpha:.3e}")  
            else:
                # otherwise, just print current test environment reward
                print(rewardList[-1])
            # record results to files if appropriate
            if (args.logging):
                if args.statepred:
                    statemodel = args.statemodel + "/"
                else:
                    statemodel = ""
                fname = f"{args.seed}H{args.hiddenSize}LR{args.qlr}EQ{args.endQPred}.csv"
                np.savetxt(f"./saved_mdiabetes_rl/{statemodel}rewards/{fname}", rewardList, delimiter="\n")
                if not actloss is None:
                    actlosslist.append(actloss.item())
                    qflosslist.append(qfloss.item())
                    rrdlosslist.append(rrdloss.item())
                    if (args.statepred):
                        testStateLossList.append(testloss)
                        statelosslist.append(statePredLoss.item())

                        np.savetxt(f"./saved_mdiabetes_rl/{statemodel}statepredtestloss/{fname}", testStateLossList, delimiter="\n")
                        np.savetxt(f"./saved_mdiabetes_rl/{statemodel}statepredloss/{fname}", statelosslist, delimiter="\n")

                    np.savetxt(f"./saved_mdiabetes_rl/{statemodel}actloss/{fname}", actlosslist, delimiter="\n")
                    np.savetxt(f"./saved_mdiabetes_rl/{statemodel}qfloss/{fname}", qflosslist, delimiter="\n")
                    np.savetxt(f"./saved_mdiabetes_rl/{statemodel}rrdloss/{fname}", rrdlosslist, delimiter="\n")

                    weeklyobsmeans = np.mean(obstraj, axis=0)

                    weeklyobsstd = np.std(obstraj, axis=0)
                    mrowtotals = np.zeros([acttraj.shape[1], testenv.maxMsgId + 1], dtype=int)
                    qrowtotals = np.zeros([acttraj.shape[1], testenv.maxQId + 1], dtype=int)
                    for week in range(acttraj.shape[1]):
                        for msgid in range(testenv.maxMsgId + 1):
                            mrowtotals[week, msgid] = (acttraj[:, week, 0:2] == msgid).sum()
                    for week in range(acttraj.shape[1]):
                        for qid in range(testenv.maxQId + 1):
                            qrowtotals[week, qid] = (acttraj[:, week, 2:4] == qid).sum()
                    for w in range(acttraj.shape[1]):        
                        with open(f"./saved_mdiabetes_rl/{statemodel}trajlog/weekobsmean{w}{fname}", "a+") as f:
                            np.savetxt(f, weeklyobsmeans[w], delimiter=', ', newline=", ")
                            f.write("\n")
                        with open(f"./saved_mdiabetes_rl/{statemodel}trajlog/weekobsstd{w}{fname}", "a+") as f:
                            np.savetxt(f, weeklyobsstd[w], delimiter=', ', newline=", ")
                            f.write("\n")
                        with open(f"./saved_mdiabetes_rl/{statemodel}trajlog/weekmsgcount{w}{fname}", "a+") as f:
                            np.savetxt(f, mrowtotals[w], delimiter=', ', newline=", ")
                            f.write("\n")
                        with open(f"./saved_mdiabetes_rl/{statemodel}trajlog/weekqcount{w}{fname}", "a+") as f:
                            np.savetxt(f, qrowtotals[w], delimiter=', ', newline=", ")
                            f.write("\n")
                    with open(f"./saved_mdiabetes_rl/{statemodel}trajlog/obsmeans{fname}", "a+") as f:
                        np.savetxt(f, weeklyobsmeans.mean(axis=0), delimiter=', ', newline=", ")
                        f.write("\n")
                    with open(f"./saved_mdiabetes_rl/{statemodel}trajlog/obsstd{fname}", "a+") as f:
                        np.savetxt(f, np.sqrt(np.square(weeklyobsstd).mean(axis=0)), delimiter=', ', newline=", ")
                        f.write("\n")
                    with open(f"./saved_mdiabetes_rl/{statemodel}trajlog/qcount{fname}", "a+") as f:
                        np.savetxt(f, qrowtotals.sum(axis=0), delimiter=', ', newline=", ")
                        f.write("\n")
                    with open(f"./saved_mdiabetes_rl/{statemodel}trajlog/msgcount{fname}", "a+") as f:
                        np.savetxt(f, mrowtotals.sum(axis=0), delimiter=', ', newline=", ")
                        f.write("\n")
                    with open(f"./saved_mdiabetes_rl/{statemodel}trajlog/partimp{fname}", "a+") as f:
                        np.savetxt(f, partImp, delimiter=', ', newline=", ")
                        f.write("\n")
                    with open(f"./saved_mdiabetes_rl/{statemodel}trajlog/totalimp{fname}", "a+") as f:
                        f.write(f"{totalImproved}\n")