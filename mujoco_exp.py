import gym
import argparse
import random
import gym.spaces
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import os
import time


def toBool(x):
    return (str(x).lower() in ['true', '1', 't'])

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--actlr", type=float, default=3e-4, help="Actor learning rate")
parser.add_argument("--statelr", type=float, default=1e-3, help="State pred learning rate")
parser.add_argument("--context", type=int, default=20, help="No. timesteps history for state prediction")
parser.add_argument("--qlr", type=float, default=3e-4, help="critic learning rate")
parser.add_argument("--alpha_lr", type=float, default=3e-4, help="alpha learning rate")
parser.add_argument("--alpha", type=float, default=0.1, help="Entropy/regularization")
parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
parser.add_argument("--numSteps", type=int, default=40000, help="iterations of overall algorithm")
parser.add_argument("--startLearning", type=int, default=1000000, help="no. samples taken before RRD and SAC training occurs")
parser.add_argument("--startLearningState", type=int, default=10000, help="no. samples taken before state pre-training occurs")
parser.add_argument("--bufferSize", type=int, default=1000000, help="no. samples in buffer")
parser.add_argument("--numHidden", type=int, default=0, help="no. parts of robot invisible each time step")
parser.add_argument("--train_batches", type=int, default=100, help="no. batches per timestep")
parser.add_argument("--envSteps", type=int, default=100, help="no. environment steps per timestep")
parser.add_argument("--consecHidden", type=int, default=1, help="no. consecutive timesteps to hide information (e.g. hide an ant limb for 10 environmental steps)")
parser.add_argument("--env", type=str, default="HalfCheetah-v2", help="MuJoCo environment")
parser.add_argument("--logging", type=toBool, default=False, help="Logs progress to file")
parser.add_argument("--cuda", type=toBool, default=False, help="Use GPU for training")
parser.add_argument("--statepred", type=toBool, default=False, help="Use explicit state prediction")
parser.add_argument("--statefill", type=toBool, default=True, help="Fill unobserved state components with previously observed values")
parser.add_argument("--contextSAC", type=toBool, default=False)
parser.add_argument("--statemodel", type=str, default="lstm", help="Type of model for predicting states")
parser.add_argument("--numBreaks", type=int, default=4, help="Minibatches for state pred training")
parser.add_argument("--stateTrainMult", type=int, default=1, help="Multiplier for state learning epochs")
parser.add_argument("--hiddenSizeLSTM", type=int, default=64, help="Hidden size for LSTM network")
parser.add_argument("--policyResetStep", type=int, default=-1, help="No. algorithmic steps at which to reset the policy")



args = parser.parse_args()

# overwrite
if args.cuda:
    args.cuda = torch.cuda.is_available()
    print(f"Using Cuda? {args.cuda}")
torch.random.manual_seed(args.seed)
np.random.seed(args.seed)

# trajectory class - helper for lessons buffer
class Trajectory:
    def __init__(self, observations, actions, rewards, dones, knownses):
        self.obs = np.stack(observations, axis=0)
        self.actions = np.stack(actions, axis=0)
        self.dones = np.stack(dones, axis=0)
        self.knownses = np.stack(knownses, axis=0)
        self.rewards = np.stack(rewards, axis=0)

        # validate trajectory
        if len(self.obs) != len(self.actions) + 1 != len(self.dones) != len(self.knownses) != len(self.rewards) + 1:
            print("ERROR IN BUFFER STORAGE")

        self.length = len(actions)

    # return specific timestep of trajectory
    def getElement(self, idx):

        return {
            'obs':self.obs[idx], 
            'actions': self.actions[idx], 
            'nextobs': self.obs[idx + 1], 
            'dones': self.dones[idx + 1], 
            'knowns': self.knownses[idx], 
            'nextknowns': self.knownses[idx + 1], 
            'rewards': self.rewards[idx]}
    
    # sample SIZE entries from trajectory
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
    
    # retrieve features for state prediction task
    def retrieveStateFeatures(self, idx):
        mindx = max(idx - args.context, 0)
        obs = self.obs[mindx:idx + 1]
        acts = self.actions[mindx:idx + 1]
        labels = np.copy(self.obs[idx + 1]) - self.obs[idx]
        feats = np.concatenate([obs, acts], axis=1)
        knowns = self.knownses[idx + 1]
        return feats, labels, knowns

# lessons buffer class
class Buffer:
    def __init__(self, numElements):
        self.n = numElements
        self.count = 0
        self.minStateVals = None
        self.maxStateVals = None
        self.els = []
        self.splits = []

    def addElement(self, obs, act, rew, dones, knownses):
        # keep track of min/max observation
        tempObsMax = np.max(obs, axis=0)
        tempObsMin = np.min(obs, axis=0)
        if (self.count == 0):
            self.maxStateVals = tempObsMax
            self.minStateVals = tempObsMin
        else:
            self.maxStateVals = np.maximum(self.maxStateVals, tempObsMax)
            self.minStateVals = np.minimum(self.minStateVals, tempObsMin)

        e = Trajectory(obs, act, rew, dones, knownses)
        self.els.append(e)
        self.count += e.length

        # keep track of divisions between different transitions in buffer
        if len(self.splits) > 0:
            self.splits.append(self.splits[-1] + len(act))
        else: 
            self.splits.append(len(act))
        if self.count > self.n:
            temp = self.els.pop(0)
            self.count -= temp.length
            self.splits = self.splits[1:]
            self.splits = list(map(lambda x: x - temp.length, self.splits))

    def sample(self, size):
        idxs = np.random.choice(self.count, size, replace = size > self.count)
        idxs = sorted(idxs)
        i = 0
        toReturn = []
        # retrieve chosen indexes from corresponding trajectories
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
        # recombine samples into their own vectors
        for i in idxs:
            temp = self.els[i].sample(subLen)
            for key in temp.keys():
                if key in toReturn:
                    toReturn[key].append(temp[key])
                else:
                    toReturn[key] = [temp[key]]
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

            # combine individually returned values into vectors for each
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

class StateLSTMNetwork(nn.Module):
    def __init__(self, envs, hidden_size=args.hiddenSizeLSTM):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(envs.observation_space.shape[0] + envs.action_space.shape[0], self.hidden_size)
        self.attnlayer = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2)
        self.outlayer = nn.Linear(self.hidden_size, envs.observation_space.shape[0])
    
    def forward(self, x, mins=None, maxes=None):
        if isinstance(x, torch.Tensor) and x.dim() < 2:
            x.unsqueeze_(0)
        output, (H,C) = self.lstm(x)
        # only use last prediction to keep context length consistent
        if isinstance(output, torch.nn.utils.rnn.PackedSequence):
            output = torch.nn.utils.rnn.pad_packed_sequence(output)[0]
        output = torch.sigmoid(output)
        output = self.attnlayer(output, output, output)[0][-1]
        output = torch.relu(output)
        output = self.outlayer(output)

        # clamp prediction to previously observed range (UNUSED)
        # if not mins is None:
        #     mins = torch.tensor(mins)
        #     maxes = torch.tensor(maxes)
        #     if args.cuda:
        #         mins = mins.cuda()
        #         maxes = maxes.cuda()
        #     output = torch.clamp(output, mins, maxes)
        return output
    
# state predictor using only linear layers
# UNUSED
class StateNNNetwork(nn.Module):
    def __init__(self, envs, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.l1 = layer_init(nn.Linear(envs.observation_space.shape[0] + envs.action_space.shape[0], self.hidden_size))
        self.l2 = layer_init(nn.Linear(self.hidden_size, self.hidden_size))
        self.outlayer = layer_init(nn.Linear(self.hidden_size, envs.observation_space.shape[0]))
    
    def forward(self, x):
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x = torch.nn.utils.rnn.pad_packed_sequence(x)[0]
        if x.dim() == 3:
            x = x.reshape([x.shape[1], x.shape[0] * x.shape[2]])
        output = self.l1(x)
        output = F.relu(output)
        output = self.l2(output)
        output = F.relu(output)
        output = self.outlayer(output)
        return output

# Q value network for SAC
class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.input_shape = envs.observation_space.shape[0] + envs.action_space.shape[0]
        if (args.contextSAC):
            hiddenSize = args.hiddenSizeLSTM
            self.lstm = nn.LSTM(self.input_shape, hiddenSize)
        else:
            hiddenSize = 256
            self.fc1 = layer_init(nn.Linear(self.input_shape, hiddenSize))
            self.fc2 = layer_init(nn.Linear(hiddenSize, hiddenSize))
        self.fc_q = layer_init(nn.Linear(hiddenSize, 1))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.relu(x)
        q_vals = self.fc_q(x)
        return q_vals

# actor network for SAC
class Actor(nn.Module):
    def __init__(self, envs: gym.Env):
        super().__init__()
        self.obs_shape = envs.observation_space.shape
        self.action_shape = envs.action_space.shape
        self.action_scale = torch.tensor((envs.action_space.high - envs.action_space.low) / 2.0).float()
        if args.cuda:
            self.action_scale = self.action_scale.cuda()
        hiddenSize = 256
        self.fc1 = layer_init(nn.Linear(self.obs_shape[0], hiddenSize))
        self.fc2 = layer_init(nn.Linear(hiddenSize, hiddenSize))
        self.fc_mean_logdev = layer_init(nn.Linear(hiddenSize, 2*self.action_shape[0]))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.relu(x)
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
            action = torch.tanh(samp) * self.action_scale
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
        action = torch.tanh(samp) * self.action_scale
        logprob = policy_dist.log_prob(samp)
        # Enforcing Action Bound
        logprob -= torch.log(self.action_scale * (1 - action.pow(2)) + epsilon)
        logprob = logprob.sum(1, keepdim=True)

        return action, logdev, logprob

# reward redistribution model
class RRDModel(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.input_shape = 2 * envs.observation_space.shape[0] + envs.action_space.shape[0]
        contextSize = 256
        self.fc1 = layer_init(nn.Linear(self.input_shape, contextSize))
        self.fc2 = layer_init(nn.Linear(contextSize, contextSize))
        self.fc3 = layer_init(nn.Linear(contextSize, 1))

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out
    

# defines components of each robot
# full components (subarrays) are what is hudden

antParts = [[0, 1, 2, 3, 4, 13, 14, 15, 16, 17, 18],
    [5, 6, 19, 20],
    [7, 8, 21, 22],
    [9, 10, 23, 24],
    [11, 12, 25, 26]]

cheetahParts = [[0, 1, 8, 9, 10],
                [2, 11],
                [3, 12],
                [4, 13],
                [5, 14],
                [6, 15],
                [7, 16]]

humanParts = [[0, 1, 2, 3, 4, 22, 23, 24, 25, 26, 27],
              [5, 6, 7, 28, 29, 30],
              [8, 9, 10, 31, 32, 33],
              [11, 34],
              [12, 13, 14, 35, 36, 37],
              [15, 38],
              [16, 17, 39, 40],
              [18, 41],
              [19, 20, 42, 43],
              [21, 44]]

walkerParts = [[0, 1, 8, 9, 10],
               [2, 11],
               [3, 12],
               [4, 13],
               [5, 14],
               [6, 15],
               [7, 16]]

hopperParts = [[0, 1, 5, 6, 7],
               [2, 8],
               [3, 9],
               [4, 10]]

# easily remap versions for debugging
partsLookup = {"Ant-v2": antParts,
               "Ant-v3": antParts,
               "HalfCheetah-v2": cheetahParts,
               "Humanoid-v2": humanParts,
               "Hopper-v2": hopperParts,
               "Walker2d-v2": walkerParts}

# returns filtered observation based on how much information should be hidden
# this accomplishes the modification to the MuJoCo Gym environments 
def obsFilter(observation, numHidden, lastKnown, leftTilRandom):
    parts = partsLookup[args.env]
    if (numHidden > 0):
        # NEW parts need to be hidden
        if leftTilRandom <= 1:
            hiddenParts = np.random.choice(len(parts), numHidden, replace=False)
            known = np.ones_like(observation)
            leftReturn = args.consecHidden
            observation = np.copy(observation)
            for part in hiddenParts:
                observation[parts[part]] = 0
                known[parts[part]] = 0
        else:
            # hide the same parts as the last timestep (UNUSED)
            known = np.copy(lastKnown)
            observation = observation * known
            leftReturn = leftTilRandom - 1
    
        return observation, known, leftReturn
    # default
    return observation, np.ones_like(observation), leftTilRandom

# return agent's state belief based on the previous state, current action, and trajectory before that
# will be combined with observable components to yield final predicted observation
def getStateBelief(observations, knowns, acts=None, statepred = None, mins=None, maxes=None):
    # explicit prediction
    if args.statepred:
        obsfeat = torch.tensor(np.concatenate([observations, acts], axis=-1)).float()
        obsfeat = obsfeat.unsqueeze_(1)
        if args.cuda:
            obsfeat = obsfeat.cuda()
        with torch.no_grad():
            toReturn = statepred(obsfeat, mins, maxes).squeeze()
        if args.cuda:
            toReturn = toReturn.cpu().detach().numpy()
        else:
            toReturn = toReturn.detach().numpy()
        # state predictive model output is actually DIFFERENCE between observations
        # add to previous observation
        return toReturn + observations[-1]
    
    # no explicit permission - assume values unchanged
    return observations[-1]

# set seeds for reproducibility 
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# set up default (training) environment, and one for testing
# seed them with the correct value
env = gym.make(args.env)
env.seed(args.seed)
testenv = gym.make(args.env)
testenv.seed(args.seed)

# initialize networks
actor = Actor(env)
qf1 = SoftQNetwork(env)
qf2 = SoftQNetwork(env)
qf1_target = SoftQNetwork(env)
qf2_target = SoftQNetwork(env)
rrder = RRDModel(env)
if (args.statepred):
    if args.statemodel == "nn":
        statepred = StateNNNetwork(env)
        statepred_target = StateNNNetwork(env)
    elif args.statemodel == "lstm":
        # exit()
        statepred = StateLSTMNetwork(env)
        statepred_target = StateLSTMNetwork(env)
else:
    statepred = None
    statepred_target = None
if args.cuda:
    actor = actor.cuda()
    qf1 = qf1.cuda()
    qf2 = qf2.cuda()
    qf1_target = qf1_target.cuda()
    qf2_target = qf2_target.cuda()
    rrder = rrder.cuda()
    if (args.statepred):
        statepred = statepred.cuda()
        statepred_target = statepred_target.cuda()
        statepred_target.load_state_dict(statepred.state_dict())
for p1, p2 in zip(qf1_target.parameters(), qf2_target.parameters()):
    p1.requires_grad = False
    p2.requires_grad = False

# initialize alpha optimization, if used
if (args.alpha_lr > 0):
    logalpha = torch.tensor(0).float()
    if (args.cuda):
        logalpha = logalpha.cuda()
    logalpha.requires_grad = True
    alpha_opt = optim.Adam([logalpha], lr=args.alpha_lr)
    alpha = torch.exp(logalpha)
else: 
    alpha = args.alpha

# copy values to target networks
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())

# initialize optimizers
q_opt = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.qlr)
act_opt = optim.Adam(list(actor.parameters()), lr=args.actlr)
rrd_opt = optim.Adam(list(rrder.parameters()), lr=3e-4)
if args.statepred:
    stateopt = optim.Adam(list(statepred.parameters()), lr=args.statelr)


# method to evaluate current policy
def evaluatePolicy(numRollouts=10):
    totalReward = 0
    totalLoss = 0
    for x in range(numRollouts):
        episodicReward = 0
        episodicLoss = 0
        done = False
        testobs = testenv.reset()
        hiddenLeft = args.consecHidden
        knowns = np.ones_like(testobs)
        obsbuff = [testobs]
        actbuff = []
        while not done:
            start = min(args.context, len(obsbuff))
            if args.contextSAC:
                obsfeat = torch.tensor(np.array(obsbuff[-start:])).float()
            else:
                obsfeat = torch.tensor(testobs).float().unsqueeze(0)
            if args.cuda:
                obsfeat = obsfeat.cuda()
            testaction = actor.only_action(obsfeat, exploration=False)
            if args.cuda:
                testaction = testaction.detach().cpu().numpy().squeeze()
            else:
                testaction = testaction.detach().numpy().squeeze()
            nextobsUn, testreward, done, _info = testenv.step(testaction)
            actbuff.append(testaction)
            nextObsPred = getStateBelief(obsbuff[-start:], knowns, actbuff[-start:], statepred, mins=buff.minStateVals, maxes=buff.maxStateVals)
            nextobs, knowns, hiddenLeft = obsFilter(nextobsUn, args.numHidden, knowns, hiddenLeft)
            testobs = (knowns * nextobs) + ((1 - knowns) * (nextObsPred))
            testObsIdxs = (1 - knowns) != 0

            # don't track loss for timesteps where num hidden is 0 (first observation)
            if testObsIdxs.sum() > 0:
                testLoss = np.mean(np.square(nextObsPred[testObsIdxs] - nextobsUn[testObsIdxs]))
                episodicLoss += testLoss
            obsbuff.append(testobs)
            episodicReward += testreward

        totalLoss += episodicLoss / len(actbuff)
        totalReward += episodicReward
    return totalReward / numRollouts, totalLoss / numRollouts

# setup directories to save data over training
if args.statepred:
    statemodel = args.statemodel
elif args.statefill:
    statemodel = "Fill"
else:
    statemodel = "NoFill"
if args.contextSAC:
    sacstr = f"{args.context}cSAC"
else:
    sacstr = "NoContext"
foldern = f"./saved_mujoco/{args.env}/{sacstr}/{statemodel}/"

# try to avoid collisions when queueing up many experiments to HPC
if (args.logging and args.seed == 1):
    if not os.path.exists(f"{foldern}rewards"):
        os.makedirs(f"{foldern}rewards")
    if not os.path.exists(f"{foldern}actloss"):
        os.makedirs(f"{foldern}actloss")
    if not os.path.exists(f"{foldern}qfloss"):
        os.makedirs(f"{foldern}qfloss")
    if not os.path.exists(f"{foldern}rrdloss"):
        os.makedirs(f"{foldern}rrdloss")
    if not os.path.exists(f"{foldern}statepredloss"):
        os.makedirs(f"{foldern}statepredloss")
    if not os.path.exists(f"{foldern}statepredtestloss"):
        os.makedirs(f"{foldern}statepredtestloss")
    if not os.path.exists(f"{foldern}statepredunreducedloss"):
        os.makedirs(f"{foldern}statepredunreducedloss")
    if not os.path.exists(f"{foldern}staterange"):
        os.makedirs(f"{foldern}staterange")


# keep track of values to report over the course of training
rewardSet = []
rewardList = []
testStateLossList = []
qflosslist = []
actlosslist = []
rrdlosslist = []
statelosslist = []

# initialize trajectory buffer
buff = Buffer(args.bufferSize)

# initialize values for environmental sampling
observation = env.reset()
obs = [observation]
acts = []
dones = [0]
knowns = np.ones_like(observation)
knownses = [knowns]
rewards = []
numSteps = 0
hiddenLeft = 0

actloss = None
qfloss = None


# keep track of runtime
starttime = time.time()

# iterations of complete algorithm
for step in range(int(args.numSteps)):
    # reset policy to re-train
    # UNUSED
    if step == args.policyResetStep:
        qf1 = SoftQNetwork(env)
        qf2 = SoftQNetwork(env)
        qf1_target = SoftQNetwork(env)
        qf2_target = SoftQNetwork(env)
        if args.cuda:
            actor = actor.cuda()
            qf1 = qf1.cuda()
            qf2 = qf2.cuda()
            qf1_target = qf1_target.cuda()
            qf2_target = qf2_target.cuda()
            if (args.statepred):
                statepred = statepred.cuda()
                statepred_target = statepred_target.cuda()
                statepred_target.load_state_dict(statepred.state_dict())
        for p1, p2 in zip(qf1_target.parameters(), qf2_target.parameters()):
            p1.requires_grad = False
            p2.requires_grad = False
        if (args.alpha_lr > 0):
            logalpha = torch.tensor(0).float()
            if (args.cuda):
                logalpha = logalpha.cuda()
            logalpha.requires_grad = True
            alpha_opt = optim.Adam([logalpha], lr=args.alpha_lr)
            alpha = torch.exp(logalpha)
        else: 
            alpha = args.alpha
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())
        q_opt = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.qlr)
        act_opt = optim.Adam(list(actor.parameters()), lr=args.actlr)

    # sample steps from environment
    for envStep in range(args.envSteps):
        with torch.no_grad():
            start = min(args.context, len(obs))
            if args.contextSAC:
                obsfeat = torch.tensor(np.array(obs[-start:])).float()
            else:
                obsfeat = torch.tensor(obs[-1]).float().unsqueeze(0)
            if (envStep % 100) == 55 and step % 10 == 1:
                # print(obsfeat.max())
                None
            if args.cuda:
                obsfeat = obsfeat.cuda()
            action = actor.only_action(obsfeat)
            if args.cuda:
                action = action.detach().cpu().numpy().squeeze()
            else:
                action = action.detach().numpy().squeeze()
            nextObs, reward, done, info = env.step(action)
            acts.append(action)
            # get predicted state
            nextObsPred = getStateBelief(obs[-start:], knowns, acts[-start:], statepred_target, mins=buff.minStateVals, maxes=buff.maxStateVals)
            # get actually observed state
            nextObs, knowns, hiddenLeft = obsFilter(nextObs, args.numHidden, knowns, hiddenLeft)
            # combine predictions with known values, treat as observation going forward
            observation = (knowns * nextObs) + ((1 - knowns) * (nextObsPred))
            knownses.append(knowns)
            rewards.append(reward)
            obs.append(observation)

            # trajectory is finished
            if done:
                # if ended due to time limit - do not treat as "done" for SAC
                if (info.get('TimeLimit.truncated', False)):
                    dones.append(0)
                else:
                    # terminated due to the state, not just truncated
                    dones.append(1)

                # add completed trajectory to buffer
                buff.addElement(obs, acts, rewards, dones, knownses)

                # keep track of environmental steps taken
                numSteps += len(acts)

                # reset environment and variables 
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
    # train state predictive model if we've sampled enough transitions
    if (numSteps) >= args.startLearningState:
        # train state pred network
        if args.statepred:
            bsize = (args.train_batches * 256) // args.numBreaks
            for x in range(args.stateTrainMult*args.numBreaks):
                feats, labels, mask, lengths = buff.sampleForStatePred(bsize)
                feats = torch.nn.utils.rnn.pack_padded_sequence(feats, lengths, enforce_sorted=False)
                labels = torch.tensor(labels).float()
                mask = torch.tensor(mask).bool()
                if args.cuda:
                    feats = feats.cuda()
                    labels = labels.cuda()
                    mask = mask.cuda()

                preds = statepred(feats, buff.minStateVals, buff.maxStateVals)
                stateLossUnreduced = (preds - labels)**2

                # compute loss for observed components ONLY!! crucial!
                statePredLoss = (stateLossUnreduced * mask).mean()
                if args.cuda:
                    stateLossUnreduced = stateLossUnreduced.cpu()
                    mask = mask.cpu()
                stateLossUnreduced = stateLossUnreduced.detach().numpy()
                mask = mask.detach().numpy()

                # keep track of per-component loss for analysis
                stateLossUnreduced = np.average(stateLossUnreduced, axis=0, weights=mask)

                stateopt.zero_grad()
                statePredLoss.backward()
                nn.utils.clip_grad_value_(statepred.parameters(), 1.0)
                stateopt.step()

    if (numSteps) >= args.startLearning:
        for trainstep in range(args.train_batches):
            # 1 update here for every environment step
            # train RRD network
            samples = buff.sampleSubSeqs(64, 4)
            # transform samples to np arrays to prep for training
            obSamp = np.array(samples['obs'])
            actSamp = np.array(samples['actions'])
            if len(actSamp.shape) < len(obSamp.shape):
                actSamp = np.expand_dims(actSamp, -1)
            ob2Samp = np.array(samples['nextobs'])
            rewSamp = np.array(samples['rewards'])
            knownSamp = np.array(samples['knowns'])

            # construct features for RRD network
            feats = torch.tensor(np.concatenate((obSamp, actSamp, obSamp - ob2Samp), axis=-1)).float()
            if args.cuda:
                feats = feats.cuda()
            
            # compute predicted rewards
            rHat = rrder(feats).squeeze(-1)

            # calculate per-trajectory sums
            episodicSums = torch.mean(rHat, dim=-1, keepdim=True)

            # labels are per-trajectory totals averaged out to each transition (see RRD paper)
            # stochastically strikes a balance between least-squares and uniform reward decomp
            labels = torch.tensor(rewSamp).float()
            if args.cuda:
                labels = labels.cuda()
            rrdloss = F.mse_loss(episodicSums, labels)

            # apply gradient descent
            rrd_opt.zero_grad()
            rrdloss.backward()
            rrd_opt.step()

            # train Q-value networks
            samples = buff.sample(256)

            # transform sample, prep for learning
            obSamp = np.array([samp['obs'] for samp in samples])
            actSamp = np.array([samp['actions'] for samp in samples])
            if len(actSamp.shape) < len(obSamp.shape):
                actSamp = np.expand_dims(actSamp, -1)
            nextObsSamp = np.array([samp['nextobs'] for samp in samples])
            donesSamp = torch.tensor([samp['dones'] for samp in samples]).float().unsqueeze(-1)
            if args.cuda:
                donesSamp = donesSamp.cuda()

            with torch.no_grad():
                # RRD features
                feats = torch.tensor(np.concatenate((obSamp, actSamp, obSamp - nextObsSamp), axis=-1)).float()
                # observation features (for actor network)
                obsfeats = torch.tensor(nextObsSamp).float()
                if(args.cuda):
                    feats = feats.cuda()
                    obsfeats = obsfeats.cuda()
                # calculate expected rewards
                rHat = rrder(feats)
                # re-calculate action from current observation
                nextActions, logdev, logprob = actor.get_action(obsfeats, debug=False, exploration=True)
                # using updated action, predict Q values (NOT training networks here)
                nextFeats = torch.concat([obsfeats, nextActions], dim=-1).float()
                if args.cuda:
                    nextFeats = nextFeats.cuda()
                qtarg1 = qf1_target(nextFeats)
                qtarg2 = qf2_target(nextFeats)
                qtargmin = torch.min(qtarg1, qtarg2) - (alpha * logprob)
                
                # target Q (label for Q value networks)
                # see SAC paper for more detailed understanding
                qtarget = rHat + (args.gamma * (1 - donesSamp) * qtargmin)
            
            # feats for Q value networks - training this time!
            feats = np.concatenate((obSamp, actSamp), axis=-1)
            feats1 = torch.tensor(feats).float()
            feats2 = torch.tensor(feats).float()
            if args.cuda:
                feats1 = feats1.cuda()
                feats2 = feats2.cuda()
            qf1vals = qf1(feats1)
            qf2vals = qf2(feats2)

            qf1loss = F.mse_loss(qf1vals, qtarget)
            qf2loss = F.mse_loss(qf2vals, qtarget)
            qfloss = qf1loss + qf2loss

            q_opt.zero_grad()
            qfloss.backward()
            q_opt.step()


            # train actor network

            # features for actor network
            obsfeat = torch.tensor(obSamp).float()
            if args.cuda:
                obsfeat = obsfeat.cuda()
            actSamp, logdev, logprob = actor.get_action(obsfeat, exploration=True)
            # features for Q value networks, using new action
            feats = torch.concat((obsfeat, actSamp), dim=-1).float()
            if args.cuda:
                feats = feats.cuda()
            # disable gradient computation for Q value networks
            # can't use torch.no_grad because we NEED grad to backpropagate to actor
            for p1, p2 in zip(qf1.parameters(), qf2.parameters()):
                p1.requires_grad = False
                p2.requires_grad = False
            qf1vals = qf1(feats)
            qf2vals = qf2(feats)
            # compute actor loss
            actloss = torch.mean((alpha * logprob) - torch.min(qf1vals, qf2vals))

            act_opt.zero_grad()
            actloss.backward()
            act_opt.step()
            # re-enable gradient for q value nets
            for p1, p2 in zip(qf1.parameters(), qf2.parameters()):
                p1.requires_grad = True
                p2.requires_grad = True
            # if we're learning alpha, update it here based on logprop from actor network and shape of action space
            if (args.alpha_lr > 0):
                alpha_opt.zero_grad()
                with torch.no_grad():
                    multiplier = logprob - np.prod(env.action_space.shape)
                aloss = -1.0*(torch.exp(logalpha) * multiplier).mean()
                aloss.backward()
                alpha_opt.step()
                alpha = torch.exp(logalpha)

            # update target networks
            with torch.no_grad():
                for qt1w, qf1w in zip(qf1_target.parameters(), qf1.parameters()):
                    qt1w.data.copy_(0.995 * qt1w.data + (.005 * qf1w.data))
                for qt2w, qf2w in zip(qf2_target.parameters(), qf2.parameters()):
                    qt2w.data.copy_(0.995 * qt2w.data + (.005 * qf2w.data))
                if args.statepred:
                    for spt, sp in zip(statepred_target.parameters(), statepred.parameters()):
                        spt.data.copy_(0.995 * spt.data + (.005 * sp.data))
    # report current results
    if numSteps >= args.startLearningState:
        if (step % 50) == 0 or (step == args.numSteps - 1):
            if args.statepred:
                statemodel = args.statemodel
            elif args.statefill:
                statemodel = "Fill"
            else:
                statemodel = "NoFill"
            if args.contextSAC:
                sacstr = f"{args.context}cSAC"
            # folder + filenames based on experiment parameters
            foldern = f"./saved_mujoco/{args.env}/{sacstr}/{statemodel}/"
            fname = f"{args.numHidden}Hidden{args.seed}QLR{args.qlr}ALR{args.actlr}SLR{args.statelr}Start{args.startLearningState},{args.startLearning}HS{args.hiddenSizeLSTM}C{args.context}RST{args.policyResetStep}.csv"

            # evaluate current policy 
            with torch.no_grad():
                testrew, testloss = evaluatePolicy()
                rewardList.append(testrew)
            
            # save state prediction losses to files
            if (args.statepred and args.logging):
                testStateLossList.append(testloss)
                statelosslist.append(statePredLoss.item())
                np.savetxt(f"{foldern}/statepredtestloss/{fname}", testStateLossList, delimiter="\n")
                np.savetxt(f"{foldern}/statepredloss/{fname}", statelosslist, delimiter="\n")

            # print statistics to terminal
            if not actloss is None:
                if args.statepred:
                    print(f"Steps: {step * args.envSteps}, Time: {time.time() - starttime:.3f}s, Test rewards: {rewardList[-1]:.3f}, Actor loss: {actloss.item():.3e}, Q loss: {qfloss.item():.3e}, RRD loss: {rrdloss.item():.3e}, Alpha: {alpha:.3e}, StateLoss: {statePredLoss.item():.3e}, Test StateLoss: {testloss:.3e}")
                    # if (args.cuda):
                    #     print(f"GPU: {torch.cuda.max_memory_allocated(device=None)}")
                else:
                    print(f"Steps: {step * args.envSteps}, Time: {time.time() - starttime:.3f}s, Test rewards: {rewardList[-1]:.3f}, Actor loss: {actloss.item():.3e}, Q loss: {qfloss.item():.3e}, RRD loss: {rrdloss.item():.3e}, Alpha: {alpha:.3e}")    
            else:
                print(rewardList[-1])

            # save rewards, RRD, and SAC losses to files
            if (args.logging):
                np.savetxt(f"{foldern}/rewards/{fname}", rewardList, delimiter="\n")
                if not actloss is None:
                    actlosslist.append(actloss.item())
                    qflosslist.append(qfloss.item())
                    rrdlosslist.append(rrdloss.item())
                    np.savetxt(f"{foldern}/actloss/{fname}", actlosslist, delimiter="\n")
                    np.savetxt(f"{foldern}/qfloss/{fname}", qflosslist, delimiter="\n")
                    np.savetxt(f"{foldern}/rrdloss/{fname}", rrdlosslist, delimiter="\n")
