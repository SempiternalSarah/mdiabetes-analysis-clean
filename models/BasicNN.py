from torch import nn
import numpy as np
import torch
from models.ModelUtils import *
from models.base import Base, N_RESP_CATEGORIES

class BasicNN(Base):
    
    def __init__(self, *args, **kw):
        # Simple model: lstm block with two linear output layers,
        #    one to predict the response of either label (question)
        # input_size:  size of linear input to model
        # hidden_size: size of lstm hidden layer
        # output_size: size of output label of data
        super().__init__(*args, **kw)
        self.inputLayer = nn.Linear(self.input_size, self.hidden_size)
        if self.splitWeeklyQuestions or self.splitModel:
            outputSize = self.output_size
        else:
            outputSize = self.output_size // 2
        if (self.transformer):
            self.tlayer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=1, dim_feedforward=self.hidden_size, batch_first=True)
        if (self.hierarchical == "Shared"):
            if (self.regression):
                osize = 1
            else:
                osize = (outputSize * 3) // 4
            self.fc_q1 = nn.Linear(self.hidden_size, osize)
            self.fc_q2 = nn.Linear(self.hidden_size, osize)
            self.fc_q1RNR = nn.Linear(self.hidden_size, 2)
            self.fc_q2RNR = nn.Linear(self.hidden_size, 2)
        else:
            self.fc_q1 = nn.Linear(self.hidden_size, outputSize)
            self.fc_q2 = nn.Linear(self.hidden_size, outputSize)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        output = self.inputLayer(x)
        out = self.relu(output)
        RvsNR = None
        if (self.regression):
            if (self.hierarchical == "Shared"):
                RvsNR = self.fc_q1RNR(out).softmax(-1)
                classpreds = self.relu(self.fc_q1(out)) + 1
                if (not self.splitWeeklyQuestions):
                    RvsNR = torch.cat([RvsNR, self.fc_q2RNR(out).softmax(-1)], 0)
                    classpreds = torch.cat([classpreds, self.relu(self.fc_q2(out))], 0)
                # print(classpreds.shape, RvsNR.shape)
                toReturn = torch.where((RvsNR[:, 0] > RvsNR[:, 1]).unsqueeze(-1), (RvsNR[:, 1]).unsqueeze(-1), classpreds)
                # print(toReturn.shape)
                if (not self.splitWeeklyQuestions):
                    toReturn = torch.cat([toReturn[0:toReturn.shape[0] // 2], toReturn[toReturn.shape[0]:]], -1)
                return toReturn, RvsNR
            out_q1 = self.relu(self.fc_q1(out))
            if (self.splitModel or self.splitWeeklyQuestions):
                return out_q1, RvsNR
            else:
                out_q2 = self.relu(self.fc_q2(out))
                return torch.cat([out_q1, out_q2],-1), RvsNR
        else:
            if (self.hierarchical == "Shared"):
                RvsNR = self.fc_q1RNR(out).softmax(-1)
                classpreds = self.fc_q1(out).softmax(-1)
                if (not self.splitWeeklyQuestions):
                    RvsNR = torch.cat([RvsNR, self.fc_q2RNR(out).softmax(-1)], 0)
                    classpreds = torch.cat([classpreds, self.fc_q2(out).softmax(-1)], 0)
                toReturn = torch.cat([RvsNR[:, 0].unsqueeze(-1), (RvsNR[:, 1]).unsqueeze(-1) * classpreds], -1)
                if (not self.splitWeeklyQuestions):
                    toReturn = torch.cat([toReturn[0:toReturn.shape[0] // 2], toReturn[toReturn.shape[0]:]], -1)
                return toReturn, RvsNR
            out_q1 = self.fc_q1(out).softmax(-1)
            if (self.splitModel or self.splitWeeklyQuestions):
                return out_q1, RvsNR
            else:
                out_q2 = self.fc_q2(out).softmax(-1)
                return torch.cat([out_q1, out_q2],-1), RvsNR

    def maybe_zero_weights(self, trainConsumption=True, trainKnowledge=True, trainPhys=True, do="All"):
        if not self.splitModel or (trainConsumption and trainKnowledge and trainPhys):
            return
        if (not trainConsumption and (do == "All" or do == "consumption")):
            self.inputLayer.weight.grad = None
            self.inputLayer.bias.grad = None
            self.fc_q1.weight.grad = None
            self.fc_q2.weight.grad = None
            self.fc_q1.bias.grad = None
            self.fc_q2.bias.grad = None
        if (not trainKnowledge and (do == "All" or do == "knowledge")):
            self.inputLayer.weight.grad = None
            self.inputLayer.bias.grad = None
            self.fc_q1.weight.grad = None
            self.fc_q2.weight.grad = None
            self.fc_q1.bias.grad = None
            self.fc_q2.bias.grad = None
        if (not trainPhys and (do == "All" or do == "physical")):
            self.inputLayer.weight.grad = None
            self.inputLayer.bias.grad = None
            self.fc_q1.weight.grad = None
            self.fc_q2.weight.grad = None
            self.fc_q1.bias.grad = None
            self.fc_q2.bias.grad = None
