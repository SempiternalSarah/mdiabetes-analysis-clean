from torch import nn
import numpy as np
import torch
from models.ModelUtils import *
from models.base import Base, N_RESP_CATEGORIES

class LogisticRegressor(Base):
    
    def __init__(self, *args, **kw):
        # Simple model: lstm block with two linear output layers,
        #    one to predict the response of either label (question)
        # input_size:  size of linear input to model
        # hidden_size: size of lstm hidden layer
        # output_size: size of output label of data
        super().__init__(*args, **kw)
        self.linear = nn.Linear(self.input_size, self.output_size)
        self.softmax = nn.Softmax(-1)
    
    def forward(self, x):
        preds = self.linear(x)
        # softmax per question, recombine
        if (self.splitModel or self.splitWeeklyQuestions):
            if (self.regression):
                return preds.clamp(0, 3)
            else:
                return self.softmax(preds)
        else:
            if (self.regression):
                return preds.clamp(0, 3)
            else:
                return torch.cat([self.softmax(preds[:, 0:(self.output_size//2)]), self.softmax(preds[:, (self.output_size//2):])], -1)

    def maybe_zero_weights(self, trainConsumption=True, trainKnowledge=True, trainPhys=True, do="All"):
        if not self.splitModel or (trainConsumption and trainKnowledge and trainPhys):
            return
        if (not trainConsumption and (do == "All" or do == "consumption")):
            self.linear.weight.grad = None
            self.linear.weight.grad = None
        if (not trainKnowledge and (do == "All" or do == "knowledge")):
            self.linear.weight.grad = None
            self.linear.weight.grad = None
        if (not trainPhys and (do == "All" or do == "physical")):
            self.linear.weight.grad = None
            self.linear.weight.grad = None