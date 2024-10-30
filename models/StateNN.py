from torch import nn
import numpy as np
import torch

class StateNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        # Simple model: lstm block with two linear output layers,
        #    one to predict the response of either label (question)
        # input_size:  size of linear input to model
        # hidden_size: size of lstm hidden layer
        # output_size: size of output label of data
        super().__init__()
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
            
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.l1(x)
        out = self.relu(output)
        out = self.relu(self.l2(out))
        return out