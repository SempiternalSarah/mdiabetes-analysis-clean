from torch import nn
import numpy as np
import torch

class StateLSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        # Simple model: lstm block with two linear output layers,
        #    one to predict the response of either label (question)
        # input_size:  size of linear input to model
        # hidden_size: size of lstm hidden layer
        # output_size: size of output label of data
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.normlayer = nn.BatchNorm1d(num_features = hidden_size)
        self.fc_q1 = nn.Linear(hidden_size, output_size)
            
        self.relu = nn.ReLU()

    def forward(self, x):
        H0, C0 = torch.zeros(1, self.hidden_size), torch.zeros(1, self.hidden_size)
        output, (H,C) = self.lstm(x, (H0, C0))
        out = self.relu(output)
        out = self.relu(self.fc_q1(out))
        return out