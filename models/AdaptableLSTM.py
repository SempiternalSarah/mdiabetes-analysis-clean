from torch import nn
import numpy as np
import torch
from models.base import Base, N_RESP_CATEGORIES
from models.ModelUtils import *

class AdaptableLSTM(Base):
    
    def __init__(self, *args, **kw):
        # Simple model: lstm block with two linear output layers,
        #    one to predict the response of either label (question)
        # input_size:  size of linear input to model
        # hidden_size: size of lstm hidden layer
        # output_size: size of output label of data
        super().__init__(*args, **kw)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size)
        self.normlayer = nn.BatchNorm1d(num_features=self.hidden_size)
        if (self.transformer):
            self.tlayer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=1, dim_feedforward=self.hidden_size, batch_first=True)

        if self.splitWeeklyQuestions or self.splitModel:
            outputSize = self.output_size
        else:
            outputSize = self.output_size // 2
        
        if (self.splitModel):
            if (self.hierarchical == "Shared"):
                # layers for the no response / response classification
                self.physicalLayerRNR = nn.Linear(self.hidden_size, 2)
                self.consumptionLayerRNR = nn.Linear(self.hidden_size, 2)
                self.knowledgeLayerRNR = nn.Linear(self.hidden_size, 2)
                # layers for the level of response classification
                if (self.regression):
                    osize = 1
                else:
                    osize = (outputSize * 3) // 4
                self.physicalLayer = nn.Linear(self.hidden_size, osize)
                self.consumptionLayer = nn.Linear(self.hidden_size, osize)
                self.knowledgeLayer = nn.Linear(self.hidden_size, osize)
            else:
                self.physicalLayer = nn.Linear(self.hidden_size, outputSize)
                self.consumptionLayer = nn.Linear(self.hidden_size, outputSize)
                self.knowledgeLayer = nn.Linear(self.hidden_size, outputSize)
        else:
            if (self.hierarchical == "Shared"):
                # layers for the no response / response classification
                self.fc_q2_1 = nn.Linear(self.hidden_size, 2)
                self.fc_q1_1 = nn.Linear(self.hidden_size, 2)
                # layers for the level of response classification
                if (self.regression):
                    osize = 1
                else:
                    osize = (outputSize * 3) // 4
                self.fc_q2_2 = nn.Linear(self.hidden_size, osize)
                self.fc_q1_2 = nn.Linear(self.hidden_size, osize)
            self.fc_q2 = nn.Linear(self.hidden_size, outputSize)
            self.fc_q1 = nn.Linear(self.hidden_size, outputSize)
            
        self.relu = nn.ReLU()

    def maybe_zero_weights(self, trainConsumption=True, trainKnowledge=True, trainPhys=True, do="All"):
        if not self.splitModel or (trainConsumption and trainKnowledge and trainPhys):
            return
        if (self.hierarchical == "Shared"):
            self.lstm.weight_ih_l0.grad = None
            self.lstm.bias_ih_l0.grad = None
            self.lstm.weight_hh_l0.grad = None
            self.lstm.bias_hh_l0.grad = None
            if (not trainConsumption):
                self.consumptionLayerRNR.weight.grad = None
                self.consumptionLayerRNR.bias.grad = None 
                self.consumptionLayer.weight.grad = None
                self.consumptionLayer.bias.grad = None
            if (not trainKnowledge):
                self.knowledgeLayerRNR.weight.grad = None
                self.knowledgeLayerRNR.bias.grad = None
                self.knowledgeLayer.weight.grad = None
                self.knowledgeLayer.bias.grad = None
            if (not trainPhys):
                self.physicalLayerRNR.weight.grad = None
                self.physicalLayerRNR.bias.grad = None
                self.physicalLayer.weight.grad = None
                self.physicalLayer.bias.grad = None
        else:
            self.lstm.weight_ih_l0.grad = None
            self.lstm.bias_ih_l0.grad = None
            self.lstm.weight_hh_l0.grad = None
            self.lstm.bias_hh_l0.grad = None
            if (not trainConsumption):
                self.consumptionLayer.weight.grad = None
                self.consumptionLayer.bias.grad = None
            if (not trainKnowledge):
                self.knowledgeLayer.weight.grad = None
                self.knowledgeLayer.bias.grad = None
            if (not trainPhys):
                self.physicalLayer.weight.grad = None
                self.physicalLayer.bias.grad = None
    
    def forward(self, x):
        if (self.splitModel):
            return self.forward_split(x)

        # One forward pass of input vector/batch x
        # Pass x thru the LSTM cell, then pass output
        #    through each linear output layer for each
        #    response prediction. 
        # Then join predictions together as one vector
        H0, C0 = self._init_hc()
        output, (H,C) = self.lstm(x, (H0, C0))
        output = self.normlayer(output)
        if (self.transformer):
            mask = torch.zeros((x.shape[0], x.shape[0]))
            for i in range(x.shape[0]):
                for j in range(x.shape[0]):
                    if (i < j):
                        mask[i, j] = 1
            mask = mask.bool()
            output = self.tlayer(output, src_mask=mask)
        # print(H, output)
        # [SEQ, hidden_size]
        RvsNR = None
        out = self.relu(output)
        if (self.regression):
            if (self.hierarchical == "Shared"):
                RvsNR = self.fc_q1_1(out).softmax(-1)
                classpreds = self.fc_q1_2(out).clamp(1, 3)
                if (not self.splitWeeklyQuestions):
                    RvsNR = torch.cat([RvsNR, self.fc_q2_1(out).softmax(-1)], 0)
                    classpreds = torch.cat([classpreds, self.fc_q2_2(out).clamp(1, 3)], 0)
                # print(classpreds.shape, RvsNR.shape)
                toReturn = torch.where((RvsNR[:, 0] > RvsNR[:, 1]).unsqueeze(-1), (RvsNR[:, 1]).unsqueeze(-1), classpreds)
                # print(toReturn.shape)
                if (not self.splitWeeklyQuestions):
                    toReturn = torch.cat([toReturn[0:toReturn.shape[0] // 2], toReturn[toReturn.shape[0]:]], -1)
                return toReturn, RvsNR

            if (self.no_response_class):
                minClamp = 0
            else:
                minClamp = 1
            out_q1 = self.fc_q1(out).clamp(minClamp, 3)
            # [SEQ, output_size//2]
            if (self.splitWeeklyQuestions):
                return out_q1, RvsNR
            else:
                out_q2 = self.fc_q2(out).clamp(minClamp, 3)
                # [SEQ, output_size//2]
                return torch.cat([out_q1, out_q2],-1), RvsNR

        else:
            if (self.hierarchical == "Shared"):
                RvsNR = self.fc_q1_1(out).softmax(-1)
                classpreds = self.fc_q1_2(out).softmax(-1)
                if (not self.splitWeeklyQuestions):
                    RvsNR = torch.cat([RvsNR, self.fc_q2_1(out).softmax(-1)], 0)
                    classpreds = torch.cat([classpreds, self.fc_q2_2(out).softmax(-1)], 0)
                toReturn = torch.cat([RvsNR[:, 0].unsqueeze(-1), (RvsNR[:, 1]).unsqueeze(-1) * classpreds], -1)
                if (not self.splitWeeklyQuestions):
                    toReturn = torch.cat([toReturn[0:toReturn.shape[0] // 2], toReturn[toReturn.shape[0]:]], -1)
                return toReturn, RvsNR
            out_q1 = self.fc_q1(out).softmax(-1)
            # [SEQ, output_size//2]
            if (self.splitWeeklyQuestions):
                return out_q1, RvsNR
            else:
                out_q2 = self.fc_q2(out).softmax(-1)
                # [SEQ, output_size//2]
                return torch.cat([out_q1, out_q2],-1), RvsNR

    def forward_split(self, x):
        # One forward pass of input vector/batch x
        # Pass x thru the LSTM cell, then pass output
        #    through each linear output layer for each
        #    response prediction. 
        # Then join predictions together as one vector
        H0, C0 = self._init_hc()
        datas, (H,C) = self.lstm(x, (H0, C0))
        datas = self.normlayer(datas)
        if (self.transformer):
            mask = torch.zeros((x.shape[0], x.shape[0]))
            for i in range(x.shape[0]):
                for j in range(x.shape[0]):
                    if (i < j):
                        mask[i, j] = 1
            mask = mask.bool()
            datas = self.tlayer(datas, src_mask=mask)
        RvsNR = None
        if (self.regression):
            osize = 1
        elif (self.hierarchical == "Shared"):
            osize = (self.output_size * 3) // 4
        else:
            osize = self.output_size
        if (not self.splitWeeklyQuestions):
            pred = torch.zeros([datas.shape[0] * 2, osize])
            pred.requires_grad = True
            # separate data by category for each weekly question
            # then, recombine the predictions using the indices
            consumptionRows2 = (torch.where(x[:, -1] == 0, 1, 0) * torch.where(x[:, -2] == 0, 1, 0)).nonzero()
            if consumptionRows2.numel() > 0:
                # print("c2")
                consumptionRows2 = consumptionRows2.squeeze(dim=-1)
                cpred2 = self.consumptionLayer(datas[consumptionRows2])
                # these are predictions for weekly question 2
                # add them to the tensor after all values for weekly question 1
                # print(cpred2.shape, consumptionRows2.shape)
                pred = pred.index_add(0, consumptionRows2 + datas.shape[0], cpred2)
            knowledgeRows2 = (torch.where(x[:, -1] == 0, 1, 0) * torch.where(x[:, -2] == 1, 1, 0)).nonzero()
            if knowledgeRows2.numel() > 0:
                # print("k2")
                knowledgeRows2 = knowledgeRows2.squeeze(dim=-1)
                kpred2 = self.knowledgeLayer(datas[knowledgeRows2])
                # these are predictions for weekly question 2
                # add them to the tensor after all values for weekly question 1
                # print(kpred2.shape, knowledgeRows2)
                pred = pred.index_add(0, knowledgeRows2 + datas.shape[0], kpred2)
            physRows2 = (torch.where(x[:, -1] == 1, 1, 0) * torch.where(x[:, -2] == 0, 1, 0)).nonzero()
            if physRows2.numel() > 0:
                # print("p2")
                physRows2 = physRows2.squeeze(dim=-1)
                ppred2 = self.physicalLayer(datas[physRows2])
                # these are predictions for weekly question 2
                # add them to the tensor after all values for weekly question 1
                pred = pred.index_add(0, physRows2 + datas.shape[0], ppred2)
            consumptionRows1 = (torch.where(x[:, -3] == 0, 1, 0) * torch.where(x[:, -4] == 0, 1, 0)).nonzero()
            if consumptionRows1.numel() > 0:
                # print("c1")
                consumptionRows1 = consumptionRows1.squeeze(dim=-1)
                cpred1 = self.consumptionLayer(datas[consumptionRows1])
                pred = pred.index_add(0, consumptionRows1, cpred1)
            knowledgeRows1 = (torch.where(x[:, -3] == 0, 1, 0) * torch.where(x[:, -4] == 1, 1, 0)).nonzero()
            if knowledgeRows1.numel() > 0:
                # print("k1")
                knowledgeRows1 = knowledgeRows1.squeeze(dim=-1)
                kpred1 = self.knowledgeLayer(datas[knowledgeRows1])
                pred = pred.index_add(0, knowledgeRows1, kpred1)
            physRows1 = (torch.where(x[:, -3] == 1, 1, 0) * torch.where(x[:, -4] == 0, 1, 0)).nonzero()
            if physRows1.numel() > 0:
                physRows1 = physRows1.squeeze(dim=-1)
                # print("p1", physRows1, datas)
                ppred1 = self.physicalLayer(datas[physRows1])
                pred = pred.index_add(0, physRows1, ppred1)
            if (self.regression):
                if (self.no_response_class and not self.hierarchical == "Shared"):
                    pred = pred.clamp(0, 3)
                else:
                    pred = pred.clamp(1, 3)
            else:
                pred = pred.softmax(-1)
        else:
            pred = torch.zeros([datas.shape[0], osize])
            pred.requires_grad = True
            # separate data by category for each weekly question
            # then, recombine the predictions using the indices
            consumptionRows = (torch.where(x[:, -1] == 0, 1, 0) * torch.where(x[:, -2] == 0, 1, 0)).nonzero()
            if consumptionRows.numel() > 0:
                # print("c2")
                consumptionRows = consumptionRows.squeeze(dim=-1)
                cpred = self.consumptionLayer(datas[consumptionRows])
                # print(pred.shape, cpred.shape, consumptionRows.shape)
                # print(cpred2.shape, consumptionRows2.shape)
                pred = pred.index_add(0, consumptionRows, cpred)
            knowledgeRows = (torch.where(x[:, -1] == 0, 1, 0) * torch.where(x[:, -2] == 1, 1, 0)).nonzero()
            if knowledgeRows.numel() > 0:
                # print("k2")
                knowledgeRows = knowledgeRows.squeeze(dim=-1)
                kpred = self.knowledgeLayer(datas[knowledgeRows])
                # print(kpred2.shape, knowledgeRows2)
                pred = pred.index_add(0, knowledgeRows, kpred)
            physRows = (torch.where(x[:, -1] == 1, 1, 0) * torch.where(x[:, -2] == 0, 1, 0)).nonzero()
            if physRows.numel() > 0:
                # print("p2")
                physRows = physRows.squeeze(dim=-1)
                ppred = self.physicalLayer(datas[physRows])
                pred = pred.index_add(0, physRows, ppred)
            if (self.regression):
                if (self.no_response_class):
                    minClamp = 0
                else:
                    minClamp = 1
                pred = pred.clamp(minClamp, 3)
            else:
                pred = pred.softmax(-1)
        if (self.hierarchical == "Shared"):
            if (not self.splitWeeklyQuestions):
                RvsNR = torch.zeros([datas.shape[0] * 2, 2])
                RvsNR.requires_grad = True
                # separate data by category for each weekly question
                # then, recombine the RvsNRictions using the indices
                consumptionRows2 = (torch.where(x[:, -1] == 0, 1, 0) * torch.where(x[:, -2] == 0, 1, 0)).nonzero()
                if consumptionRows2.numel() > 0:
                    # print("c2")
                    consumptionRows2 = consumptionRows2.squeeze(dim=-1)
                    cRvsNR2 = self.consumptionLayerRNR(datas[consumptionRows2])
                    # these are RvsNRictions for weekly question 2
                    # add them to the tensor after all values for weekly question 1
                    # print(cRvsNR2.shape, consumptionRows2.shape)
                    RvsNR = RvsNR.index_add(0, consumptionRows2 + datas.shape[0], cRvsNR2)
                knowledgeRows2 = (torch.where(x[:, -1] == 0, 1, 0) * torch.where(x[:, -2] == 1, 1, 0)).nonzero()
                if knowledgeRows2.numel() > 0:
                    # print("k2")
                    knowledgeRows2 = knowledgeRows2.squeeze(dim=-1)
                    kRvsNR2 = self.knowledgeLayerRNR(datas[knowledgeRows2])
                    # these are RvsNRictions for weekly question 2
                    # add them to the tensor after all values for weekly question 1
                    # print(kRvsNR2.shape, knowledgeRows2)
                    RvsNR = RvsNR.index_add(0, knowledgeRows2 + datas.shape[0], kRvsNR2)
                physRows2 = (torch.where(x[:, -1] == 1, 1, 0) * torch.where(x[:, -2] == 0, 1, 0)).nonzero()
                if physRows2.numel() > 0:
                    # print("p2")
                    physRows2 = physRows2.squeeze(dim=-1)
                    pRvsNR2 = self.physicalLayerRNR(datas[physRows2])
                    # these are RvsNRictions for weekly question 2
                    # add them to the tensor after all values for weekly question 1
                    RvsNR = RvsNR.index_add(0, physRows2 + datas.shape[0], pRvsNR2)
                consumptionRows1 = (torch.where(x[:, -3] == 0, 1, 0) * torch.where(x[:, -4] == 0, 1, 0)).nonzero()
                if consumptionRows1.numel() > 0:
                    # print("c1")
                    consumptionRows1 = consumptionRows1.squeeze(dim=-1)
                    cRvsNR1 = self.consumptionLayerRNR(datas[consumptionRows1])
                    RvsNR = RvsNR.index_add(0, consumptionRows1, cRvsNR1)
                knowledgeRows1 = (torch.where(x[:, -3] == 0, 1, 0) * torch.where(x[:, -4] == 1, 1, 0)).nonzero()
                if knowledgeRows1.numel() > 0:
                    # print("k1")
                    knowledgeRows1 = knowledgeRows1.squeeze(dim=-1)
                    kRvsNR1 = self.knowledgeLayerRNR(datas[knowledgeRows1])
                    RvsNR = RvsNR.index_add(0, knowledgeRows1, kRvsNR1)
                physRows1 = (torch.where(x[:, -3] == 1, 1, 0) * torch.where(x[:, -4] == 0, 1, 0)).nonzero()
                if physRows1.numel() > 0:
                    physRows1 = physRows1.squeeze(dim=-1)
                    # print("p1", physRows1, datas)
                    pRvsNR1 = self.physicalLayerRNR(datas[physRows1])
                    RvsNR = RvsNR.index_add(0, physRows1, pRvsNR1)
                RvsNR = RvsNR.softmax(-1)
            else:
                RvsNR = torch.zeros([datas.shape[0], 2])
                RvsNR.requires_grad = True
                # separate data by category for each weekly question
                # then, recombine the RvsNRictions using the indices
                consumptionRows = (torch.where(x[:, -1] == 0, 1, 0) * torch.where(x[:, -2] == 0, 1, 0)).nonzero()
                if consumptionRows.numel() > 0:
                    # print("c2")
                    consumptionRows = consumptionRows.squeeze(dim=-1)
                    cRvsNR = self.consumptionLayerRNR(datas[consumptionRows])
                    # print(cRvsNR2.shape, consumptionRows2.shape)
                    RvsNR = RvsNR.index_add(0, consumptionRows, cRvsNR)
                knowledgeRows = (torch.where(x[:, -1] == 0, 1, 0) * torch.where(x[:, -2] == 1, 1, 0)).nonzero()
                if knowledgeRows.numel() > 0:
                    # print("k2")
                    knowledgeRows = knowledgeRows.squeeze(dim=-1)
                    kRvsNR = self.knowledgeLayerRNR(datas[knowledgeRows])
                    # print(kRvsNR2.shape, knowledgeRows2)
                    RvsNR = RvsNR.index_add(0, knowledgeRows, kRvsNR)
                physRows = (torch.where(x[:, -1] == 1, 1, 0) * torch.where(x[:, -2] == 0, 1, 0)).nonzero()
                if physRows.numel() > 0:
                    # print("p2")
                    physRows = physRows.squeeze(dim=-1)
                    pRvsNR = self.physicalLayerRNR(datas[physRows])
                    # print(kRvsNR2.shape, knowledgeRows2)
                    RvsNR = RvsNR.index_add(0, physRows, pRvsNR)
                RvsNR = RvsNR.softmax(-1)

            if (self.regression):
                # print(pred.shape)
                pred = torch.where((RvsNR[:, 0] > RvsNR[:, 1]).unsqueeze(-1), RvsNR[:, 1].unsqueeze(-1), pred)
                # print(pred.shape)
            else:
                pred = torch.cat([RvsNR[:, 0].unsqueeze(-1), pred * (RvsNR[:, 1].unsqueeze(-1))], -1)
        if (not self.splitWeeklyQuestions):
            # reshape to match single model output
            # final shape is 2 questions per row (1 row = 1 week for this participant)
            pred = torch.cat((pred[0:datas.shape[0]], pred[datas.shape[0]:]), dim = -1)
        return pred, RvsNR
    
    def w1_reg(self, y):
        k = self.fc_q1.out_features
        y1 = y[:, :k]
        y2 = y[:, k:]
        t = y.shape[0]
        if (t < 2):
            return 0
        sum1 = torch.abs(y1[1:] - y1[:-1]).sum()
        sum2 = torch.abs(y2[1:] - y2[:-1]).sum()
        w1 = (sum1 + sum2) / (2 * (t-1))
        return w1
    
    def w2_reg(self, y):
        k = self.fc_q1.out_features
        y1 = y[:, :k]
        y2 = y[:, k:]
        t = y.shape[0]
        if (t < 2):
            return 0
        sum1 = torch.sqrt((y1[1:] - y1[:-1]) ** 2).sum()
        sum2 = torch.sqrt((y2[1:] - y2[:-1]) ** 2).sum()
        w2 = (sum1 + sum2) / (2 * (t-1))
        return w2
        