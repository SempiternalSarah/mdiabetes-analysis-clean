from utils.behavior_data import BehaviorData
import torch
import numpy as np
import importlib

class Experiment:
    
    def __init__(self, data_kw={}, model="BasicLSTM", model_kw={}, train_kw={}, numValFolds=5, epochsToUpdateLabelMods=5, stateZeroEpochs=0, modelSplit=False, knowSchedule=[], physSchedule=[], consumpSchedule=[], hierarchical=None, nrc=False, only_rnr=False):
        # data_kw:  dict of keyword arguments to BehaviorData instance
        # model_kw: dict of keyword arguments for Model instance
        # train_kw: dict of keyword arguments for training loop
        self.numValFolds = numValFolds
        self.stateZeroEpochs = stateZeroEpochs
        self.data_kw = data_kw
        self.model_name = model
        self.model_kw = model_kw
        self.train_kw = train_kw
        self.nrc = nrc
        self.only_rnr = only_rnr
        # similar to DQN - update label modifications based on network predictions
        # every x epochs
        # used to replace non responses with predicted responses
        self.epochsToUpdateLabelMods = epochsToUpdateLabelMods
        self.hierarchical = hierarchical
        self.knowSchedule = knowSchedule
        self.physSchedule = physSchedule
        self.consumpSchedule = consumpSchedule
        self.trainKnowledge = True
        self.trainPhysical = True
        self.trainConsumption = True

        self.bd = BehaviorData(**data_kw)

        if (self.bd.split_weekly_questions):
            output_size = self.bd.dimensions[1]
        else:
            output_size = self.bd.dimensions[1] // 2
        if modelSplit and "LSTM" not in model:
            self.modelSplit = True
            self.physicalModel = self._get_model()(
                input_size=self.bd.dimensions[0],
                output_size=output_size,
                **model_kw,
            )
            self.knowledgeModel = self._get_model()(
                input_size=self.bd.dimensions[0],
                output_size=output_size,
                **model_kw,
            )
            self.consumptionModel = self._get_model()(
                input_size=self.bd.dimensions[0],
                output_size=output_size,
                **model_kw,
            )
            if (self.hierarchical == "Separate"):
                model_kw["regression"] = False
                self.physicalModelRNR = self._get_model()(
                    input_size=self.bd.dimensions[0],
                    output_size=2,
                    **model_kw,
                )
                self.knowledgeModelRNR = self._get_model()(
                    input_size=self.bd.dimensions[0],
                    output_size=2,
                    **model_kw,
                )
                self.consumptionModelRNR = self._get_model()(
                    input_size=self.bd.dimensions[0],
                    output_size=2,
                    **model_kw,
                )
            # define one model for shared functionality (like score reporting and optimizer creation)
            self.model = self.physicalModel
        else:
            self.modelSplit = False
            self.model = self._get_model()(
                input_size=self.bd.dimensions[0],
                output_size=output_size,
                **model_kw,
            )
            if (self.hierarchical == "Separate"):
                model_kw["regression"] = False
                self.modelRNR = self._get_model()(
                    input_size=self.bd.dimensions[0],
                    output_size=2,
                    **model_kw,
                )
        # yhat = self.model(self.bd.chunkedFeatures[0])
        # make_dot(yhat, params=dict(list(self.model.named_parameters()))).render("rnn_torchviz", format="png")
        

    def runValidation(self):
        # Train the model, report parameters and results
        rep = self.__dict__
        rep = {k: v for k,v in rep.items() if "_kw" in k}
        train_loss, train_metrics, val_metrics, labels = self.train_validation()
        # results = self.evaluate()
        rep["params"] = {
            "data_kw": self.data_kw,
            "model_name": self.model_name,
            "model_kw": self.model_kw,
            "train_kw": self.train_kw,
        }
        # rep["results"] = results
        rep["loss"] = train_loss
        rep["train_metrics"] = train_metrics
        rep["metric_labels"] = labels
        rep["test_metrics"] = val_metrics
        return rep

    def run(self):
        # Train the model, report parameters and results
        rep = self.__dict__
        rep = {k: v for k,v in rep.items() if "_kw" in k}
        train_loss, train_metrics, test_metrics, labels = self.train()
        # results = self.evaluate()
        epochs = self.train_kw.get("epochs", 1)
        rec_every = self.train_kw.get("rec_every", 5)
        rep["rec_epochs"] = np.arange(start=0, stop=epochs, step=rec_every)
        rep["rec_epochs"] = np.append(rep["rec_epochs"], epochs - 1)
        rep["params"] = {
            "data_kw": self.data_kw,
            "model_name": self.model_name,
            "model_kw": self.model_kw,
            "train_kw": self.train_kw,
        }
        # rep["results"] = results
        rep["loss"] = train_loss
        rep["train_metrics"] = train_metrics
        rep["metric_labels"] = labels
        rep["test_metrics"] = test_metrics
        return rep

    def train_validation(self):
        # Loop over data and train model on each batch
        # Returns matrix of loss for each participant
        stored_losses = []
        train_metrics = []
        test_metrics = []
        epochs = self.train_kw.get("epochs", 1)
        rec_every = self.train_kw.get("rec_every", 5)
        foldsets = np.array_split(np.random.permutation(self.bd.train), self.numValFolds)
        for fold in range(self.numValFolds):
            valSet = foldsets[fold]
            trainSet = [i for i in self.bd.train if i not in valSet]
            stored_losses.append([])
            train_metrics.append([])
            test_metrics.append([])
            self.model = self._get_model()(
                input_size=self.bd.dimensions[0],
                output_size=self.bd.dimensions[1],
                **self.model_kw,
            )
            opt, sched = self.model.make_optimizer()
            for e in range(epochs):
                lh = self.train_epoch_val(opt, trainSet)
                if (e%rec_every) == 0 or e == epochs - 1:
                    print(f'{e:}\t', lh)
                    stored_losses[fold].append(lh)
                    metrics, labels = self.report_scores_subset(trainSet)
                    train_metrics[fold].append(metrics)
                    tmetrics, tlabels = self.report_scores_subset(valSet)
                    test_metrics[fold].append(tmetrics)
                sched.step()
        return np.mean(stored_losses, axis=0), np.mean(train_metrics, axis=0), np.mean(test_metrics, axis=0), labels

    def get_class_predictions(self, testset):
        if (testset):
            toUse = self.bd.test
        else:
            toUse = self.bd.train
        preds = None
        for indx in toUse:
            # extract one participants data
            data  = self.bd.get_features(indx)
            label = self.bd.chunkedLabels[indx]
            # important to note that we are not using batching here
            # one participants data is ONE sequence
            pred, RvsNR = self.getPrediction(data)
            # we have to predict with one sequence at a time and then
            # stitch together the predictions and labels to calculate our metrics
            if (preds == None):
                preds = pred
                labels = label
            else: 
                preds = torch.cat([preds, pred], dim = 0)
                labels = torch.cat([labels, label], dim = 0)
        if (not self.bd.split_weekly_questions):
            preds = torch.cat([preds[:, :3], preds[:, 3:]], 0)
            labels = torch.cat([labels[:, :3 + 1], labels[:, 3 + 1:]], 0)
        preds = preds.argmax(dim=1)
        if (self.model.regression):
            p1 = preds[labels[:, 0] == 1]
            p2 = preds[labels[:, 0] == 2]
            p3 = preds[labels[:, 0] == 3]
        else:
            p1 = preds[labels[:, 1] == 1]
            p2 = preds[labels[:, 2] == 1]
            p3 = preds[labels[:, 3] == 1]
        return p1, p2, p3

    # get prediction for sequence of data
    def getPrediction(self, datas, reporting=True):
        RvsNR = None
        if not self.modelSplit:
            pred, RvsNR = self.model.forward(datas)
        elif not self.bd.split_weekly_questions:
            pred = torch.zeros([datas.shape[0] * 2, self.model.output_size])
            if (self.hierarchical == "Shared"):
                RvsNR = torch.zeros([datas.shape[0] * 2, 2])
                RvsNR.requires_grad = True
            pred.requires_grad = True
            # separate data by category for each weekly question
            # then, recombine the predictions using the indices
            # consider computing these indices ahead of time and storing values for all participants?
            consumptionRows2 = (torch.where(datas[:, -1] == 0, 1, 0) * torch.where(datas[:, -2] == 0, 1, 0)).nonzero()
            if consumptionRows2.numel() > 0 and (reporting or self.trainConsumption):
                # print("c2")
                consumptionRows2 = consumptionRows2.squeeze(dim=-1)
                cpred2, temp = self.consumptionModel.forward(datas[consumptionRows2])
                # these are predictions for weekly question 2
                # add them to the tensor after all values for weekly question 1
                consumptionRows2 += datas.shape[0]
                if (self.hierarchical == "Shared"):
                    RvsNR = RvsNR.index_add(0, consumptionRows2, temp)
                # print(cpred2.shape, consumptionRows2.shape)
                pred = pred.index_add(0, consumptionRows2, cpred2)
            knowledgeRows2 = (torch.where(datas[:, -1] == 0, 1, 0) * torch.where(datas[:, -2] == 1, 1, 0)).nonzero()
            if knowledgeRows2.numel() > 0 and (reporting or self.trainKnowledge):
                # print("k2")
                knowledgeRows2 = knowledgeRows2.squeeze(dim=-1)
                kpred2, temp = self.knowledgeModel.forward(datas[knowledgeRows2])
                # these are predictions for weekly question 2
                # add them to the tensor after all values for weekly question 1
                knowledgeRows2 += datas.shape[0]
                if (self.hierarchical == "Shared"):
                    RvsNR = RvsNR.index_add(0, knowledgeRows2, temp)
                # print(kpred2.shape, knowledgeRows2)
                pred = pred.index_add(0, knowledgeRows2, kpred2)
            physRows2 = (torch.where(datas[:, -1] == 1, 1, 0) * torch.where(datas[:, -2] == 0, 1, 0)).nonzero()
            if physRows2.numel() > 0 and (reporting or self.trainPhysical):
                # print("p2")
                physRows2 = physRows2.squeeze(dim=-1)
                ppred2, temp = self.physicalModel.forward(datas[physRows2])
                # these are predictions for weekly question 2
                # add them to the tensor after all values for weekly question 1
                physRows2 += datas.shape[0]
                if (self.hierarchical == "Shared"):
                    RvsNR = RvsNR.index_add(0, physRows2, temp)
                pred = pred.index_add(0, physRows2, ppred2)
            consumptionRows1 = (torch.where(datas[:, -3] == 0, 1, 0) * torch.where(datas[:, -4] == 0, 1, 0)).nonzero()
            if consumptionRows1.numel() > 0 and (reporting or self.trainConsumption):
                # print("c1")
                consumptionRows1 = consumptionRows1.squeeze(dim=-1)
                cpred1, temp = self.consumptionModel.forward(datas[consumptionRows1])
                if (self.hierarchical == "Shared"):
                    RvsNR = RvsNR.index_add(0, consumptionRows1, temp)
                pred = pred.index_add(0, consumptionRows1, cpred1)
            knowledgeRows1 = (torch.where(datas[:, -3] == 0, 1, 0) * torch.where(datas[:, -4] == 1, 1, 0)).nonzero()
            if knowledgeRows1.numel() > 0 and (reporting or self.trainKnowledge):
                # print("k1")
                knowledgeRows1 = knowledgeRows1.squeeze(dim=-1)
                kpred1, temp = self.knowledgeModel.forward(datas[knowledgeRows1])
                if (self.hierarchical == "Shared"):
                    RvsNR = RvsNR.index_add(0, knowledgeRows1, temp)
                pred = pred.index_add(0, knowledgeRows1, kpred1)
            physRows1 = (torch.where(datas[:, -3] == 1, 1, 0) * torch.where(datas[:, -4] == 0, 1, 0)).nonzero()
            if physRows1.numel() > 0 and (reporting or self.trainPhysical):
                physRows1 = physRows1.squeeze(dim=-1)
                # print("p1", physRows1, datas)
                ppred1, temp = self.physicalModel.forward(datas[physRows1])
                if (self.hierarchical == "Shared"):
                    RvsNR = RvsNR.index_add(0, physRows1, temp)
                pred = pred.index_add(0, physRows1, ppred1)

        else:
            pred = torch.zeros([datas.shape[0], self.model.output_size])
            if (self.hierarchical == "Shared"):
                RvsNR = torch.zeros([datas.shape[0], 2])
            pred.requires_grad = True
            # separate data by category for each weekly question
            # then, recombine the predictions using the indices
            # consider computing these indices ahead of time and storing values for all participants?
            consumptionRows = (torch.where(datas[:, -1] == 0, 1, 0) * torch.where(datas[:, -2] == 0, 1, 0)).nonzero()
            if consumptionRows.numel() > 0 and (reporting or self.trainConsumption):
                # print("c2")
                consumptionRows = consumptionRows.squeeze(dim=-1)
                cpred, temp = self.consumptionModel.forward(datas[consumptionRows])
                if (self.hierarchical == "Shared"):
                    RvsNR = RvsNR.index_add(0, consumptionRows, temp)
                # print(cpred2.shape, consumptionRows2.shape)
                pred = pred.index_add(0, consumptionRows, cpred)
            knowledgeRows = (torch.where(datas[:, -1] == 0, 1, 0) * torch.where(datas[:, -2] == 1, 1, 0)).nonzero()
            if knowledgeRows.numel() > 0 and (reporting or self.trainKnowledge):
                # print("k2")
                knowledgeRows = knowledgeRows.squeeze(dim=-1)
                kpred, temp = self.knowledgeModel.forward(datas[knowledgeRows])
                if (self.hierarchical == "Shared"):
                    RvsNR = RvsNR.index_add(0, knowledgeRows, temp)
                # print(kpred2.shape, knowledgeRows2)
                pred = pred.index_add(0, knowledgeRows, kpred)
            physRows = (torch.where(datas[:, -1] == 1, 1, 0) * torch.where(datas[:, -2] == 0, 1, 0)).nonzero()
            if physRows.numel() > 0 and (reporting or self.trainPhysical):
                # print("p2")
                physRows = physRows.squeeze(dim=-1)
                ppred, temp = self.physicalModel.forward(datas[physRows])
                if (self.hierarchical == "Shared"):
                    RvsNR = RvsNR.index_add(0, physRows, temp)
                pred = pred.index_add(0, physRows, ppred)

        # handle work for separate hierarchical classification
        if (self.hierarchical == "Separate"):
            if not self.modelSplit:
                RvsNR, temp = self.modelRNR.forward(datas)
            # print(classpreds.shape, RvsNR.shape)
            elif not self.bd.split_weekly_questions:
                RvsNR, temp = torch.zeros([datas.shape[0] * 2, 2])
                RvsNR.requires_grad = True
                if consumptionRows2.numel() > 0 and (reporting or self.trainConsumption):
                    # print("c2")
                    consumptionRows2 = consumptionRows2.squeeze(dim=-1)
                    cRvsNR2, temp = self.consumptionModelRNR.forward(datas[consumptionRows2])
                    # these are RvsNRictions for weekly question 2
                    # add them to the tensor after all values for weekly question 1
                    consumptionRows2 += datas.shape[0]
                    # print(cRvsNR2.shape, consumptionRows2.shape)
                    RvsNR = RvsNR.index_add(0, consumptionRows2, cRvsNR2)
                if knowledgeRows2.numel() > 0 and (reporting or self.trainKnowledge):
                    # print("k2")
                    knowledgeRows2 = knowledgeRows2.squeeze(dim=-1)
                    kRvsNR2, temp = self.knowledgeModelRNR.forward(datas[knowledgeRows2])
                    # these are RvsNRictions for weekly question 2
                    # add them to the tensor after all values for weekly question 1
                    knowledgeRows2 += datas.shape[0]
                    # print(kRvsNR2.shape, knowledgeRows2)
                    RvsNR = RvsNR.index_add(0, knowledgeRows2, kRvsNR2)
                if physRows2.numel() > 0 and (reporting or self.trainPhysical):
                    # print("p2")
                    physRows2 = physRows2.squeeze(dim=-1)
                    pRvsNR2, temp = self.physicalModelRNR.forward(datas[physRows2])
                    # these are RvsNRictions for weekly question 2
                    # add them to the tensor after all values for weekly question 1
                    physRows2 += datas.shape[0]
                    RvsNR = RvsNR.index_add(0, physRows2, pRvsNR2)
                if consumptionRows1.numel() > 0 and (reporting or self.trainConsumption):
                    # print("c1")
                    consumptionRows1 = consumptionRows1.squeeze(dim=-1)
                    cRvsNR1, temp = self.consumptionModelRNR.forward(datas[consumptionRows1])
                    RvsNR = RvsNR.index_add(0, consumptionRows1, cRvsNR1)
                if knowledgeRows1.numel() > 0 and (reporting or self.trainKnowledge):
                    # print("k1")
                    knowledgeRows1 = knowledgeRows1.squeeze(dim=-1)
                    kRvsNR1, temp = self.knowledgeModelRNR.forward(datas[knowledgeRows1])
                    RvsNR = RvsNR.index_add(0, knowledgeRows1, kRvsNR1)
                if physRows1.numel() > 0 and (reporting or self.trainPhysical):
                    physRows1 = physRows1.squeeze(dim=-1)
                    # print("p1", physRows1, datas)
                    pRvsNR1, temp = self.physicalModelRNR.forward(datas[physRows1])
                    RvsNR = RvsNR.index_add(0, physRows1, pRvsNR1)
            else:
                RvsNR = torch.zeros([datas.shape[0], 2])
                RvsNR.requires_grad = True
                if consumptionRows.numel() > 0 and (reporting or self.trainConsumption):
                    # print("c2")
                    cRvsNR, temp = self.consumptionModelRNR.forward(datas[consumptionRows])
                    # print(cRvsNR2.shape, consumptionRows2.shape)
                    RvsNR = RvsNR.index_add(0, consumptionRows, cRvsNR)
                if knowledgeRows.numel() > 0 and (reporting or self.trainKnowledge):
                    # print("k2")
                    kRvsNR, temp = self.knowledgeModelRNR.forward(datas[knowledgeRows])
                    # print(kRvsNR2.shape, knowledgeRows2)
                    RvsNR = RvsNR.index_add(0, knowledgeRows, kRvsNR)
                if physRows.numel() > 0 and (reporting or self.trainPhysical):
                    # print("p2")
                    pRvsNR, temp = self.physicalModelRNR.forward(datas[physRows])
                    RvsNR = RvsNR.index_add(0, physRows, pRvsNR)
            if (not self.bd.split_weekly_questions):
                RvsNR = torch.cat((RvsNR[:, 0:RvsNR.shape[-1]], RvsNR[:, RvsNR.shape[-1]:]), dim = -1)
            if (self.model.regression):
                pred = pred + 1
                pred = torch.where((RvsNR[:, 0] > RvsNR[:, 1]).unsqueeze(-1), (RvsNR[:, 1]).unsqueeze(-1), pred)
            else:
                pred = torch.cat([RvsNR[:, 0].unsqueeze(-1), (RvsNR[:, 1]).unsqueeze(-1) * pred[:, 1:]], -1)

        if (self.modelSplit and not self.bd.split_weekly_questions):
            # reshape to match single model output
            # final shape is 2 questions per row (1 row = 1 week for this participant)
            pred = torch.cat((pred[0:datas.shape[0]], pred[datas.shape[0]:]), dim = 1)

        # print(RvsNR)
        return pred, RvsNR


    def train_epoch_val(self, opt, trainSet):
        # feed through training data one time
        loss = []
        preds = None
        labels = None
        datas = None
        RvsNRs = None
        opt.zero_grad()
        for indx in trainSet:
            # extract one participants data
            data  = self.bd.get_features(indx)
            label = self.bd.chunkedLabels[indx]
            # important to note that we are not using batching here
            # one participants data is ONE sequence
            pred, RvsNR = self.getPrediction(data)
            # we have to predict with one sequence at a time and then
            # stitch together the predictions and labels to calculate our metrics
            if (preds == None):
                preds = pred
                labels = label
                datas = data
                if (self.hierarchical):
                    RvsNRs = RvsNR
            else: 
                preds = torch.cat([preds, pred], dim = 0)
                labels = torch.cat([labels, label], dim = 0)
                datas = torch.cat([datas, data], dim = 0)
                if (self.hierarchical):
                    RvsNRs = torch.cat([RvsNRs, RvsNR], dim=0)
            
        loss1 = self.model.train_step(preds, labels, RvsNRs)
        if (loss1 != None):
            loss.append(loss1)
            loss1.backward()
            opt.step()
            loss = [l1.item() for l1 in loss]
        return loss
    
    # update feature modifications for both train and test data
    def update_all_feature_mods(self):
        with torch.no_grad():
            for set in [self.bd.train, self.bd.test]:
                for indx in set:
                    # extract one participants data
                    data = self.bd.get_features(indx)
                    pred, RvsNR = self.getPrediction(data)
                    self.bd.set_feature_response_mods(indx, pred.detach().numpy())

        
    def train(self):
        # Loop over data and train model on each batch
        # Returns matrix of loss for each participant
        stored_losses = []
        train_metrics = []
        test_metrics = []
        opts = []
        scheds = []
        modelList = []
        if (self.modelSplit):
            modelList += [self.consumptionModel, self.knowledgeModel, self.physicalModel]
            if (self.hierarchical == "Separate"):
                modelList += [self.consumptionModelRNR, self.knowledgeModelRNR, self.consumptionModelRNR]
        else:
            modelList.append(self.model)
            if (self.hierarchical == "Separate"):
                modelList.append(self.modelRNR)
        for model in modelList:
            opt, sched = model.make_optimizer()
            opts.append(opt)
            scheds.append(sched)
        epochs = self.train_kw.get("epochs", 1)
        rec_every = self.train_kw.get("rec_every", 5)
        for e in range(epochs):
            
            # enable the zeroing of state features once appropriate
            if (self.stateZeroEpochs > 0 and e == self.stateZeroEpochs):
                self.bd.zeroStateFeatures = True
            
            if e in self.knowSchedule:
                self.trainKnowledge = not self.trainKnowledge
            if e in self.consumpSchedule:
                self.trainConsumption = not self.trainConsumption
            if e in self.physSchedule:
                self.trainPhysical = not self.trainPhysical

            # record metrics every rec_every epochs
            if (e%rec_every) == 0  or (e == epochs - 1):
                # for model in [self.consumptionModel, self.knowledgeModel, self.physicalModel]:
                # for param in self.physicalModel.parameters():
                #     print(param)
                metrics, labels = self.report_scores_train()
                train_metrics.append(metrics)
                tmetrics, tlabels = self.report_scores()
                test_metrics.append(tmetrics)
                # print(len(labels))
                # print(len(metrics), len(tmetrics))
                # print(self.trainKnowledge, self.trainPhysical, self.trainConsumption)
                # print(lh)
                # print(self.model.physicalLayer.weight.data[0, 0:3], self.model.consumptionLayer.weight.data[0, 0:3], self.model.knowledgeLayer.weight.data[0, 0:3])
                # print(self.model.lstm.weight_ih_l0.data[0, 0:3])
                # print(self.physicalModel.inputLayer.weight.data[0, 0:3], self.consumptionModel.inputLayer.weight.data[0, 0:3], self.knowledgeModel.inputLayer.weight.data[0, 0:3])
                # print(f'{e}\t', f"train acc: {metrics[labels.index('Acc')]:.3%}", f"test acc: {tmetrics[labels.index('Acc')]:.3%}", f"train exerAcc: {metrics[labels.index('AccExercise')]:.3%}", f"test exerAcc: {tmetrics[labels.index('AccExercise')]:.3%}", f"train conAcc: {metrics[labels.index('AccConsumption')]:.3%}", f"test conAcc: {tmetrics[labels.index('AccConsumption')]:.3%}", f"train knowAcc: {metrics[labels.index('AccKnowledge')]:.3%}", f"test knowAcc: {tmetrics[labels.index('AccKnowledge')]:.3%}")

            lh = self.train_epoch(opts)

            # update our predictions as features when appropriate
            if (self.bd.insert_predictions):
                if (e > 0 and e % self.epochsToUpdateLabelMods == 0) or e == epochs - 1:
                    self.update_all_feature_mods()

            # record FINAL trained metrics
            if (e%rec_every) == 0 or (e == epochs - 1):
                stored_losses.append(lh)
                # metrics, labels = self.report_scores_train()
                # train_metrics.append(metrics)
                # tmetrics, tlabels = self.report_scores()
                # test_metrics.append(tmetrics)
                # print(f'{e}\t', f"train loss: {lh[0]:.4f}", f"train acc: {metrics[labels.index('Acc')]:.3%}", f"test acc: {tmetrics[labels.index('Acc')]:.3%}", f"train exerAcc: {metrics[labels.index('AccExercise')]:.3%}", f"test exerAcc: {tmetrics[labels.index('AccExercise')]:.3%}", f"train conAcc: {metrics[labels.index('AccConsumption')]:.3%}", f"test conAcc: {tmetrics[labels.index('AccConsumption')]:.3%}", f"train knowAcc: {metrics[labels.index('AccKnowledge')]:.3%}", f"test knowAcc: {tmetrics[labels.index('AccKnowledge')]:.3%}")
                if (self.only_rnr):    
                    print(f'{e}\t', f"train loss: {lh[0]:.4f}", f"train acc: {metrics[labels.index('Acc')]:.3%}", f"test acc: {tmetrics[labels.index('Acc')]:.3%}", f"train class accs: {metrics[labels.index('Acc0')]:.3%},{metrics[labels.index('Acc1')]:.3%}", f"test class accs: {tmetrics[labels.index('Acc0')]:.3%},{tmetrics[labels.index('Acc1')]:.3%}")
                elif (self.nrc):
                    # print(f'{e}\t', f"train loss: {lh[0]:.4f}", f"train acc: {metrics[labels.index('Acc')]:.3%}", f"test acc: {tmetrics[labels.index('Acc')]:.3%}", f"train class accs: {metrics[labels.index('Acc0')]:.3%},{metrics[labels.index('Acc1')]:.3%},{metrics[labels.index('Acc2')]:.3%},{metrics[labels.index('Acc3')]:.3%}", f"test class accs: {tmetrics[labels.index('Acc0')]:.3%},{tmetrics[labels.index('Acc1')]:.3%},{tmetrics[labels.index('Acc2')]:.3%},{tmetrics[labels.index('Acc3')]:.3%}")
                    print(f'{e}\t', f"train loss: {lh[0]:.4f}", f"train acc: {metrics[labels.index('Acc')]:.3%}", f"test acc: {tmetrics[labels.index('Acc')]:.3%}", f"train res accs: {metrics[labels.index('Acc0')]:.3%}, {metrics[labels.index('AccRes')]:.3%}", f"test response acc: {tmetrics[labels.index('Acc0')]:.3%}, {tmetrics[labels.index('AccRes')]:.3%}")
                elif (self.modelSplit):
                     print(f'{e}\t', f"train loss: {lh[0]:.4f}", f"train acc: {metrics[labels.index('Acc')]:.3%}", f"test acc: {tmetrics[labels.index('Acc')]:.3%}", f"train exerAcc: {metrics[labels.index('AccExercise')]:.3%}", f"test exerAcc: {tmetrics[labels.index('AccExercise')]:.3%}", f"train conAcc: {metrics[labels.index('AccConsumption')]:.3%}", f"test conAcc: {tmetrics[labels.index('AccConsumption')]:.3%}", f"train knowAcc: {metrics[labels.index('AccKnowledge')]:.3%}", f"test knowAcc: {tmetrics[labels.index('AccKnowledge')]:.3%}")
                else:
                    print(f'{e}\t', f"train loss: {lh[0]:.4f}", f"train acc: {metrics[labels.index('Acc')]:.3%}", f"test acc: {tmetrics[labels.index('Acc')]:.3%}")
            for sched in scheds:
                sched.step()
        
        return np.array(stored_losses), np.array(train_metrics), np.array(test_metrics), labels

    def calcFinalGradients(self):
        # Loop over data and train model on each batch
        # Returns matrix of loss for each participant
        grads = self.train_epoch_no_step()

        return grads

    def train_epoch_no_step(self):
        # feed through training data one time
        preds = None
        labels = None
        opt = torch.optim.SGD(self.bd.chunkedFeatures, lr=1)
        for indx in self.bd.train:
            # extract one participants data
            data  = self.bd.get_features(indx)
            data.requires_grad = True
            label = self.bd.chunkedLabels[indx]
            # important to note that we are not using batching here
            # one participants data is ONE sequence
            pred, RvsNR = self.getPrediction(data)
            # we have to predict with one sequence at a time and then
            # stitch together the predictions and labels to calculate our metrics
            if (preds == None):
                preds = pred
                labels = label
            else: 
                preds = torch.cat([preds, pred], dim = 0)
                labels = torch.cat([labels, label], dim = 0)
            
        loss1 = preds.sum()
        loss1.backward()
        grads = None
        for indx in self.bd.train:
            grad  = self.bd.get_features(indx).grad
            # we have to predict with one sequence at a time and then
            # stitch together the predictions and labels to calculate our metrics
            if (grads == None):
                grads = grad
            else: 
                grads = torch.cat([grads, grad], dim = 0)
        return grads.mean(dim=0)
    
    def train_epoch(self, opts):
        # feed through training data one time
        loss = []
        preds = None
        labels = None
        datas = None
        RvsNRs = None
        for opt in opts:
            opt.zero_grad()
        for indx in self.bd.train:
            # extract one participants data
            data  = self.bd.get_features(indx)
            label = self.bd.chunkedLabels[indx]
            # print(data.shape)
            # important to note that we are not using batching here (FOR THE LSTM)
            # one participants data is ONE sequence
            # for non LSTM models, the participants data is batched by week
            # no special trick here - the dimensions work to be non batched in LSTM (as desired)
            # and batched by week for non LSTM models (also as desired)
            pred, RvsNR = self.getPrediction(data, reporting=False)
            

            # we have to predict with one sequence at a time and then
            # stitch together the predictions and labels to calculate our metrics
            if (preds == None):
                preds = pred
                labels = label
                datas = data
                if (self.hierarchical):
                    RvsNRs = RvsNR
            else: 
                preds = torch.cat([preds, pred], dim = 0)
                labels = torch.cat([labels, label], dim = 0)
                datas = torch.cat([datas, data], dim = 0)
                if (self.hierarchical):
                    RvsNRs = torch.cat([RvsNRs, RvsNR], dim=0)
            
        loss1 = self.model.train_step(preds, labels, RvsNRs)
        if (loss1 != None):
            loss.append(loss1)
            loss1.backward()
            if (not self.trainConsumption) or (not self.trainKnowledge) or (not self.trainPhysical): 
                # print("????")     
                if hasattr(self, 'consumptionModel'):
                    self.consumptionModel.maybe_zero_weights(self.trainConsumption, self.trainKnowledge, self.trainPhysical, do="consumption")
                    self.physicalModel.maybe_zero_weights(self.trainConsumption, self.trainKnowledge, self.trainPhysical, do="physical")
                    self.knowledgeModel.maybe_zero_weights(self.trainConsumption, self.trainKnowledge, self.trainPhysical, do="knowledge")
                else:
                    self.model.maybe_zero_weights(self.trainConsumption, self.trainKnowledge, self.trainPhysical, do="All")
            # print(self.model.inputLayer.weight.grad.abs().max())
                    
            for opt in opts:
                opt.step()
            loss = [l1.item() for l1 in loss]
        else:
            loss = [-1]
        return loss
    
    def evaluate(self):
        # Evaluate the trained models predictions
        evals = []
        for indx in self.bd.test:
            # extract one participants data
            data  = self.bd.get_features(indx)
            label = self.bd.chunkedLabels[indx]
            # important to note that we are not using batching here
            # one participants data is ONE sequence
            pred, RvsNR = self.getPrediction(data)
            pred = pred.view(label.shape)
            evals.append(self.diff_matrix(label, pred))
        return evals
    
    def report_scores(self):
        with torch.no_grad():
            preds, labels, datas = None, None, None
            for indx in self.bd.test:
                # extract one participants data
                data = self.bd.get_features(indx)
                label = self.bd.chunkedLabels[indx]
                pred, RvsNR = self.getPrediction(data)
                if (preds == None):
                    datas = data
                    preds = pred
                    labels = label
                else: 
                    datas = torch.cat([datas, data], dim = 0)
                    preds = torch.cat([preds, pred], dim = 0)
                    labels = torch.cat([labels, label], dim = 0)
            scores, label = self.model.report_scores_min(labels, preds, datas)
            return scores, label
    
    def report_scores_individual_test(self):
        with torch.no_grad():
            scores = []
            for indx in self.bd.test:
                # extract one participants data
                data  = self.bd.get_features(indx)
                label = self.bd.chunkedLabels[indx]
                pred, RvsNR = self.getPrediction(data)
                score, label = self.model.report_scores_min(label, pred, data)
                if (len(score) > 0):
                    scores.append(score)
            return np.array(scores), label

    def report_scores_individual_train(self):
        with torch.no_grad():
            scores = []
            for indx in self.bd.train:
                # extract one participants data
                data  = self.bd.get_features(indx)
                label = self.bd.chunkedLabels[indx]
                pred, RvsNR = self.getPrediction(data)
                score, label = self.model.report_scores_min(label, pred, data)
                if (len(score) > 0):
                    scores.append(score)
            return np.array(scores), label
    
    
    def report_scores_train(self):
        with torch.no_grad():
            preds, labels, datas = None, None, None
            for indx in self.bd.train:
                # extract one participants data
                data  = self.bd.get_features(indx)
                label = self.bd.chunkedLabels[indx]
                pred, RvsNR = self.getPrediction(data)
                if (preds == None):
                    preds = pred
                    labels = label
                    datas = data
                else:
                    datas = torch.cat([datas, data], dim = 0)
                    preds = torch.cat([preds, pred], dim = 0)
                    labels = torch.cat([labels, label], dim = 0)
            scores, label = self.model.report_scores_min(labels, preds, datas)
            return scores, label

    def report_scores_subset(self, subset):
        with torch.no_grad():
            preds, labels, datas = None, None, None
            for indx in subset:
                # extract one participants data
                data  = self.bd.get_features(indx)
                label = self.bd.chunkedLabels[indx]
                pred, RvsNR = self.getPrediction(data)
                if (preds == None):
                    datas = data
                    preds = pred
                    labels = label
                else: 
                    datas = torch.cat([datas, data], dim = 0)
                    preds = torch.cat([preds, pred], dim = 0)
                    labels = torch.cat([labels, label], dim = 0)
            scores, label = self.model.report_scores_min(labels, preds, datas)
            return scores, label
        
            
    def diff_matrix(self, true, pred):
        # Build a matrix showing the error in predicted responses
        diff = np.zeros((2, true.shape[0]))
        diff[0] = (true[:,:4].argmax(1)-pred[:,:4].argmax(1))
        diff[1] = (true[:,-4:].argmax(1)-pred[:,-4:].argmax(1))
        return diff
        
    def totensor(self, a):
        # Convert a to a batched tensor
        a = torch.Tensor(a)
        a = a.view(a.shape[0], 1, a.shape[1])
        return a
    
    def forceBatch(self, a):
        # Convert a to a batched tensor
        a = a.view(1, 1, a.shape[0])
        return a
        
    def _get_model(self):
        mod = importlib.import_module(f"models.{self.model_name}")
        return getattr(mod, self.model_name)
