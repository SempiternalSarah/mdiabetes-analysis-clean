# script to run experiments making a predictive model of participant weekly behavior
from experiment import Experiment
import torch
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser()

def toBool(x):
    return (str(x).lower() in ['true', '1', 't'])

# expanded states
parser.add_argument('-model', type=str, default="BasicNN")
parser.add_argument('-numWeeks', type=int, default=3)
parser.add_argument('-estate', type=toBool, default=True)
parser.add_argument('-includeState', type=toBool, default=True)
parser.add_argument('-fullQ', type=toBool, default=True)
parser.add_argument('-insertPreds', type=toBool, default=False)
parser.add_argument('-splitQ', type=toBool, default=True)
parser.add_argument('-splitM', type=toBool, default=True)
parser.add_argument('-catHist', type=toBool, default=False)
parser.add_argument('-knowEpochs', type=int, default=1000)
parser.add_argument('-physEpochs', type=int, default=1000)
parser.add_argument('-conEpochs', type=int, default=1000)
parser.add_argument('-smooth', type=float, default=0)
parser.add_argument('-noise', type=float, default=0.07)
parser.add_argument('-learning_rate', type=float, default=0.001)
parser.add_argument('-respond_perc', type=float, default=0.5)
parser.add_argument('-hierarchical', type=str, default="Shared")
parser.add_argument('-regression', type=toBool, default=False)
parser.add_argument('-nrclass', type=toBool, default=False)
parser.add_argument('-sepHierLoss', type=toBool, default=False)
parser.add_argument('-seeds', type=int, default=5)
parser.add_argument('-only_rnr', type=toBool, default=False)
parser.add_argument('-transformer', type=toBool, default=False)
parser.add_argument('-save', type=toBool, default=True)
parser.add_argument('-cluster_by', type=str, default=None)
parser.add_argument('-num_clusters', type=int, default=3)
parser.add_argument('-cluster_method', type=str, default="Kmeans")



args = parser.parse_args()

if (args.hierarchical != "Shared" and args.hierarchical != "Separate"):
    args.hierarchical = None

model = args.model

respond_perc = args.respond_perc

conSched = [args.conEpochs]
knowSched = [args.knowEpochs]
physSched = [args.physEpochs]

learning_rate = args.learning_rate

smooth, noise = args.smooth, args.noise

splitQ, splitM = args.splitQ, args.splitM

catHist = args.catHist

numWeeks = args.numWeeks

insertPreds = args.insertPreds

if (args.regression):
    loss_fn = "MSELoss"
else:
    loss_fn = "CrossEntropyLoss"

include_state, estate, fullq = args.includeState, args.estate, args.fullQ

epochs=max([args.conEpochs, args.knowEpochs, args.physEpochs])

if "LSTM" in model:
    stateweek = 1
else:
    stateweek = 500

if epochs > 900:
    hiddenSize = 50
    lrmult = 1.0
else:
    if (args.transformer):
        hiddenSize = 25
    else:
        hiddenSize = 50
    lrmult = 0.9

for seed in range(args.seeds):
    np.random.seed(seed)
    torch.manual_seed(seed)
    e = Experiment(
        modelSplit = splitM,
        numValFolds = 5,
        epochsToUpdateLabelMods = 10,
        knowSchedule = knowSched,
        consumpSchedule = conSched,
        physSchedule = physSched,
        hierarchical=args.hierarchical,
        nrc=args.nrclass,
        only_rnr=args.only_rnr,
        data_kw={"minw": 2,
                "maxw": 31,
                "include_state": include_state,
                "include_pid": False,
                "expanded_states": estate,
                "top_respond_perc": respond_perc,
                "full_questionnaire": fullq,
                "num_weeks_history": numWeeks,
                "insert_predictions": insertPreds,
                "split_model_features": splitM,
                "split_weekly_questions": splitQ,
                "category_specific_history": catHist,
                "max_state_week": stateweek,
                "regression": args.regression,
                "no_response_class": args.nrclass,
                "only_rnr": args.only_rnr,
                "cluster_by": args.cluster_by,
                "num_clusters": args.num_clusters,
                "cluster_method": args.cluster_method
                },
        model=model,
        model_kw={
            "lossfn": loss_fn,
            # "lossfn": "NDCG",
            # "lossfn": "CrossEntropyLoss",
            "hidden_size": hiddenSize, 
            "lr_step_mult": lrmult, 
            "lr_step_epochs": 60,
            "opt_kw": {
                "lr": learning_rate
            },
            "labelSmoothPerc": smooth,
            "gaussianNoiseStd": noise,
            "splitModel": splitM,
            "splitWeeklyQuestions": splitQ,
            "hierarchical": args.hierarchical,
            "regression": args.regression,
            "no_response_class": args.nrclass,
            "separateHierLoss": args.sepHierLoss,
            "only_rnr": args.only_rnr,
            "transformer": args.transformer
        },
        train_kw={
            "epochs": epochs,
            "n_subj": 500,
            "rec_every": 5,
        })
    # torch.autograd.set_detect_anomaly(True)
    report = e.run()

    if not args.save:
        continue
    individual_test_scores, labels = e.report_scores_individual_test()
    individual_train_scores, labels = e.report_scores_individual_train()

    if args.transformer:
        finalDir = f"{model}Attn"
    else:
        finalDir = f"{model}"
    if args.cluster_by != None:
        finalDir = f"{finalDir}/cluster{args.cluster_by}"
    if epochs > 900:
        dire = f"./experiment_output_long/{finalDir}/"
    else:
        dire = f"./experiment_output/{finalDir}/"

    if (not os.path.exists(dire)):
        os.makedirs(dire)


    fileprefix = f"C{args.num_clusters}{args.cluster_method}R{int(args.regression)}NR{args.nrclass}H{args.hierarchical}{int(args.sepHierLoss)}W{numWeeks}LR{learning_rate}Resp{respond_perc}States{int(include_state)}Expanded{int(estate)}Full{int(fullq)}CHist{int(catHist)}Pred{int(insertPreds)}Smooth{smooth}Noise{noise}Split{int(splitQ)}{int(splitM)}"
    np.savetxt(f"{dire}TRAINMETRICS-{fileprefix}S{seed}.csv", report["train_metrics"], delimiter = ',', header = ','.join(report['metric_labels']))
    np.savetxt(f"{dire}TESTMETRICS-{fileprefix}S{seed}.csv", report["test_metrics"], delimiter = ',', header = ','.join(report['metric_labels']))
    np.savetxt(f"{dire}IDVDTESTMETRICS-{fileprefix}S{seed}.csv", individual_test_scores, delimiter = ',', header = ','.join(report['metric_labels']))
    np.savetxt(f"{dire}IDVDTRAINMETRICS-{fileprefix}S{seed}.csv", individual_train_scores, delimiter = ',', header = ','.join(report['metric_labels']))
    np.savetxt(f"{dire}TRAINLOSSES-{fileprefix}S{seed}.csv", report["loss"], delimiter = ',')

    preds1, preds2, preds3 = e.get_class_predictions(False)


    np.savetxt(f"{dire}TRAINPREDS1-{fileprefix}S{seed}.csv", preds1, delimiter = ',')
    np.savetxt(f"{dire}TRAINPREDS2-{fileprefix}S{seed}.csv", preds2, delimiter = ',')
    np.savetxt(f"{dire}TRAINPREDS3-{fileprefix}S{seed}.csv", preds3, delimiter = ',')


    preds1, preds2, preds3 = e.get_class_predictions(True)

    np.savetxt(f"{dire}TESTPREDS1-{fileprefix}S{seed}.csv", preds1, delimiter = ',')
    np.savetxt(f"{dire}TESTPREDS2-{fileprefix}S{seed}.csv", preds2, delimiter = ',')
    np.savetxt(f"{dire}TESTPREDS3-{fileprefix}S{seed}.csv", preds3, delimiter = ',')


    torch.save(e.model, f"{dire}TRAINEDMODEL-{fileprefix}S{seed}.pt")

    writer = open(f"{dire}FINALTRAINMETRICS-{fileprefix}.csv", "a")
    writer.write(",".join([str(loss) for loss in report["train_metrics"][-1, :]]))
    writer.write("\n")
    writer.close()

    writer = open(f"{dire}FINALTESTMETRICS-{fileprefix}.csv", "a")
    writer.write(",".join([str(loss) for loss in report["test_metrics"][-1, :]]))
    writer.write("\n")
    writer.close()



