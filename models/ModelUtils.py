import torch
# NOT DIFFERENTIABLE
def NDCG(pred, y):
    # calculate normalized discounted cumulative gain
    # calculate ranks (higher % prediction is lower rank)
    ranks = -1*pred
    ranks = ranks.argsort()
    ranks = ranks.argsort()
    ranks += 1 # best rank should be 1 not 0
    dcg = (torch.exp2(y) - 1) / torch.log2(1 + ranks)
    # ideal ranks (take y, set all zero terms to 3 to avoid div by 0)
    # actual rank of zero terms in y is irrelevant (2^0 - 1 = 0)
    iranks = y.clone()
    iranks[y == 0] = 3
    # calculate dcg of ideal predictions (ideal dcg)
    idcg = (torch.exp2(y) - 1) / torch.log2(1 + iranks)
    # divide dcg by idcg to get ndcg
    loss = dcg.sum() / idcg.sum()
    return loss

def MRR(pred, y):
    # calculate ranks (higher % prediction is lower rank)
    ranks = -1*pred
    ranks = ranks.argsort()
    ranks = ranks.argsort()
    ranks += 1 # best rank should be 1 not 0
    # 1/rank for correct prediction entries, 0 for others (as y is 0 then)
    mrr = (y / ranks).sum(axis=1).mean()
    return mrr

def PairwiseLogLoss(pred, y):
    None

def ApproxNDCG(pred, y):
    # differentiable approximate form of NDCG
    None