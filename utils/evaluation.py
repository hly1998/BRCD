from torch.autograd import Variable
import numpy as np
import torch
from tqdm import tqdm

def ours_compress(train, test, encode_discrete, device):
    retrievalB = list([])
    retrievalL = list([])
    for batch_step, (data, _, target) in enumerate(train):
        var_data = Variable(data.to(device))
        code = encode_discrete(var_data)
        retrievalB.extend(code.cpu().data.numpy())
        retrievalL.extend(target)

    queryB = list([])
    queryL = list([])
    for batch_step, (data, _, target) in enumerate(test):
        var_data = Variable(data.to(device))
        code = encode_discrete(var_data)
        queryB.extend(code.cpu().data.numpy())
        queryL.extend(target)

    retrievalB = np.array(retrievalB)
    retrievalL = np.stack(retrievalL)

    queryB = np.array(queryB)
    queryL = np.stack(queryL)
    return retrievalB, retrievalL, queryB, queryL

def ours_distill_compress(train, test, s_encode_discrete, t_encode_discrete, device):
    retrievalB = list([])
    retrievalL = list([])
    for batch_step, (data, _, target) in enumerate(train):
        var_data = Variable(data.to(device))
        code = t_encode_discrete(var_data)
        retrievalB.extend(code.cpu().data.numpy())
        retrievalL.extend(target)

    queryB = list([])
    queryL = list([])
    for batch_step, (data, _, target) in enumerate(test):
        var_data = Variable(data.to(device))
        code = s_encode_discrete(var_data)
        queryB.extend(code.cpu().data.numpy())
        queryL.extend(target)

    retrievalB = np.array(retrievalB)
    retrievalL = np.stack(retrievalL)

    queryB = np.array(queryB)
    queryL = np.stack(queryL)
    return retrievalB, retrievalL, queryB, queryL

def compress(train, test, encode_discrete, device):
    retrievalB = list([])
    retrievalL = list([])
    for batch_step, (data, _, target) in enumerate(train):
        var_data = Variable(data.to(device))
        code = encode_discrete(var_data)
        retrievalB.extend(code.cpu().data.numpy())
        retrievalL.extend(target)

    queryB = list([])
    queryL = list([])
    for batch_step, (data, _, target) in enumerate(test):
        var_data = Variable(data.to(device))
        code = encode_discrete(var_data)
        queryB.extend(code.cpu().data.numpy())
        queryL.extend(target)

    retrievalB = np.array(retrievalB)
    retrievalL = np.stack(retrievalL)

    queryB = np.array(queryB)
    queryL = np.stack(queryL)
    return retrievalB, retrievalL, queryB, queryL


def distill_compress(train, test, s_encode_discrete, t_encode_discrete, device):
    retrievalB = list([])
    retrievalL = list([])
    for batch_step, (data, _, target) in enumerate(train):
        var_data = Variable(data.to(device))
        code = t_encode_discrete(var_data)
        retrievalB.extend(code.cpu().data.numpy())
        retrievalL.extend(target)

    queryB = list([])
    queryL = list([])
    for batch_step, (data, _, target) in enumerate(test):
        var_data = Variable(data.to(device))
        code = s_encode_discrete(var_data)
        queryB.extend(code.cpu().data.numpy())
        queryL.extend(target)

    retrievalB = np.array(retrievalB)
    retrievalL = np.stack(retrievalL)

    queryB = np.array(queryB)
    queryL = np.stack(queryL)
    return retrievalB, retrievalL, queryB, queryL

def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    q = B2.shape[1] # max inner product value
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def calculate_euclidean(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: euclidean distance [r]
    """
    # print(type(B1), B1.shape)
    return np.sum((B1 - B2)**2, axis=1)


def calculate_top_map(qB, rB, queryL, retrievalL, topk):
    """
    :param qB: {-1,+1}^{mxq} query bits
    :param rB: {-1,+1}^{nxq} retrieval bits
    :param queryL: {0,1}^{mxl} query label
    :param retrievalL: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind] # reorder gnd

        tgnd = gnd[0:topk]
        tsum = int(np.sum(tgnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def calculate_top_map_in_euclidean_space(qB, rB, queryL, retrievalL, topk):
    """
    :param qB: {-1,+1}^{mxq} query bits
    :param rB: {-1,+1}^{nxq} retrieval bits
    :param queryL: {0,1}^{mxl} query label
    :param retrievalL: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        # hamm = calculate_hamming(qB[iter, :], rB)
        euc = calculate_euclidean(qB[iter, :], rB)
        ind = np.argsort(euc)
        gnd = gnd[ind] # reorder gnd

        tgnd = gnd[0:topk]
        tsum = int(np.sum(tgnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


# 7.3 为了回复review，增加了topk的计算
def retrieve_topk(query_b, doc_b, topK, batch_size=100):
    n_bits = doc_b.size(1)
    n_train = doc_b.size(0)
    n_test = query_b.size(0)

    topScores = torch.cuda.ByteTensor(n_test,
                                      topK + batch_size).fill_(n_bits + 1)
    topIndices = torch.cuda.LongTensor(n_test, topK + batch_size).zero_()

    testBinmat = query_b.unsqueeze(2)

    for batchIdx in tqdm(range(0, n_train, batch_size), ncols=0, leave=False):
        s_idx = batchIdx
        e_idx = min(batchIdx + batch_size, n_train)
        numCandidates = e_idx - s_idx

        trainBinmat = doc_b[s_idx:e_idx]
        trainBinmat.unsqueeze_(0)
        trainBinmat = trainBinmat.permute(0, 2, 1)
        trainBinmat = trainBinmat.expand(testBinmat.size(0), n_bits,
                                         trainBinmat.size(2))

        testBinmatExpand = testBinmat.expand_as(trainBinmat)

        scores = (trainBinmat ^ testBinmatExpand).sum(dim=1)
        indices = torch.arange(start=s_idx, end=e_idx, step=1).type(
            torch.cuda.LongTensor).unsqueeze(0).expand(n_test, numCandidates)

        topScores[:, -numCandidates:] = scores
        topIndices[:, -numCandidates:] = indices

        topScores, newIndices = topScores.sort(dim=1)
        topIndices = torch.gather(topIndices, 1, newIndices)

    return topIndices


def compute_precision_at_k(retrieved_indices, query_labels, doc_labels, topK,
                           is_single_label=True):
    n_test = query_labels.size(0)

    Indices = retrieved_indices[:, :topK]
    if is_single_label:
        print(query_labels)
        test_labels = query_labels.unsqueeze(1).expand(n_test, topK)
        topTrainLabels = [
            torch.index_select(doc_labels, 0, Indices[idx]).unsqueeze_(0)
            for idx in range(0, n_test)
        ]
        topTrainLabels = torch.cat(topTrainLabels, dim=0)
        relevances = (test_labels == topTrainLabels).type(
            torch.cuda.ShortTensor)
    else:
        topTrainLabels = [
            torch.index_select(doc_labels, 0, Indices[idx]).unsqueeze_(0)
            for idx in range(0, n_test)
        ]
        topTrainLabels = torch.cat(topTrainLabels,
                                   dim=0).type(torch.cuda.ShortTensor)
        test_labels = query_labels.unsqueeze(1).expand(
            n_test, topK, topTrainLabels.size(-1)).type(torch.cuda.ShortTensor)
        relevances = (topTrainLabels & test_labels).sum(dim=2)
        relevances = (relevances > 0).type(torch.cuda.ShortTensor)

    true_positive = relevances.sum(dim=1).type(torch.cuda.FloatTensor)
    true_positive = true_positive.div_(topK)
    prec_at_k = torch.mean(true_positive)
    return prec_at_k
