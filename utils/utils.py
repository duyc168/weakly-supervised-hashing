
import numpy as np
import torch
from tqdm import tqdm


def GenerateCode(model, data_loader, num_data, bit, batch_sz, n_class):
    B = np.zeros([num_data, bit], dtype=np.float32)
    all_labels = np.zeros([num_data, n_class], dtype=np.float32)
    for i, data in enumerate(tqdm(data_loader)):
        imgs, label, _, _, indexs = data
        imgs = imgs.cuda()
        B_code, _ = model(imgs)
        if (i+1)*batch_sz < num_data:
            B[i*batch_sz:(i+1)*batch_sz, :] = torch.sign(B_code-0.5).cpu().data.numpy()
            all_labels[i*batch_sz:(i+1)*batch_sz, :] = label.data.numpy()
        else:
            B[i*batch_sz:num_data, :] = torch.sign(B_code-0.5).cpu().data.numpy()
            all_labels[i*batch_sz:num_data, :] = label.data.numpy()          
    return B, all_labels



def CalcHammingDist(B1, B2):
    K = B2.shape[1]
    distH = 0.5 * (K - np.dot(B1, B2.transpose()))
    return distH

def CalcMap(qB, rB, queryL, retrievalL):
    num_query = queryL.shape[0]
    print("query num:{}".format(num_query))
    map = 0

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        tsum = int(tsum)
        if tsum == 0:
            continue
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map


def CalcMap_at_k(qB, rB, queryL, retrievalL, k):
    num_query = queryL.shape[0]
    # print("query num:{}".format(num_query))
    map = 0

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        gnd = gnd[0:k]
        tsum = np.sum(gnd)
        tsum = int(tsum)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map

