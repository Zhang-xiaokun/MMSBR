import numpy as np
from scipy.sparse import csr_matrix
from operator import itemgetter

import random
import numpy as np
import torch

def init_seed(seed, reproducibility):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def data_masks(all_sessions, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j]) #统计session中不同item，去重，并按照item_id排序
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i]-1)
            data.append(1)
    # indptr:session长度累加和; indices:item_id 减1, 由每个session内item组成; data:item在session内的权重，全部为1.
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))
    # 10000 * 6558 #sessions * #items H in paper 稀疏矩阵存储
    return matrix

def data_easy_masks(data_l, n_row, n_col):
    data, indices, indptr  = data_l[0], data_l[1], data_l[2]

    matrix = csr_matrix((data, indices, indptr), shape=(n_row, n_col))
    # 10000 * 6558 #sessions * #items H in paper 稀疏矩阵存储
    return matrix

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

class Data():
    def __init__(self, data, shuffle=False, n_node=None, n_price=None):
        # data formulation: 0:id_seq, 1:price_seq, 2:rating_seq, 3:cate_seq, 4:price_list, 5:rating_list, 6:cate_list, 7:labs
        self.raw = np.asarray(data[0])  # sessions, item_seq
        self.price_raw = np.asarray(data[1])  # price_seq
        self.cate_raw = np.asarray(data[2])
        self.price_list = np.asarray(data[3])
        self.cate_list = np.asarray(data[4])

        self.n_node = n_node
        self.n_price = n_price
        self.targets = np.asarray(data[5])
        self.length = len(self.raw)
        self.shuffle = shuffle

    def get_overlap(self, sessions):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)
            for j in range(i+1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap))/float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0]*len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0/degree)
        return matrix, degree

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            # 打乱session item_seq&price_seq的顺序
            self.raw = self.raw[shuffled_arg]
            self.price_raw = self.price_raw[shuffled_arg]
            self.cate_raw = self.cate_raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    def get_slice(self, index):
        items, num_node, price_seqs, cate_seqs = [], [], [], []
        inp = self.raw[index]
        inp_price = self.price_raw[index]
        inp_category = self.cate_raw[index]

        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        session_len = []
        reversed_sess_item = []
        mask = []
        for session, price, cate in zip(inp,inp_price, inp_category):
            nonzero_elems = np.nonzero(session)[0]
            session_len.append([len(nonzero_elems)])
            if max_n_node - len(nonzero_elems) == 0:
                items.append(session)
                price_seqs.append(price)
                cate_seqs.append(cate)
                mask.append([1] * len(nonzero_elems))
                reversed_sess_item.append(list(reversed(session)))
            else:
                items.append(session + (max_n_node - len(nonzero_elems)) * [0])
                price_seqs.append(price + (max_n_node - len(nonzero_elems)) * [0])
                cate_seqs.append(cate + (max_n_node - len(nonzero_elems)) * [0])
                mask.append([1] * len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
                reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])


        return self.targets[index]-1, session_len,items, reversed_sess_item, mask, price_seqs, cate_seqs


