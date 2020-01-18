import numpy as np
import scipy.sparse as sp
import torch
import pickle

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    # the labels vary when run the code, we need it run only one time and then fixed.
    # adj, features, labels, idx_train, idx_val, idx_test = utils.load_data()
    # dump_dict = {'adj': adj, 'features':features, 'labels':labels, 'idx_train':idx_train, 'idx_val':idx_val, 'idx_test': idx_test}
    # pickle.dump(dump_dict, open("cora_data.pkl", "wb"))
    load_dict = pickle.load(open(dataset+"_data.pkl", "rb"))
    adj = load_dict['adj']
    features = load_dict['features']
    labels = load_dict['labels']
    idx_train = load_dict['idx_train']
    idx_val = load_dict['idx_val']
    idx_test = load_dict['idx_test']
    # idx_train = range(120)
    # idx_val = range(120, 620)
    # idx_test = range(1000, 2000)

    # adj = torch.FloatTensor(np.array(adj.todense()))
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train).reshape(-1)
    idx_val = torch.LongTensor(idx_val).reshape(-1)
    idx_test = torch.LongTensor(idx_test).reshape(-1)

    features  = torch.tensor(features)
    adj = torch.tensor(adj)
    labels = torch.tensor(labels)

    print('adj.shape', adj.shape)
    print('features.shape', features.shape)
    label_set = set()
    for i in range(labels.shape[0]):
            #print(labels[i])
            label_set.add(labels[i].data.tolist())

    print('labels len', len(label_set))
    print('len idx_train', len(idx_train))
    print('len idx_val', len(idx_val))
    print('len idx_test', len(idx_test))

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels) #pythorch's max , 1 indicate indexes.
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

