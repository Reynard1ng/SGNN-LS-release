import torch
import math
import numpy as np
import torch.nn.functional as F

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def take_rest(x, y):
    x.sort()
    y.sort()
    res = []
    j, jmax = 0, len(y)
    for i in range(0, len(x)):
        flag = False
        while j < jmax and y[j] <= x[i]:
            if y[j] == x[i]:
                flag = True
            j += 1
        if not flag:
            res.append(x[i])
    return res

def random_splits(data, num_classes, percls_trn, val_lb, seed=42):
    index=[i for i in range(0,data.y.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    #print(train_idx)
    if len(index) < 10000:
        rest_index = [i for i in index if i not in train_idx]
        val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
        test_idx=[i for i in rest_index if i not in val_idx]
    else:
        rest_index = take_rest(index, train_idx)
        val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
        test_idx = take_rest(rest_index, val_idx)
    print(data.y.shape[0], len(train_idx), len(val_idx), len(test_idx))

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)
    return data

def random_splits_miss(data, num_classes, train_prop=.6, valid_prop=.2, seed=42):
    index = np.where(data.y != -1)[0]
    percls_trn  = int(round(0.6*len(index)/num_classes))
    val_lb      = int(round(0.2*len(index)))
    
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    if len(index) < 10000:
        rest_index = [i for i in index if i not in train_idx]
        val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
        test_idx=[i for i in rest_index if i not in val_idx]
    else:
        rest_index = take_rest(index, train_idx)
        val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
        test_idx = take_rest(rest_index, val_idx)

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)
    return data

def fixed_splits(data, num_classes, percls_trn, val_lb, name):
    seed=42
    if name in ["Chameleon","Squirrel", "Actor"]:
        seed = 1941488137
    index=[i for i in range(0,data.y.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)

    return data

def random_splits_citation(data, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)

    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data

def DataSplit(args, data_smp, data, denseAdj, batch_mask, device):
    batch_size = len(batch_mask)
    data_smp.train_mask = data.train_mask[batch_mask]
    data_smp.val_mask   = data.val_mask[batch_mask]
    data_smp.test_mask  = data.test_mask[batch_mask]
    edge_index_smp = [[], []]
    for i in range(batch_size):
        edge_index_smp[0] += [i] * batch_size
    edge_index_smp[1] = [i for i in range(batch_size)] * batch_size
    edge_index_smp = torch.tensor(edge_index_smp).to(device)
    data_smp.edge_index = edge_index_smp
    data_smp.x = data.x[batch_mask]
    data_smp.y = data.y[batch_mask]
    data_smp.denseAdj = (denseAdj[batch_mask])[:, batch_mask].flatten()

    if args.sparse_threshold > 0:  
        mask_smp = data_smp.denseAdj.abs() > args.sparse_threshold
        data_smp.edge_index = data_smp.edge_index[:, mask_smp]
        data_smp.denseAdj   = data_smp.denseAdj[mask_smp]

    return data_smp

def batch_generator(tsplit, train_mask, N, batchnum):
    res = []
    if not tsplit:
        data = torch.arange(0, N)
    else:
        data = torch.arange(0, N)[train_mask]
        N = len(data)
    
    perm = torch.randperm(N)
    batch_size = N // batchnum
    begins = 0
    ends = begins + batch_size
    while(1):
        if(ends + batch_size / 2 > N):
            res.append(data[perm[begins:]])
            return res
        else:
            res.append(data[perm[begins:ends]])
            begins = ends
            ends = begins + batch_size

def batch_generator_idx(train_set, batch_size):
    N = train_set.size()[0]
    res = []
    perm = torch.randperm(N)
    begins = 0
    ends = begins + batch_size
    while(1):
        if(ends + batch_size / 2 > N):
            res.append(train_set[perm[begins:]])
            return res
        else:
            res.append(train_set[perm[begins:ends]])
            begins = ends
            ends = begins + batch_size
            
def mag_split(seed, n, n_train, n_val = 0):
    np.random.seed(seed)
    rnd = np.random.permutation(n)

    train_idx = np.sort(rnd[:n_train])
    val_idx = np.sort(rnd[n_train:n_train + n_val])

    train_val_idx = np.concatenate((train_idx, val_idx))
    test_idx = np.sort(np.setdiff1d(np.arange(n), train_val_idx))

    return train_idx, val_idx, test_idx