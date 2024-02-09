import argparse
from signal import valid_signals
from dataset_loader import DataLoader
from utils import random_splits, random_splits_miss, random_splits_citation,fixed_splits,batch_generator
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from tqdm import tqdm
import random
import seaborn as sns
import numpy as np
import time
import math
from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from utils import  DataSplit
from torch_sparse import coalesce
from torch_geometric.utils.undirected import is_undirected, to_undirected

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, seed=520):
    labeled_nodes = np.where(label!=-1)[0]
    
    n = labeled_nodes.shape[0]
    train_num = int(n*train_prop)
    valid_num = int(n*valid_prop)
    np.random.seed(seed)
    perm = np.random.permutation(n)
    train_indices = labeled_nodes[perm[:train_num]]
    val_indices = labeled_nodes[perm[train_num:train_num+valid_num]]
    test_indices = labeled_nodes[perm[train_num+valid_num:]]
    return train_indices, val_indices, test_indices
    

if __name__ == '__main__':
    
    datasetname = ['Cora',      'Citeseer',     'Pubmed',       'Chameleon',        'Squirrel', 
                   'Actor',     'Texas',        'Cornell',      'Wisconsin',        'Photo', 
                   'Computers', 'ogbn-arxiv',   'arxiv-year',   'ogbn-products',    'penn94',
                   'genius',    'wiki',          'pokec',        'twitch-gamer',     'snap-patents',
                   'twitch-de', 'tolokers',   'minesweeper',    'roman_empire', 'amazon_ratings',   
                   'questions']

    #['arxiv-year', 'penn94', 'genius', 'pokec', 'snap-patents', 'twitch-de'] 
    
    datasetsplit_default = ['arxiv-year', 'penn94',         'genius',       'pokec',            'snap-patents', 'twitch-de',
                            'tolokers',   'minesweeper',    'roman_empire', 'amazon_ratings',   'questions'] 
    datasetsplit_10 = ['Cora',      'Citeseer',     'Pubmed',       'Chameleon',        'Squirrel', 
                       'Actor',     'Texas',        'Cornell',      'Wisconsin',        'Photo', 
                       'Computers']
    datasetsplit_5 =  ['wiki',  'twitch-gamer',]
    
    datasetsplit_miss = ['wiki']
    
    datasetsplit_ogb = ['ogbn-arxiv', 'ogbn-products']
    
    SEEDS=[1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363]
    
    for i in datasetname:
        dataset = DataLoader(i)
        data = dataset[0]
        data.edge_index = to_undirected(data.edge_index)
        data.y = data.y.flatten()
        percls_trn = int(round(0.6*len(data.y)/dataset.num_classes))
        val_lb = int(round(0.2*len(data.y)))
        saver = {}
        
        if i in datasetsplit_default:
            runs = 0
        elif i in datasetsplit_5:
            runs = 5
        elif i in datasetsplit_10:
            runs = 10
        elif i in datasetsplit_ogb:
            runs = 1
        train_mask = torch.zeros(data.num_nodes, runs, dtype = torch.bool)
        val_mask   = torch.zeros(data.num_nodes, runs, dtype = torch.bool)
        test_mask  = torch.zeros(data.num_nodes, runs, dtype = torch.bool)
        
        for j in range(runs):
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.val_mask   = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.test_mask  = torch.zeros(data.num_nodes, dtype=torch.bool)
            if i in datasetsplit_5:
                train_idx, val_idx, test_idx = rand_train_test_idx(label=data.y, seed=SEEDS[j])
                data.train_mask[train_idx] = 1
                data.val_mask  [val_idx]   = 1
                data.test_mask [test_idx]  = 1
            elif i in datasetsplit_ogb:
                split_idx = dataset.get_idx_split() 
                
                data.train_mask[split_idx['train']] = 1
                data.val_mask  [split_idx['valid']] = 1
                data.test_mask [split_idx['test']]  = 1
            else:
                data = random_splits     (data, dataset.num_classes, percls_trn, val_lb, SEEDS[j])
            train_mask[:,j], val_mask[:, j], test_mask[:, j] = data.train_mask, data.val_mask, data.test_mask
        
        if i in datasetsplit_default:
            saver['edge_index'] = data.edge_index
            saver['train_mask'] = data.train_mask
            saver['val_mask'] = data.val_mask
            saver['test_mask'] = data.test_mask
        else:
            saver['edge_index'] = data.edge_index
            saver['train_mask'] = train_mask
            saver['val_mask'] = val_mask
            saver['test_mask'] = test_mask
        
        print(data)
        for j in range(runs):
            print(train_mask[:,j].sum(), val_mask[:,j].sum(), test_mask[:,j].sum())
        
        torch.save(saver, './data_saved/'+i+'.pt')
    
    
        
        
                
        