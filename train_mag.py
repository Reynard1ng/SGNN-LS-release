import argparse
from signal import valid_signals
from dataset_loader import DataLoader
from utils import random_splits, random_splits_citation,fixed_splits, batch_generator, mag_split, batch_generator_idx
from PPR_calc import topk_ppr_matrix
from PPRGo_sup import PPRGo_mag, MLP_mag
from LSGPRL_mag import LSGPRL_mag
from LSAPPNPL_mag import LSAPPNPL_mag
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
from utils import cheby, FetchDenseAdj, FetchPolyCoefficients, DataSplit
from torch_sparse import coalesce
from torch_geometric.utils.undirected import is_undirected, to_undirected
from torch import pca_lowrank, matmul
from sklearn.metrics import roc_auc_score
import os

import torch_sparse
import scipy.sparse as sp

def RunExp(args, dataset, data, Net):

    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    host = torch.device('cpu')

    def train(model, optimizer, data, dprate):
        
        if args.net in ['GCN_RW_mini_G', 'GraphSAGE_RW_mini_G', 'GCNII_RW_mini_G']:
            batchStack = batch_generator(args.tsplit, data.train_mask, data.num_nodes, args.batch)
            for i in batchStack:
                model.train()
                optimizer.zero_grad()
                out = model(data, i.to(device))[data.train_mask[i]]
                loss = F.nll_loss(out, data.y[i][data.train_mask[i]])
                print(loss, flush=True)
                loss.backward()
                optimizer.step()
                del out
                torch.cuda.empty_cache()
        elif args.net in []:
            None
        else:
            model.train()
            optimizer.zero_grad()
            out = model(data['attr'], data['train_idx'])
            loss = F.nll_loss(out, data['y'][data['train_idx']])
            print(loss, flush=True)
            loss.backward()
            optimizer.step()
            del out
            torch.cuda.empty_cache()

    def valid(model, data):
        with torch.no_grad():
            model.eval()
            batchStack = batch_generator_idx(data['valid_idx'], args.batchsize)
            sum_test, loss_test = 0, 0
            for i in batchStack:
                out = model(data['attr'], i.to(device))
                loss = F.nll_loss(out, data['y'][i].long())
                sum_test += out.max(1)[1].eq(data['y'][i].long()).sum().item()
                loss_test += loss.item() * i.size()[0]
                torch.cuda.empty_cache()
            print("valid acc / valid loss:", sum_test / data['valid_idx'].size()[0], loss_test / data['valid_idx'].size()[0], flush=True)
            return sum_test / data['valid_idx'].size()[0], loss_test / data['valid_idx'].size()[0]
    
    def test(model, data):
        with torch.no_grad():
            model.eval()
            batchStack = batch_generator_idx(data['test_idx'], args.batchsize)
            sum_test, loss_test = 0, 0
            for i in batchStack:
                out = model(data['attr'], i.to(device))
                loss = F.nll_loss(out, data['y'][i].long())
                sum_test += out.max(1)[1].eq(data['y'][i].long()).sum().item()
                loss_test += loss.item() * i.size()[0]
                torch.cuda.empty_cache()
            print("test acc / test loss:", sum_test / data['test_idx'].size()[0], loss_test / data['test_idx'].size()[0], flush=True)
            return sum_test / data['test_idx'].size()[0], loss_test / data['test_idx'].size()[0]
            
    
    if args.net in ['LSGPRL_mag', 'LSAPPNPL_mag', 'MLP_mag']:
        tmp_net = Net(data, args)
    elif args.net in ['PPRGo_mag']:
        tmp_net = Net(args, data['num_nodes'], data['attr'].size(1), 8, args.hidden, args.nlayer, args.dropout, data['topk_idx'])
    print('MODEL CONSTRUCTED.')
    
    model = tmp_net.to(device)
    
    data['attr']        = data['attr'].to(device)
    data['train_idx']   = data['train_idx'].to(device)
    data['valid_idx']   = data['valid_idx'].to(device)
    data['test_idx']   = data['test_idx'].to(device)
    data['y']           = data['y'].to(device)
        
    if args.net in ['LSGPRL_mag']:
        optimizer = torch.optim.Adam([
            {'params': model.lins.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.att,               'weight_decay': args.prop_wd, 'lr': args.prop_lr}
        ])
    elif args.net in ['LSAPPNPL_mag']:
        optimizer = torch.optim.Adam([
            {'params': model.lins.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr}
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    time_run=[]
    for epoch in range(args.epochs):
        t_st=time.time()
        
        train(model, optimizer, data, args.dprate)
        time_epoch=time.time()-t_st  # each epoch train times
        time_run.append(time_epoch)
            
        # if epoch % 10 == 0:
        #     print(epoch, 'epochs trained. Current Status: test acc')
        #     test(model, data)
        
        val_acc, val_loss = valid(model, data)
        
        if (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            if epoch >= 20:
                test_acc, test_loss = test(model, data)
        
        # if val_acc > best_val_acc :
        #     best_val_acc = val_acc
        #     test_acc = tmp_test_acc

        # if epoch >= 0:
        #     val_loss_history.append(val_loss)
        #     val_acc_history.append(val_acc)
        #     if args.early_stopping > 0 and epoch > args.early_stopping:
        #         tmp = torch.tensor(
        #             val_loss_history[-(args.early_stopping + 1):-1])
        #         if val_loss > tmp.mean().item():
        #             break
        
        
    t = 0
    for i in time_run:
        t += i
    print("epoch time:", t / args.epochs)
    return 0, 0, 0, 0

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=300, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')    
    parser.add_argument('--prop_lr', type=float, default=0.01, help='learning rate.')    
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay.')  
    parser.add_argument('--prop_wd', type=float, default=0.0005, help='weight decay.')  
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=128, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')
    parser.add_argument('--dprate', type=float, default=0.5, help='dprate for neural networks.')

    parser.add_argument('--train_rate', type=float, default=0.6, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
    parser.add_argument('--nlayer', type=int, default=2, help='num of network layers')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha for APPN.')

    parser.add_argument('--device', type=int, default=0, help='GPU device.')
    parser.add_argument('--runs', type=int, default=1, help='number of runs.')
    parser.add_argument('--net', type=str, choices=['PPRGo_mag', 'LSGPRL_mag', 'LSAPPNPL_mag', 'MLP_mag'], default='LSGCN')
    parser.add_argument('--eps', type=float, default=0.0001, help='eps for PPRGo')
    parser.add_argument('--topk', type=int, default=32, help='topk for PPRGo')

    parser.add_argument('--train_nodes', type=int, default=105415)
    parser.add_argument('--batchsize', type=int, default=150000)
    parser.add_argument('--rws', type=int, default=20)
    parser.add_argument('--root', type=str, default='', help='root dir of data')
    parser.add_argument('--tsplit', type=bool, default=False, help='training set splitted only')
    #parser.add_argument('--osplit', type=bool, default=False, help='maintain original splits')
    #parser.add_argument('--full', type=bool, default=False, help='full-supervise')
    parser.add_argument('--K', type=int, default=2, help='The constant for ChebBase.')
    parser.add_argument('--q', type=int, default=0, help='The constant for ChebBase.')
    #parser.add_argument('--semi_rnd', type=bool, default=False, help='semi random splits')
    parser.add_argument('--ec', type=float, default=1.0, help='Constant of Edge Generating')
    
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #10 fixed seeds for random splits
    # SEEDS=[1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363]
    SEEDS=[983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363]

    print(args)
    print("---------------------------------------------")

    gnn_name = args.net
    if gnn_name == 'MLP_mag':
        Net = MLP_mag
    elif gnn_name == 'PPRGo_mag':
        Net = PPRGo_mag
    elif gnn_name == 'LSGPRL_mag':
        Net = LSGPRL_mag
    elif gnn_name == 'LSAPPNPL_mag':
        Net = LSAPPNPL_mag
    
    results = []
    time_results=[]
    
    for RP in range(args.runs):
        print("RP", RP, "Launched...")
        args.seed=SEEDS[RP]
        data = dict()
        rootname = args.root + '/MAG-scholar/'
        
        data['attr'] = torch.load(rootname + 'processed/mc_attr.pt')
        data['num_nodes'] = data['attr'].size(0)
        data['y'] = torch.load(rootname + 'processed/mc_y.pt')            
        data['train_idx'], data['valid_idx'], data['test_idx'] = mag_split(args.seed, data['num_nodes'], args.train_nodes, n_val = args.train_nodes)
        data['train_idx'] = torch.tensor(data['train_idx'])
        data['valid_idx'] = torch.tensor(data['valid_idx'])
        data['test_idx']  = torch.tensor(data['test_idx'])
        
        # data.topk_train = topk_ppr_matrix(adj_matrix, args.alpha, args.eps, data.train_idx, args.topk, normalization='sym')
        # data.topk_val   = topk_ppr_matrix(adj_matrix, args.alpha, args.eps, data.val_idx,   args.topk, normalization='sym')
        # data.topk_test  = topk_ppr_matrix(adj_matrix, args.alpha, args.eps, data.test_idx,  args.topk, normalization='sym')
        
        if gnn_name in ['PPRGo_mag']:
            dataset = np.load(rootname + 'mag_coarse.npz')
            dataset = dict(dataset)
            adj_matrix = sp.csr_matrix((dataset['adj_matrix.data'], dataset['adj_matrix.indices'], dataset['adj_matrix.indptr']), shape=dataset['adj_matrix.shape'])
            idx = torch.arange(data['num_nodes']).numpy()
            starttime=time.time()
            data['topk_idx'] = topk_ppr_matrix(adj_matrix, args.alpha, args.eps, idx, args.topk, normalization='sym')
            data['topk_idx'] = torch_sparse.SparseTensor.from_scipy(data['topk_idx'])
            print("Precomputation time:", time.time()-starttime)
            print('PPR MATRIX CALCULATED.')
            
        
        test_acc, best_val_acc, theta_0, time_run = RunExp(args, None, data, Net)
        print(f'run_{str(RP+1)} \t test_acc: {test_acc:.4f}')

    max_memory_allocated = torch.cuda.max_memory_allocated()
    print("max memory allocated: ", max_memory_allocated / 1024**3, "GB")

    # test_acc_mean, val_acc_mean, _ = np.mean(results, axis=0) * 100
    # test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    # values=np.asarray(results)[:,0]
    # uncertainty=np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values,func=np.mean,n_boot=1000),95)-values.mean()))

    

    # #print(uncertainty*100)
    # print(f'{gnn_name} on dataset {args.dataset}, in {args.runs} repeated experiment:')
    # print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty*100:.4f}  \t val acc mean = {val_acc_mean:.4f}')
    
