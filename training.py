import argparse
from signal import valid_signals
from dataset_loader import DataLoader
from utils import random_splits, random_splits_citation,fixed_splits,batch_generator
from LSGPR import *
from LSGPRM import *
from LSAPPNP import *
from LSJACOBI import *
from LSFAVARD import *
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
from utils import DataSplit
from torch_sparse import coalesce
from torch_geometric.utils.undirected import is_undirected, to_undirected
from torch import pca_lowrank, matmul
from sklearn.metrics import roc_auc_score
import os

def RunExp(args, dataset, data, Net):

    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    host = torch.device('cpu')

    def train(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        if out.shape[1] == 1:
            loss = F.binary_cross_entropy_with_logits(out.squeeze(-1), data.y[data.train_mask].to(torch.float))
        else:
            loss = F.nll_loss(out, data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        del out
        torch.cuda.empty_cache()

    def test(model, data):
        with torch.no_grad():
            model.eval()
            logits, accs, losses, preds = model(data), [], [], []
            for _, mask in data('train_mask', 'val_mask', 'test_mask'):
                if logits.shape[1] == 1:
                    pred = (logits[mask].squeeze(-1) > 0).to(torch.long)
                    acc = roc_auc_score(y_true=data.y[mask].cpu().numpy(), y_score=logits[mask].squeeze(-1).cpu().numpy())
                    loss = F.binary_cross_entropy_with_logits(logits[mask].squeeze(-1), data.y[mask].to(torch.float))
                else:
                    pred = logits[mask].max(1)[1]
                    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                    loss = F.nll_loss(model(data)[mask], data.y[mask])
                preds.append(pred.detach().cpu())
                accs.append(acc)
                losses.append(loss.detach().cpu())
            return accs, preds, losses
    
    tmp_net = Net(dataset, args)
    
    model = tmp_net.to(device)
    data = data.to(device)

    if args.net in ['LSGPR', 'LSGPRM']:
        optimizer = torch.optim.Adam([
            {'params': model.lins.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.att,               'weight_decay': args.prop_wd, 'lr': args.prop_lr}
        ])
    elif args.net in ['LSAPPNP']:
        optimizer = torch.optim.Adam([
            {'params': model.lins.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        ])
    elif args.net in ['LSJACOBI']:
        optimizer = torch.optim.Adam([
            {'params': model.lins.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.att,               'weight_decay': args.prop_wd, 'lr': args.prop_lr},
            {'params': model.alphas,            'weight_decay': args.prop_wd, 'lr': args.prop_lr},
        ])
    elif args.net in ['LSFAVARD']:
        optimizer = torch.optim.Adam([
            {'params': model.lins.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.att,               'weight_decay': args.prop_wd, 'lr': args.prop_lr},
            {'params': model.yitas,             'weight_decay': args.prop_wd, 'lr': args.prop_lr},
            {'params': model.sqrt_betas,        'weight_decay': args.prop_wd, 'lr': args.prop_lr},
        ])
    
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    time_run=[]
    for epoch in range(args.epochs):
        t_st=time.time()
        
        train(model, optimizer, data, args.dprate)
        if (args.net in ['GraphSAGE_RW_mini_GC']) and ((epoch + 1) % 10 != 0):
            print(epoch, 'epochs trained.')
            continue
        time_epoch=time.time()-t_st  # each epoch train times
        time_run.append(time_epoch)
        [train_acc, val_acc, tmp_test_acc], preds, [train_loss, val_loss, tmp_test_loss] = test(model, data)
        
        print(epoch, 'epochs trained. Current Status:', train_acc, val_acc, tmp_test_acc)

        '''
        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
        '''
        if val_acc > best_val_acc :
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            if args.net in ['LSGPR', 'LSGPRM', 'LSAPPNP', 'LSJACOBI', 'LSFAVARD']:
                theta = tmp_net.att.clone().detach().cpu().flatten().numpy()
            else:
                theta = args.alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break
    return test_acc, best_val_acc, theta, time_run

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')       
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay.')  
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')

    parser.add_argument('--rws', type=int, default=10, help='random walks per node')
    parser.add_argument('--batch', type=int, default=1, help='num of batches')
    parser.add_argument('--ban0', type=bool, default=False, help='ban own embeddings')
    parser.add_argument('--sdim', type=bool, default=False, help='data dim reduction')
    

    parser.add_argument('--train_rate', type=float, default=0.6, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
    parser.add_argument('--K', type=int, default=10, help='propagation steps.')
    parser.add_argument('--nlayer', type=int, default=1, help='num of network layers')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha for APPN.')
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')
    parser.add_argument('--skip_dropout', default=0.75, type=float, help='skip dp for ARMA.')
    parser.add_argument('--num_stack', default=1, type=int, help='num stacks for ARMA.')

    parser.add_argument('--dataset', type=str, choices=['Cora','Citeseer','Pubmed','Actor','Texas','Cornell','Wisconsin','Photo','Computers','ogbn-arxiv','penn94', 'twitch-gamer', 'twitch-de'], default='Cora')
    parser.add_argument('--device', type=int, default=0, help='GPU device.')
    parser.add_argument('--runs', type=int, default=5, help='number of runs.')
    parser.add_argument('--net', type=str, choices=['LSGPR', 'LSGPRM', 'LSAPPNP', 'LSJACOBI', 'LSFAVARD'], default='LSGPR')
    parser.add_argument('--prop_lr', type=float, default=0.01, help='learning rate for propagation layer.')
    parser.add_argument('--prop_wd', type=float, default=0.0005, help='learning rate for propagation layer.')
    parser.add_argument('--Init', type=str, choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'], default='PPR')
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--paraA', type=float, default=0.1, help='Alpha for GCNII')
    parser.add_argument('--paraB', type=float, default=0.1, help='Beta for GCNII')

    parser.add_argument('--tsplit', type=bool, default=False, help='training set splitted only')
    #parser.add_argument('--osplit', type=bool, default=False, help='maintain original splits')
    #parser.add_argument('--full', type=bool, default=False, help='full-supervise')
    parser.add_argument('--q', type=int, default=0, help='The constant for ChebBase.')
    #parser.add_argument('--semi_rnd', type=bool, default=False, help='semi random splits')
    parser.add_argument('--ec', type=float, default=1.0, help='Constant of Edge Generating')
    
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #10 fixed seeds for random splits
    SEEDS=[1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363]

    print(args)
    print("---------------------------------------------")

    gnn_name = args.net
    
    if gnn_name == "LSGPR":
        Net = LSGPR
    elif gnn_name == "LSGPRM":
        Net = LSGPRM
    elif gnn_name == "LSAPPNP":
        Net = LSAPPNP
    elif gnn_name == "LSJACOBI":
        Net = LSJACOBI
    elif gnn_name == "LSFAVARD":
        Net = LSFAVARD

    dataset = DataLoader(args.dataset)
    print(dataset.num_classes)
    saver = torch.load('./data_saved/'+args.dataset+'.pt')
    
    data = dataset[0]
    data.edge_index = saver['edge_index']
    data.y = data.y.flatten()
    args.runs = saver['train_mask'].shape[1]
    
    results = []
    time_results=[]
    
    for RP in range(args.runs):
        print("RP", RP, "Launched...")
        args.seed=SEEDS[RP]
        data.train_mask, data.val_mask, data.test_mask = saver['train_mask'][:,RP], saver['val_mask'][:,RP], saver['test_mask'][:,RP]
        test_acc, best_val_acc, theta_0, time_run = RunExp(args, dataset, data, Net)
        print(theta_0)
        time_results.append(time_run)
        results.append([test_acc, best_val_acc, theta_0])
        print(f'run_{str(RP+1)} \t test_acc: {test_acc:.4f}')

    run_sum=0
    epochsss=0
    for i in time_results:
        run_sum+=sum(i)
        epochsss+=len(i)

    print("each run avg_time:",run_sum/(args.runs),"s")
    print("each epoch avg_time:",1000*run_sum/epochsss,"ms")

    test_acc_mean, val_acc_mean, _ = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    values=np.asarray(results)[:,0]
    uncertainty=np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values,func=np.mean,n_boot=1000),95)-values.mean()))

    #print(uncertainty*100)
    print(f'{gnn_name} on dataset {args.dataset}, in {args.runs} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty*100:.4f}  \t val acc mean = {val_acc_mean:.4f}')
    
