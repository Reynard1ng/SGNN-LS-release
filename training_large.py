import argparse
from signal import valid_signals
from dataset_loader import DataLoader
from utils import random_splits, random_splits_citation, fixed_splits, batch_generator_idx
from LSGPRL import *
from LSAPPNPL import *
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
from ogb.nodeproppred import PygNodePropPredDataset

def RunExp(args, dataset, data, Net):

    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    host = torch.device('cpu')

    def train(model, optimizer, data, dprate):
        # t_st = time.time()
        # with torch.no_grad():
        #     model.eval()
        #     model.fill_hist(data)
        # print("Filling historcial embedding time:", time.time()-t_st)
        batchStack = batch_generator_idx(data.train_split, args.batchsize)
        for i in batchStack:
            model.train()
            optimizer.zero_grad()
            out = model(data, i.to(device)).to(host)
            loss = F.nll_loss(out, data.y[i].long())
            print(out.max(1)[1].eq(data.y[i].long()).sum().item() / i.size()[0], loss.item())
            loss.backward()
            optimizer.step()
            del out
            torch.cuda.empty_cache()

    def test(model, data):
        # if args.net in ['LSGPRL']:
        #     #model.to(host)
        #     model.eval()
        #     logits, accs, losses, preds = model(data, None), [], [], []
        #     for _, mask in data('train_split', 'valid_split', 'test_split'):
        #         pred = logits[mask].max(1)[1]
        #         acc = pred.eq(data.y[mask].long()).sum().item() / mask.size()[0]
        #         loss = F.nll_loss(model(data, None)[mask], data.y[mask].long())
        #         preds.append(pred.detach().cpu())
        #         accs.append(acc)
        #         losses.append(loss.detach().cpu())
        #     return accs, preds, losses
       
        model.eval()
        
        sum_val, loss_val = 0, 0
        batchStack = batch_generator_idx(data.valid_split, args.batchsize)
        for i in batchStack:
            out = model(data, i.to(device)).to(host)
            loss = F.nll_loss(out, data.y[i].long())
            sum_val += out.max(1)[1].eq(data.y[i].long()).sum().item()
            loss_val += loss.item() * i.size()[0]
            torch.cuda.empty_cache()
        print("valid_acc / valid loss:", sum_val / data.valid_split.size()[0], loss_val / data.valid_split.size()[0])
        
        sum_test, loss_test = 0, 0
        batchStack = batch_generator_idx(data.test_split, args.batchsize)
        for i in batchStack:
            out = model(data, i.to(device)).to(host)
            loss = F.nll_loss(out, data.y[i].long())
            sum_test += out.max(1)[1].eq(data.y[i].long()).sum().item()
            loss_test += loss.item() * i.size()[0]
            torch.cuda.empty_cache()
        print("test acc / test loss:", sum_test / data.test_split.size()[0], loss_test / data.test_split.size()[0])
        
        return sum_val / data.valid_split.size()[0], sum_test / data.test_split.size()[0]

    t_st=time.time()
    tmp_net = Net(dataset, args)
    model = tmp_net.to(device)
    print("Model construction time:", time.time()-t_st)

    if args.net in ['LSGPRL']:
        optimizer = torch.optim.Adam([
            {'params': model.lins.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.att,               'weight_decay': args.prop_wd, 'lr': args.prop_lr}
        ])
    elif args.net in ['LSAPPNPL']:
        optimizer = torch.optim.Adam([
            {'params': model.lins.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr}
        ])   
    time_run=[]
    model.move_to_device()
    best_val_acc, best_test_acc = 0, 0
    for epoch in range(args.epochs):
        t_st=time.time()
        train(model, optimizer, data, args.dprate)
        val_acc, test_acc = test(model, data)
        if val_acc > best_val_acc:
            best_val_acc, best_test_acc = val_acc, test_acc
        time_epoch=time.time()-t_st  
        time_run.append(time_epoch)
        print(epoch, 'epochs trained.', time_epoch, 'time spent.', flush = True)
    
    #model.to(host)
    model.move_to_host()
    
        
    return best_val_acc, best_test_acc, tmp_net.att.clone().detach().cpu().flatten().numpy(), time_run

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data')
    parser.add_argument('--seed', type=int, default=4200, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=2, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')       
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay.')  
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')

    parser.add_argument('--rws', type=int, default=10, help='random walks per node')
    parser.add_argument('--batchsize', type=int, default=100000, help='num of batches')
    parser.add_argument('--ban0', type=bool, default=False, help='ban own embeddings')
    parser.add_argument('--sdim', type=bool, default=False, help='data dim reduction')
    

    parser.add_argument('--train_rate', type=float, default=0.6, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
    parser.add_argument('--K', type=int, default=2, help='propagation steps.')
    parser.add_argument('--nlayer', type=int, default=1, help='num of network layers')
    parser.add_argument('--alpha', type=float, default=0.9, help='alpha for APPN.')
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')

    parser.add_argument('--dataset', type=str, choices=['ogbn-papers100M'],default='ogbn-papers100M')
    parser.add_argument('--device', type=int, default=0, help='GPU device.')
    parser.add_argument('--runs', type=int, default=5, help='number of runs.')
    parser.add_argument('--net', type=str, choices=['LSGPRL', 'LSAPPNPL'], default='LSGPRL')
    parser.add_argument('--prop_lr', type=float, default=0.01, help='learning rate for propagation layer.')
    parser.add_argument('--prop_wd', type=float, default=0.0005, help='learning rate for propagation layer.')
    parser.add_argument('--Init', type=str, choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'], default='PPR')

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

    t_st=time.time()
    gnn_name = args.net
    if gnn_name == "LSGPRL":
        Net = LSGPRL
    elif gnn_name == "LSAPPNPL":
        Net = LSAPPNPL
    dataset = PygNodePropPredDataset(name = args.dataset, root = args.root)
    print('Loading time:', time.time()-t_st)
    results = []
    time_results=[]
    
    data = dataset[0]
    if args.dataset in ['ogbn-papers100M']:
        split_idx = dataset.get_idx_split()
        data.y = data.y.flatten()
        data.train_split = split_idx['train']
        data.valid_split = split_idx['valid']
        data.test_split = split_idx['test']
    

    val_acc, test_acc, theta_0, time_run = RunExp(args, dataset, data, Net)
    print(theta_0)
    # time_results.append(time_run)
    # results.append([test_acc, best_val_acc, theta_0])
    print(f'best test_acc: {test_acc:.4f}')

    # run_sum=0
    # epochsss=0
    # for i in time_results:
    #     run_sum+=sum(i)
    #     epochsss+=len(i)

    # print("each run avg_time:",run_sum/(args.runs),"s")
    # print("each epoch avg_time:",1000*run_sum/epochsss,"ms")

    # test_acc_mean, val_acc_mean, _ = np.mean(results, axis=0) * 100
    # test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    # values=np.asarray(results)[:,0]
    # uncertainty=np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values,func=np.mean,n_boot=1000),95)-values.mean()))

    # #print(uncertainty*100)
    # print(f'{gnn_name} on dataset {args.dataset}, in {args.runs} repeated experiment:')
    # print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty*100:.4f}  \t val acc mean = {val_acc_mean:.4f}')
    
