import time
from scipy.fftpack import shift
import torch
import random
import math
import torch.nn.functional as F
import os.path as osp
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.autograd import Variable
from torch.nn import Parameter, Linear, ModuleList, LeakyReLU
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GCN2Conv, ChebConv, ARMAConv, APPNP
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_scipy_sparse_matrix,to_dense_adj,dense_to_sparse,add_remaining_self_loops
import scipy.sparse as sp
from torch_geometric.nn.inits import zeros
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
import networkx as nx
from torch_geometric.utils.undirected import is_undirected, to_undirected


#only for message passing
class passing(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')
    
    def forward(self, x, edge_index = None, edge_weight = None, adj_t = None):
        if adj_t is not None:
            return self.propagate(edge_index=adj_t, x=x)
        else:    
            return self.propagate(edge_index=edge_index, x=x, edge_weight=edge_weight)
    
    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)
#only for message passing

class LSFAVARD(torch.nn.Module):
    def __init__(self, dataset, args):
        super(LSFAVARD, self).__init__()
        data = dataset[0]
        #This modification is for cora only.
        #Please use looped version for other datasets.
        #Maybe better to other datasets, but unnecessary.
        
        data.edge_index = add_remaining_self_loops(to_undirected(data.edge_index))[0]
        #data.edge_index = to_undirected(data.edge_index)
        
        if dataset.num_classes == 2:
            self.out_classes = 1
        else:
            self.out_classes = dataset.num_classes
        self.num_features = dataset.num_features
        self.hidden = args.hidden
        self.dropout = args.dropout
        self.dprate = args.dprate
        self.K = args.K
        self.nlayer = args.nlayer
        self.ec = args.ec
        
        self.lins = ModuleList()
        self.lins.append(Linear(self.num_features, self.hidden))
        for i in range(args.nlayer - 1):
            self.lins.append(Linear(self.hidden, self.hidden))
        self.lins.append(Linear(self.hidden, self.out_classes))
        
        # tmptensor = args.alpha * ((1-args.alpha) ** torch.arange(0, args.K+1))
        # tmptensor[-1] = (1-args.alpha) ** args.K
        # tmptensor = tmptensor.repeat(self.nlayer, 1)
        
        tmptensor = torch.ones([self.hidden, args.K+1])
        tmptensor = tmptensor.repeat(self.nlayer, 1, 1)
        self.att = Parameter(tmptensor)

        self.yitas =        Parameter(torch.zeros(self.K+1).repeat(self.hidden,1).float()) # (n_channels, K+1)
        self.sqrt_betas =   Parameter(torch.ones (self.K+1).repeat(self.hidden,1).float()) # (n_channels, K+1)

        # rws settings 
        self.rws = args.rws
        self.device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')

        #graph information
        self.N = data.num_nodes
        self.M = data.num_edges
        self.edge_index = data.edge_index
        self.adj = SparseTensor(
            row = data.edge_index[0],
            col = data.edge_index[1],
            sparse_sizes = (data.num_nodes, data.num_nodes)
        ).coalesce()
        #self.degree = sparsesum(self.adj, dim=0)
        #self.weight = (self.degree[self.edge_index[0]] ** -0.5) * (self.degree[self.edge_index[1]] ** -0.5)
        
        self.adj.fill_value(1.)
        self.degree = sparsesum(self.adj, dim=0)
        deg_inv_sqrt = self.degree ** -0.5
        self.adj = mul(self.adj, deg_inv_sqrt.view(-1, 1))
        self.adj = mul(self.adj, deg_inv_sqrt.view(1, -1))
        
        self.passer = passing()

        self.reset_parameters()
        self.move()
        
    def calc_expansion(self, prevparas, L):
        if L == 0:
            return prevparas[0]
        if L == 1:
            tmp = torch.zeros_like(prevparas[-1], device = self.device)
            tmp[:, 1:] = prevparas[L-1][:, 0:-1]
            tmp -= self.yitas[:, L-1].unsqueeze(1) * prevparas[0]
            tmp /= self.sqrt_betas[:, L].unsqueeze(1)
            return tmp
        tmp = torch.zeros_like(prevparas[-1], device = self.device)
        tmp[:, 1:] = prevparas[L-1][:, 0:-1]
        tmp -= self.yitas[:, L-1].unsqueeze(1) * prevparas[L-1]
        tmp -= self.sqrt_betas[:, L-1].unsqueeze(1) * prevparas[L-2]
        tmp /= self.sqrt_betas[:, L].unsqueeze(1)
        return tmp
        
        
    def reset_parameters(self):
        for i in range(self.nlayer + 1):
            self.lins[i].reset_parameters()
    
    def move(self):
        self.passer = self.passer.to(self.device)
        self.degree = self.degree.to(self.device)
        self.edge_index = self.edge_index.to(self.device)
        #self.weight = self.weight.to(self.device)
        self.adj = self.adj.to(self.device)

    def spar_samp(self, att):
        res = []
        num_edges = int(self.N * math.log(1.0 * self.N) * self.ec)
        genidx = torch.arange(0, num_edges, device = self.device)
        for i in range(1, self.K+1):
            idx = torch.randint(0, self.M, (num_edges, ), device = self.device)
            idl = torch.randint(0, i,      (num_edges, ), device = self.device)
            idr = i - 1 - idl
            l = self.adj.storage.row()[idx]
            r = self.adj.storage.col()[idx]
            le = self.adj.random_walk(l, i-1)[genidx, idl]
            re = self.adj.random_walk(r, i-1)[genidx, idr]
            # mini_graph = SparseTensor(
            #     row = torch.cat((le, re)), 
            #     col = torch.cat((re, le)), 
            #     sparse_sizes = (self.N, self.N)
            # ).coalesce()
            # deg = sparsesum(mini_graph, dim=0)
            val = (self.degree[le] ** -0.5) * (self.degree[re] ** -0.5) * (self.M / num_edges)
            sparse = SparseTensor(
                row = torch.cat((le, re)),
                col = torch.cat((re, le)),
                value = torch.cat((val, val)),
                sparse_sizes = (self.N, self.N)
            ).coalesce()
            res.append(sparse)
        return res
        
    def forward(self, data):
        
        x = data.x
        for i in range(self.nlayer):
            x = self.lins[i](F.dropout(x, p = self.dropout, training = self.training))
            xs = []
            tmp0 = torch.zeros([self.hidden, self.K + 1], device = self.device)
            tmp0[:, 0] = 1 / self.sqrt_betas[:, 0]
            xs.append(tmp0)
            sum_att = self.att[i][:, 0].unsqueeze(1) * tmp0
            
            for j in range(1, self.K + 1):
                tx = self.calc_expansion(xs, j)
                sum_att += self.att[i][:, j].unsqueeze(1) * tx 
                xs.append(tx)
                
            if self.training:
                aggx = x * sum_att[:, 0]
                sparse = self.spar_samp(sum_att)
                for j in range(1, self.K+1):
                    aggx += self.passer(x, adj_t = sparse[j-1]) * sum_att[:, j]
                x = F.relu(aggx)
            else:
                aggx = x * sum_att[:, 0]
                for j in range(1, self.K+1):
                    x = self.passer(x, adj_t = self.adj)
                    aggx += x * sum_att[:, j]
                x = F.relu(aggx)
                
        x = self.lins[-1](x)
        if self.out_classes == 1:
            return x
        else:
            return F.log_softmax(x, dim=1)