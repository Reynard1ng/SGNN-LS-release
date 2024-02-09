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
from ChebnetII_pro import ChebnetII_prop
from Chebbase_pro import Chebbase_prop
from GPR_pro import GPR_prop
from torch_geometric.nn.inits import zeros
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
import networkx as nx
from torch_geometric.utils.undirected import is_undirected, to_undirected
from PPRGo_sup import MixedLinear, MixedDropout


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
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)
#only for message passing

class LSAPPNPL_mag(torch.nn.Module):
    def __init__(self, data, args):
        super(LSAPPNPL_mag, self).__init__()
        self.num_features = data['attr'].size(1)
        self.hidden = args.hidden
        self.dropout = args.dropout
        self.dprate = args.dprate
        self.K = args.K
        self.nlayer = args.nlayer
        self.ec = args.ec
        
        self.lins = ModuleList()
        self.lins.append(MixedLinear(self.num_features, self.hidden))
        self.lins.append(Linear(self.hidden, self.hidden))
        self.lins.append(Linear(self.hidden, 8))
        self.sparsedp = MixedDropout(self.dropout)
        
        tmptensor = args.alpha * ((1-args.alpha) ** torch.arange(0, args.K+1))
        tmptensor[-1] = (1-args.alpha) ** args.K
        # tmptensor = torch.ones(args.K+1)
        tmptensor = tmptensor.repeat(1, 1)
        self.att = tmptensor

        # rws settings 
        self.rws = args.rws
        self.device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
        self.host = torch.device('cpu')

        #graph information
        
        rootname = args.root + "/MAG-scholar/processed/"
        self.adj        = torch.load(rootname + 'mc_SparseTensor.pt')
        self.prop_adj   = torch.load(rootname + 'mc_SparseTensorNorm.pt')
        self.degree     = torch.load(rootname + 'mc_deg.pt')
        self.deg_inv_sqrt = torch.load(rootname + 'mc_deg_is.pt')
        self.N = data['num_nodes']
        self.M = self.prop_adj.nnz()
        #self.hist_emb = torch.zeros(data.num_nodes, self.hidden)
        
        self.inbatch = torch.zeros(self.N, dtype=torch.bool)
        self.cubatch = torch.arange(self.N, dtype=torch.long)
        
        self.passer = passing()
        self.move()
        self.reset_parameters()
        
    def reset_parameters(self):
        for i in range(3):
            self.lins[i].reset_parameters()
    
    def move(self):
        self.passer = self.passer.to(self.device)
        self.degree = self.degree.to(self.device)
        self.deg_inv_sqrt = self.deg_inv_sqrt.to(self.device)
        #self.edge_index = self.edge_index.to(self.device)
        #self.weight = self.weight.to(self.device)
        self.adj = self.adj.to(self.device)
        self.prop_adj = self.prop_adj.to(self.device)
        self.inbatch = self.inbatch.to(self.device)
        self.cubatch = self.cubatch.to(self.device)

    def spar_samp(self, batch, hop):
        dsum = self.degree[batch].sum()
        num_edges = self.rws * batch.size()[0]
        samp_head_idx = torch.multinomial(self.degree[batch], num_edges, replacement = True)
        samp_head = batch[samp_head_idx]
        samp_end = self.adj.random_walk(samp_head, hop)[:, -1]
        val = self.deg_inv_sqrt[samp_head] * self.deg_inv_sqrt[samp_end] * dsum / num_edges
        return samp_head_idx, samp_end, val
        
    def Lin_dev_dev(self, id, x, dropout):
        if id == 0:
            return self.lins[id](self.sparsedp(x))
        else:
            return self.lins[id](F.dropout(x, p = dropout, training=self.training))
    
    def forward(self, x, batch):
        # if self.training:
        self.inbatch[batch] = True
        self.cubatch[batch] = torch.arange(0, batch.size()[0], device = self.device)
        # x = x[batch].to(self.device)
        # x = self.Lin_dev_dev(0, x, self.dropout)
        aggx = self.Lin_dev_dev(0, x[batch], self.dropout)
        aggx = F.relu(aggx)
        aggx = self.Lin_dev_dev(1, aggx, self.dropout) * self.att[0, 0]
        
        for i in range(1, self.K + 1):
            heads, ends, weight = self.spar_samp(batch, i)
            tmp = self.Lin_dev_dev(0, x[ends], self.dropout)
            tmp = F.relu(tmp)
            tmp = self.Lin_dev_dev(1, tmp, self.dropout)
            propMat = SparseTensor(
                row = heads,
                col = torch.arange(0, ends.size()[0], device = self.device),
                value = weight,
                sparse_sizes = (batch.size()[0], ends.size()[0])
            ).coalesce()
            aggx += self.att[0, i] * self.passer(tmp, adj_t = propMat)
        x = F.relu(aggx)
        x = self.Lin_dev_dev(2, x, 0.0)
        self.inbatch[batch] = False
        # else:
        #     x = self.Lin_hos_dev(0, x, self.dropout)
        #     aggx = x * self.att[0, 0].to(self.host)
        #     for j in range(1, self.K + 1):
        #         x = self.passer(x, adj_t = self.prop_adj)
        #         aggx += x * self.att[0, j].to(self.host)
        #     x = F.relu(aggx)
        #     x = self.Lin_hos_dev(1, x, self.dropout)
        #     x = F.relu(x)
        #     x = self.Lin_hos_dev(2, x, 0.0)
        return F.log_softmax(x, dim=1)