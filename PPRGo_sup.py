import math
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul

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

class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        value_dropped = F.dropout(input.storage.value(), self.p, self.training)
        return torch_sparse.SparseTensor(
                row=input.storage.row(), rowptr=input.storage.rowptr(), col=input.storage.col(),
                value=value_dropped, sparse_sizes=input.sparse_sizes(), is_sorted=True)


class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if isinstance(input, torch_sparse.SparseTensor):
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)


class MixedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if isinstance(input, torch_sparse.SparseTensor):
            res = input.matmul(self.weight)
            if self.bias is not None:
                res += self.bias[None, :]
        else:
            if self.bias is not None:
                res = torch.addmm(self.bias, input, self.weight)
            else:
                res = input.matmul(self.weight)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None)


def matrix_to_torch(X):
    if sp.issparse(X):
        return torch_sparse.SparseTensor.from_scipy(X)
    else:
        return torch.FloatTensor(X)
    
    
class PPRGoMLP(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size, nlayers, dropout):
        super().__init__()

        fcs = [MixedLinear(num_features, hidden_size, bias=False)]
        for i in range(nlayers - 2):
            fcs.append(nn.Linear(hidden_size, hidden_size, bias=False))
        fcs.append(nn.Linear(hidden_size, num_classes, bias=False))
        self.fcs = nn.ModuleList(fcs)

        self.drop = MixedDropout(dropout)

    def forward(self, X):
        embs = self.drop(X)
        embs = self.fcs[0](embs)
        for fc in self.fcs[1:]:
            embs = fc(self.drop(F.relu(embs)))
        return embs


class MLP_mag(nn.Module):
    def __init__(self, data, args):
        super(MLP_mag, self).__init__()
        self.num_features = data['attr'].size(1)
        self.dropout =args.dropout
        self.lin1 = MixedLinear(self.num_features, args.hidden)
        self.lin2 = MixedLinear(args.hidden, 8)
        self.sparsedp = MixedDropout(self.dropout)
        

    def forward(self, x, batch):
        x = self.lin1(self.sparsedp(x[batch]))
        x = F.relu(x)
        x = self.lin2(F.dropout(x, p = self.dropout, training=self.training))

        return F.log_softmax(x, dim=1)

class PPRGo(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size, nlayers, dropout):
        super().__init__()
        self.mlp = PPRGoMLP(num_features, num_classes, hidden_size, nlayers, dropout)

    def forward(self, data):
        x, ppr_mat = data.x, data.topk_mat
        
        logits = self.mlp(x)
        # propagated_logits = scatter(logits * ppr_scores[:, None], ppr_idx[:, None],
        #                             dim=0, dim_size=ppr_idx[-1] + 1, reduce='sum')
        logits = ppr_mat.matmul(logits)
        logits = F.log_softmax(logits, dim=1)
        return logits
    
class PPRGo_mag(nn.Module):
    def __init__(self, args, num_nodes, num_features, num_classes, hidden_size, nlayers, dropout, topk_idx):
        super().__init__()
        self.mlp = PPRGoMLP(num_features, num_classes, hidden_size, nlayers, dropout)
        self.device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
        self.inbatch = torch.zeros(num_nodes, dtype=torch.bool)
        self.cubatch = torch.arange(num_nodes, dtype=torch.long)
        self.topk_idx = topk_idx
        self.passer = passing()
        self.move()
        
    def move(self):
        self.inbatch = self.inbatch.to(self.device)
        self.cubatch = self.cubatch.to(self.device)
        self.topk_idx = self.topk_idx.to(self.device)
        self.passer = self.passer.to(self.device)

    def forward(self, x, batch):
        self.inbatch[batch] = True
        self.cubatch[batch] = torch.arange(0, batch.size()[0], device = self.device)
        
        transer = self.topk_idx[batch]
        logits = self.mlp(x[transer.coo()[1]])
        
        propMat = SparseTensor(
                row = transer.coo()[0],
                col = torch.arange(0, transer.coo()[1].size()[0], device = self.device),
                value = transer.coo()[2],
                sparse_sizes = (batch.size()[0], transer.coo()[1].size()[0])
            ).coalesce()
        logits = self.passer(logits, adj_t = propMat)
        
        # propagated_logits = scatter(logits * ppr_scores[:, None], ppr_idx[:, None],
        #                             dim=0, dim_size=ppr_idx[-1] + 1, reduce='sum')
        # logits = ppr_mat.matmul(logits)
        logits = F.log_softmax(logits, dim=1)
        return logits