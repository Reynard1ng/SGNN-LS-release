import torch
from torch_geometric.utils.undirected import is_undirected, to_undirected
from torch_geometric.utils import to_scipy_sparse_matrix,to_dense_adj,dense_to_sparse,add_remaining_self_loops
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from ogb.nodeproppred import PygNodePropPredDataset

in_root = "./data"
out_root = "./data/ogbn_papers100M/processed/"
dataset = PygNodePropPredDataset(name = 'ogbn-papers100M', root = in_root)
data = dataset[0]
edge_index = data.edge_index
edge_index = add_remaining_self_loops(to_undirected(edge_index))
torch.save(edge_index, out_root + 'su_edge_index.pt')
adj = SparseTensor(row = edge_index[0], col = edge_index[1], sparse_sizes=(data.num_nodes, data.num_nodes))
torch.save(edge_index, out_root + 'su_SparseTensor.pt')
deg = sparsesum(adj, dim = 0)
torch.save(deg, out_root + 'su_deg.pt')
deg_inv_sqrt = deg ** -0.5
torch.save(deg_inv_sqrt, out_root + 'su_deg_is.pt')
adj = adj.fill_value(1.)
adj = mul(adj, deg_inv_sqrt.view(-1, 1))
adj = mul(adj, deg_inv_sqrt.view(1, -1))
torch.save(adj, out_root + 'su_SparseTensorNorm.pt')