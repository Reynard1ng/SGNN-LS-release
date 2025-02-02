import argparse
import torch
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, Amazon, Actor
from torch_geometric.data import Data, InMemoryDataset, download_url
from LINKX_dataset import LINKXDataset
from Hetero_dataset import HeteroDataset
from torch_geometric.utils import homophily, degree, is_undirected, to_undirected
import torch_geometric.transforms as T
from torch_geometric.transforms import LargestConnectedComponents, ToUndirected, Compose
from ogb.nodeproppred import PygNodePropPredDataset


def DataLoader(name):
    root = "./data"
    name = name.lower()

    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=root, name=name, transform=T.NormalizeFeatures())
    elif name in ['computers', 'photo']:
        dataset = Amazon(root=root, name=name, transform=T.NormalizeFeatures())
    elif name in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(root=root, name=name, transform=T.NormalizeFeatures())
    elif name in ['chameleon', 'squirrel']:
        # For directed setting
        # dataset = WikipediaNetwork(root=root, name=name, transform=T.NormalizeFeatures())
        # For GPRGNN-like dataset, use everything from "geom_gcn_preprocess=False" and
        # only the node label y from "geom_gcn_preprocess=True"
        preProcDs = WikipediaNetwork(
            root=root, name=name, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(
            root=root, name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.edge_index = preProcDs[0].edge_index
    elif name in ['film', 'actor']:
        root += '/film/'
        dataset = Actor(root=root, transform=T.NormalizeFeatures())
    elif name in ['penn94', 'genius', 'wiki', 'pokec', 'arxiv-year',
                'twitch-gamer', 'snap-patents', 'twitch-de', 'deezer-europe']:
        dataset = LINKXDataset(root=root, name=name)
        if name != 'arxiv-year' and name != 'snap-patents':
            dataset.data['edge_index'] = to_undirected(dataset.data['edge_index'])
    elif name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers']:
        if name == 'ogbn-papers':
            dataset = PygNodePropPredDataset(name='ogbn-papers100M', root=root)
        else:
            dataset = PygNodePropPredDataset(name=name, root=root)
    elif name in ['roman_empire', 'amazon_ratings', 'questions', 'minesweeper', 'tolokers']:
        dataset = HeteroDataset(root=root, name=name, transform=ToUndirected())
    return dataset


# test

# print(111)
# root = "./data/"
# parser = argparse.ArgumentParser()
# parser.add_argument("--dataset", type=str, default='cora')
# args = parser.parse_args()

# name = args.dataset


# dataset = DataLoader(name)
# data = dataset[0]

# saver = torch.load('./data_saved/'+args.dataset+'.pt')
# data.edge_index = saver['edge_index']
# data.y = data.y.flatten()
# args.runs = saver['train_mask'].shape[1]
# print(saver['train_mask'].shape)

# print(data)
# print(data.x.shape[0])
# edge_index = data.edge_index
# print(data.edge_index.shape[1])
# print(data.x.shape[1])
# print(dataset.num_classes)

# undirected = is_undirected(edge_index)
# print(undirected)
# homo = homophily(data.edge_index, data.y)
# homo_node = homophily(data.edge_index, data.y, method='node')
# print(f"{homo:.3f}", f"{homo_node:.3f}")
# #ass = assortativity(data.edge_index)
# #print(f"{ass:.3f}")
# deg = data.edge_index.shape[1] / data.x.shape[0]
# print(f"{deg:.3f}")