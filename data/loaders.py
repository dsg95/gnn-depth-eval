import os
import torch

from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB, Actor, Coauthor
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset


def load_dataset(name, split=''):

    if name in ['Cora', 'Pubmed', 'Citeseer']:
        dataset = Planetoid(root=os.getcwd() + '/datasets/', name=name,
                            transform=T.Compose([T.NormalizeFeatures()]),
                            split=split if split != '' else 'geom-gcn')[0]
    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=os.getcwd() + '/datasets/', name=name,
                                   transform=T.Compose([T.ToUndirected()]))[0]
    elif name in ['texas', 'cornell', 'wisconsin']:
        dataset = WebKB(root=os.getcwd() + '/datasets/', name=name,
                        transform=T.Compose([T.ToUndirected()]))[0]
    elif name in ['Actor']:
        dataset = Actor(root=os.getcwd() + '/datasets/' + name,
                        transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures()]))[0]
    elif name in ['CoauthorCS', 'CoauthorPhysics']:
        dataset = Coauthor(root=os.getcwd() + '/datasets/' + name, name=name[8:],
                           transform=T.Compose([T.RandomNodeSplit(num_splits=10,
                                                                  num_val=0.2,num_test=0.2)]))[0]
    elif name in ['ogbn-arxiv']:
        dataset = PygNodePropPredDataset(name=name, root=os.getcwd() + '/datasets/')
        split_idx = dataset.get_idx_split()
        dataset = dataset.data
        dataset.train_mask = torch.zeros((dataset.num_nodes, 1)).to(torch.bool)
        dataset.train_mask[split_idx['train']] = True
        dataset.val_mask = torch.zeros((dataset.num_nodes, 1)).to(torch.bool)
        dataset.val_mask[split_idx['valid']] = True
        dataset.test_mask = torch.zeros((dataset.num_nodes, 1)).to(torch.bool)
        dataset.test_mask[split_idx['test']] = True
        dataset.y = dataset.y.reshape(-1,)
    else:
        raise AssertionError('Could not find loader for dataset %s.' % name)

    return dataset
