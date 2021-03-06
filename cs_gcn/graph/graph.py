# coding=utf-8
import torch
from .node import Node
from .edge import Edge
import torch_scatter as ts


__all__ = ['Graph']


class Graph(object):
    def __init__(self,
                 node_feats,
                 node_boxes=None,
                 node_masks=None,
                 node_nums=None,
                 init_method: str = 'full',
                 cond_feats=None,
                 params=None
                 ):
        self.node = Node(node_feats, node_boxes, node_masks, node_nums, params)
        self.edge = Edge(self.node, init_method, params)
        self.cond_feats = cond_feats
        self.feats = list()
        self.params = params

    @property
    def device(self):
        return self.node.device

    @property
    def node_num(self):
        return self.node.node_num

    @property
    def batch_num(self):
        return self.node.batch_num

    @property
    def edge_num(self):
        return self.edge.edge_num

    @property
    def node_weights(self):
        return self.node.weights

    def clear(self):
        self.edge.clear_ops()

    def check(self):
        for obj in (self.node, self.edge):
            for key, value in obj.caches.items():
                if not torch.is_tensor(value):
                    for item in value:
                        if item.is_cuda:
                            raise RuntimeError()
                else:
                    if value.is_cuda:
                        raise RuntimeError()

    def pool_feats(self, method='mean'):
        if 'weight' in method:
            method = method.split('^')[-1]
            node_feats = self.node_feats * self.node_weights
        else:
            node_feats = self.node.feats

        if method == 'mean':
            fake_feats = self.node.reshape(node_feats)
            feats = fake_feats.sum(dim=1)
            feats = feats / self.node.masks.sum(dim=1)
            # feats = ts.scatter_mean(node_feats, self.node.batch_ids, dim=0)
        elif method == 'max':
            fake_feats = self.node.reshape(node_feats, fill_value=-9e10)
            feats = fake_feats.max(dim=1)[0]
            # feats = ts.scatter_max(node_feats, self.node.batch_ids, dim=0)[0]
        elif method == 'sum':
            # feats = ts.scatter_add(node_feats, self.node.batch_ids, dim=0)
            fake_feats = self.node.reshape(node_feats)
            feats = fake_feats.sum(dim=1)
        elif method == 'mix':
            fake_feats = self.node.reshape(node_feats)
            feats = fake_feats.sum(dim=1)
            mean_feat = feats / self.node.masks.sum(dim=1, keepdims=True)
            fake_feats = self.node.reshape(node_feats, fill_value=-9e10)
            max_feat = fake_feats.max(dim=1)[0]
            # max_feat = ts.scatter_max(node_feats, self.node.batch_ids, dim=0)[0]
            # mean_feat = ts.scatter_mean(node_feats, self.node.batch_ids, dim=0)
            feats = torch.cat((max_feat, mean_feat), dim=-1)
        else:
            raise NotImplementedError()
        return feats

        # if method == 'mean':
        #     return node_feats.mean(dim=1).squeeze()
        # elif method == 'max':
        #     return node_feats.max(dim=1)[0]
        # elif method == 'sum':
        #     return node_feats.sum(dim=1)
        # elif method == 'mix':
        #     max_feat = node_feats.max(dim=1)[0]
        #     mean_feat = node_feats.mean(dim=1).squeeze(1)
        #     return torch.cat((max_feat, mean_feat), dim=-1)
        # else:
        #     raise NotImplementedError()










































