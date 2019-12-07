# coding=utf-8
import torch
from .node import Node
from .edge import Edge
import numpy as np


__all__ = ['Graph']


class Graph(object):
    def __init__(self,
                 node_feats,
                 node_boxes=None,
                 node_weights=None,
                 node_masks=None,
                 init_method: str = 'full',
                 cond_feats=None,
                 ):
        self.node = Node(self, node_feats, node_boxes, node_weights, node_masks)
        self.edge = Edge(self.node, init_method)
        self.cond_feats = cond_feats
        self.feats = list()

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
    def node_feats(self):
        return self.node.feats

    @property
    def edge_feats(self):
        return self.edge.feats

    @property
    def edge_attrs(self):
        return self.edge.edge_attrs

    @property
    def node_weights(self):
        return self.node.weights

    @property
    def edge_weights(self):
        return self.edge.weights

    def clear(self):
        self.edge.clear_ops()

    def pool_feats(self, method='mean'):
        if 'weight' in method:
            method = method.split('^')[-1]
            node_feats = self.node_feats * self.node_weights
        else:
            node_feats = self.node_feats

        if method == 'mean':
            return node_feats.mean(dim=1).squeeze()
        elif method == 'max':
            return node_feats.max(dim=1)[0]
        elif method == 'sum':
            return node_feats.sum(dim=1)
        elif method == 'mix':
            max_feat = node_feats.max(dim=1)[0]
            mean_feat = node_feats.mean(dim=1).squeeze(1)
            return torch.cat((max_feat, mean_feat), dim=-1)
        else:
            raise NotImplementedError()










































