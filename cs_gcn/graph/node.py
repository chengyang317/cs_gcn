import torch
import collections
from .edge import EdgeTopK


__all__ = ['Node']


class Node(object):
    def __init__(self, graph, node_feats, node_boxes=None, node_weights=None, node_masks=None):
        self.graph = graph
        self.feats = node_feats
        self.boxes = node_boxes
        self.weights = node_weights
        self.masks = node_masks
        self.feat_layers = collections.defaultdict(None)
        self.logit_layers = {}
        self.logits = None

        if self.masks is not None:
            max_node_num = self.masks.sum(-1).max().item()
            self.feats = self.feats[:, :max_node_num]
            self.boxes = self.boxes[:, :max_node_num]
            self.weights = self.weights[:, :max_node_num] if self.weights is not None else None
            self.masks = self.masks[:, :max_node_num]

    @property
    def max_node_num(self):
        if self.masks is None:
            return self.node_num
        else:
            return self.masks.sum(-1).max().item()

    @property
    def min_node_num(self):
        if self.masks is None:
            return self.node_num
        else:
            return self.masks.sum(-1).min().item()

    def node_map(self, origin_node_i):
        return origin_node_i

    def node_feats(self, method='clean', drop_l=None):
        node_feats = drop_l(self.feats) if drop_l is not None else self.feats
        if method == 'cat':
            return torch.cat([node_feats, *self.size_center], dim=-1)
        elif method == 'clean':
            return self.feats
        else:
            raise NotImplementedError()

    @property
    def edge(self):
        return self.graph.edge

    @property
    def indexes(self):
        return torch.arange(self.node_num*self.batch_num, device=self.device)

    @property
    def node_num(self):
        return self.feats.shape[1]

    @property
    def batch_num(self):
        return self.feats.shape[0]

    @property
    def feat_dim(self):
        return self.feats.shape[-1]

    @property
    def device(self):
        return self.feats.device

    def update_feats(self, node_feats=None, node_coords=None, node_weights=None):
        if node_feats is not None:
            self.feats = node_feats
        if node_coords is not None:
            self.boxes = node_coords
        if node_weights is not None:
            self.weights = node_weights

    @property
    def shape(self):
        return self.feats.shape

    @property
    def size_center(self):
        boxes = self.boxes
        node_size = (boxes[:, :, 2:] - boxes[:, :, :2])
        node_centre = boxes[:, :, :2] + 0.5 * node_size  # b, k, 2
        return node_size, node_centre

    def pool(self, logits, norm_method, reduce_size=-1):
        assert logits.shape[:2] == (self.batch_num, self.node_num)
        if logits.shape[-1] != 1:
            logits = logits.mean(dim=-1, keepdim=True)
        logits, top_idx = EdgeTopK.attr_topk(logits, dim=1, reduce_size=reduce_size)
        self.weights = self.norm(logits, norm_method)  # b, node_num, 1
        if top_idx is not None:
            self.feats = self.feats.gather(index=top_idx.expand(-1, -1, self.feats.shape[-1]), dim=1)
            self.boxes = self.boxes.gather(index=top_idx.expand(-1, -1, self.boxes.shape[-1]), dim=1)
        self.feats = (self.weights + 1) * self.feats
        return self

    def logit2weight(self, norm_method, reduce_size=-1):
        edge_logits = self.edge.logits
        node_logits = self.edge.edge2node(edge_logits)  # b,n_num, k
        node_logits, top_indexes = logit_reduce(node_logits, 1, reduce_size)
        node_weights = logit_norm(node_logits, norm_method)
        self.weights = node_weights.squeeze()  # b, n_num
        if top_indexes is not None:
            self.feats = self.feats.gather(index=top_indexes.expand(-1, -1, self.feats.shape[-1]), dim=1)
            self.boxes = self.boxes.gather(index=top_indexes.expand(-1, -1, self.boxes.shape[-1]), dim=1)
        return node_weights

    def norm(self, attr, method):
        if method == 'softmax':
            ret_attr = attr.softmax(dim=-2)  # b, obj_num, n, k_size
            # weight = weight + 1
        elif method == 'tanh':
            ret_attr = attr.tanh()
            # weight = weight + 1  # major change
            # node_weight = node_logit.tanh()
        elif method == 'sigmoid':
            ret_attr = attr.sigmoid()
            # weight = weight + 1  # major change
        elif method == 'self':
            ret_attr = attr
        else:
            raise NotImplementedError()
        return ret_attr






