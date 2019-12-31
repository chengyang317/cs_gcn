import torch
import collections
from .edge import EdgeTopK


__all__ = ['Node']


class Node(object):
    caches = {}

    def __init__(self, node_feats, node_boxes=None, node_masks=None, node_nums=None, params=None):
        self.feat_layers = collections.defaultdict(None)
        self.logit_layers = {}
        self.logits = None
        self.geo_layer = None
        self.params = params
        self.geo_reuse, self.geo_dim, self.geo_method = params.n_geo_reuse, params.n_geo_dim, params.n_geo_method
        self.geo_out_dim = params.n_geo_out_dim
        if node_nums is not None:
            self.batch_num, self.node_num = node_nums.shape[0], node_nums.max().item()
            node_masks = self.node_num_cache.cuda(node_nums.device) < node_nums[:, None]
        if node_boxes is None:
            node_feats, node_boxes = node_feats.split((node_feats.size(-1)-4, 4), dim=-1)
        if node_feats.shape[0] == 1:
            node_feats = node_feats.squeeze(0)
            node_boxes = node_boxes.squeeze(0)
        if node_feats.dim() == 2:
            valid_node_num = node_masks.sum().item()
            if valid_node_num != node_feats.shape[0]:
                node_feats = node_feats[:valid_node_num]
                node_boxes = node_boxes[:valid_node_num]
            max_node_num = node_masks.sum(-1).max().item()
            self.feats = node_feats
            self.boxes = node_boxes
            # self.boxes = kwargs['obj_boxes_sparse'].squeeze()[:valid_node_num].clone()
            # self.boxes = node_boxes[node_masks]
            self.masks = node_masks[:, :max_node_num].contiguous()
            self.batch_num, self.node_num = self.masks.shape
            self.feat_num = node_feats.shape[-1]
        else:
            self.batch_num, self.node_num, self.feat_num = node_feats.shape
            if node_masks is not None:
                self.masks = node_masks
                # self.feats = kwargs['obj_feats_debug'].squeeze(0)[:self.masks.sum().item()]
                # self.boxes = kwargs['obj_boxes_debug'].squeeze(0)[:self.masks.sum().item()]
                self.feats = node_feats[self.masks]
                self.boxes = node_boxes[self.masks]
                self.masks = self.masks[:, :self.max_node_num].contiguous()
                self.node_num = self.max_node_num
            else:
                self.masks = None
                self.feats = node_feats.view(-1, self.feat_num)
                self.boxes = node_boxes.view(-1, 4)
        if self.masks is not None:
            self.idx_map = self.idx_map_cache.cuda(self.device)
            self.idx_map[self.masks.view(-1)] = torch.arange(self.valid_node_num, device=self.device)
            # self.select_ids = self.masks.view(-1).nonzero().squeeze()
            self.batch_ids = self.batch_ids_cache.cuda(self.device)[self.masks]
        else:
            self.idx_map = None
            # self.select_ids = None
            self.batch_ids = self.batch_ids_cache.cuda(self.device).view(-1)
        self._box_size, self._box_center = None, None
        self._geo_feats = None
        self.feats_dim = self.geo_dim + self.feat_num

    @property
    def node_num_cache(self):
        key = f'node_num_{self.batch_num}_{self.node_num}'
        if key not in self.caches:
            self.caches[key] = torch.arange(self.node_num).expand(self.batch_num, -1).contiguous()
        return self.caches[key]

    @property
    def batch_ids_cache(self):
        key = f'batch_id_{self.batch_num}_{self.node_num}'
        if key not in self.caches:
            self.caches[key] = torch.arange(self.batch_num)[:, None].expand(-1, self.node_num).contiguous()
        return self.caches[key]

    @property
    def idx_map_cache(self):
        key = f'idx_map_{self.batch_num}_{self.node_num}'
        if key not in self.caches:
            self.caches[key] = torch.full((self.batch_num*self.node_num, ), fill_value=-1).long()
        return self.caches[key]

    @property
    def valid_node_num(self):
        if self.masks is not None:
            return self.masks.sum().item()
        else:
            return self.batch_num * self.node_num

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

    def map_idx(self, origin_node_i):
        if self.idx_map is None:
            return origin_node_i
        else:
            return self.idx_map[origin_node_i]

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

    def size_center(self):
        if self._box_size is None:
            boxes = self.boxes
            self._box_size = (boxes[:, 2:] - boxes[:, :2])
            self._box_center = boxes[:, :2] + 0.5 * self._box_size  # b, k, 2
        return self._box_size, self._box_center

    def geo_feats(self, geo_layer=None):
        if self.geo_reuse and self._geo_feats is not None:
            return self._geo_feats
        # geo_feats = torch.cat([*self.size_center()], dim=-1)
        geo_feats = torch.cat([self.boxes, *self.size_center()], dim=-1)
        if self.geo_dim is not None:
            geo_feats = geo_feats.repeat(1, self.geo_dim//geo_feats.size(-1))
        if geo_layer is not None and self.geo_layer is None:
            self.geo_layer = geo_layer
        if self.geo_layer is not None:
            geo_feats = self.geo_layer(geo_feats)
        if self.geo_reuse:
            self._geo_feats = geo_feats
        return geo_feats

    def node_feats(self, geo_layer=None, drop_l=None):
        node_feats = self.feats
        if drop_l is not None:
            node_feats = drop_l(node_feats)
        if 'none' in self.geo_method:
            return node_feats
        geo_feats = self.geo_feats(geo_layer)
        if 'cat' in self.geo_method:
            node_feats = torch.cat([node_feats, geo_feats], dim=-1)
        elif 'sum' in self.geo_method:
            node_feats = node_feats + geo_feats
        elif 'none' in self.geo_method:
            node_feats = node_feats
        else:
            raise NotImplementedError()
        return node_feats

    def expand_cond_attr(self, cond_attr):
        """

        :param cond_attr: b,c
        :return:
        """
        if self.masks is None:
            return cond_attr.unsqueeze(1).repeat(1, self.node_num, 1).view(-1, cond_attr.size(-1))
        return cond_attr[self.batch_ids]

    def reshape(self, node_attr, fill_value=0.):
        fake_attr = node_attr.new_full((self.batch_num, self.node_num, node_attr.size(-1)), fill_value)
        # new_attr = new_attr.masked_fill(self.mask.unsqueeze(-1), edge_attr)
        fake_attr[self.masks] = node_attr
        return fake_attr







