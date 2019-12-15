import torch
import collections
from .edge import EdgeTopK


__all__ = ['Node']


class Node(object):
    caches = {}

    def __init__(self, node_feats, node_boxes=None, node_masks=None, node_nums=None):
        self.feat_layers = collections.defaultdict(None)
        self.logit_layers = {}
        self.logits = None
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

    def node_feats(self, method='clean', drop_l=None):
        node_feats = drop_l(self.feats) if drop_l is not None else self.feats
        if method == 'cat':
            box_feats = torch.cat([*self.size_center(), self.boxes], dim=-1).repeat(1, 8)
            return torch.cat([node_feats, box_feats], dim=-1)
        elif method == 'clean':
            return self.feats
        else:
            raise NotImplementedError()

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

    def expand_cond_attr(self, cond_attr):
        """

        :param cond_attr: b,c
        :return:
        """
        if self.masks is None:
            return cond_attr.unsqueeze(1).repeat(1, self.node_num, 1).view(-1, cond_attr.size(-1))
        return cond_attr[self.batch_ids]








