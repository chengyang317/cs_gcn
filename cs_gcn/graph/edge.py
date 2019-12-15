import torch
import collections
from typing import Dict
from pt_pack import node_intersect
import torch_scatter as ts


__all__ = ['Edge', 'EdgeAttr', 'EdgeNull', 'EdgeTopK']


EdgeAttr = collections.namedtuple('EdgeAttr', ['name', 'value', 'op_name'])


class EdgeOp(object):
    op_name = 'BASE'
    caches = {}

    def __init__(self, name, last_op=None):
        self.name = name or self.op_name
        self.next_ops = {}
        self.node_i_ids, self.node_j_ids = None, None
        self.masks = None
        # self.select_ids = None
        self.batch_num, self.node_num, self.device = None, None, None
        # self.last_op = last_op
        if last_op is not None:
            last_op.register_op(self)
            self.batch_num, self.node_num, self.device = last_op.batch_num, last_op.node_num, last_op.device

    def _attr_process(self, attr: torch.Tensor):
        raise NotImplementedError

    def attr_process(self, attr):
        if attr is None:
            return None
        if isinstance(attr, EdgeAttr):
            attr_value = self._attr_process(attr.value)
            return EdgeAttr(attr.name, attr_value, self)
        return self._attr_process(attr)

    def register_op(self, op):
        self.next_ops[op.name] = op

    def load_op(self, op_name):
        if op_name == self.name:
            return self
        return self.next_ops.get(op_name, None)

    @property
    def edge_num(self):
        return len(self.node_i_ids)

    def clear_ops(self):
        self.next_ops = {}

    def norm_attr(self, edge_attr, norm_method):
        edge_attr = edge_attr.view(-1, edge_attr.size(-1))
        edge_attr = edge_attr - edge_attr.max()
        if norm_method == 'softmax':
            exp = edge_attr.exp()
            sums = ts.scatter_add(exp, self.node_i_ids, dim=0) + 1e-9
            norm_attr = exp / sums[self.node_i_ids]
        elif norm_method == 'tanh':
            norm_attr = edge_attr.tanh()
        elif norm_method == 'sigmoid':
            norm_attr = edge_attr.sigmoid()
        elif norm_method == 'none':
            norm_attr = edge_attr
        else:
            raise NotImplementedError()
        return norm_attr

    @property
    def eye_mask_cache(self):
        key = f'eye_{self.batch_num}_{self.node_num}'
        if key not in self.caches:
            eye_mask = torch.eye(self.node_num).expand(self.batch_num, -1, -1).contiguous().bool()
            self.caches[key] = eye_mask
        return self.caches[key]

    @property
    def full_mask_cache(self):
        key = f'full_{self.batch_num}_{self.node_num}'
        if key not in self.caches:
            mask = torch.ones(self.batch_num, self.node_num, self.node_num).bool()
            self.caches[key] = mask
        return self.caches[key]

    @property
    def meshgrid_cache(self):
        key = f'meshgrid_{self.batch_num}_{self.node_num}'
        if key not in self.caches:
            batch_idx, node_i, node_j = torch.meshgrid(torch.arange(self.batch_num) * self.node_num,
                                                       torch.arange(self.node_num), torch.arange(self.node_num)
                                                       )
            node_i = batch_idx + node_i
            node_j = batch_idx + node_j
            self.caches[key] = (node_i, node_j)
        return self.caches[key]

    def expand_node_attr(self, node_attr):
        """

        :param node_attr: b,n,c
        :return:
        """
        node_attr = node_attr.view(-1, node_attr.size(-1))
        return node_attr[self.node_i_ids], node_attr[self.node_j_ids]

    def filter_edge_attr(self, edge_attr):
        """

        :param edge_attr: b,n,n,c
        :return:
        """
        edge_attr = edge_attr.view(-1, edge_attr.size(-1))
        if self.select_ids is None:
            return edge_attr
        else:
            return edge_attr[self.select_ids]

    def combine_node_attr(self, node_feats, method):
        n_i_feats, n_j_feats = self.expand_node_attr(node_feats)
        if method == 'cat':
            joint_feats = torch.cat((n_i_feats, n_j_feats), dim=-1)
        elif method == 'mul':
            joint_feats = n_i_feats * n_j_feats
        else:
            raise NotImplementedError()
        return joint_feats

    def batch_ids(self, node):
        return node.batch_ids[self.node_j_ids]

    def expand_cond_attr(self, cond_attr, node):
        """

        :param cond_attr: b,c
        :return:
        """
        if self.masks is not None:
            return cond_attr[self.batch_ids(node)]
        else:
            return cond_attr.view(self.batch_num, 1, 1, cond_attr.size(-1)).repeat(1, self.node_num, self.node_num, 1).view(-1, cond_attr.size(-1))

    def origin_reshape(self, edge_attr, fill_value=0.):
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        c_dim = edge_attr.size(-1)
        if self.masks is None:
            return edge_attr.view(self.batch_num, self.node_num, -1, c_dim)
        new_attr = edge_attr.new_full((self.batch_num, self.node_num, self.node_num, c_dim), fill_value)
        # new_attr = new_attr.masked_fill(self.mask.unsqueeze(-1), edge_attr)
        new_attr[self.masks] = edge_attr
        return new_attr


class EdgeNull(EdgeOp):
    op_name = 'NULL'

    def __init__(self):
        super().__init__('null', None)

    def attr_process(self, attr):
        if attr is None:
            return None
        if isinstance(attr, EdgeAttr):
            assert isinstance(attr.op, EdgeNull)
        return attr


class EdgeInit(EdgeOp):
    op_name = 'INIT'
    ids_cache = {}

    def __init__(self, method: str, node):
        super().__init__(f'init_{method}')
        self.method = method
        self.op_process(node)

    def op_process(self, node):
        self.batch_num, self.node_num, self.device = node.batch_num, node.node_num, node.device
        if node.masks is None and self.method == 'full':
            self.masks = None
        elif node.masks is None:
            self.masks = self.init_masks()
        else:
            self.masks = node_intersect(node.masks.unsqueeze(-1), 'mul').squeeze(-1) * self.init_masks()
        node_i, node_j = self.meshgrid_cache  # b, n, n
        if self.masks is not None:
            node_i, node_j = node_i.cuda(self.device)[self.masks], node_j.cuda(self.device)[self.masks]  # k, k
        else:
            node_i, node_j = node_i.view(-1).cuda(self.device), node_j.view(-1).cuda(self.device)
        self.node_i_ids, self.node_j_ids = node.map_idx(node_i), node.map_idx(node_j)

        # if self.mask is None:
        #     self.select_ids = torch.arange(len(self.node_i_ids), device=self.device)
        # else:
        #     self.select_ids = self.mask.view(-1).nonzero().squeeze()  # k

    def _attr_process(self, attr: torch.Tensor):
        attr = attr.view(-1, attr.shape[-1])
        assert attr.shape[0] == self.node_num * self.node_num * self.batch_num
        return attr[self.select_ids]

    def init_masks(self):
        if self.method == 'full':
            return self.full_mask_cache.cuda(self.device)
        elif self.method == 'not_eye':
            return ~self.eye_mask_cache.cuda(self.device)
        else:
            raise NotImplementedError()


class EdgeTopK(EdgeOp):
    op_name = 'TOPK'

    def __init__(self, by_attr: torch.Tensor, reduce_size, last_op, name=None, keep_self=False):
        super().__init__(name or f'top_{reduce_size}', last_op)
        # min_node_num = last_op.node.min_node_num
        # if reduce_size > min_node_num:
        #     print(f'reduce_size warning')
        #     raise IndexError()
        # self.reduce_size = min(reduce_size, min_node_num)
        self.reduce_size = reduce_size
        self.top_ids = None
        self.l_select_ids = None
        self.keep_self = keep_self
        self.op_process(by_attr, last_op)

    def op_process(self, by_attr, last_op):
        fake_by_attr = last_op.origin_reshape(by_attr, fill_value=-1e3)
        self.top_ids = self.attr_topk(fake_by_attr, -2, self.reduce_size, keep_self=self.keep_self)

        l_select_ids = last_op.origin_reshape(torch.arange(last_op.edge_num).cuda(self.device), fill_value=-1)
        l_select_ids = l_select_ids.gather(index=self.top_ids, dim=-2).view(-1)
        self.l_select_ids = l_select_ids[l_select_ids != -1]
        self.node_i_ids, self.node_j_ids = self._attr_process(last_op.node_i_ids), self._attr_process(
            last_op.node_j_ids)
        # self.select_ids = self._attr_process(last_op.select_ids)

    def attr_topk(self, attr, dim, reduce_size=-1, keep_self=False):
        """

        :param attr: b, n, n, k
        :param dim:
        :param reduce_size:
        :param keep_self
        :return: o_b, n_num, k,
        """
        c_dim = attr.shape[-1]
        if c_dim > 1:
            attr = attr.mean(dim=-1, keepdim=True)  # b, n, n, 1
        if not keep_self:
            attr, top_ids = attr.topk(reduce_size, dim=dim, sorted=False)  # b,n,k,1
        else:
            loop_mask = self.eye_mask_cache.cuda(self.device)
            fake_attr = attr.masked_fill(loop_mask.unsqueeze(-1), 1e4)
            _, top_ids = fake_attr.topk(reduce_size, dim=dim, sorted=False)
            # attr = attr.gather(index=top_ids, dim=-2)

        return top_ids

    def _attr_process(self, edge_attr: torch.Tensor):
        """

        :param edge_attr: b,c
        :return:
        """
        return edge_attr[self.l_select_ids]


class Edge(EdgeInit):
    op_name = 'EDGE_INIT'

    def __init__(self,
                 node,
                 init_method: str,
                 ):
        super().__init__(init_method, node)
        self.geo_layer = None
        self.feat_layers, self.score_layers, self.param_layers = {}, {}, {}

    def geo_feats(self, node):
        node_size, _ = node.size_center()
        node_i_box, node_j_box = self.expand_node_attr(node.boxes)
        node_i_size, node_j_size = self.expand_node_attr(node_size)

        node_dist = node_i_box - node_j_box
        node_dist = node_dist / (node_i_size.repeat(1, 2) + 1e-9)
        node_scale = node_i_size / (node_j_size + 1e-9)
        node_mul = (node_i_size[:, 0] * node_j_size[:, 1]) / (node_j_size[:, 0] * node_j_size[:, 1] + 1e-9)
        node_sum = (node_i_size[:, 0] + node_j_size[:, 1]) / (node_j_size[:, 0] + node_j_size[:, 1] + 1e-9)
        return torch.cat((node_dist, node_scale, node_mul[:, None], node_sum[:, None]), dim=-1)











