import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch
from .modules import *
import pt_pack as pt

__all__ = ['CgsQLayer', 'GraphConvLayer', 'GraphStemLayer', 'GraphClsLayer']


class CgsQLayer(nn.Module):
    def __init__(self, vocab_dim, embed_dim: int = 300, hid_dim: int = 1024, dropout: float = 0., padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_dim, embed_dim, padding_idx=padding_idx)
        self.rnn = nn.GRU(embed_dim, hid_dim, batch_first=True)
        self.dropout_l = nn.Dropout(dropout)

    def forward(self, q_labels, q_len):
        emb = self.embedding(q_labels)
        q_length = q_len.squeeze().tolist()
        if not isinstance(q_length, list):
            q_length = [q_length]
        packed = pack_padded_sequence(emb, q_length, batch_first=True, enforce_sorted=False)
        self.rnn.flatten_parameters()
        _, hid = self.rnn(packed)
        hid = self.dropout_l(hid)
        return hid.squeeze(0)


class GraphStemLayer(nn.Module):

    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 method: str = 'linear',
                 params=None,
                 ):
        super().__init__()
        self.params = params
        if params.n_geo_method in ('linear_cat', 'sum'):
            self.n_geo_layer = pt.PtLinear(params.n_geo_dim, params.n_geo_out_dim, orders=params.n_geo_orders,
                                           norm=params.n_geo_norm)
        else:
            self.n_geo_layer = None
        if params.n_geo_method == 'sum':
            n_dim = node_dim
        else:
            n_dim = node_dim + params.n_geo_out_dim
        norm, orders, drop = params.stem_norm, params.stem_orders, params.stem_drop
        if method == 'linear':
            self.linear_l = pt.PtLinear(n_dim, out_dim, norm=norm, orders=orders, drop=drop)
        elif method == 'double_linear':
            if params.stem_use_act:
                new_orders = orders
            else:
                new_orders = orders[:-1] if orders.endswith('a') else orders
            self.linear_l = nn.Sequential(
                pt.PtLinear(n_dim, out_dim, norm=norm, orders=new_orders, drop=drop),
                pt.PtLinear(out_dim, out_dim, norm=norm, orders=orders)
            )
        elif method == 'film':
            self.linear_l = FilmFusion(node_dim, cond_dim, out_dim, dropout=drop)
        else:
            raise NotImplementedError()
        self.method = method
        self.node_dim = node_dim

    def forward(self, graph):
        node_feats = graph.node.node_feats(self.n_geo_layer)
        if self.method in ('linear', 'double_linear'):
            node_feats = self.linear_l(node_feats)  # b, k, hid_dim
        elif self.method == 'film':
            node_feats = self.linear_l(node_feats, graph.cond_feats)
        graph.node.update_feats(node_feats)
        return graph

    @classmethod
    def build(cls, params):
        return cls(params.stem_in_dim, params.q_hid_dim, params.stem_out_dim, params.stem_method, params)


class GraphConvLayer(nn.Module):
    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 edge_dim: int,
                 out_dim: int,
                 e_feat_method: str = 'mul_film',
                 e_weight_method: str = 'linear_softmax_8',
                 e_param_method: str = 'linear',
                 n_feat_method: str = 'film',
                 params=None,
                 ):
        super().__init__()
        self.params = params
        self.e_feat_l = EdgeFeat(node_dim, cond_dim, edge_dim, e_feat_method, params)
        self.e_weight_l = EdgeWeight(edge_dim, e_weight_method, params)
        self.e_param_l = EdgeParam(edge_dim, out_dim, e_param_method, params)
        self.n_feat_l = NodeFeat(node_dim, cond_dim, out_dim, n_feat_method, params)
        self.act_l = nn.ReLU()

    def forward(self, graph):
        e_feats = self.e_feat_l(graph)
        e_weights = self.e_weight_l(graph, e_feats)
        weights_op = graph.edge.load_op(e_weights.op_name)
        e_feats = weights_op.attr_process(e_feats)
        e_params = self.e_param_l(graph, e_feats)
        n_feats = self.n_feat_l(graph)

        e_weights = e_weights.value * e_params.value
        n_j_feats = n_feats[weights_op.node_j_ids]
        nb_feats = e_weights * n_j_feats
        # nb_feats = ts.scatter_add(nb_feats, weights_op.node_i_ids, dim=0)
        nb_feats = nb_feats.view(graph.node.valid_node_num, -1, nb_feats.size(-1)).sum(dim=1)
        nb_feats = self.act_l(nb_feats)
        graph.node.feats = nb_feats
        return graph

    @classmethod
    def build(cls, params):
        return cls(params.stem_out_dim, params.q_hid_dim, params.e_dim, params.stem_out_dim, params.e_f_method,
                   params.e_w_method, params.e_p_method, params.n_f_method, params)


class GraphClsLayer(nn.Module):
    def __init__(self,
                 v_dim: int,
                 q_dim: int,
                 out_dim: int,
                 method: str,
                 params=None
                 ):
        super().__init__()
        self.method = method
        self.params = params
        if method == 'linear':
            self.cls_l = nn.Sequential(
                pt.PtLinear(v_dim, out_dim//2, norm=params.cls_norm, orders=params.cls_orders, drop=params.cls_drop,
                            act=params.cls_act),  # layer norm worse
                pt.PtLinear(out_dim//2, out_dim, norm=params.cls_norm, orders='ln')
            )
        else:
            raise NotImplementedError()

    def forward(self, n_feats, q_feats):
        if self.method == 'linear':
            logits = self.cls_l(n_feats)
        return logits

























