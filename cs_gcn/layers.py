import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch
from .modules import *
import torch_scatter as ts

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
        _, hid = self.rnn(packed)
        hid = self.dropout_l(hid)
        return hid.squeeze(0)


class GraphStemLayer(nn.Module):

    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 method: str = 'linear',
                 dropout: float = 0.,
                 ):
        super().__init__()
        if method == 'linear':
            self.linear_l = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(node_dim+4, out_dim)),
                nn.ReLU(),
            )
        elif method == 'film':
            self.linear_l = FilmFusion(node_dim, cond_dim, out_dim, dropout=dropout)
        else:
            raise NotImplementedError()
        self.drop_l = nn.Dropout(dropout)
        self.method = method
        self.node_dim = node_dim

    def forward(self, graph):
        node_feats = graph.node.node_feats('cat', self.drop_l)
        if self.method == 'linear':
            node_feats = self.linear_l(node_feats)  # b, k, hid_dim
        elif self.method == 'film':
            node_feats = self.linear_l(node_feats, graph.cond_feats)
        graph.node.update_feats(node_feats)
        return graph


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
                 dropout: float = 0.,
                 ):
        super().__init__()
        self.e_feat_l = EdgeFeat(node_dim, cond_dim, edge_dim, e_feat_method, dropout)
        self.e_weight_l = EdgeWeight(edge_dim, e_weight_method, dropout)
        self.e_param_l = EdgeParam(edge_dim, out_dim, e_param_method, dropout)
        self.n_feat_l = NodeFeat(node_dim, cond_dim, out_dim, n_feat_method, dropout)
        self.act_l = nn.ReLU()

    def forward(self, graph):
        graph = self.e_feat_l(graph)
        graph = self.e_weight_l(graph)
        graph = self.e_param_l(graph)
        graph = self.n_feat_l(graph)

        b_num, n_num, c_num = graph.batch_num, graph.node_num, graph.node.feat_dim
        e_weights = graph.edge_attrs['weights'].value * graph.edge_attrs['params'].value
        n_feats = graph.node.feats
        last_op = graph.edge_attrs['weights'].op
        n_j_feats = n_feats.view(b_num*n_num, c_num)[last_op.node_j_ids]
        nb_feats = e_weights * n_j_feats
        nb_feats = ts.scatter_add(nb_feats, last_op.node_i_ids, dim=0)
        nb_feats = self.act_l(nb_feats)
        n_feats = nb_feats.new_zeros((b_num*n_num, c_num))
        n_feats[:nb_feats.size(0)] = nb_feats
        graph.node.update_feats(n_feats.view(b_num, n_num, c_num))
        return graph


class GraphClsLayer(nn.Module):
    def __init__(self,
                 v_dim: int,
                 q_dim: int,
                 out_dim: int,
                 method: str,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.method = method
        if method == 'linear':
            self.cls_l = nn.Sequential(
                nn.Dropout(dropout),
                nn.utils.weight_norm(nn.Linear(v_dim, out_dim // 2)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(out_dim // 2, out_dim))
            )
        else:
            raise NotImplementedError()

    def forward(self, n_feats, q_feats):
        if self.method == 'linear':
            logits = self.cls_l(n_feats)
        return logits

























