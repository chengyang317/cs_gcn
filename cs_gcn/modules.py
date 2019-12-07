import torch.nn as nn
import pt_pack as pt
import torch
from .graph.edge import EdgeAttr
from .graph import Graph


__all__ = ['FilmFusion', 'EdgeFeat', 'EdgeWeight', 'EdgeParam', 'NodeFeat']


class FilmFusion(nn.Module):
    def __init__(self,
                 in_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 norm_type='layer',
                 dropout: float = 0.,
                 act_type: str = 'relu'
                 ):
        super().__init__()
        self.cond_proj_l = nn.utils.weight_norm(nn.Linear(cond_dim, out_dim*2))
        self.drop_l = nn.Dropout(dropout)
        self.film_l = pt.Linear(in_dim, out_dim, norm_type=norm_type, norm_affine=False, orders=('linear', 'norm', 'cond'))
        self.act_l = pt.Act(act_type)

    def forward(self, x, cond):
        gamma, beta = self.cond_proj_l(cond).chunk(2, dim=-1)
        gamma += 1.
        x = self.drop_l(x)
        x = self.film_l(x, gamma, beta)
        x = self.act_l(x)
        return x


class EdgeFeat(nn.Module):
    def __init__(self,
                 node_dim,
                 cond_dim,
                 edge_dim,
                 method: str,
                 dropout=0.,
                 ):
        super().__init__()
        self.method = method
        self.node_dim = node_dim
        self.cond_dim = cond_dim
        self.edge_dim = edge_dim
        if self.method in ('share', 'none'):
            return
        self.n2n, self.n2c = method.split('_')
        self.n_proj = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(node_dim+4, edge_dim)),
            nn.ReLU()
        )
        self.e_geo_linear = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(8, 30)),
            nn.ReLU()
        )
        self.drop_l = nn.Dropout(dropout)
        join_dim = edge_dim + 30 if self.n2n in ('sum', 'mul', 'max') else edge_dim * 2 + 30
        self.n2c_fusion = FilmFusion(join_dim, cond_dim, edge_dim)

    @property
    def layer_key(self):
        return f'{self.node_dim}_{self.cond_dim}_{self.edge_dim}'

    def forward(self, graph):
        if self.method == 'none':
            return graph
        elif self.method == 'share':
            e_feat_l = graph.edge.feat_layers[self.layer_key]
            return e_feat_l(graph)
        b_num, n_num = graph.batch_num, graph.node_num
        node_feats = self.n_proj(graph.node.node_feats('cat', self.drop_l))
        n_join_feats = graph.edge.combine_node_attr(node_feats, self.n2n)

        if graph.edge.geo_layer is None:
            graph.edge.geo_layer = self.e_geo_linear
        e_geo_feats = graph.edge.geo_layer(graph.edge.geo_feats())
        join_feats = torch.cat((n_join_feats, e_geo_feats), dim=-1)

        if self.n2c == 'film':
            e_feats = self.n2c_fusion(join_feats, graph.edge.expand_cond_attr(graph.cond_feats))
        else:
            raise NotImplementedError()
        graph.edge.edge_attrs['feats'] = EdgeAttr('feats', e_feats, graph.edge.init_op())

        if self.layer_key not in graph.edge.feat_layers:
            graph.edge.feat_layers[self.layer_key] = self
        return graph


class EdgeWeight(nn.Module):
    def __init__(self,
                 edge_dim: int,
                 method: str,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.edge_dim = edge_dim
        self.score_method, self.norm_method, self.reduce_size = pt.str_split(method, '_')
        if self.score_method == 'share':
            self.score_l = None
        elif self.score_method == 'linear':
            self.score_l = nn.Sequential(
                nn.Dropout(dropout),
                nn.utils.weight_norm(nn.Linear(edge_dim, edge_dim//2)),
                nn.ReLU(),
                nn.utils.weight_norm(nn.Linear(edge_dim//2, 1))
            )
        else:
            raise NotImplementedError()

    @property
    def layer_key(self):
        return f'{self.edge_dim}'

    def forward(self, graph: Graph):
        edge = graph.edge
        score_l = self.score_l or edge.score_layers[self.layer_key]
        if self.layer_key not in edge.score_layers:
            edge.score_layers[self.layer_key] = self.score_l

        e_feats = edge.edge_attrs['feats']
        e_scores = EdgeAttr('scores', score_l(e_feats.value), e_feats.op)
        edge.edge_attrs['scores'] = e_scores
        e_weights = e_scores.op.norm_attr(e_scores.value, self.norm_method)
        topk_op = edge.topk_op(e_weights, self.reduce_size, keep_self=True)
        topk_weights = topk_op.attr_process(e_weights)
        graph.edge.edge_attrs['weights'] = EdgeAttr('weights', topk_weights, topk_op)
        return graph


class EdgeParam(nn.Module):
    def __init__(self,
                 edge_dim: int,
                 out_dim: int,
                 method: str,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.method = method
        if method in ('share', 'none'):
            return
        else:
            self.e_param_l = nn.Sequential(
                nn.Dropout(dropout),
                nn.utils.weight_norm(nn.Linear(edge_dim, out_dim//2)),
                nn.ReLU(),
                nn.utils.weight_norm(nn.Linear(out_dim//2, out_dim)),
                nn.Tanh()
            )

    @property
    def layer_key(self):
        return f'{self.edge_dim}_{self.out_dim}'

    def forward(self, graph: Graph):
        if self.method == 'none':
            return graph
        elif self.method == 'share':
            e_param_l = graph.edge.param_layers[self.layer_key]
            return e_param_l(graph)
        if self.layer_key not in graph.edge.param_layers:
            graph.edge.param_layers[self.layer_key] = self

        e_feats = graph.edge_attrs['feats']
        e_params = EdgeAttr('params', self.e_param_l(e_feats.value), e_feats.op)
        e_weights = graph.edge_attrs['weights']
        e_params = e_weights.op.attr_process(e_params)
        graph.edge.edge_attrs['params'] = e_params
        return graph


class NodeFeat(nn.Module):
    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 method: str,
                 dropout: float = 0.,
                 ):
        super().__init__()
        self.method = method
        self.node_dim = node_dim
        self.cond_dim = cond_dim
        self.out_dim = out_dim
        if method in ('none', 'share'):
            return
        elif self.method == 'film':
            self.feat_l = FilmFusion(node_dim+4, cond_dim, out_dim)
        elif self.method == 'linear':
            self.feat_l = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(node_dim, out_dim//2)),
                nn.utils.weight_norm(nn.Linear(out_dim//2, out_dim)),
                nn.ReLU()
            )
        elif self.method == 'catLinear':
            self.q_feat_l = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(cond_dim, out_dim//2))
            )
            self.feat_l = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(node_dim+out_dim//2, out_dim//2)),
                nn.utils.weight_norm(nn.Linear(out_dim // 2, out_dim)),
                nn.ReLU()
            )
        elif self.method == 'mulLinear':
            self.q_feat_l = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(cond_dim, out_dim))
            )
            self.node_feat_l = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(node_dim, out_dim))
            )
            self.feat_l = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(out_dim, out_dim//2)),
                nn.utils.weight_norm(nn.Linear(out_dim // 2, out_dim)),
                nn.ReLU()
            )
        else:
            raise NotImplementedError()
        self.drop_l = nn.Dropout(dropout)


    @property
    def layer_key(self):
        return f'{self.node_dim}_{self.cond_dim}'

    def forward(self, graph: Graph):
        if self.method == 'share':
            n_feat_l = graph.node.feat_layers[self.layer_key]
            return n_feat_l(graph)
        if self.layer_key not in graph.node.feat_layers:
            graph.node.feat_layers[self.layer_key] = self

        node_feats = graph.node.node_feats('cat', self.drop_l)

        if self.method == 'film':
            node_feats = self.feat_l(node_feats, graph.cond_feats)
        elif self.method == 'linear':
            node_feats = self.feat_l(node_feats)
        elif self.method == 'catLinear':
            q_feats = self.q_feat_l(graph.cond_feats).unsqueeze(1)
            node_feats = self.feat_l(torch.cat((node_feats, q_feats.expand(-1, node_feats.size(1), -1)), dim=-1))
        elif self.method == 'mulLinear':
            q_feats = self.q_feat_l(graph.cond_feats).unsqueeze(1)
            node_feats = self.node_feat_l(node_feats)
            node_feats = self.feat_l(q_feats*node_feats)
        else:
            raise NotImplementedError()

        graph.node.update_feats(node_feats)
        return graph



