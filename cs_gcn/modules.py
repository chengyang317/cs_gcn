import torch.nn as nn
import pt_pack as pt
import torch
from .graph.edge import EdgeAttr, EdgeTopK
from .graph import Graph
import pt_pack as pt


__all__ = ['FilmFusion', 'EdgeFeat', 'EdgeWeight', 'EdgeParam', 'NodeFeat']


class FilmFusion(nn.Module):
    def __init__(self,
                 in_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 norm_type='layer',
                 dropout: float = 0.,
                 act_type: str = 'relu',
                 params=None,
                 ):
        super().__init__()
        self.cond_proj = pt.PtLinear(cond_dim, out_dim*2, orders=params.f_c_orders, norm=params.f_c_norm,
                                     drop=params.f_c_drop)
        self.x_linear = pt.PtLinear(in_dim, out_dim, orders=params.f_x_orders, drop=dropout, norm=params.f_x_norm,
                                    norm_affine=False, bias=False)
        self.cond_l = pt.Cond()
        self.act_l = pt.Act(params.f_act)

    def forward(self, x, cond, batch_ids=None):
        gamma, beta = self.cond_proj(cond).chunk(2, dim=-1)
        gamma += 1.
        if batch_ids is not None:
            beta, gamma = beta[batch_ids], gamma[batch_ids]
        x = self.x_linear(x)
        x = self.cond_l(x, gamma, beta)
        x = self.act_l(x)
        return x


class EdgeFeat(nn.Module):
    def __init__(self,
                 node_dim,
                 cond_dim,
                 edge_dim,
                 method: str,
                 params=None,
                 ):
        super().__init__()
        self.method = method
        self.node_dim = node_dim
        self.cond_dim = cond_dim
        self.edge_dim = edge_dim
        self.params = params
        self.use_n_geo = params.e_f_use_nGeo
        if self.method in ('share', 'none'):
            return
        self.n2n, self.n2c = method.split('_')

        if params.n_geo_method in ('sum', 'none') or not self.use_n_geo:
            n_dim = node_dim
        else:
            n_dim = node_dim + params.n_geo_out_dim

        orders = params.e_f_orders if self.n2n == 'mul' else params.e_f_orders
        self.n_proj = pt.PtLinear(n_dim, edge_dim, drop=params.e_f_drop, norm=params.e_f_norm, orders=orders)

        if params.e_geo_method == 'linear':
            orders = params.e_geo_orders if self.n2n == 'mul' else params.e_geo_orders
            self.e_geo_linear = pt.PtLinear(params.e_geo_dim, params.e_geo_out_dim, orders=orders, norm=params.e_geo_norm)
        else:
            self.e_geo_linear = None
        join_dim = edge_dim + params.e_geo_out_dim if self.n2n in ('sum', 'mul', 'max') else edge_dim * 2 + params.e_geo_out_dim
        self.n2c_fusion = FilmFusion(join_dim, cond_dim, edge_dim, params=params)

    @property
    def layer_key(self):
        return f'{self.node_dim}_{self.cond_dim}_{self.edge_dim}'

    def forward(self, graph):
        node, edge = graph.node, graph.edge
        if self.method == 'none':
            return graph
        elif self.method == 'share':
            e_feat_l = graph.edge.feat_layers[self.layer_key]
            return e_feat_l(graph)
        if self.layer_key not in graph.edge.feat_layers:
            graph.edge.feat_layers[self.layer_key] = self

        node_feats = node.node_feats() if self.use_n_geo else node.feats
        node_feats = self.n_proj(node_feats)
        n_join_feats = graph.edge.combine_node_attr(node_feats, self.n2n)
        e_geo_feats = edge.geo_feats(node, self.e_geo_linear)
        join_feats = torch.cat((n_join_feats, e_geo_feats), dim=-1)

        if self.n2c == 'film':
            e_feats = self.n2c_fusion(join_feats, graph.cond_feats, edge.batch_ids(node))
        else:
            raise NotImplementedError()
        e_feats = EdgeAttr('feats', e_feats, graph.edge.name)
        return e_feats


class EdgeWeight(nn.Module):
    def __init__(self,
                 edge_dim: int,
                 method: str,
                 params=None,
                 ):
        super().__init__()
        self.edge_dim = edge_dim
        self.params = params
        self.score_method, self.norm_method, self.reduce_size = pt.str_split(method, '_')
        if self.score_method == 'share':
            self.score_l = None
        elif self.score_method == 'linear':
            self.score_l = nn.Sequential(
                pt.PtLinear(edge_dim, edge_dim//2, norm=params.e_w_norm, drop=params.e_w_drop, orders=params.e_w_orders),
                pt.PtLinear(edge_dim//2, 1, norm='weight', orders='ln')  # for the ln orders, layer norm is terrible
            )
        else:
            raise NotImplementedError()

    @property
    def layer_key(self):
        return f'{self.edge_dim}'

    def forward(self, graph: Graph, e_feats):
        edge = graph.edge
        score_l = self.score_l or edge.score_layers[self.layer_key]
        if self.layer_key not in edge.score_layers:
            edge.score_layers[self.layer_key] = self.score_l

        e_scores = EdgeAttr('scores', score_l(e_feats.value), e_feats.op_name)
        op = edge.load_op(e_scores.op_name)
        e_weights = op.norm_attr(e_scores.value, self.norm_method)
        topk_op = EdgeTopK(e_weights, self.reduce_size, edge, keep_self=True)
        topk_weights = topk_op.attr_process(e_weights)
        e_weights = EdgeAttr('weights', topk_weights, topk_op.name)
        return e_weights


class EdgeParam(nn.Module):
    def __init__(self,
                 edge_dim: int,
                 out_dim: int,
                 method: str,
                 params=None,
                 ):
        super().__init__()
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.method = method
        self.params = params
        if method in ('share', 'none'):
            return
        else:
            self.params_l = nn.Sequential(
                pt.PtLinear(edge_dim, out_dim//2, norm=params.e_p_norm, drop=params.e_p_drop, orders=params.e_p_orders, act=params.e_p_act),
                pt.PtLinear(out_dim//2, out_dim, norm=params.e_p_norm, orders='lna', act='tanh')
            )

    @property
    def layer_key(self):
        return f'{self.edge_dim}_{self.out_dim}'

    def forward(self, graph: Graph, e_feats):
        if self.method == 'none':
            return graph
        elif self.method == 'share':
            e_param_l = graph.edge.param_layers[self.layer_key]
            return e_param_l(graph)
        if self.layer_key not in graph.edge.param_layers:
            graph.edge.param_layers[self.layer_key] = self

        e_params = EdgeAttr('params', self.params_l(e_feats.value), e_feats.op_name)
        return e_params


class NodeFeat(nn.Module):
    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 method: str,
                 params=None,
                 ):
        super().__init__()
        self.method = method
        self.node_dim = node_dim
        self.cond_dim = cond_dim
        self.out_dim = out_dim
        self.params = params
        if params.n_geo_method in ('sum', 'none'):
            n_dim = node_dim
        else:
            n_dim = node_dim + params.n_geo_out_dim
        if method in ('none', 'share'):
            return
        elif self.method == 'film':
            self.feat_l = FilmFusion(n_dim, cond_dim, out_dim, params=params, dropout=params.n_f_drop)
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

    @property
    def layer_key(self):
        return f'{self.node_dim}_{self.cond_dim}'

    def forward(self, graph: Graph):
        node, edge = graph.node, graph.edge
        if self.method == 'share':
            n_feat_l = graph.node.feat_layers[self.layer_key]
            return n_feat_l(graph)
        if self.layer_key not in graph.node.feat_layers:
            graph.node.feat_layers[self.layer_key] = self

        node_feats = graph.node.node_feats()

        if self.method == 'film':
            node_feats = self.feat_l(node_feats, graph.cond_feats, node.batch_ids)
        elif self.method == 'linear':
            node_feats = self.feat_l(node_feats)
        elif self.method == 'catLinear':
            q_feats = self.q_feat_l(cond_feats).unsqueeze(1)
            node_feats = self.feat_l(torch.cat((node_feats, q_feats.expand(-1, node_feats.size(1), -1)), dim=-1))
        elif self.method == 'mulLinear':
            q_feats = self.q_feat_l(cond_feats).unsqueeze(1)
            node_feats = self.node_feat_l(node_feats)
            node_feats = self.feat_l(q_feats*node_feats)
        else:
            raise NotImplementedError()

        return node_feats



