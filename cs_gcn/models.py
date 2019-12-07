import pytorch_lightning as pl
import pt_pack as pt
from torch.utils.data import DataLoader
from .layers import *
from test_tube import HyperOptArgumentParser
import torch.nn as nn
from .graph import Graph
import torch
from torch.optim import Adam
from .criterions import GqaCrossEntropy


class CGCNModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.hparams = params
        vocab_num, answer_num = self.train_dataloader().vocab_num, self.train_dataloader().answer_num
        self.hparams.vocab_num, self.hparams.answer_num = vocab_num, answer_num
        self.q_layer = CgsQLayer(vocab_num, hid_dim=params.q_hid_dim, dropout=params.q_drop, padding_idx=params.padding_idx)

        self.g_stem_l = GraphStemLayer(params.n_dim, params.q_hid_dim, params.n_hid_dim, params.n_stem_method, params.n_drop)

        g_conv_layers = []
        for step in range(params.iter_nums):
            g_conv_layers.append(
                GraphConvLayer(params.n_hid_dim, params.q_hid_dim, params.e_dim, params.n_hid_dim, params.e_feat_method,
                               params.e_weight_method, params.e_param_method, params.n_feat_method, params.n_drop)
            )
        self.g_conv_layers = nn.ModuleList(g_conv_layers)

        v_dim = (params.n_hid_dim * 2 if params.pool_method == 'mix' else params.n_hid_dim) * params.iter_nums
        self.g_cls_l = GraphClsLayer(v_dim, params.q_hid_dim, answer_num, params.cls_method, params.n_drop)

        if params.dataset == 'vqa2_cp':
            self.criterion = pt.Vqa2CrossEntropy()
        elif params.dataset in ('gqa_lcgn',):
            self.criterion = GqaCrossEntropy()

        self.reset_parameters()

    def reset_parameters(self):
        if self.hparams.net_init == 'k_u':
            from torch.nn.init import kaiming_uniform_
            func = kaiming_uniform_
        else:
            raise NotImplementedError()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                func(m.weight)
        if self.hparams.dataset == 'vqa2_cp':
            self.q_layer.embedding.weight.data.copy_(self.q_vocab.glove_embed('glove.6B.300d'))
        elif self.hparams.dataset == 'gqa_lcgn':
            import numpy as np
            embed_file = self.hparams.data_dir + '/gloves_gqa_no_pad.npy'
            self.q_layer.embedding.weight.data[1:].copy_(torch.Tensor(np.load(embed_file)))
        else:
            raise NotImplementedError()

    def forward(self, obj_feats, obj_boxes, q_labels, q_nums, obj_masks=None):
        q_feats = self.q_layer(q_labels, q_nums)
        graph = Graph(obj_feats, obj_boxes, node_masks=obj_masks, init_method=self.hparams.g_init_method, cond_feats=q_feats)

        graph = self.g_stem_l(graph)

        pool_feats = list()
        for conv_l in self.g_conv_layers:
            graph.clear()
            graph = conv_l(graph)
            pool_feats.append(graph.pool_feats(self.hparams.pool_method))
        v_feats = torch.cat(pool_feats, dim=-1)
        logits = self.g_cls_l(v_feats, graph.cond_feats)
        return logits

    def training_step(self, data_batch, batch_nb):
        inputs = {key: data_batch.get(key, None) for key in ('obj_feats', 'obj_boxes', 'q_labels', 'q_nums', 'obj_masks')}
        logits = self.forward(**inputs)
        loss, acc = self.criterion(logits, data_batch['a_labels'])
        output = {'loss': loss, 'progress_bar': {'train_acc': acc},
                  'log': {'train_loss': loss, 'train_acc': acc}
                  }
        return output

    def validation_step(self, data_batch, batch_nb):
        inputs = {key: data_batch.get(key, None) for key in ('obj_feats', 'obj_boxes', 'q_labels', 'q_nums', 'obj_masks')}
        logits = self.forward(**inputs)
        loss, acc = self.criterion(logits, data_batch['a_labels'])
        return {'val_loss': loss, 'val_acc': torch.tensor(acc)}

    def validation_end(self, outputs):
        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss_mean += output['val_loss']
            val_acc_mean += output['val_acc']

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean.item(), 'val_acc': val_acc_mean.item()}

        # show val_loss and val_acc in progress bar but only log val_loss
        results = {
            'progress_bar': tqdm_dict,
            'log': {'val_loss': val_loss_mean.item(), 'val_acc': val_acc_mean.item()}
        }
        return results

    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=self.hparams.lr)
        return optim

    @pl.data_loader
    def train_dataloader(self):
        if self.hparams.dataset == 'vqa2_cp':
            dataset = pt.GraphVqa2CpDataset(add_field_name=False)
            loader = DataLoader(dataset, self.hparams.batch_size, True, num_workers=self.hparams.num_workers,
                                collate_fn=dataset.collate_fn)
        elif self.hparams.dataset == 'gqa_lcgn':
            from datasets.lcgn import load_train_data
            loader = load_train_data(self.hparams)
        else:
            raise NotImplementedError()

        return loader

    @pl.data_loader
    def val_dataloader(self):
        if self.hparams.dataset == 'vqa2_cp':
            dataset = pt.GraphVqa2CpDataset(split='test', add_field_name=False)
            loader = DataLoader(dataset, self.hparams.batch_size, False, num_workers=self.hparams.num_workers,
                                collate_fn=dataset.collate_fn)
        elif self.hparams.dataset == 'gqa_lcgn':
            from datasets.lcgn import load_eval_data
            loader = load_eval_data(self.hparams)
        else:
            raise NotImplementedError()

        return loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser], add_help=False)
        parser.opt_list('--q_drop', default=0.2, options=['0.1', '0.2'], type=float)
        parser.add_argument('--q_hid_dim', default=1024, type=int)
        parser.add_argument('--padding_idx', type=int)

        parser.add_argument('--net_init', default='k_u')
        parser.add_argument('--n_dim', default=2048, type=int)
        parser.add_argument('--n_hid_dim', default=1024, type=int)
        parser.add_argument('--e_dim', default=512, type=int)
        parser.opt_list('--n_drop', default=0.2, options=['0.1', '0.2'], type=float)
        parser.opt_list('--n_stem_method', default='linear', options=['linear', 'film'], type=str)
        parser.opt_list('--e_feat_method', default='mul_film', options=['mul_film', 'cat_film'], type=str)
        parser.opt_list('--e_weight_method', default='linear_softmax_8', options=['linear_softmax_8', 'linear_sigmoid_8'], type=str)
        parser.opt_list('--e_param_method', default='linear', options=['linear', 'share'], type=str)
        parser.opt_list('--n_feat_method', default='film', options=['linear', 'film'], type=str)

        parser.add_argument('--g_init_method', default='full', type=str)

        parser.opt_list('--iter_nums', default=3, options=[1, 2, 3, 4, 5], type=int)

        parser.opt_list('--cls_method', default='linear', options=['linear'], type=str)
        parser.opt_list('--pool_method', default='mix', options=['mix', 'mean', 'max'], type=str)

        parser.opt_list('--dataset', default='vqa2_cp', type=str, options=['vaq2_cp', 'gqa_lcgn'])
        parser.add_argument('--epochs', default=20, type=int)
        parser.add_argument('--batch_size', default=20, type=int)
        parser.add_argument('--num_workers', default=0, type=int)
        parser.add_argument('--lr', default=3e-4, type=float)
        return parser