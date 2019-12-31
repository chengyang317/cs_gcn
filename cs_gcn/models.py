import pytorch_lightning as pl
import pt_pack as pt
from torch.utils.data import DataLoader
from .layers import *
from test_tube import HyperOptArgumentParser
import torch.nn as nn
from .graph import Graph
import torch
import torch.optim as torch_optim
import torch.optim.lr_scheduler as lr_sched
from .criterions import GqaCrossEntropy
import numpy as np
import pytorch_warmup as warmup
from .lr import WarmupScheduler


class CGCNModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.hparams = params
        vocab_num, answer_num = self.vocab_num, self.answer_num
        self.hparams.vocab_num, self.hparams.answer_num = vocab_num, answer_num
        self.q_layer = CgsQLayer(vocab_num, hid_dim=params.q_hid_dim, dropout=params.q_drop, padding_idx=params.padding_idx)

        self.g_stem_l = GraphStemLayer.build(params)

        g_conv_layers = []
        for step in range(params.iter_num):
            g_conv_layers.append(
                GraphConvLayer.build(params)
            )
        self.g_conv_layers = nn.ModuleList(g_conv_layers)

        n_hid_dim = params.stem_out_dim
        v_dim = (n_hid_dim * 2 if params.pool_method == 'mix' else n_hid_dim) * params.iter_num
        self.g_cls_l = GraphClsLayer(v_dim, params.q_hid_dim, answer_num, params.cls_method, params)

        if params.dataset == 'vqa2_cp':
            self.criterion = pt.Vqa2CrossEntropy()
        elif params.dataset in ('gqa_lcgn', 'gqa_graph'):
            self.criterion = GqaCrossEntropy()
        else:
            NotImplementedError()
        self.warmup_sched = None
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
            q_vocab = self.train_dataloader().dataset.question_vocab
            self.q_layer.embedding.weight.data.copy_(q_vocab.glove_embed('glove.6B.300d'))
        elif self.hparams.dataset == 'gqa_lcgn':
            import numpy as np
            embed_file = self.hparams.data_dir + '/gloves_gqa_no_pad.npy'
            self.q_layer.embedding.weight.data[1:].copy_(torch.Tensor(np.load(embed_file)))
        elif self.hparams.dataset == 'gqa_graph':
            q_vocab = self.train_dataloader().dataset.question_vocab
            self.q_layer.embedding.weight.data[1:].copy_(torch.Tensor(q_vocab.embed_init))
        else:
            raise NotImplementedError()

    def forward(self, obj_feats, obj_boxes, q_labels, q_nums, obj_masks=None, obj_nums=None):
        q_feats = self.q_layer(q_labels, q_nums)
        graph = Graph(obj_feats, obj_boxes, obj_masks, obj_nums, self.hparams.g_init_method, cond_feats=q_feats, params=self.hparams)
        graph = self.g_stem_l(graph)
        pool_feats = list()
        for conv_l in self.g_conv_layers:
            graph.clear()
            graph = conv_l(graph)
            pool_feats.append(graph.pool_feats(self.hparams.pool_method))
        v_feats = torch.cat(pool_feats, dim=-1)
        logits = self.g_cls_l(v_feats, graph.cond_feats)
        return logits

    def get_inputs(self, data_batch):
        if self.hparams.dataset == 'gqa_lcgn':
            return {key: data_batch.get(key, None) for key in ('obj_feats', 'obj_boxes', 'q_labels', 'q_nums', 'obj_masks')}, {'a_labels': data_batch['a_labels']}
        elif self.hparams.dataset == 'vqa2_cp':
            obj_feats, obj_boxes = data_batch['img_obj_feats'].split((2048, 4), dim=-1)
            return {'obj_feats': obj_feats, 'obj_boxes': obj_boxes, 'q_labels': data_batch['q_labels'], 'q_nums': data_batch['q_lens']}, \
                   {'a_label_scores': data_batch['a_label_scores'], 'a_label_counts': data_batch['a_label_counts']}
        elif self.hparams.dataset == 'gqa_graph':
            return {'obj_feats': data_batch['img_obj_feats'], 'obj_boxes': data_batch.get('img_obj_boxes', None),
                    'q_labels': data_batch['q_labels'], 'q_nums': data_batch['q_lens'], 'obj_nums': data_batch['img_obj_nums'],
                    }, {'a_labels': data_batch['a_labels']}
        else:
            raise NotImplementedError()

    def training_step(self, data_batch, batch_nb):
        forward_inputs, criterion_inputs = self.get_inputs(data_batch)
        logits = self.forward(**forward_inputs)
        # del forward_inputs
        loss, acc = self.criterion(logits, **criterion_inputs)
        # del criterion_inputs
        output = {'loss': loss, 'acc': acc}
        return output

    def training_end(self, outputs):
        loss = outputs['loss'].mean()
        acc = outputs['acc'].mean()
        output = {'loss': loss, 'progress_bar': {'train_acc': acc},
                  'log': {'train_loss': loss, 'train_acc': acc}
                  }
        return output

    def validation_step(self, data_batch, batch_nb):
        forward_inputs, criterion_inputs = self.get_inputs(data_batch)
        logits = self.forward(**forward_inputs)
        # del forward_inputs
        loss, acc = self.criterion(logits, **criterion_inputs)
        # del criterion_inputs
        return {'val_loss': loss, 'val_acc': acc}

    def validation_end(self, outputs):
        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss_mean += output['val_loss'].mean()
            val_acc_mean += output['val_acc'].mean()

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}

        # show val_loss and val_acc in progress bar but only log val_loss
        results = {
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
            'val_loss': val_loss_mean,
            'val_acc': val_acc_mean
        }
        self.logger.experiment.add_scalar('val_epoch_acc', val_acc_mean.item(), self.current_epoch)
        return results

    def configure_optimizers(self):
        params = self.hparams
        if params.optimizer == 'sgd':
            optimizer = torch_optim.SGD(self.parameters(), lr=params.lr, weight_decay=params.weight_decay, momentum=0.9)
        elif params.optimizer == 'adam':
            optimizer = torch_optim.Adam(self.parameters(), lr=params.lr, weight_decay=params.weight_decay)
        elif params.optimizer == 'adabound':
            import adabound
            optimizer = adabound.AdaBound(self.parameters(), lr=params.lr, final_lr=params.lr*10,
                                          weight_decay=params.weight_decay)
        else:
            raise NotImplementedError()

        if params.sched == 'plat':
            sched = lr_sched.ReduceLROnPlateau(optimizer, patience=0, factor=params.sched_factor, verbose=True, min_lr=0.0004)
            return [optimizer], [sched]
        elif self.hparams.sched == 'sgdr':
            sched = lr_sched.CosineAnnealingWarmRestarts(optimizer, self.hparams.sched_factor)
            return [optimizer], [sched]
        elif self.hparams.sched == 'step':
            sched = lr_sched.MultiStepLR(optimizer, milestones=[3, 6], gamma=0.3)
            return [optimizer], [sched]
        elif params.sched == 'none':
            return optimizer
        else:
            raise NotImplementedError()

    @pl.data_loader
    def train_dataloader(self):
        if self.hparams.dataset == 'vqa2_cp':
            dataset = pt.GraphVqa2CpDataset()
            loader = DataLoader(dataset, self.hparams.batch_size, True, num_workers=self.hparams.num_workers,
                                collate_fn=dataset.collate_fn)
        elif self.hparams.dataset == 'gqa_lcgn':
            from datasets.lcgn import load_train_data
            loader = load_train_data(self.hparams)
        elif self.hparams.dataset == 'gqa_graph':
            dataset = pt.GraphGqaDataset(gpu_num=len(self.hparams.gpus.split(',')), use_filter=True)
            loader = DataLoader(dataset, self.hparams.batch_size, True, num_workers=self.hparams.num_workers,
                                collate_fn=dataset.collate_fn, worker_init_fn=lambda x: np.random.seed(self.hparams.seed+x))
        else:
            raise NotImplementedError()

        return loader

    @pl.data_loader
    def val_dataloader(self):
        if self.hparams.dataset == 'vqa2_cp':
            dataset = pt.GraphVqa2CpDataset(split='test',)
            loader = DataLoader(dataset, self.hparams.batch_size, False, num_workers=self.hparams.num_workers,
                                collate_fn=dataset.collate_fn)
        elif self.hparams.dataset == 'gqa_lcgn':
            from datasets.lcgn import load_eval_data
            loader = load_eval_data(self.hparams)
        elif self.hparams.dataset == 'gqa_graph':
            dataset = pt.GraphGqaDataset(split='val', gpu_num=len(self.hparams.gpus.split(',')), use_filter=True)
            loader = DataLoader(dataset, self.hparams.batch_size, False, num_workers=self.hparams.num_workers,
                                collate_fn=dataset.collate_fn, worker_init_fn=lambda x: np.random.seed(self.hparams.seed + x))
        else:
            raise NotImplementedError()

        return loader

    @property
    def vocab_num(self):
        if self.hparams.dataset == 'gqa_lcgn':
            return self.train_dataloader().vocab_num
        elif self.hparams.dataset in ('vqa2_cp', 'gqa_graph'):
            return len(self.train_dataloader().dataset.question_vocab)
        else:
            raise NotImplementedError()

    @property
    def answer_num(self):
        if self.hparams.dataset == 'gqa_lcgn':
            return self.train_dataloader().answer_num
        elif self.hparams.dataset in ('vqa2_cp', 'gqa_graph'):
            return len(self.train_dataloader().dataset.answer_vocab)
        else:
            raise NotImplementedError()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser], add_help=False)
        parser.opt_list('--q_drop', default=0.35, options=[0.25, 0.3, 0.35, 0.15, 0.4, 0.1], type=float, tunable=False)
        parser.add_argument('--q_hid_dim', default=1024, type=int)
        parser.add_argument('--padding_idx', type=int, default=0)

        parser.add_argument('--net_init', default='k_u')
        parser.opt_list('--iter_num', default=3, options=[1, 2, 3, 4, 5], type=int)
        parser.add_argument('--g_init_method', default='full', type=str)

        parser.add_argument('--stem_in_dim', default=2048, type=int)
        parser.add_argument('--stem_out_dim', default=1024, type=int)
        parser.opt_list('--stem_norm', default='custom', options=['custom', 'layer', 'weight'], type=str, tunable=False)
        parser.opt_list('--stem_orders', default='dlna', options=['dlna', 'dnla'], type=str, tunable=False)
        parser.opt_list('--stem_drop', default=0.35, options=[0.3, 0.35, 0.4], type=int)
        parser.opt_list('--stem_method', default='linear', options=['linear', 'film', 'double_linear'], type=str)
        parser.add_argument('--stem_use_act', action='store_true')

        parser.opt_list('--e_f_norm', default='weight', options=['custom', 'layer', 'weight'], type=str, tunable=False)
        parser.opt_list('--e_f_orders', default='dlna', options=['dlna', 'dnla'], type=str, tunable=False)
        parser.opt_list('--e_f_drop', default=0.35, options=[0.3, 0.35, 0.4], type=int)
        parser.add_argument('--e_dim', default=512, type=int)
        parser.opt_list('--e_f_method', default='mul_film', options=['mul_film', 'cat_film'], type=str)
        parser.add_argument('--e_f_use_nGeo', action='store_true', default=True)

        parser.opt_list('--e_w_method', default='linear_softmax_8',
                        options=['linear_softmax_10', 'linear_softmax_12'], type=str)
        parser.opt_list('--e_w_norm', default='weight', options=['custom', 'layer', 'weight'], type=str,
                        tunable=False)
        parser.opt_list('--e_w_orders', default='dlna', options=['dlna', 'dnla'], type=str, tunable=False)
        parser.opt_list('--e_w_drop', default=0.35, options=[0.3, 0.35, 0.4], type=int)

        parser.opt_list('--e_p_method', default='linear', options=['linear', 'share'], type=str)
        parser.opt_list('--e_p_norm', default='weight', options=['custom', 'layer', 'weight'], type=str,
                        tunable=False)
        parser.opt_list('--e_p_orders', default='dlna', options=['dlna', 'dnla'], type=str, tunable=False)
        parser.opt_list('--e_p_drop', default=0.35, options=[0.3, 0.35, 0.4], type=int)
        parser.opt_list('--e_p_act', default='relu', options=['relu', 'swish', 'elu'], type=str, tunable=False)

        parser.opt_list('--n_f_method', default='film', options=['linear', 'film'], type=str)
        parser.opt_list('--n_f_drop', default=0.35, options=[0.3, 0.35, 0.4], type=int)

        parser.opt_list('--n_geo_method', default='cat', options=['cat', 'sum', 'linear_cat'], type=str)
        parser.add_argument('--n_geo_reuse', action='store_true', default=True)
        parser.add_argument('--n_geo_dim', default=64, type=int)
        parser.add_argument('--n_geo_out_dim', default=64, type=int)
        parser.opt_list('--n_geo_norm', default='weight', options=['custom', 'layer', 'weight'], type=str, tunable=False)
        parser.opt_list('--n_geo_orders', default='lna', options=['lna', 'nla'], type=str, tunable=False)

        parser.opt_list('--e_geo_method', default='linear', options=['linear', 'cat'], type=str)
        parser.add_argument('--e_geo_dim', default=128, type=int)
        parser.add_argument('--e_geo_out_dim', default=128, type=int)
        parser.add_argument('--e_geo_reuse', action='store_true', default=True)
        parser.add_argument('--e_geo_aug', action='store_true')
        parser.opt_list('--e_geo_norm', default='weight', options=['custom', 'layer', 'weight'], type=str,
                        tunable=False)
        parser.opt_list('--e_geo_orders', default='lna', options=['lna', 'nla'], type=str, tunable=False)

        parser.opt_list('--cls_method', default='linear', options=['linear'], type=str)
        parser.opt_list('--cls_norm', default='weight', options=['custom', 'layer', 'weight'], type=str, tunable=False)
        parser.opt_list('--cls_orders', default='dlna', options=['dlna', 'dnla'], type=str, tunable=False)
        parser.opt_list('--cls_drop', default=0.35, options=[0.3, 0.35, 0.4], type=int)
        parser.opt_list('--cls_act', default='relu', options=['relu', 'swish'], type=str, tunable=False)

        parser.opt_list('--f_c_norm', type=str, default='weight', options=['weight', 'custom', 'layer'])
        parser.opt_list('--f_c_drop', default=0., options=[0.3, 0.35, 0.4], type=int)
        parser.opt_list('--f_c_orders', default='dln', options=['dln', 'dnl'], type=str, tunable=False)
        parser.opt_list('--f_x_norm', default='layer', options=['custom', 'layer', 'weight'], type=str,
                        tunable=False)
        parser.opt_list('--f_x_orders', default='dln', options=['dln', 'dnl'], type=str, tunable=False)
        parser.add_argument('--f_x_norm_affine', action='store_true')
        parser.opt_list('--f_act', default='relu', options=['relu', 'swish', 'elu'], type=str, tunable=False)

        parser.opt_list('--pool_method', default='mix', options=['mix', 'mean', 'max'], type=str)
        parser.opt_list('--lr', default=2.5e-4, type=float, options=[2.5e-4, 3.5e-4], tunable=False)
        parser.opt_list('--sched_factor', default=0.5,  type=float, options=[0.1, 0.8, 0.6, 0.3], tunable=False)

        parser.opt_list('--optimizer', type=str, default='adam', options=['adam', 'sgd'])
        parser.opt_list('--sched', type=str, default='plat', options=['plat', 'cyclic', 'sgdr'])
        parser.opt_list('--dataset', default='vqa2_cp', type=str, options=['vaq2_cp', 'gqa_lcgn'])
        parser.add_argument('--epochs', default=12, type=int)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--num_workers', default=5, type=int)
        parser.add_argument('--grad_clip', default=0., type=float)
        parser.add_argument('--weight_decay', default=0., type=float)
        parser.add_argument('--use_warmup', action='store_true')
        return parser

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if self.hparams.use_warmup:
            if self.warmup_sched is None:
                print('using warmup')
                if self.hparams.optimizer == 'adam':
                    self.warmup_sched = WarmupScheduler(by_epoch=False, warmup='linear', warmup_iters=150, warmup_ratio=0.3/3)
                    # self.warmup_sched = warmup.RAdamWarmup(optimizer)
                    # self.warmup_sched = warmup.LinearWarmup(optimizer, warmup_period=10)
                else:
                    # self.warmup_sched = warmup.LinearWarmup(optimizer, warmup_period=10)
                    self.warmup_sched = WarmupScheduler(by_epoch=False, warmup='linear', warmup_iters=150, warmup_ratio=0.3/3)
                self.warmup_sched.before_run(optimizer)
            self.warmup_sched.step(optimizer)
        optimizer.step()
        optimizer.zero_grad()





