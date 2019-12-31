import sys
sys.path.append('./')
import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
from cs_gcn.models import CGCNModel
from pytorch_lightning.logging import TestTubeLogger
import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np


def get_args():
    parser = HyperOptArgumentParser()
    parser.add_argument('--work_dir', metavar='DIR', default="./work_dir", type=str, help='path to save output')
    parser.add_argument('--name', type=str)
    parser.add_argument('--gpus', type=str, default='7', help='how many gpus')
    parser.add_argument('--dist_bd', type=str, default='dp',  choices=('dp', 'ddp', 'ddp2'),
                               help='supports three options dp, ddp, ddp2')
    parser.add_argument('--use_16bit', dest='use_16bit', action='store_true',
                               help='if true uses 16 bit precision')
    parser.add_argument('--eval', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--load_mem', action='store_true')
    parser.add_argument('--track_grad_norm', action='store_true')

    parser = CGCNModel.add_model_specific_args(parser)
    return parser.parse_args()


def init_seed(params):
    random.seed(params.seed)
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    torch.cuda.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def auto_set_name(params):
    if params.name is None:
        params.name = f'gpu^{params.gpu_num}_batchSize^{params.batch_size}_lr^{params.lr}'
    else:
        params.name = f'{params.name}_gpu^{params.gpu_num}_batchSize^{params.batch_size}_lr^{params.lr}'
    if params.use_sched:
        params.name = f'{params.name}_sched^{params.sched_factor}'
    params.name = f'{params.name}_iter^{params.iter_num}_filmOrder^{params.film_order}_qDrop^{params.q_drop}_nDrop^{params.n_drop}'
    if params.use_layerNorm:
        params.name = f'{params.name}_layerNorm'
    print(f'the name is {params.name}')


def main(params, gpus=None, results_dict=None):
    init_seed(params)
    params.gpu_num = len(params.gpus.split(','))
    params.dataset = 'gqa_graph'
    if params.work_dir == './work_dir':
        params.work_dir = params.work_dir + f'/{params.dataset}'
    if not params.track_grad_norm:
        params.track_grad_norm = -1
    else:
        params.track_grad_norm = 1
    params.lr = params.lr * params.gpu_num
    params.batch_size = params.batch_size * params.gpu_num
    auto_set_name(params)

    logger = TestTubeLogger(params.work_dir, name=params.name)

    model = CGCNModel(params)

    trainer = pl.Trainer(
        logger=logger,
        default_save_path=params.work_dir,
        gpus=params.gpus,
        max_nb_epochs=params.epochs,
        distributed_backend=params.dist_bd,
        use_amp=params.use_16bit,
        # nb_sanity_val_steps=0,
        # val_check_interval=0.01,
        # val_percent_check=0.001,
        # train_percent_check=0.001,
        early_stop_callback=False,
        max_epochs=params.epochs,
        track_grad_norm=params.track_grad_norm,
        # log_gpu_memory='all',
    )




if __name__ == '__main__':
    params = get_args()
    main(params)






