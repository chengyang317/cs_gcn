import sys
sys.path.append('./')
import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
from cs_gcn.models import CGCNModel
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np
import copy


def get_args():
    parser = HyperOptArgumentParser()
    parser.add_argument('--work_dir', metavar='DIR', default="./work_dir", type=str, help='path to save output')
    parser.add_argument('--proj_name', type=str)
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
        params.name = f'gpu^{params.gpu_num}'
    else:
        params.name = f'{params.name}_gpu^{params.gpu_num}'
    params.name = f'{params.name}_bs^{params.batch_size}_optim^{params.optimizer}_lr^{params.lr}_sd^{params.sched}^{params.sched_factor}'
    params.name = f'{params.name}_nGeo^{params.n_geo_method}^{params.n_geo_dim}^{params.n_geo_out_dim}^{params.n_geo_reuse}^{params.n_geo_norm}'
    params.name = f'{params.name}_eGeo^{params.e_geo_method}^{params.e_geo_dim}^{params.e_geo_out_dim}^{params.e_geo_reuse}^{params.e_geo_norm}'
    params.name = f'{params.name}_stem^{params.stem_method}^{params.stem_norm}^{params.stem_orders}'
    params.name = f'{params.name}_eF^{params.e_f_norm}^{params.e_f_orders}'
    params.name = f'{params.name}_eW^{params.e_w_norm}^{params.e_w_orders}'
    params.name = f'{params.name}_eP^{params.e_p_norm}^{params.e_p_orders}'
    params.name = f'{params.name}_cls^{params.cls_method}^{params.cls_norm}^{params.cls_orders}'
    params.name = f'{params.name}_film^{params.f_c_norm}^{params.f_x_norm}'
    print(f'the name is {params.name}')


def main(params, gpus=None, results_dict=None):
    init_seed(params)
    params.gpu_num = len(params.gpus.split(','))
    params.dataset = 'gqa_graph'
    if params.proj_name is None:
        params.proj_name = params.dataset
    params.work_dir = params.work_dir + f'/{params.proj_name}'

    if not params.track_grad_norm:
        params.track_grad_norm = -1
    else:
        params.track_grad_norm = 1
    params.lr = params.lr * params.gpu_num * (params.batch_size / 64.)
    params.batch_size = params.batch_size * params.gpu_num
    auto_set_name(params)

    logger = TestTubeLogger(params.work_dir, name=params.name)

    model = CGCNModel(params)
    # checkpoint = ModelCheckpoint()

    trainer = pl.Trainer(
        logger=logger,
        default_save_path=params.work_dir,
        gpus=params.gpus,
        distributed_backend=params.dist_bd,
        use_amp=params.use_16bit,
        # nb_sanity_val_steps=0,
        # val_check_interval=0.01,
        # val_percent_check=0.001,
        # train_percent_check=0.001,
        early_stop_callback=False,
        max_epochs=params.epochs,
        # max_epochs=1,
        track_grad_norm=params.track_grad_norm,
        # log_gpu_memory='all',
        # checkpoint_callback=False,
        row_log_interval=100,
        gradient_clip_val=params.grad_clip
    )
    if params.evaluate:
        trainer.run_evaluation()
    else:
        trainer.fit(model)


if __name__ == '__main__':
    params = get_args()
    main(params)

    # new_params = copy.deepcopy(params)
    # new_params.stem_norm = 'custom'
    # new_params.e_f_norm = 'weight'
    # new_params.e_w_norm = 'weight'
    # new_params.e_p_norm = 'weight'
    # new_params.cls_norm = 'weight'
    # # new_params.e_p_act = 'swish'
    # # new_params.name = 'ePActSwish'
    # new_params.f_c_norm = 'weight'
    # new_params.f_x_norm = 'layer'
    # new_params.e_geo_norm = 'weight'
    # new_params.n_geo_norm = 'weight'
    # main(new_params)


    # new_params = copy.deepcopy(params)
    # new_params.stem_norm = 'custom'
    # new_params.e_f_norm = 'custom'
    # new_params.e_w_norm = 'weight'
    # new_params.e_p_norm = 'weight'
    # new_params.cls_norm = 'weight'
    # new_params.cls_act = 'elu'
    # new_params.name = 'clsActElu'
    # new_params.f_c_norm = 'custom'
    # new_params.f_x_norm = 'layer'
    # new_params.e_geo_norm = 'custom'
    # new_params.n_geo_norm = 'weight'
    # main(new_params)
    #
    # new_params = copy.deepcopy(params)
    # new_params.e_f_norm = 'custom'
    # new_params.e_w_norm = 'custom'
    # new_params.e_p_norm = 'custom'
    # new_params.cls_norm = 'custom'
    # new_params.f_c_norm = 'weight'
    # new_params.f_x_norm = 'layer'
    # new_params.e_geo_norm = 'custom'
    # new_params.n_geo_norm = 'weight'
    # main(new_params)





    # for trial_hparams in params.trials(10):
    #     main(trial_hparams)








