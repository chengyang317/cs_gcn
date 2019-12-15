import sys
sys.path.append('./')
import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
from cs_gcn.models import CGCNModel
from pytorch_lightning.logging import TestTubeLogger


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

    parser = CGCNModel.add_model_specific_args(parser)
    return parser.parse_args()


def init_seed(params):
    import torch
    import numpy as np
    import random
    import torch.backends.cudnn as cudnn
    random.seed(params.seed)
    torch.manual_seed(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)
    cudnn.deterministic = True


def main(params, gpus=None, results_dict=None):
    init_seed(params)
    params.padding_idx = 0

    if params.work_dir == './work_dir':
        params.work_dir = params.work_dir + f'/{params.dataset}'
    if params.name is not None:
        params.name = f'{params.name}_lr^{params.lr}'

    logger = TestTubeLogger(params.work_dir, name=params.name)
    model = CGCNModel(params)

    gpu_num = len(params.gpus.split(','))
    params.lr = params.lr * gpu_num

    trainer = pl.Trainer(
        logger=logger,
        default_save_path=params.work_dir,
        gpus=params.gpus,
        max_nb_epochs=params.epochs,
        distributed_backend=params.dist_bd,
        use_amp=params.use_16bit,
        # nb_sanity_val_steps=0,
        # val_check_interval=0.01,
        # val_percent_check=0.0001,
        # train_percent_check=0.001,
        early_stop_callback=False,
        # log_gpu_memory='all',
        max_epochs=params.epochs
    )
    if params.evaluate:
        trainer.run_evaluation()
    else:
        trainer.fit(model)


if __name__ == '__main__':
    params = get_args()
    main(params)
    # params.optimize_parallel_gpu(main, gpu_ids=['5', '6', '7'], )
    for trial_hparams in params.trials(10):
        main(params)






