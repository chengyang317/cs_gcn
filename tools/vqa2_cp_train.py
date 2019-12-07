import sys
sys.path.append('./')
import pt_pack as pt
import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
from cs_gcn.models import CGCNModel
from pytorch_lightning.logging import TestTubeLogger
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
import multiprocessing as mp


def get_args():
    parser = HyperOptArgumentParser()
    parser.add_argument('--save-path', metavar='DIR', default="./work_dir", type=str,
                               help='path to save output')
    parser.add_argument('--gpus', type=str, default='0',
                               help='how many gpus')
    parser.add_argument('--dist_backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                               help='supports three options dp, ddp, ddp2')
    parser.add_argument('--use-16bit', dest='use_16bit', action='store_true',
                               help='if true uses 16 bit precision')
    parser.add_argument('--eval', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set')
    parser.add_argument('--seed', default=8465, type=int)

    parser = CGCNModel.add_model_specific_args(parser)
    return parser.parse_args()


def main(params):
    if params.seed is not None:
        random.seed(params.seed)
        torch.manual_seed(params.seed)
        random.seed(params.seed)
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        torch.cuda.manual_seed(params.seed)
        cudnn.deterministic = True

    logger = TestTubeLogger(params.save_path, name=params.dataset)
    model = CGCNModel(params)

    trainer = pl.Trainer(
        logger=logger,
        default_save_path=params.save_path,
        gpus=params.gpus,
        max_nb_epochs=params.epochs,
        distributed_backend=params.dist_backend,
        use_amp=params.use_16bit,
        nb_sanity_val_steps=0,
        # val_check_interval=0.01,
        val_percent_check=0.01,
        train_percent_check=0.002,
        early_stop_callback=False,
    )
    if params.evaluate:
        trainer.run_evaluation()
    else:
        trainer.fit(model)


if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)
    main(get_args())








