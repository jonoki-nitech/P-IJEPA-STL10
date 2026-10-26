"""Trains the MC-Dropout baseline I-JEPA (see train_mcdropout.py).

Same training setup as python run_training.py (same architecture
embdim=512, patchsize=8, 300 epochs, batch 192, lr 3e-4, wd 1e-5, the EMA
gamma schedule, the STL10 unlabeled split, the periodic saves and the
periodic STL10 logistic-regression evaluations) with two differences:
the loss is the vanilla I-JEPA MSE (no sigma-predictor / NLL) and only the
mu-predictor carries Dropout (--mu_dropout, default 10%) for MC-Dropout
uncertainty estimates at inference time.

NOTE: -save_model_dir defaults to models/mcdropout_embdim=512_epoch=300_patchsize=8
so the probabilistic (sigma-based) checkpoints in
models/embdim=512_epoch=300_patchsize=8 are NOT overwritten.
NOTE: the sigma-predictor weights in the resulting training_model.pth are
at their initialization (untrained) -- the sigma-based strategies are
meaningless with this checkpoint.
"""
import argparse

from train_mcdropout import train_ijepa_mcdropout
from data_utils import get_stl10_root
import os

parser = argparse.ArgumentParser(description='MC-Dropout baseline I-JEPA')
parser.add_argument('--dataset_path',
                    default=None,
                    help='Path where datasets will be saved (未指定時は自動判定)')
parser.add_argument('--dataset_name',
                    default='stl10',
                    help='Dataset name',
                    choices=['stl10'])
parser.add_argument('-save_model_dir',
                    default='./models/mcdropout_embdim=512_epoch=300_patchsize=8',
                    help='Path where models')
parser.add_argument('--num_epochs',
                    default=300,
                    type=int,
                    help='Number of epochs for training')
parser.add_argument('-b',
                    '--batch_size',
                    default=192,
                    type=int,
                    help='Batch size')
parser.add_argument('-lr', '--learning_rate', default=3e-4, type=float)
parser.add_argument('-wd', '--weight_decay', default=1e-5, type=float)
parser.add_argument('--fp16_precision',
                    action='store_true',
                    help='Whether to use 16-bit precision for GPU training')

parser.add_argument('--emb_dim',
                    default=512,
                    type=int,
                    help='Transofmer embedding dimm')
parser.add_argument('--log_every_n_steps',
                    default=200,
                    type=int,
                    help='Log every n steps')
parser.add_argument('--gamma',
                    default=0.996,
                    type=float,
                    help='Initial EMA coefficient')
parser.add_argument('--update_gamma_after_step',
                    default=1,
                    type=int,
                    help='Update EMA gamma after this step')
parser.add_argument('--update_gamma_every_n_steps',
                    default=1,
                    type=int,
                    help='Update EMA gamma after this many steps')
parser.add_argument('--mu_dropout',
                    default=0.1,
                    type=float,
                    help='Dropout probability inside the mu-predictor '
                         '(MC-Dropout; 0 disables dropout)')
parser.add_argument('--ckpt_path',
                    default=None,
                    type=str,
                    help='Specify path to training_model.pth to resume training')


def main():
    args = parser.parse_args()
    if args.dataset_path is None:
        args.dataset_path = get_stl10_root()
    os.makedirs(args.save_model_dir, exist_ok=True)
    train_ijepa_mcdropout(args)


if __name__ == "__main__":
    main()
