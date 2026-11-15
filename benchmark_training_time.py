"""Measure the wall-clock cost of the two pretrainings and extrapolate it to
the full runs (for the paper's reproducibility statement: training duration).

Two training procedures are benchmarked exactly as they are trained:

  ijepa : the σ-head I-JEPA of train.py / run_training.py
          (ViT 512/6, STL-10 unlabeled, batch 192, Adam 3e-4 / 1e-5, EMA
          target encoder updated every step, fp16 OFF by default,
          300 epochs -> ceil(100000/192) = 521 steps/epoch ~= 156,300 steps)
  mae   : the MAE of mae-pytorch (config/mae.yaml + data/stl10_mae.yaml)
          (96/8, keep 0.25, enc 768/12, dec 384/6, STL-10 unlabeled with the
          resize/pad/affine/crop/flip + Normalize augmentation, batch 64,
          AdamW effective lr 1.5e-4*64/256, 16-mixed, drop_last=True,
          200 epochs -> 1562 steps/epoch = 312,400 steps)

The script runs a few warm-up steps and then `--steps` timed steps of
forward + backward + optimizer (+ EMA for ijepa), including the data fetch,
and reports the per-step time, the per-epoch time and the extrapolated total
training time. Run it on the TRAINING machine; the numbers are hardware
dependent. The periodic linear-probe monitoring of train.py (every 1000
steps) is NOT part of the timed loop.

Examples:
  python benchmark_training_time.py                       # both, full-size, GPU
  python benchmark_training_time.py --model ijepa --steps 50
  python benchmark_training_time.py --model mae --batch_size 64 --steps 30
  python benchmark_training_time.py --model both --dataset_size 512 \
      --num_workers 0 --device cpu --steps 2 --warmup 1     # quick smoke
"""

import argparse
import importlib.util
import math
import os
import random
import statistics
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from data_utils import load_stl10                       # noqa: E402
from simple_ijepa.ijepa import IJEPA                    # noqa: E402
from simple_ijepa.transformer import VisionTransformer  # noqa: E402
from simple_ijepa.dataset import (MaskedImageDataset,   # noqa: E402
                                  collate_fn)
from simple_ijepa.utils import training_transforms      # noqa: E402

# mae-pytorch/mae.py under an alias: a plain `import mae` would resolve to the
# top-level mae.py (the evaluation copy, different forward signature)
_spec = importlib.util.spec_from_file_location(
    "mae_pt", os.path.join(HERE, "mae-pytorch", "mae.py"))
mae_pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mae_pt)

STL10_UNLABELED = 100000
STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2603, 0.2566, 0.2713)


def make_loader(dataset, batch_size, num_workers, shuffle=True,
                drop_last=False, collate=None):
    return DataLoader(dataset, batch_size=batch_size,
                      num_workers=num_workers, pin_memory=True,
                      shuffle=shuffle, drop_last=drop_last,
                      collate_fn=collate)


def next_batch(it, loader):
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it


def timed_loop(loader, step_fn, warmup, steps, device):
    """Run warmup + timed steps; every timed step includes the data fetch."""
    it = iter(loader)
    for _ in range(warmup):
        batch, it = next_batch(it, loader)
        step_fn(batch)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    per_step = []
    for _ in range(steps):
        ts = time.perf_counter()
        batch, it = next_batch(it, loader)
        step_fn(batch)
        per_step.append(time.perf_counter() - ts)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    total = time.perf_counter() - t0
    return per_step, total


def report(tag, per_step, total, steps_ran, steps_per_epoch, epochs, device,
           batch_size, dataset_capped):
    mean = total / steps_ran
    med = statistics.median(per_step)
    std = statistics.pstdev(per_step) if steps_ran > 1 else 0.0
    epoch_s = mean * steps_per_epoch
    total_h = epoch_s * epochs / 3600.0
    print(f'\n==== {tag} ====')
    dev_name = (torch.cuda.get_device_name(0)
                if device.type == 'cuda' else 'CPU')
    print(f'device: {dev_name} | batch {batch_size} | '
          f'{args_warmup} warmup + {steps_ran} timed steps')
    print(f'per-step: mean {mean:.3f} s | median {med:.3f} s | std {std:.3f} s'
          f' | {1.0 / mean:.2f} steps/s')
    if dataset_capped:
        print('!! capped dataset: the extrapolation below is NOT valid for '
              'the paper (run with the full unlabeled split)')
    print(f'extrapolation: {steps_per_epoch} steps/epoch x {epochs} epochs = '
          f'{steps_per_epoch * epochs:,} steps -> {epoch_s / 60:.1f} '
          f'min/epoch -> ~{total_h:.1f} h total (optimizer steps only)')
    if device.type == 'cuda':
        gb = 1024 ** 3
        print(f'peak GPU memory: '
              f'{torch.cuda.max_memory_allocated() / gb:.2f} GB')
    return mean, total_h


# the warmup count is needed by report() for the header line only
args_warmup = 0


def benchmark_ijepa(args, device):
    global args_warmup
    args_warmup = args.warmup
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    bs = args.batch_size or 192
    dim, image_size, patch_size, depth, heads = 512, 96, 8, 6, 6
    encoder = VisionTransformer(image_size=image_size, patch_size=patch_size,
                                dim=dim, depth=depth, heads=heads,
                                mlp_dim=dim * 2)
    ijepa = IJEPA(encoder, hidden_emb_dim=dim, patch_size=patch_size,
                  num_targets=4).to(device)

    params = (list(ijepa.context_encoder.parameters())
              + [ijepa.mask_token]
              + list(ijepa.mu_predictor.parameters())
              + list(ijepa.sigma_predictor.parameters())
              + list(ijepa.sigma_predictor_head.parameters()))
    optimizer = torch.optim.Adam(params, lr=3e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16
                                       and device.type == 'cuda')

    ds = load_stl10(args.split, transform=training_transforms((96, 96)))
    capped = ((bool(args.dataset_size) and args.dataset_size < len(ds))
              or args.split != 'unlabeled')
    if args.dataset_size:
        ds = Subset(ds, range(min(args.dataset_size, len(ds))))
    loader = make_loader(MaskedImageDataset(ds, num_patches=144,
                                            num_targets=4),
                         bs, args.num_workers, drop_last=False,
                         collate=collate_fn)
    gamma = 0.996

    def step(batch):
        images, context_indices, target_indices_list = batch
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=args.fp16
                                     and device.type == 'cuda'):
            loss = ijepa(images, context_indices, target_indices_list)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ijepa.update_params(gamma)

    per_step, total = timed_loop(loader, step, args.warmup, args.steps,
                                 device)
    steps_per_epoch = math.ceil(STL10_UNLABELED / bs)   # drop_last=False
    report('sigma-head I-JEPA (STL-10 unlabeled, as train.py)', per_step,
           total, args.steps, steps_per_epoch, 300, device, bs, capped)


def benchmark_mae(args, device):
    global args_warmup
    args_warmup = args.warmup
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

    from torchvision import transforms as T
    bs = args.batch_size or 64
    tfm = T.Compose([
        T.Resize((96, 96)),
        T.Pad(96 // 8, padding_mode='reflect'),
        T.RandomAffine((-10, 10), (0, 1 / 8), (1, 1.2)),
        T.CenterCrop((96, 96)),
        T.RandomHorizontalFlip(0.5),
        T.ToTensor(),
        T.Normalize(STL10_MEAN, STL10_STD),
    ])
    ds = load_stl10(args.split, transform=tfm)
    capped = ((bool(args.dataset_size) and args.dataset_size < len(ds))
              or args.split != 'unlabeled')
    if args.dataset_size:
        ds = Subset(ds, range(min(args.dataset_size, len(ds))))
    loader = make_loader(ds, bs, args.num_workers, drop_last=True,
                         collate=None)   # default collate: (img, label)

    model = mae_pt.MAE(image_size=(96, 96), patch_size=8, keep=0.25,
                       enc_width=768, dec_width=0.5, enc_depth=12,
                       dec_depth=6, lr=1.5e-4,
                       base_batch_size=256).to(device)
    # effective lr = base * batch / base_batch_size (configure_optimizers)
    lr = 1.5e-4 * bs / 256
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.95), weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=args.warmup + args.steps,
        pct_start=0.1, cycle_momentum=False)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    amp = device.type == 'cuda'          # training used Lightning 16-mixed

    def step(batch):
        x = batch[0].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            pred = model.mae(x)
            loss = model.masked_mse_loss(x, pred)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        sched.step()

    per_step, total = timed_loop(loader, step, args.warmup, args.steps,
                                 device)
    steps_per_epoch = STL10_UNLABELED // bs              # drop_last=True
    report('MAE (STL-10 unlabeled, as mae-pytorch config)', per_step, total,
           args.steps, steps_per_epoch, 200, device, bs, capped)


def main():
    ap = argparse.ArgumentParser(
        description='Benchmark the wall-clock cost of the pretrainings')
    ap.add_argument('--model', choices=['both', 'ijepa', 'mae'],
                    default='both')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available()
                    else 'cpu')
    ap.add_argument('--batch_size', type=int, default=0,
                    help='0 = the training default (ijepa 192, mae 64)')
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--steps', type=int, default=30,
                    help='number of timed steps')
    ap.add_argument('--num_workers', type=int, default=8,
                    help='data-loader workers (training uses 8)')
    ap.add_argument('--fp16', action='store_true',
                    help='ijepa: enable the fp16 path of train.py '
                         '(the actual training ran WITHOUT it)')
    ap.add_argument('--dataset_size', type=int, default=0,
                    help='cap the dataset for quick smoke tests '
                         '(0 = the full unlabeled split)')
    ap.add_argument('--split', default='unlabeled', choices=['unlabeled',
                    'test'],
                    help="smoke-test only: 'test' avoids the 2.6 GB "
                         "unlabeled split on small machines (the "
                         "extrapolation is then marked invalid)")
    args = ap.parse_args()
    device = torch.device(args.device)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    if args.model in ('both', 'ijepa'):
        benchmark_ijepa(args, device)
    if args.model in ('both', 'mae'):
        benchmark_mae(args, device)
    print('\nNOTE: the periodic linear-probe monitoring of train.py '
          '(every 1000 steps) is NOT included in the extrapolation.')


if __name__ == '__main__':
    main()
