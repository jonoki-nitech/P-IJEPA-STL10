# Uncertainty-sum trajectory recorder for the sigma selection sequences
# (standalone; does not modify any other script).
#
# This answers: "for each image, how did the total uncertainty (sum of sigma
# over the still-unobserved patches) evolve as blocks were observed?"
#
# It REPLAYS the stored selection sequences (selected_patch_index/
# block_{b}_tmax_32_{variant}.json, e.g. sigma_greedy / sigma_center) through
# the I-JEPA predictors: for every prefix of t blocks of every image it runs
# the standard forward (same math as ijepa_forward_batch /
# patch_selection_reconstruction_eval.py; verified identical) and records
#
#   sigma_sum[t]  =  sum of sigma over the 144 - t*b^2 patches that are
#                    NOT in the first t blocks of the context
#
# for t = 1..(that image's final block count). Entries for steps an image
# never took (it saturated/its window budget was reached) are NaN; an image
# whose context covers the whole image has 0 for its last entry.
#
# Since the sigma map is a deterministic function of the context, replaying
# the stored JSONs reproduces EXACTLY the uncertainty the selection saw -
# no re-rolling of the random elements, no changes to any other script.
#
# Outputs per block size b:
#   images/rec_images_tmax32/b{b}/sigma_sum_traj.pt
#     {"results": {variant: (N, max_steps) float32 tensor with NaN padding},
#      "variants": [...], "block_size", "t_max", "n_images", "aligned_max_t",
#      "definition": ...}
#
# Usage:
#   python sigma_uncertainty_traj.py --N 5 --bs 1 2                  (smoke)
#   python sigma_uncertainty_traj.py --bs 1 2 3 4                    (full)
#   ... --variants sigma_greedy sigma_center (default)
# Note: one batched forward per (variant, t), so the cost is the same as one
# sigma-selection sweep (~15 min per variant for b=1 on a GPU at N=8000).

import argparse
import json
import os

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from data_utils import load_stl10
from simple_ijepa.transformer import VisionTransformer
from simple_ijepa.ijepa import IJEPA

from patch_selection_reconstruction_eval import (
    CHUNK_SIZE,
    ALL_PATCHES,
    ijepa_forward_batch,
    iter_chunks,
)


def replay_sigma_sums(ijepa, img_t, contexts_flats, b, device, desc):
    """Replay the stored sequences and record the sigma sums.

    contexts_flats: per image, the flat patch list (prefix t = first t*b^2
    patches). Returns (n, max_steps) float64 array with NaN padding."""
    N = len(contexts_flats)
    max_steps = 144 // (b * b)
    steps = [len(c) // (b * b) for c in contexts_flats]
    out = np.full((N, max_steps), np.nan, dtype=np.float64)

    for t in tqdm(range(1, max_steps + 1), desc=desc):
        rows = [i for i in range(N) if steps[i] >= t]
        if not rows:
            break
        for c0, c1 in iter_chunks(len(rows)):
            act = rows[c0:c1]
            idx = torch.tensor(act, device=device)
            batch = img_t[idx]
            ctx = [contexts_flats[i][:(t * b * b)] for i in act]
            with torch.no_grad():
                sigmas, _, _ = ijepa_forward_batch(ijepa, batch, ctx)
            sig = sigmas.squeeze(-1)                     # (A, 144)
            ctx_idx = torch.tensor(ctx, device=device)
            mask = torch.ones(batch.shape[0], len(ALL_PATCHES),
                              dtype=torch.bool, device=device)
            mask[torch.arange(batch.shape[0], device=device)[:, None],
                 ctx_idx] = False
            total = torch.where(mask, sig, torch.zeros_like(sig))
            out[np.asarray(act), t - 1] = total.sum(dim=1).cpu().numpy()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='falls back to cpu when cuda is unavailable')
    parser.add_argument('--model_path', type=str,
                        default='models/embdim=512_epoch=300_patchsize=8/'
                                'training_model.pth')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--N', type=int, default=0,
                        help='number of test images (0 = all; must match '
                             'the N of the stored selection JSONs)')
    parser.add_argument('--bs', type=int, nargs='+', default=[1, 2, 3, 4],
                        help='block sizes to process')
    parser.add_argument('--variants', type=str, nargs='+',
                        default=['sigma_greedy', 'sigma_center'],
                        help=('selection JSON variants to replay (files '
                              'selected_patch_index/block_{b}_tmax_32_'
                              '<variant>.json)'))
    parser.add_argument('--out_dir', type=str,
                        default='images/rec_images_tmax32')
    parser.add_argument('--json_dir', type=str,
                        default='selected_patch_index',
                        help='where the block_{b}_tmax_32_{variant}.json '
                             'files live (e.g. selected_patch_index/seed_0)')
    args = parser.parse_args()

    if args.device == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = args.device

    stl10_ds = load_stl10(split="test", transform=transforms.ToTensor())
    start = args.start_index
    n = (len(stl10_ds) - start if args.N <= 0
         else min(args.N, len(stl10_ds) - start))
    xs = [stl10_ds[i][0] for i in range(start, start + n)]
    print(f'images: N={n} (start_index={args.start_index}), block sizes: '
          f'{args.bs}, variants={args.variants}, device={device}')

    encoder = VisionTransformer(
        image_size=96, patch_size=8, dim=512, depth=6, heads=6, mlp_dim=1024,
    ).to(device)
    ijepa = IJEPA(encoder, hidden_emb_dim=512, patch_size=8, num_targets=1)
    ijepa.load_state_dict(torch.load(args.model_path, map_location=device))
    ijepa = ijepa.to(device)
    ijepa.eval()

    for b in args.bs:
        print(f'\n===== block size b = {b} (uncertainty trajectories) =====')
        b_dir = os.path.join(args.out_dir, f"b{b}")
        results, kept_variants = {}, []
        max_steps = 144 // (b * b)
        img_t = torch.stack(xs).to(device)
        for variant in args.variants:
            jpath = os.path.join(args.json_dir, f"block_{b}_tmax_32_{variant}.json")
            if not os.path.isfile(jpath):
                print(f'  [skip] {jpath} not found')
                continue
            with open(jpath) as f:
                contexts = json.load(f)
            if len(contexts) != n:
                print(f'  [skip] {jpath} holds {len(contexts)} images, '
                      f'not N={n}')
                continue
            kept_variants.append(variant)
            results[variant] = replay_sigma_sums(
                ijepa, img_t, contexts, b, device,
                desc=f'  {variant} b={b} (t=1..{max_steps})')

        if not kept_variants:
            continue
        out_path = os.path.join(b_dir, "sigma_sum_traj.pt")
        torch.save({
            "results": {v: torch.tensor(results[v]) for v in kept_variants},
            "variants": kept_variants,
            "block_size": b,
            "t_max": max_steps,
            "n_images": n,
            "definition": ("sum of sigma over the patches still unobserved "
                           "when t blocks (of the stored selection sequence) "
                           "are in the context; NaN = the image never took "
                           "step t, 0 = nothing unobserved"),
        }, out_path)
        print(f'  saved: {out_path}')

        # quick console summary of the mean trajectory
        for v in kept_variants:
            arr = results[v]
            cols = np.isfinite(arr).any(axis=0)
            mean = np.full(arr.shape[1], np.nan)
            mean[cols] = np.nanmean(arr[:, cols], axis=0)
            shown = [f"t={t + 1}:{m:.3f}" for t, m in enumerate(mean)
                     if np.isfinite(m)][::max(1, int(cols.sum()) // 8)]
            print(f'  {v:>14}: mean sigma-sum: ' + "  ".join(shown))

    print('done.')


if __name__ == "__main__":
    main()
