# Calibration evaluation for the sigma (uncertainty) head of Probabilistic I-JEPA (STL10).
# Adapted from uncertainty_eval.py: inference loop is shared with it.
#
# What this script measures (sigma is treated as a pooled variance v per patch,
# following the training loss e/v + log(v)):
#   1. Reliability (calibration) diagram: equal-count bins by sigma, comparing
#      mean predicted v vs mean squared residual per bin, with the y=x reference.
#   2. Scalar diagnostics: Pearson / Spearman, mean(e)/mean(v), mean(e/v),
#      log-log regression of error on sigma (calibrated <=> intercept 0, slope 1).
#   3. Proper scoring rule comparisons (mean NLL over patches):
#      - our sigma            (heteroscedastic prediction)
#      - homoscedastic baseline (single global variance = mean residual)
#      - shuffled sigma       (same marginal distribution, pairing destroyed)
#   4. Per-dimension 95% coverage: fraction of |residual_d| <= 1.96 * sqrt(v).
#
# Two modes:
#   A) inference mode (default): runs the model over --N images and --positions,
#      collects (sigma, pred_error) per unobserved patch, then analyzes.
#   B) analysis-only mode (--points_pt <path>): skips model loading / dataset /
#      inference entirely and analyzes previously saved calibration_points.pt.
#      The input .pt is never overwritten in this mode.
#
# Usage example (inference mode):
#   python calibration_eval.py --N 10 --b 4 --positions 4:4
# Usage example (analysis only):
#   python calibration_eval.py --points_pt images/calibration/calibration_points.pt

import argparse
import csv
import os
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torchvision import transforms

from data_utils import load_stl10
from simple_ijepa.transformer import VisionTransformer
from simple_ijepa.ijepa import IJEPA


def block_to_index(left, top, w, h):
    unpatchfy = Rearrange("(h w)->h w",
                          h=12, w=12)

    blocks = torch.arange(0, 144)
    blocks = unpatchfy(blocks)

    blocks = blocks[top:top + h, left:left + w]

    index = blocks.flatten().tolist()

    return index


def rank_data(x):
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman_corr(x, y):
    return float(np.corrcoef(rank_data(x), rank_data(y))[0, 1])


def gaussian_nll(err, var):
    # Proper (halved) Gaussian NLL with variance var, per patch (pooled over 512 dims).
    # Additive constants (0.5*log(2*pi)) are dropped: identical across methods.
    return float(np.mean(0.5 * (err / var + np.log(var))))


def analyze(sigma_arr, err_arr, bins, out_dir, meta, save_points, source_note):
    """Compute calibration metrics + reliability diagram from collected points.

    meta: dict with keys n_images, positions_str, block_size, context_included,
          coverage_count, coverage_total (may be None).
    save_points: False in analysis-only mode (never overwrite the input .pt).
    """
    n_points = len(sigma_arr)
    eps = 1e-12
    pearson = float(np.corrcoef(sigma_arr, err_arr)[0, 1])
    sp = spearman_corr(sigma_arr, err_arr)

    mean_e = float(err_arr.mean())
    mean_v = float(sigma_arr.mean())
    mean_ratio = mean_e / mean_v                      # calibrated <=> 1.0
    mean_of_ratio = float((err_arr / sigma_arr).mean())  # calibrated <=> 1.0

    log_slope, log_intercept = np.polyfit(np.log(sigma_arr + eps),
                                          np.log(err_arr + eps), 1)

    nll_ours = gaussian_nll(err_arr, sigma_arr)
    v_homo = np.full_like(err_arr, mean_e)
    nll_homo = gaussian_nll(err_arr, v_homo)
    rng = np.random.default_rng(0)
    nll_shuffled = gaussian_nll(err_arr, sigma_arr[rng.permutation(n_points)])

    if (meta.get('coverage_count') is not None
            and meta.get('coverage_total') is not None
            and meta['coverage_total'] > 0):
        coverage = meta['coverage_count'] / meta['coverage_total']
    else:
        coverage = float('nan')

    # ------------------------------------------------------------------ #
    #  reliability diagram (equal-count bins by sigma)
    # ------------------------------------------------------------------ #
    order = np.argsort(sigma_arr)
    chunks = np.array_split(order, bins)
    bin_v = np.array([sigma_arr[c].mean() for c in chunks])
    bin_e = np.array([err_arr[c].mean() for c in chunks])

    lo = min(sigma_arr.min(), err_arr.min())
    hi = max(sigma_arr.max(), err_arr.max())
    pad = 0.05 * (hi - lo)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='y = x (calibrated)')
    ax.plot(bin_v, bin_e, 'o-', color='tab:blue', label=f'mean of {bins} bins')
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('predicted variance (sigma, binned mean)')
    ax.set_ylabel('squared residual (binned mean)')
    ax.set_title('Reliability diagram')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.scatter(sigma_arr, err_arr, s=6, alpha=0.15, color='tab:blue')
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('predicted variance (sigma)')
    ax.set_ylabel('squared residual')
    ax.set_title(f'Raw scatter (N={n_points} patches)')
    ax.grid(alpha=0.3)

    fig.suptitle(
        f'calibration of sigma vs squared residual ({source_note}, '
        f'images={meta["n_images"]}, positions={meta["positions_str"]}, '
        f'b={meta["block_size"]}, '
        f'context={"included" if meta["context_included"] else "excluded"})',
        fontsize=10)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'calibration_diagram.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # ------------------------------------------------------------------ #
    #  save metrics and raw points
    # ------------------------------------------------------------------ #
    csv_path = os.path.join(out_dir, 'calibration_metrics.csv')
    metrics = [
        ('source', source_note),
        ('n_points', n_points),
        ('n_images', meta['n_images']),
        ('positions', meta['positions_str']),
        ('block_size', meta['block_size']),
        ('context_included', int(meta['context_included'])),
        ('pearson_corr', pearson),
        ('spearman_corr', sp),
        ('mean_pred_var', mean_v),
        ('mean_residual_mse', mean_e),
        ('mean_ratio_mean_e_over_mean_v', mean_ratio),
        ('mean_of_ratio_e_over_v', mean_of_ratio),
        ('loglog_intercept_a', log_intercept),
        ('loglog_slope_b', log_slope),
        ('nll_ours_sigma', nll_ours),
        ('nll_homo_baseline', nll_homo),
        ('nll_shuffled_sigma', nll_shuffled),
        ('nll_gain_vs_homo', nll_homo - nll_ours),
        ('coverage_95', coverage),
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for name, value in metrics:
            writer.writerow([name, value])

    if save_points:
        torch.save({
            'sigma': torch.tensor(sigma_arr),
            'pred_error': torch.tensor(err_arr),
            'coverage_count': meta['coverage_count'],
            'coverage_total': meta['coverage_total'],
        }, os.path.join(out_dir, 'calibration_points.pt'))

    print('=' * 60)
    print(f'points: {n_points} (images: {meta["n_images"]}, '
          f'positions: {meta["positions_str"]})')
    print(f'Pearson  corr (sigma, error): {pearson:.4f}')
    print(f'Spearman corr (sigma, error): {sp:.4f}')
    print(f'mean(pred var)     = {mean_v:.4f}')
    print(f'mean(residual MSE) = {mean_e:.4f}')
    print(f'mean(e)/mean(v)    = {mean_ratio:.4f}  (1.0 = calibrated)')
    print(f'mean(e/v)          = {mean_of_ratio:.4f}  (1.0 = calibrated)')
    print(f'log-log fit: err = {np.exp(log_intercept):.4f} * sigma^{log_slope:.3f}  '
          f'(calibrated <=> slope 1.0, intercept factor 1.0)')
    print(f'NLL ours={nll_ours:.4f} | homoscedastic={nll_homo:.4f} | '
          f'shuffled={nll_shuffled:.4f} | gain vs homo={nll_homo - nll_ours:.4f}')
    print(f'95% coverage (per-dim): {coverage:.4f} (target 0.95)')
    print('=' * 60)
    print(f'saved: {fig_path}')
    print(f'saved: {csv_path}')
    if save_points:
        print(f'saved: {os.path.join(out_dir, "calibration_points.pt")}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='which device to use on local machine (inference mode only; '
             'falls back to cpu when cuda is unavailable)')
    parser.add_argument(
        '--model_path', type=str,
        help='path to the model (inference mode only)',
        default='models/embdim=512_epoch=300_patchsize=8/training_model.pth')
    parser.add_argument(
        '--start_index', type=int, default=0,
        help='the first index of test data (inference mode only)')
    parser.add_argument(
        '--N', type=int, default=0,
        help='number of test images to evaluate (0 = all) (inference mode only)')
    parser.add_argument(
        '--b', type=int, default=4,
        help='block size (inference mode only)')
    parser.add_argument(
        '--positions', type=str, nargs='+', default=['4:4'],
        help='context block positions as "top:left" (patch grid coordinates on '
             'the 12x12 patch grid); rows/cols covered are top..top+b-1 and '
             'left..left+b-1 (inference mode only)')
    parser.add_argument(
        '--bins', type=int, default=10,
        help='number of equal-count bins for the reliability diagram')
    parser.add_argument(
        '--include_context', action='store_true',
        help='keep patches inside the observed block (default: excluded, '
             'matching the training target condition)')
    parser.add_argument(
        '--out_dir', type=str, default='images/calibration',
        help='output directory')
    parser.add_argument(
        '--points_pt', type=str, default='',
        help='path to an existing calibration_points.pt; if set, the script '
             'runs analysis only (no model loading, no inference) and never '
             'overwrites the input file')

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    #  analysis-only mode: load previously collected points
    # ------------------------------------------------------------------ #
    if args.points_pt:
        data = torch.load(args.points_pt, map_location="cpu")
        sigma_arr = data['sigma'].cpu().numpy()
        err_arr = data['pred_error'].cpu().numpy()

        sample_ids = None
        if 'sample_ids' in data:
            sample_ids = np.asarray(data['sample_ids'])
        n_images = len(set(sample_ids.tolist())) if sample_ids is not None else -1

        positions = data.get('positions', None)
        if positions is not None:
            positions_str = ' '.join(f'{t}:{l}' for t, l in positions)
        else:
            positions_str = 'unknown'

        meta = {
            'n_images': n_images,
            'positions_str': positions_str,
            'block_size': data.get('block_size', 'unknown'),
            'context_included': bool(data.get('include_context', False)),
            'coverage_count': data.get('coverage_count', None),
            'coverage_total': data.get('coverage_total', None),
        }

        print(f'[analysis-only mode] loading points from: {args.points_pt}')
        analyze(sigma_arr, err_arr, args.bins, args.out_dir, meta,
                save_points=False, source_note=f'points_pt={args.points_pt}')
        return

    # ------------------------------------------------------------------ #
    #  inference mode: collect (sigma, pred_error) from the model
    # ------------------------------------------------------------------ #
    if args.device == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = args.device

    # dataset (same preprocessing as uncertainty_eval.py)
    stl10_ds = load_stl10(split="test",
                          transform=transforms.ToTensor())

    # model (same configuration as uncertainty_eval.py)
    encoder = VisionTransformer(
    image_size=96,
    patch_size=8,
    dim=512,
    depth=6,
    heads=6,
    mlp_dim=1024,
    ).to(device)

    ijepa = IJEPA(encoder,
                hidden_emb_dim=512,
                patch_size=8,
                num_targets=1)

    state_dict = torch.load(args.model_path, map_location=device)
    ijepa.load_state_dict(state_dict)
    ijepa = ijepa.to(device)

    # inference
    ijepa.eval()

    positions = []
    for pos in args.positions:
        top, left = pos.split(':')
        positions.append((int(top), int(left)))

    N = args.N if args.N > 0 else len(stl10_ds)
    N = min(N, len(stl10_ds) - args.start_index)

    os.makedirs(args.out_dir, exist_ok=True)

    sigmas_list = []
    errors_list = []
    sample_ids = []
    pos_ids = []
    cover_count = 0
    cover_total = 0

    for n in range(N):
        i = args.start_index + n
        img, label = stl10_ds[i]
        img = img.unsqueeze(0).to(device)

        for p_idx, (top, left) in enumerate(positions):
            context_indices_list = [block_to_index(left, top, args.b, args.b)]
            context_set = set(context_indices_list[0])

            # observed-block filter (applied to all analyses incl. coverage)
            keep = np.ones(144, dtype=bool)
            if not args.include_context:
                keep = np.array([i not in context_set for i in range(144)])

            target_indices_list = [[list(range(144))]]

            with torch.no_grad():
                target_features = ijepa.target_encoder(img).detach()
                target_features = torch.nn.functional.layer_norm(
                    target_features, (target_features.size(-1),))

                # Extract patches from the image
                patches = ijepa.context_encoder.to_patch_embedding(img)
                B, num_patches, _ = patches.size()

                # Extract context features
                context_tensor = torch.tensor(context_indices_list).to(patches.device)
                context_features = ijepa.context_encoder(
                    patches, patchify=False, mask=context_tensor)

                # Apply positional embeddings to context features
                pos_emb = ijepa.context_encoder.pos_embedding.to(context_features.device)
                context_pos_embed = torch.stack(
                    [pos_emb[context_tensor[i], :] for i in range(B)])
                context_features += context_pos_embed

                min_target_size = min(
                    min(len(indices) for indices in target_list)
                    for target_list in target_indices_list)
                for j in range(ijepa.num_targets):
                    # Init mask tokens and apply pos encodings
                    target_indices_batch = torch.stack(
                        [torch.tensor(target_indices_list[i][j], device=img.device)
                         for i in range(B)])
                    mask_tokens = ijepa.mask_token.expand(
                        B, min_target_size, -1).clone()
                    mask_pos_embed = torch.stack(
                        [pos_emb[target_indices_batch[i], :] for i in range(B)])
                    mask_tokens += mask_pos_embed

                    # Concat context_features and mask_tokens, obtain predictions
                    predictor_input = torch.cat([context_features, mask_tokens], dim=1)
                    predictor_output = ijepa.mu_predictor(
                        predictor_input, patchify=False, pos_embed=False)

                    predictor_output = torch.nn.functional.layer_norm(
                        predictor_output, (predictor_output.size(-1),))
                    preds = predictor_output[:, -min_target_size:]

                    sigma_predictor_output = ijepa.sigma_predictor(
                        predictor_input, patchify=False, pos_embed=False)
                    sigma_predictor_output = ijepa.sigma_predictor_head(
                        sigma_predictor_output)
                    sigmas = sigma_predictor_output[:, -min_target_size:]

                    targets = torch.stack(
                        [target_features[i, target_indices_list[i][j], :]
                         for i in range(B)])

                    resid = preds - targets
                    err = (resid ** 2).mean(dim=-1)  # per-patch mean squared error

                    # per-dim 95% coverage: |resid_d| <= 1.96 * sqrt(v)
                    # (same observed-block filter as the other analyses)
                    keep_mask = torch.from_numpy(keep).to(resid.device)
                    resid_kept = resid[:, keep_mask]
                    sigmas_kept = sigmas[:, keep_mask]
                    cover_count += (resid_kept.abs() <=
                                    1.96 * torch.sqrt(sigmas_kept)).sum().item()
                    cover_total += resid_kept.numel()

            sigma_np = sigmas.reshape(B, -1).float().cpu().numpy()
            err_np = err.reshape(B, -1).float().cpu().numpy()

            sigmas_list.append(sigma_np[0][keep])
            errors_list.append(err_np[0][keep])
            sample_ids.extend([i] * int(keep.sum()))
            pos_ids.extend([p_idx] * int(keep.sum()))

    sigma_arr = np.concatenate(sigmas_list)
    err_arr = np.concatenate(errors_list)

    meta = {
        'n_images': N,
        'positions_str': ' '.join(args.positions),
        'block_size': args.b,
        'context_included': bool(args.include_context),
        'coverage_count': cover_count,
        'coverage_total': cover_total,
    }

    analyze(sigma_arr, err_arr, args.bins, args.out_dir, meta,
            save_points=True, source_note='inference')

    # raw points incl. ids for later analysis (unchanged format + ids)
    torch.save({
        'sigma': torch.tensor(sigma_arr),
        'pred_error': torch.tensor(err_arr),
        'sample_ids': torch.tensor(sample_ids),
        'position_ids': torch.tensor(pos_ids),
        'positions': positions,
        'block_size': args.b,
        'include_context': args.include_context,
        'coverage_count': cover_count,
        'coverage_total': cover_total,
    }, os.path.join(args.out_dir, 'calibration_points.pt'))


if __name__ == "__main__":
    main()
