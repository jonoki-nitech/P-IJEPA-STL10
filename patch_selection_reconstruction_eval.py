# Patch selection -> reconstruction evaluation with 8 selection variants
# (7 baselines + the AME baseline). Style/output follow the reference
# test_mae_vs_t.py.
#
# Selection variants:
#   sigma_greedy      existing sigma-greedy: random initial b x b block, then
#                     repeatedly add the (sliding) b x b window with the largest
#                     sigma among fully-unobserved windows
#   sigma_center      same as sigma_greedy but the initial block is the center
#                     block (same center coordinates as the other center
#                     variants)
#   random_greedy     existing random: random initial block, then add a random
#                     fully-unobserved (sliding) b x b window per step
#   random_center     same as random_greedy but the initial block is the center
#   center_outward    deterministic: observe tiles in distance order from the
#                     image center outward
#   raster            deterministic reading order from the top-left
#   raster_center     center start (same center coordinates as the other center
#                     variants), then reading order
#   ame               Attention-Map Entropy baseline (Pardyl et al., IJCAI
#                     2023, apardyl/AME): random initial b x b block (shared
#                     seed), then repeatedly add the (sliding) b x b window
#                     with the largest attention-map entropy of the MAE
#                     decoder (see patch_selection_ame_eval.py)
#
# Truncation: when no fully-unobserved b x b window remains, that image stops
# observing (breaking even if steps are left). Without this, the sigma variants
# would argmax over -inf and re-add the same block. Evaluation aligns all
# variants on the common axis aligned_max_t = min(tile_total - 1, min over
# (variant, image) of the observed-block budget): reconstruction (MAE) is
# evaluated for t = 1..aligned_max_t using the prefix of each selection list
# as-is (no padding, no rewinding).
#
# For each variant, per (t, image): give MAE the visible patches
# (first t*b*b indices), reconstruct the unobserved region, and compute the
# pixel MSE over the unobserved region (identical formula to
# test_mae_on_selected_patches.py).
#
# Outputs (default --out_dir images/rec_images_tmax32):
#   b{b}/rec_error_vs_t.png    one chart, x = t, y = mean pixel MSE, one line
#                              per variant
#   b{b}/rec_error_vs_t.csv    mean values, rows = variants, cols = t
#   b{b}/rec_error_vs_t.pt     raw per-sample MSE arrays + alignment metadata
#                              (variant_max_t, aligned_max_t) for later
#                              analysis; re-plottable via --points_pt
#   b{b}/gallery/nXXXX.png     per-sample gallery (rows: original / observed
#                              (black = unobserved) / each variant's
#                              reconstruction at the final t), first
#                              --gallery_n samples
#   selected_patch_index/block_{b}_tmax_32_{variant}.json
#                              duplicate-free observed index lists (8 per b;
#                              new names avoid the old tmax=20)
#
# Calibration artifacts are produced separately by calibration_eval.py.
# Bundling into result.zip is intentionally left out (done later).
#
# MSE is measured in the inverse-normalized pixel domain, identical to
# test_mae_on_selected_patches.py.
#
# Batching: the whole batch is processed in CHUNK-sized chunks (env CHUNK,
# default 1024) so the 8000-image single batch never has to fit in GPU
# memory at once; selection and evaluation results are identical.
#
# Usage example (smoke test):
#   python patch_selection_reconstruction_eval.py --N 1 --bs 2
# Full run (heavy on CPU for b = 1):
#   python patch_selection_reconstruction_eval.py --N 100
# Analysis-only mode (re-plot from previously collected arrays):
#   python patch_selection_reconstruction_eval.py --points_pt \
#     images/rec_images_tmax32/b2/rec_error_vs_t.pt --out_dir images/tmp

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from data_utils import load_stl10
from simple_ijepa.transformer import VisionTransformer
from simple_ijepa.ijepa import IJEPA
from mae import MAE

VARIANTS = [
    "sigma_greedy",
    "sigma_center",
    "random_greedy",
    "random_center",
    "center_outward",
    "raster",
    "raster_center",
    "ame",
]
SIGMA_VARIANTS = ["sigma_greedy", "sigma_center"]
OUT_DIR = "images/rec_images_tmax32"

ALL_PATCHES = list(range(144))

# Process large batches in chunks of this many images (env CHUNK to override).
# Results are identical to one big batch; the peak GPU memory stays at the
# chunk size (roughly: I-JEPA per-image attention transient ~5.3 MB ->
# ~5.4 GB at chunk 1024; the 8000-image single batch would need ~43 GB).
CHUNK_SIZE = int(os.environ.get("CHUNK", "1024"))


def iter_chunks(n, chunk_size=None):
    """Yield (start, end) slices of the batch dimension."""
    c = CHUNK_SIZE if chunk_size is None else chunk_size
    for start in range(0, n, c):
        yield start, min(start + c, n)


def block_to_index(left, top, w, h):
    unpatchfy = Rearrange("(h w)->h w", h=12, w=12)
    blocks = torch.arange(0, 144)
    blocks = unpatchfy(blocks)
    blocks = blocks[top:top + h, left:left + w]
    return blocks.flatten().tolist()


def center_window_patches(b):
    """Index list of the center b x b block, shared by all center variants."""
    c = (12 - b) // 2
    return block_to_index(c, c, b, b)


def sliding_anchors(b):
    n = 12 - b + 1
    return [(top, left) for top in range(n) for left in range(n)]


def tile_anchors(b):
    return [(top, left) for top in range(0, 12, b) for left in range(0, 12, b)]


# --------------------------------------------------------------------------- #
#  selection
# --------------------------------------------------------------------------- #
def ijepa_forward_batch(ijepa, img, context_indices_list):
    """One I-JEPA pass for a batch with per-image context lists (equal length).

    Returns (sigmas (B,144,1), resid (B,144,D), pred_error (B,144)) computed
    over all 144 target patches (same inference as uncertainty_eval.py)."""
    with torch.no_grad():
        target_features = ijepa.target_encoder(img).detach()
        target_features = torch.nn.functional.layer_norm(
            target_features, (target_features.size(-1),))

        patches = ijepa.context_encoder.to_patch_embedding(img)
        B = patches.size(0)

        context_tensor = torch.tensor(context_indices_list).to(patches.device)
        context_features = ijepa.context_encoder(
            patches, patchify=False, mask=context_tensor)

        pos_emb = ijepa.context_encoder.pos_embedding.to(context_features.device)
        context_features += torch.stack(
            [pos_emb[context_tensor[i], :] for i in range(B)])

        target_indices_batch = torch.tensor([ALL_PATCHES] * B, device=img.device)
        mask_tokens = ijepa.mask_token.expand(
            B, len(ALL_PATCHES), -1).clone()
        mask_tokens += torch.stack(
            [pos_emb[target_indices_batch[i], :] for i in range(B)])

        predictor_input = torch.cat([context_features, mask_tokens], dim=1)
        predictor_output = ijepa.mu_predictor(
            predictor_input, patchify=False, pos_embed=False)
        predictor_output = torch.nn.functional.layer_norm(
            predictor_output, (predictor_output.size(-1),))
        preds = predictor_output[:, -len(ALL_PATCHES):]

        sigma_predictor_output = ijepa.sigma_predictor(
            predictor_input, patchify=False, pos_embed=False)
        sigma_predictor_output = ijepa.sigma_predictor_head(sigma_predictor_output)
        sigmas = sigma_predictor_output[:, -len(ALL_PATCHES):]

        targets = target_features[:, ALL_PATCHES, :]
        resid = preds - targets
        pred_error = (resid ** 2).mean(dim=-1)

    return sigmas, resid, pred_error


def select_sigma(variant, ijepa, imgs, b, rng, device):
    """sigma_greedy / sigma_center (model-based).

    Returns (contexts per image [flat patch lists], observed-block steps)."""
    N = len(imgs)
    max_steps = 144 // (b * b)

    if variant == "sigma_center":
        init = center_window_patches(b)
        contexts = [list(init) for _ in range(N)]
    else:  # sigma_greedy: random initial block (sliding anchor)
        anchors = sliding_anchors(b)
        contexts = []
        for _ in range(N):
            top, left = anchors[rng.integers(len(anchors))]
            contexts.append(block_to_index(left, top, b, b))

    img_t = torch.stack(imgs).to(device)
    active = list(range(N))
    steps = [1] * N
    step = 1
    while active and step < max_steps:
        batch = torch.stack([img_t[i] for i in active]).to(device)
        ctx = [contexts[i] for i in active]
        # chunked I-JEPA forward (the custom attention materializes the
        # B x heads x tokens x tokens matrix -> must stay chunk-bounded)
        sig_c, resid_c, err_c = [], [], []
        for c0, c1 in iter_chunks(len(active)):
            s, r, e = ijepa_forward_batch(
                ijepa, batch[c0:c1], ctx[c0:c1])
            sig_c.append(s)
            resid_c.append(r)
            err_c.append(e)
        sigmas = torch.cat(sig_c, dim=0)
        resid = torch.cat(resid_c, dim=0)
        pred_error = torch.cat(err_c, dim=0)

        # mask observed patches, greedy-select the max-sigma b x b window
        sig_masked = sigmas.clone()
        bidx = torch.arange(len(active), device=sigmas.device)[:, None]
        ctx_tensor = torch.tensor(ctx, device=sigmas.device)
        sig_masked[bidx, ctx_tensor] = -torch.inf
        pooled = F.avg_pool2d(sig_masked.reshape(len(active), 1, 12, 12),
                              kernel_size=b, stride=1)          # (A,1,H',W')
        A, _, Hp, Wp = pooled.shape
        pooled_flat = pooled.reshape(A, -1)
        best = pooled_flat.argmax(dim=1)
        best_val = pooled_flat.max(dim=1).values

        still = []
        for row, i in enumerate(active):
            if not torch.isfinite(best_val[row]):
                continue      # truncation: no fully-unobserved window remains
            top, left = np.unravel_index(best[row].item(), (Hp, Wp))
            contexts[i].extend(block_to_index(int(left), int(top), b, b))
            steps[i] += 1
            still.append(i)
        active = still
        step += 1
        print(f"\r    {variant}: step {step}/{max_steps} "
              f"(active {len(active)}/{N})", end="", flush=True)
    print()
    return contexts, steps


def select_random(variant, b, rng, N):
    """random_greedy / random_center (no model).

    Returns (contexts per image [flat patch lists], observed-block steps)."""
    max_steps = 144 // (b * b)
    anchors = sliding_anchors(b)
    # precompute every anchor's patch list / patch set once: the naive
    # block_to_index-per-(image, step, anchor) loop builds a torch tensor
    # and an einops Rearrange module per call (~0.2 ms), which used to
    # silently take hours for N=8000 (7 h for b=1 alone)
    anchor_patches = [block_to_index(left, top, b, b)
                      for (top, left) in anchors]
    anchor_sets = [frozenset(p) for p in anchor_patches]

    # NOTE: the rng draw pattern must stay exactly as before (one init draw
    # per image for the greedy variants, then one draw per step taken from
    # the valid anchors in anchor order) so seeded runs keep reproducing
    # the very same selection sequences.
    if variant == "random_center":
        init = [[center_window_patches(b)] for _ in range(N)]
    else:  # random initial block
        init = []
        for _ in range(N):
            init.append([list(anchor_patches[rng.integers(len(anchors))])])

    contexts, steps = [], []
    for i in range(N):
        observed = set(init[i][0])
        seq = [list(init[i][0])]
        for _ in range(max_steps - 1):
            valid = [k for k in range(len(anchors))
                     if observed.isdisjoint(anchor_sets[k])]
            if not valid:
                break      # truncation: no fully-unobserved window remains
            k = valid[rng.integers(len(valid))]
            window = anchor_patches[k]
            observed.update(anchor_sets[k])
            seq.append(list(window))
        contexts.append([p for blk in seq for p in blk])
        steps.append(len(seq))
        if (i + 1) % 500 == 0 or i + 1 == N:
            print(f"\r    {variant}: sequences {i + 1}/{N}",
                  end="", flush=True)
    print()
    return contexts, steps


def select_deterministic(variant, b, N):
    """center_outward / raster / raster_center (no model; tiles in reading or
    distance order; tiles overlapping already-observed ones are skipped)."""
    if variant == "center_outward":
        tiles = tile_anchors(b)
        tiles.sort(key=lambda tl: (((tl[0] + b / 2 - 6) ** 2 +
                                    (tl[1] + b / 2 - 6) ** 2), tl[0], tl[1]))
        start_center = False
    elif variant == "raster":
        tiles = tile_anchors(b)
        start_center = False
    else:  # raster_center
        tiles = tile_anchors(b)
        start_center = True

    if start_center:
        blocks = [center_window_patches(b)]
        observed = set(center_window_patches(b))
    else:
        blocks, observed = [], set()

    for top, left in tiles:
        window = block_to_index(left, top, b, b)
        if observed & set(window):
            continue
        observed |= set(window)
        blocks.append(list(window))

    flat = [p for blk in blocks for p in blk]
    steps = len(blocks)
    return [list(flat) for _ in range(N)], [steps] * N


def select_ame_variant(model_mae, xs_norm, b, device, attention_layer,
                       rng=None, random_first=True):
    """ame: Attention-Map Entropy baseline (Pardyl et al., IJCAI 2023),
    MAE decoder attention; duplicate-free truncation like the other
    strategies (details in patch_selection_ame_eval.py).

    The INITIAL b x b block is drawn uniformly at random from the valid
    sliding anchors with the shared rng (same seed as the other
    random-initial variants, sigma_greedy / random_greedy); the subsequent
    blocks come from the attention-map entropy. Set random_first=False for
    the official default instead (zero-step attention argmax, deterministic
    and identical for every image).

    The attention capture needs the timm attention modules of the MAE
    decoder to be patched; they are restored afterwards so every other
    variant keeps the untouched forward path.

    Returns (contexts per image [flat patch lists], observed-block steps)."""
    from patch_selection_ame_eval import select_ame, patch_attention_modules
    patch_attention_modules(model_mae)
    try:
        return select_ame(model_mae, xs_norm, b, device, attention_layer,
                          random_first=random_first, rng=rng)
    finally:
        for block in model_mae.decoder.layers:
            attn_mod = block.attn
            if "forward" in attn_mod.__dict__:
                del attn_mod.__dict__["forward"]


# --------------------------------------------------------------------------- #
#  MAE evaluation over the common t axis
# --------------------------------------------------------------------------- #
def obs_visualization(x, target_mask):
    """observed-only image (unobserved patches blacked out) and the boolean
    observed mask (True = observed), both (B,3,96,96)."""
    patchify = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=8, p2=8)
    unpatchify = Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                           p1=8, p2=8, h=12, w=12)
    B = x.shape[0]
    img_patches = patchify(x)
    observed = torch.ones(B, 144, dtype=torch.bool, device=x.device)
    observed[torch.arange(B, device=x.device)[:, None], target_mask] = False
    zeros = torch.zeros_like(img_patches)
    obs_img = torch.where(observed[:, :, None], img_patches, zeros)
    mask_img = unpatchify(observed[:, :, None]
                          .expand(-1, -1, img_patches.size(-1))
                          .float()).bool()
    return unpatchify(obs_img), mask_img


def evaluate_mae(model, xs_norm, contexts_blocks, b, device, inv, t_max,
                 gallery_n=10, desc=""):
    """Batch MAE reconstruction for t = 1..t_max over the selection prefixes.
    contexts_blocks: per image, {flat patch list or list of block lists}.
    Returns (t_list, mse (T, N) numpy, final_t gallery dict or None) where the
    gallery dict has {'observed': (n_g,3,96,96), 'rec': (n_g,3,96,96)} of the
    first gallery_n samples at the final t (already inverse-normalized)."""
    N = len(xs_norm)
    n_ctx = b * b
    # normalize input to block lists (sigma/random variants store flat lists)
    blocks_per_img = [
        [c[k * n_ctx:(k + 1) * n_ctx]
         for k in range(len(c) // n_ctx)] for c in contexts_blocks]
    t_list = list(range(1, t_max + 1))

    x = torch.stack(xs_norm).to(device)
    inv_x = inv(x)
    mse = np.full((len(t_list), N), np.nan, dtype=np.float64)
    final_gallery = None

    for t_idx, t in enumerate(tqdm(t_list, desc=f"  {desc} (t=1..{t_max})")):
        input_masks = [[p for blk in blocks_per_img[i][:t] for p in blk]
                       for i in range(N)]
        target_masks = [[p for p in range(144) if p not in set(input_masks[i])]
                        for i in range(N)]
        input_m = torch.tensor(input_masks, device=device)
        target_m = torch.tensor(target_masks, device=device)

        # process the batch in chunks (identical results, bounded memory)
        mse_row = np.empty(N, dtype=np.float64)
        gallery_obs, gallery_rec = [], []
        want = min(gallery_n, N) if (t == t_max and gallery_n > 0) else 0
        for c0, c1 in iter_chunks(N):
            x_c = x[c0:c1]
            with torch.no_grad():
                pred, _, _, _ = model.mae(
                    x_c, input_m[c0:c1], target_m[c0:c1])
            x_input, obs_mask_img = obs_visualization(x_c, target_m[c0:c1])
            pred_image = model.mae.tokens_as_image(pred.detach())
            pred_image[obs_mask_img] = x_c[obs_mask_img]

            x_input_inv = inv(x_input)
            # unobserved patches should be pure black after the inverse
            # transform (zeros in the normalized domain would render as
            # the dataset mean)
            x_input_inv = x_input_inv * obs_mask_img.to(x_input_inv.dtype)
            pred_image_inv = inv(pred_image)

            # per-sample pixel MSE over the unobserved region (inv domain)
            unobs = ~obs_mask_img
            se = ((pred_image_inv - inv_x[c0:c1]) ** 2) * unobs
            mse_row[c0:c1] = (se.flatten(1).sum(dim=1).cpu().numpy()
                              / unobs.flatten(1).sum(dim=1).cpu().numpy())

            if want and len(gallery_obs) < want:
                take = min(want - len(gallery_obs), x_c.shape[0])
                gallery_obs.append(x_input_inv[:take].detach().cpu())
                gallery_rec.append(pred_image_inv[:take].detach().cpu())

        mse[t_idx] = mse_row
        if want:
            final_gallery = {
                "observed": torch.cat(gallery_obs)[:want],
                "rec": torch.cat(gallery_rec)[:want],
            }

    return t_list, mse, final_gallery


# --------------------------------------------------------------------------- #
#  outputs (styled after test_mae_vs_t.py)
# --------------------------------------------------------------------------- #
def as_display(t):
    """(3, H, W) float tensor -> HxWx3 numpy in [0, 1] for imshow."""
    return t.permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def save_gallery_images(gallery_cache, variants, out_dir, b, t_final):
    """Save one reconstruction gallery PNG per sample.

    One PNG per sample: top row = original (every column), middle row = the
    observation given to MAE (unobserved = black), bottom row = each variant's
    reconstruction at the final aligned t with the observed patches replaced
    by the actual pixels (same compositing as test_mae_on_selected_patches.py).
    gallery_cache keys: 'original' {n: tensor},
                        'observed' {n: {variant: tensor}},
                        'rec'      {n: {variant: tensor}}
    """
    samples = sorted(gallery_cache["original"].keys())
    if not samples:
        return None
    gallery_dir = os.path.join(out_dir, "gallery")
    os.makedirs(gallery_dir, exist_ok=True)

    for n in samples:
        fig, axes = plt.subplots(3, len(variants), squeeze=False,
                                 figsize=(2 * len(variants), 6.6))
        for j, variant in enumerate(variants):
            axes[0, j].imshow(as_display(gallery_cache["original"][n]))
            axes[0, j].axis("off")
            axes[0, j].set_title(variant, fontsize=8)
            axes[1, j].imshow(as_display(gallery_cache["observed"][n][variant]))
            axes[1, j].axis("off")
            axes[2, j].imshow(as_display(gallery_cache["rec"][n][variant]))
            axes[2, j].axis("off")
        axes[1, 0].set_ylabel("observed")
        axes[2, 0].set_ylabel("reconstruction")
        fig.suptitle(f"sample {n} @ final t={t_final} (b={b})", fontsize=10)
        plt.tight_layout()
        path = os.path.join(gallery_dir, f"n{n:04d}.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
    print(f"saved: {len(samples)} gallery images "
          f"(e.g. {os.path.join(gallery_dir, f'n{samples[0]:04d}.png')})")
    return gallery_dir


def plot_results(results, t_list, out_dir, log_scale, title):
    """One chart: x = t, y = mean pixel MSE, one line per variant."""
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for variant, arr in results.items():
        mean = arr.mean(axis=1)
        ax.plot(t_list, mean, marker="o", ms=4, lw=1.8, label=variant)

    ax.set_xlabel("observation steps t")
    ax.set_ylabel("pixel MSE (unobserved region)")
    if len(t_list) <= 30:
        ax.set_xticks(t_list)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    png_path = os.path.join(out_dir, "rec_error_vs_t.png")
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path


def print_summary(results, t_list):
    print("=" * 70)
    for variant, arr in results.items():
        mean = arr.mean(axis=1)
        row = "  ".join(f"t={t}:{m:.4f}" for t, m in zip(t_list, mean))
        print(f"{variant:>22}: {row}")
    print("=" * 70)


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='falls back to cpu when cuda is unavailable')
    parser.add_argument('--model_path', type=str,
                        default='models/embdim=512_epoch=300_patchsize=8/training_model.pth')
    parser.add_argument('--mae_model_path', type=str,
                        default='models/mae/latest.ckpt')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--N', type=int, default=10,
                        help='number of test images to evaluate (0 = all)')
    parser.add_argument('--bs', type=int, nargs='+', default=[1, 2, 3, 4],
                        help='block sizes to evaluate')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for the random elements')
    parser.add_argument('--attention-layer', type=int, default=-1,
                        help='MAE decoder block whose attention drives the '
                             'AME selection (0-based; -1 = last decoder '
                             'block, as in the official AME)')
    parser.add_argument('--tlist', type=int, nargs='+', default=None,
                        help='observation step counts to sweep '
                             '(default: 1..aligned max t)')
    parser.add_argument('--log', action='store_true',
                        help='use a log scale on the y axis')
    parser.add_argument('--gallery_n', type=int, default=10,
                        help='save reconstruction gallery PNGs (original / '
                             'observed / each variant reconstruction at the '
                             'final t) for the first this-many samples; '
                             '0 disables')
    parser.add_argument('--out_dir', type=str, default=OUT_DIR)
    parser.add_argument(
        '--points_pt', type=str, default='',
        help='path to an existing rec_error_vs_t.pt; if set, the script does '
             'plotting only (no model loading, no dataset, no forward passes)')
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    #  analysis-only mode: plot from previously collected arrays
    # ------------------------------------------------------------------ #
    if args.points_pt:
        data = torch.load(args.points_pt, map_location="cpu")
        t_list = [int(t) for t in data["t_list"]]
        results = {v: np.asarray(a, dtype=float)
                   for v, a in data["results"].items()}
        b_pt = int(data.get("block_size", args.bs[0]))

        # alignment: truncate at the smallest observable budget
        # (variant_max_t / aligned_max_t were stored with the arrays)
        t_eval_pt = int(data.get("aligned_max_t", max(t_list)))
        keep = [i for i, t in enumerate(t_list) if t <= t_eval_pt]
        if len(keep) < len(t_list):
            t_list = [t_list[i] for i in keep]
            results = {v: a[keep] for v, a in results.items()}
        t_list = [t for t in t_list
                  if args.tlist is None or t in args.tlist]

        title = (f"MAE reconstruction error vs t "
                 f"(b={b_pt}, T_max={data.get('t_max', '?')}, "
                 f"N={data.get('n_images', '?')}, aligned max t={t_eval_pt})")
        print(f"[analysis-only mode] plotting from: {args.points_pt}")
        png_path = plot_results(results, t_list,
                                os.path.join(args.out_dir, f"b{b_pt}"),
                                args.log, title)
        print_summary(results, t_list)
        print(f"saved: {png_path}")
        sys.exit(0)

    if args.device == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = args.device

    rng = np.random.default_rng(args.seed)

    # dataset (ToTensor only, same preprocessing as uncertainty_eval.py)
    stl10_ds = load_stl10(split="test", transform=transforms.ToTensor())
    start = args.start_index
    n = (len(stl10_ds) - start if args.N <= 0
         else min(args.N, len(stl10_ds) - start))

    normalize = transforms.Normalize(
        mean=(0.4467, 0.4398, 0.4066),
        std=(0.2603, 0.2566, 0.2713))
    inv = transforms.Normalize(
        mean=[-0.4467 / 0.2603, -0.4398 / 0.2566, -0.4066 / 0.2713],
        std=[1 / 0.2603, 1 / 0.2566, 1 / 0.2713])

    xs_raw, xs_norm = [], []
    for i in range(start, start + n):
        img, _ = stl10_ds[i]
        xs_raw.append(img)
        xs_norm.append(normalize(img))
    originals = {i: inv(xs_norm[i]).cpu() for i in range(n)}
    print(f'images: N={n} (start_index={args.start_index}), block sizes: '
          f'{args.bs}, seed={args.seed}, device={device}')

    # models
    encoder = VisionTransformer(
        image_size=96, patch_size=8, dim=512, depth=6, heads=6, mlp_dim=1024,
    ).to(device)
    ijepa = IJEPA(encoder, hidden_emb_dim=512, patch_size=8, num_targets=1)
    ijepa.load_state_dict(torch.load(args.model_path, map_location=device))
    ijepa = ijepa.to(device)
    ijepa.eval()

    model = MAE().to(device)
    model.load_state_dict(
        torch.load(args.mae_model_path, map_location=device)["state_dict"])
    model.eval()

    if args.attention_layer == -1:
        attention_layer = model.mae.dec_depth - 1
    else:
        attention_layer = args.attention_layer
        assert 0 <= attention_layer < model.mae.dec_depth
    print(f'MAE: dec_depth={model.mae.dec_depth}, '
          f'attention_layer={attention_layer} (AME selection)')

    os.makedirs("selected_patch_index", exist_ok=True)

    for b in args.bs:
        print(f'\n===== block size b = {b} =====')
        b_dir = os.path.join(args.out_dir, f"b{b}")
        os.makedirs(b_dir, exist_ok=True)
        tile_total = 144 // (b * b)

        # ---- selection sequences ---------------------------------------- #
        seqs, stops = {}, {}
        for variant in VARIANTS:
            if variant in SIGMA_VARIANTS:
                seqs[variant], stops[variant] = select_sigma(
                    variant, ijepa, xs_raw, b, rng, device)
            elif variant in ("random_greedy", "random_center"):
                seqs[variant], stops[variant] = select_random(
                    variant, b, rng, n)
            elif variant == "ame":
                seqs[variant], stops[variant] = select_ame_variant(
                    model.mae, xs_norm, b, device, attention_layer, rng)
            else:
                seqs[variant], stops[variant] = select_deterministic(
                    variant, b, n)
            print(f'  {variant}: steps min={min(stops[variant])}, '
                  f'max={max(stops[variant])}', flush=True)

        # JSONs (8 per b: 7 sigma/random/fixed variants + AME, tmax_32 names)
        for variant in VARIANTS:
            jpath = os.path.join(
                "selected_patch_index", f"block_{b}_tmax_32_{variant}.json")
            with open(jpath, "w") as f:
                json.dump(seqs[variant], f)
            print(f'  saved: {jpath}')

        # ---- alignment (per-variant saturation budget) ------------------- #
        # the largest t for which EVERY image still receives fresh patches at
        # step t (saturation of one image would otherwise repeat the same
        # visible set and show as a flat segment); also leave at least one
        # patch unobserved so the unobserved-region MSE stays defined
        variant_max_t = {v: min(stops[v]) for v in VARIANTS}
        T_plot = min(t for t in [tile_total] + list(variant_max_t.values()))
        t_eval = min(T_plot, tile_total - 1)
        dropped = [f"{v}(budget={variant_max_t[v]})" for v in VARIANTS
                   if variant_max_t[v] > t_eval]
        print(f"aligned max t = {t_eval} "
              f"(truncated variants beyond their saturation: {dropped})",
              flush=True)
        t_list = (args.tlist if args.tlist is not None
                  else list(range(1, t_eval + 1)))
        t_list = [t for t in t_list if t <= t_eval]

        # ---- MAE sweep over the common t axis ---------------------------- #
        results = {v: np.zeros((len(t_list), n)) for v in VARIANTS}
        gallery_on = args.gallery_n > 0
        gallery_cache = {"original": dict(originals), "observed": {}, "rec": {}}

        for variant in VARIANTS:
            t_list_v, mse, final_gallery = evaluate_mae(
                model, xs_norm, seqs[variant], b, device, inv, t_eval,
                gallery_n=args.gallery_n, desc=f'b={b} {variant}')
            ti_map = {t: i for i, t in enumerate(t_list_v)}
            for ti, t in enumerate(t_list):
                results[variant][ti] = mse[ti_map[t]]

            if gallery_on and final_gallery is not None:
                for i in range(final_gallery["observed"].shape[0]):
                    gallery_cache["observed"].setdefault(i, {})[variant] = \
                        final_gallery["observed"][i]
                    gallery_cache["rec"].setdefault(i, {})[variant] = \
                        final_gallery["rec"][i]

        # ---- gallery PNGs (first samples, at the aligned final t) --------- #
        if gallery_on:
            save_gallery_images(gallery_cache, VARIANTS, b_dir, b, t_list[-1])

        # ---- plot --------------------------------------------------------- #
        title = (f"MAE reconstruction error vs t "
                 f"(b={b}, T_max={tile_total}, N={n}, "
                 f"aligned max t={t_eval})")
        png_path = plot_results(results, t_list, b_dir, args.log, title)

        # ---- csv + raw arrays --------------------------------------------- #
        csv_path = os.path.join(b_dir, "rec_error_vs_t.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["variant"] + [f"t={t}" for t in t_list])
            for variant in VARIANTS:
                mean = results[variant].mean(axis=1)
                writer.writerow([variant] + [f"{m:.6f}" for m in mean])

        pt_path = os.path.join(b_dir, "rec_error_vs_t.pt")
        torch.save({
            "results": {v: torch.tensor(results[v]) for v in VARIANTS},
            "t_list": t_list,
            "block_size": b,
            "t_max": tile_total,
            "n_images": n,
            "variant_max_t": variant_max_t,
            "aligned_max_t": t_eval,
            "seed": args.seed,
        }, pt_path)

        # ---- console summary ---------------------------------------------- #
        print_summary(results, t_list)
        print(f"saved: {png_path}")
        print(f"saved: {csv_path}")
        print(f"saved: {pt_path}")

    print('done.')


if __name__ == "__main__":
    main()
