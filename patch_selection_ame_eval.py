# Attention-Map Entropy (AME; Pardyl et al., IJCAI 2023,
# https://github.com/apardyl/AME) patch selection for the STL10 MAE, used as
# a COMPARISON baseline for the existing 7-strategy evaluation.
#
# Method (official AttentionGlimpseSelector, adapted):
#   * zero step: run the MAE decoder once on mask tokens only (no context) to
#     initialize the decoder attention weights; the encoder is skipped (the
#     official implementation likewise runs one forward pass before the
#     first selection);
#   * each selection step: take the softmax attention weights of a decoder
#     block (default: last decoder layer), zero out the attention rows
#     (queries) of already-observed patches and compute
#         entropy = sum over (heads, keys) of -a * log2(a)   (nan -> 0)
#     for every unobserved patch; pool with avg_pool2d(kernel=b, stride=1)
#     and select the b x b window with the largest pooled entropy as the
#     next observation;
#   * truncation: when no fully-unobserved b x b window remains (or the
#     step cap 144/b^2 is reached), that image stops observing. This is the
#     one deviation from the official code, which allows re-observing
#     patches; here the selection must stay duplicate-free to match the
#     other strategies and the t-sweep evaluation protocol.
#
# The attention maps are obtained by replacing the forward of the timm
# attention modules inside the MAE decoder at runtime (mae.py is NOT
# modified). The reconstruction error-vs-t sweep reuses the exact machinery
# of patch_selection_reconstruction_eval.py, and AME is deterministic, so no
# seed is needed (--random_first is off by default, like the official code).
#
# Outputs (per block size b, default --out_dir images/rec_images_tmax32):
#   selected_patch_index/block_{b}_tmax_32_ame.json  observed index lists
#   b{b}/ame/rec_error_vs_t.png                      AME-only chart
#   b{b}/rec_error_vs_t_ame.csv / .pt                AME-only raw results
#   b{b}/ame/gallery/nXXXX.png                       AME gallery (first
#                                                    --gallery_n samples)
#   b{b}/rec_error_vs_t_comparison_ame.png           one chart with the 7
#                                                    stored variants (from
#                                                    the existing
#                                                    rec_error_vs_t.pt of
#                                                    the same N) + the AME
#                                                    line, for direct
#                                                    comparison
#
# Usage:
#   python patch_selection_ame_eval.py --N 1 --bs 2        # smoke test
#   python patch_selection_ame_eval.py                     # all images, b=1..4

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from data_utils import load_stl10
from mae import MAE
from patch_selection_reconstruction_eval import (
    CHUNK_SIZE,
    block_to_index,
    evaluate_mae,
    iter_chunks,
    plot_results,
    print_summary,
    save_gallery_images,
)

N_PATCHES = 144
GRID = 12
VARIANT = "ame"


# --------------------------------------------------------------------------- #
#  attention capture (no modification of mae.py)
# --------------------------------------------------------------------------- #
def patch_attention_modules(model_mae):
    """Replace the forward of every timm attention module inside the MAE
    decoder so that it stores the softmax attention weights (B, heads, 144,
    144) as `last_attn`. Only the decoder is patched: the official AME reads
    the attention from the decoder (its attention_layer defaults to the last
    decoder block)."""
    for block in model_mae.decoder.layers:
        attn_mod = block.attn
        num_heads = attn_mod.num_heads
        head_dim = getattr(attn_mod, "head_dim",
                           attn_mod.qkv.out_features // (3 * num_heads))
        scale = attn_mod.scale

        def forward(x, attn_mask=None, is_causal=False,
                    _mod=attn_mod, _heads=num_heads, _dim=head_dim,
                    _scale=scale):
            B, N, _ = x.shape
            qkv = _mod.qkv(x).reshape(B, N, 3, _heads, _dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * _scale
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            _mod.last_attn = attn.detach()
            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
            x = _mod.proj(x)
            x = _mod.proj_drop(x)
            return x

        attn_mod.forward = forward


def get_decoder_attn(model_mae, layer_idx):
    """Stored softmax attention of the chosen decoder block:
    (B, heads, 144, 144)."""
    return model_mae.decoder.layers[layer_idx].attn.last_attn


# --------------------------------------------------------------------------- #
#  AME selection
# --------------------------------------------------------------------------- #
def pooled_entropy(attn, observed, b):
    """Pooled attention-map entropy, following the official
    AttentionGlimpseSelector.

    attn: (B, heads, 144, 144) softmax attention (decoder, no cls token)
    observed: None or (B, 144) bool, True = already-observed patch
    Returns the pooled entropy map (B, 1, h', w') for anchors a stride-1
    b x b window sliding over the patch grid (-inf where the window would
    overlap an already-observed patch; the official AME allows overlaps,
    here the selection must stay duplicate-free)."""
    B = attn.shape[0]
    if observed is not None:
        attn = attn * (~observed).view(B, 1, N_PATCHES, 1)
    ent = torch.nan_to_num(-attn * torch.log2(attn), nan=0.0)
    ent = ent.sum(dim=(1, 3)).reshape(B, 1, GRID, GRID)
    pooled = F.avg_pool2d(ent, kernel_size=b, stride=1)
    if observed is not None:
        # a b x b window is valid iff it contains no observed patch
        valid = F.max_pool2d(observed.float().view(B, 1, GRID, GRID),
                             kernel_size=b, stride=1)
        pooled = pooled.masked_fill(valid > 0, -torch.inf)
    return pooled


def forward_attention(model_mae, xs, contexts, attention_layer):
    """Standard MAE forward with per-image context lists (the same call as
    test_mae_on_selected_patches.py); returns the decoder attention that the
    patched modules stored during this pass."""
    input_mask = torch.tensor(contexts, device=xs.device)
    target_rows = [[p for p in range(N_PATCHES) if p not in set(ctx)]
                   for ctx in contexts]
    target_mask = torch.tensor(target_rows, device=xs.device)
    with torch.no_grad():
        model_mae(xs, input_mask, target_mask)
    return get_decoder_attn(model_mae, attention_layer)


def select_ame(model_mae, xs_norm, b, device, attention_layer,
               random_first=False, rng=None, init_blocks=None):
    """AME selection for block size b.

    init_blocks: shared per-image initial block (flat patch list per image;
    when given it replaces the --random-first draw, so the initial glimpses
    are identical to the other random-initial variants).

    Returns (flat observed patch lists per image, observed-block steps)."""
    N = len(xs_norm)
    max_steps = N_PATCHES // (b * b)
    x = torch.stack(xs_norm).to(device)

    contexts = []
    observed_sets = []
    if init_blocks is not None:
        # shared initial block (paired comparison with the other
        # random-initial variants); no rng draw here
        contexts = [list(c) for c in init_blocks]
    elif random_first and rng is not None:
        # official --random-first option: first glimpse uniformly random
        for _ in range(N):
            top, left = rng.integers(0, GRID - b + 1, size=2)
            contexts.append(block_to_index(int(left), int(top), b, b))
    else:
        # zero step: decoder on mask tokens only, image-independent, so the
        # first glimpse is the same for every image (official behaviour)
        with torch.no_grad():
            dec_in = model_mae.mask_token.expand(
                1, N_PATCHES, -1) + model_mae.pos_decoder
            model_mae.decoder(dec_in)
            attn = get_decoder_attn(model_mae, attention_layer)
        pooled = pooled_entropy(attn, None, b)
        best = pooled.flatten(1).argmax(dim=1).item()
        Hp, Wp = pooled.shape[2], pooled.shape[3]
        top, left = np.unravel_index(best, (Hp, Wp))
        first = block_to_index(int(left), int(top), b, b)
        contexts = [list(first) for _ in range(N)]

    observed_sets = [set(c) for c in contexts]
    active = list(range(N))
    steps = [1] * N
    step = 1
    while active and step < max_steps:
        batch = torch.stack([x[i] for i in active]).to(device)
        ctx = [contexts[i] for i in active]
        observed = torch.zeros(len(active), N_PATCHES, dtype=torch.bool,
                               device=batch.device)
        observed[torch.arange(len(active), device=batch.device)[:, None],
                 torch.tensor(ctx, device=batch.device)] = True
        # chunked forward + entropy pooling: the patched attention modules
        # materialize the B x heads x 144 x 144 attention, so keep one chunk
        # of images at a time instead of the whole active batch
        pooled_parts = []
        for c0, c1 in iter_chunks(len(active)):
            attn_c = forward_attention(
                model_mae, batch[c0:c1], ctx[c0:c1], attention_layer)
            pooled_parts.append(pooled_entropy(attn_c, observed[c0:c1], b))
        pooled = torch.cat(pooled_parts, dim=0)
        A, _, Hp, Wp = pooled.shape
        pooled_flat = pooled.reshape(A, -1)
        best = pooled_flat.argmax(dim=1)
        best_val = pooled_flat.max(dim=1).values

        still = []
        for row, i in enumerate(active):
            if not torch.isfinite(best_val[row]):
                continue      # truncation: no fully-unobserved window remains
            top, left = np.unravel_index(best[row].item(), (Hp, Wp))
            window = block_to_index(int(left), int(top), b, b)
            contexts[i].extend(window)
            observed_sets[i].update(window)
            steps[i] += 1
            still.append(i)
        active = still
        step += 1
        print(f"\r    {VARIANT}: step {step}/{max_steps} "
              f"(active {len(active)}/{N})", end="", flush=True)
    print()
    return contexts, steps


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='falls back to cpu when cuda is unavailable')
    parser.add_argument('--mae_model_path', type=str,
                        default='models/mae/latest.ckpt')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--N', type=int, default=0,
                        help='number of test images to evaluate (0 = all)')
    parser.add_argument('--bs', type=int, nargs='+', default=[1, 2, 3, 4],
                        help='block sizes to evaluate')
    parser.add_argument('--attention-layer', type=int, default=-1,
                        help='decoder block whose attention supplies the '
                             'entropy (0-based, -1 = last decoder block; '
                             'official AME defaults to its last of 8 blocks)')
    parser.add_argument('--random_first', action='store_true',
                        help='select the first block at random (official '
                             '--random-first option; default off)')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for --random_first')
    parser.add_argument('--log', action='store_true',
                        help='use a log scale on the y axis')
    parser.add_argument('--gallery_n', type=int, default=10,
                        help='AME gallery PNGs for the first this-many '
                             'samples; 0 disables')
    parser.add_argument('--out_dir', type=str,
                        default='images/rec_images_tmax32')
    args = parser.parse_args()

    if args.device == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = args.device

    rng = (np.random.default_rng(args.seed) if args.random_first
           else None)

    # dataset (same preprocessing as patch_selection_reconstruction_eval.py)
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

    xs_norm = []
    for i in range(start, start + n):
        img, _ = stl10_ds[i]
        xs_norm.append(normalize(img))
    originals = {i: inv(xs_norm[i]).cpu() for i in range(n)}
    print(f'images: N={n} (start_index={args.start_index}), block sizes: '
          f'{args.bs}, device={device}')

    # MAE (the only model AME needs) + attention capture
    model = MAE().to(device)
    model.load_state_dict(
        torch.load(args.mae_model_path, map_location=device)["state_dict"])
    model.eval()
    patch_attention_modules(model.mae)
    if args.attention_layer == -1:
        attention_layer = model.mae.dec_depth - 1
    else:
        attention_layer = args.attention_layer
        assert 0 <= attention_layer < model.mae.dec_depth
    print(f'MAE: dec_depth={model.mae.dec_depth}, '
          f'attention_layer={attention_layer}')

    os.makedirs("selected_patch_index", exist_ok=True)

    for b in args.bs:
        print(f'\n===== block size b = {b} (AME) =====')
        b_dir = os.path.join(args.out_dir, f"b{b}")
        os.makedirs(b_dir, exist_ok=True)
        tile_total = N_PATCHES // (b * b)

        # ---- AME selection ----------------------------------------------- #
        contexts, stops = select_ame(
            model.mae, xs_norm, b, device, attention_layer,
            random_first=args.random_first, rng=rng)
        print(f'  {VARIANT}: steps min={min(stops)}, max={max(stops)}')

        jpath = os.path.join(
            "selected_patch_index", f"block_{b}_tmax_32_{VARIANT}.json")
        with open(jpath, "w") as f:
            json.dump(contexts, f)
        print(f'  saved: {jpath}')

        # ---- alignment ---------------------------------------------------- #
        variant_max_t = {VARIANT: min(stops)}
        t_eval = max(min(min(stops), tile_total - 1), 1)
        print(f'  aligned max t = {t_eval} '
              f'(observed-block budget=min steps={min(stops)})')
        t_list = list(range(1, t_eval + 1))

        # ---- MAE sweep over the AME prefixes ------------------------------ #
        t_list_v, mse, final_gallery = evaluate_mae(
            model, xs_norm, contexts, b, device, inv, t_eval,
            gallery_n=args.gallery_n, desc=f'AME b={b}')
        results = {VARIANT: mse}

        # ---- AME-only outputs --------------------------------------------- #
        # (AME-only chart goes to b{b}/ame/ so the stored
        #  rec_error_vs_t.png of the 7 variants is not overwritten)
        title = (f"MAE reconstruction error vs t "
                 f"(b={b}, T_max={tile_total}, N={n}, "
                 f"aligned max t={t_eval}, variant={VARIANT})")
        png_path = plot_results(results, t_list, os.path.join(b_dir, "ame"),
                                args.log, title)
        print(f'  saved: {png_path}')

        csv_path = os.path.join(b_dir, "rec_error_vs_t_ame.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["variant"] + [f"t={t}" for t in t_list])
            mean = results[VARIANT].mean(axis=1)
            writer.writerow([VARIANT] + [f"{m:.6f}" for m in mean])
        print(f'  saved: {csv_path}')

        pt_path = os.path.join(b_dir, "rec_error_vs_t_ame.pt")
        torch.save({
            "results": {VARIANT: torch.tensor(results[VARIANT])},
            "t_list": t_list,
            "block_size": b,
            "t_max": tile_total,
            "n_images": n,
            "variant_max_t": variant_max_t,
            "aligned_max_t": t_eval,
            "attention_layer": attention_layer,
        }, pt_path)
        print(f'  saved: {pt_path}')

        if args.gallery_n > 0 and final_gallery is not None:
            gallery_cache = {
                "original": dict(originals),
                "observed": {},
                "rec": {},
            }
            for i in range(final_gallery["observed"].shape[0]):
                gallery_cache["observed"].setdefault(i, {})[VARIANT] = \
                    final_gallery["observed"][i]
                gallery_cache["rec"].setdefault(i, {})[VARIANT] = \
                    final_gallery["rec"][i]
            save_gallery_images(gallery_cache, [VARIANT],
                                os.path.join(b_dir, "ame"), b, t_list[-1])

        # ---- comparison chart (7 stored variants + AME) -------------------- #
        existing_pt = os.path.join(b_dir, "rec_error_vs_t.pt")
        if os.path.isfile(existing_pt):
            data = torch.load(existing_pt, map_location="cpu")
            if int(data.get("n_images", -1)) != n:
                print(f'  [skip comparison] existing {existing_pt} was made '
                      f'with N={data.get("n_images")}, not N={n}')
            else:
                t_comp = min(int(data.get("aligned_max_t", t_eval)), t_eval)
                comp_results = {}
                for v, arr in data["results"].items():
                    comp_results[v] = np.asarray(arr, dtype=float)
                t_stored = [int(t) for t in data["t_list"]]
                keep = [i for i, t in enumerate(t_stored) if t <= t_comp]
                t_comp_list = [t_stored[i] for i in keep]
                comp_results = {v: a[keep] for v, a in comp_results.items()}
                # add the AME line on the same common axis
                ame_rows = [results[VARIANT][t - 1] for t in t_comp_list]
                ame_arr = np.stack(ame_rows, axis=0)
                comp_results[VARIANT] = ame_arr
                comp_title = (f"MAE reconstruction error vs t "
                              f"(b={b}, N={n}, aligned max t={t_comp}, "
                              f"AME vs stored variants)")
                fig_path = os.path.join(
                    b_dir, "rec_error_vs_t_comparison_ame.png")
                import patch_selection_reconstruction_eval as pse
                fig, ax = plt.subplots(figsize=(8, 5.5))
                for vi, (v, arr) in enumerate(comp_results.items()):
                    mean = arr.mean(axis=1)
                    ax.plot(t_comp_list, mean, marker="o", ms=4, lw=1.8,
                            color=(f'C{pse.VARIANTS.index(v)}'
                                   if v in pse.VARIANTS else 'k'),
                            linestyle='-' if v in pse.VARIANTS else '--',
                            label=v)
                ax.set_xlabel("observation steps t")
                ax.set_ylabel("pixel MSE (unobserved region)")
                if len(t_comp_list) <= 30:
                    ax.set_xticks(t_comp_list)
                if args.log:
                    ax.set_yscale("log")
                ax.set_title(comp_title)
                ax.legend(fontsize=8, loc="best")
                ax.grid(alpha=0.3)
                plt.tight_layout()
                fig.savefig(fig_path, dpi=150)
                plt.close(fig)
                print(f'  saved: {fig_path}')

        print_summary(results, t_list)

    print('done.')


if __name__ == "__main__":
    main()
