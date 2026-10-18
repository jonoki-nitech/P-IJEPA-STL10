# Time-series (saccade-style) visualization of the ATTENTION maps that drive
# the AME patch selection, in the same spirit as the old saccade_sim.py:
# one PNG per image, a grid of one column per selection step t with
#
#   row "context": the patches observed BEFORE the action at step t (so
#                  t=1 shows only the initial b x b block chosen from the
#                  zero-step attention),
#   row "entropy": per-patch attention entropy (-a log2 a summed over heads
#                  and keys) the next b x b block is selected from; already
#                  observed patches are zeroed and the block selected by the
#                  action at this step is outlined in white.
#
# The selection loop is exactly the one of patch_selection_ame_eval.py
# (zero step -> decoder attention -> per-patch entropy -> pooled argmax with
#  fully-unobserved-window validity -> duplicate-free truncation), run one
# image at a time so the intermediate maps can be recorded.
#
# Usage:
#   python ame_saccade_sim.py --N 1 --b 2 --T_max 10
#   python ame_saccade_sim.py            # default N=10, b=2, T_max=10
#
# Outputs: images/ame_saccade/b{b}/image_{n}.png   (one grid PNG per image)
#
# (AME forward passes are chunked by the shared CHUNK_SIZE of
#  patch_selection_reconstruction_eval.py, env CHUNK, default 1024.)

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torchvision import transforms

from data_utils import load_stl10
from mae import MAE
from patch_selection_reconstruction_eval import block_to_index
from patch_selection_ame_eval import (
    GRID,
    N_PATCHES,
    get_decoder_attn,
    patch_attention_modules,
    pooled_entropy,
)

NORM_MEAN, NORM_STD = (0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)


# --------------------------------------------------------------------------- #
#  visualization helpers (saccade_sim.py style)
# --------------------------------------------------------------------------- #
def reconstruct_img_from_patch(ori_img, patch_index, patch_size=8):
    """Observed patches of ori_img (1,3,96,96); the others black."""
    h, w = 96 // patch_size, 96 // patch_size
    patchfy = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                        p1=patch_size, p2=patch_size)
    unpatchfy = Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                          p1=patch_size, p2=patch_size, h=h, w=w)
    patches = patchfy(ori_img)
    for i in range(patches.shape[1]):
        if i not in patch_index:
            patches[:, i] = torch.zeros(patch_size ** 2 * 3)
    return unpatchfy(patches)


def patch_map_to_color(pmap, vmin, vmax, cmap="viridis", patch_size=8):
    """(144,) float patch map -> (3, 96, 96) RGB tensor (viridis)."""
    grid = pmap.reshape(GRID, GRID)
    colormap = matplotlib.colormaps[cmap]
    x_norm = torch.clamp((grid - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
    color = torch.from_numpy(colormap(x_norm.numpy())[:, :, :3])
    color = color.permute(2, 0, 1).unsqueeze(0)          # (1,3,12,12)
    return F.interpolate(color, size=(96, 96), mode="nearest")[0]


# --------------------------------------------------------------------------- #
#  AME selection with recording
# --------------------------------------------------------------------------- #
def forward_attention(model_mae, img, contexts, attention_layer):
    """Standard MAE forward with per-image context lists; returns the stored
    decoder attention (1, heads, 144, 144)."""
    input_mask = torch.tensor(contexts, device=img.device)
    target_rows = [[p for p in range(N_PATCHES) if p not in set(ctx)]
                   for ctx in contexts]
    target_mask = torch.tensor(target_rows, device=img.device)
    with torch.no_grad():
        model_mae(img, input_mask, target_mask)
    return get_decoder_attn(model_mae, attention_layer)


def select_ame_trace(model_mae, img_norm, b, device, attention_layer,
                     t_max):
    """AME selection for ONE image, recording the maps of the timeline.

    Returns (contexts, trace); trace has one dict per recorded step t, where
    every entry shows the state BEFORE the action taken at t:
      observed   : sorted flat list of patches observed up to step t-1
                   (t=1 therefore contains only the initial b x b block)
      entropy    : (144,) per-patch attention entropy used at this step
                   (0 on already-observed patches)
      next_top / next_left : anchor of the block selected by the action at
                   this step, None when the image got truncated at this step
    """
    img = img_norm.unsqueeze(0).to(device)
    max_steps = N_PATCHES // (b * b)
    n_record = min(t_max, max_steps)

    # zero step: the first block comes from the mask-token-only decoder
    with torch.no_grad():
        dec_in = model_mae.mask_token.expand(1, N_PATCHES, -1) \
            + model_mae.pos_decoder
        model_mae.decoder(dec_in)
    pooled = pooled_entropy(get_decoder_attn(model_mae, attention_layer),
                            None, b)
    best = int(pooled.flatten(1).argmax(dim=1).item())
    top, left = np.unravel_index(best, (pooled.shape[2], pooled.shape[3]))
    contexts = [block_to_index(int(left), int(top), b, b)]

    trace = []
    for step in range(1, n_record + 1):
        attn = forward_attention(model_mae, img, contexts, attention_layer)
        observed = torch.zeros(1, N_PATCHES, dtype=torch.bool,
                               device=attn.device)
        observed[0, torch.tensor(contexts[0], device=attn.device)] = True

        # per-patch attention entropy (the selection criterion)
        a = attn * (~observed).view(1, 1, N_PATCHES, 1)
        entropy = torch.nan_to_num(-a * torch.log2(a), nan=0.0)
        entropy = entropy.sum(dim=(1, 3)).squeeze(0).cpu()

        # action: next block from the pooled entropy (duplicate-free, like
        # the evaluation script)
        pooled = pooled_entropy(attn, observed, b)
        if not torch.isfinite(pooled.flatten(1).max()):
            trace.append({
                "observed": sorted(set(contexts[0])),
                "entropy": entropy.numpy(),
                "next_top": None,
                "next_left": None,
            })
            break                                    # truncation
        best = int(pooled.flatten(1).argmax(dim=1).item())
        next_top, next_left = np.unravel_index(
            best, (pooled.shape[2], pooled.shape[3]))
        trace.append({
            "observed": sorted(set(contexts[0])),
            "entropy": entropy.masked_fill(
                observed.squeeze(0).cpu(), 0.0).numpy(),
            "next_top": next_top,
            "next_left": next_left,
        })
        contexts[0].extend(
            block_to_index(int(next_left), int(next_top), b, b))

    return contexts, trace


# --------------------------------------------------------------------------- #
#  timeline figure
# --------------------------------------------------------------------------- #
def plot_timeline(context_panels, trace, b, patch_size, out_path):
    """One figure: rows context/entropy, one column per step.

    context_panels: T numpy arrays (96,96,3) of the observed image (BEFORE
    the action at each step).
    """
    T = len(trace)
    row_labels = ["context", "entropy"]
    col_labels = [f"t = {i}" for i in range(1, T + 1)]

    entropy_raw = [r["entropy"] for r in trace]

    fig, axes = plt.subplots(2, T, figsize=(max(T * 2, 4), 5.5),
                             squeeze=False)
    for r, label in enumerate(row_labels):
        for t in range(T):
            if label == "context":
                axes[r, t].imshow(context_panels[t])
            else:
                vmin = min(p.min() for p in entropy_raw)
                vmax = max(p.max() for p in entropy_raw)
                img = patch_map_to_color(torch.tensor(entropy_raw[t]),
                                         vmin, vmax,
                                         patch_size=patch_size)
                axes[r, t].imshow(img.permute(1, 2, 0).numpy())
                if trace[t]["next_top"] is not None:
                    axes[r, t].add_patch(plt.Rectangle(
                        (trace[t]["next_left"] * patch_size,
                         trace[t]["next_top"] * patch_size),
                        width=patch_size * b, height=patch_size * b,
                        fill=False, edgecolor="white", linewidth=1.5))
            axes[r, t].axis("off")
            if r == 0:
                axes[r, t].set_title(col_labels[t], fontsize=10)
    for ax, label in zip(axes[:, 0], row_labels):
        ax.set_ylabel(label, fontsize=12, rotation=0, labelpad=40,
                      va="center")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='falls back to cpu when cuda is unavailable')
    parser.add_argument('--mae_model_path', type=str,
                        default='models/mae/latest.ckpt')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--N', type=int, default=10,
                        help='number of test images to visualize')
    parser.add_argument('--b', type=int, default=2, help='block size')
    parser.add_argument('--attention-layer', type=int, default=-1,
                        help='decoder block whose attention is visualized '
                             '(0-based, -1 = last decoder block; official '
                             'AME uses its last decoder block)')
    parser.add_argument('--T_max', type=int, default=10,
                        help='number of selection steps shown per image')
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--out_dir', type=str, default='images/ame_saccade')
    args = parser.parse_args()

    if args.device == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = args.device

    normalize = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    inv = transforms.Normalize(
        mean=[-m / s for m, s in zip(NORM_MEAN, NORM_STD)],
        std=[1 / s for s in NORM_STD])

    # dataset (same preprocessing as patch_selection_ame_eval.py)
    stl10_ds = load_stl10(split="test", transform=transforms.ToTensor())
    start = args.start_index
    n = min(args.N, len(stl10_ds) - start)

    # MAE + attention capture
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
          f'attention_layer={attention_layer}, device={device}, '
          f'N={n}, b={args.b}, T_max={args.T_max}')

    out_dir = os.path.join(args.out_dir, f"b{args.b}")
    os.makedirs(out_dir, exist_ok=True)

    for offset in range(n):
        idx = start + offset
        img, _ = stl10_ds[idx]
        img_norm = normalize(img)

        contexts, trace = select_ame_trace(
            model.mae, img_norm, args.b, device, attention_layer,
            args.T_max)
        n_observed = len(set(contexts[0]))
        print(f'image {idx}: {n_observed}/{N_PATCHES} patches observed in '
              f'{len(trace)} recorded steps')

        # context panels need the actual image, so they are recomposed here
        img_display = inv(img_norm)          # denormalize for display
        context_panels = []
        for record in trace:
            context_img = reconstruct_img_from_patch(
                img_display.unsqueeze(0), record["observed"],
                patch_size=args.patch_size)
            context_panels.append(
                context_img.squeeze(0).permute(1, 2, 0).numpy())

        plot_timeline(context_panels, trace, args.b, args.patch_size,
                      os.path.join(out_dir, f"image_{idx}.png"))
        print(f'  saved: {os.path.join(out_dir, f"image_{idx}.png")}')

    print('done.')


if __name__ == "__main__":
    main()
