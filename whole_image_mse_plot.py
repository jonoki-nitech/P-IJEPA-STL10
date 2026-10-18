# Post-hoc re-plot of the reconstruction-error-vs-t results with the
# WHOLE-IMAGE denominator.
#
# The stored rec_error_vs_t.pt arrays (patch_selection_reconstruction_eval.py)
# use
#     mse = sum of squared error over the UNOBSERVED pixels (RGB sum)
#           / number of unobserved pixels,
# i.e. the evaluated set shrinks with t and the tail is dominated by the last
# remaining patches. Here each value is re-scaled so the denominator is ALL
# pixels of the frame:
#     mse_whole(t) = mse_stored(t) * (144 - t * b^2) / 144
# (unobserved pixel count / total pixel count on the 12x12 patch grid; the
#  numerator, RGB sum, is unchanged). Because the selections are
#  duplicate-free, every image on the common aligned axis has exactly
#  t * b^2 observed patches, so this factor is exact. The result is the
#  image-level MSE (observed patches contribute zero: they are the true
#  pixels). Note the whole-image scale differs from the stored scale by this
#  t-dependent factor; the RGB-sum convention (3x the per-channel mean) is
#  kept as in the original pipeline.
#
# Usage (defaults to every b under images/rec_images_tmax32):
#   python whole_image_mse_plot.py
#   python whole_image_mse_plot.py --log
#   python whole_image_mse_plot.py --pt images/rec_images_tmax32/b2/rec_error_vs_t.pt
#
# Outputs (one pair per pt):
#   b{b}/rec_error_vs_t_wholeimg.png   re-scaled chart (does not touch the
#                                      original rec_error_vs_t.png/csv/pt)
#   b{b}/rec_error_vs_t_wholeimg.csv   re-scaled mean rows

import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

N_PATCHES = 144
FRAME_PIXELS = 96 * 96


def load_pt(pt_path):
    data = torch.load(pt_path, map_location="cpu")
    b = int(data["block_size"])
    t_list = [int(t) for t in data["t_list"]]
    results = {v: np.asarray(a, dtype=float) for v, a in data["results"].items()}
    return b, t_list, results, data


def convert_to_whole_image(mse, t_list, b):
    """(T, N) unobserved-region MSE -> whole-image MSE."""
    factors = np.asarray(
        [(N_PATCHES - t * b * b) / N_PATCHES for t in t_list])[:, None]
    return mse * factors


def plot_whole_image(results, t_list, b, out_png, title, log_scale=False):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for variant, arr in results.items():
        ax.plot(t_list, arr.mean(axis=1), marker="o", ms=4, lw=1.8,
                label=variant)
    ax.set_xlabel("observation steps t")
    ax.set_ylabel("pixel MSE (whole image)")
    if len(t_list) <= 30:
        ax.set_xticks(t_list)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def save_csv(results, t_list, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant"] + [f"t={t}" for t in t_list])
        for variant, arr in results.items():
            mean = arr.mean(axis=1)
            writer.writerow([variant] + [f"{m:.6f}" for m in mean])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt', default='', help='specific pt path; '
                        'default: images/rec_images_tmax32/b*/rec_error_vs_t.pt')
    parser.add_argument('--out_dir', default='images/rec_images_tmax32')
    parser.add_argument('--log', action='store_true',
                        help='log scale on the y axis')
    args = parser.parse_args()

    if args.pt:
        pt_paths = [args.pt]
    else:
        pt_paths = sorted(glob.glob(
            os.path.join(args.out_dir, "b*", "rec_error_vs_t.pt")))
    if not pt_paths:
        raise SystemExit(f"no rec_error_vs_t.pt found under {args.out_dir}")

    for pt_path in pt_paths:
        b, t_list, results, data = load_pt(pt_path)
        print(f"\n===== pt: {pt_path} "
              f"(b={b}, N={data.get('n_images', '?')}, "
              f"aligned max t={data.get('aligned_max_t', '?')}) =====")
        conv = {v: convert_to_whole_image(a, t_list, b)
                for v, a in results.items()}

        out_dir = os.path.join(args.out_dir, f"b{b}")
        title = (f"MAE reconstruction error vs t (whole-image MSE, "
                 f"b={b}, T_max={data.get('t_max', '?')}, "
                 f"N={data.get('n_images', '?')}, "
                 f"aligned max t={data.get('aligned_max_t', '?')})")
        png_path = plot_whole_image(conv, t_list, b,
                                    os.path.join(out_dir,
                                                 "rec_error_vs_t_wholeimg.png"),
                                    title, args.log)
        print(f"saved: {png_path}")
        csv_path = os.path.join(out_dir, "rec_error_vs_t_wholeimg.csv")
        save_csv(conv, t_list, csv_path)
        print(f"saved: {csv_path}")

        for variant, arr in conv.items():
            mean = arr.mean(axis=1)
            row = "  ".join(f"t={t}:{m:.4f}" for t, m in zip(t_list, mean))
            print(f"{variant:>22}: {row}")


if __name__ == "__main__":
    main()
