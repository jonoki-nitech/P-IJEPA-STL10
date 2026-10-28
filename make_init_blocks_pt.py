# Extract the shared per-image initial b x b blocks from a previous
# patch-selection run's selection JSONs and save them for later runs.
#
# Why: the shared initial block is drawn once per image from a numpy rng
# whose stream continues across block sizes and across the random variants
# (random_greedy / random_center consume draws per step). A follow-up run
# with a different --bs / --variants set therefore cannot reproduce the
# initial blocks by seeding alone. This tool recovers them directly from
# the stored selections instead: the FIRST b*b observed patches of each
# image ARE the initial block (verified identical across the random-initial
# variants: sigma_greedy / random_greedy / ame / sigma_lookahead).
#
# Output (torch.save):
#   {1: [[patch, ...], ...],          per-image initial blocks for each b
#    ...,
#    "n_images": N,
#    "sources": {b: {variant: json-path, ...}},
#    "description": "..."}
#
# The consumer is patch_selection_reconstruction_eval.py --init-blocks-pt.
#
# Usage (old 5-seed results as extracted zips, e.g.
#        results/STL10_1/result_seed_0/selected_patch_index/seed_0):
#   python make_init_blocks_pt.py \
#       --json_dir results/STL10_1/result_seed_0/selected_patch_index/seed_0 \
#       --bs 1 2 3 4 --out init_blocks_seed_0.pt
#
# The --bs list may be a subset (e.g. only `--bs 2 3 4` when b=1 is dropped
# from the new run); only the sizes present in the pt may run later.

import argparse
import json
import os

import torch

INIT_CANDIDATE_VARIANTS = ["sigma_greedy", "random_greedy", "ame",
                           "sigma_lookahead", "mc_greedy", "mc_lookahead"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir', type=str, required=True,
                        help='directory holding block_{b}_tmax_32_'
                             '{variant}.json files of a previous run')
    parser.add_argument('--bs', type=int, nargs='+', default=[1, 2, 3, 4],
                        help='block sizes to extract (default: all four)')
    parser.add_argument('--variants', type=str, nargs='+',
                        default=None, metavar='VARIANT',
                        help='random-initial variant JSONs to read the '
                             'initial blocks from (default: try ' +
                             ' / '.join(INIT_CANDIDATE_VARIANTS) +
                             '; the first existing is primary, the others '
                             'are cross-checked against it)')
    parser.add_argument('--n', type=int, default=0,
                        help='expected number of images per JSON '
                             '(0 = take it from the first file and '
                             'cross-check the rest)')
    parser.add_argument('--out', type=str, default='init_blocks.pt',
                        help='output pt path')
    args = parser.parse_args()

    variants = args.variants or INIT_CANDIDATE_VARIANTS
    data = {}
    sources = {}
    n_images = args.n if args.n else None

    for b in args.bs:
        blocks = None
        src = {}
        for variant in variants:
            jpath = os.path.join(args.json_dir,
                                 f"block_{b}_tmax_32_{variant}.json")
            if not os.path.isfile(jpath):
                continue
            with open(jpath) as f:
                seqs = json.load(f)
            if n_images is None:
                n_images = len(seqs)
            if n_images != len(seqs):
                parser.error(f'{jpath} holds {len(seqs)} images, '
                             f'expected N={n_images}')
            first = [list(c[:b * b]) for c in seqs]
            if any(any(p < 0 or p >= 144 for p in blk) for blk in first):
                parser.error(f'{jpath}: block_{b} contains invalid patch '
                             f'indices (outside 0..143)')
            if blocks is None:
                blocks = first
                src[variant] = 'primary'
            else:
                if blocks != first:
                    parser.error(
                        f'{jpath}: the initial blocks of "{variant}" DIFFER '
                        f'from the primary source for b={b} (image examples '
                        f'differ at index ' + str(
                            next(i for i, (a, c) in
                                 enumerate(zip(blocks, first))
                                 if a != c)) + '); the JSONs are from '
                        f'different runs or an older code version')
                src[variant] = 'verified identical'
        if blocks is None:
            parser.error(
                f'no candidate initial-block JSON found for b={b} in '
                f'{args.json_dir} (looked for: ' +
                ', '.join(f'block_{b}_tmax_32_{v}.json' for v in variants)
                + ')')
        data[b] = blocks
        sources[b] = src

    torch.save({
        **data,
        "n_images": n_images,
        "sources": sources,
        "description": ("per-image initial b x b blocks extracted as the "
                        "first b*b observed patches of stored selection "
                        "sequences (identical across the random-initial "
                        "variants); consumed by "
                        "patch_selection_reconstruction_eval.py "
                        "--init-blocks-pt"),
    }, args.out)
    print(f'created: {args.out} '
          f'(N={n_images}, block sizes={args.bs}, N x b*b ints per size)')
    print(f'  sources: {sources}')
    print(f'use with: patch_selection_reconstruction_eval.py '
          f'--init-blocks-pt {args.out}')


if __name__ == "__main__":
    main()
