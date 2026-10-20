#!/usr/bin/env bash
# =============================================================================
# revision/run.sh
#
#   Two experiments, end to end, for the hand-off run:
#     Step 0: uncertainty (sigma) calibration (seed-independent, runs once)
#             -> images/calibration/
#     Step 1: SEEDS (default "0 1 2 3 4") of strategies x block sizes b=1..4,
#             one patch_selection_reconstruction_eval.py call per seed
#             (selection + MAE sweep; all strategies of a seed share one
#              aligned t axis)
#
#   Per-seed outputs (nothing of the other seeds is touched):
#     images/rec_images_tmax32/seed_{s}/b{b}/
#         rec_error_vs_t.png / .csv / .pt, sigma_sum_traj.pt,
#         gallery/nXXXX.png (first --gallery_n samples)
#     selected_patch_index/seed_{s}/block_{b}_tmax_32_{variant}.json
#     result_seed_{s}.zip     packed at the end of the seed's run (the seed's
#                             artefacts + the shared calibration)
#     result.zip              bundles all result_seed_{s}.zip of the run
#                             (nested, stored without re-compression)
#
#   Strategies (default VARIANTS_LIST = 7 baselines + AME + lookahead):
#     sigma_greedy    : sigma-based selection, random initial block
#     sigma_center    : sigma-based selection, center initial block
#     random_greedy   : random selection, random initial block
#     random_center   : random selection, center initial block
#     center_outward  : fixed, center-outward order
#     raster          : fixed, raster order (top-left initial)
#     raster_center   : fixed, raster order (center initial block)
#     ame             : Attention-Map Entropy baseline (Pardyl et al.,
#                       IJCAI 2023) on the MAE decoder attention;
#                       random initial block (the shared per-image draw)
#     sigma_lookahead : 1-step lookahead sigma minimisation, random initial
#                       block (simulate the top-M candidates with their
#                       mu-predicted features and pick the one that minimises
#                       the total sigma left unobserved) -- only for block
#                       sizes in LOOKAHEAD_BS (default "2 3 4": b=1 is skipped
#                       because its 144-step candidate simulation is by far
#                       the most expensive part)
#   sigma_lookahead_center (center-init lookahead) also exists in the script,
#   but is NOT part of this multi-seed sweep: it is seed-independent, so add
#   it to VARIANTS_LIST once if it is ever needed.
#
#   While selecting, the sigma-based and lookahead variants record, per image
#   and step, the total sigma still unobserved; the lookahead variants also
#   record the sigma rank of each chosen block. All of it lands in
#   images/rec_images_tmax32/seed_{s}/b{b}/sigma_sum_traj.pt.
#
#   Each variant is truncated at its observable budget (no fully-unobserved
#   b x b window remains -> the image stops observing, even if steps are
#   left); the MAE evaluation aligns all strategies of a block size at
#   aligned_max_t = min(144/b^2 - 1, min over (variant, image) budgets),
#   i.e. plotting stops where duplicated observations would begin.
#   The random-initial variants (sigma_greedy / random_greedy / ame /
#   sigma_lookahead) share ONE initial block per image (paired comparison;
#   different images keep independent draws).
#
#   Step 0 (calibration) outputs images/calibration/:
#   reliability diagram + metrics CSV + calibration_points.pt (per-point
#   sigma / prediction error + ids, for later re-analysis:
#   python calibration_eval.py --points_pt <path>),
#   context (b=4 at "4:4") patches excluded from the calibration points.
#   NOTE: a (re)run OVERWRITES the artefacts of the seeds it processes and
#   the calibration directory.
#
# Usage:
#   PYTHON=/path/to/python bash revision/run.sh
#   (PY defaults to "python". Scripts auto-fall back to CPU when CUDA is
#    unavailable, so no device flag is needed unless you want to pin one.)
#
#   Overrides:
#     N=100 PYTHON=... bash revision/run.sh      # first 100 test images only
#     SEEDS="0 1 2" ...                          # seeds to run (default 0-4)
#     BS="2 4" ...                               # block sizes (default 1 2 3 4)
#     DEVICE=cpu ...                             # pin the device
#     GALLERY_N=0 ...                            # skip gallery PNGs
#     ATTENTION_LAYER=2 ...                      # AME decoder attention block
#     CHUNK=512 ...                              # chunk size (default 1024)
#     M=8 ...                                    # lookahead candidate prescreen
#                                                #  (default 16; ALL_CANDIDATES=1
#                                                #  disables the prescreen)
#     LOOKAHEAD_BS="1 2 3 4" ...                 # block sizes with lookahead
#                                                #  (default "2 3 4"; add 1 for
#                                                #  the full but very slow set)
#     VARIANTS_LIST="..." ...                    # strategies (default: the 9
#                                                #  above, WITHOUT
#                                                #  sigma_lookahead_center)
#     SKIP_ZIP=1 ...                             # skip packaging (the per-seed
#                                                #  zips AND their result.zip)
#     ZIP_PREFIX=result ...                      # zip base name
#
#   Runtime on one 24 GB GPU at N=8000: ~3-3.5 h per seed (~17-19 h for the
#   five seeds), split roughly into the 8 fast variants' selection ~40-60 min,
#   the lookahead selection (b=2-4) ~30-50 min and the MAE sweeps ~1.5-2 h.
#
#   Requires: torch, torchvision, numpy, matplotlib, einops, tqdm.
#   Models: I-JEPA at
#   models/embdim=512_epoch=300_patchsize=8/training_model.pth and MAE at
#   models/mae/latest.ckpt (both --model_path / --mae_model_path overridable
#   in the patch-selection call below). STL10 data is resolved by
#   data_utils.py: if the server data directory exists it is used directly,
#   otherwise only the required splits are streamed into data/stl10_binary/.
# =============================================================================

set -e
set -u

export PYTHONUNBUFFERED=1 # real-time log lines even when piped through tee

PY=${PYTHON:-python}
N=${N:-0}               # number of test images (0 = all 8000)
BS=${BS:-"1 2 3 4"}      # block sizes to evaluate
SEEDS=${SEEDS:-"0 1 2 3 4"} # seeds for the stochastic strategies
DEVICE=${DEVICE:-cuda:0} # override with e.g. DEVICE=cpu
GALLERY_N=${GALLERY_N:-10}
AME_LAYER=${AME_LAYER:--1} # AME attention layer (-1 = last decoder block)
LOOKAHEAD_M=${M:-16}     # lookahead candidate prescreen size
LOOKAHEAD_BS=${LOOKAHEAD_BS:-"2 3 4"} # block sizes that run the lookahead
VARIANTS_LIST=${VARIANTS_LIST:-"sigma_greedy sigma_center random_greedy random_center center_outward raster raster_center ame sigma_lookahead"}
LOOKAHEAD_FLAGS=""
if [ "${ALL_CANDIDATES:-0}" = "1" ]; then
    LOOKAHEAD_FLAGS="--all-candidates"
fi
export CHUNK=${CHUNK:-1024} # chunk size of the batched forward passes

echo "python : $($PY --version 2>&1)"
echo "params : N=$N   bs=$BS   seeds=[$SEEDS]   device=$DEVICE   gallery_n=$GALLERY_N   ame_attention_layer=$AME_LAYER   lookahead_M=$LOOKAHEAD_M   lookahead_bs=[$LOOKAHEAD_BS]"

# -----------------------------------------------------------------------------
# Step 0: calibration of the sigma predictor (seed-independent, once)
#   default: b=4, one b x b-patch evaluation position per image ("4:4"),
#   context patches excluded from the calibration points.
# -----------------------------------------------------------------------------
echo ""
echo "=== Step 0: calibration ==========================================="
$PY calibration_eval.py \
    --b 4 \
    --positions 4:4 \
    --N "$N" \
    --device "$DEVICE"

# -----------------------------------------------------------------------------
# Step 1: per-seed selection + MAE sweep + per-seed zip
# -----------------------------------------------------------------------------
for SEED in $SEEDS; do
    SEED_DIR="seed_${SEED}"
    REC_DIR="images/rec_images_tmax32/${SEED_DIR}"
    JSON_DIR="selected_patch_index/${SEED_DIR}"

    echo ""
    echo "=== Step 1: seed=$SEED (selection + MAE reconstruction vs t) ======"
    $PY patch_selection_reconstruction_eval.py \
        --bs $BS \
        --N "$N" \
        --seed "$SEED" \
        --attention-layer "$AME_LAYER" \
        --M "$LOOKAHEAD_M" \
        --lookahead-bs $LOOKAHEAD_BS \
        --variants $VARIANTS_LIST \
        $LOOKAHEAD_FLAGS \
        --gallery_n "$GALLERY_N" \
        --out_dir "$REC_DIR" \
        --json_dir "$JSON_DIR" \
        --device "$DEVICE"

    # -------------------------------------------------------------------------
    # Package this seed's artefacts (+ the shared calibration) into one zip
    # -------------------------------------------------------------------------
    if [ "${SKIP_ZIP:-0}" != "1" ]; then
        ZIP="${ZIP_PREFIX:-result}_${SEED_DIR}.zip"
        echo ""
        echo "=== Pack artefacts into $ZIP ===================================="
        rm -f "$ZIP"
        "$PY" - "$ZIP" "$SEED_DIR" <<'PYEOF'
import os, sys, zipfile
zip_name, seed_dir = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for root in ("images/calibration",
                 f"images/rec_images_tmax32/{seed_dir}",
                 f"selected_patch_index/{seed_dir}"):
        for dirpath, _, files in os.walk(root):
            for f in files:
                zf.write(os.path.join(dirpath, f))
print(f"created: {zip_name} ({os.path.getsize(zip_name) / 1e6:.1f} MB)")
PYEOF
    fi
done

# -----------------------------------------------------------------------------
# Bundle all per-seed zips into one result zip (nested: the seed zips are
# stored as-is, without re-compression)
# -----------------------------------------------------------------------------
if [ "${SKIP_ZIP:-0}" != "1" ]; then
    RESULT_ZIP="${ZIP_PREFIX:-result}.zip"
    echo ""
    echo "=== Bundle per-seed zips into $RESULT_ZIP ======================="
    rm -f "$RESULT_ZIP"
    "$PY" - "$RESULT_ZIP" "${ZIP_PREFIX:-result}_seed_" "$SEEDS" <<'PYEOF'
import os, sys, zipfile
result_zip, prefix, seeds = sys.argv[1], sys.argv[2], sys.argv[3]
seed_zips = [f"{prefix}{s}.zip" for s in seeds.split()]
with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_STORED) as zf:
    for n in seed_zips:
        zf.write(n, os.path.basename(n))
print(f"created: {result_zip} ({os.path.getsize(result_zip) / 1e6:.1f} MB, "
      f"{len(seed_zips)} seed zips)")
PYEOF
fi

# -----------------------------------------------------------------------------
# Summary of the produced artefacts
# -----------------------------------------------------------------------------
echo ""
echo "=== Done. Produced artefacts ======================================"
echo "--- calibration -------------------------------------------------"
ls -d images/calibration 2>/dev/null || echo "(missing: images/calibration)"
for SEED in $SEEDS; do
    SEED_DIR="seed_${SEED}"
    echo "--- seed $SEED --------------------------------------------------"
    ls "selected_patch_index/${SEED_DIR}"/*tmax_32*.json 2>/dev/null \
        || echo "(missing: selection JSONs)"
    for b in 1 2 3 4; do
        ls "images/rec_images_tmax32/${SEED_DIR}/b$b"/rec_error_vs_t.pt 2>/dev/null || true
        ls -d "images/rec_images_tmax32/${SEED_DIR}/b$b/gallery" 2>/dev/null || true
    done
    ls "result_seed_${SEED_DIR}.zip" 2>/dev/null || true
done
ls "${ZIP_PREFIX:-result}.zip" 2>/dev/null || true

exit 0
