#!/usr/bin/env bash
# =============================================================================
# revision/run.sh -- MC-Dropout baseline EVALUATION pipeline
# (the training script that produced the checkpoint lives in
#  revision/archive/run_mcdropout_train.sh)
#
#   Evaluates the MC-Dropout I-JEPA with the same protocol as the archived
#   5-seed sigma-model sweep (revision/archive/run.sh), so the two runs can
#   be compared later from their stored artefacts.
#
#     Step 0: MC-uncertainty calibration (seed-independent, runs once)
#             -> images/calibration_mcdropout/
#     Step 0b: per-seed init-block extraction from the OLD sigma run's
#             selection JSONs (make_init_blocks_pt.py) -> init_blocks/seed_{s}.pt
#             The old and new runs were seeded independently and the numpy
#             draw chain depends on the variant set, so seeding alone cannot
#             reproduce the old initial blocks; they are recovered from the
#             old run's stored selections instead (the first b*b observed
#             patches of each image ARE the initial block). This keeps the
#             per-image comparison paired on the initialization luck.
#     Step 1: SEEDS (default "0 1 2 3 4") of mc_* strategies x block sizes
#             BS (default "2 3 4"; b=1 is dropped like the lookahead stage of
#             the old sweep -- its 144-step lookahead candidate simulation is
#             by far the most expensive part), one
#             patch_selection_reconstruction_eval.py call per seed
#             (selection + MAE sweep; all strategies of a seed share one
#              aligned t axis, incl. sigma_sum_traj.pt for the MC variance)
#
#   Per-seed outputs (nothing of the other seeds is touched):
#     images/rec_images_mcdropout/seed_{s}/b{b}/
#         rec_error_vs_t.png / .csv / .pt, sigma_sum_traj.pt,
#         gallery/nXXXX.png (first --gallery_n samples)
#     selected_patch_index/mcdropout/seed_{s}/block_{b}_tmax_32_{variant}.json
#     init_blocks/seed_{s}.pt (shared init blocks, reproduced from the old run)
#     result_mcdropout_seed_{s}.zip   packed at the end of the seed's run
#                                     (the seed's artefacts + the shared
#                                     calibration + the init blocks pt)
#     result_mcdropout.zip            bundles all seed zips (nested, stored
#                                     without re-compression)
#
#   Strategies (default VARIANTS_LIST = mc_greedy + mc_lookahead):
#     mc_greedy           : MC-variance greedy stepwise minimisation,
#                           random initial block (the shared per-image draw,
#                           restored from the old run)
#     mc_center           : MC-variance greedy, center initial block
#     mc_lookahead        : 1-step lookahead MC-variance minimisation, random
#                           initial block (simulate the top-M candidates with
#                           their K-sample predictors and pick the one that
#                           minimises the total MC variance left unobserved)
#                           -- only for block sizes in LOOKAHEAD_BS
#     mc_lookahead_center : center-init version of the above
#   The center variants are NOT in the default sweep (identical algorithm,
#   only a different init; the old sweep has no sigma_lookahead_center
#   counterpart either). Add them via VARIANTS_LIST=... when the sensitivity
#   to the initialization position is of interest.
#
#   The predictor's dropout is ON during selection (K=MC_K forwards per
#   step); encoders always run in eval mode once. The recorded residual
#   error uses the K-sample AVERAGE prediction (no dropout noise in the
#   evaluation).
#
#   NOTE: a (re)run OVERWRITES the artefacts of the seeds it processes,
#   the calibration directory and init_blocks/. The run is idempotent for
#   the init blocks: they are re-extracted only when the pt is missing
#   (or FORCE_INIT=1 is set).
#
# Usage:
#   PYTHON=/path/to/python bash revision/run.sh
#   (PY defaults to "python". Scripts auto-fall back to CPU when CUDA is
#    unavailable, so no device flag is needed unless you want to pin one.)
#
#   Overrides:
#     N=100 PYTHON=... bash revision/run.sh      # first 100 test images only
#     SEEDS="0 1" ...                            # seeds to run (default 0-4)
#     BS="2 4" ...                               # block sizes (default 2 3 4)
#     DEVICE=cpu ...                             # pin the device
#     OLD_JSON_BASE=/path/to/old/results ...     # directory holding the OLD
#                                                #  run's per-seed JSON dirs
#                                                #  (defaults to
#                                                #  selected_patch_index,
#                                                #  i.e. the old sweep's own
#                                                #  layout result_seed_{s} zips
#                                                #  extract to); each seed dir
#                                                #  must contain the
#                                                #  block_{b}_tmax_32_*.json
#     MC_K=8 ...                                 # MC samples per step (default 16)
#     MC_SEED=0 ...                              # torch seed for the dropout stream
#     MU_DROPOUT=0.1 ...                         # predictor dropout p (constructor)
#     GALLERY_N=0 ...                            # skip gallery PNGs
#     LOOKAHEAD_M=8 ...                          # lookahead candidate prescreen
#                                                #  (default 16; ALL_CANDIDATES=1
#                                                #  disables the prescreen; halving
#                                                #  halves the lookahead cost)
#     LOOKAHEAD_BS="2 3 4" ...                   # block sizes with lookahead
#     VARIANTS_LIST="mc_greedy mc_center" ...    # strategies (default: the 4
#                                                #  mc_* above)
#     FORCE_INIT=1 ...                           # re-extract the init blocks
#     SKIP_INIT=1 ...                            # skip the init-block step
#                                                #  (then --init-blocks-pt is
#                                                #  NOT passed: the seeds re-draw
#                                                #  the initial blocks instead,
#                                                #  why: the old pairing is lost)
#     SKIP_CALIB=1 ...                           # skip Step 0
#     SKIP_ZIP=1 ...                             # skip packaging (the per-seed
#                                                #  zips AND their result zip)
#     ZIP_PREFIX=result_mcdropout ...            # zip base name
#
#   Runtime at N=8000, K=16 (24 GB GPU): roughly a day per seed, dominated
#   by the 1-step lookahead (MC_K x M candidate simulations per step; the old
#   sigma lookahead took ~30-50 min per seed and this runs its candidate
#   sim MC_K times). The greedy/center selection is about as fast as the old
#   8-variant sweep and the MAE sweeps are unchanged (~1.5-2 h). For a
#   cheaper lookahead: LOOKAHEAD_M=8, and/or a smaller MC_K (document any
#   deviation from the defaults when reporting).
#
#   Requires: torch, torchvision, numpy, matplotlib, einops, tqdm.
#   Models: MC-Dropout I-JEPA at
#   models/mcdropout_embdim=512_epoch=300_patchsize=8/training_model.pth and
#   MAE at models/mae/latest.ckpt (both --model_path / --mae_model_path
#   overridable in the patch-selection call below). STL10 data is resolved
#   by data_utils.py: if the server data directory exists it is used
#   directly, otherwise only the required splits are streamed into
#   data/stl10_binary/.
# =============================================================================

set -e
set -u

export PYTHONUNBUFFERED=1 # real-time log lines even when piped through tee

PY=${PYTHON:-python}
N=${N:-0}                 # number of test images (0 = all 8000)
BS=${BS:-"2 3 4"}         # block sizes to evaluate (=the old sweep minus b=1)
SEEDS=${SEEDS:-"0 1 2 3 4"} # seeds for the stochastic strategies
DEVICE=${DEVICE:-cuda:0}  # override with e.g. DEVICE=cpu
GALLERY_N=${GALLERY_N:-10}
LOOKAHEAD_M=${M:-16}      # lookahead candidate prescreen size
LOOKAHEAD_BS=${LOOKAHEAD_BS:-"2 3 4"} # block sizes that run the lookahead
VARIANTS_LIST=${VARIANTS_LIST:-"mc_greedy mc_lookahead"}
MC_K=${MC_K:-16}          # MC-Dropout samples per step
MC_SEED=${MC_SEED:-0}     # torch seed of the dropout sample stream
MU_DROPOUT=${MU_DROPOUT:-0.1} # predictor dropout p (IJEPA constructor arg)
MODEL_PATH=${MODEL_PATH:-models/mcdropout_embdim=512_epoch=300_patchsize=8/training_model.pth}
MAE_MODEL_PATH=${MAE_MODEL_PATH:-models/mae/latest.ckpt}
export CHUNK=${CHUNK:-1024} # chunk size of the batched forward passes

CAL_DIR="images/calibration_mcdropout"
INIT_DIR="init_blocks"
OLD_JSON_BASE=${OLD_JSON_BASE:-selected_patch_index} # old sweep's per-seed JSON dirs

LOOKAHEAD_FLAGS=""
if [ "${ALL_CANDIDATES:-0}" = "1" ]; then
    LOOKAHEAD_FLAGS="--all-candidates"
fi

echo "python : $($PY --version 2>&1)"
echo "params : N=$N   bs=$BS   seeds=[$SEEDS]   device=$DEVICE   gallery_n=$GALLERY_N   mc_k=$MC_K   mc_seed=$MC_SEED   mu_dropout=$MU_DROPOUT   lookahead_M=$LOOKAHEAD_M   lookahead_bs=[$LOOKAHEAD_BS]   old_json_base=$OLD_JSON_BASE"

# per-seed sanity check of the old-JSON locations before any heavy step
if [ "${SKIP_INIT:-0}" != "1" ]; then
    for SEED in $SEEDS; do
        DIR="${OLD_JSON_BASE}/seed_${SEED}"
        if [ ! -f "${DIR}/block_2_tmax_32_sigma_greedy.json" ] \
           && [ ! -f "${DIR}/block_2_tmax_32_random_greedy.json" ] \
           && [ ! -f "${DIR}/block_1_tmax_32_sigma_greedy.json" ]; then
            echo ""
            echo "The old run's selection JSONs are missing at ${DIR}/" >&2
            echo "(looked for block_{b}_tmax_32_sigma_greedy.json or" >&2
            echo " block_{b}_tmax_32_random_greedy.json)" >&2
            echo "Set OLD_JSON_BASE=/path/to/dir-holding-seed_{s}-json-dirs" >&2
            echo "(e.g. one level above the old run's selected_patch_index)," >&2
            echo "or SKIP_INIT=1 to re-draw the initial blocks instead of" >&2
            echo "pairing with the old sweep." >&2
            exit 1
        fi
    done
fi

# -----------------------------------------------------------------------------
# Step 0: calibration of the MC-Dropout uncertainty (seed-independent, once)
#   default: b=4, one b x b-patch evaluation position per image ("4:4"),
#   context patches excluded from the calibration points. The recorded
#   uncertainty is the pooled K-sample prediction variance of the
#   dropout-enabled predictor (same definition as the mc_* strategies).
# -----------------------------------------------------------------------------
if [ "${SKIP_CALIB:-0}" != "1" ]; then
    echo ""
    echo "=== Step 0: MC calibration ========================================"
    $PY calibration_eval.py \
        --model_path "$MODEL_PATH" \
        --b 4 \
        --positions 4:4 \
        --N "$N" \
        --mc-samples "$MC_K" \
        --mc-seed "$MC_SEED" \
        --mu-dropout "$MU_DROPOUT" \
        --out_dir "$CAL_DIR" \
        --device "$DEVICE"
fi

# -----------------------------------------------------------------------------
# Step 0b: per-seed init-block extraction from the old run's selections
#   (idempotent: skipped when the pt already exists, unless FORCE_INIT=1)
# -----------------------------------------------------------------------------
if [ "${SKIP_INIT:-0}" != "1" ]; then
    echo ""
    echo "=== Step 0b: init blocks from the old run ========================="
    for SEED in $SEEDS; do
        INIT_PT="${INIT_DIR}/seed_${SEED}.pt"
        if [ -f "$INIT_PT" ] && [ "${FORCE_INIT:-0}" != "1" ]; then
            echo "already there: ${INIT_PT} (FORCE_INIT=1 to redo)"
            continue
        fi
        mkdir -p "$INIT_DIR"
        $PY make_init_blocks_pt.py \
            --json_dir "${OLD_JSON_BASE}/seed_${SEED}" \
            --bs $BS \
            --out "$INIT_PT"
    done
fi

# -----------------------------------------------------------------------------
# Step 1: per-seed selection + MAE sweep + per-seed zip
# -----------------------------------------------------------------------------
for SEED in $SEEDS; do
    SEED_DIR="seed_${SEED}"
    REC_DIR="images/rec_images_mcdropout/${SEED_DIR}"
    JSON_DIR="selected_patch_index/mcdropout/${SEED_DIR}"
    INIT_ARGS=""
    if [ "${SKIP_INIT:-0}" != "1" ]; then
        INIT_ARGS="--init-blocks-pt ${INIT_DIR}/seed_${SEED}.pt"
    fi

    echo ""
    echo "=== Step 1: seed=$SEED (selection + MAE reconstruction vs t) ======"
    $PY patch_selection_reconstruction_eval.py \
        --model_path "$MODEL_PATH" \
        --mae_model_path "$MAE_MODEL_PATH" \
        --bs $BS \
        --N "$N" \
        --seed "$SEED" \
        --M "$LOOKAHEAD_M" \
        --lookahead-bs $LOOKAHEAD_BS \
        --variants $VARIANTS_LIST \
        $LOOKAHEAD_FLAGS \
        --mc-samples "$MC_K" \
        --mc-seed "$MC_SEED" \
        --mu-dropout "$MU_DROPOUT" \
        $INIT_ARGS \
        --gallery_n "$GALLERY_N" \
        --out_dir "$REC_DIR" \
        --json_dir "$JSON_DIR" \
        --device "$DEVICE"

    # -------------------------------------------------------------------------
    # Package this seed's artefacts (+ the shared calibration and the seed's
    # init blocks) into one zip
    # -------------------------------------------------------------------------
    if [ "${SKIP_ZIP:-0}" != "1" ]; then
        ZIP="${ZIP_PREFIX:-result_mcdropout}_${SEED_DIR}.zip"
        echo ""
        echo "=== Pack artefacts into $ZIP ===================================="
        rm -f "$ZIP"
        "$PY" - "$ZIP" "$SEED_DIR" <<'PYEOF'
import os, sys, zipfile
zip_name, seed_dir = sys.argv[1], sys.argv[2]
roots = ["images/calibration_mcdropout",
         f"images/rec_images_mcdropout/{seed_dir}",
         f"selected_patch_index/mcdropout/{seed_dir}"]
init_pt = f"init_blocks/{seed_dir}.pt"
if os.path.isfile(init_pt):
    roots.append(init_pt)
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for root in roots:
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
    RESULT_ZIP="${ZIP_PREFIX:-result_mcdropout}.zip"
    echo ""
    echo "=== Bundle per-seed zips into $RESULT_ZIP ======================="
    rm -f "$RESULT_ZIP"
    "$PY" - "$RESULT_ZIP" "${ZIP_PREFIX:-result_mcdropout}_seed_" "$SEEDS" <<'PYEOF'
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

echo ""
echo "all done."
