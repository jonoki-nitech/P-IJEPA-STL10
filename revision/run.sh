#!/usr/bin/env bash
# =============================================================================
# revision/run.sh -- EXTRA EXPERIMENT: sampled-lookahead on the SIGMA-head model
# (the MC-Dropout evaluation pipeline lives in
#  revision/archive/runmcdropout_infer.sh; the old 5-seed sigma sweep in
#  revision/archive/run.sh)
#
#   sigma_lookahead_sample*: the family of sampled-lookahead variants of
#   sigma_lookahead. Every simulated candidate state uses SAMPLES
#   z^(k) = mu + sqrt(sigma^2) * eps of the sigma head's predictive
#   distribution instead of the predictive mean (K = sim_samples, eps stream
#   seeded by SAMPLE_SEED).
#
#     sigma_lookahead_sample     : L = 1 (one sampled simulated state per
#                                  candidate) -- data already produced
#     sigma_lookahead_sample_ms  : the greedy ROLLOUT version up to L =
#                                  --lookahead-steps: addition 1 is the
#                                  candidate, then the next added anchor is
#                                  the fully-unobserved anchor with the
#                                  largest pooled sigma of the current
#                                  simulated state (greedy) and its feature
#                                  is sampled from that state's own
#                                  predictive distribution; the rollout
#                                  score is the sigma sum of the final
#                                  simulated state (mean over K)
#     sigma_lookahead_sample_msn : the NESTED version (depth 2, K = 1 default)
#     sigma_lookahead_sample_ms2 : the DEFAULT run -- the SAME nested depth-2
#                                  planning as msn, but the plan's TWO chosen
#                                  blocks are BOTH OBSERVED per real step
#                                  (candidate + its best follow-up), so the
#                                  observations advance 2 blocks per step
#                                  (the last step adds a single block when
#                                  only one block's budget remains); the
#                                  t axis still counts OBSERVED blocks, so
#                                  the curves stay comparable with the
#                                  1-block-per-step variants on the same
#                                  axis; selection cost ~= half of msn
#                                  (half as many real steps).
#
#   The prescreen and every other mechanism (top-M prescreen, duplicate-free
#   truncation, sigma_sum / sigma_rank recording) are identical to
#   sigma_lookahead; only the counterfactual state(s) are sampled instead of
#   deterministic.
#
#     Step 0 (optional, RUN_CALIB=1): plain sigma calibration (the same
#             b=4 / "4:4" protocol as the old sweep; normally SKIPPED here
#             because the identical artefacts already exist in the old
#             sweep's result zips)
#     Step 0b: per-seed init-block extraction from the OLD sigma run's
#             selection JSONs (make_init_blocks_pt.py) -> init_blocks/seed_{s}.pt
#             (the numpy draw chain depends on the variant set, so the old
#             initial blocks are recovered from the old run's stored
#             selections; the comparison stays per-image paired on the
#             initialization luck)
#     Step 1: SEEDS (default "0 1 2 3 4") x block sizes BS (default "2 3 4";
#             b=1 is skipped like in the old sweep's lookahead stage) x sample
#             counts KS (default "1 4"), one patch_selection_reconstruction_
#             eval.py call per (K, seed) running sigma_lookahead_sample ONLY
#             (default VARIANTS_LIST): the counterpart selection curves
#             already exist in the OLD sweep's artefacts (sigma_greedy /
#             sigma_lookahead, results/STL10_1) and are model outputs of the
#             same checkpoint, so they are REUSED instead of being
#             recomputed; compare through the overlay mode of
#             patch_selection_reconstruction_eval.py (--points_pt <old.pt>
#             <new.pt>; NOTE the two runs' aligned_max_t can differ because
#             the axis is the min budget over each run's own variant set --
#             compare only up to the common min t). The old run's per-image
#             initial blocks are restored (--init-blocks-pt) so the
#             comparisons stay paired on the initialization luck with the
#             same shared init blocks as sigma_greedy / sigma_lookahead.
#             Selection + MAE reconstruction evaluation (rec_error_vs_t +
#             sigma_sum_traj + gallery) all happen in that call.
#
#   Models: sigma-head I-JEPA at MODEL_PATH (default
#   models/embdim=512_epoch=300_patchsize=8/training_model.pth) and MAE at
#   models/mae/latest.ckpt. REQUIRES the sigma-head checkpoint: the
#   MC-Dropout baseline checkpoint (untrained sigma head) is rejected
#   upfront by the checkpoint path name ("mcdropout"), since sampling from
#   its untrained head would silently degenerate.
#
#   Per-(K, seed) outputs (nothing of the other Ks/seeds is touched):
#     images/rec_images_sigmasample_ms2_k{K}/seed_{s}/b{b}/     (ms2 default)
#         rec_error_vs_t.png / .csv / .pt, sigma_sum_traj.pt
#         (sigma_sum + sigma_rank),
#         gallery/nXXXX.png (first --gallery_n samples)
#     selected_patch_index/sigmasample_ms2_k{K}/seed_{s}/
#         block_{b}_tmax_32_{variant}.json
#     (msn lands in sigmasample_msn_k{K}, the greedy ms in
#      sigmasample_L{L}_k{K})
#     init_blocks/seed_{s}.pt (shared init blocks, reproduced from the old run)
#     result_sigmasample_ms2_k{K}_seed_{s}.zip packed at the end of the run
#     result_sigmasample_ms2_k{K}.zip          bundles the K's seed zips
#     result.zip                               bundles EVERY per-(K, seed) zip
#
#   NOTE: a (re)run OVERWRITES the artefacts of the seeds it processes and
#   the init blocks they re-extract (existing init pts are reused unless
#   FORCE_INIT=1).
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
#                                                #  sigma run's per-seed JSON
#                                                #  dirs (default
#                                                #  selected_patch_index,
#                                                #  one seed_{s} dir each)
#     KS="1" ...                                 # predictive-distribution
#                                                #  samples (K rollouts per
#                                                #  candidate) to run
#                                                #  (default "1"; each K is
#                                                #  its own pse run with
#                                                #  --mc-samples K; K=1 means
#                                                #  a single eps draw per
#                                                #  candidate -- NOT the
#                                                #  predictive mean)
#     LSTEPS=3 ...                               # simulated rollout length
#                                                #  of sigma_lookahead_sample_ms
#                                                #  (default 2; L=1 is the
#                                                #  plain sigma_lookahead_sample;
#                                                #  msn, the default, ignores it)
#     SAMPLE_SEED=0 ...                          # torch seed of the eps
#                                                #  streams (default 0;
#                                                #  =--mc-seed; identical
#                                                #  runs repeat exactly)
#     LOOKAHEAD_M=8 ...                          # lookahead candidate prescreen
#                                                #  (default 16 = the old sweep;
#                                                #  ALL_CANDIDATES=1 disables
#                                                #  the prescreen; for msn this
#                                                #  is ALSO the nested c2 panel)
#     VARIANTS_LIST="..." ...                    # (default:
#                                                #  sigma_lookahead_sample_ms2;
#                                                #  the 1-step
#                                                #  sigma_lookahead_sample data
#                                                #  already exists; sigma_greedy /
#                                                #  sigma_lookahead come from
#                                                #  the OLD sweep's artefacts)
#     RUN_CALIB=1 ...                            # redo the plain sigma
#                                                #  calibration (b=4, "4:4")
#     FORCE_INIT=1 ...                           # re-extract the init blocks
#     SKIP_INIT=1 ...                            # skip the init-block step
#                                                #  (then --init-blocks-pt is
#                                                #  NOT passed: the seeds
#                                                #  re-draw the initial blocks
#                                                #  instead, which breaks the
#                                                #  pairing with the old run)
#     SKIP_ZIP=1 ...                             # skip packaging
#     ZIP_PREFIX=result_sigmasample_ms2_k{K} ... # zip base name per K
#
#   Runtime at N=8000 (24 GB GPU) per seed (ms2 default, K=1): the nested
#   scoring runs ~M*(2+M) forwards per real step but each real step consumes
#   TWO blocks, i.e. half as many real steps as msn -> ~half of the msn
#   selection cost (~4.5-7.5 h), plus the MAE sweep (~1.5-2 h) -> roughly
#   6-9 h per seed (~1.3-2 days for the 5 seeds sequentially). The greedy ms
#   variant (L=2, K=1) is ~3x the old cost (~1.5-2.5 h selection). Measure
#   with N=100 first.
#
#   Requires: torch, torchvision, numpy, matplotlib, einops, tqdm.
#   STL10 data is resolved by data_utils.py: if the server data directory
#   exists it is used directly, otherwise only the required splits are
#   streamed into data/stl10_binary/.
# =============================================================================

set -e
set -u

export PYTHONUNBUFFERED=1 # real-time log lines even when piped through tee

PY=${PYTHON:-python}
N=${N:-0}                 # number of test images (0 = all 8000)
BS=${BS:-"2 3 4"}         # block sizes to evaluate (old sweep minus b=1)
SEEDS=${SEEDS:-"0 1 2 3 4"} # seeds for the random-inital strategies
DEVICE=${DEVICE:-cuda:0}  # override with e.g. DEVICE=cpu
GALLERY_N=${GALLERY_N:-10}
LOOKAHEAD_M=${M:-16}      # lookahead candidate prescreen size
LOOKAHEAD_BS=${LOOKAHEAD_BS:-"2 3 4"} # block sizes that run the lookahead
VARIANTS_LIST=${VARIANTS_LIST:-"sigma_lookahead_sample_ms2"}
KS=${KS:-"1"}             # sample-count runs (1 = single draw; e.g. KS="1 4")
LSTEPS=${LSTEPS:-2}       # rollout length of the GREEDY ms variant (ignored
                          # by ms2/msn, the depth-2 variants)
SAMPLE_SEED=${SAMPLE_SEED:-0} # torch seed of the eps streams (-> --mc-seed)
MODEL_PATH=${MODEL_PATH:-models/embdim=512_epoch=300_patchsize=8/training_model.pth}
MAE_MODEL_PATH=${MAE_MODEL_PATH:-models/mae/latest.ckpt}
export CHUNK=${CHUNK:-1024} # chunk size of the batched forward passes

OUT_TAG="sigmasample"
INIT_DIR="init_blocks"
OLD_JSON_BASE=${OLD_JSON_BASE:-selected_patch_index} # old sweep's per-seed JSON dirs
ALL_SEED_ZIPS=""          # names of every per-(K, seed) zip created by Step 1,
                          # collected for the final single result.zip bundle

LOOKAHEAD_FLAGS=""
if [ "${ALL_CANDIDATES:-0}" = "1" ]; then
    LOOKAHEAD_FLAGS="--all-candidates"
fi

echo "python : $($PY --version 2>&1)"
echo "params : N=$N   bs=$BS   seeds=[$SEEDS]   device=$DEVICE   gallery_n=$GALLERY_N   KS=[$KS]   L=$LSTEPS   sample_seed=$SAMPLE_SEED   M=$LOOKAHEAD_M   lookahead_bs=[$LOOKAHEAD_BS]   old_json_base=$OLD_JSON_BASE   model=$MODEL_PATH"

# per-seed sanity check of the old-JSON locations before any heavy step
if [ "${SKIP_INIT:-0}" != "1" ]; then
    for SEED in $SEEDS; do
        DIR="${OLD_JSON_BASE}/seed_${SEED}"
        if [ ! -f "${DIR}/block_2_tmax_32_sigma_greedy.json" ] \
           && [ ! -f "${DIR}/block_2_tmax_32_random_greedy.json" ] \
           && [ ! -f "${DIR}/block_1_tmax_32_sigma_greedy.json" ]; then
            echo "" >&2
            echo "The old sigma run's selection JSONs are missing at ${DIR}/" >&2
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
# Step 0 (optional): plain sigma calibration of the sigma head (cool because the
#   identical artefacts exist in the old sweep's result zips -- normally skip)
# -----------------------------------------------------------------------------
if [ "${RUN_CALIB:-0}" = "1" ]; then
    echo ""
    echo "=== Step 0: sigma calibration ====================================="
    $PY calibration_eval.py \
        --model_path "$MODEL_PATH" \
        --b 4 \
        --positions 4:4 \
        --N "$N" \
        --mc-samples 1 \
        --out_dir "images/calibration" \
        --device "$DEVICE"
fi

# -----------------------------------------------------------------------------
# Step 0b: per-seed init-block extraction from the old run's selections
#   (idempotent: skipped when the pt already exists, unless FORCE_INIT=1)
# -----------------------------------------------------------------------------
if [ "${SKIP_INIT:-0}" != "1" ]; then
    echo ""
    echo "=== Step 0b: init blocks from the old sigma run ==================="
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
# Step 1: per-K sample count of sigma_lookahead_sample (each K is a SEPARATE
#         pse run: K = 1 = single-sample lookahead, K > 1 = mean over K
#         sampled simulations) + MAE sweep + per-(K, seed) zip
# -----------------------------------------------------------------------------
for K in $KS; do
for SEED in $SEEDS; do
    SEED_DIR="seed_${SEED}"
    case "$VARIANTS_LIST" in
        *ms2*) TAG_K="sigmasample_ms2_k${K}";;
        *msn*) TAG_K="sigmasample_msn_k${K}";;
        *)     TAG_K="sigmasample_L${LSTEPS}_k${K}";;
    esac
    REC_DIR="images/rec_images_${TAG_K}/${SEED_DIR}"
    JSON_DIR="selected_patch_index/${TAG_K}/${SEED_DIR}"
    INIT_ARGS=""
    if [ "${SKIP_INIT:-0}" != "1" ]; then
        INIT_ARGS="--init-blocks-pt ${INIT_DIR}/seed_${SEED}.pt"
    fi

    echo ""
    echo "=== Step 1: K=$K seed=$SEED (selection + MAE reconstruction vs t) ==="
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
        --lookahead-steps "$LSTEPS" \
        --mc-samples "$K" \
        --mc-seed "$SAMPLE_SEED" \
        $INIT_ARGS \
        --gallery_n "$GALLERY_N" \
        --out_dir "$REC_DIR" \
        --json_dir "$JSON_DIR" \
        --device "$DEVICE"

    # -------------------------------------------------------------------------
    # Package this (K, seed)'s artefacts (+ the seed's init blocks) into a zip
    # -------------------------------------------------------------------------
    if [ "${SKIP_ZIP:-0}" != "1" ]; then
        ZIP="${ZIP_PREFIX:-result_${TAG_K}}_${SEED_DIR}.zip"
        echo ""
        echo "=== Pack artefacts into $ZIP ===================================="
        rm -f "$ZIP"
        "$PY" - "$ZIP" "$SEED_DIR" "$TAG_K" "$INIT_DIR" <<'PYEOF'
import os, sys, zipfile
zip_name, seed_dir, out_tag, init_dir = sys.argv[1:5]
roots = [f"images/rec_images_{out_tag}/{seed_dir}",
         f"selected_patch_index/{out_tag}/{seed_dir}"]
init_pt = f"{init_dir}/{seed_dir}.pt"
if os.path.isfile(init_pt):
    roots.append(init_pt)
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for f in files:
                zf.write(os.path.join(dirpath, f))
print(f"created: {zip_name} ({os.path.getsize(zip_name) / 1e6:.1f} MB)")
PYEOF
        ALL_SEED_ZIPS="$ALL_SEED_ZIPS $ZIP"
    fi
done
done

# -----------------------------------------------------------------------------
# Bundle all per-(K, seed) zips into one result zip per K (nested: the seed
# zips are stored as-is, without re-compression)
# -----------------------------------------------------------------------------
if [ "${SKIP_ZIP:-0}" != "1" ]; then
    for K in $KS; do
        K_PREFIX="${ZIP_PREFIX:-result_${TAG_K}}"
        RESULT_ZIP="${K_PREFIX}.zip"
        echo ""
        echo "=== Bundle per-seed zips into $RESULT_ZIP ======================="
        rm -f "$RESULT_ZIP"
        "$PY" - "$RESULT_ZIP" "${K_PREFIX}_seed_" "$SEEDS" <<'PYEOF'
import os, sys, zipfile
result_zip, prefix, seeds = sys.argv[1], sys.argv[2], sys.argv[3]
seed_zips = [f"{prefix}{s}.zip" for s in seeds.split()]
with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_STORED) as zf:
    for n in seed_zips:
        zf.write(n, os.path.basename(n))
print(f"created: {result_zip} ({os.path.getsize(result_zip) / 1e6:.1f} MB, "
      f"{len(seed_zips)} seed zips)")
PYEOF
    done
fi

# -----------------------------------------------------------------------------
# Final bundle: EVERY per-(K, seed) zip collected in Step 1 into ONE result.zip
# (the single file to share; the seed zips inside are stored as-is, without
# re-compression, like the old sweep's result.zip)
# -----------------------------------------------------------------------------
if [ "${SKIP_ZIP:-0}" != "1" ] && [ -n "$ALL_SEED_ZIPS" ]; then
    echo ""
    echo "=== Bundle EVERYTHING into result.zip ==========================="
    rm -f result.zip
    # shellcheck disable=SC2086
    "$PY" - result.zip $ALL_SEED_ZIPS <<'PYEOF'
import os, sys, zipfile
result_zip = sys.argv[1]
seed_zips = sys.argv[2:]
with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_STORED) as zf:
    for n in seed_zips:
        zf.write(n, os.path.basename(n))
print(f"created: {result_zip} ({os.path.getsize(result_zip) / 1e6:.1f} MB, "
      f"{len(seed_zips)} seed zips)")
PYEOF
fi

echo ""
echo "all done."
