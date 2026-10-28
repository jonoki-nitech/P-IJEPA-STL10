#!/usr/bin/env bash
# =============================================================================
# revision/run.sh
#
#   Trains the MC-Dropout baseline I-JEPA on the GPU:
#     $PY run_training_mcdropout.py
#
#   Identical setup to the probabilistic run_training.py (architecture
#   embdim=512_epoch=300_patchsize=8, 300 epochs, batch 192, lr 3e-4, wd 1e-5,
#   the cosine EMA gamma schedule, the STL10 unlabeled split, the periodic
#   training_model.pth / encoder.pth saves every --log_every_n_steps and the
#   periodic STL10 logistic-regression evaluations), EXCEPT:
#     * the loss is the vanilla I-JEPA MSE between the mu-predictor output
#       and the (EMA) target representation: NO sigma-predictor / NLL term
#     * ONLY the mu-predictor carries Dropout (--mu_dropout, default 10%;
#       MC-Dropout uncertainty baseline for the comparison runs)
#     * the sigma-predictor stays at its random initialization (its weights
#       are still saved, so the evaluation pipeline can load the checkpoint
#       strictly, but the sigma-based strategies are MEANINGLESS with this
#       checkpoint -- use MC-Dropout uncertainties instead)
#
#   Outputs go to models/mcdropout_embdim=512_epoch=300_patchsize=8/:
#     training_model.pth          full I-JEPA state dict (sigma keys included,
#                                 untrained -- overwritten every log period)
#     encoder.pth                 target-encoder state dict (EMA)
#     result.zip                  packed at the end of the run: the save
#                                 directory's contents, stored under
#                                 mcdropout_embdim=.../training_model.pth etc.
#                                 (SKIP_ZIP=1 disables packaging)
#   NOTE: the default save directory differs from the probabilistic
#   embdim=512_epoch=300_patchsize=8 so the sigma-based checkpoints are
#   NEVER overwritten; pass -save_model_dir to change it.
#
# Usage:
#   bash revision/run.sh                                 # GPU hand-off run
#   PYTHON=/path/to/python bash revision/run.sh          # pin the interpreter
#   DEVICE=cuda:1 bash revision/run.sh                   # pin the GPU index
#
#   Overrides:
#     N_EPOCHS=10 ...                  # epochs (default 300)
#     BATCH=64 ...                     # batch size (default 192)
#     LR=3e-4 / WD=1e-5 ...            # optimizer
#     MU_DROPOUT=0.2 ...               # dropout probability (default 0.1)
#     GAMMA=0.996 ...                  # initial EMA coefficient
#     LOG_EVERY=200 ...                # save/eval period in steps
#     FP16=0 ...                       # 0 = fp32 (plain run_training.py
#                                      #   behaviour); 1 = --fp16_precision
#                                      #   (faster, default on a GPU)
#     SAVE_DIR=./models/xx ...         # -save_model_dir
#     CKPT=path/training_model.pth ... # resume training
#     SMOKE=1 ...                      # 1-epoch, every-1-step sanity run
#     SKIP_ZIP=1 ...                   # skip the final result.zip
#     ZIP=result.zip ...               # zip base name
#
#   The STL10 data is resolved by data_utils.py (auto-detected server
#   directory, otherwise streamed into data/stl10_binary/).
# =============================================================================

set -e
set -u

export PYTHONUNBUFFERED=1 # real-time log lines even when piped through tee

PY=${PYTHON:-python}

N_EPOCHS=${N_EPOCHS:-300}
BATCH=${BATCH:-192}
LR=${LR:-3e-4}
WD=${WD:-1e-5}
MU_DROPOUT=${MU_DROPOUT:-0.1}
GAMMA=${GAMMA:-0.996}
LOG_EVERY=${LOG_EVERY:-200}
SAVE_DIR=${SAVE_DIR:-./models/mcdropout_embdim=512_epoch=300_patchsize=8}
DEVICE=${DEVICE:-cuda:0} # scripts auto-pick "cuda"; the index is pinned below

if [ "${SMOKE:-0}" = "1" ]; then
    N_EPOCHS=1
    LOG_EVERY=1
fi

# the training script itself picks "cuda"; honour DEVICE=cuda:<i> by making
# the physical GPU <i> the only visible device (=> cuda == cuda:<i>)
if [ "$DEVICE" != "cpu" ]; then
    export CUDA_VISIBLE_DEVICES=${DEVICE#cuda:}
fi

FP16_FLAGS=""
if [ "${FP16:-1}" = "1" ]; then
    if [ "$DEVICE" = "cpu" ]; then
        echo "NOTE: FP16 ignored on cpu."
    else
        FP16_FLAGS="--fp16_precision"
    fi
fi

CKPT_FLAGS=""
if [ -n "${CKPT:-}" ]; then
    CKPT_FLAGS="--ckpt_path $CKPT"
fi

echo "python : $($PY --version 2>&1)"
echo "params : epochs=$N_EPOCHS   batch=$BATCH   lr=$LR   wd=$WD   " \
     "mu_dropout=$MU_DROPOUT   gamma=$GAMMA   log_every=$LOG_EVERY"
echo "         save_dir=$SAVE_DIR   device=$DEVICE   " \
     "cuda_visible_devices=${CUDA_VISIBLE_DEVICES-(unset)}" \
     "   fp16=${FP16:-1}"

echo ""
echo "=== MC-Dropout baseline I-JEPA training ==========================="
$PY run_training_mcdropout.py \
    -save_model_dir "$SAVE_DIR" \
    --num_epochs "$N_EPOCHS" \
    -b "$BATCH" \
    -lr "$LR" \
    -wd "$WD" \
    --mu_dropout "$MU_DROPOUT" \
    --gamma "$GAMMA" \
    --log_every_n_steps "$LOG_EVERY" \
    $FP16_FLAGS \
    $CKPT_FLAGS

# -----------------------------------------------------------------------------
# Package the artefacts into result.zip
# -----------------------------------------------------------------------------
if [ "${SKIP_ZIP:-0}" != "1" ]; then
    INFIX=$(basename "$SAVE_DIR")
    ZIP="${ZIP:-result.zip}"
    echo ""
    echo "=== Pack artefacts into $ZIP ===================================="
    rm -f "$ZIP"
    "$PY" - "$ZIP" "$SAVE_DIR" "$INFIX" <<'PYEOF'
import os, sys, zipfile
zip_name, save_dir, infix = sys.argv[1], sys.argv[2], sys.argv[3]
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for dirpath, _, files in os.walk(save_dir):
        for f in files:
            src = os.path.join(dirpath, f)
            dst = os.path.join(infix, os.path.relpath(src, save_dir))
            zf.write(src, dst)
print(f"created: {zip_name} ({os.path.getsize(zip_name) / 1e6:.1f} MB, "
      f"from {save_dir})")
PYEOF
fi

# -----------------------------------------------------------------------------
# Summary of the produced artefacts
# -----------------------------------------------------------------------------
echo ""
echo "=== Done. Produced artefacts ======================================"
ls -l "$SAVE_DIR"/training_model.pth "$SAVE_DIR"/encoder.pth 2>/dev/null \
    || echo "(missing artefacts in $SAVE_DIR)"
ls "${ZIP:-result.zip}" 2>/dev/null || true
echo ""
echo "NOTE: the sigma-predictor in this checkpoint is untrained --"
echo "      do NOT run the sigma-based strategies with it."

exit 0
