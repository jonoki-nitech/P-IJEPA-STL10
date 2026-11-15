#!/usr/bin/env bash
# =============================================================================
# revision/run_2.sh -- TRAINING-TIME BENCHMARK (for the paper's
# reproducibility statement: "report ... training duration")
#
#   Runs benchmark_training_time.py for BOTH pretrainings exactly as they
#   are trained, with --steps 50 timed steps each, and appends everything to
#   a single log file:
#
#     ijepa : the sigma-head I-JEPA of train.py
#             (STL-10 unlabeled, batch 192, Adam 3e-4/1e-5, EMA every step,
#              fp16 OFF; 300 epochs = 156,300 steps)
#     mae   : the MAE of mae-pytorch (config/mae.yaml)
#             (STL-10 unlabeled with the training augmentation, batch 64,
#              AdamW effective lr 3.75e-5, 16-mixed; 200 epochs = 312,400 steps)
#
#   The script reports the measured per-step time, per-epoch time, the
#   extrapolated TOTAL training time and the peak GPU memory -- run it on
#   the TRAINING machine (the numbers are hardware dependent). The output
#   lines starting with "extrapolation: ... -> ~X.X h total" are what the
#   paper's training-duration statement needs.
#
#   Usage (from anywhere; the script cds to the repo root):
#       bash revision/run_2.sh
#   Overridable environment variables:
#       STEPS=50          timed steps per model        (default 50)
#       WARMUP=5          warm-up steps per model      (default 5)
#       NUM_WORKERS=8     data-loader workers          (default 8, as training)
#       DEVICE=cuda:0     torch device                 (default cuda:0)
#       MODELS="ijepa mae"  which models to run        (default both)
#       LOG=...           log file                     (default
#                          benchmark_training_time_steps${STEPS}.log)
#       SPLIT=...         smoke-test only: 'test' avoids the 2.6 GB unlabeled
#                         split (unset on the server = the training split)
#       DATASET_SIZE=...  smoke-test only: cap the dataset size
#       PYTHON=python     python interpreter
#   Example:  DEVICE=cuda:1 STEPS=50 bash revision/run_2.sh
# =============================================================================

set -e
set -u
set -o pipefail

export PYTHONUNBUFFERED=1 # real-time log lines even when piped through tee

PY=${PYTHON:-python}
STEPS=${STEPS:-50}        # timed steps per model
WARMUP=${WARMUP:-5}       # warm-up steps per model
NUM_WORKERS=${NUM_WORKERS:-8}
DEVICE=${DEVICE:-cuda:0}
MODELS=${MODELS:-"ijepa mae"}
LOG=${LOG:-benchmark_training_time_steps${STEPS}.log}

# always run from the repository root (data_utils resolves the STL-10 path
# relative to the server data directory)
cd "$(dirname "$0")/.."

if [ ! -f benchmark_training_time.py ]; then
    echo "benchmark_training_time.py not found in $(pwd)" >&2
    exit 1
fi

touch "$LOG"
{
    echo ""
    echo "==============================================================="
    echo " training-time benchmark | $(date '+%F %T') | host: $(hostname)"
    echo " device: $DEVICE | steps: $STEPS | warmup: $WARMUP | workers: $NUM_WORKERS"
    command -v nvidia-smi >/dev/null 2>&1 && \
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
        | sed 's/^/ gpu: /'
    echo "==============================================================="
} >> "$LOG"

for MODEL in $MODELS; do
    echo "=== benchmark: $MODEL (steps=$STEPS) ==="
    EXTRA_ARGS=""
    if [ -n "${SPLIT:-}" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --split $SPLIT"
    fi
    if [ -n "${DATASET_SIZE:-}" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --dataset_size $DATASET_SIZE"
    fi
    # shellcheck disable=SC2086
    "$PY" benchmark_training_time.py \
        --model "$MODEL" \
        --device "$DEVICE" \
        --steps "$STEPS" \
        --warmup "$WARMUP" \
        --num_workers "$NUM_WORKERS" $EXTRA_ARGS 2>&1 | tee -a "$LOG"
    echo ""
done

echo "saved: $LOG"
