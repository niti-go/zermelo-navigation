#!/usr/bin/env bash
# One-shot overnight experiment: generate → thin → train everything → evaluate.
#
# Pipeline (all settings read from zermelo_config.yaml — 5x5 sensors / 50 flow
# readings, wandb_project_name, thinned dataset_save_path):
#   1. GENERATE  a large RAW dataset (NUM_EPISODES, default 50k) in conda env
#                `zermelo`. The raw path is injected into a throwaway config so
#                the committed zermelo_config.yaml keeps pointing at the thinned
#                set (what training + eval read).
#   2. THIN      RAW → THINNED via trim_dataset.py (poor-quality target shape).
#   3. TRAIN     BC, DT, MeanFlowBC, and the MFQL alpha sweep (alpha=1,3,10 with
#                use_dynamic_alpha=False) in conda env `flowrl`. Jobs are packed
#                onto whatever GPUs are free: each job gets one GPU, and the next
#                queued job launches as soon as a GPU frees up (parallel up to
#                MAX_PARALLEL, sequential beyond that).
#   4. EVALUATE  with EVAL_EPISODES (default 500) per algorithm. evaluate_models
#                auto-discovers every run-group dir, so each sweep variant shows
#                up as its own line in every plot. A training that crashed leaves
#                no checkpoints and is silently skipped.
#
# The whole run executes inside a detached tmux session so it survives logout.
#
# Usage:
#   bash scripts/run_overnight.sh
#   SEED=1 NUM_EPISODES=60000 bash scripts/run_overnight.sh
#   MAX_PARALLEL=4 bash scripts/run_overnight.sh      # cap concurrent trainings
#   SKIP_GEN=1 SKIP_TRIM=1 bash scripts/run_overnight.sh   # reuse existing data
#   NO_TMUX=1 bash scripts/run_overnight.sh           # run in the foreground
#
# Env-var overrides (all optional):
#   SEED         random seed                                   (default 0)
#   NUM_EPISODES raw dataset size                              (default 50000)
#   BC_STEPS     BC train_steps                                (default 500000)
#   DT_STEPS     DT train_steps                                (default 500000)
#   MFQL_STEPS   MeanFlowQL offline_steps                      (default 1000000)
#   MFBC_STEPS   MeanFlowBC offline_steps                      (default 1000000)
#   EVAL_EPISODES  eval episodes per algo per segment          (default 500)
#   MAX_PARALLEL   max concurrent trainings                    (default: #GPUs)
#   RAW_DATASET    raw dataset path        (default datasets/waypoint_5x5_gridsensors.npz)
#   SKIP_GEN=1     reuse RAW_DATASET if it already exists
#   SKIP_TRIM=1    reuse the thinned dataset if it already exists
#   NO_TMUX=1      run inline instead of in a detached tmux session
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
DATASET_ENV="zermelo"   # conda env for dataset generation
TRAIN_ENV="flowrl"      # conda env for training + evaluation

# ── Tunable knobs (env overrides honored) ─────────────────────────────────────
SEED="${SEED:-0}"
NUM_EPISODES="${NUM_EPISODES:-50000}"
BC_STEPS="${BC_STEPS:-500000}"
DT_STEPS="${DT_STEPS:-500000}"
MFQL_STEPS="${MFQL_STEPS:-1000000}"
MFBC_STEPS="${MFBC_STEPS:-1000000}"
EVAL_EPISODES="${EVAL_EPISODES:-500}"
SKIP_GEN="${SKIP_GEN:-0}"
SKIP_TRIM="${SKIP_TRIM:-0}"
RAW_DATASET="${RAW_DATASET:-datasets/waypoint_5x5_gridsensors.npz}"

# ── MFQL alpha sweep: "run_group|<agent overrides>" ───────────────────────────
MFQL_SPECS=(
    "mfql_alpha1|--agent.alpha=1   --agent.use_dynamic_alpha=False"
    "mfql_alpha3|--agent.alpha=3   --agent.use_dynamic_alpha=False"
    "mfql_alpha10|--agent.alpha=10 --agent.use_dynamic_alpha=False"
)

# ══════════════════════════════════════════════════════════════════════════════
# OUTER: preflight, then relaunch the real work inside tmux (or inline).
# ══════════════════════════════════════════════════════════════════════════════
if [ -z "${OVERNIGHT_INNER:-}" ]; then
    for cmd in python3 nvidia-smi tmux; do
        command -v "$cmd" >/dev/null || { echo "ERROR: '$cmd' not found." >&2; exit 1; }
    done
    [ -f "$CONDA_SH" ] || { echo "ERROR: $CONDA_SH not found. Edit CONDA_SH." >&2; exit 1; }

    _pyread() { python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]])" \
        "$REPO_ROOT/zermelo_config.yaml" "$1"; }
    PROJ_WANDB=$(_pyread wandb_project_name)
    THINNED=$(_pyread dataset_save_path)

    GPU_COUNT=$(nvidia-smi -L | wc -l)
    MAX_PARALLEL="${MAX_PARALLEL:-$GPU_COUNT}"
    [ "$MAX_PARALLEL" -ge 1 ] 2>/dev/null || { echo "ERROR: no GPUs available." >&2; exit 1; }

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="$REPO_ROOT/logs/$PROJ_WANDB"
    SESSION="overnight_${PROJ_WANDB}"
    mkdir -p "$LOG_DIR"

    # Hand the resolved knobs to the inner run via a sourced env file (tmux does
    # not reliably inherit the launching shell's environment).
    ENVFILE=$(mktemp /tmp/overnight_env_XXXXXX.sh)
    cat > "$ENVFILE" <<EOF
SEED=$SEED
NUM_EPISODES=$NUM_EPISODES
BC_STEPS=$BC_STEPS
DT_STEPS=$DT_STEPS
MFQL_STEPS=$MFQL_STEPS
MFBC_STEPS=$MFBC_STEPS
EVAL_EPISODES=$EVAL_EPISODES
SKIP_GEN=$SKIP_GEN
SKIP_TRIM=$SKIP_TRIM
RAW_DATASET=$RAW_DATASET
THINNED_DATASET=$THINNED
PROJ_WANDB=$PROJ_WANDB
MAX_PARALLEL=$MAX_PARALLEL
TIMESTAMP=$TIMESTAMP
LOG_DIR=$LOG_DIR
EOF

    echo "════════════════════════════════════════════════════════════"
    echo "  Overnight experiment : $PROJ_WANDB"
    echo "  Raw dataset          : $RAW_DATASET  ($NUM_EPISODES episodes)"
    echo "  Thinned dataset      : $THINNED"
    echo "  Trainings            : bc, dt, meanflow_bc, ${MFQL_SPECS[*]%%|*}"
    echo "  GPUs / max parallel  : $GPU_COUNT / $MAX_PARALLEL"
    echo "  Eval episodes        : $EVAL_EPISODES"
    echo "  Logs                 : $LOG_DIR"
    echo "════════════════════════════════════════════════════════════"

    if [ "${NO_TMUX:-0}" = "1" ]; then
        OVERNIGHT_INNER=1 OVERNIGHT_ENV="$ENVFILE" bash --login "$SELF"
        exit $?
    fi

    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Killing existing tmux session '$SESSION'..."
        tmux kill-session -t "$SESSION"
    fi
    tmux new -d -s "$SESSION" \
        "OVERNIGHT_INNER=1 OVERNIGHT_ENV='$ENVFILE' bash --login '$SELF'; \
         echo; echo '[overnight complete — press Enter to close]'; read; exec bash"

    echo
    echo "Launched in tmux session '$SESSION'."
    echo "  Attach : tmux attach -t $SESSION   (detach: Ctrl-b then d)"
    echo "  Logs   : tail -f $LOG_DIR/*_$TIMESTAMP.log"
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════════
# INNER: the actual pipeline.
# ══════════════════════════════════════════════════════════════════════════════
source "$OVERNIGHT_ENV"
cd "$REPO_ROOT"
source "$CONDA_SH"

RAW="$RAW_DATASET"
THINNED="$THINNED_DATASET"
WB="--proj_wandb=$PROJ_WANDB"

echo
echo "[$(date +%T)] ===== Overnight pipeline: $PROJ_WANDB (seed=$SEED) ====="

# ── 1/4  Generate raw dataset ─────────────────────────────────────────────────
if [ "$SKIP_GEN" = "1" ] && [ -f "$RAW" ]; then
    echo "[$(date +%T)] 1/4  SKIP_GEN set and $RAW exists — skipping generation."
else
    echo "[$(date +%T)] 1/4  Generating raw dataset ($NUM_EPISODES episodes) → $RAW"
    conda activate "$DATASET_ENV"
    GEN_CFG=$(mktemp /tmp/overnight_gencfg_XXXXXX.yaml)
    python3 - "$REPO_ROOT/zermelo_config.yaml" "$GEN_CFG" "$RAW" "$NUM_EPISODES" <<'PY'
import sys, yaml
src, dst, raw, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
cfg = yaml.safe_load(open(src))
cfg['dataset_save_path'] = raw          # write the RAW set, not the thinned one
cfg.setdefault('run', {})['num_episodes'] = n
yaml.safe_dump(cfg, open(dst, 'w'), sort_keys=False)
PY
    python scripts/generate_dataset.py --config "$GEN_CFG" \
        2>&1 | tee "$LOG_DIR/gen_$TIMESTAMP.log"
    echo "[$(date +%T)] Generation complete."
fi

# ── 2/4  Thin the dataset ─────────────────────────────────────────────────────
if [ "$SKIP_TRIM" = "1" ] && [ -f "$THINNED" ]; then
    echo "[$(date +%T)] 2/4  SKIP_TRIM set and $THINNED exists — skipping thinning."
else
    echo "[$(date +%T)] 2/4  Thinning $RAW → $THINNED"
    conda activate "$DATASET_ENV"   # numpy-only; either env works
    TRIM_INPUT="$RAW" TRIM_OUTPUT="$THINNED" TRIM_SEED="$SEED" \
        python scripts/trim_dataset.py 2>&1 | tee "$LOG_DIR/trim_$TIMESTAMP.log"
    echo "[$(date +%T)] Thinning complete."
fi
[ -f "$THINNED" ] || { echo "ERROR: thinned dataset $THINNED missing — aborting." >&2; exit 1; }

# ── 3/4  Train everything, packed onto free GPUs ──────────────────────────────
echo "[$(date +%T)] 3/4  Training (max $MAX_PARALLEL concurrent)…"
conda activate "$TRAIN_ENV"

# Build the job list as "label|command".
JOBS=()
JOBS+=("bc|python scripts/bc_zermelo.py --seed=$SEED --train_steps=$BC_STEPS $WB --run_group=bc --zermelo_dataset=$THINNED")
JOBS+=("dt|python scripts/dt_zermelo.py --seed=$SEED --train_steps=$DT_STEPS $WB --run_group=dt --zermelo_dataset=$THINNED")
JOBS+=("meanflow_bc|python scripts/meanflowbc_zermelo.py --seed=$SEED --offline_steps=$MFBC_STEPS $WB --run_group=meanflow_bc --zermelo_dataset=$THINNED")
for spec in "${MFQL_SPECS[@]}"; do
    label="${spec%%|*}"; overrides="${spec#*|}"
    JOBS+=("$label|python scripts/meanflowql_zermelo.py --seed=$SEED --offline_steps=$MFQL_STEPS $WB --run_group=$label --zermelo_dataset=$THINNED $overrides")
done
NJOBS=${#JOBS[@]}

# Free GPUs, least-loaded first, capped to the pool size.
pick_gpus_sorted() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | awk -F, '{gsub(/ /, "", $1); print $1}'
}
mapfile -t ALL_GPUS < <(pick_gpus_sorted)
POOL=$MAX_PARALLEL
[ "$POOL" -gt "$NJOBS" ] && POOL=$NJOBS
[ "$POOL" -gt "${#ALL_GPUS[@]}" ] && POOL=${#ALL_GPUS[@]}
free_gpus=( "${ALL_GPUS[@]:0:$POOL}" )
echo "[$(date +%T)]   GPU pool: ${free_gpus[*]}   ($NJOBS jobs)"

DONE_FILE=$(mktemp /tmp/overnight_done_XXXXXX)
: > "$DONE_FILE"
declare -A EXITCODE

launch_one() {
    local label="$1" cmd="$2" gpu="$3"
    local log="$LOG_DIR/${label}_$TIMESTAMP.log"
    echo "[$(date +%T)]   ▶ $label  → GPU $gpu  → $(basename "$log")"
    (
        set +e   # always record the exit code, even when the training fails
        CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd" > "$log" 2>&1
        echo "$label $? $gpu" >> "$DONE_FILE"
    ) &
}

i=0; running=0; processed=0
while [ "$i" -lt "$NJOBS" ] || [ "$running" -gt 0 ]; do
    # Launch while GPUs are free and jobs remain.
    while [ "${#free_gpus[@]}" -gt 0 ] && [ "$i" -lt "$NJOBS" ]; do
        spec="${JOBS[$i]}"; label="${spec%%|*}"; cmd="${spec#*|}"
        gpu="${free_gpus[0]}"; free_gpus=( "${free_gpus[@]:1}" )
        launch_one "$label" "$cmd" "$gpu"
        i=$((i + 1)); running=$((running + 1))
    done
    [ "$running" -eq 0 ] && break
    # Block until some background job exits, then reconcile via the done-file.
    wait -n 2>/dev/null || true
    mapfile -t done_lines < "$DONE_FILE"
    while [ "$processed" -lt "${#done_lines[@]}" ]; do
        read -r dlabel dcode dgpu <<< "${done_lines[$processed]}"
        EXITCODE["$dlabel"]=$dcode
        free_gpus+=( "$dgpu" )
        running=$((running - 1))
        processed=$((processed + 1))
        if [ "$dcode" -eq 0 ]; then
            echo "[$(date +%T)]   ✓ $dlabel finished (GPU $dgpu freed)"
        else
            echo "[$(date +%T)]   ✗ $dlabel FAILED (exit $dcode) — see ${dlabel}_$TIMESTAMP.log"
        fi
    done
done

SUCCEEDED=""; FAILED=""
for label in "${!EXITCODE[@]}"; do
    if [ "${EXITCODE[$label]}" -eq 0 ]; then
        SUCCEEDED="${SUCCEEDED}${SUCCEEDED:+,}$label"
    else
        FAILED="${FAILED}${FAILED:+,}$label"
    fi
done
echo "[$(date +%T)] Training done.  Succeeded: ${SUCCEEDED:-(none)}   Failed: ${FAILED:-(none)}"

if [ -z "$SUCCEEDED" ]; then
    echo "[$(date +%T)] All trainings failed — skipping evaluation." >&2
    exit 1
fi

# ── 4/4  Evaluate ─────────────────────────────────────────────────────────────
# evaluate_models auto-discovers every run-group with checkpoints, so all
# succeeded variants are plotted; crashed ones have no checkpoints and drop out.
echo "[$(date +%T)] 4/4  Evaluating ($EVAL_EPISODES episodes/algo)…"
ZERMELO_EVAL_EPISODES="$EVAL_EPISODES" \
    python scripts/evaluate_models.py 2>&1 | tee "$LOG_DIR/eval_$TIMESTAMP.log"

echo
echo "════════════════════════════════════════════════════════════"
echo "  Overnight experiment complete: $PROJ_WANDB"
echo "  Succeeded : ${SUCCEEDED}"
[ -n "$FAILED" ] && echo "  Failed    : ${FAILED}"
echo "  Results   : results/$PROJ_WANDB/"
echo "  Logs      : $LOG_DIR/"
echo "════════════════════════════════════════════════════════════"
