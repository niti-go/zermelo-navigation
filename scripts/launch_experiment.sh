#!/usr/bin/env bash
# Full experiment pipeline: dataset generation → parallel training → evaluation.
# Everything runs inside a single tmux session named exp_<wandb_project_name>,
# so multiple experiments can run in parallel by launching this script multiple
# times (with different configs).
#
# Usage:
#   bash scripts/launch_experiment.sh
#   SEED=1 bash scripts/launch_experiment.sh
#
# Env-var overrides (all optional):
#   SEED        — random seed                           (default 0)
#   BC_STEPS    — BC train_steps                        (default 500000)
#   DT_STEPS    — DT train_steps                        (default 500000)
#   MFQL_STEPS  — MeanFlowQL offline_steps              (default 1000000)
#   MFBC_STEPS  — MeanFlowBC (critic-free) offline_steps (default 1000000)
#
# Config keys read from zermelo_config.yaml:
#   wandb_project_name — names the tmux session and wandb project
#   dataset_type       — 'straight' or 'diverse'
#   dataset_save_path  — where the dataset is written / read from
set -euo pipefail

SEED="${SEED:-0}"
BC_STEPS="${BC_STEPS:-500000}"
DT_STEPS="${DT_STEPS:-500000}"
MFQL_STEPS="${MFQL_STEPS:-1000000}"
MFBC_STEPS="${MFBC_STEPS:-1000000}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
DATASET_ENV="zermelo"   # conda env for dataset generation
TRAIN_ENV="flowrl"      # conda env for training + evaluation

# ── Read project config ──────────────────────────────────────────────────────
_pyread() { python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]])" "$REPO_ROOT/zermelo_config.yaml" "$1"; }
PROJ_WANDB=$(  _pyread wandb_project_name)
DATASET_TYPE=$(  _pyread dataset_type)

SESSION="exp_${PROJ_WANDB}"
LOG_DIR="$REPO_ROOT/logs/$PROJ_WANDB"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── Pre-flight checks ────────────────────────────────────────────────────────
if [[ "$DATASET_TYPE" != "straight" && "$DATASET_TYPE" != "diverse" ]]; then
    echo "ERROR: dataset_type must be 'straight' or 'diverse', got '$DATASET_TYPE'" >&2
    exit 1
fi
for cmd in python3 nvidia-smi tmux; do
    command -v "$cmd" >/dev/null || { echo "ERROR: '$cmd' not found" >&2; exit 1; }
done
[[ -f "$CONDA_SH" ]] || { echo "ERROR: $CONDA_SH not found. Edit CONDA_SH in this script." >&2; exit 1; }

# ── Kill any stale session ───────────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Killing existing tmux session '$SESSION'..."
    tmux kill-session -t "$SESSION"
fi

# ── Write the inner experiment script to a temp file ────────────────────────
# GPU selection is deferred to just before training starts (after dataset gen)
# so we pick the least-loaded GPUs at the moment they're actually needed.
TMPSCRIPT=$(mktemp /tmp/zermelo_exp_XXXXXX.sh)
chmod +x "$TMPSCRIPT"

# Note: unquoted INNER_EOF → outer $VAR are expanded now (baked in).
#       Inner script's own dynamic vars are written as \$VAR.
cat > "$TMPSCRIPT" << INNER_EOF
#!/bin/bash
set -euo pipefail
cd "$REPO_ROOT"
source "$CONDA_SH"

PROJ_WANDB="$PROJ_WANDB"
DATASET_TYPE="$DATASET_TYPE"
TIMESTAMP="$TIMESTAMP"
LOG_DIR="$LOG_DIR"
SEED="$SEED"
BC_STEPS="$BC_STEPS"
DT_STEPS="$DT_STEPS"
MFQL_STEPS="$MFQL_STEPS"
MFBC_STEPS="$MFBC_STEPS"
DATASET_ENV="$DATASET_ENV"
TRAIN_ENV="$TRAIN_ENV"

mkdir -p "\$LOG_DIR"

echo "════════════════════════════════════════════════════════════"
echo "  Experiment : \$PROJ_WANDB"
echo "  Dataset    : \$DATASET_TYPE  (seed=\$SEED)"
echo "  Log dir    : \$LOG_DIR"
echo "════════════════════════════════════════════════════════════"
echo

# ── 1/3  Dataset generation ──────────────────────────────────────────────────
echo "[\$(date +%T)] 1/3  Generating \$DATASET_TYPE dataset..."
conda activate "\$DATASET_ENV"
DATASET_LOG="\$LOG_DIR/dataset_\$TIMESTAMP.log"
if [ "\$DATASET_TYPE" = "straight" ]; then
    python scripts/generate_straight_dataset.py 2>&1 | tee "\$DATASET_LOG"
else
    python scripts/generate_dataset.py 2>&1 | tee "\$DATASET_LOG"
fi
echo "[\$(date +%T)] Dataset complete."
echo

# ── 2/3  Launch 3 trainings in parallel ──────────────────────────────────────
echo "[\$(date +%T)] 2/3  Launching BC, DT, MeanFlowQL, MeanFlowBC in parallel..."
conda activate "\$TRAIN_ENV"

# Pick the 4 least-loaded GPUs right now (dataset gen is done, GPUs are free).
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used \
    --format=csv,noheader,nounits \
    | sort -t, -k2 -n | head -n 4 \
    | awk -F, '{gsub(/ /, "", \$1); print \$1}')
if [[ "\${#GPUS[@]}" -lt 4 ]]; then
    echo "ERROR: need 4 GPUs for training, found \${#GPUS[@]}" >&2; exit 1
fi
BC_GPU="\${GPUS[0]}"
DT_GPU="\${GPUS[1]}"
MFQL_GPU="\${GPUS[2]}"
MFBC_GPU="\${GPUS[3]}"

BC_LOG="\$LOG_DIR/bc_\$TIMESTAMP.log"
DT_LOG="\$LOG_DIR/dt_\$TIMESTAMP.log"
MFQL_LOG="\$LOG_DIR/mfql_\$TIMESTAMP.log"
MFBC_LOG="\$LOG_DIR/mfbc_\$TIMESTAMP.log"

CUDA_VISIBLE_DEVICES=\$BC_GPU python scripts/bc_zermelo.py \
    --seed=\$SEED --train_steps=\$BC_STEPS \
    > "\$BC_LOG" 2>&1 &
BC_PID=\$!

CUDA_VISIBLE_DEVICES=\$DT_GPU python scripts/dt_zermelo.py \
    --seed=\$SEED --train_steps=\$DT_STEPS \
    > "\$DT_LOG" 2>&1 &
DT_PID=\$!

CUDA_VISIBLE_DEVICES=\$MFQL_GPU python scripts/meanflowql_zermelo.py \
    --seed=\$SEED --offline_steps=\$MFQL_STEPS \
    > "\$MFQL_LOG" 2>&1 &
MFQL_PID=\$!

CUDA_VISIBLE_DEVICES=\$MFBC_GPU python scripts/meanflowbc_zermelo.py \
    --seed=\$SEED --offline_steps=\$MFBC_STEPS \
    > "\$MFBC_LOG" 2>&1 &
MFBC_PID=\$!

echo "  BC         → GPU \$BC_GPU    (PID \$BC_PID)    → \$(basename \$BC_LOG)"
echo "  DT         → GPU \$DT_GPU    (PID \$DT_PID)    → \$(basename \$DT_LOG)"
echo "  MeanFlowQL → GPU \$MFQL_GPU  (PID \$MFQL_PID)  → \$(basename \$MFQL_LOG)"
echo "  MeanFlowBC → GPU \$MFBC_GPU  (PID \$MFBC_PID)  → \$(basename \$MFBC_LOG)"
echo
echo "  Tail a log:  tail -f \$MFQL_LOG"
echo "  Waiting for all 3 to finish..."
echo

# ── 3/3 (prep)  Collect training exit codes ──────────────────────────────────
# set +e so a failed training doesn't abort the script before we run eval.
set +e
wait \$BC_PID;   BC_EXIT=\$?
wait \$DT_PID;   DT_EXIT=\$?
wait \$MFQL_PID; MFQL_EXIT=\$?
wait \$MFBC_PID; MFBC_EXIT=\$?
set -e

SUCCEEDED=""
FAILED=""
_check() {
    local name=\$1 code=\$2
    if [ "\$code" -eq 0 ]; then
        SUCCEEDED="\${SUCCEEDED}\${SUCCEEDED:+,}\$name"
        echo "  ✓ \$name succeeded"
    else
        FAILED="\${FAILED}\${FAILED:+,}\$name"
        echo "  ✗ \$name FAILED (exit \$code) — see \$LOG_DIR/\${name,,}_\$TIMESTAMP.log"
    fi
}
_check BC         \$BC_EXIT
_check DT         \$DT_EXIT
_check MeanFlowQL \$MFQL_EXIT
_check MeanFlowBC \$MFBC_EXIT
echo

if [ -z "\$SUCCEEDED" ]; then
    echo "[\$(date +%T)] All trainings failed — skipping evaluation."
    exit 1
fi
if [ -n "\$FAILED" ]; then
    echo "  Warning: \$FAILED failed. Running evaluation for: \$SUCCEEDED"
fi

# ── 3/3  Evaluation ──────────────────────────────────────────────────────────
echo "[\$(date +%T)] 3/3  Running evaluation (algos: \$SUCCEEDED)..."
EVAL_LOG="\$LOG_DIR/eval_\$TIMESTAMP.log"
ZERMELO_EVAL_ALGOS="\$SUCCEEDED" python scripts/evaluate_models.py 2>&1 | tee "\$EVAL_LOG"

echo
echo "════════════════════════════════════════════════════════════"
echo "  Experiment complete: \$PROJ_WANDB"
echo "  Logs    : \$LOG_DIR"
echo "  Results : results/\$PROJ_WANDB/"
echo "════════════════════════════════════════════════════════════"
INNER_EOF

# ── Launch the tmux session ──────────────────────────────────────────────────
tmux new -d -s "$SESSION" \
    "bash --login '$TMPSCRIPT'; echo; echo '[Session complete — press Enter to close]'; read; exec bash"

echo "Experiment '$PROJ_WANDB' launched."
echo
echo "  tmux session : $SESSION"
echo "  Attach       : tmux attach -t $SESSION"
echo "  Detach       : Ctrl-b then d"
echo "  Logs         : $LOG_DIR/"
echo
echo "List all running experiments:  tmux ls"
