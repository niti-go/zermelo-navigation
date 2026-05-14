#!/usr/bin/env bash
# Launch BC, DT, and MeanFlowQL training runs in parallel detached tmux
# sessions, each pinned to a different GPU.
#
# Picks the N least-loaded GPUs (sorted by memory.used ascending) from
# `nvidia-smi`. Refuses to clobber an existing tmux session with the same
# name — kill it first if you want a fresh run.
#
# Usage:
#   bash scripts/launch_trainings.sh
#   SEED=42 bash scripts/launch_trainings.sh
#   PROJ_WANDB=straight20k_v1 bash scripts/launch_trainings.sh
#
# Env-var overrides (all optional):
#   SEED        — random seed                           (default 0)
#   BC_STEPS    — BC train_steps                        (default 500000)
#   DT_STEPS    — DT train_steps                        (default 500000)
#   MFQL_STEPS  — MeanFlowQL offline_steps              (default 1000000)
#   PROJ_WANDB  — wandb project                         (default "zermelo")
#
# Wandb entity is always RL_Control_JX (not configurable here).
#
# Tee'd logs land in logs/{bc,dt,mfql}_<timestamp>.log so detached runs
# stay recoverable.
set -euo pipefail

SEED="${SEED:-0}"
BC_STEPS="${BC_STEPS:-500000}"
DT_STEPS="${DT_STEPS:-500000}"
MFQL_STEPS="${MFQL_STEPS:-1000000}"
PROJ_WANDB="${PROJ_WANDB:-zermelo}"
WANDB_ENTITY="RL_Control_JX"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="flowrl"
LOG_DIR="$REPO_ROOT/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [[ ! -f "$CONDA_SH" ]]; then
    echo "ERROR: $CONDA_SH not found. Edit CONDA_SH in this script." >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null; then
    echo "ERROR: nvidia-smi not found." >&2
    exit 1
fi
if ! command -v tmux >/dev/null; then
    echo "ERROR: tmux not installed." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

# Pick the N least-loaded GPU indices, sorted by memory.used ascending.
pick_free_gpus() {
    local n=$1
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n \
        | head -n "$n" \
        | awk -F, '{gsub(/ /, "", $1); print $1}'
}

mapfile -t GPUS < <(pick_free_gpus 3)
if [[ "${#GPUS[@]}" -lt 3 ]]; then
    echo "ERROR: needed 3 GPUs, only found ${#GPUS[@]}." >&2
    exit 1
fi
BC_GPU="${GPUS[0]}"
DT_GPU="${GPUS[1]}"
MFQL_GPU="${GPUS[2]}"

# Refuse to clobber existing sessions.
for name in bc_zermelo dt_zermelo mfql_zermelo; do
    if tmux has-session -t "$name" 2>/dev/null; then
        echo "ERROR: tmux session '$name' already exists. Kill it first:" >&2
        echo "    tmux kill-session -t $name" >&2
        exit 1
    fi
done

# Launch one detached tmux session.
launch() {
    local session=$1 gpu=$2 log=$3 cmd=$4
    # Outer bash -lc wraps the whole thing so `conda activate` works inside
    # tmux's non-interactive shell. `; exec bash` keeps the session alive
    # after the command finishes so you can inspect output before closing.
    tmux new -d -s "$session" "bash -lc '
        set -e
        cd \"$REPO_ROOT\"
        source \"$CONDA_SH\"
        conda activate $CONDA_ENV
        export CUDA_VISIBLE_DEVICES=$gpu
        echo \"[\$(date +%T)] Starting $session on GPU $gpu (logs: $log)\"
        $cmd 2>&1 | tee \"$log\"
        echo
        echo \"[\$(date +%T)] $session finished. Press Enter to close session.\"
        read
    '; exec bash"
    echo "  $session  ->  GPU $gpu  ->  $log"
}

echo "Launching 3 training runs (seed=$SEED, conda env=$CONDA_ENV)"
echo "  wandb: project=$PROJ_WANDB (entity=$WANDB_ENTITY)"
echo "Picked GPUs (least-loaded): $BC_GPU, $DT_GPU, $MFQL_GPU"
echo

WANDB_ARGS="--proj_wandb=$PROJ_WANDB"

launch bc_zermelo   "$BC_GPU"   "$LOG_DIR/bc_${TIMESTAMP}.log" \
    "python scripts/bc_zermelo.py --seed=$SEED --train_steps=$BC_STEPS $WANDB_ARGS"
launch dt_zermelo   "$DT_GPU"   "$LOG_DIR/dt_${TIMESTAMP}.log" \
    "python scripts/dt_zermelo.py --seed=$SEED --train_steps=$DT_STEPS $WANDB_ARGS"
launch mfql_zermelo "$MFQL_GPU" "$LOG_DIR/mfql_${TIMESTAMP}.log" \
    "python scripts/meanflowql_zermelo.py --seed=$SEED --offline_steps=$MFQL_STEPS $WANDB_ARGS"

echo
echo "All sessions launched."
echo
echo "Attach with:"
echo "  tmux attach -t bc_zermelo"
echo "  tmux attach -t dt_zermelo"
echo "  tmux attach -t mfql_zermelo"
echo
echo "Detach from a session with Ctrl-b then d."
echo "Tail logs from anywhere:"
echo "  tail -f $LOG_DIR/bc_${TIMESTAMP}.log"
echo
echo "List your tmux sessions:  tmux ls"
echo "Kill one (after done):    tmux kill-session -t bc_zermelo"
echo
echo "Wandb runs will appear at:"
echo "  https://wandb.ai/$WANDB_ENTITY/$PROJ_WANDB"
