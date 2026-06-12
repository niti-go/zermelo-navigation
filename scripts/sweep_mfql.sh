#!/usr/bin/env bash
# Sweep MeanFlowQL configs: launch one MFQL run per spec, each pinned to its own
# least-loaded GPU in a detached tmux session. All runs read the SAME dataset
# from zermelo_config.yaml (dataset_save_path) and log to the same wandb project,
# separated by run_group so they stay distinguishable.
#
# Each spec is "LABEL|<agent overrides>". LABEL names the wandb run_group, the
# tmux session (mfql_<LABEL>) and the log file. The overrides are passed through
# verbatim to meanflowql_zermelo.py (config_flags accepts --agent.<key>=<val>).
#
# Usage:
#   bash scripts/sweep_mfql.sh
#   SEED=1 bash scripts/sweep_mfql.sh
#   MFQL_STEPS=500000 bash scripts/sweep_mfql.sh
#   PROJ_WANDB=my_sweep bash scripts/sweep_mfql.sh
#
# Edit the SPECS array below to define the sweep.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEED="${SEED:-0}"
MFQL_STEPS="${MFQL_STEPS:-1000000}"
PROJ_WANDB="${PROJ_WANDB:-$(python3 -c "import yaml; print(yaml.safe_load(open('$REPO_ROOT/zermelo_config.yaml'))['wandb_project_name'])")}"
WANDB_ENTITY="RL_Control_JX"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="flowrl"
LOG_DIR="$REPO_ROOT/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── Define the sweep here ─────────────────────────────────────────────────────
# "label|<agent overrides>"  — one tmux session + GPU per entry.
SPECS=(
    "alpha1|--agent.alpha=1   --agent.use_dynamic_alpha=False"
    "alpha3|--agent.alpha=3   --agent.use_dynamic_alpha=False"
    "alpha10|--agent.alpha=10 --agent.use_dynamic_alpha=False"
)
# ──────────────────────────────────────────────────────────────────────────────

N="${#SPECS[@]}"

if [[ ! -f "$CONDA_SH" ]]; then
    echo "ERROR: $CONDA_SH not found. Edit CONDA_SH in this script." >&2
    exit 1
fi
for cmd in python3 nvidia-smi tmux; do
    command -v "$cmd" >/dev/null || { echo "ERROR: '$cmd' not found." >&2; exit 1; }
done

mkdir -p "$LOG_DIR"

# Pick the N least-loaded GPU indices, sorted by memory.used ascending.
pick_free_gpus() {
    local n=$1
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n \
        | head -n "$n" \
        | awk -F, '{gsub(/ /, "", $1); print $1}'
}

mapfile -t GPUS < <(pick_free_gpus "$N")
if [[ "${#GPUS[@]}" -lt "$N" ]]; then
    echo "ERROR: need $N GPUs (one per spec), only found ${#GPUS[@]}." >&2
    exit 1
fi

# Launch one detached tmux session (mirrors launch_trainings.sh).
launch() {
    local session=$1 gpu=$2 log=$3 cmd=$4
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

echo "MeanFlowQL sweep ($N runs, seed=$SEED, conda env=$CONDA_ENV)"
echo "  wandb: project=$PROJ_WANDB (entity=$WANDB_ENTITY)"
echo "  GPUs (least-loaded): ${GPUS[*]}"
echo

for i in "${!SPECS[@]}"; do
    spec="${SPECS[$i]}"
    label="${spec%%|*}"
    overrides="${spec#*|}"
    session="mfql_${label}"
    gpu="${GPUS[$i]}"
    log="$LOG_DIR/mfql_${label}_${TIMESTAMP}.log"

    # Kill any stale session with the same name so we start fresh.
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "Killing existing tmux session '$session'..."
        tmux kill-session -t "$session"
    fi

    launch "$session" "$gpu" "$log" \
        "python scripts/meanflowql_zermelo.py \
            --seed=$SEED --offline_steps=$MFQL_STEPS \
            --proj_wandb=$PROJ_WANDB --run_group=mfql_${label} \
            $overrides"
done

echo
echo "All sweep runs launched. Each run_group is mfql_<label> in wandb."
echo "Reattach:  tmux attach -t mfql_${SPECS[0]%%|*}"
echo "List:      tmux ls"
echo "Wandb:     https://wandb.ai/$WANDB_ENTITY/$PROJ_WANDB"
