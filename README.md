# Zermelo Navigation

Offline reinforcement learning for a Zermelo navigation problem — a point agent navigating a maze with a background fluid flow field.

**The core question:** can an offline RL agent learn to *exploit* fluid currents (like a boat using river currents) by stitching together the best parts of many suboptimal demonstration trajectories?

## Structure

```
configs/                  # Environment and dataset generation config
scripts/                  # Dataset generation, visualization, evaluation
zermelo_env/              # Custom Gymnasium environment (self-contained)
ext/                      # External dependencies (git submodules)
  ogbench/                # Fork of seohongpark/ogbench (training infrastructure)
  MeanFlowQL/             # Fork of HiccupRL/MeanFlowQL (offline RL algorithm)
experiments/              # Training scripts and results for all experiments
results/                  # Runtime outputs: datasets, checkpoints (gitignored)
```

## Setup

```bash
# Clone with submodules
git clone --recurse-submodules <repo-url>

# Install the Zermelo environment
conda activate zermelo
pip install -e .

# (Optional) Install ogbench for training infrastructure
pip install -e ext/ogbench
```

## Workflow

```bash
# 1. Edit configs/zermelo_config.yaml to set env params

# 2. Generate dataset
python scripts/generate_dataset.py

# 3. Train (e.g. with MeanFlowQL)
cd ext/MeanFlowQL
python main.py --env_name=zermelo-pointmaze-medium-v0 ...

# 4. Evaluate
python scripts/test_env.py --policy oracle
```

## Environment

The `zermelo_env` package provides Gymnasium environments:

- `zermelo-pointmaze-medium-v0` — maze with fluid flow
- `zermelo-pointarena-medium-v0` — open arena (boundary only) with fluid flow

All parameters (maze layout, flow field, start/goal, rewards, etc.) are controlled by `configs/zermelo_config.yaml`.
