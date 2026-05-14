# Zermelo Navigation

Offline RL for Zermelo navigation: a point agent navigates an arena (or
maze) with a time-varying background fluid flow. The goal is to learn
policies that **exploit** flow currents — like a boat using a river — by
stitching together the useful parts of many suboptimal demonstrations.

## Conda envs

Two envs, one for each phase of the pipeline:

```bash
# Generate datasets, run the env, build the flow cache.
conda env create -f environment-zermelo.yaml
conda activate zermelo
pip install -e .

# Train BC / DT / MeanFlowQL.
conda env create -f environment-flowrl.yaml
conda activate flowrl
```

The split exists because the training stack (jax + torch + wandb) doesn't
need to coexist with the data-gen stack (mujoco + xarray + netCDF).

## Layout

```
zermelo_config.yaml       # Single source of truth for env + dataset + reward.
zermelo_env/              # Custom Gymnasium env.
  zermelo_point.py          # Point-mass dynamics with HIT flow.
  zermelo_maze.py           # + maze walls, reward shaping, rendering.
  hit_chain.py / hit_cache.py   # Read flow snapshots via local-SSD cache.
  zermelo_config.py         # YAML loader → env kwargs.
scripts/
  build_hit_cache.py        # Transcode HIT*.nc → fast .bin (run once).
  generate_dataset.py       # Diverse offline dataset (waypoints + personality).
  generate_straight_dataset.py  # Straight-line oracle dataset.
  recompute_rewards.py      # Re-score an existing dataset with new weights.
  bc_zermelo.py             # Behavior Cloning training.
  dt_zermelo.py             # Decision Transformer training.
  meanflowql_zermelo.py     # MeanFlowQL training.
  visualize.py / test_env.py / evaluate_models.py / analyze_rewards.py
  _dataset_common.py        # Shared helpers for the two generators.
  _training_common.py       # Shared dataset loader + eval env for the trainers.
ext/MeanFlowQL/           # MeanFlowQL algo (git submodule).
datasets/                 # HIT*.nc inputs + generated .npz outputs.
```

## Running an experiment

```bash
# (Once) Build the local-SSD flow cache from HIT*.nc.
conda activate zermelo
PYTHONPATH=. python scripts/build_hit_cache.py

# 1. Edit zermelo_config.yaml — set num_episodes, reward weights, etc.

# 2. Generate the dataset (uses `zermelo` env).
PYTHONPATH=. python scripts/generate_dataset.py --num_workers=16
# or, for a clean straight-line baseline:
PYTHONPATH=. python scripts/generate_straight_dataset.py --num_workers=16

# 3. (Optional) Re-score an existing dataset with new reward weights —
#    no need to regenerate.
PYTHONPATH=. python scripts/recompute_rewards.py --from_config

# 4. (Optional) Visualize a handful of trajectories from the most recent
#    .npz to datasets/video.mp4.
PYTHONPATH=. python scripts/visualize.py

# 5. Train (uses `flowrl` env).
conda activate flowrl
python scripts/bc_zermelo.py        --seed=0 --train_steps=500000
python scripts/dt_zermelo.py        --seed=0 --train_steps=500000
python scripts/meanflowql_zermelo.py --seed=0 --offline_steps=1000000
```

All training scripts default to the dataset path in `zermelo_config.yaml`
(`run.save_path`) and log to wandb. Pass `--zermelo_dataset=/path/to/foo.npz`
to override.

## Reward

Per-step reward used both at generation time and inside the env:

```
reward = goal_reward · (reached this step)
       − (action_weight · ||action|| + fixed_hover_cost)
       + progress_weight · (prev_dist − curr_dist)
```

Weights live under `reward:` in `zermelo_config.yaml`. The energy term
bundles a dynamic action cost with a fixed hover cost (baseline power to
stay airborne in still air) — both are charged every step.

## Flow split

`flow.train_max_file: 45` means dataset generators only see `HIT1..HIT45`.
`HIT46..HIT49` are reserved for held-out evaluation.

## Registered envs

- `zermelo-pointmaze-medium-v0` — arena or maze (`maze.enabled` in config).
- `zermelo-pointmaze-medium-singletask{-task1..5}-v0` — single fixed task.
