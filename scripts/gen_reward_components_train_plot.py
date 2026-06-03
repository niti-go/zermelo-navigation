"""Generate the reward-component breakdown plot for the TRAINING flow segment.

Same approach as gen_reward_components_plot.py but uses train-segment start
frames.  Writes plots/reward_components_train.png into the latest results dir.

Usage
-----
    conda activate flowrl
    cd ~/zermelo-navigation
    python scripts/gen_reward_components_train_plot.py
"""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))   # so `import evaluate_models` works

import numpy as np

import evaluate_models as ev  # triggers GPU pick + env imports

# How many candidate episodes to try per algo when hunting for a
# success (so the goal component is visible). Falls back to best-return.
MAX_TRIES = 40
SEGMENT = 'train'


def main():
    import torch

    cfg = ev.load_config(None)
    ev.EXP_PROJECT = cfg['wandb_project_name']
    ev.SEGMENT_LABEL = ev._make_segment_labels(cfg)
    device = torch.device(ev.DEVICE)
    dataset_path = os.path.join(_REPO_ROOT, cfg['dataset_save_path'])

    # ── Resolve checkpoints + load policies ──────────────────────────────────
    print('Resolving checkpoints…')
    run_dirs = {}
    ckpts = {}
    skipped = []
    for a in ev.ALGO_ORDER:
        try:
            run_dirs[a] = ev.find_run_dir(a)
            ckpts[a] = ev.select_checkpoint(run_dirs[a], a, ev.CHECKPOINT_POLICY[a])
        except FileNotFoundError as e:
            print(f'  {a:12s}  SKIPPED — {e}')
            skipped.append(a)
    algos = [a for a in ev.ALGO_ORDER if a not in skipped]
    if not algos:
        raise RuntimeError('No algorithms have checkpoints — nothing to evaluate.')
    if skipped:
        print(f'  ⚠ Proceeding with: {", ".join(algos)}')
    ev.ALGO_ORDER = tuple(algos)

    print('Loading policies…')
    policies = {}
    if 'BC' in algos:
        policies['BC'] = ev.load_bc(run_dirs['BC'], ckpts['BC'], device)
    if 'DT' in algos:
        policies['DT'] = ev.load_dt(run_dirs['DT'], ckpts['DT'], device, dataset_path)[0]
    if 'MeanFlowQL' in algos:
        policies['MeanFlowQL'] = ev.load_mfql(run_dirs['MeanFlowQL'], ckpts['MeanFlowQL'])

    # ── Train-segment env + episode schedule ─────────────────────────────────
    env = ev.make_env(cfg)
    n_total = int(env.unwrapped.n_frames)
    n_train = ev.n_train_frames(cfg, env)
    print(f'Flow: total={n_total}, train=[0,{n_train}), held-out=[{n_train},{n_total})')

    start_frames, _ = ev.episode_start_frames(SEGMENT, MAX_TRIES, cfg, env)
    cells = ev.free_cells(env)
    tasks = ev.sample_episode_tasks(cells, MAX_TRIES, ev.SEED)

    # ── One representative episode per algo (first success, else best return) ─
    results = {a: [] for a in ev.ALGO_ORDER}
    for algo in ev.ALGO_ORDER:
        chosen, best = None, None
        for k in range(MAX_TRIES):
            init_ij, goal_ij = tasks[k]
            ep = ev.run_episode(env, policies[algo], float(start_frames[k]),
                                init_ij, goal_ij, record_video=False)
            ep.pop('frames', None)
            if best is None or ep['return'] > best['return']:
                best = ep
            if ep['success'] > 0.5:
                chosen = ep
                break
        chosen = chosen or best
        results[algo] = [chosen]
        print(f'  {algo:12s} chosen episode: success={chosen["success"]:.0f} '
              f'return={chosen["return"]:.1f} length={chosen["length"]}')

    env.close()

    # ── Plot ─────────────────────────────────────────────────────────────────
    import glob
    from datetime import datetime
    proj_root = os.path.join(ev.RESULTS_ROOT, ev.EXP_PROJECT)
    run_dirs_existing = sorted(
        d for d in glob.glob(os.path.join(proj_root, '*')) if os.path.isdir(d))
    run_dir = run_dirs_existing[-1] if run_dirs_existing else os.path.join(
        proj_root, datetime.now().strftime('%Y%m%d_%H%M%S'))
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, 'reward_components_train.png')
    ev.plot_reward_component_breakdown({SEGMENT: results}, out_path,
                                       segment=SEGMENT, cfg=cfg)
    print(f'\nDone. Plot: {out_path}')


if __name__ == '__main__':
    main()
