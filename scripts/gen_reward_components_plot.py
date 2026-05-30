"""Generate ONLY the reward-component breakdown plot, fast.

Reuses evaluate_models.py's loaders to run one representative held-out episode
per algorithm (BC / DT / MeanFlowQL) and writes plots/reward_components.png —
without running the full 200-episode × 2-segment evaluation.

Usage
-----
    conda activate flowrl
    cd ~/zermelo-navigation
    python scripts/gen_reward_components_plot.py
"""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))   # so `import evaluate_models` works

import numpy as np

import evaluate_models as ev  # triggers GPU pick + env imports

# How many candidate held-out episodes to try per algo when hunting for a
# success (so the goal component is visible). Falls back to best-return.
MAX_TRIES = 40
SEGMENT = 'heldout'


def main():
    import torch

    cfg = ev.load_config(None)
    ev.EXP_PROJECT = cfg['wandb_project_name']
    ev.SEGMENT_LABEL = ev._make_segment_labels(cfg)
    device = torch.device(ev.DEVICE)
    dataset_path = os.path.join(_REPO_ROOT, cfg['dataset_save_path'])

    # ── Resolve checkpoints + load policies ──────────────────────────────────
    print('Resolving checkpoints…')
    run_dirs = {a: ev.find_run_dir(a) for a in ev.ALGO_ORDER}
    ckpts = {a: ev.select_checkpoint(run_dirs[a], a, ev.CHECKPOINT_POLICY[a])
             for a in ev.ALGO_ORDER}

    print('Loading policies…')
    policies = {}
    if 'BC' in ev.ALGO_ORDER:
        policies['BC'] = ev.load_bc(run_dirs['BC'], ckpts['BC'], device)
    if 'DT' in ev.ALGO_ORDER:
        policies['DT'] = ev.load_dt(run_dirs['DT'], ckpts['DT'], device, dataset_path)[0]
    if 'MeanFlowQL' in ev.ALGO_ORDER:
        policies['MeanFlowQL'] = ev.load_mfql(run_dirs['MeanFlowQL'], ckpts['MeanFlowQL'])

    # ── Held-out env + episode schedule ──────────────────────────────────────
    env = ev.make_env(cfg)
    n_total = int(env.unwrapped.n_frames)
    n_train = ev.n_train_frames(cfg, env)
    print(f'Flow: total={n_total}, train={n_train}, held-out=[{n_train},{n_total})')

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
    # Write into the SAME plots/ folder evaluate_models.py uses: the latest
    # results/<EXP_PROJECT>/<timestamp>/plots dir. If no run exists yet, make a
    # fresh timestamped dir so the file still lands under the standard layout.
    import glob
    from datetime import datetime
    proj_root = os.path.join(ev.RESULTS_ROOT, ev.EXP_PROJECT)
    run_dirs_existing = sorted(
        d for d in glob.glob(os.path.join(proj_root, '*')) if os.path.isdir(d))
    run_dir = run_dirs_existing[-1] if run_dirs_existing else os.path.join(
        proj_root, datetime.now().strftime('%Y%m%d_%H%M%S'))
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, 'reward_components.png')
    ev.plot_reward_component_breakdown({SEGMENT: results}, out_path,
                                       segment=SEGMENT, cfg=cfg)
    print(f'\nDone. Plot: {out_path}')


if __name__ == '__main__':
    main()
