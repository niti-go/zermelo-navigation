"""Analyze the reward distribution in an offline Zermelo dataset.

Reads a .npz, re-derives per-episode reward components from the raw
arrays, prints a clean breakdown at the current config weights, shows
how sensitive the mean total reward is to scaling each weight, and saves
a few distribution / scatter plots.

The per-step reward formula matches the env exactly:

    reward = goal_reward · (reached this step)
           − (action_weight · ||action|| + fixed_hover_cost)
           + progress_weight · (prev_dist − curr_dist)

This script doesn't trust the saved `data['rewards']` — it recomputes
components from raw `actions`, `dist_to_goal`, `terminals`, and
`goal_reward_components`. So you can analyze a dataset under any weights
without regenerating it. To bake new weights into the .npz, follow up
with `recompute_rewards.py`.

Usage:
    # Analyze with current config weights:
    PYTHONPATH=. python scripts/helpers/analyze_rewards.py

    # Override one or more weights for this run only (config unchanged):
    PYTHONPATH=. python scripts/helpers/analyze_rewards.py \
        --action_weight=0.7 --fixed_hover_cost=0.15

    # Choose a different "what fraction of goal_reward should each
    # component contribute?" target for the recommendation block:
    PYTHONPATH=. python scripts/helpers/analyze_rewards.py --target_fraction=0.4
"""
import os
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags

# This file lives at scripts/helpers/. Add its own dir for sibling imports
# (training_common is in the same folder) and the repo root for zermelo_env.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

import training_common as tc  # noqa: E402
from zermelo_env.zermelo_config import load_config  # noqa: E402

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', None, 'Path to .npz dataset. Defaults to config save_path.')
flags.DEFINE_string('config', None, 'Path to zermelo_config.yaml.')
flags.DEFINE_string('save_dir', 'datasets/hyperparameter_tuning',
                    'Directory for plot outputs.')
flags.DEFINE_float('target_fraction', 0.3,
                   'For recommendations: target |component| / goal_reward '
                   'at the median episode (0.3 = 30%).')
flags.DEFINE_float('goal_reward', None, 'Override goal_reward for this run.')
flags.DEFINE_float('action_weight', None, 'Override action_weight for this run.')
flags.DEFINE_float('fixed_hover_cost', None, 'Override fixed_hover_cost for this run.')
flags.DEFINE_float('progress_weight', None, 'Override progress_weight for this run.')

# Scatter plots cap at this many points (subsampled deterministically) so
# they remain readable + memory-bounded on 20k-episode datasets.
MAX_SCATTER_POINTS = 2000


# ---------------------------------------------------------------------------
# Loading + per-episode components
# ---------------------------------------------------------------------------

def load_segments(dataset_path):
    """Return (data, segments) without shuffling — analyzes whole dataset."""
    print(f'Loading dataset: {dataset_path}')
    data = dict(np.load(dataset_path))
    for k in ('actions', 'dist_to_goal', 'terminals', 'goal_reward_components'):
        if k in data:
            data[k] = data[k].astype(np.float32)
    terminals = data['terminals']
    ends = np.where(terminals > 0.5)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])
    segments = [(int(s), int(e) + 1) for s, e in zip(starts, ends)]
    print(f'  {len(segments)} episodes, {len(terminals)} total transitions')
    return data, segments


def compute_episode_components(data, segments):
    """Re-derive per-episode quantities needed to score any weight choice.

    Returns a dict of numpy arrays of length n_episodes:
      length            — number of steps in the episode
      action_norms_sum  — sum of ||action|| over the episode
      success           — bool, did the agent reach the goal
      init_dist         — distance to goal at start (approx; see note)
      final_dist        — distance to goal at last stored step
      dist_travelled    — init_dist - final_dist
                          (what progress_weight × telescopes to)

    Note: `init_dist` uses the first stored `dist_to_goal[s]`, which is
    the POST-step-1 distance (the env stores per-step values after the
    dynamics update). The approximation drops < a few % progress per
    episode and is fine for analysis-grade weight selection. For an
    exact recomputation use `recompute_rewards.py`.
    """
    actions = data['actions']
    dist_to_goal = data['dist_to_goal']
    goal_components = data['goal_reward_components']

    length, action_norms_sum, success, init_dist, final_dist = [], [], [], [], []
    for s, e in segments:
        length.append(e - s)
        action_norms_sum.append(np.linalg.norm(actions[s:e], axis=1).sum())
        success.append(goal_components[s:e].sum() > 0.5)
        init_dist.append(float(dist_to_goal[s]))
        final_dist.append(float(dist_to_goal[e - 1]))
    out = dict(
        length=np.array(length),
        action_norms_sum=np.array(action_norms_sum, dtype=np.float64),
        success=np.array(success),
        init_dist=np.array(init_dist),
        final_dist=np.array(final_dist),
    )
    out['dist_travelled'] = out['init_dist'] - out['final_dist']
    return out


def score(components, goal_reward, action_weight, fixed_hover_cost, progress_weight):
    """Per-episode reward breakdown at the given weights."""
    goal = goal_reward * components['success'].astype(np.float32)
    action_cost = -action_weight * components['action_norms_sum']
    hover_cost = -fixed_hover_cost * components['length']
    progress = progress_weight * components['dist_travelled']
    total = goal + action_cost + hover_cost + progress
    return dict(goal=goal, action_cost=action_cost, hover_cost=hover_cost,
                progress=progress, total=total)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_summary(scored, components, goal_reward, action_weight,
                  fixed_hover_cost, progress_weight):
    n = len(scored['total'])
    succ_count = int(components['success'].sum())
    succ_frac = succ_count / max(n, 1)

    print(f'\n{"=" * 72}')
    print(f'REWARD ANALYSIS  ({n} episodes, success rate {succ_frac:.1%})')
    print(f'{"=" * 72}')
    print(f'Weights: goal={goal_reward}, action_weight={action_weight}, '
          f'fixed_hover_cost={fixed_hover_cost}, progress_weight={progress_weight}')
    print()
    print(f'{"Component":<14} {"mean":>9} {"std":>9} {"min":>9} {"max":>9} {"% of goal":>10}')
    print('-' * 72)
    for name, arr in [
        ('Total',       scored['total']),
        ('Goal',        scored['goal']),
        ('Action cost', scored['action_cost']),
        ('Hover cost',  scored['hover_cost']),
        ('Progress',    scored['progress']),
    ]:
        pct = 100 * abs(arr.mean()) / max(abs(goal_reward), 1e-9)
        print(f'{name:<14} {arr.mean():>9.2f} {arr.std():>9.2f} '
              f'{arr.min():>9.2f} {arr.max():>9.2f} {pct:>9.1f}%')
    print()
    print(f'Episode length — mean: {components["length"].mean():.0f}, '
          f'median: {np.median(components["length"]):.0f}, '
          f'min: {components["length"].min()}, max: {components["length"].max()}')

    # Diagnostics.
    print('\nDIAGNOSTICS:')
    succ_mask = components['success'].astype(bool)
    if succ_mask.any():
        succ_total = scored['total'][succ_mask]
        succ_neg = int((succ_total < 0).sum())
        succ_neg_pct = 100 * succ_neg / len(succ_total)
        print(f'  Successful episodes with negative total reward: '
              f'{succ_neg}/{len(succ_total)} ({succ_neg_pct:.1f}%)')
        if succ_neg_pct > 25:
            print('  [!] Penalties likely too high — most successes are punished.')
        elif succ_neg_pct < 1 and components['length'].max() > 200:
            print('  [i] Even long-path successes stay positive — energy could go higher '
                  'if you want more differentiation between short and long paths.')
    if abs(scored['progress'].mean()) < 0.05 * goal_reward:
        print('  [i] Progress contribution is <5% of goal_reward — bump '
              'progress_weight if you want a stronger dense signal.')
    if abs(scored['hover_cost'].mean()) < 0.02 * goal_reward and fixed_hover_cost > 0:
        print('  [i] Hover cost is <2% of goal — set fixed_hover_cost=0 if you '
              'don\'t want a time-pressure term, or raise it for stronger pressure.')


def print_sensitivity(components, goal_reward, action_weight,
                      fixed_hover_cost, progress_weight):
    """How does mean total reward change when each weight is scaled?"""
    scales = [0.25, 0.5, 1.0, 2.0, 4.0]
    print(f'\n--- Sensitivity: mean total reward when scaling one weight at a time ---')
    print(f'{"scale":<8} {"action_weight":>16} {"fixed_hover_cost":>20} {"progress_weight":>17}')
    print('-' * 64)
    for s in scales:
        m_aw = score(components, goal_reward, action_weight * s,
                     fixed_hover_cost, progress_weight)['total'].mean()
        m_hc = score(components, goal_reward, action_weight,
                     fixed_hover_cost * s, progress_weight)['total'].mean()
        m_pw = score(components, goal_reward, action_weight,
                     fixed_hover_cost, progress_weight * s)['total'].mean()
        marker = '  <-- current' if s == 1.0 else ''
        print(f'{f"{s}x":<8} {m_aw:>16.2f} {m_hc:>20.2f} {m_pw:>17.2f}{marker}')


def print_recommendations(components, goal_reward, target_fraction):
    """Solve for weights so each component ≈ `target_fraction * goal_reward`
    in magnitude at the *median* episode."""
    median_len = float(np.median(components['length']))
    median_action_sum = float(np.median(components['action_norms_sum']))
    median_dist_travelled = float(np.median(components['dist_travelled']))
    target = goal_reward * target_fraction

    rec_aw = target / max(median_action_sum, 1e-9)
    rec_hc = target / max(median_len, 1e-9)
    rec_pw = target / max(median_dist_travelled, 1e-9)

    print(f'\n--- Recommended weights (each component ≈ {target_fraction:.0%} of '
          f'goal_reward at the median episode) ---')
    print(f'  action_weight    ≈ {rec_aw:.4f}    '
          f'(median action_sum = {median_action_sum:.1f})')
    print(f'  fixed_hover_cost ≈ {rec_hc:.4f}    '
          f'(median episode length = {median_len:.0f})')
    print(f'  progress_weight  ≈ {rec_pw:.4f}    '
          f'(median dist_travelled = {median_dist_travelled:.1f})')
    print('\nThese three are tuned independently — combining all of them brings the')
    print(f'median total down by roughly {3 * target:.1f}, leaving '
          f'~{goal_reward - 3 * target:.1f} of net goal_reward at the median.')


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save(fig, save_dir, name):
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    p = os.path.join(save_dir, f'{name}.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f'  Saved {p}')
    plt.close(fig)


def plot_distributions(scored, components, save_dir, weights_label):
    """2×3 grid of histograms over per-episode quantities."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Reward distributions ({weights_label})', fontsize=12)

    def _hist(ax, data, title, color, xlabel):
        ax.hist(data, bins=40, color=color, edgecolor='white', linewidth=0.4)
        ax.axvline(data.mean(), color='black', linestyle='--', linewidth=1.2,
                   label=f'mean={data.mean():.2f}')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('count')
        ax.legend(fontsize=8)

    _hist(axes[0, 0], scored['total'],        'Total reward',    '#2196F3', 'reward')
    _hist(axes[0, 1], components['length'],   'Episode length',  '#9C27B0', 'steps')
    _hist(axes[0, 2], scored['goal'],         'Goal reward',     '#4CAF50', 'reward')
    _hist(axes[1, 0], scored['action_cost'],  'Action cost',     '#F44336', 'reward (negative)')
    _hist(axes[1, 1], scored['hover_cost'],   'Hover cost',      '#FF9800', 'reward (negative)')
    _hist(axes[1, 2], scored['progress'],     'Progress reward', '#3F51B5', 'reward')

    fig.tight_layout()
    _save(fig, save_dir, 'distributions')


def _subsample(n, cap=MAX_SCATTER_POINTS):
    if n <= cap:
        return np.arange(n)
    return np.random.RandomState(0).choice(n, cap, replace=False)


def plot_length_vs_reward(scored, components, save_dir, weights_label):
    """Episode length vs total reward, colored by success."""
    n = len(scored['total'])
    idx = _subsample(n)
    succ = components['success'][idx].astype(bool)
    x = components['length'][idx]
    y = scored['total'][idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x[succ], y[succ], c='#4CAF50', alpha=0.5, s=12,
               edgecolors='none', label=f'success ({int(succ.sum())})')
    ax.scatter(x[~succ], y[~succ], c='#F44336', alpha=0.5, s=12,
               edgecolors='none', label=f'fail ({int((~succ).sum())})')
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xlabel('episode length (steps)')
    ax.set_ylabel('total reward')
    note = f'subsampled to {len(idx)} of {n}' if n > len(idx) else f'{n} episodes'
    ax.set_title(f'Episode length vs total reward  ({weights_label})  [{note}]')
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, save_dir, 'length_vs_reward')


def plot_component_correlations(scored, save_dir, weights_label):
    """Pairwise scatter between the three penalty/shaping components."""
    n = len(scored['total'])
    idx = _subsample(n)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    note = f'subsampled to {len(idx)} of {n}' if n > len(idx) else f'{n} episodes'
    fig.suptitle(f'Component correlations ({weights_label})  [{note}]', fontsize=11)
    pairs = [
        (scored['action_cost'][idx], scored['hover_cost'][idx],
         'Action cost', 'Hover cost'),
        (scored['action_cost'][idx], scored['progress'][idx],
         'Action cost', 'Progress'),
        (scored['hover_cost'][idx], scored['progress'][idx],
         'Hover cost', 'Progress'),
    ]
    for ax, (x, y, xl, yl) in zip(axes, pairs):
        ax.scatter(x, y, c='#2196F3', alpha=0.4, s=10, edgecolors='none')
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    fig.tight_layout()
    _save(fig, save_dir, 'component_correlations')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(_):
    cfg = load_config(FLAGS.config)
    reward_cfg = cfg['reward']
    energy_cfg = reward_cfg['energy']

    # Config defaults, with optional CLI overrides for what-if analysis.
    goal_reward = (FLAGS.goal_reward if FLAGS.goal_reward is not None
                   else reward_cfg['goal_reward'])
    action_weight = (FLAGS.action_weight if FLAGS.action_weight is not None
                     else energy_cfg['action_weight'])
    fixed_hover_cost = (FLAGS.fixed_hover_cost if FLAGS.fixed_hover_cost is not None
                        else energy_cfg['fixed_hover_cost'])
    progress_weight = (FLAGS.progress_weight if FLAGS.progress_weight is not None
                       else reward_cfg['progress_weight'])
    weights_label = (f'goal={goal_reward}, aw={action_weight}, '
                     f'hc={fixed_hover_cost}, pw={progress_weight}')

    dataset_path = FLAGS.dataset or tc.default_dataset_path(cfg)
    data, segments = load_segments(dataset_path)

    components = compute_episode_components(data, segments)
    scored = score(components, goal_reward, action_weight,
                   fixed_hover_cost, progress_weight)

    print_summary(scored, components, goal_reward, action_weight,
                  fixed_hover_cost, progress_weight)
    print_sensitivity(components, goal_reward, action_weight,
                      fixed_hover_cost, progress_weight)
    print_recommendations(components, goal_reward, FLAGS.target_fraction)

    matplotlib.use('Agg')
    print(f'\nGenerating plots in {FLAGS.save_dir}/ ...')
    plot_distributions(scored, components, FLAGS.save_dir, weights_label)
    plot_length_vs_reward(scored, components, FLAGS.save_dir, weights_label)
    plot_component_correlations(scored, FLAGS.save_dir, weights_label)
    print('Done.')


if __name__ == '__main__':
    app.run(main)
