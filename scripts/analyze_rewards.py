"""Reward analysis and hyperparameter tuning for the Zermelo dataset.

Loads a generated .npz dataset, reconstructs per-episode reward breakdowns,
and produces:
  1. Per-episode stacked bar chart: goal vs energy vs time vs distance components.
  2. Distribution plots: total reward, episode length, energy/time/distance fractions.
  3. Weight sweep: sweeps energy_weight, time_weight, and distance_weight over a
     grid and shows how the reward gap between best and worst episodes changes —
     useful for picking weights before re-running the environment.

Key insight: trajectories are fixed after generation. Changing reward weights
only changes *how you score* the same trajectories, not what they look like.
So you can sweep weights analytically using just the stored actions, episode
lengths, and distances, without re-running the environment.

Plots are saved to datasets/hyperparameter_tuning/.

Usage:
    # Analyze the default dataset with current weights from config:
    python scripts/analyze_rewards.py

    # Analyze a specific dataset file:
    python scripts/analyze_rewards.py --dataset datasets/my_dataset.npz

    # Use a specific config (for reward weight context):
    python scripts/analyze_rewards.py --config configs/zermelo_config.yaml
"""
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags

import zermelo_env  # noqa — registers gymnasium envs
from zermelo_env.zermelo_config import load_config

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', None, 'Path to .npz dataset. Defaults to the save_path in config.')
flags.DEFINE_string('config', None, 'Path to zermelo_config.yaml. Uses built-in defaults if omitted.')

SAVE_DIR = 'datasets/hyperparameter_tuning'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_latest_dataset(default_path: str) -> str:
    """Return default_path if it exists, otherwise search datasets/ for newest .npz."""
    if pathlib.Path(default_path).exists():
        return default_path
    npz_files = sorted(pathlib.Path('datasets').glob('*.npz'), key=lambda p: p.stat().st_mtime)
    if npz_files:
        return str(npz_files[-1])
    raise FileNotFoundError(
        'No dataset found. Run python scripts/generate_dataset.py first.')


def _split_episodes(terminals: np.ndarray):
    """Return list of (start_idx, end_idx_exclusive) for each episode."""
    ends = np.where(terminals > 0.5)[0]
    episodes = []
    start = 0
    for end in ends:
        episodes.append((start, end + 1))
        start = end + 1
    if start < len(terminals):  # truncated final episode
        episodes.append((start, len(terminals)))
    return episodes


def _get_dist_to_goal(data):
    """Return per-step distance-to-goal array, computing from obs if not stored."""
    if 'dist_to_goal' in data:
        return data['dist_to_goal']
    # obs[:, :2] = agent xy, obs[:, 2:4] = goal xy
    qpos = data['qpos']
    obs = data['observations']
    goal_xy = obs[:, 2:4]
    return np.linalg.norm(qpos - goal_xy, axis=1)


def _compute_episode_stats(data: dict, episodes: list, goal_reward: float,
                            energy_weight: float, time_weight: float,
                            distance_weight: float):
    """Compute per-episode reward breakdown using given weights.

    Works regardless of whether the dataset was generated with those weights,
    because we recompute from raw action norms, distances, and terminal flags.
    """
    actions = data['actions']          # (T, 2)
    dist_to_goal = _get_dist_to_goal(data)

    # Use stored goal_reward_components if available (nonzero on the success step).
    goal_components = data.get('goal_reward_components', None)

    ep_lengths, ep_goal, ep_energy, ep_time, ep_dist, ep_total = [], [], [], [], [], []
    for start, end in episodes:
        length = end - start
        action_norms = np.linalg.norm(actions[start:end], axis=1)

        # Determine whether goal was actually reached.
        if goal_components is not None:
            reached_goal = goal_components[start:end].sum() > 0.5
        else:
            # Fallback for older datasets without goal_reward_components:
            # check if any reward in the episode equals goal_reward (sparse).
            ep_rewards = data['rewards'][start:end]
            reached_goal = ep_rewards.max() >= goal_reward - 1e-6

        g = goal_reward if reached_goal else 0.0
        e = -energy_weight * action_norms.sum()
        t = -time_weight * length
        d = -distance_weight * dist_to_goal[start:end].sum()

        ep_lengths.append(length)
        ep_goal.append(g)
        ep_energy.append(e)
        ep_time.append(t)
        ep_dist.append(d)
        ep_total.append(g + e + t + d)

    return (np.array(ep_lengths), np.array(ep_goal),
            np.array(ep_energy), np.array(ep_time),
            np.array(ep_dist), np.array(ep_total))


def _save_fig(fig, name: str):
    out = pathlib.Path(SAVE_DIR) / f'{name}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  Saved {out}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1: Per-episode stacked bar chart
# ---------------------------------------------------------------------------

def plot_episode_breakdown(ep_goal, ep_energy, ep_time, ep_dist, ep_total,
                           goal_reward, energy_weight, time_weight, distance_weight):
    n = len(ep_goal)
    order = np.argsort(ep_total)  # sort by total reward ascending
    ep_goal = ep_goal[order]
    ep_energy = ep_energy[order]
    ep_time = ep_time[order]
    ep_dist = ep_dist[order]
    ep_total = ep_total[order]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), 5))
    x = np.arange(n)
    w = 0.2

    ax.bar(x - 1.5 * w, ep_goal, width=w, label='Goal reward', color='#4CAF50')
    ax.bar(x - 0.5 * w, ep_energy, width=w, label='Energy cost', color='#F44336')
    ax.bar(x + 0.5 * w, ep_time, width=w, label='Time cost', color='#FF9800')
    ax.bar(x + 1.5 * w, ep_dist, width=w, label='Distance cost', color='#9C27B0')
    ax.plot(x, ep_total, 'k.--', markersize=6, linewidth=1.2, label='Total')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xlabel('Episode (sorted by total reward)')
    ax.set_ylabel('Reward')
    ax.set_title(
        f'Per-episode reward breakdown\n'
        f'goal={goal_reward}, ew={energy_weight:.5f}, '
        f'tw={time_weight:.5f}, dw={distance_weight:.5f}')
    ax.legend(loc='upper left')
    fig.tight_layout()
    _save_fig(fig, 'episode_breakdown')


# ---------------------------------------------------------------------------
# Plot 2: Distributions
# ---------------------------------------------------------------------------

def plot_distributions(ep_lengths, ep_goal, ep_energy, ep_time, ep_dist, ep_total,
                       goal_reward, energy_weight, time_weight, distance_weight):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f'Reward distributions  (goal={goal_reward}, '
        f'ew={energy_weight}, tw={time_weight}, dw={distance_weight})',
        fontsize=12)

    def _hist(ax, data, title, color, xlabel):
        ax.hist(data, bins=30, color=color, edgecolor='white', linewidth=0.5)
        ax.axvline(data.mean(), color='black', linestyle='--', linewidth=1.2,
                   label=f'mean={data.mean():.3f}')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

    _hist(axes[0, 0], ep_total, 'Total reward per episode', '#2196F3', 'total reward')
    _hist(axes[0, 1], ep_lengths, 'Episode length', '#9C27B0', 'steps')
    _hist(axes[0, 2], ep_goal, 'Goal reward per episode', '#4CAF50', 'goal reward')
    _hist(axes[1, 0], ep_energy, 'Energy cost per episode', '#F44336', 'energy cost (negative)')
    _hist(axes[1, 1], ep_time, 'Time cost per episode', '#FF9800', 'time cost (negative)')
    _hist(axes[1, 2], ep_dist, 'Distance cost per episode', '#9C27B0', 'distance cost (negative)')

    fig.tight_layout()
    _save_fig(fig, 'distributions')


# ---------------------------------------------------------------------------
# Plot 3: Weight sweep heatmaps
# ---------------------------------------------------------------------------

def sweep_weights(data, episodes, goal_reward):
    """Sweep energy_weight, time_weight, and distance_weight over a grid.

    Selection criteria (in priority order):
      1. All successful episodes must have positive total reward (goal dominates).
      2. Median penalty fraction among successes should be in [10%, 30%] of
         goal_reward — penalties matter but don't dominate.
      3. Balance: each active penalty contributes >= 25% of total penalty.
      4. Maximize reward std among successful episodes only — so the offline
         algorithm can differentiate efficient vs inefficient goal-reaching.

    Falls back to relaxed constraints if nothing meets the strict criteria.
    """
    action_norms_per_ep = []
    dist_sums_per_ep = []
    lengths = []
    successes = []
    actions = data['actions']
    dist_to_goal = _get_dist_to_goal(data)
    goal_components = data.get('goal_reward_components', None)

    for start, end in episodes:
        action_norms_per_ep.append(np.linalg.norm(actions[start:end], axis=1).sum())
        dist_sums_per_ep.append(dist_to_goal[start:end].sum())
        lengths.append(end - start)
        if goal_components is not None:
            successes.append(goal_components[start:end].sum() > 0.5)
        else:
            successes.append(data['rewards'][start:end].max() >= goal_reward - 1e-6)

    action_norms_per_ep = np.array(action_norms_per_ep)
    dist_sums_per_ep = np.array(dist_sums_per_ep)
    lengths = np.array(lengths)
    successes = np.array(successes)
    succ_mask = successes.astype(bool)

    # Log data ranges.
    print(f'  Action norm totals — min: {action_norms_per_ep.min():.1f}, '
          f'max: {action_norms_per_ep.max():.1f}, mean: {action_norms_per_ep.mean():.1f}')
    print(f'  Episode lengths    — min: {lengths.min()}, '
          f'max: {lengths.max()}, mean: {lengths.mean():.0f}')
    print(f'  Dist-to-goal sums  — min: {dist_sums_per_ep.min():.1f}, '
          f'max: {dist_sums_per_ep.max():.1f}, mean: {dist_sums_per_ep.mean():.1f}')
    print(f'  Successful episodes: {succ_mask.sum()} / {len(succ_mask)}')

    # Scale sweep ranges so that at max weight, the median penalty equals goal_reward.
    succ_norms = action_norms_per_ep[succ_mask] if succ_mask.any() else action_norms_per_ep
    succ_lens = lengths[succ_mask] if succ_mask.any() else lengths
    succ_dists = dist_sums_per_ep[succ_mask] if succ_mask.any() else dist_sums_per_ep

    n_grid = 20  # per dimension (20^3 = 8000 combos)
    ew_max = goal_reward / (np.median(succ_norms) + 1e-9)
    tw_max = goal_reward / (np.median(succ_lens) + 1e-9)
    dw_max = goal_reward / (np.median(succ_dists) + 1e-9)
    ew_vals = np.linspace(0.0, ew_max, n_grid)
    tw_vals = np.linspace(0.0, tw_max, n_grid)
    dw_vals = np.linspace(0.0, dw_max, n_grid)

    best_combo = None
    best_spread = -1.0

    for lo, hi in [(0.10, 0.30), (0.05, 0.50), (0.0, 1.0)]:
        for min_balance in [0.25, 0.1, 0.0]:
            for ew in ew_vals:
                for tw in tw_vals:
                    for dw in dw_vals:
                        if ew == 0 and tw == 0 and dw == 0:
                            continue

                        g = successes.astype(float) * goal_reward
                        e = -ew * action_norms_per_ep
                        t = -tw * lengths
                        d = -dw * dist_sums_per_ep
                        total = g + e + t + d

                        if not succ_mask.any():
                            continue

                        succ_total = total[succ_mask]

                        # Constraint 1: all successful eps must have positive reward.
                        if np.any(succ_total <= 0):
                            continue

                        # Constraint 2: median penalty fraction in [lo, hi].
                        succ_penalties = goal_reward - succ_total
                        med_frac = float(np.median(succ_penalties)) / goal_reward
                        if not (lo <= med_frac <= hi):
                            continue

                        # Constraint 3: balance among active components.
                        active = []
                        if ew > 0:
                            active.append(np.median(np.abs(e[succ_mask])))
                        if tw > 0:
                            active.append(np.median(np.abs(t[succ_mask])))
                        if dw > 0:
                            active.append(np.median(np.abs(d[succ_mask])))

                        if len(active) >= 2:
                            total_active = sum(active)
                            if total_active > 1e-12:
                                fracs = [a / total_active for a in active]
                                if any(f < min_balance for f in fracs):
                                    continue

                        spread = float(np.std(succ_total))
                        if spread > best_spread:
                            best_spread = spread
                            best_combo = (ew, tw, dw, med_frac, spread)

            if best_combo is not None:
                print(f'  Selection: penalty frac [{lo:.0%}, {hi:.0%}], '
                      f'balance>={min_balance:.0%}, '
                      f'best success_std={best_spread:.4f}')
                break
        if best_combo is not None:
            break

    if best_combo is None:
        # Unconstrained fallback: just maximize spread.
        print('  Selection: unconstrained fallback (no combo met criteria)')
        for ew in ew_vals:
            for tw in tw_vals:
                for dw in dw_vals:
                    if ew == 0 and tw == 0 and dw == 0:
                        continue
                    g = successes.astype(float) * goal_reward
                    total = g - ew * action_norms_per_ep - tw * lengths - dw * dist_sums_per_ep
                    if succ_mask.any():
                        spread = float(np.std(total[succ_mask]))
                        if spread > best_spread:
                            best_spread = spread
                            succ_total = total[succ_mask]
                            med_frac = float(np.median(goal_reward - succ_total)) / goal_reward
                            best_combo = (ew, tw, dw, med_frac, spread)

    best_ew, best_tw, best_dw = best_combo[0], best_combo[1], best_combo[2]

    return dict(
        ew_vals=ew_vals, tw_vals=tw_vals, dw_vals=dw_vals,
        best_ew=best_ew, best_tw=best_tw, best_dw=best_dw,
        best_spread=best_spread,
        # Pass per-episode data for plots.
        action_norms_per_ep=action_norms_per_ep,
        dist_sums_per_ep=dist_sums_per_ep,
        lengths=lengths, successes=successes,
    )


def plot_penalty_scatter(sweep, goal_reward):
    """Scatter plots of penalty components against each other."""
    best_ew, best_tw, best_dw = sweep['best_ew'], sweep['best_tw'], sweep['best_dw']
    norms = sweep['action_norms_per_ep']
    lens = sweep['lengths']
    dists = sweep['dist_sums_per_ep']
    succs = sweep['successes'].astype(bool)

    energy_costs = best_ew * norms
    time_costs = best_tw * lens
    dist_costs = best_dw * dists

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f'Penalty scatter at recommended weights\n'
        f'ew={best_ew:.5f}, tw={best_tw:.5f}, dw={best_dw:.5f}',
        fontsize=11)

    for ax, (x_data, y_data, xlabel, ylabel) in zip(axes, [
        (energy_costs, time_costs, 'Energy cost', 'Time cost'),
        (energy_costs, dist_costs, 'Energy cost', 'Distance cost'),
        (time_costs, dist_costs, 'Time cost', 'Distance cost'),
    ]):
        ax.scatter(x_data[succs], y_data[succs],
                   c='#4CAF50', label='Reached goal', alpha=0.7, edgecolors='white', s=50)
        ax.scatter(x_data[~succs], y_data[~succs],
                   c='#F44336', label='Failed', alpha=0.7, edgecolors='white', s=50)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save_fig(fig, 'penalty_scatter')


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(ep_lengths, ep_goal, ep_energy, ep_time, ep_dist, ep_total,
                  goal_reward, energy_weight, time_weight, distance_weight):
    n = len(ep_lengths)
    successes = (ep_goal > 0).sum()
    abs_total = np.abs(ep_total).mean() + 1e-9

    print(f'\n{"="*65}')
    print(f'REWARD ANALYSIS SUMMARY')
    print(f'{"="*65}')
    print(f'Episodes: {n}  |  Successful: {successes} ({100*successes/n:.0f}%)')
    print(f'Current weights: goal_reward={goal_reward}, '
          f'energy_weight={energy_weight}, time_weight={time_weight}, '
          f'distance_weight={distance_weight}')
    print()
    print(f'{"Component":<18} {"Mean":>8} {"Std":>8} {"Min":>8} {"Max":>8} {"% of |total|":>13}')
    print('-' * 65)
    for name, arr in [('Total reward', ep_total), ('Goal reward', ep_goal),
                      ('Energy cost', ep_energy), ('Time cost', ep_time),
                      ('Distance cost', ep_dist)]:
        pct = 100 * abs(arr.mean()) / abs_total
        print(f'{name:<18} {arr.mean():>8.3f} {arr.std():>8.3f} '
              f'{arr.min():>8.3f} {arr.max():>8.3f} {pct:>12.0f}%')
    print()

    print(f'Episode lengths  — mean: {ep_lengths.mean():.0f}  '
          f'min: {ep_lengths.min()}  max: {ep_lengths.max()}')
    print(f'Reward gap (max-min): {ep_total.max() - ep_total.min():.3f}')
    print()

    goal_frac = abs(ep_goal.mean()) / abs_total
    energy_frac = abs(ep_energy.mean()) / abs_total
    time_frac = abs(ep_time.mean()) / abs_total
    dist_frac = abs(ep_dist.mean()) / abs_total

    print('TUNING GUIDANCE:')
    if goal_frac < 0.5:
        print('  [!] Goal reward is <50% of total — penalties may dominate, consider reducing weights.')
    if goal_frac > 0.95:
        print('  [!] Goal reward is >95% of total — penalties are negligible, consider increasing weights.')
    if energy_frac < 0.05 and energy_weight > 0:
        print('  [!] Energy cost is very small — try doubling energy_weight.')
    if time_frac < 0.05 and time_weight > 0:
        print('  [!] Time cost is very small — try doubling time_weight.')
    if dist_frac < 0.05 and distance_weight > 0:
        print('  [!] Distance cost is very small — try doubling distance_weight.')
    if ep_total.std() < 0.05:
        print('  [!] Very low reward variance — trajectories are indistinguishable. Increase weights.')
    succ_negative = ((ep_goal > 0) & (ep_total < 0)).sum()
    if succ_negative > 0:
        print(f'  [!] {succ_negative} successful episodes have negative total reward — penalties too high.')
    elif (ep_total < 0).mean() > 0.5:
        print('  [!] >50% of all episodes have negative total reward.')

    print(f'Plots saved to {SAVE_DIR}/')
    print(f'{"="*65}\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(_):
    cfg = load_config(FLAGS.config)
    reward_cfg = cfg['reward']
    goal_reward = reward_cfg['goal_reward']

    dataset_path = FLAGS.dataset or _find_latest_dataset(cfg['dataset']['save_path'])
    print(f'Loading dataset: {dataset_path}')
    data = dict(np.load(dataset_path))

    episodes = _split_episodes(data['terminals'])
    print(f'Found {len(episodes)} episodes, {len(data["terminals"])} total steps.')

    pathlib.Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    matplotlib.use('Agg')

    # --- Step 1: Sweep weight grid to find best combo ---
    print('Sweeping weight combinations...')
    sweep = sweep_weights(data, episodes, goal_reward)
    best_ew, best_tw, best_dw = sweep['best_ew'], sweep['best_tw'], sweep['best_dw']
    print(f'Best weights found: energy_weight={best_ew:.5f}, '
          f'time_weight={best_tw:.5f}, distance_weight={best_dw:.5f}')

    # --- Step 2: Compute episode stats at the best weights ---
    (ep_lengths, ep_goal, ep_energy, ep_time, ep_dist, ep_total) = _compute_episode_stats(
        data, episodes, goal_reward, best_ew, best_tw, best_dw)

    print_summary(ep_lengths, ep_goal, ep_energy, ep_time, ep_dist, ep_total,
                  goal_reward, best_ew, best_tw, best_dw)

    # --- Step 3: Generate all plots at the best weights ---
    print(f'Generating plots to {SAVE_DIR}/ ...')
    plot_penalty_scatter(sweep, goal_reward)
    plot_episode_breakdown(ep_goal, ep_energy, ep_time, ep_dist, ep_total,
                           goal_reward, best_ew, best_tw, best_dw)
    plot_distributions(ep_lengths, ep_goal, ep_energy, ep_time, ep_dist, ep_total,
                       goal_reward, best_ew, best_tw, best_dw)
    print(f'\nRecommended config values:')
    print(f'  energy_weight:   {best_ew:.5f}')
    print(f'  time_weight:     {best_tw:.5f}')
    print(f'  distance_weight: {best_dw:.5f}')
    print('Done.')


if __name__ == '__main__':
    app.run(main)
