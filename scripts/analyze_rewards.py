"""Reward weight selection for the Zermelo offline RL dataset.

BACKGROUND
----------
The Zermelo environment rewards each trajectory with four components:

    reward_per_step = goal_reward * (reached goal this step)
                    - energy_weight * ||action||
                    - time_weight * 1
                    - distance_weight * dist_to_goal

The goal_reward is a large sparse bonus on the step the agent reaches the
goal. The three penalty weights control how harshly the reward function
penalizes energy use, elapsed time, and distance from the goal at every step.

We need to choose the three penalty weights so that:
  (a) Goal-reaching dominates: successful episodes have positive total reward.
  (b) Penalties meaningfully differentiate efficient vs. inefficient paths.
  (c) The per-step reward provides a useful dense learning signal for the
      offline RL algorithm (not just a sparse goal bonus).

KEY INSIGHT
-----------
Trajectories are fixed after dataset generation. Changing reward weights only
changes how you *score* existing trajectories, not what they look like. So we
can sweep weights analytically from stored actions, episode lengths, and
distances — no need to re-run the environment.

WHY TWO STAGES
--------------
Energy, time, and distance penalties are highly correlated: longer episodes
accumulate more energy, more steps, and more cumulative distance. Jointly
sweeping all three on a grid doesn't work well because:

  1. The penalties move together, so a "balance" constraint (each component
     contributing a meaningful fraction) is hard to satisfy with 3 correlated
     axes — one always dominates.
  2. The distance penalty serves a fundamentally different purpose than
     energy/time. Energy and time differentiate *trajectory quality* at the
     episode level (was this path efficient?). Distance provides a *dense
     per-step learning signal* (am I getting closer to the goal right now?).
     An objective that maximizes episode-level reward spread doesn't capture
     the value of per-step signal quality.
  3. Distance sums can be orders of magnitude larger than energy sums or step
     counts, so interesting distance_weight values fall between grid points.

Because of this, energy/time and distance are tuned separately.

STAGE 1 — energy_weight and time_weight
----------------------------------------
These two weights control trajectory quality differentiation. We sweep a 2D
grid of (energy_weight, time_weight) combinations and pick the one that
maximizes reward variance among successful episodes, subject to:

  - All successful episodes keep reward above goal_reward/3 after subtracting energy+time
    penalties. That leftover third is reserved for the distance penalty added
    in Stage 2 (so the total can stay positive once all three penalties apply).
  - The median energy+time penalty fraction falls in a moderate band (the code
    tries 10%-30% first, relaxing to wider bands if no combo qualifies).

To make the grid meaningful, raw signals (raw total action norms (energy)
and raw step counts (time)) are normalized to unit variance
before sweeping. This ensures the grid covers comparable effect ranges for
both axes regardless of their raw magnitude differences. The grid is
log-spaced for better coverage across dynamic ranges, and a refinement pass
runs a fine grid around the best coarse result.

STAGE 2 — distance_weight
--------------------------
Rather than optimizing distance_weight for episode-level spread (where it
adds little beyond energy/time), we set it to produce a useful per-step
signal for the RL algorithm. Specifically:

  distance_weight is scaled so that the median per-step distance penalty is
  comparable in magnitude (1:1 ratio) to the median per-step energy + time
  penalty.

This means at every step, the agent "feels" the distance gradient at roughly
the same scale as the action cost — enough to provide learning signal without
overwhelming it. The weight is then capped so that the total distance penalty
stays within the reserved budget, and at least 90% of successful episodes
keep positive total reward. The remaining ~10% that go negative are the
longest, most inefficient trajectories — exactly the ones the RL algorithm
should learn to avoid.

OUTPUTS
-------
  - Recommended energy_weight, time_weight, distance_weight values.
  - Per-episode stacked bar chart: goal vs. energy vs. time vs. distance.
  - Distribution plots: total reward, episode length, component breakdowns.
  - Penalty scatter plots: correlations between penalty components.

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
    # Fallback for datasets without dist_to_goal stored.
    # obs layout: [qpos_x, qpos_y, flow_vx, flow_vy, goal_x, goal_y]
    qpos = data['qpos']
    if 'goal_xy' in data:
        goal_xy = data['goal_xy']
    else:
        obs = data['observations']
        goal_xy = obs[:, 4:6]
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

def _compute_distance_weight(data, episodes, succ_mask, action_norms_per_ep,
                              lengths, energy_weight, time_weight,
                              target_ratio=1.0):
    """Set distance_weight so per-step distance penalty ≈ target_ratio × per-step energy+time penalty.

    This gives the RL agent a dense gradient signal at every step that is
    comparable in magnitude to the other per-step penalties, without needing
    to jointly optimize it in the sweep.

    Args:
        target_ratio: desired ratio of median per-step distance penalty to
            median per-step (energy + time) penalty. 1.0 means equal magnitude.

    Returns:
        distance_weight (float)
    """
    dist_to_goal = _get_dist_to_goal(data)
    actions = data['actions']

    per_step_energy = []  # per-step energy penalty magnitude
    per_step_time = []    # per-step time penalty (just the weight)
    per_step_dist = []    # per-step raw distance (before weighting)

    for i, (start, end) in enumerate(episodes):
        if not succ_mask[i]:
            continue
        length = end - start
        norms = np.linalg.norm(actions[start:end], axis=1)
        per_step_energy.append(energy_weight * norms.mean())
        per_step_time.append(time_weight)
        per_step_dist.append(dist_to_goal[start:end].mean())

    if not per_step_dist:
        return 0.0

    median_other = np.median(per_step_energy) + np.median(per_step_time)
    median_raw_dist = np.median(per_step_dist)

    if median_raw_dist < 1e-12:
        return 0.0

    dw = target_ratio * median_other / median_raw_dist
    return float(dw)


def sweep_weights(data, episodes, goal_reward):
    """Two-stage weight selection: sweep ew×tw on a log-spaced 2D grid, then
    set distance_weight independently based on per-step signal magnitude.

    Stage 1 — energy_weight × time_weight sweep:
      Normalizes raw signals to unit variance so the grid covers comparable
      effect ranges. Uses a log-spaced grid for better dynamic range coverage.
      Constraints:
        1. All successful episodes have positive total reward.
        2. Median penalty fraction in [10%, 30%] of goal_reward.
      Objective: maximize std(reward) among successful episodes.

    Stage 2 — distance_weight (set independently):
      Scales dw so the median per-step distance penalty is comparable to the
      median per-step energy+time penalty. This ensures dense learning signal
      without needing to jointly optimize — distance serves a fundamentally
      different purpose (per-step gradient) than energy/time (trajectory quality).
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
    lengths = np.array(lengths, dtype=float)
    successes = np.array(successes)
    succ_mask = successes.astype(bool)

    # Log data ranges.
    print(f'  Action norm totals — min: {action_norms_per_ep.min():.1f}, '
          f'max: {action_norms_per_ep.max():.1f}, mean: {action_norms_per_ep.mean():.1f}')
    print(f'  Episode lengths    — min: {lengths.min():.0f}, '
          f'max: {lengths.max():.0f}, mean: {lengths.mean():.0f}')
    print(f'  Dist-to-goal sums  — min: {dist_sums_per_ep.min():.1f}, '
          f'max: {dist_sums_per_ep.max():.1f}, mean: {dist_sums_per_ep.mean():.1f}')
    print(f'  Successful episodes: {succ_mask.sum()} / {len(succ_mask)}')

    # --- Normalize raw signals to unit variance ---
    # This makes the grid uniform in "effect space" so energy and time
    # axes have comparable scales regardless of raw magnitude differences.
    norm_std = np.std(action_norms_per_ep[succ_mask]) if succ_mask.sum() > 1 else np.std(action_norms_per_ep)
    len_std = np.std(lengths[succ_mask]) if succ_mask.sum() > 1 else np.std(lengths)
    norm_std = max(norm_std, 1e-9)
    len_std = max(len_std, 1e-9)

    normed_energy = action_norms_per_ep / norm_std  # unit-variance
    normed_time = lengths / len_std

    print(f'  Normalization — energy_std: {norm_std:.2f}, length_std: {len_std:.2f}')

    # --- Estimate distance budget ---
    # We'll reserve headroom for distance in Stage 1. Estimate the ratio of
    # median per-step distance to median per-step (energy+time) so we know
    # roughly how much penalty budget distance will need at target_ratio=1.0.
    # With target_ratio=1.0, distance penalty ≈ energy+time penalty in total,
    # so we need to reserve ~50% of the total penalty budget for distance.
    # Use a conservative 1/3 reservation (distance gets 1/3, energy+time get 2/3).
    dist_budget_fraction = 1.0 / 3.0

    # --- Stage 1: Log-spaced 2D sweep over ew_norm × tw_norm ---
    # Weights are in normalized space; real weights = w_norm / std.
    # At max normalized weight, median penalty ≈ goal_reward.
    succ_normed_energy = normed_energy[succ_mask] if succ_mask.any() else normed_energy
    succ_normed_time = normed_time[succ_mask] if succ_mask.any() else normed_time

    n_grid = 40  # 40×40 = 1600 combos (fast, much finer than old 20×20 slice)
    ew_norm_max = goal_reward / (np.median(succ_normed_energy) + 1e-9)
    tw_norm_max = goal_reward / (np.median(succ_normed_time) + 1e-9)

    # Log-spaced from a small fraction to max, plus zero.
    ew_norm_min = ew_norm_max * 1e-3
    tw_norm_min = tw_norm_max * 1e-3
    ew_norm_vals = np.concatenate([[0.0], np.geomspace(ew_norm_min, ew_norm_max, n_grid - 1)])
    tw_norm_vals = np.concatenate([[0.0], np.geomspace(tw_norm_min, tw_norm_max, n_grid - 1)])

    best_combo = None
    best_spread = -1.0

    g = successes.astype(float) * goal_reward  # precompute goal vector

    # Penalty fraction targets for energy+time only (leaving room for distance).
    et_frac_bands = [
        (0.10 * (1 - dist_budget_fraction), 0.30 * (1 - dist_budget_fraction)),
        (0.05 * (1 - dist_budget_fraction), 0.50 * (1 - dist_budget_fraction)),
        (0.0, 1.0),
    ]
    for lo, hi in et_frac_bands:
        for ew_n in ew_norm_vals:
            for tw_n in tw_norm_vals:
                if ew_n == 0 and tw_n == 0:
                    continue

                total = g - ew_n * normed_energy - tw_n * normed_time

                if not succ_mask.any():
                    continue

                succ_total = total[succ_mask]

                # Constraint 1: all successful eps positive reward (with headroom
                # for distance — require at least dist_budget_fraction of
                # goal_reward remaining).
                min_headroom = goal_reward * dist_budget_fraction
                if np.any(succ_total <= min_headroom):
                    continue

                # Constraint 2: median ew+tw penalty fraction in [lo, hi].
                succ_penalties = goal_reward - succ_total
                med_frac = float(np.median(succ_penalties)) / goal_reward
                if not (lo <= med_frac <= hi):
                    continue

                spread = float(np.std(succ_total))
                if spread > best_spread:
                    best_spread = spread
                    best_combo = (ew_n, tw_n, med_frac, spread)

        if best_combo is not None:
            print(f'  Stage 1 selection: ew+tw penalty frac [{lo:.0%}, {hi:.0%}] '
                  f'(reserving {dist_budget_fraction:.0%} for distance), '
                  f'best success_std={best_spread:.4f}')
            break

    if best_combo is None:
        # Unconstrained fallback: maximize spread.
        print('  Stage 1: unconstrained fallback (no combo met criteria)')
        for ew_n in ew_norm_vals:
            for tw_n in tw_norm_vals:
                if ew_n == 0 and tw_n == 0:
                    continue
                total = g - ew_n * normed_energy - tw_n * normed_time
                if succ_mask.any():
                    spread = float(np.std(total[succ_mask]))
                    if spread > best_spread:
                        best_spread = spread
                        succ_total = total[succ_mask]
                        med_frac = float(np.median(goal_reward - succ_total)) / goal_reward
                        best_combo = (ew_n, tw_n, med_frac, spread)

    # --- Refine: fine grid around best combo ---
    coarse_ew_n, coarse_tw_n = best_combo[0], best_combo[1]
    refine_n = 20
    refine_ew = np.linspace(max(coarse_ew_n * 0.5, 0), coarse_ew_n * 1.5, refine_n)
    refine_tw = np.linspace(max(coarse_tw_n * 0.5, 0), coarse_tw_n * 1.5, refine_n)

    for ew_n in refine_ew:
        for tw_n in refine_tw:
            if ew_n == 0 and tw_n == 0:
                continue
            total = g - ew_n * normed_energy - tw_n * normed_time
            if not succ_mask.any():
                continue
            succ_total = total[succ_mask]
            if np.any(succ_total <= 0):
                continue
            succ_penalties = goal_reward - succ_total
            med_frac = float(np.median(succ_penalties)) / goal_reward
            spread = float(np.std(succ_total))
            if spread > best_spread:
                best_spread = spread
                best_combo = (ew_n, tw_n, med_frac, spread)

    best_ew_n, best_tw_n = best_combo[0], best_combo[1]

    # Convert normalized weights back to real weights.
    best_ew = best_ew_n / norm_std
    best_tw = best_tw_n / len_std

    print(f'  Stage 1 result: ew={best_ew:.6f}, tw={best_tw:.6f} '
          f'(normed: ew_n={best_ew_n:.4f}, tw_n={best_tw_n:.4f})')

    # --- Stage 2: Set distance_weight from per-step signal ratio ---
    best_dw = _compute_distance_weight(
        data, episodes, succ_mask, action_norms_per_ep, lengths,
        energy_weight=best_ew, time_weight=best_tw, target_ratio=1.0)

    # Verify adding distance doesn't break the positivity constraint.
    # Use the reserved budget: distance penalty should use at most
    # dist_budget_fraction of goal_reward for the *median* successful episode.
    # We allow some worst-case episodes to go slightly negative if needed —
    # the dense signal value outweighs perfect positivity for outlier episodes.
    if succ_mask.any() and best_dw > 0:
        median_dist_sum = np.median(dist_sums_per_ep[succ_mask])
        budget = goal_reward * dist_budget_fraction
        max_dw_from_budget = budget / (median_dist_sum + 1e-12)

        if best_dw > max_dw_from_budget:
            print(f'  Stage 2: capped dw from {best_dw:.6f} to {max_dw_from_budget:.6f} '
                  f'(median distance budget = {dist_budget_fraction:.0%} of goal_reward)')
            best_dw = max_dw_from_budget

        # Hard check: at least 90% of successful eps must stay positive.
        total_with_dist = (g - best_ew * action_norms_per_ep
                           - best_tw * lengths
                           - best_dw * dist_sums_per_ep)
        succ_positive_frac = np.mean(total_with_dist[succ_mask] > 0)
        if succ_positive_frac < 0.90:
            # Scale down to hit 90% threshold using the 90th percentile dist sum.
            p90_dist = np.percentile(dist_sums_per_ep[succ_mask], 90)
            headroom = (goal_reward - best_ew * action_norms_per_ep[succ_mask]
                        - best_tw * lengths[succ_mask])
            p90_headroom = np.percentile(headroom, 10)  # 10th percentile = tightest
            safe_dw = max(p90_headroom / (p90_dist + 1e-12) * 0.95, 0.0)
            print(f'  Stage 2: further scaled dw to {safe_dw:.6f} '
                  f'(90% positivity constraint, was {succ_positive_frac:.0%})')
            best_dw = safe_dw

    print(f'  Stage 2 result: dw={best_dw:.6f} (per-step signal targeting '
          f'1.0x energy+time magnitude)')

    return dict(
        ew_vals=ew_norm_vals / norm_std, tw_vals=tw_norm_vals / len_std,
        dw_vals=np.array([best_dw]),
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
    n_succ = (ep_goal > 0).sum()
    if succ_negative > 0 and n_succ > 0:
        neg_pct = 100 * succ_negative / n_succ
        if neg_pct > 15:
            print(f'  [!] {succ_negative} ({neg_pct:.0f}%) successful episodes have negative total reward — penalties may be too high.')
        else:
            print(f'  [i] {succ_negative} ({neg_pct:.0f}%) successful episodes have negative total reward '
                  f'(expected for long/inefficient trajectories with dense distance signal).')
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
