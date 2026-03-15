"""
Visualize per-episode evaluation returns for all algorithms.
Generates histograms + KDE curves showing the distribution of returns.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load results
with open('/ehome/niti/Results/sb3_results.json') as f:
    sb3 = json.load(f)
with open('/ehome/niti/Results/meanflowql_results.json') as f:
    mfql = json.load(f)

# Merge all algorithms, ordered by avg_return descending
all_results = {**mfql, **sb3}
all_results = dict(sorted(all_results.items(), key=lambda x: x[1]['avg_return'], reverse=True))

n_algos = len(all_results)
colors = plt.cm.Set2(np.linspace(0, 1, n_algos))

# ---------- Figure 1: Individual histograms with KDE overlay ----------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

# Find global return range for consistent x-axis
all_vals = []
for data in all_results.values():
    all_vals.extend(data['all_returns'])
global_min = int(min(all_vals))
global_max = int(max(all_vals))
bins = np.arange(global_min - 0.5, global_max + 1.5, 1)  # integer bins

for idx, (algo, data) in enumerate(all_results.items()):
    ax = axes[idx]
    returns = np.array(data['all_returns'])
    avg = np.mean(returns)
    std = np.std(returns)

    # Histogram
    counts, _, bars = ax.hist(returns, bins=bins, color=colors[idx],
                               edgecolor='white', alpha=0.85, zorder=2)

    # Mean line
    ax.axvline(avg, color='red', linestyle='--', linewidth=2, label=f'Mean={avg:.2f}')
    # Std shading
    ax.axvspan(avg - std, avg + std, alpha=0.12, color='red', zorder=1)

    ax.set_title(f'{algo}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode Return', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_xlim(global_min - 1, global_max + 1)
    ax.set_xticks(range(global_min, global_max + 1))
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Annotate std
    ax.text(0.02, 0.92, f'std={std:.2f}', transform=ax.transAxes,
            fontsize=10, color='gray', verticalalignment='top')

# Hide unused subplot
for j in range(n_algos, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('PointMaze UMaze-v3: Per-Episode Return Distributions (50 episodes each)',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('/ehome/niti/Results/eval_histograms.png', dpi=150, bbox_inches='tight')
print('Saved: eval_histograms.png')

# ---------- Figure 2: Overlaid comparison ----------
fig2, ax2 = plt.subplots(figsize=(12, 6))

bar_width = 0.13
return_vals = np.arange(global_min, global_max + 1)
offsets = np.arange(n_algos) - (n_algos - 1) / 2

for idx, (algo, data) in enumerate(all_results.items()):
    returns = np.array(data['all_returns'])
    # Count occurrences of each return value
    counts = np.array([np.sum(returns == v) for v in return_vals])
    x_pos = return_vals + offsets[idx] * bar_width
    ax2.bar(x_pos, counts, width=bar_width, label=f'{algo} (avg={np.mean(returns):.2f})',
            color=colors[idx], edgecolor='white', alpha=0.9)

ax2.set_xlabel('Episode Return', fontsize=13)
ax2.set_ylabel('Number of Episodes', fontsize=13)
ax2.set_title('PointMaze UMaze-v3: Return Distribution Comparison',
              fontsize=15, fontweight='bold')
ax2.set_xticks(return_vals)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig2.savefig('/ehome/niti/Results/eval_comparison.png', dpi=150, bbox_inches='tight')
print('Saved: eval_comparison.png')

# ---------- Figure 3: Box plot ----------
fig3, ax3 = plt.subplots(figsize=(10, 6))

algo_names = list(all_results.keys())
all_data = [np.array(all_results[a]['all_returns']) for a in algo_names]

bp = ax3.boxplot(all_data, tick_labels=algo_names, patch_artist=True, widths=0.5,
                  showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=7))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax3.set_ylabel('Episode Return', fontsize=13)
ax3.set_title('PointMaze UMaze-v3: Return Distributions',
              fontsize=15, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig3.savefig('/ehome/niti/Results/eval_boxplot.png', dpi=150, bbox_inches='tight')
print('Saved: eval_boxplot.png')

print('\nAll plots saved to /ehome/niti/Results/')
