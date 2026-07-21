"""Generate benchmark charts from Wan2.1 14B Dynamic SP×DP results.

Usage:
    python examples/dynamic_ulysses/plot_bench.py

Outputs PNG files in examples/dynamic_ulysses/figures/
"""
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ----------------------------------------------------------------------------
# Experiment data (from bench_wan21_14b.py, B300 x8, 40 layers, official 14B)
# ----------------------------------------------------------------------------
scenarios = [
    'all_short\n2K×8',
    'bimodal\n2×32K+6×2K',
    'uniform\n8K×8',
    'one_long_tail\n1×32K+7×2K',
    'uniform\n32K×2',
    'mixed\nvaried',
]
scenarios_short = ['all_short', 'bimodal', 'uniform_8K', 'one_long', 'uniform_32K', 'mixed']

# tokens/s
static_tps = [6176, 20667, 22528, 14579, 35996, 21970]
dynamic_tps = [41235, 37206, 46916, 22476, 38706, 19447]

# wall-clock ms
static_ms = [2652.9, 3765.5, 2909.1, 3230.9, 1820.6, 3542.3]
dynamic_ms = [397.3, 2091.7, 1396.9, 2095.8, 1693.2, 4001.8]

speedups = [s / d for s, d in zip(static_ms, dynamic_ms)]

# Dynamic SP schedule: {sp_size: count}
sp_schedules = [
    {1: 8},       # all_short
    {4: 2, 1: 6}, # bimodal
    {2: 8},       # uniform_8K
    {4: 1, 1: 7}, # one_long_tail
    {4: 2},       # uniform_32K
    {4: 2, 2: 4, 1: 2}, # mixed
]

# Sequence lengths per scenario
seq_lengths = {
    'all_short':  [2048]*8,
    'bimodal':    [32768, 32768, 2048, 2048, 2048, 2048, 2048, 2048],
    'uniform_8K': [8192]*8,
    'one_long':   [32768, 2048, 2048, 2048, 2048, 2048, 2048, 2048],
    'uniform_32K':[32768, 32768],
    'mixed':      [32768, 16384, 8192, 8192, 4096, 4096, 2048, 2048],
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette
C_STATIC  = '#4A90D9'   # blue
C_DYNAMIC = '#E85D3A'   # orange-red
C_SPEEDUP_POS = '#2ECC71' # green
C_SPEEDUP_NEG = '#E74C3C' # red
C_SP = {1: '#3498DB', 2: '#9B59B6', 4: '#F39C12', 8: '#E74C3C'}

# ----------------------------------------------------------------------------
# Figure 1: Throughput comparison (tokens/s)
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(scenarios))
w = 0.35
bars1 = ax.bar(x - w/2, static_tps, w, label='Static SP=8 (dp=1)', color=C_STATIC, edgecolor='white', linewidth=0.8)
bars2 = ax.bar(x + w/2, dynamic_tps, w, label='Dynamic SP×DP', color=C_DYNAMIC, edgecolor='white', linewidth=0.8)

# Add value labels
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 500, f'{h/1000:.1f}K',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 500, f'{h/1000:.1f}K',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Throughput (tokens/s)', fontsize=12)
ax.set_title('Wan2.1 T2V-14B Training Throughput: Static SP=8 vs Dynamic SP×DP\n'
             '(B300 ×8, 40 layers, 14.056B params, official weights, FSDP2)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontsize=9)
ax.legend(fontsize=11, loc='upper right')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v/1000:.0f}K'))
ax.set_ylim(0, max(max(static_tps), max(dynamic_tps)) * 1.15)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_throughput.png'), dpi=150)
plt.close()
print(f'Saved fig1_throughput.png')

# ----------------------------------------------------------------------------
# Figure 2: Speedup with SP schedule annotation
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5.5))
colors = [C_SPEEDUP_POS if s >= 1.0 else C_SPEEDUP_NEG for s in speedups]
bars = ax.bar(x, speedups, 0.5, color=colors, edgecolor='white', linewidth=0.8)

# Add speedup value labels
for i, (bar, s) in enumerate(zip(bars, speedups)):
    label = f'{s:.2f}x'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            label, ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add SP schedule annotations below each bar
for i, sched in enumerate(sp_schedules):
    parts = [f'SP={sp}×{cnt}' for sp, cnt in sorted(sched.items())]
    text = ', '.join(parts)
    ax.text(i, -0.12, text, ha='center', va='top', fontsize=8, color='#555',
            transform=ax.get_xaxis_transform())

# Geometric mean line
geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
ax.axhline(y=geo, color='#2C3E50', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(len(scenarios) - 0.5, geo + 0.03, f'Geo mean: {geo:.3f}x',
        fontsize=10, color='#2C3E50', fontweight='bold', ha='right')

ax.axhline(y=1.0, color='#7F8C8D', linestyle='-', linewidth=0.8, alpha=0.5)
ax.set_ylabel('Speedup (Static SP=8 / Dynamic SP×DP)', fontsize=12)
ax.set_title('Dynamic SP×DP Speedup over Static SP=8\n'
             'Green = Dynamic wins, Red = Static wins', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontsize=9)
ax.set_ylim(0, max(speedups) * 1.2)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_speedup.png'), dpi=150)
plt.close()
print(f'Saved fig2_speedup.png')

# ----------------------------------------------------------------------------
# Figure 3: Sequence length → optimal SP size assignment
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

# Map seq_len → assigned SP size (from BalancedDataLoader)
seq_sp_map = {}
for name, seqs in seq_lengths.items():
    for s in seqs:
        if s not in seq_sp_map:
            # Reproduce assign_sp_size logic
            if s <= 2048:
                sp = 1
            elif s <= 8192:
                sp = 2
            elif s <= 32768:
                sp = 4
            else:
                sp = 8
            seq_sp_map[s] = sp

all_seqs = sorted(seq_sp_map.keys())
all_sps = [seq_sp_map[s] for s in all_seqs]

# Staircase plot
ax.step(all_seqs, all_sps, where='mid', linewidth=2.5, color='#2C3E50', alpha=0.8)
for s, sp in zip(all_seqs, all_sps):
    ax.scatter([s], [sp], s=100, color=C_SP[sp], zorder=5, edgecolors='white', linewidth=1.5)
    ax.annotate(f'SP={sp}', (s, sp), textcoords='offset points',
                xytext=(8, 8), fontsize=9, fontweight='bold', color=C_SP[sp])

ax.set_xlabel('Sequence Length (tokens)', fontsize=12)
ax.set_ylabel('Assigned SP Size', fontsize=12)
ax.set_title('BalancedDataLoader: Sequence Length → SP Size Assignment\n'
             '(shorter sequences → smaller SP → more DP copies)', fontsize=13, fontweight='bold')
ax.set_xscale('log', base=2)
ax.set_xticks(all_seqs)
ax.set_xticklabels([f'{s//1024}K' if s >= 1024 else str(s) for s in all_seqs])
ax.set_yticks([1, 2, 4, 8])
ax.set_yticklabels(['SP=1\n(DP=8)', 'SP=2\n(DP=4)', 'SP=4\n(DP=2)', 'SP=8\n(DP=1)'])
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.invert_yaxis()  # SP=1 at top (short seq), SP=8 at bottom (long seq)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_sp_assignment.png'), dpi=150)
plt.close()
print(f'Saved fig3_sp_assignment.png')

# ----------------------------------------------------------------------------
# Figure 4: Wall-clock breakdown — stacked by (sp_size, seq_len) groups
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5.5))

# For each scenario, show the dynamic SP execution timeline (normalized)
# Each (sp_size, seq_len) group is a colored segment
for i, name in enumerate(scenarios_short):
    sched = sp_schedules[i]
    seqs = seq_lengths[name]

    # Group by (sp_size, seq_len)
    groups = {}
    for s in seqs:
        sp = seq_sp_map.get(s, 1)
        key = (sp, s)
        groups[key] = groups.get(key, 0) + 1

    # Estimate relative time per group (proportional to seq_len * count / sp_size)
    total_est = 0
    group_times = []
    for (sp, s), cnt in sorted(groups.items(), key=lambda k: (-k[0][0], -k[0][1])):
        est = s * cnt / sp
        group_times.append((sp, s, cnt, est))
        total_est += est

    # Draw stacked bar
    bottom = 0
    for sp, s, cnt, est in group_times:
        frac = est / total_est if total_est > 0 else 0
        label = f'SP={sp}\n{s//1024}K×{cnt}'
        ax.bar(i, frac, bottom=bottom, color=C_SP[sp], edgecolor='white',
               linewidth=0.8, width=0.6)
        if frac > 0.05:  # Only label if segment is big enough
            ax.text(i, bottom + frac/2, label, ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold')
        bottom += frac

ax.set_xticks(range(len(scenarios)))
ax.set_xticklabels(scenarios, fontsize=9)
ax.set_ylabel('Normalized Execution Time', fontsize=12)
ax.set_title('Dynamic SP×DP Execution Time Breakdown by (SP size, seq_len) Group\n'
             'Each color = different SP size', fontsize=13, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=C_SP[sp], label=f'SP={sp}') for sp in sorted(C_SP.keys())]
ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
ax.set_ylim(0, 1.1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_breakdown.png'), dpi=150)
plt.close()
print(f'Saved fig4_breakdown.png')

# ----------------------------------------------------------------------------
# Figure 5: Summary — throughput + speedup combined
# ----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: throughput
x = np.arange(len(scenarios))
w = 0.35
ax1.bar(x - w/2, [t/1000 for t in static_tps], w, label='Static SP=8', color=C_STATIC, edgecolor='white')
ax1.bar(x + w/2, [t/1000 for t in dynamic_tps], w, label='Dynamic SP×DP', color=C_DYNAMIC, edgecolor='white')
ax1.set_ylabel('Throughput (K tokens/s)', fontsize=11)
ax1.set_title('Throughput', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([s.replace('\n', ' ') for s in scenarios], fontsize=8, rotation=15, ha='right')
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Right: speedup
colors_sp = [C_SPEEDUP_POS if s >= 1.0 else C_SPEEDUP_NEG for s in speedups]
ax2.bar(x, speedups, 0.5, color=colors_sp, edgecolor='white')
for i, s in enumerate(speedups):
    ax2.text(i, s + 0.05, f'{s:.2f}x', ha='center', fontsize=9, fontweight='bold')
ax2.axhline(y=1.0, color='#7F8C8D', linestyle='-', linewidth=0.8, alpha=0.5)
ax2.axhline(y=geo, color='#2C3E50', linestyle='--', linewidth=1.2, alpha=0.7)
ax2.text(len(scenarios)-0.5, geo+0.05, f'Geo: {geo:.3f}x', fontsize=9, ha='right', fontweight='bold')
ax2.set_ylabel('Speedup', fontsize=11)
ax2.set_title('Speedup (Dynamic / Static)', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([s.replace('\n', ' ') for s in scenarios], fontsize=8, rotation=15, ha='right')
ax2.set_ylim(0, max(speedups)*1.2)
ax2.grid(axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle('Wan2.1 T2V-14B Dynamic SP×DP Benchmark Summary (B300 ×8, 40 layers, official weights, FSDP2)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved fig5_summary.png')

print(f'\nAll figures saved to {OUTPUT_DIR}/')
