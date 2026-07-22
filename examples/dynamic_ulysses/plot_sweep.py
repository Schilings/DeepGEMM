"""Generate curve plots from sweep benchmark data.

Reads sweep_results.json and produces 4 curve charts:
1. Speedup vs sequence length
2. Speedup vs number of sequences
3. Throughput vs SP size
4. Speedup vs long:short ratio
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, 'sweep_results.json')
OUT_DIR = os.path.join(HERE, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

with open(JSON_PATH) as f:
    data = json.load(f)

C_STATIC = '#4A90D9'
C_DYN = '#E85D3A'
C_SPEEDUP = '#2ECC71'
C_LINE = '#2C3E50'

# ----------------------------------------------------------------------------
# Fig 6: Speedup vs Sequence Length
# ----------------------------------------------------------------------------
s1 = data['sweep_seq_len']
seq_lens = [s // 1024 for s in s1['seq_lens']]  # K
static_ms = s1['static_sp8']
dyn_ms = s1['dynamic']
speedups = [st / dn for st, dn in zip(static_ms, dyn_ms)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: wall-clock time
ax1.plot(seq_lens, static_ms, 'o-', color=C_STATIC, linewidth=2, markersize=8, label='Static SP=8')
ax1.plot(seq_lens, dyn_ms, 's-', color=C_DYN, linewidth=2, markersize=8, label='Dynamic SP×DP')
ax1.set_xlabel('Sequence Length (K tokens)', fontsize=12)
ax1.set_ylabel('Wall-clock Time (ms)', fontsize=12)
ax1.set_title('Wall-clock Time vs Sequence Length\n(8 sequences, 14B model)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)
ax1.set_xticks(seq_lens)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Right: speedup
ax2.plot(seq_lens, speedups, 'D-', color=C_SPEEDUP, linewidth=2.5, markersize=9)
ax2.axhline(y=1.0, color='#7F8C8D', linestyle='--', linewidth=1, alpha=0.7, label='Break-even (1.0x)')
for x, y in zip(seq_lens, speedups):
    ax2.annotate(f'{y:.2f}x', (x, y), textcoords='offset points',
                xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')
ax2.set_xlabel('Sequence Length (K tokens)', fontsize=12)
ax2.set_ylabel('Speedup (Static / Dynamic)', fontsize=12)
ax2.set_title('Speedup vs Sequence Length\n(shorter sequences benefit more from DP)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)
ax2.set_xticks(seq_lens)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig6_sweep_seq_len.png'), dpi=150)
plt.close()
print('Saved fig6_sweep_seq_len.png')

# ----------------------------------------------------------------------------
# Fig 7: Speedup vs Number of Sequences
# ----------------------------------------------------------------------------
s2 = data['sweep_num_seqs']
counts = s2['counts']
static_ms2 = s2['static_sp8']
dyn_ms2 = s2['dynamic']
speedups2 = [st / dn for st, dn in zip(static_ms2, dyn_ms2)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: wall-clock time
ax1.plot(counts, static_ms2, 'o-', color=C_STATIC, linewidth=2, markersize=8, label='Static SP=8')
ax1.plot(counts, dyn_ms2, 's-', color=C_DYN, linewidth=2, markersize=8, label='Dynamic SP×DP')
ax1.set_xlabel('Number of Sequences', fontsize=12)
ax1.set_ylabel('Wall-clock Time (ms)', fontsize=12)
ax1.set_title('Wall-clock Time vs Number of Sequences\n(8K tokens each, 14B model)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)
ax1.set_xticks(counts)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Right: speedup
ax2.plot(counts, speedups2, 'D-', color=C_SPEEDUP, linewidth=2.5, markersize=9)
ax2.axhline(y=1.0, color='#7F8C8D', linestyle='--', linewidth=1, alpha=0.7, label='Break-even (1.0x)')
for x, y in zip(counts, speedups2):
    ax2.annotate(f'{y:.2f}x', (x, y), textcoords='offset points',
                xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')
ax2.set_xlabel('Number of Sequences', fontsize=12)
ax2.set_ylabel('Speedup (Static / Dynamic)', fontsize=12)
ax2.set_title('Speedup vs Number of Sequences\n(more sequences → more DP parallelism)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)
ax2.set_xticks(counts)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig7_sweep_num_seqs.png'), dpi=150)
plt.close()
print('Saved fig7_sweep_num_seqs.png')

# ----------------------------------------------------------------------------
# Fig 8: Throughput vs SP Size
# ----------------------------------------------------------------------------
s3 = data['sweep_sp_size']
sps = s3['sp_sizes']
times = s3['times']
tps = s3['throughput']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: latency
bars = ax1.bar([f'SP={s}\n(DP={8//s})' for s in sps], times, color=['#3498DB', '#9B59B6', '#F39C12', '#E74C3C'],
               edgecolor='white', linewidth=0.8, width=0.6)
for bar, t in zip(bars, times):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             f'{t:.0f}ms', ha='center', fontsize=10, fontweight='bold')
ax1.set_ylabel('Latency per Sequence (ms)', fontsize=12)
ax1.set_title('Single 8K Sequence Latency vs SP Size\n(14B model)', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Right: throughput
bars2 = ax2.bar([f'SP={s}\n(DP={8//s})' for s in sps], [t/1000 for t in tps],
                color=['#3498DB', '#9B59B6', '#F39C12', '#E74C3C'],
                edgecolor='white', linewidth=0.8, width=0.6)
for bar, t in zip(bars2, tps):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{t/1000:.1f}K', ha='center', fontsize=10, fontweight='bold')
ax2.set_ylabel('Throughput (K tokens/s)', fontsize=12)
ax2.set_title('Single 8K Sequence Throughput vs SP Size\n(larger SP = faster for single seq)', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig8_sweep_sp_size.png'), dpi=150)
plt.close()
print('Saved fig8_sweep_sp_size.png')

# ----------------------------------------------------------------------------
# Fig 9: Speedup vs Long:Short Ratio
# ----------------------------------------------------------------------------
s4 = data['sweep_mixed_ratio']
ratios = s4['ratios']
labels = [f'{l}:{s}' for l, s in ratios]
static_ms4 = s4['static_sp8']
dyn_ms4 = s4['dynamic']
speedups4 = [st / dn for st, dn in zip(static_ms4, dyn_ms4)]
n_long = [l for l, s in ratios]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: wall-clock time
x = np.arange(len(labels))
w = 0.35
ax1.bar(x - w/2, static_ms4, w, color=C_STATIC, label='Static SP=8', edgecolor='white')
ax1.bar(x + w/2, dyn_ms4, w, color=C_DYN, label='Dynamic SP×DP', edgecolor='white')
ax1.set_xlabel('Long:Short Ratio (32K:2K)', fontsize=12)
ax1.set_ylabel('Wall-clock Time (ms)', fontsize=12)
ax1.set_title('Wall-clock Time vs Sequence Composition\n(8 total sequences, 14B model)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Right: speedup curve
ax2.plot(n_long, speedups4, 'D-', color=C_SPEEDUP, linewidth=2.5, markersize=10)
ax2.axhline(y=1.0, color='#7F8C8D', linestyle='--', linewidth=1, alpha=0.7, label='Break-even (1.0x)')
for x_val, y_val in zip(n_long, speedups4):
    ax2.annotate(f'{y_val:.2f}x', (x_val, y_val), textcoords='offset points',
                xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')
ax2.set_xlabel('Number of Long Sequences (32K)', fontsize=12)
ax2.set_ylabel('Speedup (Static / Dynamic)', fontsize=12)
ax2.set_title('Speedup vs Sequence Composition\n(more short sequences → higher speedup)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)
ax2.set_xticks(n_long)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig9_sweep_ratio.png'), dpi=150)
plt.close()
print('Saved fig9_sweep_ratio.png')

print(f'\nAll sweep figures saved to {OUT_DIR}/')
