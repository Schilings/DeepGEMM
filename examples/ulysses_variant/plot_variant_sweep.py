"""Generate curve charts from Ulysses variant sweep data.

Reads sweep_throughput.json and produces:
- fig_var1: Throughput vs sequence length (serial vs fused_var)
- fig_var2: Peak memory vs sequence length (serial vs fused_var)
- fig_var3: Throughput-memory tradeoff scatter
- fig_var4: BWD time breakdown vs sequence length
- fig_var5: Speedup/slowdown ratio vs sequence length
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, 'sweep_throughput.json')
OUT_DIR = os.path.join(HERE, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

with open(JSON_PATH) as f:
    data = json.load(f)

C_SERIAL = '#4A90D9'
C_VAR = '#E85D3A'
C_MEM_SAVE = '#2ECC71'
C_SLOWDOWN = '#E74C3C'

seqs = [s // 1024 for s in data['seq_lens']]  # K
serial = data['data']['serial']
fused_var = data['data']['fused_var']

# Filter None
valid = [i for i in range(len(seqs)) if serial['throughput'][i] is not None]
seqs = [seqs[i] for i in valid]
s_tps = [serial['throughput'][i] for i in valid]
v_tps = [fused_var['throughput'][i] for i in valid]
s_mem = [serial['peak_mb'][i] for i in valid]
v_mem = [fused_var['peak_mb'][i] for i in valid]
s_fwd = [serial['fwd'][i] for i in valid]
v_fwd = [fused_var['fwd'][i] for i in valid]
s_bwd = [serial['bwd'][i] for i in valid]
v_bwd = [fused_var['bwd'][i] for i in valid]
s_wall = [serial['wall'][i] for i in valid]
v_wall = [fused_var['wall'][i] for i in valid]

# ----------------------------------------------------------------------------
# Fig var1: Throughput vs sequence length
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(seqs, s_tps, 'o-', color=C_SERIAL, linewidth=2.5, markersize=9, label='serial (baseline)')
ax.plot(seqs, v_tps, 's-', color=C_VAR, linewidth=2.5, markersize=9, label='fused_var (Wo sharded)')
for x, y in zip(seqs, s_tps):
    ax.annotate(f'{y/1000:.1f}K', (x, y), textcoords='offset points', xytext=(0, 10),
                ha='center', fontsize=9, color=C_SERIAL)
for x, y in zip(seqs, v_tps):
    ax.annotate(f'{y/1000:.1f}K', (x, y), textcoords='offset points', xytext=(0, -15),
                ha='center', fontsize=9, color=C_VAR)
ax.set_xlabel('Sequence Length (K tokens)', fontsize=12)
ax.set_ylabel('Throughput (tokens/s)', fontsize=12)
ax.set_title('Ulysses Variant: Throughput vs Sequence Length\n'
             '(B300 x8, 40 layers, 14.056B params, official weights, FSDP2, DDP)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.set_xticks(seqs)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1000:.0f}K'))
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_var1_throughput.png'), dpi=150)
plt.close()
print('Saved fig_var1_throughput.png')

# ----------------------------------------------------------------------------
# Fig var2: Peak memory vs sequence length
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(seqs, [m/1024 for m in s_mem], 'o-', color=C_SERIAL, linewidth=2.5, markersize=9, label='serial')
ax.plot(seqs, [m/1024 for m in v_mem], 's-', color=C_VAR, linewidth=2.5, markersize=9, label='fused_var')
for x, y in zip(seqs, s_mem):
    ax.annotate(f'{y/1024:.1f}G', (x, y/1024), textcoords='offset points', xytext=(0, 10),
                ha='center', fontsize=9, color=C_SERIAL)
for x, y in zip(seqs, v_mem):
    ax.annotate(f'{y/1024:.1f}G', (x, y/1024), textcoords='offset points', xytext=(0, -15),
                ha='center', fontsize=9, color=C_VAR)
ax.set_xlabel('Sequence Length (K tokens)', fontsize=12)
ax.set_ylabel('Peak Memory (GB)', fontsize=12)
ax.set_title('Ulysses Variant: Peak Memory vs Sequence Length\n'
             '(Wo sharding saves ~10.5GB across all lengths)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.set_xticks(seqs)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_var2_memory.png'), dpi=150)
plt.close()
print('Saved fig_var2_memory.png')

# ----------------------------------------------------------------------------
# Fig var3: Throughput-memory tradeoff scatter
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
for i, s in enumerate(seqs):
    ax.scatter(s_mem[i]/1024, s_tps[i]/1000, s=150, color=C_SERIAL, zorder=5, edgecolors='white', linewidth=1.5)
    ax.scatter(v_mem[i]/1024, v_tps[i]/1000, s=150, color=C_VAR, zorder=5, edgecolors='white', linewidth=1.5)
    # Draw arrow from serial to variant
    ax.annotate('', xy=(v_mem[i]/1024, v_tps[i]/1000),
                xytext=(s_mem[i]/1024, s_tps[i]/1000),
                arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1.5))
    # Label
    ax.annotate(f'{s}K', (s_mem[i]/1024, s_tps[i]/1000),
                textcoords='offset points', xytext=(-20, 10), fontsize=9, fontweight='bold')
# Legend
from matplotlib.lines import Line2D
legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=C_SERIAL, markersize=12, label='serial'),
          Line2D([0], [0], marker='s', color='w', markerfacecolor=C_VAR, markersize=12, label='fused_var'),
          Line2D([0], [0], color='#7F8C8D', linewidth=1.5, label='tradeoff direction')]
ax.legend(handles=legend, fontsize=11, loc='upper left')
ax.set_xlabel('Peak Memory (GB)', fontsize=12)
ax.set_ylabel('Throughput (K tokens/s)', fontsize=12)
ax.set_title('Throughput-Memory Tradeoff: serial -> fused_var\n'
             '(left = less memory, up = more throughput, ideal = up-left)',
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_var3_tradeoff.png'), dpi=150)
plt.close()
print('Saved fig_var3_tradeoff.png')

# ----------------------------------------------------------------------------
# Fig var4: FWD/BWD breakdown vs sequence length
# ----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# FWD
ax1.plot(seqs, s_fwd, 'o-', color=C_SERIAL, linewidth=2, markersize=8, label='serial')
ax1.plot(seqs, v_fwd, 's-', color=C_VAR, linewidth=2, markersize=8, label='fused_var')
ax1.set_xlabel('Sequence Length (K tokens)', fontsize=12)
ax1.set_ylabel('Forward Time (ms)', fontsize=12)
ax1.set_title('Forward Time', fontsize=12, fontweight='bold')
ax1.legend(fontsize=11)
ax1.set_xticks(seqs)
ax1.grid(alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# BWD
ax2.plot(seqs, s_bwd, 'o-', color=C_SERIAL, linewidth=2, markersize=8, label='serial')
ax2.plot(seqs, v_bwd, 's-', color=C_VAR, linewidth=2, markersize=8, label='fused_var')
ax2.set_xlabel('Sequence Length (K tokens)', fontsize=12)
ax2.set_ylabel('Backward Time (ms)', fontsize=12)
ax2.set_title('Backward Time (incl. DDP overlap)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=11)
ax2.set_xticks(seqs)
ax2.grid(alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle('Ulysses Variant: FWD/BWD Breakdown vs Sequence Length',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_var4_breakdown.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig_var4_breakdown.png')

# ----------------------------------------------------------------------------
# Fig var5: Speedup/slowdown + memory savings (dual axis)
# ----------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 5.5))

# Throughput ratio
throughput_ratio = [v / s for v, s in zip(v_tps, s_tps)]
color_ratio = [C_MEM_SAVE if r >= 1.0 else C_SLOWDOWN for r in throughput_ratio]
bars = ax1.bar([str(x) for x in seqs], throughput_ratio, color=color_ratio,
               edgecolor='white', linewidth=0.8, width=0.5, alpha=0.8)
ax1.axhline(y=1.0, color='#7F8C8D', linestyle='--', linewidth=1, alpha=0.7)
for bar, r in zip(bars, throughput_ratio):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{r:.3f}x', ha='center', fontsize=10, fontweight='bold')
ax1.set_xlabel('Sequence Length (K tokens)', fontsize=12)
ax1.set_ylabel('Throughput Ratio (fused_var / serial)', fontsize=12, color='#2C3E50')
ax1.set_ylim(min(throughput_ratio) * 0.95, max(throughput_ratio) * 1.08)
ax1.tick_params(axis='y', labelcolor='#2C3E50')

# Memory savings on right axis
ax2 = ax1.twinx()
mem_savings = [(s - v) / s * 100 for s, v in zip(s_mem, v_mem)]
ax2.plot([str(x) for x in seqs], mem_savings, 'D--', color='#8E44AD',
         linewidth=2, markersize=8, label='Memory savings %')
for i, ms in enumerate(mem_savings):
    ax2.annotate(f'{ms:.1f}%', (i, ms), textcoords='offset points',
                xytext=(0, 10), ha='center', fontsize=9, color='#8E44AD', fontweight='bold')
ax2.set_ylabel('Memory Savings (%)', fontsize=12, color='#8E44AD')
ax2.tick_params(axis='y', labelcolor='#8E44AD')
ax2.set_ylim(0, max(mem_savings) * 1.3)

ax1.set_title('Ulysses Variant: Throughput Ratio & Memory Savings vs Sequence Length\n'
              '(green=faster, red=slower; purple dashed=memory saved)',
              fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
fig.legend(labels=['Throughput ratio', 'Memory savings %'], loc='upper right',
           fontsize=10, bbox_to_anchor=(0.9, 0.85))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_var5_ratio_memory.png'), dpi=150)
plt.close()
print('Saved fig_var5_ratio_memory.png')

# ----------------------------------------------------------------------------
# Print summary
# ----------------------------------------------------------------------------
print('\n=== Summary ===')
print(f'{"Seq":>6} {"serial tok/s":>14} {"var tok/s":>14} {"ratio":>8} '
      f'{"serial GB":>10} {"var GB":>10} {"saved":>8}')
for i in range(len(seqs)):
    print(f'{seqs[i]:>4}K {s_tps[i]:>13.0f} {v_tps[i]:>13.0f} '
          f'{v_tps[i]/s_tps[i]:>7.3f}x {s_mem[i]/1024:>9.1f}G '
          f'{v_mem[i]/1024:>9.1f}G {(s_mem[i]-v_mem[i])/s_mem[i]*100:>6.1f}%')

print(f'\nAll variant figures saved to {OUT_DIR}/')
