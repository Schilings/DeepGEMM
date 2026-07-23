"""Generate curve charts from Ulysses variant v2 sweep data.

Reads sweep_throughput.json and produces:
- fig_v2_1: Throughput vs sequence length (serial vs fused_var vs v2)
- fig_v2_2: Peak memory vs sequence length
- fig_v2_3: BWD time comparison
- fig_v2_4: Throughput ratio + memory savings (dual axis)
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
C_V1 = '#E85D3A'
C_V2 = '#9B59B6'
C_MEM = '#2ECC71'

seqs = [s // 1024 for s in data['seq_lens']]
strategies = data['strategies']

# Filter valid entries
valid = [i for i in range(len(seqs))
         if data['data']['serial']['throughput'][i] is not None]
seqs = [seqs[i] for i in valid]

s_data = {k: [data['data']['serial'][k][i] for i in valid] for k in ['throughput', 'peak_mb', 'fwd', 'bwd', 'wall']}
has_v1 = 'fused_var' in data['data']
has_v2 = 'fused_var_v2' in data['data']

if has_v1:
    v1_data = {k: [data['data']['fused_var'][k][i] for i in valid] for k in ['throughput', 'peak_mb', 'fwd', 'bwd', 'wall']}
if has_v2:
    v2_data = {k: [data['data']['fused_var_v2'][k][i] for i in valid] for k in ['throughput', 'peak_mb', 'fwd', 'bwd', 'wall']}

# Fig 1: Throughput
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(seqs, s_data['throughput'], 'o-', color=C_SERIAL, linewidth=2.5, markersize=9, label='serial (baseline)')
if has_v1:
    ax.plot(seqs, v1_data['throughput'], 's-', color=C_V1, linewidth=2.5, markersize=9, label='fused_var (v1: fused AG+GEMM)')
if has_v2:
    ax.plot(seqs, v2_data['throughput'], 'D-', color=C_V2, linewidth=2.5, markersize=9, label='fused_var_v2 (native AG+overlap)')
ax.set_xlabel('Sequence Length (K tokens)', fontsize=12)
ax.set_ylabel('Throughput (tokens/s)', fontsize=12)
ax.set_title('Ulysses Variant v2: Throughput vs Sequence Length\n'
             '(B300 x8, 40 layers, 14B params, DDP overlap)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.set_xticks(seqs)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1000:.0f}K'))
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_v2_1_throughput.png'), dpi=150)
plt.close()
print('Saved fig_v2_1_throughput.png')

# Fig 2: Memory
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(seqs, [m/1024 for m in s_data['peak_mb']], 'o-', color=C_SERIAL, linewidth=2.5, markersize=9, label='serial')
if has_v1:
    ax.plot(seqs, [m/1024 for m in v1_data['peak_mb']], 's-', color=C_V1, linewidth=2.5, markersize=9, label='fused_var')
if has_v2:
    ax.plot(seqs, [m/1024 for m in v2_data['peak_mb']], 'D-', color=C_V2, linewidth=2.5, markersize=9, label='fused_var_v2')
ax.set_xlabel('Sequence Length (K tokens)', fontsize=12)
ax.set_ylabel('Peak Memory (GB)', fontsize=12)
ax.set_title('Peak Memory vs Sequence Length\n(v2 same as v1 — Wo sharding saves ~10.5GB)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xticks(seqs)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_v2_2_memory.png'), dpi=150)
plt.close()
print('Saved fig_v2_2_memory.png')

# Fig 3: BWD time
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(seqs, s_data['bwd'], 'o-', color=C_SERIAL, linewidth=2.5, markersize=9, label='serial')
if has_v1:
    ax.plot(seqs, v1_data['bwd'], 's-', color=C_V1, linewidth=2.5, markersize=9, label='fused_var (v1)')
if has_v2:
    ax.plot(seqs, v2_data['bwd'], 'D-', color=C_V2, linewidth=2.5, markersize=9, label='fused_var_v2 (native AG+overlap)')
ax.set_xlabel('Sequence Length (K tokens)', fontsize=12)
ax.set_ylabel('Backward Time (ms, incl. DDP overlap)', fontsize=12)
ax.set_title('Backward Time Comparison\n(v2 should narrow the BWD gap vs serial)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xticks(seqs)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_v2_3_bwd.png'), dpi=150)
plt.close()
print('Saved fig_v2_3_bwd.png')

# Fig 4: Throughput ratio + memory savings
fig, ax1 = plt.subplots(figsize=(10, 5.5))
labels = [str(s) for s in seqs]
if has_v1:
    ratio_v1 = [v / s for v, s in zip(v1_data['throughput'], s_data['throughput'])]
    bars1 = ax1.bar([x + ' v1' for x in labels], ratio_v1, color=C_V1, alpha=0.7, width=0.3, label='v1/serial')
if has_v2:
    ratio_v2 = [v / s for v, s in zip(v2_data['throughput'], s_data['throughput'])]
    bars2 = ax1.bar([x + ' v2' for x in labels], ratio_v2, color=C_V2, alpha=0.7, width=0.3, label='v2/serial')
ax1.axhline(y=1.0, color='#7F8C8D', linestyle='--', linewidth=1, alpha=0.7)
ax1.set_ylabel('Throughput Ratio', fontsize=12)
ax1.set_title('Throughput Ratio: v1/serial vs v2/serial', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_v2_4_ratio.png'), dpi=150)
plt.close()
print('Saved fig_v2_4_ratio.png')

# Summary
print('\n=== Summary ===')
header = f'{"Seq":>6} {"serial tps":>12}'
if has_v1: header += f' {"v1 tps":>12} {"v1/s":>7}'
if has_v2: header += f' {"v2 tps":>12} {"v2/s":>7}'
print(header)
for i in range(len(seqs)):
    row = f'{seqs[i]:>4}K {s_data["throughput"][i]:>11.0f}'
    if has_v1:
        row += f' {v1_data["throughput"][i]:>11.0f} {v1_data["throughput"][i]/s_data["throughput"][i]:>6.3f}x'
    if has_v2:
        row += f' {v2_data["throughput"][i]:>11.0f} {v2_data["throughput"][i]/s_data["throughput"][i]:>6.3f}x'
    print(row)
print(f'\nAll v2 figures saved to {OUT_DIR}/')
