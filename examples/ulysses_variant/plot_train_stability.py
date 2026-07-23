"""Plot 1000-step training stability: loss curve + throughput."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(HERE, 'training_stability.json')) as f:
    data = json.load(f)

C_SERIAL = '#4A90D9'
C_VAR = '#E85D3A'
C_DIFF = '#2ECC71'

s_loss = data['serial']['losses']
v_loss = data['fused_var']['losses']
steps = list(range(len(s_loss)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: loss curves
ax1.plot(steps, s_loss, '-', color=C_SERIAL, linewidth=2, label='serial (baseline)', alpha=0.9)
ax1.plot(steps, v_loss, '-', color=C_VAR, linewidth=2, label='fused_var (Wo sharded)', alpha=0.9)
ax1.set_xlabel('Training Step', fontsize=12)
ax1.set_ylabel('Loss (MSE)', fontsize=12)
ax1.set_title('1000-Step Training Loss: serial vs fused_var\n'
              '(40 layers, 8K seq, SP=8, lr=1e-4, official 14B weights)',
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.set_yscale('log')
ax1.grid(alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Right: relative difference
diffs = [abs(s - v) / (abs(s) + 1e-8) * 100 for s, v in zip(s_loss, v_loss)]
# Smooth with moving average
window = 20
smooth_diffs = []
for i in range(len(diffs)):
    start = max(0, i - window // 2)
    end = min(len(diffs), i + window // 2 + 1)
    smooth_diffs.append(sum(diffs[start:end]) / (end - start))

ax2.plot(steps, diffs, '-', color=C_DIFF, linewidth=0.8, alpha=0.3, label='Raw diff')
ax2.plot(steps, smooth_diffs, '-', color=C_DIFF, linewidth=2.5, label=f'Smoothed (window={window})')
ax2.axhline(y=5.0, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.7, label='5% threshold')
ax2.fill_between(steps, 0, 5, alpha=0.08, color='#2ECC71')
ax2.set_xlabel('Training Step', fontsize=12)
ax2.set_ylabel('Relative Loss Difference (%)', fontsize=12)
ax2.set_title('Loss Difference: |serial - var| / serial\n'
              f'(final: {diffs[-1]:.2f}%, max: {max(diffs):.1f}%)',
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.set_yscale('log')
ax2.grid(alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_var7_train_1000.png'), dpi=150)
plt.close()
print('Saved fig_var7_train_1000.png')

# Summary
print(f'\n=== 1000-Step Training Summary ===')
print(f'serial:     final_loss={s_loss[-1]:.6f}  avg_tps={data["serial"]["avg_tps"]:.0f}  '
      f'peak={data["serial"]["peak_mb"]/1024:.1f}GB')
print(f'fused_var:  final_loss={v_loss[-1]:.6f}  avg_tps={data["fused_var"]["avg_tps"]:.0f}  '
      f'peak={data["fused_var"]["peak_mb"]/1024:.1f}GB')
print(f'Max loss diff: {max(diffs):.2f}%')
print(f'Final loss diff: {diffs[-1]:.2f}%')
print(f'Throughput ratio: {data["fused_var"]["avg_tps"]/data["serial"]["avg_tps"]:.3f}x')
print(f'Memory savings: {(data["serial"]["peak_mb"]-data["fused_var"]["peak_mb"])/data["serial"]["peak_mb"]*100:.1f}%')
