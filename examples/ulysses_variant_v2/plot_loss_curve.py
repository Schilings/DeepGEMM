"""Plot loss curve comparison: serial vs fused_var_v2."""
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(HERE, 'correctness_results.json')) as f:
    data = json.load(f)

steps = data['steps']
serial = data['serial']
v2 = data['v2']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(steps, serial, 'o-', color='#4A90D9', linewidth=2.5, markersize=7, label='serial (baseline)')
ax1.plot(steps, v2, 's-', color='#9B59B6', linewidth=2.5, markersize=7, label='fused_var_v2 (native AG+overlap)')
ax1.set_xlabel('Training Step', fontsize=12)
ax1.set_ylabel('Loss (MSE)', fontsize=12)
ax1.set_title('Loss Curve: serial vs fused_var_v2\n(4 layers, 8K seq, SP=8, lr=1e-4)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

diffs = [abs(s - v) / (abs(s) + 1e-8) * 100 for s, v in zip(serial, v2)]
ax2.plot(steps, diffs, 'D-', color='#2ECC71', linewidth=2.5, markersize=8)
ax2.axhline(y=5.0, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.7, label='5% threshold')
ax2.fill_between(steps, 0, 5, alpha=0.1, color='#2ECC71')
ax2.set_xlabel('Training Step', fontsize=12)
ax2.set_ylabel('Relative Loss Difference (%)', fontsize=12)
ax2.set_title(f'Loss Difference: |serial - v2| / serial\n(max = {max(diffs):.3f}%)',
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)
ax2.set_ylim(0, max(diffs) * 3 + 0.1)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_v2_loss_curve.png'), dpi=150)
plt.close()
print('Saved fig_v2_loss_curve.png')
