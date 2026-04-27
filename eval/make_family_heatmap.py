"""Per-family score heatmap (family × solver)."""
import json, os, sys
import numpy as np
sys.path.insert(0, '/nas/ckgfs/jaunts/jahin/RELAY/eval')
from plotting import setup_style, save_fig, PALETTE
import matplotlib.pyplot as plt

setup_style()

BASE = '/nas/ckgfs/jaunts/jahin/RELAY'
SOLVERS = ['nuwls','bandhs','satlike','spb','foursat']

def family(name):
    s = name.split('__', 1)[0]
    # strip extensions
    for ext in ('.wcnf', '.cnf'):
        if s.endswith(ext): s = s[:-len(ext)]
    # family = first non-numeric token
    import re
    m = re.match(r'([a-zA-Z][a-zA-Z_\-]+)', s)
    return m.group(1).rstrip('_-').lower() if m else s.lower()

def load_flat(solver, timeout):
    key = str(timeout)
    sub = '' if timeout == 60 else '_300s'
    p = f'{BASE}/results/baselines{sub}/{solver}_results_raw.json'
    if not os.path.exists(p): return {}
    out = {}
    for t, yd in json.load(open(p)).items():
        if t != key: continue
        for y, cd in yd.items():
            for c, inst in cd.items():
                for n, s in inst.items():
                    out[(y, c, n)] = min(s) if isinstance(s, list) and s else float('inf')
    return out

bk = json.load(open(f'{BASE}/results/mse_official_best_known.json'))
def score(c, b):
    if b is None or b < 0 or c == float('inf'): return 0.0
    return max(0, min(1, (1+b)/(1+c)))

# Aggregate per-family per-solver at 60s
data = {s: load_flat(s, 60) for s in SOLVERS}
all_keys = set()
for d in data.values(): all_keys.update(d.keys())

fam_scores = {}  # {fam: {solver: [scores]}}
fam_counts = {}
for k in all_keys:
    y, c, n = k
    fam = family(n)
    if n not in bk or bk[n] < 0: continue
    for s in SOLVERS:
        fam_scores.setdefault(fam, {}).setdefault(s, []).append(score(data[s].get(k, float('inf')), bk[n]))
    fam_scores[fam].setdefault('VBS', []).append(score(min(data[sv].get(k, float('inf')) for sv in SOLVERS), bk[n]))
    fam_counts[fam] = fam_counts.get(fam, 0) + 1

# Pick families with >= 10 instances for heatmap
big_fams = [f for f, n in fam_counts.items() if n >= 10]
big_fams.sort(key=lambda f: -fam_counts[f])
top_fams = big_fams[:20]

cols = SOLVERS + ['VBS']
mat = np.zeros((len(top_fams), len(cols)))
for i, fam in enumerate(top_fams):
    for j, s in enumerate(cols):
        scores = fam_scores[fam].get(s, [])
        mat[i, j] = np.mean(scores) if scores else 0

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(len(cols)))
ax.set_xticklabels([c.upper() if c == 'VBS' else c for c in cols], rotation=30, ha='right')
ax.set_yticks(range(len(top_fams)))
ax.set_yticklabels([f'{f} ({fam_counts[f]})' for f in top_fams], fontsize=8)
ax.set_xlabel('Solver')
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                color='white' if mat[i,j] < 0.5 else 'black', fontsize=7)
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Avg Score', fontsize=9)
plt.tight_layout()
fig_dir = f'{BASE}/figures'
save_fig(fig, os.path.join(fig_dir, 'fig_family_heatmap.pdf'))
print(f'Wrote family heatmap: {len(top_fams)} families')
