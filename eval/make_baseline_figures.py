"""Quick NIPS-style figures from the baseline data we have now."""
import json, os, sys
from collections import defaultdict

sys.path.insert(0, '/nas/ckgfs/jaunts/jahin/RELAY')
sys.path.insert(0, '/nas/ckgfs/jaunts/jahin/RELAY/eval')
from plotting import setup_style, new_fig, place_legend, save_fig, PALETTE

setup_style()
import numpy as np

BASE = '/nas/ckgfs/jaunts/jahin/RELAY'
SOLVERS = ['nuwls','bandhs','satlike','spb','foursat']

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

def score(cost, b):
    if b is None or b < 0 or cost == float('inf'): return 0.0
    return max(0, min(1, (1+b)/(1+cost)))

years = ['2020','2021','2022','2023','2024']
cats = ['ms','wms']

# Grouped bars: (year, cat) x solver, one subplot per timeout
fig, (ax1, ax2) = __import__('matplotlib.pyplot', fromlist=['pyplot']).subplots(1, 2, figsize=(12, 4.2), sharey=True)
colors = {'nuwls': PALETTE['blue'], 'satlike': PALETTE['orange'], 'bandhs': PALETTE['green'],
          'spb': PALETTE['red'], 'foursat': PALETTE['purple'], 'VBS': PALETTE['black']}

for ax, timeout in [(ax1, 60), (ax2, 300)]:
    data = {s: load_flat(s, timeout) for s in SOLVERS}
    labels, pos = [], []
    group_width = 0.88
    bar_w = group_width / (len(SOLVERS) + 1)
    for i, y in enumerate(years):
        for j, c in enumerate(cats):
            labels.append(f'{c}-{y[2:]}')
            pos.append(i*2 + j)
    for idx, s in enumerate(SOLVERS + ['VBS']):
        vals = []
        for p in pos:
            y = years[p//2]
            c = cats[p%2]
            insts = [k for k in data[SOLVERS[0]] if k[0]==y and k[1]==c]
            scores = []
            for k in insts:
                bkc = bk.get(k[2])
                if bkc is None or bkc < 0: continue
                if s == 'VBS':
                    cst = min(data[sv].get(k, float('inf')) for sv in SOLVERS)
                else:
                    cst = data[s].get(k, float('inf'))
                scores.append(score(cst, bkc))
            vals.append(np.mean(scores) if scores else 0.0)
        offs = (idx - (len(SOLVERS))/2) * bar_w
        ax.bar([p + offs for p in pos], vals, width=bar_w,
               label=s.upper() if s=='VBS' else s,
               color=colors[s], edgecolor='black', linewidth=0.6)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Avg Incomplete Score' if timeout == 60 else '')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.text(0.02, 0.95, f'{timeout}s', transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='top')

# Legend outside
import matplotlib.pyplot as plt
handles, labs = ax1.get_legend_handles_labels()
fig.legend(handles, labs, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_dir = f'{BASE}/figures'
os.makedirs(out_dir, exist_ok=True)
save_fig(fig, os.path.join(out_dir, 'fig_baseline_scores.pdf'))
print('Wrote fig_baseline_scores.pdf/png')
