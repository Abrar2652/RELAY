"""Final paper artifacts: tables + figures with parallel portfolio + selector results."""
from __future__ import annotations
import json, os, sys
from collections import defaultdict
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{REPO_ROOT}/eval')
from plotting import setup_style, save_fig, PALETTE

setup_style()

BASE = f'{REPO_ROOT}/results'
FIGS = f'{REPO_ROOT}/figures'
os.makedirs(FIGS, exist_ok=True)

MAIN_SOLVERS = ['nuwls','bandhs','satlike','spb','foursat','nuwls_c','spb_c','nuwls_bandhs']
DISPLAY = ['nuwls','satlike','spb','nuwls_c','spb_c','nuwls_bandhs']
LABEL = {'nuwls':'NuWLS', 'bandhs':'BandHS', 'satlike':'SatLike', 'spb':'SPB',
         'foursat':'FourierSAT', 'nuwls_c':'NuWLS-c', 'spb_c':'SPB-c',
         'nuwls_bandhs':'NuWLS-BandHS', 'VBS':'VBS', 'Portfolio':'RELAY'}


def flat(path, key='60'):
    d = json.load(open(path))
    out = {}
    for t, yd in d.items():
        if t != key: continue
        for y, cd in yd.items():
            for c, inst in cd.items():
                for n, s in inst.items():
                    out[(y, c, n)] = min(s) if isinstance(s, list) and s else float('inf')
    return out


def load_all():
    bk = json.load(open(f'{BASE}/mse_official_best_known.json'))
    data = {}
    for s in MAIN_SOLVERS:
        for p in [f'{BASE}/baselines/{s}_results_raw.json',
                  f'{BASE}/baselines_new4/{s}_results_raw.json']:
            if os.path.exists(p):
                data[s] = flat(p)
                break
    return data, bk


def sc(c, b): return 0.0 if c == float('inf') else max(0.0, min(1.0, (1+b)/(1+c)))


def load_all_300s():
    bk = json.load(open(f'{BASE}/mse_official_best_known.json'))
    data = {}
    for s in MAIN_SOLVERS:
        for p in [f'{BASE}/baselines_300s/{s}_results_raw.json',
                  f'{BASE}/baselines_new4_300s/{s}_results_raw.json']:
            if os.path.exists(p):
                d = json.load(open(p))
                out = {}
                for t, yd in d.items():
                    if t != '300': continue
                    for y, cd in yd.items():
                        for c, inst in cd.items():
                            for n, sv in inst.items():
                                out[(y, c, n)] = min(sv) if isinstance(sv, list) and sv else float('inf')
                data[s] = out
                break
    return data, bk


def compute_totals(data, bk):
    all_keys = set(data['nuwls'])
    for s in MAIN_SOLVERS[1:]:
        if s in data: all_keys &= set(data[s])
    keys = [k for k in all_keys if k[2] in bk and bk[k[2]] >= 0]
    totals = defaultdict(list)
    for k in keys:
        for s in MAIN_SOLVERS:
            if s in data: totals[s].append(sc(data[s][k], bk[k[2]]))
        totals['VBS'].append(sc(min(data[s][k] for s in MAIN_SOLVERS if s in data), bk[k[2]]))
        totals['P2_NuWLS+SPB'].append(sc(min(data['nuwls'][k], data['spb'][k]), bk[k[2]]))
        totals['P3_NuWLS+SatLike+SPB'].append(sc(min(data['nuwls'][k], data['satlike'][k], data['spb'][k]), bk[k[2]]))
        totals['P4_top4'].append(sc(min(data[s][k] for s in ['nuwls','nuwls_c','spb','spb_c']), bk[k[2]]))
    return totals, keys


def main():
    data, bk = load_all()
    all_keys = set(data['nuwls'])
    for s in MAIN_SOLVERS[1:]: all_keys &= set(data[s])
    keys = [k for k in all_keys if k[2] in bk and bk[k[2]] >= 0]
    print(f'Full matched: {len(keys)} instances')

    # Per year-cat x solver + portfolios
    rows = defaultdict(lambda: defaultdict(list))
    for k in keys:
        y, c, _ = k
        for s in MAIN_SOLVERS:
            rows[(y, c)][s].append(sc(data[s][k], bk[k[2]]))
        # VBS
        rows[(y, c)]['VBS'].append(sc(min(data[s][k] for s in MAIN_SOLVERS), bk[k[2]]))
        # Parallel portfolios
        portfolios = {
            'P2_NuWLS+SPB':       min(data['nuwls'][k], data['spb'][k]),
            'P3_NuWLS+SatLike+SPB': min(data['nuwls'][k], data['satlike'][k], data['spb'][k]),
            'P4_top4':            min(data[s][k] for s in ['nuwls','nuwls_c','spb','spb_c']),
        }
        for name, cost in portfolios.items():
            rows[(y, c)][name].append(sc(cost, bk[k[2]]))

    # ------------- TABLE 1: Main year-cat comparison ---------------
    tbl_lines = [r"""\begin{table*}[t]
\centering\small
\caption{Average incomplete score on MaxSAT Evaluation 2020--2024, 60s timeout.
Score $= (1+c^*)/(1+c)$ capped to $[0,1]$.  $c^*$ is the official MSE best-known cost.
Bold marks best per column.  All parallel portfolios beat NuWLS with zero losses
(see Table~\ref{tab:h2h}).}
\label{tab:main}
\begin{tabular}{l"""+"c"*10+"|c"+r"}\toprule"""]
    years = ['2020','2021','2022','2023','2024']
    cats = ['ms','wms']
    header = r"\textbf{Method} & " + " & ".join(f"{c}-{y[2:]}" for y in years for c in cats) + r" & \textbf{Avg} \\"
    tbl_lines += [header, r"\midrule"]
    totals = defaultdict(list)
    methods = DISPLAY + ['VBS', 'P2_NuWLS+SPB', 'P3_NuWLS+SatLike+SPB']
    for m in methods:
        for (y, c), sd in rows.items():
            totals[m].extend(sd[m])
    for m in methods:
        row = LABEL.get(m, m.replace('P2_', 'PP2: ').replace('P3_', 'PP3: ')) + " & "
        for y in years:
            for c in cats:
                v = np.mean(rows[(y, c)][m])
                row += f"{v:.3f} & "
        row += f"{np.mean(totals[m]):.3f} \\\\"
        tbl_lines.append(row)
    tbl_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    tbl_path = f'{FIGS}/tbl_main.tex'
    with open(tbl_path, 'w') as f: f.write('\n'.join(tbl_lines))
    print(f'Wrote {tbl_path}')

    # ------------- TABLE 2: Head-to-head vs NuWLS ---------------
    def h2h(col, ref='nuwls'):
        a = totals[col]; b = totals[ref]
        w = sum(1 for x, y in zip(a, b) if x > y)
        t = sum(1 for x, y in zip(a, b) if abs(x-y) < 1e-9)
        l = sum(1 for x, y in zip(a, b) if x < y)
        return w, t, l
    h2h_lines = [r"""\begin{table}[t]\centering\small
\caption{Head-to-head instance counts vs NuWLS (previous single-solver SOTA).
``W/T/L'' means instances where method's score is strictly better / tied / worse than NuWLS.
Parallel portfolios show zero losses by construction (take min cost).}
\label{tab:h2h}
\begin{tabular}{lrrrr}\toprule
\textbf{Method} & \textbf{Avg} & \textbf{W} & \textbf{T} & \textbf{L} \\\midrule"""]
    for m in DISPLAY + ['VBS', 'P2_NuWLS+SPB', 'P3_NuWLS+SatLike+SPB']:
        if m == 'nuwls': continue
        w, t, l = h2h(m)
        avg = np.mean(totals[m])
        h2h_lines.append(f"{LABEL.get(m, m)} & {avg:.4f} & {w} & {t} & {l} \\\\")
    h2h_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(f'{FIGS}/tbl_h2h.tex', 'w') as f: f.write('\n'.join(h2h_lines))
    print(f'Wrote {FIGS}/tbl_h2h.tex')

    # ------------- FIGURE: bar chart main comparison ---------------
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    methods_plot = ['nuwls', 'satlike', 'spb', 'nuwls_c', 'spb_c',
                    'P2_NuWLS+SPB', 'P3_NuWLS+SatLike+SPB', 'VBS']
    labels = [LABEL.get(m, m.replace('P2_', 'PP2: ').replace('P3_', 'PP3: '))
              for m in methods_plot]
    vals = [np.mean(totals[m]) for m in methods_plot]
    colors_order = [PALETTE['gray'], PALETTE['cyan'], PALETTE['purple'],
                    PALETTE['green'], PALETTE['orange'],
                    PALETTE['red'], PALETTE['red'], PALETTE['black']]
    bars = ax.bar(range(len(labels)), vals, color=colors_order,
                  edgecolor='black', linewidth=0.8)
    # emphasis: outline our methods
    for i in [5, 6]:
        bars[i].set_linewidth(2.0)
        bars[i].set_hatch('//')
    # VBS bar: distinct
    bars[7].set_hatch('xx')
    bars[7].set_linewidth(2.0)

    ax.axhline(vals[0], color='black', linestyle=':', linewidth=0.8, alpha=0.6,
               label='NuWLS baseline')
    for i, (v, l) in enumerate(zip(vals, labels)):
        ax.text(i, v + 0.008, f'{v:.3f}', ha='center', fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Avg Incomplete Score (60s)')
    # y-range must cover lowest (SPB/SPB-c ~0.63) and highest (VBS ~0.90)
    ymin = max(0.0, min(vals) - 0.03)
    ymax = min(1.0, max(vals) + 0.03)
    ax.set_ylim(ymin, ymax)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    save_fig(fig, f'{FIGS}/fig_main_portfolio.pdf')
    print(f'Wrote {FIGS}/fig_main_portfolio.pdf')

    # ------------- FIGURE: per-year lift ---------------
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    years_plot = years
    width = 0.2
    xs = np.arange(len(years_plot))
    method_bundle = [
        ('nuwls', PALETTE['gray'], 'NuWLS'),
        ('P2_NuWLS+SPB', PALETTE['orange'], 'PP2: NuWLS+SPB'),
        ('P3_NuWLS+SatLike+SPB', PALETTE['red'], 'PP3: NuWLS+SatLike+SPB'),
        ('VBS', PALETTE['black'], 'VBS'),
    ]
    all_year_vals = []
    for i, (meth, color, label) in enumerate(method_bundle):
        vals = []
        for y in years_plot:
            scores = []
            for c in cats:
                scores.extend(rows[(y, c)][meth])
            vals.append(np.mean(scores) if scores else 0)
        all_year_vals.extend(vals)
        ax.bar(xs + i*width, vals, width, label=label,
               color=color, edgecolor='black', linewidth=0.6)
        for j, v in enumerate(vals):
            ax.text(xs[j] + i*width, v + 0.01, f'{v:.2f}',
                    ha='center', fontsize=6, rotation=0)
    ax.set_xticks(xs + 1.5*width)
    ax.set_xticklabels(years_plot)
    ax.set_xlabel('Year (MS + WMS combined)')
    ax.set_ylabel('Avg Incomplete Score (60s)')
    # Auto-scale to include the lowest 2021 value (~0.52)
    ymin = max(0.0, min(all_year_vals) - 0.05)
    ax.set_ylim(ymin, 1.12)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    # Legend above the plot area (bars don't reach there)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
              frameon=False, fontsize=8, ncol=4)
    plt.tight_layout()
    save_fig(fig, f'{FIGS}/fig_per_year.pdf')
    print(f'Wrote {FIGS}/fig_per_year.pdf')

    # ------------- Summary JSON ---------------
    summary = {}
    for m in MAIN_SOLVERS + ['VBS', 'P2_NuWLS+SPB', 'P3_NuWLS+SatLike+SPB', 'P4_top4']:
        summary[m] = {
            'avg_score': float(np.mean(totals[m])) if totals.get(m) else 0,
        }
        if m != 'nuwls':
            w, t, l = h2h(m)
            summary[m].update(wins_vs_nuwls=w, ties_vs_nuwls=t, losses_vs_nuwls=l)
    with open(f'{BASE}/final_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {BASE}/final_summary.json')


if __name__ == '__main__':
    main()
