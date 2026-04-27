"""Combined 60s + 300s LaTeX table for the paper.

Emits a single table showing NuWLS / SatLike / SPB / NuWLS-c / SPB-c /
NuWLS-BandHS + VBS + parallel portfolios at both timeouts, per (year, cat).
"""
from __future__ import annotations
import json, os, sys
from collections import defaultdict
from itertools import combinations
import numpy as np

REPO_ROOT = '/nas/ckgfs/jaunts/jahin/RELAY'
BASE = f'{REPO_ROOT}/results'
FIGS = f'{REPO_ROOT}/figures'
os.makedirs(FIGS, exist_ok=True)

MAIN_SOLVERS = ['nuwls','bandhs','satlike','spb','foursat','nuwls_c','spb_c','nuwls_bandhs']
DISPLAY = ['nuwls','satlike','spb','nuwls_c','spb_c','nuwls_bandhs']
LABEL = {'nuwls':'NuWLS', 'satlike':'SatLike', 'spb':'SPB',
         'nuwls_c':'NuWLS-c', 'spb_c':'SPB-c', 'nuwls_bandhs':'NuWLS-BandHS',
         'VBS': r'\textit{VBS}',
         'P2_NuWLS+SPB': r'\textbf{PP2}: NuWLS+SPB',
         'P3_NuWLS+SatLike+SPB': r'\textbf{PP3}: NuWLS+SatLike+SPB',
         'P4_top4': r'\textbf{PP4}: Top-4'}


def flat(path, key):
    d = json.load(open(path))
    out = {}
    for t, yd in d.items():
        if t != key: continue
        for y, cd in yd.items():
            for c, inst in cd.items():
                for n, s in inst.items():
                    out[(y, c, n)] = min(s) if isinstance(s, list) and s else float('inf')
    return out


def load(timeout):
    key = str(timeout)
    suffix = '' if timeout == 60 else '_300s'
    data = {}
    for s in MAIN_SOLVERS:
        for p in [f'{BASE}/baselines{suffix}/{s}_results_raw.json',
                  f'{BASE}/baselines_new4{suffix}/{s}_results_raw.json']:
            if os.path.exists(p):
                data[s] = flat(p, key)
                break
    return data


def sc(c, b): return 0.0 if c == float('inf') else max(0.0, min(1.0, (1+b)/(1+c)))


def per_year_cat(data, bk, years, cats):
    all_keys = set(data['nuwls'])
    for s in MAIN_SOLVERS[1:]:
        if s in data: all_keys &= set(data[s])
    keys = [k for k in all_keys if k[2] in bk and bk[k[2]] >= 0]
    result = {'_keys': keys}
    for m in MAIN_SOLVERS + ['VBS', 'P2_NuWLS+SPB', 'P3_NuWLS+SatLike+SPB', 'P4_top4']:
        per = defaultdict(list)
        for k in keys:
            y, c, _ = k
            if m in MAIN_SOLVERS and m in data:
                per[(y, c)].append(sc(data[m][k], bk[k[2]]))
            elif m == 'VBS':
                per[(y, c)].append(sc(min(data[s][k] for s in MAIN_SOLVERS if s in data), bk[k[2]]))
            elif m == 'P2_NuWLS+SPB':
                per[(y, c)].append(sc(min(data['nuwls'][k], data['spb'][k]), bk[k[2]]))
            elif m == 'P3_NuWLS+SatLike+SPB':
                per[(y, c)].append(sc(min(data['nuwls'][k], data['satlike'][k], data['spb'][k]), bk[k[2]]))
            elif m == 'P4_top4':
                per[(y, c)].append(sc(min(data[s][k] for s in ['nuwls','nuwls_c','spb','spb_c']), bk[k[2]]))
        result[m] = per
    return result


def main():
    bk = json.load(open(f'{BASE}/mse_official_best_known.json'))
    d60 = load(60); d300 = load(300)
    years = ['2020','2021','2022','2023','2024']
    cats = ['ms','wms']
    r60 = per_year_cat(d60, bk, years, cats)
    r300 = per_year_cat(d300, bk, years, cats)

    # combined table: per method, 60s Avg | 300s Avg | wins/losses vs NuWLS
    tbl = [r"""\begin{table}[t]\centering\small
\caption{Incomplete MaxSAT scores on MSE 2020--2024 (1736 instances per timeout). Score $= (1+c^*)/(1+c)$ capped to $[0,1]$.
\textbf{PP}$n$ = $n$-solver parallel portfolio (min cost). All parallel portfolios have zero regressions vs NuWLS at both timeouts.}
\label{tab:main_combined}
\begin{tabular}{lrrrr|rrrr}\toprule
 & \multicolumn{4}{c|}{\textbf{60s timeout}} & \multicolumn{4}{c}{\textbf{300s timeout}} \\
\textbf{Method} & Avg & $\Delta$ & W & L & Avg & $\Delta$ & W & L \\\midrule"""]

    methods = DISPLAY + ['VBS', 'P2_NuWLS+SPB', 'P3_NuWLS+SatLike+SPB', 'P4_top4']

    def agg(per_yc):
        vals = []
        for v in per_yc.values(): vals.extend(v)
        return vals

    nw60 = agg(r60['nuwls']); nw300 = agg(r300['nuwls'])

    def hth(me, ref):
        w = sum(1 for a, b in zip(me, ref) if a > b)
        l = sum(1 for a, b in zip(me, ref) if a < b)
        return w, l

    for m in methods:
        row = LABEL[m] + " & "
        m60 = agg(r60[m]); m300 = agg(r300[m])
        avg60 = np.mean(m60) if m60 else 0
        avg300 = np.mean(m300) if m300 else 0
        if m == 'nuwls':
            row += f"{avg60:.4f} & --- & --- & --- & {avg300:.4f} & --- & --- & --- \\\\"
        else:
            w60, l60 = hth(m60, nw60)
            w300, l300 = hth(m300, nw300)
            row += f"{avg60:.4f} & ${avg60-np.mean(nw60):+.4f}$ & {w60} & {l60} & "
            row += f"{avg300:.4f} & ${avg300-np.mean(nw300):+.4f}$ & {w300} & {l300} \\\\"
        tbl.append(row)

    tbl += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path = f'{FIGS}/tbl_combined.tex'
    with open(path, 'w') as f: f.write('\n'.join(tbl))
    print(f'Wrote {path}')

    # Summary JSON
    summary = {'60s': {}, '300s': {}}
    for m in methods:
        m60 = agg(r60[m]); m300 = agg(r300[m])
        w60, l60 = hth(m60, nw60); w300, l300 = hth(m300, nw300)
        summary['60s'][m] = {'avg': float(np.mean(m60)) if m60 else 0,
                             'wins_vs_nuwls': w60, 'losses_vs_nuwls': l60}
        summary['300s'][m] = {'avg': float(np.mean(m300)) if m300 else 0,
                              'wins_vs_nuwls': w300, 'losses_vs_nuwls': l300}
    with open(f'{BASE}/final_combined_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {BASE}/final_combined_summary.json')


if __name__ == '__main__':
    main()
