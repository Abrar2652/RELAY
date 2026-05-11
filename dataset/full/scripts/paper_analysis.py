"""Final paper analysis: scores + win-tie-loss tables."""
from __future__ import annotations
import json, os, sys
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLVERS = ['nuwls', 'bandhs', 'satlike', 'spb', 'foursat']


def load_baseline(solver: str, timeout_key: str):
    p = f'{BASE}/results/baselines{"" if timeout_key == "60" else "_300s"}/{solver}_results_raw.json'
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    out = {}
    for t, yd in d.items():
        if t != timeout_key: continue
        for y, cd in yd.items():
            for c, inst in cd.items():
                for n, seeds in inst.items():
                    cost = min(seeds) if isinstance(seeds, list) and seeds else float('inf')
                    out[(y, c, n)] = cost
    return out


def load_hydra_fix(timeout_key: str):
    shard_dir = f'{BASE}/results/hydra_fix/eval_{timeout_key}s'
    if not os.path.exists(shard_dir): return {}
    out = {}
    for d in os.listdir(shard_dir):
        p = os.path.join(shard_dir, d, 'results_raw.json')
        if not os.path.exists(p): continue
        dd = json.load(open(p))
        for t, yd in dd.items():
            for y, cd in yd.items():
                for c, inst in cd.items():
                    for n, seeds in inst.items():
                        cost = min(seeds) if seeds else float('inf')
                        out[(y, c, n)] = cost
    return out


def score(cost, bk):
    if bk is None or bk < 0 or cost == float('inf'): return 0.0
    return max(0.0, min(1.0, (1 + bk) / (1 + cost)))


def report(timeout_key: str):
    bk = json.load(open(f'{BASE}/results/mse_official_best_known.json'))
    data = {s: load_baseline(s, timeout_key) for s in SOLVERS}
    data['hydra_fix'] = load_hydra_fix(timeout_key)

    # VBS over baselines only
    all_keys = set()
    for d in data.values():
        all_keys.update(d.keys())
    per_inst_vbs = {}
    for k in all_keys:
        costs = [data[s].get(k, float('inf')) for s in SOLVERS]
        per_inst_vbs[k] = min(costs) if costs else float('inf')

    # Per-year-cat score
    print('=' * 100)
    print(f'  RESULTS AT TIMEOUT = {timeout_key}s')
    print('=' * 100)
    print()
    header = f"{'yr/cat':10s}{'n':>6s}"
    for s in SOLVERS + ['VBS', 'RELAY-Fix']:
        header += f"{s:>10s}"
    print(header)
    print('-' * len(header))

    totals = defaultdict(list)
    for (y, c) in sorted({(k[0], k[1]) for k in all_keys}):
        insts = [k for k in all_keys if k[0] == y and k[1] == c]
        insts_with_bk = [k for k in insts if k[2] in bk and bk[k[2]] >= 0]
        row = f"{c+'-'+y:10s}{len(insts_with_bk):>6d}"
        for s in SOLVERS:
            scores = [score(data[s].get(k, float('inf')), bk.get(k[2])) for k in insts_with_bk]
            avg = sum(scores)/max(len(scores),1)
            totals[s].extend(scores)
            row += f"{avg:>10.4f}"
        # VBS
        scores = [score(per_inst_vbs[k], bk.get(k[2])) for k in insts_with_bk]
        totals['VBS'].extend(scores)
        row += f"{sum(scores)/max(len(scores),1):>10.4f}"
        # RELAY-Fix — only score on instances that RELAY-Fix attempted
        hf_scores = []
        for k in insts_with_bk:
            if k in data['hydra_fix']:
                hf_scores.append(score(data['hydra_fix'][k], bk.get(k[2])))
        if hf_scores:
            totals['RELAY-Fix'].extend(hf_scores)
            row += f"{sum(hf_scores)/len(hf_scores):>10.4f}"
        else:
            row += f"{'--':>10s}"
        print(row)

    print('-' * len(header))
    row = f"{'OVERALL':10s}{len(totals[SOLVERS[0]]):>6d}"
    for s in SOLVERS + ['VBS']:
        row += f"{sum(totals[s])/max(len(totals[s]),1):>10.4f}"
    if totals['RELAY-Fix']:
        row += f"{sum(totals['RELAY-Fix'])/len(totals['RELAY-Fix']):>10.4f}"
    else:
        row += f"{'--':>10s}"
    print(row)
    print()

    # RELAY-Fix vs NuWLS head-to-head
    if totals['RELAY-Fix']:
        print('RELAY-Fix vs NuWLS (same instances):')
        win = tie = loss = 0
        for k in data['hydra_fix']:
            if k not in data['nuwls']: continue
            hf = data['hydra_fix'][k]
            nw = data['nuwls'][k]
            if hf < nw: win += 1
            elif hf == nw: tie += 1
            else: loss += 1
        print(f'  wins={win}  ties={tie}  losses={loss}   ({win}/{win+tie+loss})')
    print()


if __name__ == '__main__':
    for t in ['60', '300']:
        report(t)
