"""Build a representative sample of RELAY-Bench:
   - 100 instances spanning all 5 years × 2 categories (10 per year-cat)
   - Their corresponding cost entries from all 8 solvers at 60s and 300s
   - The .wcnf files themselves (small instances only — total < 50 MB)
"""
from __future__ import annotations
import json, os, random, shutil
from collections import defaultdict

REPO = '/nas/ckgfs/jaunts/jahin/RELAY'
OUT = f'{REPO}/dataset/sample'
os.makedirs(f'{OUT}/instances', exist_ok=True)
os.makedirs(f'{OUT}/results', exist_ok=True)

random.seed(42)
SOLVERS = ['nuwls','bandhs','satlike','spb','foursat','nuwls_c','spb_c','nuwls_bandhs']
YEARS = ['2020','2021','2022','2023','2024']
CATS = ['ms','wms']
PER_YC = 10  # 10 per (year, cat) → 100 sample instances


def load_costs(timeout):
    suffix = '' if timeout == 60 else '_300s'
    data = {}
    for s in SOLVERS:
        for p in [f'{REPO}/results/baselines{suffix}/{s}_results_raw.json',
                  f'{REPO}/results/baselines_new4{suffix}/{s}_results_raw.json']:
            if os.path.exists(p):
                d = json.load(open(p))
                for t, yd in d.items():
                    for y, cd in yd.items():
                        for c, inst in cd.items():
                            for n, v in inst.items():
                                data.setdefault((y, c, n), {})[s] = v[0] if isinstance(v, list) and v else float('inf')
                break
    return data


def find_instance_path(name):
    for root, _, files in os.walk(f'{REPO}/benchmarks'):
        if name in files:
            return os.path.join(root, name)
    return None


def main():
    costs60 = load_costs(60)
    costs300 = load_costs(300)
    bk = json.load(open(f'{REPO}/results/mse_official_best_known.json'))

    # Stratified sample: 10 instances per (year, cat) where ALL 8 solvers have data + bk
    by_yc = defaultdict(list)
    for k in costs60:
        if all(s in costs60[k] for s in SOLVERS) and k[2] in bk and bk[k[2]] >= 0:
            by_yc[(k[0], k[1])].append(k)

    sampled = []
    for y in YEARS:
        for c in CATS:
            pool = by_yc.get((y, c), [])
            random.shuffle(pool)
            sampled.extend(pool[:PER_YC])

    print(f'Sampled {len(sampled)} instances across {len(YEARS)*len(CATS)} year-cat groups')

    # Build sample cost JSONs
    for timeout, costs in [(60, costs60), (300, costs300)]:
        per_solver = {s: defaultdict(lambda: defaultdict(dict)) for s in SOLVERS}
        for k in sampled:
            y, c, n = k
            for s in SOLVERS:
                v = costs.get(k, {}).get(s, float('inf'))
                per_solver[s][y][c][n] = [v]
        for s in SOLVERS:
            d = {str(timeout): per_solver[s]}
            with open(f'{OUT}/results/{s}_results_raw_t{timeout}.json', 'w') as f:
                json.dump(d, f, indent=2, default=lambda x: 'inf' if x == float('inf') else x)

    # Best-known subset
    bk_sample = {n: bk[n] for (_,_,n) in sampled if n in bk}
    with open(f'{OUT}/best_known.json', 'w') as f:
        json.dump(bk_sample, f, indent=2)

    # Copy actual .wcnf files (only those <500KB to keep sample compact)
    copied = 0
    skipped_size = 0
    inst_dir = f'{OUT}/instances'
    manifest = []
    for k in sampled:
        y, c, n = k
        path = find_instance_path(n)
        if not path:
            continue
        size_kb = os.path.getsize(path) // 1024
        if size_kb > 2000:
            skipped_size += 1
            manifest.append({'name': n, 'year': y, 'cat': c, 'included': False, 'reason': f'size={size_kb}KB'})
            continue
        dst = f'{inst_dir}/{c}{y}__{n}'
        shutil.copy2(path, dst)
        manifest.append({'name': n, 'year': y, 'cat': c, 'included': True, 'size_kb': size_kb})
        copied += 1
    print(f'Copied {copied} .wcnf files; skipped {skipped_size} (too large for sample)')
    with open(f'{OUT}/instance_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'\nSample built: {OUT}')


if __name__ == '__main__':
    main()
