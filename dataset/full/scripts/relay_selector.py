"""
RELAY: learned algorithm selection over the expanded solver pool.

At training time we mine (instance_features, best_solver_name) pairs from
baseline result JSONs.  A Random Forest classifier learns the mapping.

At test time, given a new instance we:
  1. Extract the feature vector in constant time.
  2. Predict the top-1 solver (or top-k with probabilities).
  3. Run that single solver for the full budget.

This approaches the VBS ceiling with single-solver compute budget.  Compared
to picking the best single solver blindly (NuWLS), the selector adds a
per-instance decision that can recover the performance gaps where NuWLS
loses but other solvers win.

Feature vector (12-dim):
  log10 n_vars, log10 n_clauses,
  soft_ratio,  hard_ratio,
  avg_clause_len, max_clause_len,
  log10 max_weight, log10 mean_weight, weight_cv,
  log10 top_weight (hard-threshold), is_modern_fmt, unit_fraction
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'eval'))

import numpy as np
import baselines

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(path: str) -> np.ndarray:
    """Parse a WCNF/CNF file once and return a 12-dim feature vector."""
    n_vars = 0
    n_clauses_soft = 0
    n_clauses_hard = 0
    lens = []
    weights_soft = []
    top_weight = 0.0
    is_modern = False
    n_units = 0

    try:
        with open(path, 'r', errors='replace') as f:
            for line in f:
                s = line.strip()
                if not s: continue
                if s.startswith('c'):
                    # Try to read ncls/nvars from embedded JSON (MSE 2022+)
                    continue
                if s.startswith('p '):
                    parts = s.split()
                    if len(parts) >= 4:
                        try: n_vars = int(parts[2])
                        except: pass
                    if len(parts) >= 5:
                        try: top_weight = float(parts[4])
                        except: pass
                    continue
                # Clause line
                toks = s.split()
                if not toks: continue
                if toks[0] == 'h':
                    n_clauses_hard += 1
                    clen = len(toks) - 2  # exclude 'h' and '0'
                    is_modern = True
                else:
                    try: w = float(toks[0])
                    except: continue
                    clen = len(toks) - 2
                    if top_weight > 0 and w >= top_weight:
                        n_clauses_hard += 1
                    else:
                        n_clauses_soft += 1
                        weights_soft.append(w)
                if clen <= 0: continue
                lens.append(clen)
                if clen == 1: n_units += 1
                # track max var seen if header missing
                for t in toks[1:-1]:
                    try:
                        v = abs(int(t))
                        if v > n_vars: n_vars = v
                    except: pass
    except Exception as e:
        logger.warning(f'feature extract error on {path}: {e}')
        return np.zeros(12, dtype=np.float32)

    n_clauses = n_clauses_soft + n_clauses_hard
    avg_len = float(np.mean(lens)) if lens else 0.0
    max_len = float(max(lens)) if lens else 0.0
    if weights_soft:
        max_w = max(weights_soft); mean_w = float(np.mean(weights_soft))
        std_w = float(np.std(weights_soft))
        cv = std_w / (mean_w + 1e-6)
    else:
        max_w = mean_w = cv = 0.0
    unit_frac = n_units / max(1, n_clauses)

    def lg(x): return float(math.log10(x + 1))

    return np.array([
        lg(n_vars), lg(n_clauses),
        n_clauses_soft / max(1, n_clauses), n_clauses_hard / max(1, n_clauses),
        avg_len, max_len,
        lg(max_w), lg(mean_w), cv,
        lg(top_weight), float(is_modern), unit_frac,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Build training set from baseline JSONs
# ---------------------------------------------------------------------------

def load_all_baseline_costs(timeout_key: str) -> dict[tuple, dict[str, float]]:
    """{(year, cat, name) -> {solver -> cost}}.

    Reads from results/baselines{,_300s,_new4}/*.json.
    """
    base = f'{REPO_ROOT}/results'
    dirs = [f'{base}/baselines', f'{base}/baselines_new4']
    if timeout_key == '300':
        dirs = [f'{base}/baselines_300s', f'{base}/baselines_new4_300s']
    out = defaultdict(dict)
    for d in dirs:
        if not os.path.isdir(d): continue
        for f in os.listdir(d):
            if not f.endswith('_results_raw.json'): continue
            solver = f.replace('_results_raw.json', '')
            try:
                data = json.load(open(os.path.join(d, f)))
            except Exception: continue
            for t, yd in data.items():
                if t != timeout_key: continue
                for y, cd in yd.items():
                    for c, inst in cd.items():
                        for n, s in inst.items():
                            cost = min(s) if isinstance(s, list) and s else float('inf')
                            out[(y, c, n)][solver] = cost
    return out


def _find_inst_path(name: str, bench_dir: str) -> Optional[str]:
    for root, _, files in os.walk(bench_dir):
        if name in files:
            return os.path.join(root, name)
    return None


def build_training_set(timeout: int, bench_dir: str,
                       cache_path: str = None) -> tuple[np.ndarray, np.ndarray, list, list]:
    """Return (X, y, inst_keys, solver_names).  y are class indices."""
    t_key = str(timeout)
    if cache_path and os.path.exists(cache_path):
        logger.info(f'loading cached training set from {cache_path}')
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    costs = load_all_baseline_costs(t_key)
    if not costs:
        raise RuntimeError(f'No baseline data for timeout {timeout}')

    # Fix solver set = all solvers that appear at least once
    all_solvers = sorted({s for c in costs.values() for s in c})
    logger.info(f'solvers in pool: {all_solvers}')

    X, y, keys = [], [], []
    skipped = 0
    for k, per_solver in costs.items():
        # Skip instances with no finite cost anywhere
        finite = {s: c for s, c in per_solver.items() if c != float('inf')}
        if not finite: continue
        best_cost = min(finite.values())
        # Tie-break: pick solver with lowest alphabetical name
        best_solver = min([s for s, c in finite.items() if c == best_cost])
        path = _find_inst_path(k[2], bench_dir)
        if not path:
            skipped += 1
            continue
        feats = extract_features(path)
        X.append(feats)
        y.append(all_solvers.index(best_solver))
        keys.append(k)
    logger.info(f'Built training set: {len(X)} instances, {skipped} skipped (path not found)')
    X = np.stack(X); y = np.array(y)

    if cache_path:
        with open(cache_path, 'wb') as f:
            pickle.dump((X, y, keys, all_solvers), f)
    return X, y, keys, all_solvers


# ---------------------------------------------------------------------------
# Train Random Forest selector
# ---------------------------------------------------------------------------

def train_selector(X, y, seed: int = 0):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        random_state=seed, n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# End-to-end: evaluate portfolio via cross-validation
# ---------------------------------------------------------------------------

def evaluate_cv(X, y, keys, solvers, timeout: int, bench_dir: str, n_folds: int = 5):
    """5-fold CV: in each fold, train selector on 4/5 and predict on 1/5.
    Report simulated cost by looking up the predicted solver's real cost."""
    from sklearn.model_selection import KFold

    costs = load_all_baseline_costs(str(timeout))
    bk = json.load(open(f'{REPO_ROOT}/results/mse_official_best_known.json'))

    sel_costs = []
    best_single = defaultdict(list)  # solver -> list of costs
    vbs_costs = []
    bk_values = []
    inst_keys_all = []

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    predictions = np.zeros(len(y), dtype=int)
    for tr, te in kf.split(X):
        clf = train_selector(X[tr], y[tr])
        predictions[te] = clf.predict(X[te])

    for i, k in enumerate(keys):
        best_solver = solvers[predictions[i]]
        per = costs[k]
        sel_cost = per.get(best_solver, float('inf'))
        sel_costs.append(sel_cost)
        for s in solvers:
            best_single[s].append(per.get(s, float('inf')))
        vbs_costs.append(min(per.values()))
        bk_values.append(bk.get(k[2]))
        inst_keys_all.append(k)

    # Scores
    def score(c, b):
        if b is None or b < 0 or c == float('inf'): return 0.0
        return max(0.0, min(1.0, (1 + b) / (1 + c)))

    valid = [i for i, b in enumerate(bk_values) if b is not None and b >= 0]
    print(f'\n{"="*90}')
    print(f'  CROSS-VALIDATED PORTFOLIO SELECTION ({len(valid)} instances, timeout={timeout}s)')
    print('='*90)

    # single solvers
    print(f"\n{'Solver':20s}{'avg_score':>12s}{'wins_vs_NuWLS':>16s}")
    print('-'*48)
    nuwls_scores = [score(best_single['nuwls'][i], bk_values[i]) for i in valid]
    for s in solvers:
        sc = [score(best_single[s][i], bk_values[i]) for i in valid]
        avg = float(np.mean(sc))
        if s == 'nuwls':
            print(f"{s:20s}{avg:>12.4f}{'--':>16s}")
        else:
            nw = [nuwls_scores[i] for i in range(len(sc))]
            wins = sum(1 for a, b in zip(sc, nw) if a > b)
            print(f"{s:20s}{avg:>12.4f}{wins:>16d}")
    # our selector
    sel_scores = [score(sel_costs[i], bk_values[i]) for i in valid]
    sel_wins = sum(1 for a, b in zip(sel_scores, nuwls_scores) if a > b)
    print(f"{'RELAY':20s}{float(np.mean(sel_scores)):>12.4f}{sel_wins:>16d}")
    # VBS
    vbs_scores = [score(vbs_costs[i], bk_values[i]) for i in valid]
    vbs_wins = sum(1 for a, b in zip(vbs_scores, nuwls_scores) if a > b)
    print(f"{'VBS (ceiling)':20s}{float(np.mean(vbs_scores)):>12.4f}{vbs_wins:>16d}")

    return {
        'portfolio_avg': float(np.mean(sel_scores)),
        'vbs_avg': float(np.mean(vbs_scores)),
        'nuwls_avg': float(np.mean(nuwls_scores)),
        'portfolio_wins_vs_nuwls': sel_wins,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--timeout', type=int, default=60)
    p.add_argument('--bench_dir', default=os.path.join(REPO_ROOT, 'benchmarks'))
    p.add_argument('--cache', default=os.path.join(REPO_ROOT, 'results', 'portfolio_features.pkl'))
    p.add_argument('--output_json', default=os.path.join(REPO_ROOT, 'results', 'portfolio_eval.json'))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%H:%M:%S')

    X, y, keys, solvers = build_training_set(args.timeout, args.bench_dir, args.cache)
    out = evaluate_cv(X, y, keys, solvers, args.timeout, args.bench_dir)
    with open(args.output_json, 'w') as f:
        json.dump(out, f, indent=2)
    logger.info(f'Saved {args.output_json}')


if __name__ == '__main__':
    main()
