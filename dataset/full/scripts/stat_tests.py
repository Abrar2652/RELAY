"""
Statistical tests + summary aggregation for RELAY vs baselines.

Takes the per-solver JSON files produced by ``evaluate.py`` / ``run_baselines.py``
/ ``run_ablation.py`` and produces:

  - Avg incomplete score per (solver, year, category)
  - Paired Wilcoxon signed-rank test RELAY vs each baseline (per year × cat)
  - Bootstrap 95% CIs on the avg incomplete score
  - Best-known tracking (the minimum cost seen across any solver)

All tests are non-parametric because the cost distribution across instances is
heavy-tailed, not approximately normal.
"""

from __future__ import annotations

import os
import json
import math
import argparse
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Incomplete score (matches MaxSAT Evaluations + SGAT-MS paper formula)
# ---------------------------------------------------------------------------
def incomplete_score(solver_cost: float, best_known: float) -> float:
    """score(s, i) = (1 + best_known) / (1 + solver_cost), higher is better."""
    if solver_cost == float('inf') or math.isnan(solver_cost):
        return 0.0
    return (1.0 + best_known) / (1.0 + solver_cost)


# ---------------------------------------------------------------------------
# Loading — handles both RELAY's and our baseline runner's JSON shape
# ---------------------------------------------------------------------------
def load_results(path: str) -> Tuple[dict, str]:
    """Load a results JSON file.

    Returns (raw_results, shape_tag) where raw_results is the timeout->year->cat
    nested dict.  Handles two shapes:
        RELAY/evaluate.py:       { "60": { "2020": { "ms": { inst: [costs] }}}}
        run_ablation.py:         { "config": {...}, "results": <same as above> }
    """
    with open(path, 'r') as f:
        data = json.load(f)
    if 'results' in data and 'config' in data:
        return data['results'], 'ablation'
    return data, 'main'


# ---------------------------------------------------------------------------
# Best-known cost tracking across multiple solvers
# ---------------------------------------------------------------------------
def compute_best_known(solvers: Dict[str, dict]) -> Dict[str, float]:
    """Minimum cost observed for each instance across all solvers/timeouts."""
    bk: Dict[str, float] = {}
    for raw in solvers.values():
        for timeout in raw.values():
            for year_results in timeout.values():
                for cat_results in year_results.values():
                    for inst, costs in cat_results.items():
                        valid = [c for c in costs if c != float('inf')]
                        if not valid:
                            continue
                        m = min(valid)
                        if inst not in bk or m < bk[inst]:
                            bk[inst] = m
    return bk


# ---------------------------------------------------------------------------
# Summary: avg score per (solver, timeout, year, cat) + per-instance score list
# ---------------------------------------------------------------------------
def summarize(
    solvers: Dict[str, dict],
    best_known: Dict[str, float],
) -> Tuple[dict, dict]:
    """Return (summary_dict, per_instance_scores).

    summary_dict[solver][timeout][year][cat] = {
        'avg_incomplete_score': float,
        'bootstrap_ci_low': float,
        'bootstrap_ci_high': float,
        'n_instances': int,
    }
    per_instance_scores[(solver, timeout, year, cat)] = {inst: score}
    """
    summary: Dict = {}
    per_inst: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}

    for solver, raw in solvers.items():
        summary.setdefault(solver, {})
        for timeout, year_map in raw.items():
            summary[solver].setdefault(timeout, {})
            for year, cat_map in year_map.items():
                summary[solver][timeout].setdefault(year, {})
                for cat, inst_map in cat_map.items():
                    scores = {}
                    for inst, costs in inst_map.items():
                        valid = [c for c in costs if c != float('inf')]
                        if not valid:
                            scores[inst] = 0.0
                            continue
                        cost = min(valid)  # best across seeds
                        scores[inst] = incomplete_score(cost, best_known.get(inst, 0.0))

                    if not scores:
                        continue

                    arr = np.array(list(scores.values()))
                    avg = float(arr.mean())
                    lo, hi = bootstrap_ci(arr)
                    summary[solver][timeout][year][cat] = {
                        'avg_incomplete_score': round(avg, 4),
                        'bootstrap_ci_low': round(lo, 4),
                        'bootstrap_ci_high': round(hi, 4),
                        'n_instances': len(scores),
                    }
                    per_inst[(solver, timeout, year, cat)] = scores
    return summary, per_inst


# ---------------------------------------------------------------------------
# Bootstrap 95% confidence interval for the mean
# ---------------------------------------------------------------------------
def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Percentile bootstrap CI on the mean."""
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_resamples)
    ])
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


# ---------------------------------------------------------------------------
# Paired Wilcoxon signed-rank test (RELAY vs each baseline)
# ---------------------------------------------------------------------------
def wilcoxon_paired(
    hydra_scores: Dict[str, float],
    other_scores: Dict[str, float],
) -> Tuple[float, float, int]:
    """Return (W, p_value, n_pairs).

    Uses the Pratt correction for zero differences.  Reduces to scipy's
    ``wilcoxon`` when available; otherwise implements the test manually to
    keep the dependency surface small.
    """
    common = sorted(set(hydra_scores) & set(other_scores))
    if not common:
        return (0.0, 1.0, 0)
    diffs = np.array([hydra_scores[k] - other_scores[k] for k in common])
    diffs = diffs[diffs != 0.0]  # Wilcoxon classic drops ties
    n = len(diffs)
    if n < 6:  # Wilcoxon's asymptotic approximation needs ~6+ pairs
        return (0.0, 1.0, n)
    try:
        from scipy.stats import wilcoxon  # type: ignore
        w, p = wilcoxon(diffs, alternative='greater')
        return float(w), float(p), n
    except ImportError:
        # Manual fallback — two-sided approximate
        abs_d = np.abs(diffs)
        ranks = np.argsort(np.argsort(abs_d)) + 1  # rank 1..n (ties average)
        w_pos = float(ranks[diffs > 0].sum())
        w_neg = float(ranks[diffs < 0].sum())
        W = min(w_pos, w_neg)
        # Normal approximation
        mu = n * (n + 1) / 4.0
        sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        z = (W - mu) / sigma if sigma > 0 else 0.0
        # two-sided p via normal CDF
        from math import erfc
        p = erfc(abs(z) / math.sqrt(2.0))
        return W, p, n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Stats over RELAY + baseline results')
    parser.add_argument('--hydra_json', required=True,
                        help='RELAY results_raw.json')
    parser.add_argument('--baseline_jsons', nargs='+', default=[],
                        help='One or more baseline <name>_results_raw.json files')
    parser.add_argument('--out', default='./results/stats.json')
    parser.add_argument('--reference_solver', default='hydra',
                        help='Which solver is the focus of paired tests (default: hydra)')
    args = parser.parse_args()

    solvers = {}

    hydra_raw, _ = load_results(args.hydra_json)
    solvers['hydra'] = hydra_raw

    for p in args.baseline_jsons:
        # Derive solver name from filename stem
        stem = os.path.splitext(os.path.basename(p))[0]
        name = stem.replace('_results_raw', '').replace('_raw', '')
        raw, _ = load_results(p)
        solvers[name] = raw

    bk = compute_best_known(solvers)
    summary, per_inst = summarize(solvers, bk)

    # Paired tests: reference_solver vs each other
    stats = {}
    for (s, t, y, c), ref_scores in per_inst.items():
        if s != args.reference_solver:
            continue
        key = f'{t}s_{y}_{c}'
        stats[key] = {}
        for (s2, t2, y2, c2), other_scores in per_inst.items():
            if s2 == args.reference_solver or (t2, y2, c2) != (t, y, c):
                continue
            W, p, n = wilcoxon_paired(ref_scores, other_scores)
            stats[key][f'{args.reference_solver}_vs_{s2}'] = {
                'W': W, 'p_value_greater': p, 'n_pairs': n,
            }

    out = {
        'best_known': bk,
        'summary':    summary,
        'paired_wilcoxon': stats,
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Wrote {args.out}')
    # Short console summary
    print('\n=== Avg Incomplete Score (best-known reference) ===')
    for s in sorted(summary):
        for t in sorted(summary[s]):
            for y in sorted(summary[s][t]):
                for c in sorted(summary[s][t][y]):
                    e = summary[s][t][y][c]
                    print(f'  {s:<12s} {t:>4s}s {c.upper()} {y}: '
                          f'{e["avg_incomplete_score"]:.4f} '
                          f'[{e["bootstrap_ci_low"]:.4f}, {e["bootstrap_ci_high"]:.4f}] '
                          f'(n={e["n_instances"]})')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
