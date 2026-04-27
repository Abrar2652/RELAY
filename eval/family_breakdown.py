"""
Per-family benchmark breakdown.

MaxSAT Evaluation instances are named with a family prefix
(``scheduling_xiaojuan-…``, ``maxcut-…``, ``MinFill_…``, etc.).  Breaking
results down by family shows reviewers where RELAY dominates versus where
baselines already work well — a qualitative diagnostic SGAT-MS skipped but
reviewers often request.

Family extraction is conservative: we split on the first ``-``, ``_`` or
``.`` that appears after a letter prefix.  Edge cases are mapped to
``'other'``.
"""

from __future__ import annotations

import os
import re
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from stat_tests import (
    load_results, compute_best_known, incomplete_score, bootstrap_ci,
)


# Manually curated family prefixes observed across MSE benchmarks.  Order
# matters — longer-prefix matches are tried first so e.g. ``scheduling_xiaojuan``
# isn't swallowed by the bare ``scheduling``.
FAMILY_PATTERNS = [
    # (regex, canonical name)
    (r'^MaxSATQueriesinInterpretableClassifiers', 'decision-tree'),
    (r'^(?:decision-tree|xai-)', 'decision-tree'),
    (r'^(?:maxcut|max-cut|dim\.brock|dim\.hamming|dim\.MANN|bip\.maxcut|ran\.max_cut|str\.p_hat)', 'max-cut'),
    (r'^(?:scheduling_xiaojuan|staff-scheduling|shiftdesign|sched)', 'scheduling'),
    (r'^(?:min-fill|MinFill)', 'min-fill'),
    (r'^(?:treewidth|TWComp)', 'treewidth'),
    (r'^(?:abstraction-refinement)', 'abstraction-refinement'),
    (r'^(?:af-synthesis|extension-enforcement)', 'argumentation'),
    (r'^(?:atcoss|mes\.atcoss|sug\.atcoss)', 'atcoss'),
    (r'^(?:BrazilInstance|GreeceWestern|instance)', 'timetabling'),
    (r'^(?:ran-scp|cra-scp|set-covering)', 'set-covering'),
    (r'^(?:close_solutions|SAT04)', 'close-solutions'),
    (r'^(?:causal-discovery|causal_n)', 'causal-discovery'),
    (r'^(?:CircuitDebugging|rsdecoder|wb_)', 'circuit-debug'),
    (r'^(?:judgment-aggregation|ja-)', 'judgment-aggregation'),
    (r'^(?:lisbon-wedding|sysadmin|uaq-)', 'misc-combinatorial'),
    (r'^(?:switchingactivitymaximization|SwitchingActivity)', 'switching-activity'),
    (r'^(?:pesp|railway-transport|timetabling)', 'transport'),
    (r'^(?:drmx-cryptogen|cryptogen|cellda|gen_mult|gen_add|sbox)', 'crypto'),
    (r'^(?:bcp|BCP-)', 'bcp'),
    (r'^(?:cnf|wcn|hs-timetabling|ms_)', 'phylogenetics'),
    (r'^(?:inconsistency-measurement|im-hit)', 'inconsistency'),
    (r'^(?:optimizing-BDDs|BDDs)', 'bdd'),
    (r'^(?:MinWidthCB|power_|mitdbsample)', 'min-width'),
    (r'^(?:ParametricRBAC|role_domino)', 'rbac'),
    (r'^(?:reversi|rev66)', 'reversi'),
    (r'^(?:navigation_)', 'navigation'),
    (r'^(?:data\.|cellda_|unw\.|wei\.)', 'misc-planning'),
]


def classify(instance_name: str) -> str:
    """Return a canonical family label for an instance filename."""
    base = os.path.basename(instance_name)
    for pattern, label in FAMILY_PATTERNS:
        if re.match(pattern, base):
            return label
    # Fallback: first token before '-', '_', '.' if it's long enough
    m = re.match(r'^([A-Za-z][A-Za-z0-9]{2,})[-_.]', base)
    if m:
        return m.group(1).lower()
    return 'other'


def breakdown(
    solvers: Dict[str, dict],
    best_known: Dict[str, float],
) -> Dict:
    """Return {family: {solver: {timeout: {year: {cat: summary}}}}}.

    summary fields: avg_incomplete_score, ci_low, ci_high, n_instances.
    """
    # Re-key per-instance scores by family
    out: Dict = {}
    for solver, raw in solvers.items():
        for timeout, year_map in raw.items():
            for year, cat_map in year_map.items():
                for cat, inst_map in cat_map.items():
                    fam_scores = defaultdict(list)
                    for inst, costs in inst_map.items():
                        valid = [c for c in costs if c != float('inf')]
                        if not valid:
                            score = 0.0
                        else:
                            score = incomplete_score(min(valid), best_known.get(inst, 0.0))
                        fam_scores[classify(inst)].append(score)

                    for fam, scores in fam_scores.items():
                        arr = np.array(scores)
                        lo, hi = bootstrap_ci(arr)
                        out.setdefault(fam, {}).setdefault(solver, {}).setdefault(
                            timeout, {}).setdefault(year, {})[cat] = {
                            'avg_incomplete_score': round(float(arr.mean()), 4),
                            'bootstrap_ci_low': round(lo, 4),
                            'bootstrap_ci_high': round(hi, 4),
                            'n_instances': len(arr),
                        }
    return out


def family_counts(bench_dir: str, years) -> Dict[str, int]:
    """Count instances per family across the test set (sanity check)."""
    counts: Dict[str, int] = defaultdict(int)
    for entry in os.listdir(bench_dir):
        p = os.path.join(bench_dir, entry)
        if not os.path.isdir(p):
            continue
        # match ms2020 / wms2021 / ...
        if not (entry.startswith('ms') or entry.startswith('wms')):
            continue
        yr_digits = entry.lstrip('msw')
        if not (yr_digits.isdigit() and yr_digits in years):
            continue
        for root, _, files in os.walk(p):
            for f in files:
                if f.endswith(('.cnf', '.wcnf')):
                    counts[classify(f)] += 1
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description='Per-family result breakdown')
    parser.add_argument('--hydra_json', required=True)
    parser.add_argument('--baseline_jsons', nargs='+', default=[])
    parser.add_argument('--out', default='./results/family_breakdown.json')
    parser.add_argument('--bench_dir', default='./benchmarks')
    parser.add_argument('--years', nargs='+',
                        default=['2020', '2021', '2022', '2023', '2024'])
    args = parser.parse_args()

    solvers = {}
    raw, _ = load_results(args.hydra_json); solvers['hydra'] = raw
    for p in args.baseline_jsons:
        stem = os.path.splitext(os.path.basename(p))[0]
        name = stem.replace('_results_raw', '').replace('_raw', '')
        raw, _ = load_results(p); solvers[name] = raw

    bk = compute_best_known(solvers)
    out = {
        'family_counts': family_counts(args.bench_dir, args.years),
        'breakdown':     breakdown(solvers, bk),
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Wrote {args.out}')
    print('\n=== Families discovered ===')
    for fam, n in sorted(out['family_counts'].items(), key=lambda kv: -kv[1]):
        print(f'  {fam:<25s} {n:>5d} instances')


if __name__ == '__main__':
    main()
