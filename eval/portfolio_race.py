"""
Portfolio Racing: a short probe of K SLS solvers, then commit full remaining
budget to the winner.  Beats any single solver on instances where the winning
solver is not the user's default choice.

Method
------
Given budget T seconds and K solvers:
  1. Run each solver in parallel for T_probe = T * probe_frac seconds.
  2. Collect best cost seen per solver during probing.
  3. Commit the best-performing solver for the remaining T_commit = T - T_probe.
  4. Return min cost across probe + commit.

A ``parallel=True`` flag controls whether all K probes run concurrently
(default; uses K CPUs) or sequentially time-slicing (K * T_probe, so caller
must budget accordingly).  Sequential is fair vs single-solver baselines
running on 1 CPU.

Implementation uses ``baselines.solve`` per-solver, so binaries are reused
verbatim.  The "o <cost>" parsing tracks incumbent cost live; we do not rely
on the solver returning early.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass
from typing import Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'eval'))

import baselines

logger = logging.getLogger(__name__)


def _run_solver(name: str, problem: str, timeout: float, seed: int):
    return baselines.solve(name, problem, timeout, seed)


def portfolio_race(
    problem: str,
    timeout: float,
    solvers: list[str],
    probe_frac: float = 0.2,
    parallel: bool = True,
    seed: int = 1,
) -> tuple[float, float, str]:
    """Return (best_cost, elapsed, winner_name).

    If parallel=True: probe all K in parallel for T_probe, best-cost winner
    gets T_commit alone.  If parallel=False: run each probe sequentially for
    T_probe; total probe budget = K * T_probe (caller must set timeout to
    reflect this if fair-budget).
    """
    t0 = time.perf_counter()
    T_probe = max(2.0, timeout * probe_frac)
    T_commit = max(1.0, timeout - T_probe)

    probe_results: dict[str, float] = {s: float('inf') for s in solvers}

    if parallel:
        with ProcessPoolExecutor(max_workers=len(solvers)) as ex:
            futures = {
                ex.submit(_run_solver, s, problem, T_probe, seed): s
                for s in solvers
            }
            for fut in futures:
                try:
                    cost, _ = fut.result(timeout=T_probe + 10)
                    probe_results[futures[fut]] = cost
                except Exception as e:
                    logger.warning(f'{futures[fut]} probe failed: {e}')
    else:
        for s in solvers:
            try:
                cost, _ = _run_solver(s, problem, T_probe, seed)
                probe_results[s] = cost
            except Exception as e:
                logger.warning(f'{s} sequential probe failed: {e}')

    # Pick winner: smallest cost
    winner = min(solvers, key=lambda s: probe_results[s])
    winner_probe_cost = probe_results[winner]

    # Commit full budget to winner
    commit_cost, _ = _run_solver(winner, problem, T_commit, seed)

    best = min(winner_probe_cost, commit_cost)
    elapsed = time.perf_counter() - t0
    return best, elapsed, winner


def _peek_clause_count(path):
    try:
        with open(path, 'r', errors='replace') as f:
            ncls_json = None
            for line in f:
                s = line.strip()
                if s.startswith('p ') and len(s.split()) >= 4:
                    return int(s.split()[3])
                if s.startswith('c'):
                    if '"ncls"' in s:
                        try:
                            part = s.split('"ncls"', 1)[1]
                            num = part.split(':', 1)[1].split(',', 1)[0].strip()
                            ncls_json = int(num)
                        except Exception: pass
                    continue
                return ncls_json
    except Exception: return None
    return None


def discover(bench_dir, years):
    out = {}
    for name in sorted(os.listdir(bench_dir)):
        path = os.path.join(bench_dir, name)
        if not os.path.isdir(path): continue
        if name.startswith('wms') and name[3:].isdigit():
            cat, yr = 'wms', name[3:]
        elif name.startswith('ms') and name[2:].isdigit():
            cat, yr = 'ms', name[2:]
        else: continue
        if yr not in years: continue
        insts = []
        for root, _, files in os.walk(path):
            for f in sorted(files):
                if f.endswith(('.cnf', '.wcnf')):
                    insts.append(os.path.join(root, f))
        if insts:
            out.setdefault(yr, {})[cat] = insts
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bench_dir', default='./benchmarks')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--timeout', type=float, default=60.0)
    p.add_argument('--probe_frac', type=float, default=0.2)
    p.add_argument('--solvers', nargs='+',
                   default=['nuwls', 'satlike', 'spb', 'bandhs'])
    p.add_argument('--years', nargs='+',
                   default=['2020','2021','2022','2023','2024'])
    p.add_argument('--max_clauses', type=int, default=10000)
    p.add_argument('--seeds', type=int, default=1)
    p.add_argument('--seed_offset', type=int, default=0)
    p.add_argument('--log_file', default=None)
    args = p.parse_args()

    handlers = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or '.', exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%H:%M:%S')

    bench = discover(args.bench_dir, args.years)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'results_raw.json')
    winner_path = os.path.join(args.output_dir, 'winners.json')

    results = {}
    winners = {}
    if os.path.exists(out_path):
        try: results = json.load(open(out_path))
        except Exception: pass
    if os.path.exists(winner_path):
        try: winners = json.load(open(winner_path))
        except Exception: pass

    t_key = str(int(args.timeout))
    results.setdefault(t_key, {})
    winners.setdefault(t_key, {})

    for year in sorted(bench):
        results[t_key].setdefault(year, {})
        winners[t_key].setdefault(year, {})
        for cat in sorted(bench[year]):
            results[t_key][year].setdefault(cat, {})
            winners[t_key][year].setdefault(cat, {})
            for inst in bench[year][cat]:
                iname = os.path.basename(inst)
                nc = _peek_clause_count(inst)
                if nc is not None and nc > args.max_clauses: continue
                prior = results[t_key][year][cat].get(iname)
                if isinstance(prior, list) and len(prior) >= args.seeds:
                    continue

                seed_costs = []
                win_seeds = []
                for s_idx in range(args.seeds):
                    seed = args.seed_offset + s_idx
                    try:
                        cost, el, winner = portfolio_race(
                            problem=inst, timeout=args.timeout,
                            solvers=args.solvers, probe_frac=args.probe_frac,
                            parallel=True, seed=seed + 1,
                        )
                        seed_costs.append(cost)
                        win_seeds.append(winner)
                        logger.info(f'  {iname:<40s} seed={seed} cost={cost:>12.1f} '
                                    f'winner={winner:>8s}  t={el:.1f}s')
                    except Exception as e:
                        logger.error(f'  {iname} seed={seed} FAILED {e}')
                        seed_costs.append(float('inf'))
                        win_seeds.append('failed')
                results[t_key][year][cat][iname] = seed_costs
                winners[t_key][year][cat][iname] = win_seeds

                # checkpoint
                for p, d in [(out_path, results), (winner_path, winners)]:
                    tmp = p + '.tmp'
                    with open(tmp, 'w') as f: json.dump(d, f)
                    os.replace(tmp, p)

    logger.info(f'portfolio_race complete -> {out_path}')


if __name__ == '__main__':
    main()
