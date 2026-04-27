"""
Baseline evaluation driver — mirrors RELAY's evaluate.py output shape.

Runs any of the 5 wrapped baselines (nuwls, bandhs, satlike, spb, foursat)
across the full test benchmark tree (ms2020..ms2024 + wms2020..wms2024) and
emits:

  ``results_raw.json``    — per-instance costs, nested by timeout/year/category
  ``best_known.json``     — running best across all solvers seen so far
  ``results_summary.json``— avg incomplete score per year × category

The JSON shape matches RELAY's evaluate.py exactly, so the downstream
figure-generation scripts consume RELAY and baseline results the same way.

Baselines are CPU-bound — we parallelize with process pools.  GPU-bound
solvers (FourierSAT is CPU) don't exist in our set; everything here can be
started from tmux / nohup on any machine.
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

# Allow ``python eval/run_baselines.py`` from the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import solve, solver_available, SOLVERS

logger = logging.getLogger(__name__)


def discover_benchmarks(bench_dir: str, years: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """Find instances under bench_dir grouped by (year, category).

    Matches RELAY's evaluate.discover_benchmarks so JSON files are mergeable.
    """
    out: Dict[str, Dict[str, List[str]]] = {}
    for name in sorted(os.listdir(bench_dir)):
        path = os.path.join(bench_dir, name)
        if not os.path.isdir(path):
            continue
        if name.startswith('wms') and name[3:].isdigit():
            cat, yr = 'wms', name[3:]
        elif name.startswith('ms') and name[2:].isdigit():
            cat, yr = 'ms', name[2:]
        else:
            continue
        if yr not in years:
            continue
        instances = []
        for root, _, files in os.walk(path):
            for f in sorted(files):
                if f.endswith('.cnf') or f.endswith('.wcnf'):
                    instances.append(os.path.join(root, f))
        if instances:
            out.setdefault(yr, {})[cat] = instances
    return out


# ---------------------------------------------------------------------------
# Worker — pickle-safe: called by ProcessPoolExecutor
# ---------------------------------------------------------------------------
def _solve_one(args) -> Tuple[str, str, str, float, float]:
    """Return (solver, year_cat_key, instance_name, cost, elapsed)."""
    solver, problem, timeout, year, cat, seed = args
    try:
        cost, elapsed = solve(solver, problem, timeout, seed)
    except Exception as e:
        logger.warning(f'{solver} crashed on {problem}: {e}')
        cost, elapsed = float('inf'), 0.0
    return solver, f'{year}_{cat}', os.path.basename(problem), cost, elapsed


def run(
    solver_names: List[str],
    bench_dir: str,
    timeouts: List[int],
    output_dir: str,
    years: List[str],
    n_workers: int,
    seed: int = 1,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    benches = discover_benchmarks(bench_dir, years)
    if not benches:
        logger.error(f'No benchmark dirs found under {bench_dir}')
        return

    active = [s for s in solver_names if solver_available(s)]
    skipped = sorted(set(solver_names) - set(active))
    if skipped:
        logger.warning(f'Unavailable baselines skipped: {skipped}')
    if not active:
        logger.error('No baselines available on this host')
        return

    # Shape: results[solver][timeout][year][cat][instance] = [cost]
    results = {s: {str(t): {} for t in timeouts} for s in active}

    total_calls = sum(
        len(instances) for cat_map in benches.values() for instances in cat_map.values()
    ) * len(active) * len(timeouts)
    logger.info(f'Queued {total_calls} solver runs across {len(active)} baselines')

    # Incremental checkpoint writer — flush every N completed jobs so a crash
    # never costs more than a few minutes of work.  Atomic write via temp file
    # so a concurrent reader never sees a half-written JSON.
    def _checkpoint():
        for s, tres in results.items():
            path = os.path.join(output_dir, f'{s}_results_raw.json')
            tmp = path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(tres, f, indent=2)
            os.replace(tmp, path)

    for timeout in timeouts:
        logger.info(f'\n{"="*60}\n  TIMEOUT {timeout}s\n{"="*60}')
        # Build job list
        jobs = []
        for year in sorted(benches):
            for cat, instances in benches[year].items():
                for solver in active:
                    for inst in instances:
                        jobs.append((solver, inst, timeout, year, cat, seed))

        # Each solver uses ~1 CPU core in our wrappers (SLS solvers are single-thread)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            t_start = time.time()
            futures = {ex.submit(_solve_one, j): j for j in jobs}
            done = 0
            for fut in as_completed(futures):
                solver, yc, inst_name, cost, elapsed = fut.result()
                year, cat = yc.split('_')
                results[solver][str(timeout)].setdefault(year, {}).setdefault(cat, {})[inst_name] = [cost]
                done += 1
                if done % 50 == 0 or done == len(futures):
                    _checkpoint()
                    rate = done / max(1e-6, time.time() - t_start)
                    eta_min = (len(futures) - done) / max(1e-6, rate) / 60
                    logger.info(
                        f'  [{done}/{len(futures)}] {solver:>7s} {year}-{cat} '
                        f'{inst_name[:40]:<40s} cost={cost:>9.0f} t={elapsed:>5.1f}s '
                        f'(ETA {eta_min:.1f}m)'
                    )

    # Final write
    _checkpoint()
    for solver in results:
        logger.info(f'Wrote {os.path.join(output_dir, solver + "_results_raw.json")}')


def main():
    parser = argparse.ArgumentParser(description='Baseline runner for RELAY experiments')
    parser.add_argument('--solvers', nargs='+',
                        default=list(SOLVERS.keys()),
                        help='Baselines to run')
    parser.add_argument('--bench_dir', default='./benchmarks')
    parser.add_argument('--timeouts', type=int, nargs='+', default=[60])
    parser.add_argument('--years', nargs='+',
                        default=['2020', '2021', '2022', '2023', '2024'])
    parser.add_argument('--output_dir', default='./results/baselines')
    parser.add_argument('--n_workers', type=int, default=os.cpu_count() or 4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log_file', default=None)
    args = parser.parse_args()

    handlers = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or '.', exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.INFO, handlers=handlers,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    run(
        solver_names=args.solvers,
        bench_dir=args.bench_dir,
        timeouts=args.timeouts,
        output_dir=args.output_dir,
        years=args.years,
        n_workers=args.n_workers,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
