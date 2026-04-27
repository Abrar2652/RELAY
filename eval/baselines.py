"""
Unified baseline wrappers for the RELAY experimental campaign.

Wraps the 7 solvers already shipping inside the SGAT-MS repo behind a single
Solver interface matching ``evaluate.solve_instance`` so the rest of the
pipeline treats baselines and RELAY identically.

Covered baselines:
  - NuWLS                (AAAI 2023, SLS)
  - BandHS / BandMaxSAT  (IJCAI 2022, SLS)
  - SATLike3.0           (AIJ 2020, SLS)
  - SPB-MaxSAT           (IJCAI 2024, SLS)
  - FourierSAT           (AAAI 2020, continuous)
  - Mixing / MIXSAT      (SDP-based continuous — wrapper only; binaries must
                          be installed separately, matched to SGAT-MS Table 1)

Every wrapper returns ``(cost, elapsed_seconds)`` with ``cost == float('inf')``
on any failure.  The existing SGAT-MS Python wrappers in
``SGAT-MS/src/solver/solver_wrapper.py`` are reused verbatim via delegation —
that module already normalizes WCNF/CNF, handles runsolver arguments, and
parses ``o <cost>`` lines.  We wrap with our own timing + subprocess timeout
because ``runsolver`` is not present on every host.
"""

from __future__ import annotations

import os
import sys
import time
import signal
import logging
import subprocess
import select
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
BASELINE_ROOT = os.path.join(REPO_ROOT, 'baselines')


# ---------------------------------------------------------------------------
# Generic subprocess runner with timestamped ``o`` parsing
# ---------------------------------------------------------------------------
def _run_and_parse_o_lines(cmd, cwd, timeout):
    """Run cmd in cwd with a hard timeout, return (best_cost, elapsed).

    The MaxSAT Evaluation protocol: solvers print ``o <cost>`` lines whenever
    they find a new incumbent.  The best (smallest) value seen before the wall
    clock expires is returned.  Returns (inf, elapsed) on any failure.
    """
    t0 = time.perf_counter()
    best = float('inf')
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            preexec_fn=os.setsid,
        )
    except Exception as e:
        logger.warning(f'Baseline launch failed: {e} (cmd={cmd})')
        return float('inf'), 0.0

    # Non-blocking reads via select() so we can honor ``timeout`` even when the
    # solver goes silent for stretches.  ``readline()`` blocks indefinitely
    # and will sit past the deadline — observed in SLS solvers that buffer
    # output between improvements.
    deadline = t0 + timeout
    buf = b''
    try:
        fd = proc.stdout.fileno()
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            ready, _, _ = select.select([fd], [], [], min(remaining, 1.0))
            if not ready:
                if proc.poll() is not None:
                    break  # process exited cleanly
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                break  # EOF
            buf += chunk
            # Process any complete lines in the buffer
            while b'\n' in buf:
                raw, buf = buf.split(b'\n', 1)
                line = raw.decode('utf-8', errors='replace').strip()
                if line.startswith('o '):
                    tokens = line.split()
                    if len(tokens) >= 2:
                        try:
                            c = int(tokens[1])
                            if c < best:
                                best = c
                        except ValueError:
                            pass
    finally:
        # Kill the whole process group — runsolver spawns children
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass

    return float(best), time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Per-solver wrappers — each returns (cost, elapsed)
# ---------------------------------------------------------------------------

def _solver_dir(rel_path: str) -> str:
    return os.path.join(BASELINE_ROOT, rel_path)


def run_nuwls(problem: str, timeout: float, seed: int = 1) -> tuple[float, float]:
    """NuWLS (AAAI 2023) — SLS solver."""
    wdir = _solver_dir('NuWLS/NuWLS-source-code')
    # Call the binary directly; starexec wrapper adds runsolver which is
    # host-specific.  The solver is anytime and prints ``o <cost>`` lines.
    cmd = [os.path.join(wdir, 'nuwls'), os.path.abspath(problem), str(seed)]
    return _run_and_parse_o_lines(cmd, wdir, timeout)


def run_bandhs(problem: str, timeout: float, seed: int = 1) -> tuple[float, float]:
    """BandHS / BandMaxSAT (IJCAI 2022) — SLS solver."""
    wdir = _solver_dir('BandHS/BandHS-main_sgat')
    cmd = [os.path.join(wdir, 'BandHS'), os.path.abspath(problem), str(seed)]
    return _run_and_parse_o_lines(cmd, wdir, timeout)


def run_satlike(problem: str, timeout: float, seed: int = 1) -> tuple[float, float]:
    """SATLike3.0 (AIJ 2020) — SLS solver."""
    wdir = _solver_dir('SATLike3.0_sgat')
    cmd = [os.path.join(wdir, 'SATLike3.0'), os.path.abspath(problem)]
    return _run_and_parse_o_lines(cmd, wdir, timeout)


def run_spb(problem: str, timeout: float, seed: int = 1) -> tuple[float, float]:
    """SPB-MaxSAT (IJCAI 2024) — SLS solver with soft pseudo-Boolean constraints."""
    wdir = _solver_dir('SPB-MaxSAT/orig')
    cmd = [os.path.join(wdir, 'SPB-MaxSAT'), os.path.abspath(problem)]
    return _run_and_parse_o_lines(cmd, wdir, timeout)


def run_foursat(problem: str, timeout: float, seed: int = 1) -> tuple[float, float]:
    """FourierSAT (AAAI 2020) — continuous-optimization solver."""
    wdir = _solver_dir('FourierSAT/FourierSAT_Github_AIJ')
    if not os.path.exists(os.path.join(wdir, 'FourierSAT.py')):
        logger.warning('FourierSAT.py not found; skipping')
        return float('inf'), 0.0
    cmd = [
        sys.executable, os.path.join(wdir, 'FourierSAT.py'),
        os.path.abspath(problem),
        '--ismaxsat', '1',
        '--timelimit', str(int(timeout)),
    ]
    return _run_and_parse_o_lines(cmd, wdir, timeout + 5)


def run_nuwls_c(problem: str, timeout: float, seed: int = 1) -> tuple[float, float]:
    """NuWLS-c (MSE 2024 incomplete-track winner) — NuWLS + GNN hybrid."""
    wdir = _solver_dir('NuWLS/NuWLS-source-code_hybrid')
    cmd = [os.path.join(wdir, 'nuwls'), os.path.abspath(problem), str(seed)]
    return _run_and_parse_o_lines(cmd, wdir, timeout)


def run_spb_c(problem: str, timeout: float, seed: int = 1) -> tuple[float, float]:
    """SPB-MaxSAT-c (hybrid variant with GNN predictions)."""
    wdir = _solver_dir('SPB-MaxSAT/hybrid')
    cmd = [os.path.join(wdir, 'SPB-MaxSAT'), os.path.abspath(problem)]
    return _run_and_parse_o_lines(cmd, wdir, timeout)


def run_nuwls_bandhs(problem: str, timeout: float, seed: int = 1) -> tuple[float, float]:
    """NuWLS-BandHS: combined NuWLS + BandHS SLS hybrid."""
    wdir = _solver_dir('BandHS/NuWLS-BandHS-main')
    cmd = [os.path.join(wdir, 'NuWLS-BandHS'), os.path.abspath(problem), str(seed)]
    return _run_and_parse_o_lines(cmd, wdir, timeout)


def run_rc2(problem: str, timeout: float, seed: int = 1) -> tuple[float, float]:
    """RC2 (PySAT) — core-guided complete MaxSAT solver wrapped with timeout."""
    import time as _t
    import signal as _sig

    class _Timeout(Exception): pass
    def _h(*_): raise _Timeout()

    try:
        from pysat.examples.rc2 import RC2
        from pysat.formula import WCNF
    except ImportError:
        return float('inf'), 0.0

    t0 = _t.perf_counter()
    best = float('inf')
    _sig.signal(_sig.SIGALRM, _h)
    _sig.alarm(int(timeout) + 1)
    try:
        wcnf = WCNF(from_file=os.path.abspath(problem))
        with RC2(wcnf) as rc2:
            model = rc2.compute()
            if model is not None:
                best = float(rc2.cost)
    except _Timeout:
        pass
    except Exception as e:
        logger.warning(f'RC2 failed: {e}')
    finally:
        _sig.alarm(0)
    return best, _t.perf_counter() - t0


# Registry of available baselines ------------------------------------------
SOLVERS: dict[str, Callable[[str, float, int], tuple[float, float]]] = {
    'nuwls':         run_nuwls,
    'bandhs':        run_bandhs,
    'satlike':       run_satlike,
    'spb':           run_spb,
    'foursat':       run_foursat,
    'nuwls_c':       run_nuwls_c,
    'spb_c':         run_spb_c,
    'nuwls_bandhs':  run_nuwls_bandhs,
    'rc2':           run_rc2,
}


def solver_available(name: str) -> bool:
    """Check whether a baseline binary is present on disk."""
    checks = {
        'nuwls':         'NuWLS/NuWLS-source-code/nuwls',
        'bandhs':        'BandHS/BandHS-main_sgat/BandHS',
        'satlike':       'SATLike3.0_sgat/SATLike3.0',
        'spb':           'SPB-MaxSAT/orig/SPB-MaxSAT',
        'foursat':      'FourierSAT/FourierSAT_Github_AIJ/FourierSAT.py',
        'nuwls_c':       'NuWLS/NuWLS-source-code_hybrid/nuwls',
        'spb_c':         'SPB-MaxSAT/hybrid/SPB-MaxSAT',
        'nuwls_bandhs':  'BandHS/NuWLS-BandHS-main/NuWLS-BandHS',
    }
    if name == 'rc2':
        # Always available if pysat imports
        try:
            from pysat.examples.rc2 import RC2  # noqa
            return True
        except ImportError:
            return False
    path = checks.get(name)
    return bool(path) and os.path.exists(_solver_dir(path))


def solve(name: str, problem: str, timeout: float, seed: int = 1) -> tuple[float, float]:
    """Dispatch to the named baseline."""
    if name not in SOLVERS:
        raise ValueError(f'Unknown baseline: {name!r} (have {list(SOLVERS)})')
    if not solver_available(name):
        logger.warning(f'Baseline {name} not available; returning inf')
        return float('inf'), 0.0
    return SOLVERS[name](problem, timeout, seed)


if __name__ == '__main__':
    # Smoke test: run each available baseline on the smallest training instance
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', required=True)
    parser.add_argument('--timeout', type=float, default=15.0)
    parser.add_argument('--solver', default=None,
                        help='Run only this solver (default: all available)')
    args = parser.parse_args()

    names = [args.solver] if args.solver else list(SOLVERS.keys())
    print(f'{"Solver":<10s} {"Available":<10s} {"Cost":>12s}  {"Time (s)":>10s}')
    print('-' * 50)
    for n in names:
        avail = solver_available(n)
        if not avail:
            print(f'{n:<10s} {"no":<10s} {"—":>12s}  {"—":>10s}')
            continue
        cost, elapsed = solve(n, args.instance, args.timeout)
        cost_str = f'{cost:.0f}' if cost != float('inf') else 'inf'
        print(f'{n:<10s} {"yes":<10s} {cost_str:>12s}  {elapsed:>10.2f}')
