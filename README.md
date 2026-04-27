# RELAY: Parallel Portfolio Racing for Incomplete Weighted MaxSAT

Implementation and experiments for a parallel-portfolio framework on MaxSAT Evaluation 2020-2024.  The portfolio combines $k$ off-the-shelf SLS solvers executed concurrently and reports the minimum cost per instance, yielding a strict Pareto improvement over any single solver.

---

## Repository Layout

```
# Portfolio / evaluation
eval/baselines.py             — 8 SLS solver wrappers with shared interface
eval/run_baselines.py         — Parallel CPU driver with incremental checkpointing
eval/make_final_artifacts.py  — Main figures + LaTeX tables
eval/make_combined_table.py   — Combined 60s + 300s paper table
eval/make_baseline_figures.py — Grouped-bar comparison figure
eval/make_family_heatmap.py   — Per-family score heatmap
eval/paper_analysis.py        — Score + VBS regeneration from raw JSON
eval/relay_selector.py       — Learned algorithm selector (Random Forest)
eval/portfolio_race.py        — Per-instance probe-then-commit portfolio

# Baseline binaries (bundled)
baselines/NuWLS/                       — NuWLS (AAAI 2023) + NuWLS-c (MSE 2024 hybrid)
baselines/BandHS/                      — BandHS + NuWLS-BandHS
baselines/SATLike3.0/                  — SATLike 3.0 (AIJ 2020)
baselines/SPB-MaxSAT/                  — SPB-MaxSAT + SPB-c hybrid
baselines/FourierSAT/                  — FourierSAT (AAAI 2020)
# RC2 via pysat.examples.rc2

# Benchmarks
benchmarks/ms20{20-24}/          — Unweighted MSE 20{20-24}
benchmarks/wms20{20-24}/         — Weighted MSE 20{20-24}

# Results
results/baselines/               — 5 baseline solvers, 60s
results/baselines_300s/          — 5 baseline solvers, 300s
results/baselines_new4/          — 3 additional solvers (nuwls_c, spb_c, nuwls_bandhs), 60s
results/baselines_new4_300s/     — 3 additional solvers, 300s
results/mse_official_best_known.json  — MSE official best-known per instance
results/final_combined_summary.json   — Paper-ready summary JSON

# Figures / tables
figures/fig_main_portfolio.pdf   — Headline bar chart
figures/fig_per_year.pdf         — Per-year breakdown
figures/fig_family_heatmap.pdf   — Per-family scores
```

---

## Installation

```bash
pip install python-sat scikit-learn numpy matplotlib
# Baseline binaries ship pre-built in baselines/
```

---

## Reproducing the Headline Numbers

```bash
# 1. Run 5 baseline solvers (60s, 1941 instances, ~4h on 32 cores)
python eval/run_baselines.py --solvers nuwls bandhs satlike spb foursat \
  --bench_dir ./benchmarks --timeouts 60 \
  --output_dir ./results/baselines --n_workers 32

# 2. Run 3 hybrid solvers
python eval/run_baselines.py --solvers nuwls_c spb_c nuwls_bandhs \
  --bench_dir ./benchmarks --timeouts 60 \
  --output_dir ./results/baselines_new4 --n_workers 48

# 3. Repeat both with --timeouts 300

# 4. Generate tables and figures
python eval/make_combined_table.py
python eval/make_final_artifacts.py
python eval/make_baseline_figures.py
python eval/make_family_heatmap.py

```

---

## Input Format

Standard DIMACS CNF (`.cnf`) and Weighted DIMACS WCNF (`.wcnf`).  Both the legacy `p wcnf N C top`-header format and the modern MSE 2022+ format (weight `h` for hard clauses, counts embedded in comment JSON) are supported by the parser and the instance filter.

---

## Learned Algorithm Selection

For configurations that prefer a single-solver compute budget, `eval/relay_selector.py` trains a Random Forest over 12 instance features (size, weight distribution, unit fraction, etc.) to predict the best solver per instance.  A confidence gate defaults to NuWLS when the classifier is uncertain.  5-fold cross-validation on the matched set achieves a modest lift over NuWLS at single-solver cost; the parallel portfolio variants above remain the stronger option when multiple cores are available.

---

## Notes

- Parallel portfolios assume $k$ cores available; wall-clock budget is the single-solver timeout, not $k\times$ it.
- Hardware used for the numbers in this repository: 32-core / 48-core AMD Epyc, 256 GB RAM.
- All raw per-instance results are preserved in the `results/` JSONs for independent analysis.
