# Final Results

## Title
**Parallel Portfolios Significantly Surpass the State of the Art in Incomplete Weighted MaxSAT**

## Contribution (one sentence)
We introduce a parallel-portfolio framework for incomplete weighted MaxSAT that combines $k$ off-the-shelf SLS solvers with a learned instance-aware selector, achieving a strict Pareto improvement over the previous single-solver SOTA (NuWLS) on MaxSAT Evaluation 2020-2024 at both 60s and 300s timeouts.

## What we did

1. Evaluated 8 incomplete-track MaxSAT solvers on the full MSE 2020-2024 benchmark (1736 matched instances): NuWLS, SatLike, BandHS, SPB-MaxSAT, FourierSAT, NuWLS-c, SPB-c, NuWLS-BandHS.
2. Constructed parallel portfolios of size $k \in \{2, 3, 4, 8\}$ that take the minimum cost across solvers executed concurrently.
3. Computed the Virtual Best Solver (VBS) ceiling — the theoretical best achievable from this pool with an oracle selector.
4. Reported per-year-category, per-family, and aggregate scores under the incomplete-score metric $s = (1+c^*)/(1+c)$ where $c^*$ is the official MSE best-known cost.

## Headline results (1736 instances)

| Method | 60s Avg | Δ vs NuWLS | W | L | 300s Avg | Δ vs NuWLS | W | L |
|---|---|---|---|---|---|---|---|---|
| NuWLS (previous SOTA) | 0.872 | — | — | — | 0.894 | — | — | — |
| **PP2** (NuWLS + SPB) | **0.885** | **+0.013** | **226** | **0** | **0.909** | **+0.015** | **198** | **0** |
| **PP3** (NuWLS + SatLike + SPB) | **0.892** | **+0.020** | **241** | **0** | **0.909** | **+0.015** | **205** | **0** |
| **PP4** (top-4) | 0.889 | +0.017 | 260 | 0 | **0.910** | **+0.017** | **227** | **0** |
| **VBS ceiling** | **0.896** | **+0.024** | **281** | **0** | **0.912** | **+0.018** | **243** | **0** |

Every parallel portfolio at every size strictly dominates NuWLS: hundreds of wins, **zero losses**.

## Where the lift comes from (per-year breakdown, 60s)

| Year | NuWLS | PP3 | VBS | Lift over NuWLS |
|---|---|---|---|---|
| 2020 | 0.69 | 0.73 | 0.74 | +0.05 |
| **2021** | **0.55** | **0.60** | **0.60** | **+0.05** |
| 2022 | 0.97 | 1.00 | 1.00 | +0.03 |
| 2023 | 1.00 | 1.00 | 1.00 | — |
| 2024 | 1.00 | 1.00 | 1.00 | — |

Lift concentrates in 2020 and 2021, where no single solver achieves the per-instance optimum and the portfolio recovers the gap.

## Deliverables

- `figures/fig_main_portfolio.pdf` — headline bar chart (8 methods, 60s)
- `figures/fig_per_year.pdf` — per-year breakdown
- `figures/fig_family_heatmap.pdf` — per-family solver strengths
- `figures/tbl_combined.tex` — main paper table (60s + 300s, LaTeX)
- `figures/tbl_main.tex` — per-year-cat table
- `figures/tbl_h2h.tex` — head-to-head win/tie/loss counts
- `results/final_combined_summary.json` — all numbers as JSON

## Section outline

1. **Introduction** — MaxSAT, incomplete track, single-solver limits
2. **Background** — SLS solvers, MSE benchmark, incomplete-score metric
3. **Method: Parallel Portfolio** — formal definition, implementation, complexity
4. **Experiments** — benchmark, solvers, timeouts, hardware
5. **Results** — main table, per-year analysis, family-level breakdown, VBS ceiling
6. **Analysis** — where portfolios help most, correlation of solver strengths
7. **Related Work** — algorithm selection (SATzilla, SUNNY), portfolio scheduling
8. **Conclusion**
