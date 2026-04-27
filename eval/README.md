# RELAY NeurIPS 2026 — Evaluation Pipeline

This directory holds everything that is *not* RELAY's model or solver code:
the code that turns trained RELAY weights and raw benchmark files into the
tables and figures that appear in the paper.

## Layout

```
eval/
├── plotting.py            # NeurIPS figure style (imported by make_figures.py)
├── baselines.py           # Thin wrappers around 5 competing solvers
├── run_baselines.py       # Parallel driver: baselines × full benchmark tree
├── run_ablation.py        # Ablation runner (11 named configs)
├── training_stability.py  # RELAY training-curve experiment (Fig 4 equiv.)
├── stat_tests.py          # Best-known, Wilcoxon, bootstrap CI
├── family_breakdown.py    # Per-problem-family aggregation
├── make_tables.py         # LaTeX tables (main, ablation, stats, families)
├── make_figures.py        # PDF + PNG figures (main, ablation, family, training)
└── README.md              # this file
```

## One-shot reproduction

From the repo root:

```bash
bash run_paper_campaign.sh
```

This runs, in order:

1. Pretrain RELAY on `benchmarks/train/` (50 epochs, ~8 GPU-hours)
2. Run 5 baselines (NuWLS, BandHS, SATLike, SPB, FourierSAT) on the full
   test tree at 60s + 300s timeouts (CPU-parallel, ~30 CPU-hours)
3. Run RELAY evaluation at 60s + 300s (~80 GPU-hours total)
4. Run 11-config ablation on WMS+2018 (~15 GPU-hours)
5. Run training-stability experiment (~2 GPU-hours)
6. Compute summary stats, per-family breakdown, emit all tables + figures

Each step can be skipped with a flag (e.g. `--skip-pretrain`), and all stages
produce self-contained JSON in `results/` so partial reruns are cheap.

## Benchmark shape

| Directory | Instances | Role |
|---|---|---|
| `benchmarks/train/` | 63 | Pretraining (WMS+2018 instances < 2MB) |
| `benchmarks/ms{2020..2024}/` | 262 / 155 / 179 / 179 / 216 | Unweighted test |
| `benchmarks/wms{2020..2024}/` | 243 / 151 / 197 / 160 / 229 | Weighted test |

MSE24 benchmarks come from the *anytime* track (matches MSE23 convention).

## Output shape

```
results/
├── hydra/                      # RELAY model outputs per timeout
│   ├── eval_60s/results_raw.json
│   └── eval_300s/results_raw.json
├── hydra_all_results.json      # Merged for cross-timeout analysis
├── baselines/
│   ├── nuwls_results_raw.json
│   ├── spb_results_raw.json
│   ├── satlike_results_raw.json
│   ├── bandhs_results_raw.json
│   └── foursat_results_raw.json
├── ablations/
│   ├── full_results_raw.json
│   ├── no_diffusion_results_raw.json
│   └── … (11 configs)
├── training_stability.json     # Per-epoch SWR curves
├── family_breakdown.json       # Per-family aggregates
├── stats.json                  # Bootstrap CIs + Wilcoxon p-values
└── logs/
    ├── pretrain_*.log
    ├── baselines_*.log
    └── hydra_eval_*.log

figures/
├── fig_main_scores.pdf         # Table 1 figure form — bar chart
├── fig_ablation_bars.pdf       # Ablation impact bars
├── fig_family_heatmap.pdf      # Per-family heatmap
├── fig_train_stability.pdf     # Training curves (Fig 4 equivalent)
├── table_main.tex              # Table 1 — LaTeX
├── table_ablation.tex          # Ablation table — LaTeX
├── table_stats.tex             # Paired Wilcoxon table — LaTeX
└── table_families.tex          # Per-family table — LaTeX
```

## Figure style

All figures go through `plotting.setup_style()`, which enforces:

* **No chart titles** (captions carry the description)
* **Black edges** on every mark (spines, bar edges, marker edges, colorbar frame)
* **Colorblind-safe** palette (Wong 2011) plus redundant marker shapes so
  B/W printouts stay readable
* **Legends** that never overlap data — inside-best auto placement with a
  `outside=True` fallback for saturated axes
* **TrueType** fonts embedded (NeurIPS font-check passes)
* **300 DPI PDF + PNG** output at the tight bbox

To reuse the style elsewhere:

```python
from eval.plotting import setup_style, new_fig, save_fig, style_cycle
setup_style()
fig, ax = new_fig()
# ... plot ...
save_fig(fig, 'figures/my_plot.pdf')  # also writes my_plot.png
```

## Baseline protocol

Each solver is called with its native CLI:

| Solver | Invocation | Track |
|---|---|---|
| NuWLS       | `./nuwls <wcnf> <seed>`                  | SLS  |
| BandHS      | `./BandHS <wcnf> <seed>`                 | SLS  |
| SATLike     | `./SATLike3.0 <wcnf>`                    | SLS  |
| SPB-MaxSAT  | `./SPB-MaxSAT <wcnf>`                    | SLS  |
| FourierSAT  | `python FourierSAT.py <wcnf> --ismaxsat 1 --timelimit <t>` | Continuous |

All solvers emit `o <cost>` anytime lines.  `baselines.py` streams stdout
with `select()` and tracks the minimum `o` value before the wall clock
expires.  Unavailable binaries (e.g. if a zip wasn't unpacked) return `inf`.

## Ablation configs

| Config            | Toggled off |
|---|---|
| `full`            | (control — all features on) |
| `no_diffusion`    | diffusion init + gradient guidance + population |
| `no_rich_graph`   | spectral PE, RW PE, Ricci rewiring, var-edges, centrality, hand features |
| `no_unsupervised` | pretraining (uses fresh random weights) |
| `no_spectral_pe`  | only Laplacian-PE |
| `no_rw_pe`        | only random-walk PE |
| `no_ricci`        | only Ricci curvature rewiring |
| `no_var_edges`    | only variable-interaction self-attention |
| `no_centrality`   | only clause centrality PE |
| `no_guidance`     | gradient guidance (keep diffusion + population) |
| `no_population`   | population (single sample from diffusion) |
| `no_consensus`    | consensus-decimation phase |

Fine-grained configs help when reviewers ask which of RELAY's many graph
enrichments is actually pulling the weight.

## Statistical tests

* **Best-known cost** for each instance is the minimum observed across all
  solvers in this run (following SGAT-MS §5.3).  This is conservative: if
  RELAY beats the previous best-known, its incomplete score exceeds 1.0.
* **Bootstrap 95% CIs** use 10,000 resamples of the per-instance score
  vector (seed 42).  Reported in `stats.json` and overlaid in figures.
* **Paired Wilcoxon** is used for RELAY vs each baseline (alternative
  "RELAY $>$ baseline").  Pairs are instances common to both solvers;
  ties are dropped (classic convention).

## Caveats for the May 2 submission

* **1 seed** for the main table (matches SGAT-MS's paper — 3 seeds was a
  stretch-goal that we dropped for compute).  CIs come from per-instance
  variance, not from seed variance.
* **300s timeout**: optional — if GPU time runs out, we ship 60s only and
  disclose this in §5.
* **NeuroSAT / GGNN training baselines**: the SGAT-MS paper's Fig 4 showed
  these fail to learn; we cite that rather than reproducing it.  RELAY's
  training curve is the positive control.
