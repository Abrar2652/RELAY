# RELAY-Bench: Datasheet

A datasheet for the RELAY-Bench dataset, following the structure of Gebru et al. (2021) and the NeurIPS 2026 D&B track guidelines.

---

## 1. Motivation

**For what purpose was the dataset created?**
RELAY-Bench was created to enable rigorous, reproducible comparison of incomplete Weighted MaxSAT solvers, and to provide a strong portfolio baseline that future neural and learned methods must beat. Single-solver comparisons against NuWLS (the previous SOTA) understate what off-the-shelf SLS solvers can collectively achieve; this dataset records the full per-instance cost matrix needed to expose that gap.

**Who created the dataset?**
Anonymous authors for the NeurIPS 2026 D&B Track (double-blind submission).

**Funding sources:** redacted for double-blind review.

---

## 2. Composition

**What do the instances represent?**
Each row in `relay_bench_runs.csv` is a (solver, timeout, year, category, instance, cost) tuple — the minimum incumbent cost reported by one of 8 SLS MaxSAT solvers on one MaxSAT Evaluation 2020-2024 instance under a 60s or 300s wall-clock budget.

**How many instances?**
- 31,056 solver runs (8 solvers × 1,941 instances × 2 timeouts; some instances have entries from fewer than all 8 solvers due to per-solver crashes — the matched analysis subset is 1,736 instances)
- 1,550 unique benchmark instances with metadata
- 1,367 instances with official MSE best-known reference cost
- 1,736 instances form the matched portfolio analysis subset

**Sampling strategy?**
The full MSE 2020-2024 incomplete track was attempted; instances with more than 10,000 clauses were filtered out at runtime to keep wall-clock feasible across $k\times$ parallel solver invocations. The 1,736 matched subset retains instances that all 8 solvers completed successfully and for which an official MSE best-known cost is published.

**What data does each instance consist of?**
Per (solver, timeout, instance):
- `solver` (str): one of {nuwls, bandhs, satlike, spb, foursat, nuwls_c, spb_c, nuwls_bandhs}
- `timeout_seconds` (int): 60 or 300
- `year` (int): 2020-2024
- `category` (str): ms (unweighted) or wms (weighted)
- `instance_name` (str): the .wcnf file basename
- `cost` (float): the best (minimum) cost reported before the deadline; +infinity if the solver returned no incumbent

Per instance:
- `n_vars`, `n_clauses`, `file_size_kb`, `family` (one of: clique, ramsey, scheduling, set-cover, etc.)

**Are there errors, sources of noise, or redundancies?**
- Solver costs at 60s are subject to single-seed (seed=1) stochastic variance.
- A small number of instance basenames collide across years (the same logical instance reused in different MSE editions); all entries are preserved with their year qualifier.
- 5 solvers were originally evaluated; 3 hybrid solvers (NuWLS-c, SPB-c, NuWLS-BandHS) were added later. Their bundled binaries are GNN-augmented variants that internally call a complete solver.

**Is any information missing?** Yes:
- Multi-seed runs are not provided (single seed per solver per instance).
- Anytime trajectories are not recorded; only the final incumbent at the deadline.
- Per-solver memory and CPU traces are not recorded.

**Are there labels?** No human labels; all values are automatic solver outputs.

**Is the data self-contained?** Yes (the cost matrix and metadata are self-contained). The raw `.wcnf` instance files are not redistributed by RELAY-Bench but are publicly available from the MaxSAT Evaluation series at `https://maxsat-evaluations.github.io/`. A representative sample of 34 raw `.wcnf` files (those <2 MB) is bundled for reviewer convenience.

**Confidential / sensitive?** None. Instances are synthetic combinatorial-optimization formulations from public sources.

---

## 3. Collection Process

**How was the data acquired?**
By executing each solver's bundled binary on each instance and parsing the MSE-standard `o <cost>` output protocol live. The minimum cost reported before the wall-clock deadline was recorded. Process groups were terminated by SIGKILL at the deadline.

**Who was involved?**
Automated scripts (`eval/run_baselines.py` + `eval/baselines.py`) using a 32-core or 48-core x86_64 server at 2.8 GHz, 256 GB RAM. No human annotation was involved.

**Over what timeframe was the data collected?**
April 2026, for the NeurIPS 2026 submission cycle.

**Ethical review?** Not applicable — no human-subject data.

**Was the dataset associated with previously published research?**
The MSE benchmark instances are from the official MaxSAT Evaluation series (2020-2024). The cost matrix in RELAY-Bench is original to this submission.

---

## 4. Preprocessing / Cleaning / Labeling

**Was any preprocessing done?**
- Instance filter: only instances with $\le 10{,}000$ clauses were retained (parses both legacy `p wcnf N C top` headers and the modern MSE 2022+ format with weight token `h` and counts in JSON-style comments).
- The matched subset retains 1,736 instances on which all 8 solvers report a cost and for which an official MSE best-known cost is available.

**Is the raw data preserved?** Per-solver per-instance per-timeout result JSONs are preserved alongside the flattened CSVs.

**Is the preprocessing software available?** Yes — `scripts/run_baselines.py` and `scripts/baselines.py` in the dataset bundle.

---

## 5. Uses

**Has the dataset been used yet?**
The RELAY-Bench paper (NeurIPS 2026 D&B submission) uses it to evaluate 8 single solvers, 4 parallel-portfolio configurations ($k=2,3,4,8$), and the Virtual Best Solver upper bound. Headline result: 3-solver parallel portfolio beats NuWLS by +2.0% absolute score with 241 wins / 0 losses on the matched subset at 60s.

**What other tasks?**
- Benchmarking new MaxSAT solvers (add a row to the cost matrix; compute relative VBS).
- Evaluating learned algorithm-selection methods (per-instance cost matrix is the supervision signal).
- Studying anytime-convergence behaviour (the `runtime_seconds` field is exported by the bundled scripts; for the deadline-final cost matrix we report `cost` only).
- Sanity-checking neural MaxSAT methods against a strong baseline portfolio.

**Tasks the dataset should NOT be used for:**
- Drawing conclusions about solvers' time-to-optimum (only deadline costs are reported, not anytime curves).
- Multi-seed CI estimation (only a single seed=1 run per solver per instance).
- Direct comparisons against papers that used different timeouts, hardware, or compilation flags — costs are sensitive to all three.

**Is there anything that could lead to unfair treatment?** No — the dataset evaluates algorithms, not humans.

---

## 6. Distribution

**Will the dataset be distributed to third parties?** Yes, publicly under CC-BY 4.0.

**How?** Zenodo deposit (DOI provided at submission; URL anonymized for double-blind review).

**When?** Upon paper acceptance; an anonymized Zenodo deposit is provided for review.

**License:** CC-BY 4.0. Bundled solver binaries inherit their upstream licenses (NuWLS: MIT; BandHS: MIT; SatLike: AIJ-2020 source; SPB-MaxSAT: IJCAI-2024 source; FourierSAT: AAAI-2020 source); see `scripts/baselines.py` for upstream pointers.

**IP-based or other restrictions?** No.

**Export controls or other regulations?** None known.

---

## 7. Maintenance

**Maintenance plan:**
The artifact is versioned. Version 1.0.0 accompanies the NeurIPS 2026 D&B submission. New MSE editions will be added as separate version increments. Issues will be tracked in the Zenodo deposit comments.

**Will older versions remain available?** Yes — Zenodo preserves all versions.

**Mechanism for community contributions?** Pull requests to a public Git mirror (released after acceptance).

---

## 8. Limitations and Biases

- **Solver-pool bias.** The 8 SLS solvers over-represent the SLS family. Complete solvers (CEGAR, MaxHS, EvalMaxSAT) and ILP-based solvers are not in the pool, which would change the VBS ceiling.
- **Year saturation.** MSE 2022-2024 instances tend to saturate the incomplete-score metric (NuWLS already achieves $\approx 1.0$). Per-year lift is concentrated in 2020-2021.
- **Single seed.** All runs use seed=1. Stochastic variance across seeds is not captured.
- **Hardware sensitivity.** Costs depend on hardware (32-core / 48-core x86 at 2.8 GHz). Different hardware will produce different numbers.
- **No anytime curves.** Only the deadline-final incumbent cost is recorded.

---

## 9. Author Statement

Authors are responsible for the dataset as described. We confirm:
- The data does not contain personal information.
- The dataset is released under CC-BY 4.0.
- Bundled solver binaries are redistributed under their respective upstream licenses.
- The dataset will be hosted on Zenodo with a permanent DOI.
