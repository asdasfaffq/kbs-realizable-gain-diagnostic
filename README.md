# Reproducibility package — *A Knowledge-Based Realizable-Gain Criterion for Guarded Deployment of LLM-Designed Heuristics*

This repository contains the code and result artifacts needed to reproduce the central,
non-LLM claims of the paper. All scripts here are **deterministic** (fixed seeds, no API keys,
no network except the one-time ASlib data download), so a reviewer can re-run them on a CPU in
minutes and obtain the numbers reported in the paper.

> Scope. This package covers the paper's *methodological* core: the realizable-gain diagnostic
> and its validation. The LLM-driven heuristic-*generation* scripts are not included because they
> require model API access and are non-deterministic; the heuristics they produced are used here
> only as fixed, transcribed baselines (see *Provenance* below).

## Layout

```
diagnostics/     the realizable-gain diagnostic pipeline (deterministic, no LLM)
  diagnostic_scale.py       34-configuration sweep: G_str vs. G_cv vs. held-out outcome
  select_probe.py           bin-packing panel, features, probe selector  (imported as BP)
  knapsack_select.py        knapsack panel + Pisinger instance classes    (KN)
  tsp_select.py             TSP construction panel                        (TP)
  tsp_construct_audit.py    TSP audit harness                             (TPC)
  sched_select.py           weighted-tardiness scheduling panel           (SC)
  probe_ood.py              online bin-packing harness (lower bound, packing loop)
  calib_analysis.py         calibration, non-inferiority (TOST), CVaR tail risk
  lcb_analysis.py           certified build rule: one-sided LCB (false-build control) + cost-sensitivity + tail-robust gain
  diagnostic_scale_results.json   saved output (so calib_analysis runs without a re-sweep)
simulation/
  controlled_sim.py         controlled study: why the classical gap over-predicts and G_cv does not
aslib/
  aslib_diagnostic.py       external validation on standard ASlib algorithm-selection scenarios
results/                    saved outputs for inspection without re-running
```

## Requirements

Python 3.10+ and:

```
pip install -r requirements.txt      # numpy, scipy, scikit-learn
```

## Reproducing each claim

| Paper result | Command | Expected output |
|---|---|---|
| **34-config diagnostic sweep** (Tables: confusion matrices; $\hat G_{\mathrm{cv}}$ precision 1.00 / recall 0.50, $G_{\mathrm{str}}$ over-predicts with 3 FP) | `cd diagnostics && python diagnostic_scale.py` | per-config `Gstr/Gcv/pred/actual` + confusion matrices; writes `diagnostic_scale_results.json` |
| **Calibration + tail risk** ($\hat G_{\mathrm{cv}}$ rank/sign-consistent, Spearman 0.90, but not numerically calibrated; scheduling non-inferiority FAIL, CVaR$_{0.95}$ up to $1.6\times10^4$) | `cd diagnostics && python calib_analysis.py` | consistency check (0 mismatches vs. the saved sweep), OLS slope/intercept, Spearman, per-config scheduling CVaR + non-inferiority |
| **Certified build rule** (one-sided LCB controls false builds: 0 false builds at every threshold, but conservative; the significance rule stays operative) | `cd diagnostics && python lcb_analysis.py` | false-build / false-retain rates of `LCB>tau` vs. the significance rule; CVaR-penalized tail-robust gain on scheduling |
| **Controlled simulation** (classical gap false-positive rate 1.00 vs. $\hat G_{\mathrm{cv}}$ 0.04 over the no-build conditions) | `cd simulation && python controlled_sim.py` | grid over feature identifiability × tail × sample size, with a known ground-truth realizable gain |
| **ASlib external validation** (on curated benchmarks the raw VBS$-$SBS gap does not over-predict) | see `aslib/` below | per-scenario confusion + PPV; needs the ASlib data (one-time clone) |

`bash reproduce.sh` runs the three self-contained studies (everything except the ASlib clone).

### ASlib external validation (one-time data download)

```
git clone https://github.com/coseal/aslib_data
export ASLIB_ROOT=$PWD/aslib_data
cd aslib && python aslib_diagnostic.py
```

The scenario list is pre-registered inside the script (`SCENARIOS`).

## Determinism and consistency

`calib_analysis.py` begins by re-running the exact diagnostic pipeline and asserting that it
reproduces `diagnostic_scale_results.json` (it reports `0 aggregate mismatches`), so the
per-instance analyses it adds are provably from the same fixed-seed computation as the sweep.

## Provenance of the audited heuristics

The published LLM-designed heuristics used as fixed baselines (FunSearch, EoH, ReEvo variants)
are transcribed verbatim from the authors' official public repositories and are the property of
their respective authors; they are included here only to make the audit reproducible. OR-Library
instances are the standard Beasley files. The knapsack optimum uses exact dynamic programming.

## License

Code in this repository is released under the MIT License (see `LICENSE`). This license covers
the code we wrote; transcribed third-party heuristics remain under their original licenses.
