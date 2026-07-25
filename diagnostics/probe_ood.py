#!/usr/bin/env python3
"""STEP 1 — OOD gate probe (NO LLM). Decides honestly whether the OOD-robustness
battleground is real BEFORE spending any LLM budget.

Faithful reproduction of FunSearch's exact online_binpack harness (full bins array of
num_items capacity slots; priority computed on the VALID subset; argmax; position-dependent
terms preserved). The 3 "hot" heuristics are transcribed VERBATIM from official repos:
  - FunSearch-Weibull : google-deepmind/funsearch  bin_packing/bin_packing.ipynb (Weibull cell)
                        (also ai4co/reevo prompts/bpp_online/seed_func.txt "best known")
  - FunSearch-OR      : same notebook, OR-datasets cell (integer-threshold heuristic)
  - EoH-best          : FeiLiu36/EoH docs/Results/Prob1_OnlineBinPacking/run1/
                        population_generation_20.json  (best individual, objective 0.00624)
Classical SOTA panel (5): First/Best/Worst/Next-Fit + Harmonic.

Pre-declared decision rule (committed before seeing numbers):
  Q1: does any hot heuristic degrade >=1% excess WORSE than Best-Fit on >=1 OOD distribution?
  Q2: is there any distribution where some heuristic robustly (Wilcoxon p<0.05) beats Best-Fit?
  Q1 yes & Q2 yes -> strict rank-#1 achievable; pursue.
  Q1 yes & Q2 no  -> aim for tied-#1-with-BF + strictly-beats-all-hot (robustness story).
  Q1 no           -> angle dead; retreat to safe-floor + reality-check paper.
"""
import math, statistics as st
import numpy as np
try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

CAP = 100.0
def lb(items): return math.ceil(sum(items) / CAP)

# ---------------- instance generators (integer items in [1,100], cap=100) ----------------
def gen(dist, n, seed):
    rng = np.random.default_rng(seed); it = []
    if dist == "weibull":           # TRAIN family / in-distribution
        for _ in range(n):
            u = rng.random(); it.append(max(1, min(100, int(round(45.0 * (-math.log(1 - u))**(1/3.0))))))
    elif dist == "weibull2":
        for _ in range(n):
            u = rng.random(); it.append(max(1, min(100, int(round(30.0 * (-math.log(1 - u))**(1/2.0))))))
    elif dist == "unif_large":      it = [int(rng.integers(35, 91)) for _ in range(n)]
    elif dist == "unif_small":      it = [int(rng.integers(1, 31)) for _ in range(n)]
    elif dist == "near_half":       it = [int(rng.integers(45, 56)) for _ in range(n)]
    elif dist == "bimodal":         it = [int(rng.integers(1, 21)) if rng.random() < 0.5 else int(rng.integers(60, 96)) for _ in range(n)]
    return it

# ---------------- FAITHFUL FunSearch online_binpack harness ----------------
def online_binpack_bins(items, priority):
    """Exact FunSearch loop. bins = num_items slots at CAP. Returns #used bins."""
    n = len(items); bins = np.full(n, CAP, dtype=float)
    for item in items:
        valid = np.nonzero(bins - item >= 0)[0]              # bins that can fit item
        with np.errstate(all="ignore"):
            pr = np.asarray(priority(float(item), bins[valid]), dtype=float)
        pr = np.nan_to_num(pr, nan=-np.inf, posinf=1e18, neginf=-np.inf)  # fragile heuristics: nan->never picked, perfect-fit inf-> picked
        best = valid[int(np.argmax(pr))]
        bins[best] -= item
    return int(np.sum(bins != CAP))

# ----- hot heuristics (VERBATIM from official repos; only signature name normalized) -----
def H_funsearch_weibull(item, bins):
    max_bin_cap = max(bins)
    score = (bins - max_bin_cap)**2 / item + bins**2 / (item**2)
    score += bins**2 / item**3
    score[bins > item] = -score[bins > item]
    score[1:] -= score[:-1]
    return score

def H_funsearch_or(item, bins):
    def s(b, it):
        d = b - it
        if d <= 2:   return 4
        elif d <= 3: return 3
        elif d <= 5: return 2
        elif d <= 7: return 1
        elif d <= 9: return 0.9
        elif d <= 12:return 0.95
        elif d <= 15:return 0.97
        elif d <= 18:return 0.98
        elif d <= 20:return 0.98
        elif d <= 21:return 0.98
        else:        return 0.99
    return np.array([s(b, item) for b in bins])

def H_eoh_best(item, bins):
    scores = (bins / np.sqrt(np.log(bins - item))) ** (bins / np.sqrt(item)) * np.exp(item * (bins - item)) * np.sqrt(item)
    scores /= (1 / bins) * np.sqrt(item)
    scores *= 100
    return scores

# ----- classical SOTA as priority functions (same faithful harness) -----
def H_best_fit(item, bins):  return -(bins - item)            # tightest fit (FunSearch seed)
def H_worst_fit(item, bins): return (bins - item)             # loosest fit
def H_first_fit(item, bins): return -np.arange(len(bins), dtype=float)  # lowest index

# Next-Fit & Harmonic are stateful -> direct simulation (harness-independent #bins)
def run_next_fit(items):
    bins = [CAP]
    for it in items:
        if bins[-1] >= it: bins[-1] -= it
        else: bins.append(CAP - it)
    return len(bins)
def run_harmonic(items, k=6):
    ob = {}; cnt = 0
    for it in items:
        c = min(k, int(1.0 / (it / CAP))) if it > 0 else k
        if c not in ob or ob[c] < it: ob[c] = CAP - it; cnt += 1
        else: ob[c] -= it
    return cnt

HOT = {"FunSearch-Weibull": H_funsearch_weibull, "FunSearch-OR": H_funsearch_or, "EoH-best": H_eoh_best}
CLASSIC_PR = {"FirstFit": H_first_fit, "BestFit": H_best_fit, "WorstFit": H_worst_fit}

def excess(used, items): return 100.0 * (used - lb(items)) / lb(items)

def method_excess(name, items):
    if name == "NextFit":   return excess(run_next_fit(items), items)
    if name == "Harmonic":  return excess(run_harmonic(items), items)
    pr = CLASSIC_PR.get(name) or HOT[name]
    return excess(online_binpack_bins(items, pr), items)

ALL = ["FirstFit", "BestFit", "WorstFit", "NextFit", "Harmonic"] + list(HOT)

if __name__ == "__main__":
    DISTS = ["weibull", "weibull2", "unif_large", "unif_small", "near_half", "bimodal"]
    TRAIN_FAMILY = "weibull"   # what the hot heuristics are (implicitly) tuned for
    N, INST = 500, 10
    per = {d: {m: [] for m in ALL} for d in DISTS}
    for d in DISTS:
        insts = [gen(d, N, 1000 + i) for i in range(INST)]
        for it in insts:
            for m in ALL:
                per[d][m].append(method_excess(m, it))
    # report
    print(f"{'dist':11s} | " + " ".join(f"{m[:10]:>10s}" for m in ALL))
    means = {d: {m: st.mean(per[d][m]) for m in ALL} for d in DISTS}
    for d in DISTS:
        win = min(means[d], key=means[d].get)
        print(f"{d:11s} | " + " ".join(f"{means[d][m]:10.2f}" for m in ALL) + f"   win={win}")
    # ---- Q1: hot degrades below BestFit on some OOD dist? ----
    q1_hits = []
    for d in DISTS:
        if d == TRAIN_FAMILY: continue
        for h in HOT:
            margin = means[d][h] - means[d]["BestFit"]   # >0 => hot is WORSE than BestFit
            if margin >= 1.0:
                q1_hits.append((d, h, round(margin, 2)))
    Q1 = len(q1_hits) > 0
    # ---- Q2: any dist where some heuristic robustly beats BestFit (Wilcoxon)? ----
    q2_hits = []
    for d in DISTS:
        bf = per[d]["BestFit"]
        for m in ALL:
            if m == "BestFit": continue
            diff_mean = means[d][m] - means[d]["BestFit"]
            if diff_mean < -0.05:   # m better than BF on mean
                if HAVE_SCIPY:
                    a = np.array(per[d][m]); b = np.array(bf)
                    try:
                        p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
                    except Exception: p = 1.0
                else:
                    p = 0.0 if all(per[d][m][i] <= bf[i] for i in range(len(bf))) and any(per[d][m][i] < bf[i] for i in range(len(bf))) else 1.0
                if p < 0.05:
                    q2_hits.append((d, m, round(diff_mean, 3), round(p, 4)))
    Q2 = len(q2_hits) > 0
    print("\n--- DECISION (pre-declared) ---")
    print("Q1 (a hot heuristic degrades >=1% excess vs BestFit on an OOD dist)?", Q1)
    for h in q1_hits: print("   OOD-degrade:", h, "(dist, method, +excess over BestFit)")
    print("Q2 (some heuristic robustly beats BestFit, Wilcoxon p<0.05)?", Q2, "(scipy=%s)" % HAVE_SCIPY)
    for h in q2_hits: print("   beats-BF:", h, "(dist, method, mean_diff, p)")
    if Q1 and Q2:
        verdict = "BRANCH A: strict rank-#1 achievable -> build GUARD-AHD, claim strict #1."
    elif Q1 and not Q2:
        verdict = "BRANCH B: aim tied-#1-with-BestFit + strictly-beat-all-hot (robustness story)."
    else:
        verdict = "BRANCH C: hot heuristics robust OOD too -> angle DEAD, retreat to safe-floor paper."
    print("VERDICT:", verdict)
    print("DONE")
