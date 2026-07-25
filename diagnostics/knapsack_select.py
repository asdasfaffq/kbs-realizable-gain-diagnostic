#!/usr/bin/env python3
"""THIRD DOMAIN — PRE-REGISTERED test of the selection diagnostic (Prop 4).
0/1 knapsack has textbook DETERMINISTIC regime structure (Pisinger instance classes:
uncorrelated / weakly / strongly correlated / subset-sum), where different greedy rules win on
different classes and there is NO stochastic baseline noise.

PRE-REGISTERED PREDICTION (written before computing the selector outcome):
  structural-G should be SIGNIFICANT  =>  regime-aware selection RANKS #1.
We compute structural-G first (print the prediction), then run the selector and report whether
the prediction holds. Either outcome validates the diagnostic (this is a test, not cherry-picking).
Metric: %gap from the DP optimum (maximization). Panel = 5 classical greedy rules.
"""
import os, sys, math, statistics as st
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import wilcoxon, friedmanchisquare
from sklearn.tree import DecisionTreeClassifier

REGIMES = ["uncorrelated", "weak_corr", "strong_corr", "subset_sum"]
def gen_knap(regime, n, seed):
    rng = np.random.default_rng(seed); w = rng.integers(1, 101, n)
    if regime == "uncorrelated": v = rng.integers(1, 101, n)
    elif regime == "weak_corr":  v = np.clip(w + rng.integers(-10, 11, n), 1, None)
    elif regime == "strong_corr": v = w + 10
    elif regime == "subset_sum":  v = w.copy()
    cap = int(0.5 * w.sum())
    return w.astype(int), v.astype(int), cap

def dp_opt(w, v, cap):
    dp = np.zeros(cap + 1, dtype=np.int64)
    for i in range(len(w)):
        wi, vi = int(w[i]), int(v[i])
        if wi <= cap: dp[wi:] = np.maximum(dp[wi:], dp[:cap + 1 - wi] + vi)
    return int(dp[cap])

def greedy(order, w, v, cap):
    val = 0; rem = cap
    for i in order:
        if w[i] <= rem: rem -= w[i]; val += v[i]
    return val
def H_ratio(w, v, cap):   return greedy(sorted(range(len(w)), key=lambda i: -v[i]/w[i]), w, v, cap)
def H_value(w, v, cap):   return greedy(sorted(range(len(w)), key=lambda i: -v[i]), w, v, cap)
def H_wlight(w, v, cap):  return greedy(sorted(range(len(w)), key=lambda i: w[i]), w, v, cap)
def H_wheavy(w, v, cap):  return greedy(sorted(range(len(w)), key=lambda i: -w[i]), w, v, cap)
def H_ratiofill(w, v, cap):  # ratio, then try to fill remaining capacity with any fitting leftover
    order = sorted(range(len(w)), key=lambda i: -v[i]/w[i]); val = 0; rem = cap; taken = [False]*len(w)
    for i in order:
        if w[i] <= rem: rem -= w[i]; val += v[i]; taken[i] = True
    for i in sorted(range(len(w)), key=lambda i: w[i]):
        if not taken[i] and w[i] <= rem: rem -= w[i]; val += v[i]
    return val
PANEL = {"ratio": H_ratio, "value": H_value, "wlight": H_wlight, "wheavy": H_wheavy, "ratiofill": H_ratiofill}
PANEL_NAMES = list(PANEL)

def gap(name, w, v, cap, opt):
    return 100.0 * (opt - PANEL[name](w, v, cap)) / opt if opt > 0 else 0.0
def features(w, v):
    w = w.astype(float); v = v.astype(float); r = v / w
    corr = float(np.corrcoef(w, v)[0, 1]) if w.std() > 0 and v.std() > 0 else 1.0
    return [float(r.mean()), float(r.std()), corr, float(v.mean()/ (w.mean()+1e-9)), float((v == w).mean())]

def structural_gain(per, nper, sbs_mean):
    reg = []
    for k in range(len(REGIMES)):
        idx = list(range(k*nper, k*nper+nper))
        best = min(PANEL_NAMES, key=lambda m: st.mean([per[m][i] for i in idx]))
        for i in idx: reg.append(per[best][i])
    return sbs_mean - st.mean(reg), reg

if __name__ == "__main__":
    n = 50
    train = [(rg, *gen_knap(rg, n, 10 + i)) for rg in REGIMES for i in range(10)]
    test  = [(rg, *gen_knap(rg, n, 400 + i)) for rg in REGIMES for i in range(12)]
    def build(rows):
        per = {m: [] for m in PANEL_NAMES}; feats = []; opts = []
        for (rg, w, v, cap) in rows:
            opt = dp_opt(w, v, cap); opts.append(opt); feats.append(features(w, v))
            for m in PANEL_NAMES: per[m].append(gap(m, w, v, cap, opt))
        return per, feats
    tr_per, tr_feat = build(train); te_per, te_feat = build(test)

    # ---- STRUCTURAL-G + PRE-REGISTERED PREDICTION (computed on TRAIN, before any selector) ----
    sbs_tr = min(PANEL_NAMES, key=lambda m: st.mean(tr_per[m])); msbs_tr = st.mean(tr_per[sbs_tr])
    strG, reg = structural_gain(tr_per, 10, msbs_tr)
    p_str = wilcoxon(reg, tr_per[sbs_tr]).pvalue if any(np.array(reg)!=np.array(tr_per[sbs_tr])) else 1.0
    predict_win = strG > 0 and p_str < 0.05
    print("=== PRE-REGISTERED (on TRAIN) ===")
    print(f"  SBS={sbs_tr} mean={msbs_tr:.3f}; structural-G={strG:.3f} (paired p={p_str:.2e})")
    print(f"  PREDICTION: regime-aware selection {'RANKS #1' if predict_win else 'CANNOT win'}")

    # ---- now build selector and TEST (confirm/refute) ----
    BASE = PANEL_NAMES  # let selector choose among all 5
    Xtr = tr_feat; ytr = [int(np.argmin([tr_per[m][i] for m in BASE])) for i in range(len(train))]
    clf = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Xtr, ytr)
    sel = [te_per[BASE[int(clf.predict([te_feat[i]])[0])]][i] for i in range(len(test))]
    keys = ["SELECTOR"] + PANEL_NAMES; allm = {"SELECTOR": sel, **te_per}
    means = {k: round(st.mean(allm[k]), 3) for k in keys}
    M = np.array([allm[k] for k in keys]); rk = np.array([(np.argsort(np.argsort(M[:,j]))+1) for j in range(M.shape[1])]).T.mean(1)
    mr = {keys[i]: round(rk[i],3) for i in range(len(keys))}
    fr = friedmanchisquare(*[M[i] for i in range(len(keys))])
    print("\n=== TEST OUTCOME ===  Friedman p=%.2e" % fr.pvalue)
    for k in sorted(means, key=means.get): print(f"  {k:12s} gap%={means[k]:7.3f}  rank={mr[k]:.3f}")
    beats = 0
    for m in PANEL_NAMES:
        a, b = np.array(sel), np.array(te_per[m]); p = wilcoxon(a,b).pvalue if any(a!=b) else 1.0
        ok = means["SELECTOR"] < means[m] and p < 0.05; beats += ok
        print(f"  vs {m:10s} {means['SELECTOR']:.3f} vs {means[m]:.3f} p={p:.4f} {'SEL-better' if ok else 'tie/worse'}")
    strict = (min(mr, key=mr.get) == "SELECTOR") and beats == len(PANEL_NAMES)
    print(f"\nSELECTOR strict rank#1? {strict} ({beats}/{len(PANEL_NAMES)})")
    print(f"PREDICTION ({'win' if predict_win else 'no-win'}) CORRECT? {predict_win == strict}")
    print("DONE")
