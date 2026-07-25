#!/usr/bin/env python3
"""Method redesign — feasibility probe for a SELECTION hyper-heuristic.
Question: can a legitimate per-instance selector (features computable WITHOUT running the
heuristics / WITHOUT the outcome -- NOT a VBS oracle) statistically RANK #1 over the panel of
5 classical SOTA + 3 hot, by dispatching to the best base heuristic per regime?

Base set for selection: {BestFit, FS-OR} (the two strong, complementary members:
FS-OR wins in-distribution Weibull, Best-Fit wins elsewhere). Online-feasible features are
estimated from a PREFIX of the item stream (first P items) -> commit to one heuristic.
Honest: selector trained on TRAIN instances, evaluated on disjoint TEST; we report oracle
(upper bound) AND the realistic feature-selector, with Friedman ranks + Wilcoxon.
"""
import os, sys, math, statistics as st
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_ood import (gen, lb, online_binpack_bins, CAP,
                       H_funsearch_weibull, H_funsearch_or, H_eoh_best,
                       H_best_fit, H_worst_fit, H_first_fit, run_next_fit, run_harmonic)
from scipy.stats import wilcoxon, friedmanchisquare
from sklearn.tree import DecisionTreeClassifier

def excess(items, pr): return 100.0 * (online_binpack_bins(items, pr) - lb(items)) / lb(items)
def m_nf(items): return 100.0 * (run_next_fit(items) - lb(items)) / lb(items)
def m_hm(items): return 100.0 * (run_harmonic(items) - lb(items)) / lb(items)

PANEL = {"FirstFit": H_first_fit, "BestFit": H_best_fit, "WorstFit": H_worst_fit,
         "FunSearch-Weibull": H_funsearch_weibull, "FunSearch-OR": H_funsearch_or, "EoH-best": H_eoh_best}
def panel_excess(name, items):
    if name == "NextFit": return m_nf(items)
    if name == "Harmonic": return m_hm(items)
    return excess(items, PANEL[name])
PANEL_NAMES = ["FirstFit", "BestFit", "WorstFit", "NextFit", "Harmonic",
               "FunSearch-Weibull", "FunSearch-OR", "EoH-best"]

# online-feasible features from a PREFIX (first P items) -- no heuristic is run, no outcome seen
def features(items, P=60):
    pre = np.array(items[:P], dtype=float) / CAP
    return [pre.mean(), pre.std(), float((pre < 0.2).mean()), float(((pre >= 0.45) & (pre <= 0.55)).mean()),
            float((pre > 0.5).mean()), float((pre > 0.35).mean() * (pre < 0.9).mean())]

BASE = ["BestFit", "FunSearch-OR"]   # complementary strong base for the selector
def base_excess(name, items): return excess(items, PANEL[name])

if __name__ == "__main__":
    DISTS = ["weibull", "weibull2", "unif_large", "unif_small", "near_half", "bimodal"]
    N = 500
    train = [(d, gen(d, N, 10 + i)) for d in DISTS for i in range(8)]
    test  = [(d, gen(d, N, 400 + i)) for d in DISTS for i in range(10)]

    # train selector: features -> argmin over BASE (on TRAIN only)
    Xtr, ytr = [], []
    for d, it in train:
        ex = {b: base_excess(b, it) for b in BASE}
        Xtr.append(features(it)); ytr.append(int(np.argmin([ex[b] for b in BASE])))
    clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xtr, ytr)

    # evaluate on TEST
    per = {m: [] for m in PANEL_NAMES}
    sel, orac = [], []
    for d, it in test:
        for m in PANEL_NAMES: per[m].append(panel_excess(m, it))
        exb = {b: base_excess(b, it) for b in BASE}
        pick = BASE[int(clf.predict([features(it)])[0])]
        sel.append(exb[pick])
        orac.append(min(exb.values()))
    # ---- ranking & significance over deployable methods ONLY (oracle excluded as it is just an upper bound) ----
    keys = ["SELECTOR"] + PANEL_NAMES
    allm = {"SELECTOR": sel, **per}
    means = {k: round(st.mean(allm[k]), 3) for k in keys}
    rank = sorted(keys, key=lambda k: means[k])
    print("mean %excess (test, deployable methods):")
    for k in rank: print(f"  {k:18s} {means[k]:7.3f}")
    print(f"  [ref] ORACLE(2) {st.mean(orac):.3f}  (non-deployable upper bound)")
    M = np.array([allm[k] for k in keys]); N_ins = M.shape[1]
    fr = friedmanchisquare(*[M[i] for i in range(len(keys))])
    rk = np.array([(np.argsort(np.argsort(M[:, j])) + 1) for j in range(N_ins)]).T.mean(1)
    mr = {keys[i]: round(rk[i], 3) for i in range(len(keys))}
    print(f"\nFriedman p={fr.pvalue:.2e}; mean ranks:")
    for k in sorted(mr, key=mr.get): print(f"  {k:18s} {mr[k]:6.3f}")
    print("\nSELECTOR vs each panel member (Wilcoxon, selector lower=better):")
    beats = 0
    for m in PANEL_NAMES:
        a, b = np.array(sel), np.array(per[m])
        p = wilcoxon(a, b).pvalue if any(a != b) else 1.0
        win = int((a < b).sum()); lose = int((a > b).sum())
        ok = means["SELECTOR"] < means[m] and p < 0.05
        beats += ok
        print(f"  vs {m:18s} {means['SELECTOR']:.3f} vs {means[m]:.3f}  W/L={win}/{lose}  p={p:.4f}  -> {'SEL sig-better' if ok else ('sig-worse' if means['SELECTOR']>means[m] and p<0.05 else 'tie')}")
    strict_rank1 = (rank[0] == "SELECTOR") and (beats == len(PANEL_NAMES))
    print(f"\nSELECTOR #1 by mean & Friedman? {rank[0]=='SELECTOR'} | STRICTLY beats ALL 8 panel (p<0.05)? {strict_rank1} ({beats}/8)")
    print(f"oracle(2)={st.mean(orac):.3f} selector={means['SELECTOR']:.3f} (gap shows selector != oracle, i.e. real predictor)")
    # decision sanity: what does it pick per distribution?
    from collections import Counter
    picks = {d: Counter() for d in DISTS}
    for d, it in test: picks[d][BASE[int(clf.predict([features(it)])[0])]] += 1
    print("selector picks per distribution:", {d: dict(picks[d]) for d in DISTS})

    # ---- robustness across 3 independent test seed-sets (selector/clf fixed) ----
    print("\n=== ROBUSTNESS over 3 independent test draws (12/dist=72 each) ===")
    for rep, base_seed in enumerate([4000, 6000, 8000]):
        tt = [(d, gen(d, N, base_seed + i)) for d in DISTS for i in range(12)]
        s = []; pe = {m: [] for m in PANEL_NAMES}
        for d, it in tt:
            for m in PANEL_NAMES: pe[m].append(panel_excess(m, it))
            exb = {b: base_excess(b, it) for b in BASE}
            s.append(exb[BASE[int(clf.predict([features(it)])[0])]])
        msel = st.mean(s)
        Kf = ["SEL"] + PANEL_NAMES; Mf = np.array([s] + [pe[m] for m in PANEL_NAMES])
        rkf = np.array([(np.argsort(np.argsort(Mf[:, j])) + 1) for j in range(Mf.shape[1])]).T.mean(1)
        sel_rank = rkf[0]; is_top = sel_rank == rkf.min()
        nbeat = sum(1 for m in PANEL_NAMES if msel < st.mean(pe[m]) and (wilcoxon(s, pe[m]).pvalue < 0.05 if any(np.array(s)!=np.array(pe[m])) else False))
        pbf = wilcoxon(s, pe["BestFit"]).pvalue if any(np.array(s)!=np.array(pe["BestFit"])) else 1.0
        pfo = wilcoxon(s, pe["FunSearch-OR"]).pvalue if any(np.array(s)!=np.array(pe["FunSearch-OR"])) else 1.0
        print(f"  draw{rep}: SEL mean={msel:.3f} rank={sel_rank:.2f} top={is_top} beats {nbeat}/8 (p_BF={pbf:.4f} p_FSOR={pfo:.4f}) strict#1={is_top and nbeat==8}")
    print("DONE")
