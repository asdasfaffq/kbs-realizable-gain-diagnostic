#!/usr/bin/env python3
"""
ASlib external validation of the realizable-gain deployment rule (G_cv) vs the raw VBS-SBS gap.

Answers TEVC R2#2 / R1#1: does G_cv predict, a priori, whether a selector beats the single-best
solver (SBS) on COMMUNITY-STANDARD algorithm-selection data, and does the raw VBS-SBS gap
over-predict? Uses ASlib precomputed performance matrices + official CV folds. CPU-only, minutes.

DATA (one-time):
    git clone https://github.com/coseal/aslib_data
    export ASLIB_ROOT=/path/to/aslib_data
DEPS: numpy scipy scikit-learn         # pure-python ARFF parser below, no liac-arff needed
RUN:  python aslib_diagnostic.py

The diagnostic math (SBS/VBS, raw gap, structural gain via feature-partition, cross-validated
probe selector, MDL noise floor) mirrors tevc-llm-ec/experiments/{diagnostic.py,select_probe.py}.
Only the data loader is ASlib-specific.
"""
import os, sys, math, glob
import numpy as np
from scipy.stats import wilcoxon
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

ASLIB_ROOT = os.environ.get("ASLIB_ROOT", "./aslib_data")

# ---- pre-registered scenario list (frozen at plan time; see EXPERIMENT_PLAN_ASLIB.md) ----
SCENARIOS = ["SAT11-INDU","SAT11-HAND","SAT12-ALL","MAXSAT12-PMS","ASP-POTASSCO","CSP-2010",
             "QBF-2011","PROTEUS-2014","PREMARSHALLING-ASTAR-2015","TSP-LION2015","GRAPHS-2015",
             "SAT15-INDU"]

# ------------------------------- minimal ARFF parser -------------------------------
def parse_arff(path):
    attrs, data, in_data = [], [], False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            low = s.lower()
            if not in_data:
                if low.startswith("@attribute"):
                    attrs.append(s.split()[1])
                elif low.startswith("@data"):
                    in_data = True
            else:
                data.append([c.strip() for c in s.split(",")])
    return attrs, data

def _read_desc(dirpath):
    """Return (cutoff, maximize) parsing YAML that may put values inline or on the next line."""
    lines = open(os.path.join(dirpath, "description.txt"), encoding="utf-8", errors="replace").readlines()
    cutoff, maximize = 3600.0, False
    for k, line in enumerate(lines):
        low = line.lower()
        if low.startswith("algorithm_cutoff_time"):
            try: cutoff = float(line.split(":", 1)[1])
            except ValueError: pass
        if low.startswith("maximize"):
            val = line.split(":", 1)[1].strip()
            if not val and k + 1 < len(lines):                      # value on next line, list style
                val = lines[k + 1].lstrip("- ").strip()
            maximize = val.lower() in ("yes", "true")
    return cutoff, maximize

def load_scenario(name):
    """Return (X features [n,d], P PAR10 matrix [n,n_algo], algo_names, folds [n], inst_ids)."""
    d = os.path.join(ASLIB_ROOT, name)
    cutoff, maximize = _read_desc(d)
    # features
    fa, fd = parse_arff(os.path.join(d, "feature_values.arff"))
    fcols = fa[2:]                                   # drop instance_id, repetition
    feat = {}
    for row in fd:
        vals = [np.nan if v in ("?","") else float(v) for v in row[2:]]
        feat[row[0]] = vals
    # runs -> PAR10 per (instance, algorithm)
    ra, rd = parse_arff(os.path.join(d, "algorithm_runs.arff"))
    # columns: instance_id, repetition, algorithm, <perf...>, runstatus
    perf_idx, status_idx = 3, len(ra) - 1
    algos = sorted({row[2] for row in rd})
    aidx = {a: j for j, a in enumerate(algos)}
    insts = sorted(feat.keys())
    iidx = {i: k for k, i in enumerate(insts)}
    P = np.full((len(insts), len(algos)), 10.0 * cutoff)   # default PAR10 = timeout
    for row in rd:
        inst, alg, status = row[0], row[2], row[status_idx].lower()
        if inst not in iidx:
            continue
        try:
            t = float(row[perf_idx])
        except ValueError:
            t = 10.0 * cutoff
        P[iidx[inst], aidx[alg]] = t if status == "ok" else 10.0 * cutoff
    # features matrix, impute column medians
    X = np.array([feat[i] for i in insts], dtype=float)
    med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])
    keep = ~np.isnan(med)                                    # drop all-NaN feature columns
    X = X[:, keep]
    # normalize performance so scenarios are comparable and huge cutoffs (e.g. GRAPHS 1e8)
    # do not explode the noise floor. Runtime: cost = PAR10 / (10*cutoff) in [0,1].
    if maximize:                                             # quality-maximize -> minimization cost
        P = P.max(axis=1, keepdims=True) - P
    P = P / (10.0 * cutoff)
    # folds
    ca, cd = parse_arff(os.path.join(d, "cv.arff"))
    folds = np.zeros(len(insts), dtype=int)
    for row in cd:
        if row[0] in iidx:
            folds[iidx[row[0]]] = int(float(row[-1]))
    return X, P, algos, folds, insts, cutoff

# ------------------------------- diagnostic quantities -------------------------------
def sbs_vbs(P):
    sbs_j = int(np.argmin(P.mean(axis=0)))
    return sbs_j, P[:, sbs_j], P.min(axis=1)          # sbs col idx, SBS per-inst, VBS per-inst

def noise_floor(B, N, m, delta=0.05):                 # Occam/MDL floor (diagnostic.py:noise_floor)
    return B * math.sqrt(math.log(max(N,2) / delta) / (2 * max(m,1)))

def structural_gain(P, X, K):
    """Feature-partition structural gain: best fixed algorithm per KMeans cell (Prop. 2)."""
    lab = KMeans(n_clusters=min(K, len(X)), n_init=4, random_state=0).fit_predict(X)
    reg = np.empty(len(X))
    for c in np.unique(lab):
        idx = lab == c
        best = int(np.argmin(P[idx].mean(axis=0)))
        reg[idx] = P[idx, best]
    return reg                                        # per-instance cost of regime-best policy

def g_cv(P, X, folds, algos, Clf=lambda: DecisionTreeClassifier(max_depth=3, random_state=0)):
    """Cross-validated probe-selector cost per instance, using official folds. Returns per-inst cost."""
    y = P.argmin(axis=1)                              # oracle label = best algo per instance
    sel_cost = np.empty(len(X))
    for fo in np.unique(folds):
        te = folds == fo; tr = ~te
        if len(np.unique(y[tr])) < 2:                 # degenerate fold -> play SBS-on-train
            j = int(np.argmin(P[tr].mean(axis=0)))
            sel_cost[te] = P[te, j]; continue
        clf = Clf().fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        sel_cost[te] = P[np.where(te)[0], pred]
    return sel_cost

def sig(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return wilcoxon(a, b).pvalue if np.any(a != b) else 1.0

# ------------------------------- per-scenario run -------------------------------
def run(name):
    X, P, algos, folds, insts, cutoff = load_scenario(name)
    n = len(insts)
    sbs_j, sbs, vbs = sbs_vbs(P)
    raw_gap = sbs.mean() - vbs.mean()                              # classical predictor
    B = float(np.percentile(P, 99))                               # clipped range for the floor
    nf = noise_floor(B, N=len(algos), m=n)
    # GROUND TRUTH: does a competent, properly-tuned selector (RandomForest) beat SBS on the
    # official folds, significant? (decoupled from the cheap probe used as the predictor)
    sel_strong = g_cv(P, X, folds, algos,
                      Clf=lambda: RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1))
    p_strong = sig(sel_strong, sbs)
    gt_win = int(sel_strong.mean() < sbs.mean() and p_strong < 0.05)
    gap_closed = (sbs.mean() - sel_strong.mean()) / max(raw_gap, 1e-9)
    # PREDICTOR (ours): cheap depth-3 probe's cross-validated gain, floor-calibrated
    sel_probe = g_cv(P, X, folds, algos)
    gcv = sbs.mean() - sel_probe.mean()
    p_cv = sig(sel_probe, sbs)
    pred_cv  = int(gcv > nf and p_cv < 0.05 and sel_probe.mean() < sbs.mean())
    pred_raw = int(raw_gap > nf)                                  # raw VBS-SBS gap (classical)
    gstr = {K: sbs.mean() - structural_gain(P, X, K).mean() for K in (4, 8, 16)}
    return dict(name=name, n=n, algos=len(algos), sbs=algos[sbs_j],
                raw_gap=raw_gap, gcv=gcv, floor=nf, gap_closed=gap_closed,
                gstr=gstr, gt=gt_win, pred_cv=pred_cv, pred_raw=pred_raw,
                p_cv=p_cv, p_strong=p_strong)

def confusion(rows, key):
    tp=fp=tn=fn=0
    for r in rows:
        pr, gt = r[key], r["gt"]
        tp += pr and gt; fp += pr and not gt; tn += (not pr) and (not gt); fn += (not pr) and gt
    prec = tp/max(1,tp+fp); rec = tp/max(1,tp+fn)
    return tp,fp,tn,fn,round(prec,3),round(rec,3)

def main():
    if not os.path.isdir(ASLIB_ROOT):
        sys.exit(f"ASlib data not found at {ASLIB_ROOT}. Run:\n"
                 f"  git clone https://github.com/coseal/aslib_data && export ASLIB_ROOT=$PWD/aslib_data")
    rows=[]
    print(f"{'scenario':<26}{'n':>6}{'alg':>4}  {'rawgap':>8}{'Gcv':>8}{'floor':>8}"
          f"{'%closed':>9}  gt pC pR")
    for s in SCENARIOS:
        try:
            r = run(s); rows.append(r)
            print(f"{r['name']:<26}{r['n']:>6}{r['algos']:>4}  {r['raw_gap']:>8.3f}{r['gcv']:>8.3f}"
                  f"{r['floor']:>8.3f}{100*r['gap_closed']:>8.0f}%   {r['gt']}  {r['pred_cv']}  {r['pred_raw']}")
        except Exception as e:
            print(f"{s:<26} EXCLUDED: {type(e).__name__}: {e}")     # report, never silently drop
    print("\n=== predictor quality over", len(rows), "scenarios ===")
    for key,label in [("pred_cv","G_cv (ours)"),("pred_raw","raw VBS-SBS gap")]:
        tp,fp,tn,fn,pr,rc = confusion(rows, key)
        print(f"{label:20s} TP={tp} FP={fp} TN={tn} FN={fn}  precision={pr} recall={rc}")

if __name__ == "__main__":
    main()
