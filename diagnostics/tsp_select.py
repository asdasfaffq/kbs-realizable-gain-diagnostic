#!/usr/bin/env python3
"""SECOND DOMAIN (generality) — TSP construction: does the selection-ranks-#1 story replicate?
Panel: 5 classical (NN, NN+2opt, CheapestInsertion, GreedyEdge, Random+2opt) + 2 hot published
LLM heuristics (ReEvo-weighted seed; EoH tsp_construct_class discovered, obj=6.35). A per-instance
selector (cheap geometry features -> base heuristic) dispatches among a complementary base.
Metric: %excess over per-instance best. Shift axis: uniform / clustered / grid cities.
"""
import os, sys, math, statistics as st
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsp_construct_audit as T   # make_cities, dmat, tour_cost, nn_select, reevo_select, construct, two_opt, two_opt_restart
from scipy.stats import wilcoxon, friedmanchisquare
from sklearn.tree import DecisionTreeClassifier

# ---- EoH tsp_construct_class discovered heuristic (VERBATIM logic) ----
def eoh_ratio_select(current_node, destination_node, unvisited_nodes, D):
    unv = list(unvisited_nodes)
    if len(unv) == 1: return unv[0]
    dfc = np.array([D[current_node][j] for j in unv])
    avg = np.array([np.mean([D[j][k] for k in unv]) for j in unv])
    return unv[int(np.argmin(dfc / (avg + 1e-10)))]

# ---- extra classical constructors ----
def cheapest_insertion(D):
    n = len(D); start = 0
    tour = [start, int(np.argmax(D[start]))]  # farthest as 2nd
    inset = set(tour)
    while len(tour) < n:
        best = None
        for c in range(n):
            if c in inset: continue
            # cheapest position
            for i in range(len(tour)):
                a, b = tour[i], tour[(i + 1) % len(tour)]
                delta = D[a][c] + D[c][b] - D[a][b]
                if best is None or delta < best[0]: best = (delta, i + 1, c)
        tour.insert(best[1], best[2]); inset.add(best[2])
    return tour
def greedy_edge(D):
    n = len(D); edges = []
    for i in range(n):
        for j in range(i + 1, n): edges.append((D[i][j], i, j))
    edges.sort(); deg = [0]*n; uf = list(range(n)); adj = {i: [] for i in range(n)}
    def find(x):
        while uf[x] != x: uf[x] = uf[uf[x]]; x = uf[x]
        return x
    cnt = 0
    for d, i, j in edges:
        if deg[i] < 2 and deg[j] < 2 and find(i) != find(j):
            adj[i].append(j); adj[j].append(i); deg[i]+=1; deg[j]+=1; uf[find(i)] = find(j); cnt += 1
            if cnt == n - 1: break
    ends = [i for i in range(n) if deg[i] < 2]
    if len(ends) == 2: i, j = ends; adj[i].append(j); adj[j].append(i)
    tour = [0]; prev = -1; cur = 0
    for _ in range(n - 1):
        nxt = adj[cur][0] if adj[cur][0] != prev else (adj[cur][1] if len(adj[cur])>1 else adj[cur][0])
        tour.append(nxt); prev, cur = cur, nxt
    return tour

def features(pts):
    D = T.dmat(pts); n = len(pts)
    nn = np.partition(D + np.eye(n)*1e9, 1, axis=1)[:, 1]   # nearest-neighbor distance per point
    allpair = D[np.triu_indices(n, 1)]
    # regularity: low CV of NN distance => grid-like; high => clustered
    cv_nn = nn.std() / (nn.mean() + 1e-12)
    return [float(pts[:,0].std()), float(pts[:,1].std()), float(nn.mean()), float(cv_nn),
            float(nn.mean() / (allpair.mean() + 1e-12))]

def m_excess(name, pts):
    D = T.dmat(pts)
    if name == "NN": t = T.construct(D, T.nn_select)
    elif name == "NN+2opt": t = T.two_opt(D, T.construct(D, T.nn_select))
    elif name == "CheapestInsertion": t = cheapest_insertion(D)
    elif name == "GreedyEdge": t = greedy_edge(D)
    elif name == "Random+2opt": return T.two_opt_restart(D, 8, 7)  # returns length
    elif name == "ReEvo-weighted": t = T.construct(D, T.reevo_select)
    elif name == "EoH-ratio": t = T.construct(D, eoh_ratio_select)
    return T.tour_cost(D, t)

PANEL = ["NN", "NN+2opt", "CheapestInsertion", "GreedyEdge", "Random+2opt", "ReEvo-weighted", "EoH-ratio"]
def length(name, pts): return m_excess(name, pts)
BASE = ["NN+2opt", "ReEvo-weighted"]   # complementary: NN+2opt wins uniform/clustered, ReEvo wins grid

def inst_excess(pts):
    D = T.dmat(pts)
    raw = {m: length(m, pts) for m in PANEL}
    orac = min(min(raw.values()), T.two_opt_restart(D, 25, 321))
    return {m: 100.0 * (raw[m] - orac) / orac for m in PANEL}, raw

if __name__ == "__main__":
    DISTS = ["uniform", "clustered", "grid"]; n = 60
    train = [(d, T.make_cities(d, n, 100 + i)) for d in DISTS for i in range(8)]
    test  = [(d, T.make_cities(d, n, 500 + i)) for d in DISTS for i in range(12)]

    # train selector: features -> argmin over BASE
    Xtr, ytr = [], []
    for d, pts in train:
        b = {m: length(m, pts) for m in BASE}
        Xtr.append(features(pts)); ytr.append(int(np.argmin([b[m] for m in BASE])))
    clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xtr, ytr)

    per = {m: [] for m in PANEL}; sel = []
    for d, pts in test:
        ex, raw = inst_excess(pts)
        for m in PANEL: per[m].append(ex[m])
        pick = BASE[int(clf.predict([features(pts)])[0])]
        D = T.dmat(pts); orac = min(min(raw.values()), T.two_opt_restart(D, 25, 321))
        sel.append(100.0 * (raw[pick] - orac) / orac)
    keys = ["SELECTOR"] + PANEL; allm = {"SELECTOR": sel, **per}
    means = {k: round(st.mean(allm[k]), 3) for k in keys}
    M = np.array([allm[k] for k in keys]); rk = np.array([(np.argsort(np.argsort(M[:,j]))+1) for j in range(M.shape[1])]).T.mean(1)
    mr = {keys[i]: round(rk[i],3) for i in range(len(keys))}
    fr = friedmanchisquare(*[M[i] for i in range(len(keys))])
    print("TSP construction (2nd domain) — mean %excess over per-instance best, test:")
    for k in sorted(means, key=means.get): print(f"  {k:18s} {means[k]:7.3f}  rank={mr[k]:.3f}")
    print(f"Friedman p={fr.pvalue:.2e}")
    beats = 0
    print("SELECTOR vs each panel (Wilcoxon):")
    for m in PANEL:
        a, b = np.array(sel), np.array(per[m]); p = wilcoxon(a, b).pvalue if any(a!=b) else 1.0
        ok = means["SELECTOR"] < means[m] and p < 0.05; beats += ok
        print(f"  vs {m:18s} {means['SELECTOR']:.3f} vs {means[m]:.3f} p={p:.4f} {'SEL-better' if ok else 'tie/worse'}")
    strict = (min(mr, key=mr.get) == "SELECTOR") and beats == len(PANEL)
    print(f"\nSELECTOR strict rank#1 over panel? {strict} ({beats}/{len(PANEL)})")
    from collections import Counter
    picks = {d: Counter() for d in DISTS}
    for d, pts in test: picks[d][BASE[int(clf.predict([features(pts)])[0])]] += 1
    print("picks per dist:", {d: dict(picks[d]) for d in DISTS})
    print("DONE")
