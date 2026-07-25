#!/usr/bin/env python3
"""A6 second problem — TSP constructive heuristic. Audit whether a published LLM-EC
constructive heuristic with tuned magic constants generalizes across city distributions
or overfits / underperforms the trivial nearest-neighbor baseline. No LLM.

Faithful to EoH's tsp_construct harness (examples/tsp_construct/evaluation): greedy
construction via `select_next_node(current, destination, unvisited, distance_matrix)`,
euclidean tour cost, return-to-start.

Heuristics:
  - NearestNeighbor : argmin distance  (= EoH seed `heuristic.py`; the classical baseline)
  - ReEvo-weighted  : VERBATIM ai4co/reevo prompts/tsp_constructive/seed_func.txt
                      (tuned constants c1..c4 + destination term) -- the "hot" LLM-EC heuristic
  - NN+2opt         : nearest-neighbor refined by 2-opt (stronger classical reference)
Metric: %excess of tour cost over the per-instance best (incl. a 2-opt-restart oracle).
Shift axis: uniform / clustered / grid cities.
"""
import math, statistics as st
import numpy as np

def make_cities(kind, n, seed):
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        return rng.random((n, 2))
    if kind == "clustered":
        nc = 4; centers = rng.random((nc, 2)); pts = []
        for _ in range(n):
            c = centers[rng.integers(nc)]; pts.append(np.clip(c + rng.normal(0, 0.06, 2), 0, 1))
        return np.array(pts)
    if kind == "grid":
        side = int(math.ceil(math.sqrt(n))); g = [(i / side, j / side) for i in range(side) for j in range(side)]
        idx = rng.permutation(len(g))[:n]; return np.array([g[i] for i in idx])

def dmat(pts):
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))
def tour_cost(D, t):
    return float(sum(D[t[j], t[j + 1]] for j in range(len(t) - 1)) + D[t[-1], t[0]])

# ---- classical: nearest neighbor (EoH seed) ----
def nn_select(current_node, destination_node, unvisited_nodes, distance_matrix):
    nid = int(np.argmin([distance_matrix[current_node][i] for i in unvisited_nodes if i != current_node]))
    cand = [i for i in unvisited_nodes if i != current_node]
    return cand[nid] if cand else unvisited_nodes[0]

# ---- hot: ReEvo tsp_constructive seed (VERBATIM, tuned constants) ----
def reevo_select(current_node, destination_node, unvisited_nodes, distance_matrix):
    threshold = 0.7
    c1, c2, c3, c4 = 0.4, 0.3, 0.2, 0.1
    scores = {}
    for node in unvisited_nodes:
        all_distances = [distance_matrix[node][i] for i in unvisited_nodes if i != node]
        if not all_distances:
            scores[node] = distance_matrix[current_node][node]; continue
        average_distance_to_unvisited = np.mean(all_distances)
        std_dev_distance_to_unvisited = np.std(all_distances)
        score = (c1 * distance_matrix[current_node][node] - c2 * average_distance_to_unvisited
                 + c3 * std_dev_distance_to_unvisited - c4 * distance_matrix[destination_node][node])
        scores[node] = score
    return min(scores, key=scores.get)

def construct(D, select, start=0):
    n = len(D); unv = [i for i in range(n) if i != start]; t = [start]; cur = start
    while unv:
        nxt = select(cur, start, unv, D); t.append(nxt); unv.remove(nxt); cur = nxt
    return t

def two_opt(D, t, max_pass=40):
    n = len(t); t = t[:]; improved = True; p = 0
    while improved and p < max_pass:
        improved = False; p += 1
        for i in range(n - 1):
            a, b = t[i], t[i + 1]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1: continue
                c, d = t[j], t[(j + 1) % n]
                if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-12:
                    t[i + 1:j + 1] = t[i + 1:j + 1][::-1]; improved = True; a, b = t[i], t[i + 1]
    return t

def two_opt_restart(D, restarts, seed):
    rng = np.random.default_rng(seed); n = len(D); best = math.inf
    for _ in range(restarts):
        t = list(rng.permutation(n)); best = min(best, tour_cost(D, two_opt(D, t)))
    return best

PANEL = ["NearestNeighbor", "ReEvo-weighted", "NN+2opt"]
def eval_instance(pts):
    D = dmat(pts)
    nn = construct(D, nn_select); re = construct(D, reevo_select)
    raw = {"NearestNeighbor": tour_cost(D, nn),
           "ReEvo-weighted": tour_cost(D, re),
           "NN+2opt": tour_cost(D, two_opt(D, nn))}
    oracle = min(min(raw.values()), two_opt_restart(D, 20, 999))
    return {m: 100.0 * (raw[m] - oracle) / oracle for m in PANEL}

if __name__ == "__main__":
    N = 50; SEEDS = range(12); dists = ["uniform", "clustered", "grid"]
    per = {d: {m: [] for m in PANEL} for d in dists}
    for d in dists:
        for s in SEEDS:
            ex = eval_instance(make_cities(d, N, 100 + s))
            for m in PANEL: per[d][m].append(ex[m])
    means = {d: {m: st.mean(per[d][m]) for m in PANEL} for d in dists}
    print(f"TSP construction audit, n={N}, {len(list(SEEDS))} seeds/dist, %excess over per-instance best")
    print(f"{'dist':10s} | " + " ".join(f"{m:>16s}" for m in PANEL))
    for d in dists:
        win = min(means[d], key=means[d].get)
        print(f"{d:10s} | " + " ".join(f"{means[d][m]:16.2f}" for m in PANEL) + f"  win={win}")
    print("\n=== ReEvo-weighted vs NearestNeighbor (overfitting check) ===")
    for d in dists:
        diff = means[d]["ReEvo-weighted"] - means[d]["NearestNeighbor"]
        tag = "ReEvo WORSE than trivial NN" if diff > 0.1 else ("ReEvo better" if diff < -0.1 else "≈ tie")
        print(f"  {d:10s}: ReEvo={means[d]['ReEvo-weighted']:.2f}%  NN={means[d]['NearestNeighbor']:.2f}%  diff={diff:+.2f}  -> {tag}")
    print("DONE")
