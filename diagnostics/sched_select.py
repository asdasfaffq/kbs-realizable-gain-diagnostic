#!/usr/bin/env python3
"""FOURTH DOMAIN — PRE-REGISTERED diagnostic test. Single-machine total weighted tardiness
(1||sum w_j T_j) with classical dispatching rules (EDD, WSPT, MDD, ATC, SPT), which are
regime-complementary across (tardiness-factor TF, due-date-range RDD) instance classes and are
deterministic. PRE-REGISTERED PREDICTION: structural-G significant => selection ranks #1.
Metric: %excess over per-instance best (panel + adjacent-swap local search). Compute structural-G
on TRAIN first (prediction), then run a selector on held-out (confirm/refute).
"""
import os, sys, math, statistics as st
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import wilcoxon, friedmanchisquare
from sklearn.tree import DecisionTreeClassifier

REGIMES = [("tight", 0.6, 0.2), ("loose", 0.2, 0.4), ("wide", 0.4, 0.8), ("verytight", 0.8, 0.2)]
def gen_sched(tf, rdd, n, seed):
    rng = np.random.default_rng(seed)
    p = rng.integers(1, 101, n); w = rng.integers(1, 11, n); P = p.sum()
    lo = max(0, P*(1 - tf - rdd/2)); hi = max(lo+1, P*(1 - tf + rdd/2))
    d = rng.uniform(lo, hi, n)
    return p.astype(float), w.astype(float), d.astype(float)

def wtard(order, p, w, d):
    C = 0.0; tot = 0.0
    for j in order:
        C += p[j]; tot += w[j]*max(0.0, C - d[j])
    return tot
# dispatching rules -> order
def R_SPT(p,w,d):  return sorted(range(len(p)), key=lambda j: p[j])
def R_EDD(p,w,d):  return sorted(range(len(p)), key=lambda j: d[j])
def R_WSPT(p,w,d): return sorted(range(len(p)), key=lambda j: p[j]/w[j])
def R_MDD(p,w,d):
    n=len(p); rem=set(range(n)); C=0.0; order=[]
    while rem:
        j=min(rem, key=lambda j: max(C+p[j], d[j])); order.append(j); C+=p[j]; rem.discard(j)
    return order
def R_ATC(p,w,d,k=2.0):
    n=len(p); rem=set(range(n)); C=0.0; order=[]; pbar=np.mean(p)
    while rem:
        j=max(rem, key=lambda j:(w[j]/p[j])*math.exp(-max(0.0,d[j]-C-p[j])/(k*pbar+1e-9)))
        order.append(j); C+=p[j]; rem.discard(j)
    return order
PANEL={"SPT":R_SPT,"EDD":R_EDD,"WSPT":R_WSPT,"MDD":R_MDD,"ATC":R_ATC}
PANEL_NAMES=list(PANEL)
def local_search(order,p,w,d,passes=6):
    order=order[:]; best=wtard(order,p,w,d); improved=True; c=0
    while improved and c<passes:
        improved=False; c+=1
        for i in range(len(order)-1):
            order[i],order[i+1]=order[i+1],order[i]; v=wtard(order,p,w,d)
            if v<best-1e-9: best=v; improved=True
            else: order[i],order[i+1]=order[i+1],order[i]
    return best
def features(p,w,d):
    P=p.sum(); slack=(d-p)/ (P+1e-9)
    return [float(np.mean(d)/P), float(np.std(d)/P), float(np.mean(w/p)), float(np.std(w/p)),
            float(np.mean(slack)), float((d<0.5*P).mean())]
def per_inst(inst):
    p,w,d=inst; raw={m:wtard(PANEL[m](p,w,d),p,w,d) for m in PANEL_NAMES}
    orac=min(min(raw.values()), min(local_search(PANEL[m](p,w,d),p,w,d) for m in PANEL_NAMES))
    orac=max(orac,1e-9)
    return {m:100.0*(raw[m]-orac)/orac for m in PANEL_NAMES}

def structural_gain(per,nper,sbs_mean):
    reg=[]
    for k in range(len(REGIMES)):
        idx=list(range(k*nper,k*nper+nper))
        best=min(PANEL_NAMES,key=lambda m: st.mean([per[m][i] for i in idx]))
        for i in idx: reg.append(per[best][i])
    return sbs_mean-st.mean(reg), reg

if __name__=="__main__":
    n=40
    train=[(rg,gen_sched(tf,rdd,n,10+i)) for (rg,tf,rdd) in REGIMES for i in range(10)]
    test =[(rg,gen_sched(tf,rdd,n,400+i)) for (rg,tf,rdd) in REGIMES for i in range(12)]
    def build(rows):
        per={m:[] for m in PANEL_NAMES}; feats=[]
        for (rg,inst) in rows:
            ex=per_inst(inst); feats.append(features(*inst))
            for m in PANEL_NAMES: per[m].append(ex[m])
        return per,feats
    trp,trf=build(train); tep,tef=build(test)
    sbs=min(PANEL_NAMES,key=lambda m:st.mean(trp[m])); msbs=st.mean(trp[sbs])
    strG,reg=structural_gain(trp,10,msbs)
    pstr=wilcoxon(reg,trp[sbs]).pvalue if any(np.array(reg)!=np.array(trp[sbs])) else 1.0
    predict=strG>0 and pstr<0.05
    print("=== PRE-REGISTERED (TRAIN) ===")
    print(f"  SBS={sbs} mean={msbs:.3f}; structural-G={strG:.3f} (paired p={pstr:.2e})")
    print(f"  PREDICTION: selection {'RANKS #1' if predict else 'CANNOT win'}")
    # selector
    ytr=[int(np.argmin([trp[m][i] for m in PANEL_NAMES])) for i in range(len(train))]
    clf=DecisionTreeClassifier(max_depth=4,random_state=0).fit(trf,ytr)
    sel=[tep[PANEL_NAMES[int(clf.predict([tef[i]])[0])]][i] for i in range(len(test))]
    keys=["SELECTOR"]+PANEL_NAMES; allm={"SELECTOR":sel,**tep}
    means={k:round(st.mean(allm[k]),3) for k in keys}
    M=np.array([allm[k] for k in keys]); rk=np.array([(np.argsort(np.argsort(M[:,j]))+1) for j in range(M.shape[1])]).T.mean(1)
    mr={keys[i]:round(rk[i],3) for i in range(len(keys))}; fr=friedmanchisquare(*[M[i] for i in range(len(keys))])
    print(f"\n=== TEST === Friedman p={fr.pvalue:.2e}")
    for k in sorted(means,key=means.get): print(f"  {k:10s} %excess={means[k]:8.3f} rank={mr[k]:.3f}")
    beats=0
    for m in PANEL_NAMES:
        a,b=np.array(sel),np.array(tep[m]); pp=wilcoxon(a,b).pvalue if any(a!=b) else 1.0
        ok=means["SELECTOR"]<means[m] and pp<0.05; beats+=ok
        print(f"  vs {m:8s} p={pp:.4f} {'SEL-better' if ok else 'tie/worse'}")
    strict=(min(mr,key=mr.get)=="SELECTOR") and beats==len(PANEL_NAMES)
    print(f"\nSELECTOR strict rank#1? {strict} ({beats}/{len(PANEL_NAMES)})")
    print(f"PREDICTION ({'win' if predict else 'no-win'}) CORRECT? {predict==strict}")
    print("DONE")
