#!/usr/bin/env python3
"""Scale-validate the selection diagnostic as a PREDICTOR (no LLM).
Across 4 problems x many (regime-subset, panel) configurations, compute on TRAIN two predictors:
  - oracle structural gain G_str  (the classical VBS/SBS-style per-regime-best gap)
  - cross-validated REALIZABLE gain G_cv  (k-fold selector gain over the single-best method)
and on disjoint TEST the ACTUAL outcome (does a trained selector strictly beat the single-best,
Wilcoxon p<0.05?). Report each predictor's confusion matrix / accuracy, and where the classical
G_str systematically MISPREDICTS (false positives), confirming the heavy-tail/weak-feature
correction (N2).
"""
import os, sys, math, statistics as st, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import wilcoxon
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

# ---- adapters: each returns make(regime,seed)->inst, exvec(inst)->{method:cost}, feats(inst), regimes, panel ----
import select_probe as BP           # BP panel + features
import knapsack_select as KN
import tsp_select as TP
import tsp_construct_audit as TPC
import sched_select as SC

def bp_adapter():
    regs=["weibull","weibull2","unif_large","unif_small","near_half","bimodal"]
    make=lambda r,s: BP.gen(r,400,s)
    exvec=lambda it:{m:BP.panel_excess(m,it) for m in BP.PANEL_NAMES}
    return regs, make, exvec, (lambda it: BP.features(it)), BP.PANEL_NAMES
def kn_adapter():
    regs=[r for r in KN.REGIMES]
    def make(r,s):
        w,v,cap=KN.gen_knap(r,50,s); opt=KN.dp_opt(w,v,cap); return (w,v,cap,opt)
    def exvec(inst):
        w,v,cap,opt=inst; return {m:KN.gap(m,w,v,cap,opt) for m in KN.PANEL_NAMES}
    feats=lambda inst: KN.features(inst[0],inst[1])
    return regs, make, exvec, feats, KN.PANEL_NAMES
def tsp_adapter():
    regs=["uniform","clustered","grid"]
    make=lambda r,s: TPC.make_cities(r,60,s)
    exvec=lambda pts:{m:TP.length(m,pts) for m in TP.PANEL}
    feats=lambda pts: TP.features(pts)
    return regs, make, exvec, feats, TP.PANEL
def sched_adapter():
    regs=SC.REGIMES  # (name,tf,rdd)
    rnames=[r[0] for r in regs]
    tfrdd={r[0]:(r[1],r[2]) for r in regs}
    def make(r,s):
        tf,rdd=tfrdd[r]; return SC.gen_sched(tf,rdd,40,s)
    def exvec(inst):
        return SC.per_inst(inst)   # dict over dispatch rules, %excess over per-instance best
    feats=lambda inst: SC.features(*inst)
    return rnames, make, exvec, feats, SC.PANEL_NAMES

ADAPTERS={"binpack":bp_adapter(),"knapsack":kn_adapter(),"tsp":tsp_adapter(),"sched":sched_adapter()}

def eval_config(adapter, regime_subset, ntr=30, nte=30):
    regs, make, exvec, feats, panel = adapter
    # build train/test instances over the regime subset (disjoint seeds)
    tr=[(r,make(r,100+i)) for r in regime_subset for i in range(ntr)]
    te=[(r,make(r,5000+i)) for r in regime_subset for i in range(nte)]
    def mat(rows):
        per={m:[] for m in panel}; X=[]; lab=[]
        for (r,inst) in rows:
            ev=exvec(inst); X.append(feats(inst))
            for m in panel: per[m].append(ev[m])
            lab.append(panel[int(np.argmin([ev[m] for m in panel]))])
        return per,np.array(X),lab
    trp,Xtr,_=mat(tr); tep,Xte,_=mat(te)
    sbs=min(panel,key=lambda m:st.mean(trp[m]))
    # ---- oracle structural gain (per-regime best applied within regime) ----
    reg_sel=[]
    idx=0
    for r in regime_subset:
        sub=[trp[m][idx:idx+ntr] for m in panel]
        best=min(range(len(panel)),key=lambda j:st.mean(sub[j]))
        reg_sel+=list(sub[best]); idx+=ntr
    Gstr=st.mean(trp[sbs])-st.mean(reg_sel)
    pstr=wilcoxon(reg_sel,trp[sbs]).pvalue if any(np.array(reg_sel)!=np.array(trp[sbs])) else 1.0
    # ---- cross-validated realizable gain on TRAIN ----
    ybase=[int(np.argmin([trp[m][i] for m in panel])) for i in range(len(Xtr))]
    cv_sel=[]; cv_sbs=[]
    kf=KFold(n_splits=4,shuffle=True,random_state=0)
    for tri,vai in kf.split(Xtr):
        clf=DecisionTreeClassifier(max_depth=4,random_state=0).fit(Xtr[tri],[ybase[i] for i in tri])
        for i in vai:
            pick=panel[int(clf.predict([Xtr[i]])[0])]
            cv_sel.append(trp[pick][i]); cv_sbs.append(trp[sbs][i])
    Gcv=st.mean(cv_sbs)-st.mean(cv_sel)
    pcv=wilcoxon(cv_sel,cv_sbs).pvalue if any(np.array(cv_sel)!=np.array(cv_sbs)) else 1.0
    # ---- ACTUAL: train on full train, test on held-out ----
    clf=DecisionTreeClassifier(max_depth=4,random_state=0).fit(Xtr,ybase)
    sel_te=[tep[panel[int(clf.predict([Xte[i]])[0])]][i] for i in range(len(Xte))]
    sbs_te=tep[sbs]
    pact=wilcoxon(sel_te,sbs_te).pvalue if any(np.array(sel_te)!=np.array(sbs_te)) else 1.0
    actual = (st.mean(sel_te)<st.mean(sbs_te)) and pact<0.05
    pred_str = (Gstr>1e-9) and pstr<0.05
    pred_cv  = (Gcv>1e-9) and pcv<0.05
    return dict(Gstr=round(Gstr,3),pstr=round(pstr,3),Gcv=round(Gcv,3),pcv=round(pcv,3),
                actual=bool(actual),pred_str=bool(pred_str),pred_cv=bool(pred_cv),sbs=sbs)

def configs_for(regs):
    cfgs=[(r,) for r in regs]                       # homogeneous (expect no gain)
    if len(regs)>=2:
        for c in itertools.combinations(regs,2): cfgs.append(c)   # pairs
    cfgs.append(tuple(regs))                          # full mix
    return cfgs

if __name__=="__main__":
    rows=[]
    for prob,adp in ADAPTERS.items():
        regs=adp[0]
        cfgs=configs_for(regs)
        # cap pairs to keep runtime bounded
        if len(cfgs)>9: cfgs=[cfgs[i] for i in range(0,len(cfgs),max(1,len(cfgs)//9))][:9]
        for cfg in cfgs:
            try:
                r=eval_config(adp,list(cfg)); r["prob"]=prob; r["cfg"]="+".join(cfg); rows.append(r)
                print(f"{prob:8s} {('+'.join(cfg))[:22]:22s} Gstr={r['Gstr']:+.2f}(p{r['pstr']:.2f}) Gcv={r['Gcv']:+.2f}(p{r['pcv']:.2f}) | pred_str={int(r['pred_str'])} pred_cv={int(r['pred_cv'])} actual={int(r['actual'])}",flush=True)
            except Exception as e:
                print(f"{prob} {cfg} ERR {e}",flush=True)
    # confusion matrices
    def cm(key):
        tp=sum(1 for r in rows if r[key] and r['actual']); fp=sum(1 for r in rows if r[key] and not r['actual'])
        tn=sum(1 for r in rows if not r[key] and not r['actual']); fn=sum(1 for r in rows if not r[key] and r['actual'])
        acc=(tp+tn)/max(1,len(rows)); return tp,fp,tn,fn,round(acc,3)
    print("\n=== predictor accuracy over",len(rows),"configs ===")
    for key,name in [("pred_cv","REALIZABLE G_cv (ours)"),("pred_str","oracle/structural G_str (classical-style)")]:
        tp,fp,tn,fn,acc=cm(key)
        print(f"{name:42s} TP={tp} FP={fp} TN={tn} FN={fn}  acc={acc}")
    # where classical mispredicts but ours is right
    fp_str=[(r['prob'],r['cfg']) for r in rows if r['pred_str'] and not r['actual']]
    print("classical G_str FALSE POSITIVES (predict win, selection actually fails):", fp_str)
    import json; json.dump(rows,open("experiments/diagnostic_scale_results.json","w"),indent=1)
    print("DONE")
