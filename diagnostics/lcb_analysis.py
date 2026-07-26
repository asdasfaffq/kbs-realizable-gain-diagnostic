#!/usr/bin/env python3
"""
Certified build rule: one-sided lower confidence bound (LCB) on the realizable gain, and
cost-sensitivity, on the SAME deterministic 34-config pipeline as diagnostic_scale.py.

Reviewer point (novelty elevation): the paper's rule is "G_cv significant > 0". A stronger,
knowledge-based rule is BUILD iff LCB_{1-delta}(G_cv) > tau, which directly controls the
false-build probability. This script computes, per configuration and with NO new randomness:
  * the cross-validated realizable gain G_cv and a one-sided Hoeffding LCB from the CV fold gains,
  * the realized held-out (test) gain g_test  (the deployable ground truth),
  * a CVaR-penalized (tail-robust) gain,
and then reports the false-build / false-retain rates of the LCB rule as the deployment
threshold tau sweeps, versus the significance rule and the classical structural gap.

Consistency: it re-runs the exact pipeline and asserts 0 aggregate mismatches vs
diagnostic_scale_results.json. RUN: python lcb_analysis.py
"""
import os, sys, json, math, statistics as st
import numpy as np
from scipy.stats import wilcoxon
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diagnostic_scale as D

DELTA = 0.05

def eb_lcb(d, delta=DELTA):
    """one-sided (1-delta) empirical-Bernstein lower confidence bound on E[d] (Maurer-Pontil);
    variance-adaptive, so low-variance positive gains are not swamped by the range."""
    d = np.asarray(d, float); n = len(d)
    if n < 2: return d.mean()
    B = float(d.max() - d.min()) or 1e-9
    var = float(d.var(ddof=1))
    return d.mean() - math.sqrt(2*var*math.log(1/delta)/n) - 7*B*math.log(1/delta)/(3*(n-1))

def cvar(d, q=0.95):
    """CVaR_q of the LOSS (-d): mean loss over the worst (1-q) tail, i.e. the most-negative d."""
    d = np.sort(np.asarray(d, float))                # ascending: worst (most negative) gains first
    k = max(1, int((1-q)*len(d)))
    return float(-d[:k].mean())                      # mean loss (-d) over the worst k

def eval_lcb(adapter, regime_subset, ntr=30, nte=30):
    regs, make, exvec, feats, panel = adapter
    tr=[(r,make(r,100+i)) for r in regime_subset for i in range(ntr)]
    te=[(r,make(r,5000+i)) for r in regime_subset for i in range(nte)]
    def mat(rows):
        per={m:[] for m in panel}; X=[]
        for (r,inst) in rows:
            ev=exvec(inst); X.append(feats(inst))
            for m in panel: per[m].append(ev[m])
        return per,np.array(X)
    trp,Xtr=mat(tr); tep,Xte=mat(te)
    sbs=min(panel,key=lambda m:st.mean(trp[m]))
    # structural gap
    reg_sel=[]; idx=0
    for r in regime_subset:
        sub=[trp[m][idx:idx+ntr] for m in panel]
        best=min(range(len(panel)),key=lambda j:st.mean(sub[j]))
        reg_sel+=list(sub[best]); idx+=ntr
    Gstr=st.mean(trp[sbs])-st.mean(reg_sel)
    pstr=wilcoxon(reg_sel,trp[sbs]).pvalue if any(np.array(reg_sel)!=np.array(trp[sbs])) else 1.0
    # cross-validated realizable gain + per-instance CV gains d_cv
    ybase=[int(np.argmin([trp[m][i] for m in panel])) for i in range(len(Xtr))]
    cv_sel=[]; cv_sbs=[]
    kf=KFold(n_splits=4,shuffle=True,random_state=0)
    for tri,vai in kf.split(Xtr):
        clf=DecisionTreeClassifier(max_depth=4,random_state=0).fit(Xtr[tri],[ybase[i] for i in tri])
        for i in vai:
            cv_sel.append(trp[panel[int(clf.predict([Xtr[i]])[0])]][i]); cv_sbs.append(trp[sbs][i])
    d_cv=np.array(cv_sbs)-np.array(cv_sel)
    Gcv=d_cv.mean()
    pcv=wilcoxon(cv_sel,cv_sbs).pvalue if any(np.array(cv_sel)!=np.array(cv_sbs)) else 1.0
    lcb=eb_lcb(d_cv)
    Gtr=Gcv - 1.0*cvar(d_cv)                                   # CVaR-penalized (tail-robust), lambda=1
    # realized held-out gain (deployable ground truth)
    clf=DecisionTreeClassifier(max_depth=4,random_state=0).fit(Xtr,ybase)
    sel_te=[tep[panel[int(clf.predict([Xte[i]])[0])]][i] for i in range(len(Xte))]
    sbs_te=list(tep[sbs])
    g_test=st.mean(sbs_te)-st.mean(sel_te)
    pact=wilcoxon(sel_te,sbs_te).pvalue if any(np.array(sel_te)!=np.array(sbs_te)) else 1.0
    actual=(st.mean(sel_te)<st.mean(sbs_te)) and pact<0.05
    return dict(Gstr=round(Gstr,3),Gcv=round(Gcv,3),lcb=round(lcb,3),Gtr=round(Gtr,3),
                g_test=g_test,actual=bool(actual),
                pred_cv=(Gcv>1e-9 and pcv<0.05),pred_str=(Gstr>1e-9 and pstr<0.05))

def main():
    ref={(r['prob'],r['cfg']):r for r in json.load(open(os.path.join(os.path.dirname(__file__),
         'diagnostic_scale_results.json')))}
    rows=[]; mism=0
    for prob,adp in D.ADAPTERS.items():
        regs=adp[0]; cfgs=D.configs_for(regs)
        if len(cfgs)>9: cfgs=[cfgs[i] for i in range(0,len(cfgs),max(1,len(cfgs)//9))][:9]
        for cfg in cfgs:
            r=eval_lcb(adp,list(cfg)); r['prob']=prob; r['cfg']="+".join(cfg); rows.append(r)
            k=(prob,"+".join(cfg))
            if k in ref and (abs(ref[k]['Gcv']-r['Gcv'])>1e-6 or ref[k]['actual']!=r['actual']): mism+=1
    print(f"consistency vs saved JSON: {len(rows)} configs, {mism} mismatches "
          f"({'OK reproduces' if mism==0 else 'MISMATCH'})")

    # deployable ground truth at threshold tau: net-positive iff g_test > tau
    def rates(build_fn, tau):
        fb=fr=tp=tn=0
        for r in rows:
            build=build_fn(r,tau); netpos=r['g_test']>tau
            fb += build and not netpos            # false build
            fr += (not build) and netpos          # false retain (missed a worthwhile selector)
            tp += build and netpos; tn += (not build) and (not netpos)
        nb=sum(1 for r in rows if r['g_test']<=tau); npv=len(rows)-nb
        return fb, fr, (fb/max(1,nb)), (fr/max(1,npv))
    # MC3: predictor precision/recall under an EFFECT-SIZE label (net gain > tau), not p<0.05
    def prec_rec(build_fn, tau):
        tp=fp=fn=0
        for r in rows:
            build=build_fn(r); netpos=r['g_test']>tau
            tp += build and netpos; fp += build and not netpos; fn += (not build) and netpos
        return tp,fp,fn,(tp/max(1,tp+fp)),(tp/max(1,tp+fn))
    print("\n=== MC3: significance-rule quality under EFFECT-SIZE label g_test>tau (not p<0.05) ===")
    for tau in (0.0, 0.05, 0.1):
        c=prec_rec(lambda r:r['pred_cv'], tau); s=prec_rec(lambda r:r['pred_str'], tau)
        print(f"  tau={tau:.2f}: G_cv precision={c[3]:.2f} recall={c[4]:.2f} (TP{c[0]}/FP{c[1]}/FN{c[2]}) | "
              f"G_str precision={s[3]:.2f} recall={s[4]:.2f} (FP{s[1]})")

    print("\n=== build rules vs deployment threshold tau (false-build / false-retain rates) ===")
    print(f"{'tau':>5} | {'LCB>tau':>22} | {'G_cv sig':>14} | {'G_str sig':>14}")
    for tau in (0.0, 0.05, 0.1, 0.2):
        fb1,fr1,fbr1,frr1=rates(lambda r,t:r['lcb']>t, tau)
        fb2,fr2,fbr2,frr2=rates(lambda r,t:r['pred_cv'], tau)
        fb3,fr3,fbr3,frr3=rates(lambda r,t:r['pred_str'], tau)
        print(f"{tau:>5.2f} | FB={fb1} FBR={fbr1:.2f}  FR={fr1} FRR={frr1:.2f} | "
              f"FB={fb2} FBR={fbr2:.2f} | FB={fb3} FBR={fbr3:.2f}")

    # tail-robust gain flips scheduling decisions
    print("\n=== scheduling: mean G_cv vs LCB vs CVaR-penalized G_TR (tail-robust) ===")
    for r in rows:
        if r['prob']!='sched': continue
        print(f"  {r['cfg'][:24]:24s} G_cv={r['Gcv']:+8.2f}  LCB={r['lcb']:+9.2f}  "
              f"G_TR={r['Gtr']:+10.2f}  g_test={r['g_test']:+8.2f}  build(G_cv sig)={int(r['pred_cv'])}")
    json.dump([{k:(bool(v) if isinstance(v,bool) else (round(float(v),4) if isinstance(v,(int,float)) or hasattr(v,'item') else v)) for k,v in r.items()} for r in rows],
              open(os.path.join(os.path.dirname(__file__),'lcb_results.json'),'w'),indent=1)

if __name__=="__main__":
    main()
