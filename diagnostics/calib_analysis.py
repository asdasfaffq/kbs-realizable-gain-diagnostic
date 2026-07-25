#!/usr/bin/env python3
"""
Calibration / non-inferiority / tail-risk analysis for the selection diagnostic.

Reuses the EXACT deterministic pipeline of diagnostic_scale.py (same adapters, seeds, folds), but
additionally captures, per configuration, the CONTINUOUS realized test gain g_test and the
per-instance test gains d_i = cost(SBS,i) - cost(selector,i). It then:
  1. verifies Gstr/Gcv/actual reproduce diagnostic_scale_results.json (consistency guard);
  2. calibrates the predictor G_cv against the realized g_test (OLS slope/intercept, Spearman, MAE);
  3. reports tail risk (CVaR_0.95 of -d) and a non-inferiority (TOST-style) check on scheduling.
No new randomness; nothing is fabricated. RUN: python calib_analysis.py
"""
import os, sys, json, statistics as st
import numpy as np
from scipy.stats import wilcoxon, spearmanr
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diagnostic_scale as D   # module-level ADAPTERS, configs_for

def eval_ext(adapter, regime_subset, ntr=30, nte=30):
    """Verbatim copy of diagnostic_scale.eval_config, extended to also return g_test and d_te."""
    regs, make, exvec, feats, panel = adapter
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
    reg_sel=[]; idx=0
    for r in regime_subset:
        sub=[trp[m][idx:idx+ntr] for m in panel]
        best=min(range(len(panel)),key=lambda j:st.mean(sub[j]))
        reg_sel+=list(sub[best]); idx+=ntr
    Gstr=st.mean(trp[sbs])-st.mean(reg_sel)
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
    pstr=wilcoxon(reg_sel,trp[sbs]).pvalue if any(np.array(reg_sel)!=np.array(trp[sbs])) else 1.0
    clf=DecisionTreeClassifier(max_depth=4,random_state=0).fit(Xtr,ybase)
    sel_te=[tep[panel[int(clf.predict([Xte[i]])[0])]][i] for i in range(len(Xte))]
    sbs_te=list(tep[sbs])
    pact=wilcoxon(sel_te,sbs_te).pvalue if any(np.array(sel_te)!=np.array(sbs_te)) else 1.0
    actual=(st.mean(sel_te)<st.mean(sbs_te)) and pact<0.05
    d_te=[sbs_te[i]-sel_te[i] for i in range(len(sel_te))]        # per-instance realized gain
    g_test=st.mean(d_te)
    return dict(Gstr=round(Gstr,3),Gcv=round(Gcv,3),pcv=round(pcv,3),pstr=round(pstr,3),
                actual=bool(actual),pred_cv=(Gcv>1e-9 and pcv<0.05),pred_str=(Gstr>1e-9 and pstr<0.05),
                g_test=g_test, d_te=d_te, sbs=sbs)

def main():
    ref={ (r['prob'],r['cfg']):r for r in json.load(open(os.path.join(os.path.dirname(__file__),
          'diagnostic_scale_results.json'))) }
    rows=[]; mism=0
    for prob,adp in D.ADAPTERS.items():
        regs=adp[0]; cfgs=D.configs_for(regs)
        if len(cfgs)>9: cfgs=[cfgs[i] for i in range(0,len(cfgs),max(1,len(cfgs)//9))][:9]
        for cfg in cfgs:
            try:
                r=eval_ext(adp,list(cfg)); r['prob']=prob; r['cfg']="+".join(cfg); rows.append(r)
                k=(prob,"+".join(cfg))
                if k in ref and (abs(ref[k]['Gcv']-r['Gcv'])>1e-6 or abs(ref[k]['Gstr']-r['Gstr'])>1e-6
                                 or ref[k]['actual']!=r['actual']): mism+=1
            except Exception as e:
                print("ERR",prob,cfg,e)
    print(f"consistency vs saved JSON: {len(rows)} configs, {mism} aggregate mismatches "
          f"({'OK reproduces' if mism==0 else 'MISMATCH -- investigate'})")

    # ---- calibration: realized g_test vs predictor G_cv ----
    x=np.array([r['Gcv'] for r in rows]); y=np.array([r['g_test'] for r in rows])
    beta,alpha=np.polyfit(x,y,1); yhat=alpha+beta*x
    ss=1-np.sum((y-yhat)**2)/np.sum((y-y.mean())**2)
    rho=spearmanr(x,y).correlation; mae=np.mean(np.abs(y-x))
    print(f"\n=== calibration of G_cv against realized test gain (n={len(rows)}) ===")
    print(f"  OLS  g_test = {alpha:+.3f} + {beta:.3f}*G_cv     (ideal alpha=0, beta=1)")
    print(f"  R^2={ss:.3f}   Spearman={rho:.3f}   MAE|g_test-G_cv|={mae:.3f}")
    sign=np.mean(np.sign(x)==np.sign(y)); print(f"  sign agreement={sign:.2f}")

    # ---- tail risk + non-inferiority on scheduling configs ----
    print("\n=== scheduling: tail risk & non-inferiority (per-instance realized gain d) ===")
    for r in rows:
        if r['prob']!='sched': continue
        d=np.array(r['d_te'])
        cvar=-np.mean(np.sort(d)[:max(1,int(0.05*len(d)))])   # CVaR_0.95 of the LOSS -d
        # non-inferiority at margin eps=0.5: is selector NOT worse than SBS by more than eps? one-sided
        eps=0.5; ni_lcb=d.mean()-1.96*d.std(ddof=1)/np.sqrt(len(d))
        ni = "PASS" if ni_lcb>-eps else "FAIL"
        print(f"  {r['cfg'][:26]:26s} g_test={r['g_test']:+7.2f}  CVaR95(loss)={cvar:7.2f}  "
              f"NI(eps={eps}) LCB={ni_lcb:+.2f} -> {ni}")
    json.dump([{k:v for k,v in r.items() if k!='d_te'} for r in rows],
              open(os.path.join(os.path.dirname(__file__),'calib_results.json'),'w'),indent=1)

if __name__=="__main__":
    main()
