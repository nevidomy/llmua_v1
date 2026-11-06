import numpy as np
from scipy import optimize, stats
np.set_printoptions(precision=5, suppress=True)
# ---------- shares with delta-method CIs (df 2) ----------
b8=np.array([1.5400,1.5417,1.5441]); b16=np.array([1.5160,1.5183,1.5169])
arms={"EMA":np.array([1.5291,1.5253,1.5349]),"wd":np.array([1.5344,1.5344,1.5320]),"lr":np.array([1.5372,1.5401,1.5420]),"warm":np.array([1.5420,1.5419,1.5387]),"all4":np.array([1.5127,1.5159,1.5141])}
pen=b8-b16; k=3; tq=stats.t.ppf(0.975,2)
print("=== shares (ratio of means, delta method, mbs-8 arm shared) ===")
for lab,a in arms.items():
    rec=b8-a; R=rec.mean()/pen.mean()
    # per-seed influence: d(R)= (dr/P - r dP/P^2); use covariance of (rec,pen)
    X=np.vstack([rec,pen]); C=np.cov(X)/k
    g=np.array([1/pen.mean(), -rec.mean()/pen.mean()**2])
    se=np.sqrt(g@C@g)
    print(f"{lab:5s} share {100*R:5.1f}%  95% CI [{100*(R-tq*se):6.1f}%, {100*(R+tq*se):6.1f}%]   recovery {rec.mean():.4f} CI [{rec.mean()-tq*rec.std(ddof=1)/np.sqrt(k):+.5f},{rec.mean()+tq*rec.std(ddof=1)/np.sqrt(k):+.5f}]")
print("EMA vs wd paired t:", stats.ttest_rel(b8-arms["EMA"], b8-arms["wd"]))
ssum=sum(b8-arms[x] for x in ["EMA","wd","lr","warm"]); print("sum singles %.5f = %.1f%% of penalty, %.1f%% of joint"%(ssum.mean(),100*ssum.mean()/pen.mean(),100*ssum.mean()/(b8-arms['all4']).mean()))
inter=(b8-arms["all4"])-ssum; se=inter.std(ddof=1)/np.sqrt(3); print("interaction %.5f CI [%.5f, %.5f]"%(inter.mean(),inter.mean()-tq*se,inter.mean()+tq*se))
dbl=np.array([1.5153,1.5204,1.5181])-b16; se=dbl.std(ddof=1)/np.sqrt(3); print("doubling CI [%.4f, %.4f]"%(dbl.mean()-tq*se, dbl.mean()+tq*se))
# ---------- LR penalty sensitivity ----------
print("\n=== 20x/40x penalties by curvature ===")
for a in [0.0120,0.0137,0.014,0.018,0.0192,0.023]:
    p20=a*np.log(0.850/0.450)**2; p40=a*np.log(0.605/0.450)**2
    print(f"a={a:.4f}: p20={p20:.4f} p40={p40:.4f}")
L20,L40,L80=1.0062,0.9747,0.9549
def fit3(l20,l40,l80):
    g1=l20-l40; g2=l40-l80; b=np.log(g1/g2)/np.log(2); A=g1/(1-2**-b); return b,A,g1,g2
print("target beta/A across corrections (same a both horizons):")
for a in [0,0.0120,0.0137,0.014,0.018,0.0192,0.023]:
    p20=a*np.log(0.850/0.450)**2; p40=a*np.log(0.605/0.450)**2
    b,A,g1,g2=fit3(L20-p20,L40-p40,L80); print(f"  a={a:.4f}: corrected L20={L20-p20:.4f} err20={L20-p20-1.0091:+.4f} ({100*(L20-p20-1.0091)/1.0091:+.2f}%) L40={L40-p40:.4f} err40={L40-p40-0.9858:+.4f} beta={b:.3f} A={A:.4f} gain20-80={g1+g2:.4f}")
# 80x itself detuned if gamma=-0.20 (optimum 0.644e-3 vs 0.45e-3)
for a in [0.0127,0.0192]:
    print("  80x self-penalty if optimum 0.644e-3: a=%.4f -> %.4f"%(a, a*np.log(0.644/0.450)**2))
# ---------- ladder detuning (N1) ----------
print("\n=== ladder 80x detuning from the probes ===")
# P2: 3pt parabola vertex 4.196e-3 a=0.01917, 20x vertex 5.541e-3 ; P3: vertex 1.936e-3 a=0.01268, 20x vertex 3.65e-3
for lab,v80,v20,a,L in [("P2",4.196e-3,5.541e-3,0.01917,(1.36550,1.33104,1.30838)),("P3",1.936e-3,3.65e-3,0.01268,(1.27222,1.23934,1.21866))]:
    gam=np.log(v80/v20)/np.log(4); p80=a*np.log(v20/v80)**2; v40=v20*2**gam; p40=a*np.log(v20/v40)**2
    b0,A0,_,_=fit3(*L); b1,A1,g1,g2=fit3(L[0],L[1]-p40,L[2]-p80)
    print(f"{lab}: gamma={gam:.3f} ratio80={v20/v80:.2f} p80={p80:.4f} p40={p40:.4f} | beta {b0:.3f}->{b1:.3f} A {A0:.4f}->{A1:.4f} gain20-80 {L[0]-L[2]:.4f}->{g1+g2:.4f} ({100*((g1+g2)/(L[0]-L[2])-1):+.1f}%)")
# ---------- LOO & refits with corrected P3 count ----------
N=np.array([5326464,9833664,16422144,33826176],float); Nold=N.copy(); Nold[2]=16416768
Lm=np.array([1.50010,1.36550,1.27222,1.15967]); Ntgt=124278528
def fit(Nv,L):
    f=lambda p,x:p[0]+p[1]*x**(-p[2])
    best=None
    for a0 in [0.5,0.6,0.7,0.8]:
        for b0 in [0.2,0.29,0.35]:
            s=optimize.least_squares(lambda p:f(p,Nv)-L,[a0,(L[0]-a0)*Nv[0]**b0,b0],max_nfev=20000,xtol=1e-14,ftol=1e-14)
            if best is None or s.cost<best.cost: best=s
    return best.x,f
print("\n=== LOO with corrected P3 count ===")
for lab,Nv in [("old",Nold),("new",N)]:
    out=[]
    for d in range(4):
        m=np.ones(4,bool); m[d]=False; p,f=fit(Nv[m],Lm[m]); out.append((f(p,Ntgt),f(p,5e8),f(p,1e9)))
    out=np.array(out); print(lab, "tgt",np.round(out[:,0],4),"500M",np.round(out[:,1],4),"1B",np.round(out[:,2],4))
    p,f=fit(Nv,Lm); print("   unweighted fit E=%.4f A=%.1f al=%.4f tgt=%.4f"%(p[0],p[1],p[2],f(p,Ntgt)))
print("\n=== five-rung refit over corrected 20x points, with LOO ===")
for Lt in [0.9969,0.9985,1.0005,1.0013]:
    Nv=np.append(N,Ntgt); Lv=np.append(Lm,Lt); p,f=fit(Nv,Lv); loo=[]
    for d in range(5):
        m=np.ones(5,bool); m[d]=False; q,g=fit(Nv[m],Lv[m]); loo.append((g(q,5e8),g(q,1e9)))
    loo=np.array(loo); print(f"Lt={Lt}: 500M={f(p,5e8):.4f} LOO[{loo[:,0].min():.4f},{loo[:,0].max():.4f}]  1B={f(p,1e9):.4f} LOO[{loo[:,1].min():.4f},{loo[:,1].max():.4f}]")
# ---------- direct 80x rays ----------
print("\n=== direct 80x rays (fit on 80x endpoints), target measured ===")
sets={"val":([1.44138,1.30838,1.21866,1.11020],0.9549),"docs":([1.5395,1.4131,1.3271,1.2234],1.0756),"paras":([1.6534,1.5294,1.4442,1.3412],1.1943),"sents":([1.8351,1.7129,1.6308,1.5303],1.3936)}
for lab,(L,meas) in sets.items():
    L=np.array(L); p,f=fit(N,L); loo=[]
    for d in range(4):
        m=np.ones(4,bool); m[d]=False; q,g=fit(N[m],L[m]); loo.append(g(q,Ntgt))
    pred=f(p,Ntgt); print(f"{lab:6s} pred={pred:.4f} meas={meas:.4f} err={meas-pred:+.4f} ({100*(meas-pred)/pred:+.2f}%) LOO[{min(loo):.4f},{max(loo):.4f}] {'inside' if min(loo)<=meas<=max(loo) else 'OUTSIDE'}")
# ---------- masking exponents ----------
print("\n=== masking effect exponents ===")
for lab,e in [("a (P1 pre-retune)",[0.0063,0.0045,0.0033,0.0030]),("b (5-seed sets)",[0.0047,0.0055,0.0027,0.0029]),("paper hybrid",[0.0048,0.0045,0.0033,0.0030])]:
    s,i,_,_,_=stats.linregress(np.log(N),np.log(e)); s3,i3,_,_,_=stats.linregress(np.log(N[1:]),np.log(e[1:]))
    print(f"{lab:18s} slope4={s:.3f} target={np.exp(i)*Ntgt**s:.4f} | P2-P4 slope={s3:.3f} target={np.exp(i3)*Ntgt**s3:.4f}")
# ---------- chi2 variants ----------
print("\n=== chi2 variants ===")
b=np.array([0.5455,0.6049,0.6461,0.6366]); se=np.array([0.0278,0.0044,0.0185,0.0208])
def chi(bb,ss):
    w=1/ss**2; m=(bb*w).sum()/w.sum(); c=(((bb-m)/ss)**2); return c.sum(),c,m
c,contrib,m=chi(b,se); print("all: chi2=%.2f p=%.4f contrib="%(c,1-stats.chi2.cdf(c,3)),np.round(contrib,2))
for d,lab in [(0,"no P1"),(1,"no P2")]:
    m_=np.ones(4,bool); m_[d]=False; c,_,_=chi(b[m_],se[m_]); print(f"{lab}: chi2={c:.2f} p={1-stats.chi2.cdf(c,2):.4f}")
print("P1 vs P3: diff %.3f, SE %.3f, z %.2f"%(b[2]-b[0], np.hypot(se[0],se[2]), (b[2]-b[0])/np.hypot(se[0],se[2])))
print("P2 per-seed sd implied by SE 0.0044 at n=5:", 0.0044*np.sqrt(5), " others:", np.round(se*np.sqrt([5,5,5,3]),3))
print("beta from (0.0347,0.0234):", np.log(0.0347/0.0234)/np.log(2))
# ---------- Muennighoff predictions at P2 ----------
print("\n=== Muennighoff-type predicted penalties at P2 (A=.1024, beta=.605, r=4), R*_D=15.39 ===")
A,beta,RD=0.1024,0.605,15.39
for lam in [4,8,16]:
    f=1+RD*(1-np.exp(-(lam-1)/RD)); Dp=4*f/lam; print(f"lambda={lam}: D'/D={f/lam:.3f} penalty={A*(Dp**-beta-4**-beta):.4f}")
print("unique exposure: pool 787M at lambda 1:", 787*(1-np.exp(-1)), " pool 2522M at 786.8M:", 2522*(1-np.exp(-786.8/2522)))
# ---------- misc ----------
print("\nF P1 vs pooled:", 0.00287**2/0.00103**2, "one-sided p", 1-stats.f.cdf(0.00287**2/0.00103**2,4,10), "two-sided", 2*(1-stats.f.cdf(0.00287**2/0.00103**2,4,10)))
print("census sum:", 80.26+17.12+1.59, "fertility 16k->32k:", 4.008/3.618-1, "32k->49k per doubling extrapolated:", (4.219/4.008)**(1/np.log2(49152/32768))-1)
print("Muon sd ratio 0.00424/0.00237 =", 0.00424/0.00237, "var", (0.00424/0.00237)**2)
print("1/ln2^2 =", 1/np.log(2)**2, "; 0.7223*2.0814=", 0.7223/np.log(2)**2)
print("loose match: 2615+1217 =", 2615+1217)
print("branch vs scratch P2-P4 means: P2", np.mean([1.3664,1.3664,1.3663])-1.36550, "P3", np.mean([1.2718,1.2719,1.2721])-1.27222, "P4", 1.1600-1.15967)
print("tight masked sds: P2", np.std([1.3664,1.3664,1.3663],ddof=1), "P3", np.std([1.2718,1.2719,1.2721],ddof=1))
import numpy as np
from scipy import optimize
N=np.array([5326464,9833664,16422144,33826176],float); Lm=np.array([1.50010,1.36550,1.27222,1.15967]); Ntgt=124278528
def fit(Nv,L):
    f=lambda p,x:p[0]+p[1]*x**(-p[2]); best=None
    for a0 in [0.5,0.6,0.7,0.8]:
        for b0 in [0.2,0.29,0.35]:
            s=optimize.least_squares(lambda p:f(p,Nv)-L,[a0,(L[0]-a0)*Nv[0]**b0,b0],max_nfev=20000,xtol=1e-14,ftol=1e-14)
            if best is None or s.cost<best.cost: best=s
    return best.x,f
for Lt in [1.0062,1.0013,0.9969]:
    Nv=np.append(N,Ntgt); Lv=np.append(Lm,Lt); p,f=fit(Nv,Lv); loo=[]
    for d in range(5):
        m=np.ones(5,bool); m[d]=False; q,g=fit(Nv[m],Lv[m]); loo.append((g(q,5e8),g(q,1e9)))
    loo=np.array(loo); print(f"Lt={Lt}: 500M={f(p,5e8):.4f} LOO[{loo[:,0].min():.4f},{loo[:,0].max():.4f}] 1B={f(p,1e9):.4f} LOO[{loo[:,1].min():.4f},{loo[:,1].max():.4f}]  512M={f(p,512e6):.4f}  E={p[0]:.4f} al={p[2]:.3f} maxres={np.abs(f(p,Nv)-Lv).max():.4f}")
print("±1.6% of 0.894:", 0.894*0.984, 0.894*1.016, " of 0.853:", 0.853*0.984, 0.853*1.016)
# A(1-4^-b) bracket with A in [0.085,0.091], beta in [0.45,0.67]
for A in [0.085,0.091]:
    for b in [0.45,0.67]:
        print("A=%.3f beta=%.2f gain=%.4f"%(A,b,A*(1-4**-b)))
