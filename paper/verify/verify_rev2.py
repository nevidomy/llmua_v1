import numpy as np
from scipy import optimize, stats
np.set_printoptions(precision=5, suppress=True)
N = np.array([5326464, 9833664, 16416768, 33826176], float)
N_fix = N.copy(); N_fix[2] = 16422144
Lm = np.array([1.50010, 1.36550, 1.27222, 1.15967])   # masked 20x
Lc = np.array([1.50484, 1.37098, 1.27492, 1.16253])   # causal 20x
sd = np.array([0.00287, 0.00109, 0.00091, 0.00112]); n = np.array([5,5,5,3])
Ntgt = 124278528

def fit(Nv, L, w=None):
    f = lambda p, x: p[0] + p[1]*x**(-p[2])
    def res(p):
        r = f(p, Nv) - L
        return r if w is None else r*np.sqrt(w)
    best=None
    for a0 in [0.5,0.6,0.7,0.8]:
        for b0 in [0.2,0.29,0.35]:
            p0=[a0, (L[0]-a0)*Nv[0]**b0, b0]
            try:
                s = optimize.least_squares(res, p0, max_nfev=20000, xtol=1e-14, ftol=1e-14)
            except Exception: continue
            if best is None or s.cost < best.cost: best=s
    p = best.x
    return p, f(p, Nv)-L, f

print("=== Ray fits (masked 20x) ===")
for name, w, Nv in [("unweighted", None, N), ("1/sd^2", 1/sd**2, N), ("n/sd^2", n/sd**2, N), ("unweighted, P3 N fixed", None, N_fix)]:
    p, r, f = fit(Nv, Lm, w)
    print(f"{name:26s} E={p[0]:.4f} A={p[1]:.2f} alpha={p[2]:.4f} maxres={abs(r).max():.5f} tgt={f(p,Ntgt):.4f} 500M={f(p,5e8):.4f} 1B={f(p,1e9):.4f}")
# paper's stated ray
pp=[0.6803,73.16,0.290]; fp=lambda x: pp[0]+pp[1]*x**(-pp[2])
print("paper ray: tgt",round(fp(Ntgt),4),"500M",round(fp(5e8),4),"1B",round(fp(1e9),4),"512M",round(fp(512e6),4),"resid",np.round(fp(N)-Lm,5))
print("iso-compute: ray at 248.5M =", round(fp(248.5e6),4))
# LOO (exact 3-pt fits)
print("\n=== LOO (3-point exact fits) ===")
for drop in range(4):
    m = np.ones(4,bool); m[drop]=False
    p,r,f = fit(N[m], Lm[m])
    print(f"drop P{drop+1}: tgt={f(p,Ntgt):.4f} 500M={f(p,5e8):.4f} 1B={f(p,1e9):.4f}  (E={p[0]:.4f},alpha={p[2]:.3f}) maxres={abs(r).max():.1e}")
# 5-rung refit with target point (corrected / uncorrected)
print("\n=== 5-rung refits ===")
for lab, Lt in [("uncorrected 1.0062",1.0062),("corr -0.0057",1.0062-0.0057),("corr -0.0073",1.0062-0.0073),("corr -0.0077",1.0062-0.0077)]:
    p,r,f = fit(np.append(N,Ntgt), np.append(Lm,Lt))
    print(f"{lab:20s} 500M={f(p,5e8):.4f} 1B={f(p,1e9):.4f} maxres={abs(r).max():.4f}")

print("\n=== masked - causal means ===", np.round(Lm-Lc,5))
print("\n=== LR transfer exponents (4-point OLS in log-log) ===")
d = np.array([128,192,256,384.]); mu=np.array([8.16e-3,5.45e-3,3.63e-3,2.00e-3]); ad=np.array([4.93e-3,4.00e-3,3.00e-3,2.14e-3])
for lab, lr in [("muon",mu),("adamw",ad)]:
    s,i,r,p,se = stats.linregress(np.log(d), np.log(lr)); res=np.log(lr)-(i+s*np.log(d))
    loo=[stats.linregress(np.log(np.delete(d,k)), np.log(np.delete(lr,k)))[0] for k in range(4)]
    print(f"{lab}: slope={s:.3f} se={se:.3f} resid_sd={res.std(ddof=2):.3f} LOO={np.round(loo,3)} P1->P4 two-point={np.log(lr[3]/lr[0])/np.log(3):.3f} P1->target(768) pred={lr[0]*(768/128)**s:.2e}")
print("adamw P1->target exponent from 4.93e-3 -> 1.19e-3:", round(np.log(1.19e-3/4.93e-3)/np.log(6),3), " muon 8.16e-3->0.85e-3 (20x extrap):", round(np.log(0.85e-3/8.16e-3)/np.log(6),3))

print("\n=== LR penalty ===")
for a in [0.014,0.018,0.019]:
    print(f"a={a}: 20x penalty (0.85/0.45) = {a*np.log(0.85/0.45)**2:.4f};  40x (0.60/0.45)={a*np.log(0.60/0.45)**2:.4f} (0.62/0.45)={a*np.log(0.62/0.45)**2:.4f}")

print("\n=== Target horizon fits & repetition ===")
L20,L40,L80=1.0062,0.9747,0.9549
def abfit(l20,l40,l80):
    g1=l20-l40; g2=l40-l80; beta=np.log(g1/g2)/np.log(2)
    # L = E + A r^-beta, r=1,2,4  => g1 = A(1-2^-b)
    A = g1/(1-2**-beta); E = l20-A
    return beta,A,E,g1,g2
for lab,(p20,p40) in [("uncorr",(0,0)),("corr lo",(0.0057,0.0012)),("corr hi",(0.0073,0.0016)),("corr hi2",(0.0077,0.0016))]:
    b,A,E,g1,g2 = abfit(L20-p20,L40-p40,L80)
    print(f"{lab:8s} beta={b:.3f} A={A:.4f} E={E:.4f} gain20->40={g1:.4f} 40->80={g2:.4f} gain20->80={g1+g2:.4f} vs ladder 0.0387: {100*((g1+g2)/0.0387-1):+.0f}%")
    for beta in [0.50,0.60,0.67]:
        pred=g1*2**-beta; print(f"      beta={beta}: predicted 40->80 {pred:.4f}, measured {g2:.4f}, measured-pred {g2-pred:+.4f}")
print("ladder A trend: A=",[0.1102,0.1024,0.0890,0.0844])
Av=np.array([0.1102,0.1024,0.0890,0.0844]); s,i,_,_,_=stats.linregress(np.log(N),np.log(Av)); print("A ∝ N^%.3f -> target A pred %.4f"%(s,np.exp(i)*Ntgt**s))
# pre-registered 40x/80x predictions from A=0.0683, beta=0.60 at L20=1.0091
for beta in [0.60,0.6069]:
    A=0.0683; print(f"prereg beta={beta}: L40={1.0091-A*(1-2**-beta):.4f} L80={1.0091-A*(1-4**-beta):.4f}")
print("P1 20% more tokens gain: A=0.1102,beta=0.546 ->", round(0.1102*(1-1.2**-0.546),4))

print("\n=== Table 4 ===")
base8=np.array([1.5400,1.5417,1.5441]); base16=np.array([1.5160,1.5183,1.5169]); base32=np.array([1.5153,1.5204,1.5181])
A_=np.array([1.5344,1.5344,1.5320]); C_=np.array([1.5420,1.5419,1.5387]); D_=np.array([1.5372,1.5401,1.5420]); E_=np.array([1.5291,1.5253,1.5349]); F_=np.array([1.5127,1.5159,1.5141])
pen=base8-base16; print("halving penalty per seed",pen, "mean %.4f sd %.4f t %.1f"%(pen.mean(),pen.std(ddof=1),pen.mean()/(pen.std(ddof=1)/np.sqrt(3))))
dbl=base32-base16; print("doubling", dbl, "mean %.4f t %.1f"%(dbl.mean(), dbl.mean()/(dbl.std(ddof=1)/np.sqrt(3))))
recs={}
for lab,arr in [("A wd",A_),("C warm",C_),("D lr",D_),("E betas",E_),("F all",F_)]:
    r=base8-arr; recs[lab]=r; print(f"{lab:8s} recovery per seed {r} mean {r.mean():.4f} share {100*r.mean()/pen.mean():.0f}% t {r.mean()/(r.std(ddof=1)/np.sqrt(3)):.1f}")
ssum=recs["A wd"]+recs["C warm"]+recs["D lr"]+recs["E betas"]; print("sum of singles per seed",ssum,"mean %.4f"%ssum.mean())
inter=recs["F all"]-ssum; print("interaction per seed",inter,"mean %.4f sd %.4f se %.4f t %.2f"%(inter.mean(),inter.std(ddof=1),inter.std(ddof=1)/np.sqrt(3),inter.mean()/(inter.std(ddof=1)/np.sqrt(3))))
fv=F_-base16; print("F vs mbs16 per seed",fv,"mean %.4f t %.1f"%(fv.mean(), fv.mean()/(fv.std(ddof=1)/np.sqrt(3))))
# Muon
m8=np.array([1.5295,1.5386,1.5309,1.5299,1.5328]); m16=np.array([1.5007,1.5074,1.5031,1.5059,1.5071])
dm=m8-m16; print("Muon causal halving per seed",dm,"mean %.4f sd %.4f t %.1f"%(dm.mean(),dm.std(ddof=1),dm.mean()/(dm.std(ddof=1)/np.sqrt(5))))
print("Muon all-four mbs8 mean", np.mean([1.4989,1.5040,1.5000]), "vs masked mbs16 1.5001 ->", round(np.mean([1.4989,1.5040,1.5000])-1.5001,4), "recovery vs 0.0275:", round((1.5001+0.0275-np.mean([1.4989,1.5040,1.5000]))/0.0275,3))
print("timescale check: (eta/sqrt2)(w/sqrt2)=eta w/2 -> token timescale B/(eta w) preserved at B/2:", (1/np.sqrt(2))**2)

print("\n=== N_total ===")
V=32768
for lab,(dd,L,h) in {"P1":(128,6,320),"P2":(192,8,512),"P3":(256,10,704),"P4":(384,12,1024),"tgt":(768,14,2048)}.items():
    emb=V*dd; lay=L*(4*dd*dd+3*dd*h); rms=L*2*dd+dd; qk=L*2*64
    print(f"{lab}: emb+layers={emb+lay:,} +rms={emb+lay+rms:,} +qk={emb+lay+rms+qk:,}")

print("\n=== FLOPs T=4096 at target (fwd per token) ===")
dd,L,h,T=768,14,2048,2048
mat=2*L*(4*dd*dd+3*dd*h); att=lambda T: L*2*dd*T  # 2dT per layer (causal avg)
head=2*V*dd
for T2 in [2048,4096]:
    tot=mat+att(T2)+head; print(f"T={T2}: layers {mat/1e6:.1f}M attn {att(T2)/1e6:.1f}M head {head/1e6:.1f}M total {tot/1e6:.1f}M")
print("ratio with head:", (mat+att(4096)+head)/(mat+att(2048)+head), " without head:", (mat+att(4096))/(mat+att(2048)))

print("\n=== beta sensitivities (P2) ===")
l20,l40,l80=1.36550,1.33104,1.30838; g1=l20-l40; g2=l40-l80
print("beta=%.4f dβ/dL20=%.1f dβ/dL40=%.1f dβ/dL80=%.1f"%(np.log(g1/g2)/np.log(2), 1/(np.log(2)*g1), -(1/g1+1/g2)/np.log(2), 1/(np.log(2)*g2)))

print("\n=== chi2 constant beta ===")
b=np.array([0.546,0.605,0.646,0.637]); se=np.array([0.028,0.004,0.019,0.021]); w=1/se**2; m=(b*w).sum()/w.sum(); chi=(((b-m)/se)**2).sum()
print("weighted mean %.4f chi2 %.2f p %.4f"%(m,chi,1-stats.chi2.cdf(chi,3)))
b2=np.array([0.5455,0.6049,0.6461,0.6366]); se2=np.array([0.0278,0.0044,0.0185,0.0208]); w=1/se2**2; m=(b2*w).sum()/w.sum(); chi=(((b2-m)/se2)**2).sum(); print("unrounded: mean %.4f chi2 %.2f p %.4f"%(m,chi,1-stats.chi2.cdf(chi,3)))
print("sign test 5/5: one-sided %.4f two-sided %.4f"%(0.5**5, 2*0.5**5))
d3=np.array([0.0103,0.0149,0.0124]); print("QK-norm paired t:", stats.ttest_1samp(d3,0))
print("\n=== MDE power check (k=3, sigma .00106, independent arms) ===")
for k in [3,5,8]:
    sig=0.00106; se_=sig*np.sqrt(2/k); mde=(stats.t.ppf(0.975,k-1)+stats.t.ppf(0.80,k-1))*se_
    nc=mde/se_; crit=stats.t.ppf(0.975,k-1); power=1-stats.nct.cdf(crit,k-1,nc)+stats.nct.cdf(-crit,k-1,nc)
    print(f"k={k}: MDE={mde:.4f} power(noncentral t)={power:.3f}")
print("\n=== Muennighoff effective data at 4 epochs, U=60B, R_D*=15.4 ===")
U=60; RD=3; print("D' =", U + U*15.39*(1-np.exp(-RD/15.39)))
print("\n=== sd/var ===", 0.00101/0.00219, (0.00101/0.00219)**2, (0.00219/0.00101)**2)
print("subset offsets P1-P3 spread:", 0.0305-0.0269, " N ratio P3/P1:", 16416768/5326464)
print("\n=== uk4b token/char ===  21.4B tok * ? ; ours 9.94B tok * 4.03 c/t = %.1fB chars"%(9.94*4.0305))
