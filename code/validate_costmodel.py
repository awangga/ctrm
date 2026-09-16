#!/usr/bin/env python3
"""Validasi cost model microbench J/step = a (P D)^b terhadap 49 run faithful (fase BM, respons reviewer).
Prediksi: J_mb(h,D) x batch/128 (microbench: batch 128, seq 81). Realised: smi_net_Wh*3600/realised_steps.
Usage: python3 validate_costmodel.py [data_root] [microbench_csv]"""
import csv, json, glob, os, sys, math, statistics as st
ROOT=sys.argv[1] if len(sys.argv)>1 else os.path.dirname(os.path.abspath(__file__))
MB=sys.argv[2] if len(sys.argv)>2 else os.path.join(ROOT,"..","kalibrasi","microbench_results.csv")
if not os.path.exists(MB): MB=os.path.join(ROOT,"microbench_results.csv")
mb=list(csv.DictReader(open(MB)))
def f(r,*ks):
    for k in ks:
        if k in r and r[k] not in ("",None): return float(r[k])
    raise KeyError(ks)
P={}; J={}
for r in mb:
    h=int(f(r,"hidden","hidden_size")); D=int(f(r,"D_eff","deff","depth")); P[h]=f(r,"params_M","params","params_m"); J[(h,D)]=f(r,"J_per_step","J_step")
xs=[math.log(P[h]*D) for (h,D) in J]; ys=[math.log(v) for v in J.values()]
def fit(xs,ys):
    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
    b=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs); a=my-b*mx
    ss=sum((y-(a+b*x))**2 for x,y in zip(xs,ys)); tot=sum((y-my)**2 for y in ys); return a,b,1-ss/tot
a,b,r2=fit(xs,ys)
small=[(x,y) for x,y in zip(xs,ys) if math.exp(x)<50]; a_s,b_s,_=fit([x for x,_ in small],[y for _,y in small])
# two-exponent fit J = c P^p D^q (least squares in log space)
import itertools
X=[(1.0,math.log(P[h]),math.log(D)) for (h,D) in J]; Y=ys
# solve normal equations 3x3
def solve(X,Y):
    import copy
    n=3; A=[[sum(x[i]*x[j] for x in X) for j in range(n)] for i in range(n)]; B=[sum(x[i]*y for x,y in zip(X,Y)) for i in range(n)]
    for i in range(n):
        piv=A[i][i]
        for j in range(i,n): A[i][j]/=piv
        B[i]/=piv
        for k in range(n):
            if k!=i:
                fac=A[k][i]
                for j in range(i,n): A[k][j]-=fac*A[i][j]
                B[k]-=fac*B[i]
    return B
c,pexp,qexp=solve(X,Y)
out=[]
out.append("# Validasi cost model microbench terhadap run faithful (auto-generated)\n")
out.append(f"Fit tunggal 14 titik: b={b:.3f}, R2={r2:.3f}. Fit pada 8 titik P*D<50: b={b_s:.3f}. Dua eksponen: J ~ P^{pexp:.2f} D^{qexp:.2f}.\n")
out.append("| run | task | h | D | batch | steps_real | J/step realised | J/step predicted (mb x batch/128) | error |\n|---|---|---|---|---|---|---|---|---|")
errs={}
for d,task,seq in [("recipe_out","Sudoku",81),("maze_depth_out","Maze",900),("arc_depth_out","ARC",900)]:
    for r in csv.DictReader(open(os.path.join(ROOT,d,"recipe_summary.csv"))):
        if "baseline" in r["tag"]: continue
        h=int(r["hidden"]); D=int(r["D_eff"]); B=int(r["batch"])
        mx=0
        for ln in open(os.path.join(ROOT,d,f"progress_{r['tag']}.jsonl")):
            if '"phase": "train"' in ln:
                try: mx=max(mx,json.loads(ln)["step"])
                except: pass
        real=float(r["smi_net_Wh"])*3600/mx
        pred=math.exp(a+b*math.log(P[h]*D))*B/128 if (h,D) not in J else J[(h,D)]*B/128
        e=(pred-real)/real*100; errs.setdefault(task,[]).append(e)
        out.append(f"| {r['tag']} | {task} (seq {seq}) | {h} | {D} | {B} | {mx} | {real:.2f} | {pred:.2f} | {e:+.0f}% |")
out.append("\n## Ringkasan galat prediksi (positif = over-prediksi)\n")
for t,v in errs.items(): out.append(f"- {t}: mean {st.mean(v):+.0f}%, min {min(v):+.0f}%, max {max(v):+.0f}%, MAPE {st.mean(abs(x) for x in v):.0f}% (n={len(v)})")
open(os.path.join(ROOT,"ablation","costmodel_validation.md"),"w").write("\n".join(out)+"\n")
print("\n".join(out[:3])); print("\n".join(out[-4:]))
