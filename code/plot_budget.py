#!/usr/bin/env python3
"""Budget-resolved curves: exact/token accuracy vs training step per recursion depth (fixed width).
Shows the budget-dependent optimal depth (crossover). Usage: python plot_budget.py <budget_out> <figdir>"""
import json, os, sys, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "eksperimen/frontier/budget_out"
FIG = sys.argv[2] if len(sys.argv) > 2 else "eksperimen/frontier/pilot_figures"
os.makedirs(FIG, exist_ok=True)
DEPTHS = [9, 18, 36]

def curve(d):
    xs=[]; ex=[]; tok=[]
    for l in open(os.path.join(OUT, f"progress_h128_d{d}.jsonl")):
        r=json.loads(l)
        if r.get("phase")=="eval":
            xs.append(r["step"]); ex.append(r.get("all/exact_accuracy",0)*100); tok.append(r.get("all/accuracy",0)*100)
    return xs, ex, tok

# consolidated long-form csv
with open(os.path.join(OUT,"budget_curves.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["D_eff","step","exact_pct","token_pct"])
    for d in DEPTHS:
        xs,ex,tok=curve(d)
        for s,e,t in zip(xs,ex,tok): w.writerow([d,s,round(e,3),round(t,3)])

for field,idx,ylab,fname,title in [("exact",1,"exact (puzzle) accuracy (%)","fig_budget_exact_acc.png","Accuracy vs compute per recursion depth"),
                                   ("token",2,"token accuracy (%)","fig_budget_token_acc.png","Token accuracy vs compute per depth")]:
    plt.figure(figsize=(6.2,4.2))
    for d in DEPTHS:
        c=curve(d); plt.plot(c[0], c[idx], marker="o", ms=3, label=f"D={d}")
    plt.xlabel("training step (compute)"); plt.ylabel(ylab); plt.title(title)
    plt.legend(title="recursion depth"); plt.grid(True,alpha=0.3); plt.tight_layout()
    p=os.path.join(FIG,fname); plt.savefig(p,dpi=200); plt.close(); print("wrote",p)
print("DONE")
