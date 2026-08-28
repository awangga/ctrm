#!/usr/bin/env python3
"""Iso-FLOP analysis: consolidated ablation table + figures (accuracy heatmap over PxD, ~constant energy).
Usage: python plot_isoflop.py <isoflop_out> <figdir>"""
import csv, json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "eksperimen/frontier/isoflop_out"
FIG = sys.argv[2] if len(sys.argv) > 2 else "eksperimen/frontier/pilot_figures"
os.makedirs(FIG, exist_ok=True)

rows = []
for r in csv.DictReader(open(os.path.join(OUT, "isoflop_summary.csv"))):
    try: m = json.loads(r["final_eval_json"])
    except: m = {}
    rows.append(dict(hidden=int(r["hidden"]), D=int(r["D_eff"]), params=float(r["params_M"]),
                     energy=float(r["net_energy_Wh"]), exact=float(m.get("all/exact_accuracy",0)),
                     token=float(m.get("all/accuracy",0)), loss=float(m.get("all/lm_loss",0))))

# consolidated ablation csv
abl = os.path.join(OUT, "ablation_isoflop.csv")
with open(abl, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["hidden","D","params","energy","exact","token","loss"]); w.writeheader(); w.writerows(rows)
print("ablation table ->", abl)

H = sorted(set(r["hidden"] for r in rows)); Dl = sorted(set(r["D"] for r in rows))
def grid(field):
    g = np.full((len(H), len(Dl)), np.nan)
    for r in rows: g[H.index(r["hidden"]), Dl.index(r["D"])] = r[field]
    return g

# heatmap exact accuracy
g = grid("exact")*100
plt.figure(figsize=(5.5,4.2))
im = plt.imshow(g, cmap="viridis", aspect="auto", origin="lower")
plt.colorbar(im, label="exact accuracy (%)")
plt.xticks(range(len(Dl)), [f"D={d}" for d in Dl]); plt.yticks(range(len(H)), [f"h={h}" for h in H])
for i in range(len(H)):
    for j in range(len(Dl)):
        if not np.isnan(g[i,j]): plt.text(j,i,f"{g[i,j]:.1f}",ha="center",va="center",color="w",fontsize=9)
plt.xlabel("recursion depth"); plt.ylabel("width (hidden)")
plt.title("Iso-FLOP exact accuracy over P×D\n(smaller+shallower wins at equal compute)")
plt.tight_layout(); p=os.path.join(FIG,"fig_isoflop_heatmap.png"); plt.savefig(p,dpi=200); plt.close(); print("wrote",p)

# energy ~constant bar + exact overlay
rows_s = sorted(rows, key=lambda r:(r["hidden"], r["D"]))
labels = [f"h{r['hidden']}\nD{r['D']}" for r in rows_s]
E = [r["energy"] for r in rows_s]; A = [r["exact"]*100 for r in rows_s]
fig, ax1 = plt.subplots(figsize=(8,4)); x=range(len(rows_s))
ax1.bar(x, E, color="#55A868", alpha=0.85); ax1.set_ylabel("net energy (Wh)", color="#55A868")
ax1.set_xticks(list(x)); ax1.set_xticklabels(labels, fontsize=8); ax1.set_ylim(0, max(E)*1.3)
ax2 = ax1.twinx(); ax2.plot(x, A, "o-", color="#C44E52"); ax2.set_ylabel("exact accuracy (%)", color="#C44E52")
plt.title("Iso-FLOP: energy ~constant (fair), accuracy varies (smaller+shallower best)")
plt.tight_layout(); p=os.path.join(FIG,"fig_isoflop_energy.png"); plt.savefig(p,dpi=200); plt.close(); print("wrote",p)
print("DONE")
