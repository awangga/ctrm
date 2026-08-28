#!/usr/bin/env python3
"""Plot learning curves + energy-vs-depth from frontier artifacts.
Inputs: <frontier_out>/learning_curves.csv (long: tag,phase,step,lm_loss,exact_accuracy,accuracy)
        <frontier_out>/frontier_summary.csv  (per-config energy + final eval json)
Outputs PNGs into <figdir> (default eksperimen/frontier/pilot_figures/)."""
import csv, json, os, sys, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "eksperimen/frontier/frontier_out"
FIG = sys.argv[2] if len(sys.argv) > 2 else "eksperimen/frontier/pilot_figures"
os.makedirs(FIG, exist_ok=True)

# learning curves (eval phase)
curves = collections.defaultdict(lambda: collections.defaultdict(list))  # tag -> field -> [(step,val)]
for r in csv.DictReader(open(os.path.join(OUT, "learning_curves.csv"))):
    if r["phase"] != "eval": continue
    tag = r["tag"]; st = int(r["step"])
    for f in ("accuracy", "exact_accuracy", "lm_loss"):
        v = r.get(f, "")
        if v not in ("", "None"):
            curves[tag][f].append((st, float(v)))
tags = sorted(curves)

def line_plot(field, ylabel, fname, title):
    plt.figure(figsize=(6,4))
    for t in tags:
        pts = sorted(curves[t][field])
        if pts:
            xs, ys = zip(*pts); plt.plot(xs, ys, marker="o", label=t)
    plt.xlabel("training step"); plt.ylabel(ylabel); plt.title(title)
    plt.legend(title="config"); plt.grid(True, alpha=0.3); plt.tight_layout()
    p = os.path.join(FIG, fname); plt.savefig(p, dpi=200); plt.close(); print("wrote", p)

line_plot("accuracy", "eval token accuracy", "fig_learning_token_acc.png", "Learning curve (token accuracy)")
line_plot("exact_accuracy", "eval exact (puzzle) accuracy", "fig_learning_exact_acc.png", "Learning curve (exact accuracy)")
line_plot("lm_loss", "eval loss", "fig_learning_loss.png", "Learning curve (loss)")

# energy vs depth (bar) + final exact-acc overlay
rows = []
for r in csv.DictReader(open(os.path.join(OUT, "frontier_summary.csv"))):
    try: m = json.loads(r.get("final_eval_json","{}") or "{}")
    except: m = {}
    rows.append((int(r["D_eff"]), float(r["net_energy_Wh"]), float(m.get("all/exact_accuracy", 0))))
rows.sort()
if rows:
    D, E, A = zip(*rows)
    fig, ax1 = plt.subplots(figsize=(6,4))
    x = range(len(D))
    ax1.bar(x, E, color="#4C72B0", alpha=0.8); ax1.set_xticks(list(x)); ax1.set_xticklabels([f"D={d}" for d in D])
    ax1.set_ylabel("net training energy (Wh)", color="#4C72B0"); ax1.set_xlabel("recursion depth")
    ax2 = ax1.twinx(); ax2.plot(x, [a*100 for a in A], "o-", color="#C44E52")
    ax2.set_ylabel("final exact accuracy (%)", color="#C44E52")
    plt.title("Energy rises with depth; exact accuracy does not (iso-step)")
    plt.tight_layout(); p = os.path.join(FIG, "fig_energy_vs_depth.png"); plt.savefig(p, dpi=200); plt.close(); print("wrote", p)
print("DONE")
