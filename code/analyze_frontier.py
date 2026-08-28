#!/usr/bin/env python3
"""Build learning curves + frontier summary from frontier_out/.
Inputs: frontier_out/frontier_summary.csv + progress_<tag>.jsonl (per-step train + per-checkpoint eval).
Outputs: learning_curves.csv (long form, plot-ready), frontier_pilot_report.md."""
import csv, json, os, sys, glob
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-1000/-home-adb-awangga-trm/63edeaef-bf18-4055-b033-2ef40a499d44/scratchpad/frontier_out"
REPORT = sys.argv[2] if len(sys.argv) > 2 else "/home/adb/awangga/trm/eksperimen/frontier/frontier_pilot_report.md"
LCCSV  = os.path.join(OUTDIR, "learning_curves.csv")

def g(d, *subs):
    for k, v in d.items():
        if all(s in k.lower() for s in subs):
            try: return float(v)
            except: return None
    return None

# learning curves (long form) from progress JSONLs
lc_rows = []
for pj in sorted(glob.glob(os.path.join(OUTDIR, "progress_*.jsonl"))):
    tag = os.path.basename(pj)[len("progress_"):-len(".jsonl")]
    for ln in open(pj):
        ln = ln.strip()
        if not ln: continue
        try: r = json.loads(ln)
        except: continue
        lc_rows.append({"tag": tag, "phase": r.get("phase"), "step": r.get("step"),
                        "lm_loss": g(r, "lm_loss") if any("lm_loss" in k for k in r) else g(r, "loss"),
                        "exact_accuracy": g(r, "exact", "acc"),
                        "accuracy": g(r, "accuracy")})
if lc_rows:
    with open(LCCSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag","phase","step","lm_loss","exact_accuracy","accuracy"])
        w.writeheader(); w.writerows(lc_rows)

# summary table
summ = []
sp = os.path.join(OUTDIR, "frontier_summary.csv")
if os.path.exists(sp):
    for r in csv.DictReader(open(sp)):
        try: m = json.loads(r.get("final_eval_json","{}") or "{}")
        except: m = {}
        summ.append({**r, "eval_acc": g(m,"exact","acc"), "eval_loss": g(m,"lm_loss") or g(m,"loss")})
    summ.sort(key=lambda x: int(x["D_eff"]))

L = ["# Frontier — energi & learning curve vs recursion depth (hidden=256, Sudoku-Extreme)", "",
     "Artefak historis: `learning_curves.csv` (train-loss/step + eval acc/loss/checkpoint, long-form,",
     "siap plot), `progress_<tag>.jsonl`, `pw_<tag>.csv` (time-series daya). Untuk grafik learning,",
     "kurva akurasi/loss, dan studi ablasi.", "",
     "| hidden | D_eff | epochs | wall (s) | net energy (Wh) | eval acc | eval loss | #records |",
     "|---|---|---|---|---|---|---|---|"]
for x in summ:
    pj = os.path.join(OUTDIR, x.get("progress_jsonl",""))
    n = sum(1 for _ in open(pj)) if pj and os.path.exists(pj) else 0
    a = f"{x['eval_acc']:.4f}" if isinstance(x['eval_acc'],float) else "n/a"
    l = f"{x['eval_loss']:.4f}" if isinstance(x['eval_loss'],float) else "n/a"
    L.append(f"| {x['hidden']} | {x['D_eff']} | {x['epochs']} | {float(x['wall_s']):.0f} | {x['net_energy_Wh']} | {a} | {l} | {n} |")
neval = len([r for r in lc_rows if r['phase']=='eval'])
ntr = len([r for r in lc_rows if r['phase']=='train'])
L += ["", f"Total records: {ntr} train-step + {neval} eval-checkpoint (lihat learning_curves.csv).",
      "Catatan: pilot under-trained bila akurasi ~0; frontier final naikkan budget hingga τ tercapai."]
open(REPORT, "w").write("\n".join(L) + "\n")
print("\n".join(L)); print("\nlearning_curves.csv rows:", len(lc_rows))
