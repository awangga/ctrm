#!/usr/bin/env python3
"""Sensitivitas kontras kedalaman terhadap aturan checkpoint + uji permutasi eksak
(fase BM, respons reviewer R2-M2). Usage: python3 estimator_sensitivity.py [data_root]"""
import glob, itertools, json, os, re, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_table import welch
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
def curves(d):
    res = {}
    for f in sorted(glob.glob(os.path.join(ROOT, d, "progress_*.jsonl"))):
        v = []
        for ln in open(f):
            if '"phase": "eval"' not in ln: continue
            try: v.append(json.loads(ln)["all/accuracy"] * 100)
            except Exception: pass
        cfg = re.sub(r"_s\d+$", "", os.path.basename(f)[len("progress_"):-6])
        res.setdefault(cfg, []).append(v)
    return res
def perm_p(a, b):
    obs = st.mean(a) - st.mean(b); pool = a + b; n = len(a); hit = tot = 0
    for idx in itertools.combinations(range(len(pool)), n):
        x = [pool[i] for i in idx]; y = [pool[i] for i in range(len(pool)) if i not in idx]
        tot += 1; hit += abs(st.mean(x) - st.mean(y)) >= abs(obs) - 1e-12
    return hit / tot, 2 / tot
EST = [("best checkpoint", max), ("final checkpoint", lambda v: v[-1]),
       ("mean of last 5", lambda v: st.mean(v[-5:])), ("median of curve", lambda v: st.median(v))]
out = ["# Sensitivitas kontras D9-vs-D36 terhadap aturan checkpoint (auto-generated)\n",
       "Uji permutasi eksak dua-sisi atas selisih rata-rata; C(10,5)=252 partisi, sehingga p minimum = 0,0079.\n",
       "| task | aturan | selisih D9-D36 (poin) | Welch p | permutasi p |", "|---|---|---:|---:|---:|"]
for d, task in [("arc_depth_out", "ARC-AGI-1"), ("maze_depth_out", "Maze-Hard")]:
    r = curves(d)
    k9 = [k for k in r if "_d9_" in k][0]; k36 = [k for k in r if "_d36" in k][0]
    for nm, fn in EST:
        a = [fn(v) for v in r[k9]]; b = [fn(v) for v in r[k36]]
        t, df, p, _ = welch(a, b); pp, floor = perm_p(a, b)
        out.append(f"| {task} | {nm} | {st.mean(a)-st.mean(b):+.2f} | {p:.4f} | {pp:.4f} |")
        print(f"{task:10s} {nm:16s} {st.mean(a)-st.mean(b):+6.2f}  Welch p={p:.4f}  perm p={pp:.4f}")
out.append("\nAmbang Bonferroni per sumbu: 0,0167.")
open(os.path.join(ROOT, "ablation", "estimator_sensitivity.md"), "w").write("\n".join(out) + "\n")
print("tulis ablation/estimator_sensitivity.md")
