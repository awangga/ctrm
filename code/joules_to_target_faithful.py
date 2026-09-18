#!/usr/bin/env python3
"""Emit Wh-to-target and step-to-target for every faithful-recipe run.

Aturan #1 CLAUDE.md: tiap angka di naskah harus telusur ke artefak yang di-commit.
Nilai Joules-to-target (268/329/161/155/353 Wh) sebelumnya hanya dihitung on-the-fly
oleh make_manuscript_figures.py dan tidak tersimpan di mana pun, sehingga kalimat naskah
"the released files record the training step number at which the target was first reached,
beside the energy" tidak dapat diperiksa. Skrip ini menuliskannya.

Energi net = integral trapesium (P - IDLE_W) atas deret nvidia-smi 1 Hz, TIDAK di-clip
pada nol, identik dengan run_recipe.py dan make_manuscript_figures.py.

Pakai:  python3 joules_to_target_faithful.py
Keluar: ablation/joules_to_target_faithful.csv
"""
import csv, datetime, glob, json, os, re

IDLE_W = 4.7
HERE = os.path.dirname(os.path.abspath(__file__))


def _mean_best_token(folder, deff):
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, folder, "recipe_summary.csv")))
            if r["D_eff"] == deff]
    return round(sum(float(r["best_token_pct"]) for r in rows) / len(rows), 2)


ARC_G400_D36_BEST = _mean_best_token("arc_d36_g400_out", "36")   # 62.52 pada fase BS

# (folder, metrik yang dilaporkan, daftar target beserta asalnya)
TASKS = [
    ("recipe_out",     "Sudoku-Extreme", "all/exact_accuracy", [
        (50.0,  "50% exact, the headline target"),
        (49.67, "best accuracy of the non-recursive baseline"),
        (50.07, "best accuracy of D_eff=18"),
        (36.26, "best accuracy of D_eff=36"),
    ]),
    ("maze_depth_out", "Maze-Hard",      "all/accuracy", [
        (86.80, "best token accuracy of D_eff=18"),
    ]),
    # REKAMAN SAJA: subset lama arc1-aug1k-e512 ternyata augmentasi SATU task ARC (fase BS),
    # jadi baris ini tidak dipakai untuk klaim. Targetnya berasal dari subset lama itu.
    ("arc_depth_out",  "ARC-AGI-1 subset lama e512 (1 task, TIDAK SAH)", "all/accuracy", [
        (30.59, "best token accuracy of D_eff=36, OLD e512 subset"),
        (33.30, "best token accuracy of D_eff=18, OLD e512 subset"),
    ]),
    # Subset sah arc1-aug1k-g400 (400 task): hanya D9 dan D36. Target = rerata akurasi token
    # terbaik D36 pada subset ini, dihitung dari summary CSV (bukan dipatok).
    ("arc_d9_g400_out",  "ARC-AGI-1 (g400)", "all/accuracy", [
        (ARC_G400_D36_BEST, "best token accuracy of D_eff=36, g400 subset"),
    ]),
    ("arc_d36_g400_out", "ARC-AGI-1 (g400)", "all/accuracy", [
        (ARC_G400_D36_BEST, "best token accuracy of D_eff=36, g400 subset"),
    ]),
    ("maze_real_out",  "Maze-Hard",      "all/accuracy", []),
]


def power_series(path):
    ts, pw = [], []
    for row in csv.reader(open(path)):
        if len(row) < 2:
            continue
        try:
            ts.append(datetime.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp())
            pw.append(float(row[1]))
        except ValueError:
            pass
    return ts, pw


def cumulative_wh(ts, pw):
    """Wh kumulatif, sejajar dengan ts. Tanpa clip, lihat Eq. (2) naskah."""
    out = [0.0]
    for i in range(1, len(ts)):
        seg = 0.5 * ((pw[i] - IDLE_W) + (pw[i - 1] - IDLE_W)) * (ts[i] - ts[i - 1])
        out.append(out[-1] + seg / 3600.0)
    return out


def interp(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    f = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + f * (ys[hi] - ys[lo])


rows = []
for folder, task, key, targets in TASKS:
    for prog in sorted(glob.glob(os.path.join(HERE, folder, "progress_*.jsonl"))):
        tag = re.sub(r"^progress_|\.jsonl$", "", os.path.basename(prog))
        pwf = os.path.join(HERE, folder, f"pw_{tag}.csv")
        if not os.path.exists(pwf):
            continue
        ts, pw = power_series(pwf)
        if len(ts) < 2:
            continue
        wh = cumulative_wh(ts, pw)
        ev = []
        for line in open(prog):
            d = json.loads(line)
            if d.get("phase") == "eval" and key in d:
                ev.append((d["step"], d["t"], d[key] * 100.0))
        if not ev:
            continue
        best = float("-inf")
        run_best = []
        for step, t, acc in ev:
            best = max(best, acc)
            run_best.append((step, t, best))
        # tag rekursif: h512_d18_recipe_b192_s0 ; baseline non-rekursif: h512_transformers_baseline_b192_s0
        m = re.search(r"h(\d+)_d(\d+)_.*_s(\d+)$", tag)
        if m:
            hidden, deff, seed = m.group(1), m.group(2), m.group(3)
        else:
            b = re.search(r"h(\d+)_transformers_baseline_.*_s(\d+)$", tag)
            hidden, deff, seed = (b.group(1), "none", b.group(2)) if b else ("", "", "")
        for value, origin in targets:
            hit = next((r for r in run_best if r[2] >= value), None)
            rows.append(dict(
                task=task, tag=tag, hidden=hidden, D_eff=deff, seed=seed,
                metric=key.split("/")[-1], target_pct=f"{value:g}", target_origin=origin,
                reached="yes" if hit else "no",
                step_at_target=hit[0] if hit else "",
                net_Wh_at_target=f"{interp(hit[1], ts, wh):.2f}" if hit else "",
                final_net_Wh=f"{wh[-1]:.2f}",
            ))

out = os.path.join(HERE, "ablation", "joules_to_target_faithful.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} baris -> {out}")
reached = sum(1 for r in rows if r["reached"] == "yes")
print(f"target tercapai pada {reached} dari {len(rows)} pasangan (run x target)")
