#!/usr/bin/env python3
"""Bangun figur manuscript dari artefak run yang di-commit.

Gaya mengikuti figur yang sudah dipakai naskah: vektor PDF, DejaVu Serif, lebar 6.6 in,
palet colorblind-safe Okabe-Ito, tanpa judul-plot, spine bersih.

Hanya memakai rezim faithful-recipe (recipe_out/ dan microbench); data pilot TIDAK dipakai
untuk figur hasil.

Pakai:  python3 make_manuscript_figures.py [--out ../../manuscript/figures]
"""
import argparse
import csv
import glob
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IDLE_W = 4.7           # daya idle terukur, dikurangkan spt Eq. (2) di naskah
GRID = 0.67593         # kg CO2e/kWh, dari CodeCarbon (Indonesia), konstan di semua run

# Okabe-Ito
OI = {"blue": "#0072B2", "orange": "#E69F00", "vermillion": "#D55E00",
      "green": "#009E73", "sky": "#56B4E9", "grey": "#555555",
      "reddish": "#CC79A7"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "lines.linewidth": 1.3,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def read_power(path):
    """Kembalikan (waktu unix, daya W) dari log nvidia-smi 1 Hz."""
    ts, w = [], []
    for line in open(path):
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            t = datetime.strptime(parts[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
            p = float(parts[1])
        except ValueError:
            continue
        ts.append(t)
        w.append(p)
    return np.array(ts), np.array(w)


def cumulative_wh(ts, w):
    """Energi net kumulatif (Wh) via trapesium atas (P - P_idle), Eq. (2)."""
    net = np.clip(w - IDLE_W, 0, None)
    if len(ts) < 2:
        return np.zeros_like(net)
    dt = np.diff(ts)
    seg = 0.5 * (net[1:] + net[:-1]) * dt          # Joule
    return np.concatenate([[0.0], np.cumsum(seg)]) / 3600.0


def eval_points(path):
    """(step, exact%) dari tiap checkpoint eval, plus waktu unix."""
    out = []
    for line in open(path):
        if '"phase": "eval"' not in line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append((float(d["t"]), int(d["step"]), 100.0 * float(d.get("all/exact_accuracy", 0.0))))
    return out


def trajectory(tag, out_dir):
    """Lintasan (Wh kumulatif, exact%) untuk satu run."""
    pw = os.path.join(out_dir, f"pw_{tag}.csv")
    pg = os.path.join(out_dir, f"progress_{tag}.jsonl")
    if not (os.path.exists(pw) and os.path.exists(pg)):
        return None
    ts, w = read_power(pw)
    if len(ts) < 2:
        return None
    cum = cumulative_wh(ts, w)
    xs, ys = [], []
    for t, _step, acc in eval_points(pg):
        i = int(np.searchsorted(ts, t))
        if i >= len(cum):
            i = len(cum) - 1
        xs.append(cum[i])
        ys.append(acc)
    return np.array(xs), np.array(ys)


def fig_energy_accuracy(out_path, data_root):
    """Lintasan energi-akurasi Sudoku h512, tiga kedalaman x tiga seed."""
    rec = os.path.join(data_root, "recipe_out")
    specs = [(9, "b192", OI["blue"], "o"), (18, "b192", OI["orange"], "s"),
             (36, "b96", OI["vermillion"], "^")]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    TARGET = 50.0
    for depth, batch, colour, marker in specs:
        curves = []
        for seed in (0, 1, 2):
            tr = trajectory(f"h512_d{depth}_recipe_{batch}_s{seed}", rec)
            if tr is None:
                continue
            curves.append(tr)
            ax.plot(tr[0], tr[1], color=colour, alpha=0.28, lw=0.8, zorder=1)
        if not curves:
            continue
        n = min(len(c[0]) for c in curves)
        xm = np.mean([c[0][:n] for c in curves], axis=0)
        ym = np.mean([c[1][:n] for c in curves], axis=0)
        ax.plot(xm, ym, color=colour, marker=marker, markersize=2.6, markevery=3,
                label=rf"$D_{{\mathrm{{eff}}}}={depth}$", zorder=3)
        # Titik capai-target dihitung PER SEED lalu dirata-rata, konsisten dengan tabel
        # di naskah; kurva rata-rata tidak dipakai untuk ini karena seed yang tak pernah
        # mencapai target akan menyeret rata-ratanya.
        reach = []
        for cx, cy in curves:
            h = np.where(cy >= TARGET)[0]
            if len(h):
                reach.append(cx[h[0]])
        if reach:
            mx = float(np.mean(reach))
            ax.plot(mx, TARGET, marker="*", markersize=11, color=colour,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=4)
            # Label ditaruh di bawah sumbu target dan digeser per kedalaman supaya tidak
            # ditembus kurva tetangga; sebelumnya kurva D18 melintasi teks label D9.
            dy = {9: -30, 18: -18, 36: -18}.get(depth, -18)
            dx = {9: -46, 18: 6, 36: 6}.get(depth, 6)
            ax.annotate(f"{mx:.0f} Wh ({len(reach)}/{len(curves)})", (mx, TARGET),
                        textcoords="offset points", xytext=(dx, dy), fontsize=7.5, color=colour,
                        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.85))
    ax.axhline(TARGET, color=OI["grey"], ls=":", lw=0.9, zorder=0)
    ax.annotate("target 50% exact", (ax.get_xlim()[1], TARGET), textcoords="offset points",
                xytext=(-4, 4), ha="right", fontsize=7.5, color=OI["grey"])
    ax.set_xlabel("Cumulative net GPU energy (Wh)")
    ax.set_ylabel("Exact accuracy (\\%)" if False else "Exact accuracy (%)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left", handlelength=1.6)
    sec = ax.secondary_xaxis("top", functions=(lambda wh: wh / 1000 * GRID * 1000,
                                               lambda g: g / (GRID * 1000) * 1000))
    sec.set_xlabel("Cumulative CO$_2$e (g)", fontsize=8)
    sec.tick_params(labelsize=7.5)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  tulis {out_path}")


def fig_cost_model(out_path, data_root):
    """Cost-model energi per langkah dari microbench hardware, log-log + CI bootstrap."""
    csv_path = os.path.join(data_root, "microbench_results.csv")
    P, J = [], []
    for r in csv.DictReader(open(csv_path)):
        try:
            P.append(float(r["params_M"]) * float(r["D_eff"]))
            J.append(float(r["J_per_step"]))
        except (KeyError, ValueError):
            continue
    P, J = np.array(P), np.array(J)
    lx, ly = np.log(P), np.log(J)
    b, la = np.polyfit(lx, ly, 1)
    pred = la + b * lx
    r2 = 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
    # Bootstrap direplikasi PERSIS spt fit_extrapolation.py (RNG dan konvensi persentil
    # yang sama) supaya CI di figur identik dengan yang dilaporkan di naskah.
    import random as _random
    _random.seed(20260627)
    bs = []
    n = len(P)
    for _ in range(5000):
        idx = [_random.randrange(n) for _ in range(n)]
        bxx = lx[idx]
        if len(set(bxx.tolist())) < 2:
            continue
        bs.append(np.polyfit(bxx, ly[idx], 1)[0])
    bs.sort()
    lo, hi = bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs))]

    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    xs = np.linspace(lx.min(), lx.max(), 100)
    ax.fill_between(np.exp(xs), np.exp(la + lo * xs), np.exp(la + hi * xs),
                    color=OI["blue"], alpha=0.13, lw=0)
    ax.plot(np.exp(xs), np.exp(la + b * xs), color=OI["blue"], lw=1.2)
    ax.scatter(P, J, s=18, color=OI["vermillion"], zorder=3, edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"params (M) $\times\ D_{\mathrm{eff}}$")
    ax.set_ylabel("Energy per step (J)")
    ax.text(0.04, 0.94, f"$b={b:.2f}$ [{lo:.2f}, {hi:.2f}]\n$R^2={r2:.2f}$, $n={len(P)}$",
            transform=ax.transAxes, va="top", fontsize=7.5)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  tulis {out_path}  (b={b:.3f} CI[{lo:.2f},{hi:.2f}] R2={r2:.3f})")


def joules_to_target_table(data_root, target=50.0):
    """Tabel Joules-to-target per konfigurasi, dihitung per seed lalu dirata-rata."""
    rec = os.path.join(data_root, "recipe_out")
    cfgs = [("TRM D_eff=9", "h512_d9_recipe_b192"), ("TRM D_eff=18", "h512_d18_recipe_b192"),
            ("TRM D_eff=36", "h512_d36_recipe_b96"),
            ("non-recursive baseline", "h512_transformers_baseline_b192")]
    rows = []
    for name, base in cfgs:
        per = []
        for seed in (0, 1, 2):
            tr = trajectory(f"{base}_s{seed}", rec)
            if tr is None:
                continue
            x, y = tr
            h = np.where(y >= target)[0]
            per.append(float(x[h[0]]) if len(h) else None)
        ok = [v for v in per if v is not None]
        rows.append((name, per, len(ok), len(per),
                     float(np.mean(ok)) if ok else None,
                     float(np.std(ok, ddof=1)) if len(ok) > 1 else None))
    print(f"\n  Joules-to-target ({target:.0f}% exact), per seed lalu dirata-rata:")
    for name, per, nok, ntot, m, sd in rows:
        cells = ", ".join(f"{v:.0f}" if v is not None else "--" for v in per)
        tail = (f"{m:.0f}" + (f" +/- {sd:.0f}" if sd else "") +
                f" Wh, {m/1000*GRID*1000:.0f} g CO2e") if m else "never reached"
        print(f"    {name:24s} [{cells}]  {nok}/{ntot}  -> {tail}")
    return rows


def perseed_table(data_root):
    """Cetak nilai akurasi per seed untuk tiap konfigurasi rezim faithful.

    Dipakai untuk Tabel per-seed di naskah; dicetak di sini supaya angkanya
    diregenerasi dari summary CSV, bukan disalin tangan.
    """
    import collections
    import re as _re
    import statistics as _st
    src = [("Sudoku-Extreme (exact)", "recipe_out/recipe_summary.csv", "best_exact_pct"),
           ("Maze-Hard (token)", "maze_depth_out/recipe_summary.csv", "final_token_pct"),
           ("ARC-AGI-1 (token)", "arc_depth_out/recipe_summary.csv", "final_token_pct")]
    print("\n  Akurasi per seed (rezim faithful):")
    for task, rel, col in src:
        path = os.path.join(data_root, rel)
        if not os.path.exists(path):
            continue
        groups = collections.defaultdict(dict)
        for r in csv.DictReader(open(path)):
            m = _re.search(r"_s(\d+)$", r["tag"])
            if not m:
                continue
            groups[r["tag"][:m.start()]][int(m.group(1))] = float(r[col])
        print(f"    {task}")
        for base in sorted(groups):
            v = groups[base]
            got = [v[s] for s in sorted(v)]
            cells = " ".join(f"{v[s]:6.2f}" if s in v else "    --" for s in (0, 1, 2))
            sd = _st.stdev(got) if len(got) > 1 else 0.0
            print(f"      {base:40s} {cells}  -> {_st.mean(got):5.1f} +/- {sd:.1f}")


def pilot_vs_faithful(data_root):
    """Bandingkan peringkat kedalaman rezim pilot vs faithful pada Sudoku.

    Menjawab: bisakah run murah dipakai menyaring pilihan kedalaman lebih dulu?
    Angka dicetak dari CSV yang di-commit supaya klaim di naskah telusur.
    """
    import statistics as _st
    rep = os.path.join(data_root, "budget_out", "replication_summary.csv")
    xv = os.path.join(data_root, "budget_out", "energy_xval.csv")
    rec = os.path.join(data_root, "recipe_out", "recipe_summary.csv")
    if not (os.path.exists(rep) and os.path.exists(xv) and os.path.exists(rec)):
        return
    pilot = {}
    for r in csv.DictReader(open(rep)):
        pilot[int(r["D_eff"])] = [float(r[f"seed{i}_exact"]) for i in (0, 1, 2)]
    for r in csv.DictReader(open(xv)):          # seed 3-4 disimpan sebagai fraksi
        pilot.setdefault(int(r["D_eff"]), []).append(float(r["exact_acc"]) * 100)
    faith = {}
    for r in csv.DictReader(open(rec)):
        m = None
        for d, tag in ((9, "h512_d9_recipe_b192"), (18, "h512_d18_recipe_b192"),
                       (36, "h512_d36_recipe_b96")):
            if r["tag"].startswith(tag):
                m = d
        if m:
            faith.setdefault(m, []).append(float(r["best_exact_pct"]))
    print("\n  Pilot vs faithful, sumbu kedalaman Sudoku (exact %):")
    for label, data in (("pilot h128 (5 seed)", pilot), ("faithful h512 (3 seed)", faith)):
        means = {d: _st.mean(v) for d, v in sorted(data.items())}
        sds = {d: _st.stdev(v) for d, v in data.items() if len(v) > 1}
        rank = " > ".join(f"D{d}" for d in sorted(means, key=lambda k: -means[k]))
        spread = max(means.values()) - min(means.values())
        print(f"    {label:24s} " +
              "  ".join(f"D{d}={means[d]:5.2f}+/-{sds.get(d, 0):.2f}" for d in sorted(means)))
        print(f"    {'':24s} peringkat {rank};  sebaran {spread:.1f} poin, "
              f"sd {min(sds.values()):.1f}-{max(sds.values()):.1f}")


def agreement_floor(data_root):
    """Lantai kesepakatan CodeCarbon vs nvidia-smi di rezim pilot, memisahkan run batal."""
    import glob as _glob
    ok, short = [], []
    for f in _glob.glob(os.path.join(data_root, "*", "*summary*.csv")) + \
            [os.path.join(data_root, "budget_out", "energy_xval.csv")]:
        if not os.path.exists(f):
            continue
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        ak = [k for k in rows[0] if "agree" in k.lower()]
        wk = [k for k in rows[0] if k in ("wall_s", "smi_dur_s")]
        if not ak:
            continue
        for r in rows:
            v = r.get(ak[0])
            if not v:
                continue
            dur = float(r.get(wk[0], 1e9)) if wk else 1e9
            (short if dur < 120 else ok).append((float(v), r.get("tag", ""), dur))
    if ok:
        print(f"\n  Kesepakatan dua instrumen, run pilot yang selesai: "
              f"min {min(a for a, _, _ in ok):.2f}%  (n={len(ok)})")
    if short:
        print(f"    run batal <120 s (dikecualikan): " +
              ", ".join(f"{t} {a:.2f}% @{d:.0f}s" for a, t, d in sorted(short)))


def iso_accuracy_energy(data_root):
    """Energi yang dibutuhkan D9 untuk MENYAMAI akurasi akhir tiap pesaing.

    Perbandingan iso-akurasi: metrik Joules-to-target yang sama, tetapi target
    diambil dari akurasi akhir yang benar-benar dicapai pesaing, bukan satu ambang
    tetap. Ini yang menjawab "berapa energi untuk mutu yang sama".
    """
    import statistics as _st
    rec = os.path.join(data_root, "recipe_out")
    rivals = [("D_eff=36", 36.26, 340.4), ("D_eff=18", 50.07, 358.0),
              ("non-recursive baseline", 49.67, 384.0)]
    print("\n  Energi iso-akurasi: biaya D9 untuk menyamai akurasi akhir pesaing")
    for name, target, rival_wh in rivals:
        per = []
        for seed in (0, 1, 2):
            tr = trajectory(f"h512_d9_recipe_b192_s{seed}", rec)
            if tr is None:
                continue
            x, y = tr
            h = np.where(y >= target)[0]
            per.append(float(x[h[0]]) if len(h) else None)
        ok = [v for v in per if v is not None]
        if not ok:
            continue
        m = _st.mean(ok)
        sd = _st.stdev(ok) if len(ok) > 1 else 0.0
        cells = ", ".join(f"{v:.0f}" if v is not None else "--" for v in per)
        print(f"    vs {name:24s} target {target:5.2f}%  D9=[{cells}] -> {m:5.1f}+/-{sd:.0f} Wh "
              f"({len(ok)}/{len(per)});  pesaing {rival_wh:.0f} Wh, hemat {100*(rival_wh-m)/rival_wh:.0f}%"
              f";  CO2e {rival_wh/1000*GRID*1000:.0f} -> {m/1000*GRID*1000:.0f} g")


def fig_crosstask(out_path, data_root):
    """Efek kedalaman di tiga task, memakai SELURUH seed yang tersedia per task."""
    import collections
    import statistics as _st
    src = [("Sudoku-Extreme", "recipe_out/recipe_summary.csv", "best_exact_pct",
            "Exact accuracy (%)", ("h512_d9_recipe", "h512_d18_recipe", "h512_d36_recipe")),
           ("ARC-AGI-1", "arc_depth_out/recipe_summary.csv", "final_token_pct",
            "Token accuracy (%)", ("h256_d9_recipe", "h256_d18_recipe", "h256_d36_recipe")),
           ("Maze-Hard", "maze_depth_out/recipe_summary.csv", "final_token_pct",
            "Token accuracy (%)", ("h256_d9_recipe", "h256_d18_recipe", "h256_d36_recipe"))]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5))
    for ax, (task, rel, col, ylab, prefixes) in zip(axes, src):
        path = os.path.join(data_root, rel)
        if not os.path.exists(path):
            continue
        vals = collections.defaultdict(list)
        for r in csv.DictReader(open(path)):
            for d, pre in zip((9, 18, 36), prefixes):
                if r["tag"].startswith(pre):
                    vals[d].append(float(r[col]))
        xs = sorted(vals)
        m = [_st.mean(vals[d]) for d in xs]
        e = [_st.stdev(vals[d]) if len(vals[d]) > 1 else 0.0 for d in xs]
        n = min(len(vals[d]) for d in xs)
        for i, d in enumerate(xs):                       # titik tiap seed
            ax.scatter([i] * len(vals[d]), vals[d], s=9, color=OI["grey"], alpha=0.45, zorder=2)
        ax.errorbar(range(len(xs)), m, yerr=e, color=OI["blue"], marker="o", markersize=4,
                    capsize=3, lw=1.3, zorder=3)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"{d}" for d in xs])
        ax.set_xlabel(r"$D_{\mathrm{eff}}$")
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_title(f"{task}  ($n{{=}}{n}$)", fontsize=8.5)
        ax.margins(x=0.22)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  tulis {out_path}")


def fig_learning(out_path, data_root):
    """Kurva belajar Sudoku h512: akurasi vs langkah, semua seed, tiga kedalaman.

    Menjawab pertanyaan reviewer yang belum dijawab figur mana pun: apakah run
    benar-benar dilatih cukup lama? Sekaligus memperlihatkan rancangan iso-compute
    bekerja (D9 menjalankan dua kali lipat langkah D18/D36).
    """
    rec = os.path.join(data_root, "recipe_out")
    specs = [(9, "b192", OI["blue"], "o"), (18, "b192", OI["orange"], "s"),
             (36, "b96", OI["vermillion"], "^")]
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    for depth, batch, colour, marker in specs:
        curves = []
        for seed in (0, 1, 2):
            pg = os.path.join(rec, f"progress_h512_d{depth}_recipe_{batch}_s{seed}.jsonl")
            if not os.path.exists(pg):
                continue
            xs, ys = [], []
            for _t, step, acc in eval_points(pg):
                xs.append(step)
                ys.append(acc)
            if xs:
                curves.append((np.array(xs), np.array(ys)))
                ax.plot(xs, ys, color=colour, alpha=0.25, lw=0.8, zorder=1)
        if not curves:
            continue
        n = min(len(c[0]) for c in curves)
        xm = curves[0][0][:n]
        ym = np.mean([c[1][:n] for c in curves], axis=0)
        ax.plot(xm, ym, color=colour, marker=marker, markersize=2.6, markevery=3,
                label=rf"$D_{{\mathrm{{eff}}}}={depth}$ ({xm[-1]/1000:.0f}k steps)", zorder=3)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Exact accuracy (%)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left", handlelength=1.6)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  tulis {out_path}")


def fig_iso_accuracy(out_path, data_root):
    """Energi untuk menyamai akurasi akhir tiap pesaing: batang berpasangan."""
    import statistics as _st
    rec = os.path.join(data_root, "recipe_out")
    rivals = [(r"$D_{\mathrm{eff}}{=}36$", 36.26, 340.4), (r"$D_{\mathrm{eff}}{=}18$", 50.07, 358.0),
              ("non-recursive", 49.67, 384.0)]
    labels, rival_wh, d9_wh, d9_sd = [], [], [], []
    for name, target, rwh in rivals:
        per = []
        for seed in (0, 1, 2):
            tr = trajectory(f"h512_d9_recipe_b192_s{seed}", rec)
            if tr is None:
                continue
            x, y = tr
            h = np.where(y >= target)[0]
            if len(h):
                per.append(float(x[h[0]]))
        if not per:
            continue
        labels.append(name)
        rival_wh.append(rwh)
        d9_wh.append(_st.mean(per))
        d9_sd.append(_st.stdev(per) if len(per) > 1 else 0.0)
    idx = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.bar(idx - 0.19, rival_wh, 0.36, color=OI["grey"], label="rival, full budget")
    ax.bar(idx + 0.19, d9_wh, 0.36, yerr=d9_sd, capsize=3, color=OI["blue"],
           label=r"$D_{\mathrm{eff}}{=}9$ to match")
    for i, (r, d) in enumerate(zip(rival_wh, d9_wh)):
        ax.text(i + 0.19, d + 14, f"\u2212{100*(r-d)/r:.0f}%", ha="center", fontsize=8.5,
                color=OI["blue"], fontweight="bold")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Net energy (Wh)")
    ax.set_ylim(0, max(rival_wh) * 1.30)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  tulis {out_path}")


def eval_share_table(data_root):
    """Pangsa wall-time yang dipakai evaluasi berkala, per kedalaman.

    Menopang klaim protokol: biaya evaluasi TIDAK dominan setelah di-subset ke 512
    puzzle, tetapi tumbuh sebanding dengan D_eff -- itulah alasan jadwal evaluasi
    harus dikunci lintas konfigurasi, sebab jadwal yang bervariasi akan memberi
    keuntungan sistematis kepada model dangkal.
    """
    import bisect
    import statistics as _st
    rec = os.path.join(data_root, "recipe_out")
    print("\n=== EVAL-SHARE (pangsa wall-time untuk evaluasi) ===")
    rows = []
    for depth, batch in ((9, "b192"), (18, "b192"), (36, "b96")):
        vals = []
        for seed in (0, 1, 2):
            path = os.path.join(rec, f"progress_h512_d{depth}_recipe_{batch}_s{seed}.jsonl")
            if not os.path.exists(path):
                continue
            train_t, eval_t = [], []
            for line in open(path):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "t" not in d:
                    continue
                (eval_t if d.get("phase") == "eval" else train_t).append(d["t"])
            if len(train_t) < 10:
                continue
            train_t.sort()
            eval_t.sort()
            total = train_t[-1] - train_t[0]
            step = _st.median([b - a for a, b in zip(train_t, train_t[1:])])
            spent = 0.0
            for e in eval_t:
                i = bisect.bisect_left(train_t, e)
                if 0 < i < len(train_t):
                    spent += max(0.0, (train_t[i] - train_t[i - 1]) - step)
            vals.append(100.0 * spent / total)
        if vals:
            m = _st.mean(vals)
            sd = _st.stdev(vals) if len(vals) > 1 else 0.0
            rows.append((depth, m, sd, len(vals)))
            print(f"  D_eff={depth:>2}  eval {m:5.2f}% +- {sd:.2f}  (n={len(vals)} seed)")
    if len(rows) >= 2:
        print(f"  rasio D36/D9 = {rows[-1][1] / rows[0][1]:.2f}x untuk kedalaman {rows[-1][0] // rows[0][0]}x")
    return rows


def _tcrit(df, alpha=0.05):
    """Kuantil t dua-sisi via bisection pada betai (tanpa SciPy)."""
    from stats_table import betai
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betai(df / 2.0, 0.5, df / (df + mid * mid)) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fig_regime_map(out_path, data_root):
    """Peta rezim: posisi metrik (sebab) di kiri, efek kedalaman bertanda (akibat) di kanan.

    Sumbu pengorganisasi SENGAJA bukan 'spread', karena spread secara aljabar sama
    dengan |efek| (urutan kedalaman monoton di ketiga task) sehingga plot efek-vs-spread
    akan tautologis. Panel kiri membawa informasi yang tidak ada di panel kanan:
    di mana konfigurasi duduk pada skala metriknya, yakni berapa sisa ruang ke langit-langit.
    """
    from stats_table import welch
    rec = os.path.join(data_root, "recipe_out")
    specs = [
        ("Sudoku-Extreme", "exact", os.path.join(rec, "recipe_summary.csv"), "best_exact_pct",
         lambda r: r["hidden"] == "512" and "recipe" in r["tag"], "discriminating"),
        ("ARC-AGI-1", "token", os.path.join(data_root, "arc_depth_out", "recipe_summary.csv"),
         "final_token_pct", lambda r: True, "unresolved"),
        ("Maze-Hard", "token", os.path.join(data_root, "maze_depth_out", "recipe_summary.csv"),
         "final_token_pct", lambda r: True, "saturated"),
    ]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True,
                                   gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.08})
    shades = {"discriminating": OI["blue"], "unresolved": OI["grey"], "saturated": OI["vermillion"]}
    marks = {"9": "o", "18": "s", "36": "^"}
    ylabels = []
    for row, (task, metric, path, col, filt, regime) in enumerate(specs):
        y = len(specs) - 1 - row
        groups = {}
        for r in csv.DictReader(open(path)):
            if filt(r):
                groups.setdefault(r["D_eff"], []).append(float(r[col]))
        means = {k: float(np.mean(v)) for k, v in groups.items()}
        # panel kiri: posisi tiap kedalaman pada skala metrik 0-100
        axL.plot([0, 100], [y, y], color="0.88", lw=5, solid_capstyle="butt", zorder=1)
        for k in ("9", "18", "36"):
            axL.errorbar(means[k], y, xerr=np.std(groups[k], ddof=1), fmt=marks[k], ms=4.4,
                         color=shades[regime], ecolor=shades[regime], elinewidth=1, capsize=2,
                         zorder=3, mfc="white" if regime == "unresolved" else shades[regime])
        best = max(means.values())
        axL.annotate("", xy=(100, y + 0.27), xytext=(best, y + 0.27),
                     arrowprops=dict(arrowstyle="<->", color="0.55", lw=0.7, shrinkA=0, shrinkB=0))
        axL.text((best + 100) / 2, y + 0.40, f"{100 - best:.0f} pts left", ha="center",
                 fontsize=8.2, color="0.35")
        # panel kanan: efek bertanda + CI 95%
        d9, d36 = groups["9"], groups["36"]
        t, df, pval, _ = welch(d36, d9)
        diff = means["36"] - means["9"]
        se = abs(diff / t)
        ci = _tcrit(df) * se
        # Ambang Bonferroni per sumbu (3 kontras), sama dgn Tabel 8. ARC lolos 0.05
        # tetapi TIDAK lolos 0.0167, dan penanda terisi berarti 'mapan' -- jadi 0.05
        # akan menyesatkan.
        sig = pval < 0.05 / 3
        axR.errorbar(diff, y, xerr=ci, fmt="D", ms=5, color=shades[regime], ecolor=shades[regime],
                     elinewidth=1.3, capsize=3, mfc=shades[regime] if sig else "white", zorder=3)
        axR.text(diff, y - 0.34, (f"{diff:+.1f}" + ("" if sig else " n.s.")), ha="center",
                 fontsize=8.6, color=shades[regime])
        ylabels.append((y, f"{task}\n({metric}), {regime}"))
    axL.set_xlim(0, 108)
    axL.set_ylim(-0.75, len(specs) - 0.35)
    axL.set_yticks([y for y, _ in ylabels])
    axL.set_yticklabels([lab for _, lab in ylabels], fontsize=8.6)
    axL.set_xlabel("Where the metric sits (%)")
    axR.axvline(0, color="0.45", lw=0.9, zorder=2)
    axR.set_xlabel(r"Depth effect, $D_{36}-D_9$ (points)")
    axR.set_xlim(-33, 12)
    for ax in (axL, axR):
        ax.grid(axis="x", alpha=0.25, lw=0.5)
    hs = [plt.Line2D([], [], marker=m, ls="", color="0.35", ms=5.2,
                     label=rf"$D_{{\mathrm{{eff}}}}{{=}}{k}$") for k, m in marks.items()]
    axL.legend(handles=hs, frameon=False, fontsize=8.0, loc="lower left",
               ncol=3, handletextpad=0.2, columnspacing=0.9, borderpad=0.1)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  tulis {out_path}")


def fig_protocol(out_path):
    """Diagram alur protokol pengukuran. Skema, bukan figur data.

    Sengaja memuat gerbang penolakan yang benar-benar dipakai (kesepakatan dua
    instrumen >=98.8%, run <120 s dibuang) supaya diagram ini menyampaikan
    keputusan, bukan sekadar hiasan kotak-panah.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    # Beri margin: tanpa ini kotak paling kiri dan paling kanan terpotong tepi gambar.
    ax.set_xlim(-4, 108)
    ax.set_ylim(-2, 47)
    ax.axis("off")

    def box(x, y, w, h, lines, fc="white", ec="0.35", fs=6.3, mono=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=0.9, zorder=2))
        ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center",
                fontsize=fs, zorder=3,
                fontfamily="monospace" if mono else None, linespacing=1.45)

    def arrow(x1, y1, x2, y2, color="0.35", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                     color=color, lw=0.9, ls=ls, shrinkA=1, shrinkB=1, zorder=1))

    box(0, 19, 21, 12, ["Task and", "iso-compute budget",
                        r"$P \times D_\mathrm{eff} \times$ steps", "held fixed"])
    box(24.5, 19, 22, 12, ["Train, one GPU", "fixed thermal env.", "eval schedule pinned",
                           "to a 512-puzzle subset"])
    box(50, 33, 15, 10, ["nvidia-smi", "power.draw, 1 Hz"], fc="#EAF2FA")
    box(50, 5, 15, 10, ["CodeCarbon", "NVML energy"], fc="#EAF2FA")
    box(68.5, 19, 20, 12, ["idle subtraction", "4.7 W constant", "cross-validate:",
                           r"agreement $\geq$ 98.8%"])
    box(91.5, 19, 12.5, 12, ["Joules to", "target", "accuracy"], fc="#FDF0E3")

    arrow(21.3, 25, 24.2, 25)
    arrow(46.8, 27.6, 49.7, 35.0)
    arrow(46.8, 22.4, 49.7, 13.0)
    arrow(65.3, 36.0, 68.2, 27.6)
    arrow(65.3, 12.0, 68.2, 22.4)
    arrow(88.8, 25, 91.2, 25)

    arrow(78, 18.3, 78, 3.6, color=OI["vermillion"], ls=(0, (2.5, 1.6)))
    ax.text(50, 1.2, "rejected: instruments disagree, or wall time < 120 s (aborted / OOM)",
            ha="center", va="bottom", fontsize=7.6, color=OI["vermillion"], style="italic")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  tulis {out_path}")


def fig_sudoku_frontier(out_path, data_root):
    """Akurasi akhir Sudoku terhadap energi neto. SATU panel.

    Panel "akurasi vs kedalaman" yang dulu ada di sini dibuang karena menduplikasi
    panel pertama fig_crosstask_depth: data dan bentuk plotnya sama persis.
    """
    rec = os.path.join(data_root, "recipe_out")
    pts = []
    for depth, batch in ((9, "b192"), (18, "b192"), (36, "b96")):
        acc, wh = [], []
        for r in csv.DictReader(open(os.path.join(rec, "recipe_summary.csv"))):
            if r["hidden"] == "512" and r["D_eff"] == str(depth) and "recipe" in r["tag"]:
                acc.append(float(r["best_exact_pct"])); wh.append(float(r["smi_net_Wh"]))
        pts.append((depth, float(np.mean(wh)), float(np.mean(acc))))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
    ax.plot(xs, ys, "-", color="0.25", lw=1.4, zorder=1)
    for d, x, y in pts:
        ax.plot(x, y, "s", ms=6, mfc="white", mec="0.15", mew=1.2, zorder=3)
        off = (-30, -14) if d == 9 else (8, -4)
        ax.annotate(rf"$D_{{{d}}}$", (x, y), textcoords="offset points", xytext=off, fontsize=8.5)
    ax.set_xlabel("Net training energy (Wh)")
    ax.set_ylabel("Exact accuracy (%)")
    ax.set_xlim(min(xs) - 12, max(xs) + 12)
    ax.set_ylim(min(ys) - 4, max(ys) + 5)
    ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(out_path); plt.close(fig)
    print(f"  tulis {out_path}")


def fig_width_optimum(out_path, data_root):
    """Sumbu lebar pada D_eff=18: optimum di tengah pada h=512."""
    rec = os.path.join(data_root, "recipe_out", "recipe_summary.csv")
    g = {}
    for r in csv.DictReader(open(rec)):
        if r["D_eff"] == "18" and "recipe" in r["tag"]:
            g.setdefault(int(r["hidden"]), []).append(float(r["best_exact_pct"]))
    hs = sorted(g)
    m = [float(np.mean(g[h])) for h in hs]
    sd = [float(np.std(g[h], ddof=1)) for h in hs]
    fig, ax = plt.subplots(figsize=(2.9, 2.6))
    ax.errorbar(range(len(hs)), m, yerr=sd, fmt="D-", ms=5.5, mfc="white",
                color=OI["reddish"], ecolor=OI["reddish"], capsize=3, lw=1.3, mew=1.2)
    ax.set_xticks(range(len(hs))); ax.set_xticklabels([str(h) for h in hs])
    ax.set_xlabel("hidden size $h$")
    ax.set_ylabel("Exact accuracy (%)")
    ax.set_xlim(-0.35, len(hs) - 0.65)
    ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(out_path); plt.close(fig)
    print(f"  tulis {out_path}")


def fig_design_plane(out_path, data_root):
    """Bidang rancangan (P, D_eff): titik yang benar-benar dijalankan."""
    fig, ax = plt.subplots(figsize=(3.9, 3.0))
    ax.plot([512, 512, 512], [9, 18, 36], "-", color="0.6", lw=0.9, zorder=1)
    ax.plot([256, 512, 768], [18, 18, 18], "-", color="0.6", lw=0.9, zorder=1)
    ax.plot([512] * 3, [9, 18, 36], "o", ms=8, mfc="none", mec=OI["blue"], mew=1.6,
            label=r"depth sweep ($h{=}512$)", zorder=3)
    ax.plot([256, 512, 768], [18] * 3, "s", ms=8, mfc="none", mec=OI["reddish"], mew=1.6,
            label=r"width sweep ($D_{\mathrm{eff}}{=}18$)", zorder=3)
    ax.plot([512], [1], "^", ms=9, mfc="none", mec="0.15", mew=1.6,
            label="non-recursive baseline", zorder=3)
    ax.plot([256], [9], "o", ms=8, mfc="none", mec=OI["blue"], mew=1.6, alpha=0.45, zorder=2)
    ax.plot([256], [36], "o", ms=8, mfc="none", mec=OI["blue"], mew=1.6, alpha=0.45, zorder=2)
    ax.plot([256, 256, 256], [9, 18, 36], "-", color="0.75", lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.text(600, 1.15, "faded: Maze and ARC\ndepth sweep ($h{=}256$)", fontsize=7.2, color="0.45")
    ax.set_yscale("log"); ax.set_yticks([1, 9, 18, 36])
    ax.set_yticklabels(["1", "9", "18", "36"])
    ax.set_xticks([256, 512, 768])
    ax.set_xlabel(r"hidden size $h$   ($P \propto h^2$)")
    ax.set_ylabel(r"recursion depth $D_{\mathrm{eff}}$")
    ax.set_xlim(175, 880); ax.set_ylim(0.62, 62)
    ax.grid(alpha=0.25, lw=0.5)
    # Legenda ditaruh di luar kanan-atas area data: di dalam plot ia selalu
    # bertabrakan, entah dgn titik D=36, penanda baseline, atau titik D=9.
    ax.legend(frameon=False, fontsize=7.2, loc="upper left", bbox_to_anchor=(0.015, 0.46),
              handletextpad=0.4, borderpad=0.2, labelspacing=0.35)
    fig.savefig(out_path); plt.close(fig)
    print(f"  tulis {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=HERE)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "manuscript", "figures"))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    root = os.path.abspath(a.data_root)
    fig_energy_accuracy(os.path.join(out, "fig_energy_accuracy.pdf"), root)
    fig_crosstask(os.path.join(out, "fig_crosstask_depth.pdf"), root)
    fig_learning(os.path.join(out, "fig_learning.pdf"), root)
    fig_iso_accuracy(os.path.join(out, "fig_iso_accuracy.pdf"), root)
    fig_regime_map(os.path.join(out, "fig_regime_map.pdf"), root)
    fig_protocol(os.path.join(out, "fig_protocol.pdf"))
    fig_sudoku_frontier(os.path.join(out, "fig_sudoku_frontier.pdf"), root)
    fig_width_optimum(os.path.join(out, "fig_width_optimum.pdf"), root)
    fig_design_plane(os.path.join(out, "fig_design_plane.pdf"), root)
    mb = os.path.join(root, "microbench_results.csv")
    if not os.path.exists(mb):
        mb_alt = os.path.join(root, "..", "kalibrasi", "microbench_results.csv")
        root = os.path.dirname(os.path.abspath(mb_alt))
    fig_cost_model(os.path.join(out, "fig_cost_model.pdf"), root)
    joules_to_target_table(os.path.abspath(a.data_root))
    perseed_table(os.path.abspath(a.data_root))
    pilot_vs_faithful(os.path.abspath(a.data_root))
    agreement_floor(os.path.abspath(a.data_root))
    iso_accuracy_energy(os.path.abspath(a.data_root))
    eval_share_table(os.path.abspath(a.data_root))


if __name__ == "__main__":
    main()
