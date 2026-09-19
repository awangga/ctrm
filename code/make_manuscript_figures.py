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
# Fase BS: ARC dibaca HANYA dari subset sah arc1-aug1k-g400 (400 task). Folder lama
# (arc_depth_out, arc_d36_accum_out, arc_d9_preds_out) memakai subset arc1-aug1k-e512 yang
# ternyata augmentasi SATU task, sehingga tidak dipakai untuk figur hasil.
ARC_G400_D9 = "arc_d9_g400_out"
ARC_G400 = [f"{ARC_G400_D9}/recipe_summary.csv", "arc_d36_g400_out/recipe_summary.csv"]
ARC_COPY_INPUT = 60.75  # baseline copy-input token pada subset g400, ablation/trivial_baselines.md

# Gbr. 6 = dua panel berdampingan; ukuran fontnya dikunci lewat dua konstanta ini supaya
# panel kiri tidak terbaca lebih kecil daripada kanan. Skala cetak diukur dari lebar PDF
# keluaran (bukan figsize, karena bbox="tight" memangkas): textwidth elsarticle 390 pt,
# panel kiri 0.50\linewidth / 225,7 pt = 0,864x, panel kanan 0.43\linewidth / 194,1 pt = 0,864x.
PANEL6_TICK = 9.5      # efektif 8,21 pt
PANEL6_LABEL = 10.0    # efektif 8,64 pt
# Penanda panel "(a)"/"(b)". Elsevier menyarankan TIDAK menggabung beberapa gambar jadi satu
# berkas (alasan aksesibilitas); kedua panel Gbr. 6 memang sudah berkas terpisah, jadi yang
# ditambahkan hanya penandanya supaya caption dan teks bisa menyebut panel secara eksplisit
# dan pembaca layar bisa membedakannya. Ditaruh di ATAS area sumbu (bukan di dalamnya) agar
# tidak mungkin menimpa titik atau error bar; bbox="tight" lalu memperlebar margin atas
# sebesar tinggi teks, sama besar di kedua panel karena font dan offsetnya identik.
PANEL6_PANELLAB = 10.5  # efektif 9,07 pt: sedikit di atas tick panel (8,21 pt)
PANEL6_LAB_PAD = 3.0    # offset vertikal, pt


def panel_label(ax, text):
    """Tempel penanda panel di pojok kiri atas, tepat di luar area data."""
    ax.annotate(text, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(0.0, PANEL6_LAB_PAD), textcoords="offset points",
                ha="left", va="bottom", fontsize=PANEL6_PANELLAB, fontweight="bold",
                color="0.15", annotation_clip=False)

def legend_outside(fig, axes, where="below", ncol=3, pad_pt=5.0, **kw):
    """Legenda bersama di LUAR semua area sumbu (permintaan penulis, fase BU).

    Tidak ada kotak legenda yang boleh duduk di dalam sumbu data. Legenda ditaruh sebagai
    baris figur tepat di bawah (atau di atas) kotak-ketat gabungan semua sumbu, termasuk
    label dan tick, lalu savefig bbox="tight" memperluas kanvas untuk memuatnya. Hanya
    tata letak: tidak menyentuh data, warna, penanda, maupun batas sumbu.
    """
    axes = list(np.atleast_1d(axes))
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh); labels.append(ll)
    if "handles" in kw:
        handles = kw.pop("handles")
        labels = [h.get_label() for h in handles]
    order = kw.pop("order", None)
    if order is not None:
        handles = [handles[i] for i in order]; labels = [labels[i] for i in order]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    tb = [inv.transform_bbox(ax.get_tightbbox(r)) for ax in axes]
    pos = [ax.get_position() for ax in axes]
    xc = 0.5 * (min(b.x0 for b in pos) + max(b.x1 for b in pos))
    dy = pad_pt / 72.0 / fig.get_size_inches()[1]
    if where == "below":
        y, loc = min(b.y0 for b in tb) - dy, "upper center"
    else:
        y, loc = max(b.y1 for b in tb) + dy, "lower center"
    kw.setdefault("frameon", False)
    return fig.legend(handles, labels, loc=loc, bbox_to_anchor=(xc, y),
                      bbox_transform=fig.transFigure, ncol=ncol, borderaxespad=0.0, **kw)


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
    """Energi net kumulatif (Wh) via trapesium atas (P - P_idle), Eq. (2).

    TIDAK di-clip pada nol, supaya identik dengan run_recipe.py yang menghasilkan
    angka energi yang dilaporkan. Sampel di bawah lantai idle mengurangi total.
    Selisih terhadap versi ter-clip di seluruh run yang dilaporkan < 0,001%.
    """
    net = w - IDLE_W
    if len(ts) < 2:
        return np.zeros_like(net)
    dt = np.diff(ts)
    seg = 0.5 * (net[1:] + net[:-1]) * dt          # Joule
    return np.concatenate([[0.0], np.cumsum(seg)]) / 3600.0


def eval_points(path, metric="all/exact_accuracy"):
    """(step, acc%) dari tiap checkpoint eval, plus waktu unix."""
    out = []
    for line in open(path):
        if '"phase": "eval"' not in line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append((float(d["t"]), int(d["step"]), 100.0 * float(d.get(metric, 0.0))))
    return out


def trajectory(tag, out_dir, metric="all/exact_accuracy"):
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
    for t, _step, acc in eval_points(pg, metric):
        i = int(np.searchsorted(ts, t))
        if i >= len(cum):
            i = len(cum) - 1
        xs.append(cum[i])
        ys.append(acc)
    return np.array(xs), np.array(ys)


def fig_energy_accuracy(out_path, data_root):
    """Lintasan energi-akurasi Sudoku h512, tiga kedalaman x tiga seed."""
    rec = os.path.join(data_root, "recipe_out")
    # Linestyle membedakan D18 dan D36 juga dalam cetak hitam-putih: #E69F00 dan
    # #D55E00 nyaris sama begitu dikonversi ke grayscale.
    specs = [(9, "b192", OI["blue"], "o", "-"), (18, "b192", OI["orange"], "s", (0, (4.5, 1.8))),
             (36, "b96", OI["vermillion"], "^", (0, (1.4, 1.4)))]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    TARGET = 50.0
    for depth, batch, colour, marker, ls in specs:
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
        ax.plot(xm, ym, color=colour, marker=marker, markersize=2.6, markevery=3, ls=ls,
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
            # Label ditaruh di ruang kosong dengan garis penunjuk: posisi lama (kanan-bawah
            # bintang, berlatar putih) menutup kurva D18 dan ujung kurva D36. D9 ke kiri-atas
            # (celah di atas kurva D9 sebelum ia memotong target), D18 ke kanan-bawah (celah
            # antara ujung kurva D18 dan D36).
            off, ha = {9: ((-14, 16), "right")}.get(depth, ((14, -15), "left"))
            ax.annotate(f"{mx:.0f} Wh ({len(reach)}/{len(curves)})", (mx, TARGET),
                        textcoords="offset points", xytext=off, ha=ha, va="center",
                        fontsize=8, color=colour,
                        arrowprops=dict(arrowstyle="-", color=colour, lw=0.6,
                                        shrinkA=1.5, shrinkB=4.5))
    ax.axhline(TARGET, color=OI["grey"], ls=":", lw=0.9, zorder=0)
    # Teks target dipindah ke ruang kosong di atas garis putus-putus (sebelum kurva mana
    # pun memotongnya) supaya tidak berdesakan dengan bintang capai-target.
    # Dipindah ke ujung kiri garis (daerah kosong, semua kurva masih < 30%) supaya tidak
    # berdesakan dengan label capai-target D9 yang kini di kiri-atas bintangnya.
    ax.annotate("target 50% exact", (ax.get_xlim()[0], TARGET), textcoords="offset points",
                xytext=(4, 3), ha="left", va="bottom", fontsize=8, color=OI["grey"])
    ax.set_xlabel("Cumulative net GPU energy (Wh)")
    ax.set_ylabel("Exact accuracy (\\%)" if False else "Exact accuracy (%)")
    ax.set_ylim(bottom=0)
    sec = ax.secondary_xaxis("top", functions=(lambda wh: wh / 1000 * GRID * 1000,
                                               lambda g: g / (GRID * 1000) * 1000))
    sec.set_xlabel(f"Cumulative CO$_2$e (g), at {GRID * 1000:.1f} g/kWh", fontsize=8.5)
    sec.tick_params(labelsize=8)
    legend_outside(fig, ax, "below", ncol=3, handlelength=2.4, columnspacing=2.0)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  tulis {out_path}")


def fig_cost_model(out_path, data_root):
    """Cost-model energi per langkah dari microbench hardware, log-log + CI bootstrap."""
    csv_path = os.path.join(data_root, "microbench_results.csv")
    P, J, Dv = [], [], []
    for r in csv.DictReader(open(csv_path)):
        try:
            P.append(float(r["params_M"]) * float(r["D_eff"]))
            J.append(float(r["J_per_step"]))
            Dv.append(int(float(r["D_eff"])))
        except (KeyError, ValueError):
            continue
    P, J, Dv = np.array(P), np.array(J), np.array(Dv)
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
                    color="0.55", alpha=0.16, lw=0)
    ax.plot(np.exp(xs), np.exp(la + b * xs), color="0.2", lw=1.2, zorder=2,
            label=rf"fit, $b={b:.2f}$")
    # Acuan kemiringan 1 diikat ke titik terkecil: tanpa garis ini sub-linearitas
    # (b < 1) tidak terlihat pada sumbu log-log, karena mata tak punya pembanding.
    i0 = int(np.argmin(P))
    ax.plot(np.exp(xs), J[i0] * np.exp(xs) / P[i0], ls=(0, (4, 2)), color="0.55", lw=1.0,
            zorder=2, label="slope 1 (linear)")
    # Warna+penanda per D_eff: 14 titik itu grid 5 lebar x 3 kedalaman, dan tanpa
    # pembedaan ini beberapa titik terbaca saling tindih. Palet sama dgn Gbr. 4-5.
    dcol = {9: OI["blue"], 18: OI["orange"], 36: OI["vermillion"]}
    dmark = {9: "o", 18: "s", 36: "^"}
    for d in (9, 18, 36):
        sel = Dv == d
        if not sel.any():
            continue
        ax.scatter(P[sel], J[sel], s=20, color=dcol[d], marker=dmark[d], zorder=4,
                   edgecolor="white", linewidth=0.4,
                   label=rf"$D_{{\mathrm{{eff}}}}{{=}}{d}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$P \cdot D_{\mathrm{eff}}$ (M)")
    ax.set_ylabel("Energy per step (J)")
    ax.tick_params(labelsize=9)
    ax.text(0.04, 0.96, f"$b={b:.2f}$ [{lo:.2f}, {hi:.2f}]\n$R^2={r2:.2f}$, $n={len(P)}$",
            transform=ax.transAxes, va="top", fontsize=9)
    # Legenda 8.5 pt: figur ini dicetak pada 0,82x (0.5\linewidth elsarticle preprint),
    # jadi apa pun di bawah 8.5 pt jatuh di bawah ambang 7 pt Elsevier.
    # Urutan: kolom kiri tiga penanda D_eff, kolom kanan dua garis.
    legend_outside(fig, ax, "below", ncol=2, fontsize=9, handlelength=1.8, order=[2, 3, 4, 0, 1],
                   handletextpad=0.45, labelspacing=0.3, columnspacing=1.6)
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
    # Fase BS: ARC dibaca dari subset sah arc1-aug1k-g400 (400 task). Folder lama arc_depth_out
    # memakai subset yang ternyata satu task beserta augmentasinya, jadi tidak dipakai lagi.
    src = [("Sudoku-Extreme (exact)", ["recipe_out/recipe_summary.csv"], "best_exact_pct", (0, 1, 2)),
           ("Maze-Hard (token)", ["maze_depth_out/recipe_summary.csv"], "best_token_pct", (0, 1, 2)),
           ("ARC-AGI-1 (token, subset g400)", ARC_G400, "best_token_pct", (0, 1, 2, 3, 4))]
    print("\n  Akurasi per seed (rezim faithful):")
    for task, rels, col, seeds in src:
        paths = [os.path.join(data_root, rel) for rel in rels]
        if not all(os.path.exists(p) for p in paths):
            continue
        groups = collections.defaultdict(dict)
        for path in paths:
            for r in csv.DictReader(open(path)):
                m = _re.search(r"_s(\d+)$", r["tag"])
                if not m:
                    continue
                groups[r["tag"][:m.start()]][int(m.group(1))] = float(r[col])
        print(f"    {task}")
        for base in sorted(groups):
            v = groups[base]
            got = [v[s] for s in sorted(v)]
            cells = " ".join(f"{v[s]:6.2f}" if s in v else "    --" for s in seeds)
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
    """Energi yang dibutuhkan D9 untuk MENYAMAI akurasi checkpoint-terbaik tiap pesaing.

    Perbandingan iso-akurasi: metrik Joules-to-target yang sama, tetapi target
    diambil dari akurasi akhir yang benar-benar dicapai pesaing, bukan satu ambang
    tetap. Ini yang menjawab "berapa energi untuk mutu yang sama".
    """
    import statistics as _st
    rec = os.path.join(data_root, "recipe_out")
    rivals = [("D_eff=36", 36.26, 340.4), ("D_eff=18", 50.07, 358.0),
              ("non-recursive baseline", 49.67, 384.0)]
    print("\n  Energi iso-akurasi: biaya D9 untuk menyamai akurasi terbaik pesaing")
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


def iso_accuracy_energy_arc(data_root):
    """ARC-AGI-1 (token, checkpoint terbaik): energi D9 untuk menyamai akurasi terbaik D36.

    Fase BS: subset sah arc1-aug1k-g400. Target = rerata akurasi token terbaik D36 pada
    subset itu, dihitung dari summary CSV (tidak dipatok).
    """
    import statistics as _st
    d9dir = os.path.join(data_root, ARC_G400_D9)
    d36sp = os.path.join(data_root, ARC_G400[1])
    if not os.path.exists(d36sp):
        return
    rows = list(csv.DictReader(open(d36sp)))
    d36 = [float(r["best_token_pct"]) for r in rows if r["D_eff"] == "36"]
    wh36 = [float(r["smi_net_Wh"]) for r in rows if r["D_eff"] == "36"]
    target, rival_wh = _st.mean(d36), _st.mean(wh36)
    per = []
    for seed in range(5):
        tr = trajectory(f"h256_d9_recipe_b48_s{seed}", d9dir, metric="all/accuracy")
        if tr is None:
            continue
        x, y = tr
        h = np.where(y >= target)[0]
        per.append(float(x[h[0]]) if len(h) else None)
    ok = [v for v in per if v is not None]
    cells = ", ".join(f"{v:.0f}" if v is not None else "--" for v in per)
    if ok:
        m = _st.mean(ok); sd = _st.stdev(ok) if len(ok) > 1 else 0.0
        print(f"\n  ARC iso-akurasi (token, checkpoint terbaik, subset g400): D9 menyamai D36 {target:.2f}%: "
              f"D9=[{cells}] -> {m:.0f}+/-{sd:.0f} Wh ({len(ok)}/{len(per)}); D36 spent {rival_wh:.0f} Wh, "
              f"hemat {100*(rival_wh-m)/rival_wh:.0f}%")


def fig_crosstask(out_path, data_root):
    """Efek kedalaman di tiga task, memakai SELURUH seed yang tersedia per task."""
    import collections
    import statistics as _st
    # Fase BM: tiap panel menampilkan garis baseline sepele metriknya (trivial_baselines.py),
    # karena tanpa garis itu panel Maze menyesatkan: seluruh rentangnya ada DI BAWAH 87,51%.
    # Fase BS: ARC dari subset sah arc1-aug1k-g400 (dua folder, D9 dan D36 saja; tidak ada D18
    # pada subset sah). Garis baseline ARC = copy-input 60,75% pada subset yang sama
    # (ablation/trivial_baselines.md), setara Maze yang juga memakai copy-input.
    src = [("Sudoku-Extreme", ["recipe_out/recipe_summary.csv"], "best_exact_pct",
            "Exact accuracy (%)", ("h512_d9_recipe", "h512_d18_recipe", "h512_d36_recipe"), 0.0),
           ("ARC-AGI-1", ARC_G400, "best_token_pct",
            "Token accuracy (%)", ("h256_d9_recipe", None, "h256_d36_recipe"), ARC_COPY_INPUT),
           ("Maze-Hard", ["maze_depth_out/recipe_summary.csv"], "best_token_pct",
            "Token accuracy (%)", ("h256_d9_recipe", "h256_d18_recipe", "h256_d36_recipe"), 87.51)]
    # Lebar 5,45 in (bukan 6,6): figur dicetak 0.95\linewidth, jadi pada 6,6 in semua teks
    # menyusut ke 0,81x dan label baseline jatuh ke 5,5 pt.
    fig, axes = plt.subplots(1, 3, figsize=(5.45, 2.45))
    for ax, (task, rels, col, ylab, prefixes, base) in zip(axes, src):
        paths = [os.path.join(data_root, rel) for rel in rels]
        if not all(os.path.exists(p) for p in paths):
            continue
        vals = collections.defaultdict(list)
        for path in paths:
            for r in csv.DictReader(open(path)):
                for d, pre in zip((9, 18, 36), prefixes):
                    if pre is not None and r["tag"].startswith(pre):
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
        ax.margins(x=0.22, y=0.18)
        lo, hi = ax.get_ylim()
        allv = [v for d in xs for v in vals[d]]
        if base > min(allv) - 50:                       # baseline dekat/di dalam rentang data
            ax.set_ylim(min(lo, base - 0.45 * (hi - lo)), max(hi, base + 0.30 * (hi - lo)))
            ax.axhline(base, color=OI["vermillion"], ls="--", lw=1.0, zorder=1)
            ax.text(0.98, base, f"trivial baseline\n{base:g}%", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=7.6, color=OI["vermillion"],
                    linespacing=1.1)
    fig.tight_layout(w_pad=1.0)
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
    specs = [(9, "b192", OI["blue"], "o", "-"), (18, "b192", OI["orange"], "s", (0, (4.5, 1.8))),
             (36, "b96", OI["vermillion"], "^", (0, (1.4, 1.4)))]
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    for depth, batch, colour, marker, ls in specs:
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
        ax.plot(xm, ym, color=colour, marker=marker, markersize=2.6, markevery=3, ls=ls,
                label=rf"$D_{{\mathrm{{eff}}}}={depth}$ ({xm[-1]/1000:.0f}k steps)", zorder=3)
        if depth == 18:
            # Checkpoint terbaik per seed ditandai: angka yang dibahas caption berasal
            # dari puncak tiap seed, bukan dari puncak kurva rata-rata, dan tanpa
            # penanda ini pembaca tak bisa menemukannya di figur.
            bx = [float(cx[int(np.argmax(cy))]) for cx, cy in curves]
            by = [float(np.max(cy)) for cx, cy in curves]
            ax.plot(bx, by, "*", ms=8.5, mfc=colour, mec="white", mew=0.6, ls="", zorder=5)
            # Label ke kiri-atas bintang, di ruang kosong di atas kurva, dengan garis penunjuk:
            # posisi lama (kanan-atas) menimpa kurva D9 sekitar langkah 40k.
            ax.annotate(rf"best checkpoints, mean {np.mean(by):.1f}\%".replace("\\%", "%"),
                        (float(np.mean(bx)), float(np.max(by))), textcoords="offset points",
                        xytext=(-16, 13), ha="right", va="center", fontsize=8, color=colour,
                        arrowprops=dict(arrowstyle="-", color=colour, lw=0.6,
                                        shrinkA=1.5, shrinkB=4.5))
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Exact accuracy (%)")
    ax.set_ylim(bottom=0)
    legend_outside(fig, ax, "below", ncol=3, handlelength=2.4, columnspacing=2.0)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  tulis {out_path}")


def fig_iso_accuracy(out_path, data_root):
    """Energi untuk menyamai akurasi akhir tiap pesaing: batang berpasangan."""
    import statistics as _st
    rec = os.path.join(data_root, "recipe_out")
    rivals = [(r"$D_{\mathrm{eff}}{=}36$", 36.26, 340.4), (r"$D_{\mathrm{eff}}{=}18$", 50.07, 358.0),
              ("non-recursive", 49.67, 384.0)]
    labels, rival_wh, d9_wh, d9_sd, targets = [], [], [], [], []
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
        targets.append(target)
        rival_wh.append(rwh)
        d9_wh.append(_st.mean(per))
        d9_sd.append(_st.stdev(per) if len(per) > 1 else 0.0)
    idx = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.bar(idx - 0.19, rival_wh, 0.36, color=OI["grey"], label="rival, full budget")
    ax.bar(idx + 0.19, d9_wh, 0.36, yerr=d9_sd, capsize=3, color=OI["blue"],
           label=r"$D_{\mathrm{eff}}{=}9$ to match")
    # Label persen ditaruh DI ATAS ujung error bar: dengan offset tetap 14 Wh ia
    # ditembus error bar yang s.d.-nya mencapai 19 Wh.
    for i, (r, d, s) in enumerate(zip(rival_wh, d9_wh, d9_sd)):
        ax.text(i + 0.19, d + s + 13, f"\u2212{100*(r-d)/r:.0f}%", ha="center", fontsize=8.5,
                color=OI["blue"], fontweight="bold")
    ax.set_xticks(idx)
    # Akurasi target ikut dicetak: angka inilah yang mendefinisikan arti "matched".
    ax.set_xticklabels([f"{n}\nmatch {t:.1f}%" for n, t in zip(labels, targets)], fontsize=8.6)
    ax.set_ylabel("Net energy (Wh)")
    ax.tick_params(axis="y", labelsize=8.6)
    # 1,30 -> 1,12: ruang kosong di atas batang dulu disediakan untuk legenda, yang kini
    # di luar sumbu; label persen tetap di bawah batas ini.
    ax.set_ylim(0, max(rival_wh) * 1.12)
    ax.set_yticks([0, 100, 200, 300, 400])
    legend_outside(fig, ax, "above", ncol=2, fontsize=8.6, handlelength=1.4, columnspacing=1.0)
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


def fig_baseline_distance(out_path, data_root):
    """Kiri: posisi tiap task relatif terhadap baseline sepele dan langit-langit skalanya.
    Kanan: efek kedalaman bertanda dengan CI 95% Welch.

    Fase BM: panel kiri tidak lagi memakai 'headroom' (sumbu peta rezim lama), karena
    reviewer benar bahwa jarak-ke-100% tak sebanding antar metrik. Yang ditampilkan kini
    adalah acuan yang bisa diperiksa siapa pun: baseline copy-input / majority-class pada
    subset evaluasi yang sama. Maze-Hard duduk DI BAWAH baseline copy-input, sehingga
    metrik tokennya tidak mengukur penguasaan task pada anggaran ini.
    """
    from stats_table import welch
    rec = os.path.join(data_root, "recipe_out")
    # Baseline sepele dari ablation/trivial_baselines.md. Sudoku memakai akurasi EXACT, yang
    # baseline sepelenya 0%; Maze dan ARC memakai akurasi token (copy-input 87,51% dan 60,75%).
    # Fase BS: ARC dari subset sah arc1-aug1k-g400, hanya D9 dan D36 (tak ada D18 di sana).
    specs = [
        ("Sudoku-Extreme", "exact", os.path.join(rec, "recipe_summary.csv"), "best_exact_pct",
         lambda r: r["hidden"] == "512" and "recipe" in r["tag"], 0.0, "informative"),
        ("ARC-AGI-1", "token", [os.path.join(data_root, rel) for rel in ARC_G400],
         "best_token_pct", lambda r: True, ARC_COPY_INPUT, "informative"),
        ("Maze-Hard", "token", os.path.join(data_root, "maze_depth_out", "recipe_summary.csv"),
         "best_token_pct", lambda r: True, 87.51, "at floor"),
    ]
    # Lebar 5,75 in (bukan 6,6): figur dicetak \linewidth (390 pt), sehingga teksnya kini
    # tercetak ~0,97x alih-alih 0,85x.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(5.75, 2.9), sharey=True,
                                   gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.10})
    shades = {"informative": OI["blue"], "at floor": OI["vermillion"]}
    marks = {"9": "o", "18": "s", "36": "^"}
    ylabels = []
    for row, (task, metric, path, col, filt, base, status) in enumerate(specs):
        y = len(specs) - 1 - row
        groups = {}
        for pth in (path if isinstance(path, list) else [path]):
            for r in csv.DictReader(open(pth)):
                if filt(r):
                    groups.setdefault(r["D_eff"], []).append(float(r[col]))
        means = {k: float(np.mean(v)) for k, v in groups.items()}
        # Fase BM: sumbu kiri adalah JARAK ke baseline sepele, bukan akurasi absolut. Pada skala
        # absolut 0-100% jarak Maze (-1 poin) tak terlihat, sehingga pesan utama figur hilang.
        axL.plot([-6, 104], [y, y], color="0.88", lw=5, solid_capstyle="butt", zorder=1)
        depths = [k for k in ("9", "18", "36") if k in groups]
        for k in depths:
            axL.errorbar(means[k] - base, y, xerr=np.std(groups[k], ddof=1), fmt=marks[k], ms=4.4,
                         color=shades[status], ecolor=shades[status], elinewidth=1, capsize=2,
                         zorder=3, mfc="white" if status == "at floor" else shades[status])
        lo_d = min(means[k] - base for k in depths)
        hi_d = max(means[k] - base for k in depths)
        axL.annotate(f"{lo_d:+.1f} to {hi_d:+.1f}", (hi_d, y), textcoords="offset points",
                     xytext=(8, -2), fontsize=8, color=shades[status], va="center")
        d9, d36 = groups["9"], groups["36"]
        t, df, pval, _ = welch(d36, d9)
        diff = means["36"] - means["9"]
        ci = _tcrit(df) * abs(diff / t)
        sig = pval < 0.05 / 3
        axR.errorbar(diff, y, xerr=ci, fmt="D", ms=5, color=shades[status], ecolor=shades[status],
                     elinewidth=1.3, capsize=3, mfc=shades[status] if sig else "white", zorder=3)
        axR.text(diff - 1.2, y - 0.34, (f"{diff:+.1f}" + ("" if sig else " n.s.")), ha="right",
                 fontsize=8.6, color=shades[status])
        print(f"    {task:15s} jarak ke baseline {base:g}%: "
              + ", ".join(f"D{k} {means[k] - base:+.2f}" for k in depths)
              + f";  D36-D9 {diff:+.2f}, CI95 Welch [{diff - ci:+.2f}, {diff + ci:+.2f}], p={pval:.4g}")
        ylabels.append((y, f"{task}\n({metric}), {status}"))
    axL.axvline(0, color=OI["vermillion"], lw=1.1, zorder=2)
    axL.text(0, len(specs) - 0.5, "trivial baseline", rotation=90, ha="right", va="top",
             fontsize=7.6, color=OI["vermillion"])
    # Batas kanan 94, bukan 82: label rentang Sudoku ("+36.3 to +62.4", 53 pt pada 7.4 pt)
    # mulai 8 pt di kanan penanda D=9 dan dulu menembus spine, lalu tertutup latar putih
    # panel kanan sehingga teksnya terbaca terpotong. Batas ini murni ruang gambar.
    # Fase BU: dinaikkan ke 104 karena pada lebar 5,75 in label itu kembali menyentuh spine.
    axL.set_xlim(-6, 104)
    axL.set_ylim(-0.75, len(specs) - 0.35)
    axL.set_yticks([y for y, _ in ylabels])
    axL.set_yticklabels([lab for _, lab in ylabels], fontsize=8.2)
    # Label sumbu dua baris: satu baris, kedua label sumbu-x saling bertabrakan di celah panel.
    axL.set_xlabel("Distance from the trivial baseline\n(points)")
    axR.axvline(0, color="0.45", lw=0.9, zorder=2)
    axR.set_xlabel("Depth effect,\n" + r"$D_{36}-D_9$ (points)")
    # Idem: "-26.2" rata-kanan di bawah diamond Sudoku melewati spine kiri pada batas -33.
    axR.set_xlim(-41, 12)
    # Tick ditetapkan eksplisit: pada lebar 5,75 in locator otomatis menjarangkannya.
    axL.set_xticks([0, 20, 40, 60, 80])
    axR.set_xticks([-30, -20, -10, 0, 10])
    for ax in (axL, axR):
        ax.grid(axis="x", alpha=0.25, lw=0.5)
    hs = [plt.Line2D([], [], marker=m, ls="", color="0.35", ms=5.2,
                     label=rf"$D_{{\mathrm{{eff}}}}{{=}}{k}$") for k, m in marks.items()]
    legend_outside(fig, (axL, axR), "below", ncol=3, handles=hs, fontsize=8.0,
                   handletextpad=0.2, columnspacing=1.6)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  tulis {out_path}")

def fig_protocol(out_path):
    """Diagram alur protokol pengukuran. Skema, bukan figur data.

    Sengaja memuat gerbang penolakan yang benar-benar dipakai (kesepakatan dua
    instrumen >=98.8%, run <120 s dibuang) supaya diagram ini menyampaikan
    keputusan, bukan sekadar hiasan kotak-panah.

    Tata letak dihitung dalam satuan POIN (sumbu mengisi seluruh kanvas, 1 unit = 1 pt),
    dan tiap kotak diukur dari teksnya sendiri: versi lama memakai kotak berukuran tetap,
    sehingga beberapa baris meluber melewati tepi kotak dan panah ke/dari kotak instrumen
    menabrak sudut kotak. Kini semua panah horizontal/vertikal di antara tepi kotak.
    Kotak "Joules to target accuracy" ditaruh di bawah kotak idle agar lebar total muat
    di lebar teks (\\linewidth) tanpa mengecilkan font di bawah ~8 pt saat dicetak.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    W, H = 400.0, 175.0
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis("off")
    FS, PADX, PADY, GAP = 8.0, 5.0, 4.5, 11.0
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def measure(lines):
        t = ax.text(0, 0, "\n".join(lines), fontsize=FS, linespacing=1.4, ha="center",
                    va="center")
        bb = inv.transform_bbox(t.get_window_extent(rend))
        t.remove()
        return bb.width + 2 * PADX, bb.height + 2 * PADY

    def box(cx, cy, w, h, lines, fc="white"):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0,rounding_size=3.5",
                                    fc=fc, ec="0.35", lw=0.9, zorder=2))
        ax.text(cx, cy, "\n".join(lines), ha="center", va="center", fontsize=FS,
                linespacing=1.4, zorder=3)

    def arrow(x1, y1, x2, y2, color="0.35", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                     color=color, lw=0.9, ls=ls, shrinkA=0, shrinkB=0.5,
                                     zorder=1))

    T_task = ["Task and", "iso-compute", "budget", r"$P \times D_\mathrm{eff} \times$ steps",
              "held fixed"]
    T_train = ["Train, one GPU", "fixed thermal env.", "eval schedule", "pinned to a",
               "512-puzzle subset"]
    T_smi = ["nvidia-smi", "power.draw, 1 Hz"]
    T_cc = ["CodeCarbon", "NVML energy"]
    T_idle = ["idle subtraction", "4.7 W constant", "cross-validate:", "agree 98.8-99.95%"]
    T_j = ["Joules to", "target", "accuracy"]
    w1, h1 = measure(T_task); w2, h2 = measure(T_train)
    ws, hs_ = measure(T_smi); wc, hc = measure(T_cc)
    wi, hi = measure(T_idle); wj, hj = measure(T_j)
    wm = max(ws, wc); hm = max(hs_, hc)
    total = w1 + w2 + wm + wi + 3 * GAP
    if total > W - 4:
        print(f"  PERINGATAN fig_protocol: lebar {total:.0f} pt > kanvas {W:.0f} pt")
    y0 = H - 8 - max(h1, h2, 2 * hm + 8) / 2          # garis tengah baris utama
    x = (W - total) / 2
    c1 = x + w1 / 2; x += w1 + GAP
    c2 = x + w2 / 2; x += w2 + GAP
    c3 = x + wm / 2; x += wm + GAP
    c4 = x + wi / 2
    dy = hm / 2 + 4                                    # lajur instrumen atas/bawah
    box(c1, y0, w1, h1, T_task)
    box(c2, y0, w2, h2, T_train)
    box(c3, y0 + dy, wm, hm, T_smi, fc="#EAF2FA")
    box(c3, y0 - dy, wm, hm, T_cc, fc="#EAF2FA")
    box(c4, y0, wi, hi, T_idle)
    yj = y0 - hi / 2 - GAP - hj / 2
    box(c4, yj, wj, hj, T_j, fc="#FDF0E3")
    arrow(c1 + w1 / 2, y0, c2 - w2 / 2, y0)
    for yy in (y0 + dy, y0 - dy):
        arrow(c2 + w2 / 2, yy, c3 - wm / 2, yy)
        arrow(c3 + wm / 2, yy, c4 - wi / 2, yy)
    arrow(c4, y0 - hi / 2, c4, yj + hj / 2)
    yr = min(y0 - h2 / 2, y0 - dy - hm / 2, yj - hj / 2) - 16
    arrow(c2, y0 - h2 / 2, c2, yr + 1, color="0.45", ls=(0, (2.5, 1.6)))
    tr = ax.text(c2 - w2 / 2, yr, "rejected: wall time < 120 s (aborted / OOM) or incomplete budget",
                 ha="left", va="top", fontsize=FS, color="0.45", style="italic")
    # Kanvas dipangkas ke isi (1 unit tetap = 1 pt): bbox="tight" tidak memangkas karena
    # sumbu (tak terlihat) mengisi seluruh kanvas.
    rb = inv.transform_bbox(tr.get_window_extent(rend))
    x0 = min(c1 - w1 / 2, rb.x0) - 2; x1 = max(c4 + wi / 2, rb.x1) + 2
    y1 = y0 + max(h1, h2, 2 * dy + hm) / 2 + 2; ylo = rb.y0 - 2
    fig.set_size_inches((x1 - x0) / 72, (y1 - ylo) / 72)
    ax.set_xlim(x0, x1); ax.set_ylim(ylo, y1)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  tulis {out_path}")


def fig_sudoku_frontier(out_path, data_root):
    """Akurasi akhir Sudoku terhadap energi neto. SATU panel.

    Panel "akurasi vs kedalaman" yang dulu ada di sini dibuang karena menduplikasi
    panel pertama fig_crosstask_depth: data dan bentuk plotnya sama persis.
    """
    rec = os.path.join(data_root, "recipe_out")
    # Encoding kedalaman disamakan dengan Gbr. 4-5 (biru/oranye/vermillion, o/s/^);
    # kotak hitam-putih yang lama tak bisa dihubungkan pembaca ke figur tetangga.
    style = {9: (OI["blue"], "o"), 18: (OI["orange"], "s"), 36: (OI["vermillion"], "^")}
    pts = []
    for depth, batch in ((9, "b192"), (18, "b192"), (36, "b96")):
        acc, wh = [], []
        for r in csv.DictReader(open(os.path.join(rec, "recipe_summary.csv"))):
            if r["hidden"] == "512" and r["D_eff"] == str(depth) and "recipe" in r["tag"]:
                acc.append(float(r["best_exact_pct"])); wh.append(float(r["smi_net_Wh"]))
        pts.append((depth, float(np.mean(wh)), float(np.mean(acc)),
                    float(np.std(wh, ddof=1)) if len(wh) > 1 else 0.0,
                    float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0, wh, acc))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
    ax.plot(xs, ys, "-", color="0.35", lw=1.2, zorder=1)
    for d, x, y, sx, sy, whs, accs in pts:
        col, mk = style[d]
        # Titik per seed + error bar: rata-rata saja membuang sebaran yang justru
        # menentukan apakah jarak antar kedalaman berarti.
        ax.scatter(whs, accs, s=11, color=col, alpha=0.40, marker=mk, zorder=2, linewidth=0)
        ax.errorbar(x, y, xerr=sx, yerr=sy, fmt=mk, ms=6, mfc="white", mec=col, mew=1.3,
                    color=col, ecolor=col, elinewidth=1.0, capsize=2.5, zorder=3)
        # D36 duduk paling bawah: labelnya digeser ke kanan-bawah supaya tidak menimpa
        # cap error bar horizontalnya (yang lebarnya s.d. energi).
        # D9 kini di kiri titik, rata kanan, di atas garis penghubung: posisi lama
        # (kiri-bawah) ditembus garis itu.
        off, ha, va = {9: ((-11, 1), "right", "center"), 36: ((12, -13), "left", "baseline")}.get(
            d, ((9, -3), "left", "baseline"))
        ax.annotate(rf"$D_{{{d}}}$", (x, y), textcoords="offset points", xytext=off,
                    ha=ha, va=va, fontsize=PANEL6_TICK, color=col)
    # Font dinaikkan lewat konstanta PANEL6_*: pada tick 8 pt lama, hasil cetaknya hanya
    # 6,9 pt, di bawah ambang 7 pt Elsevier.
    ax.set_xlabel("Net training energy (Wh)", fontsize=PANEL6_LABEL)
    ax.set_ylabel("Exact accuracy (%)", fontsize=PANEL6_LABEL)
    ax.tick_params(labelsize=PANEL6_TICK)
    allx = [v for p in pts for v in p[5]]; ally = [v for p in pts for v in p[6]]
    ax.set_xlim(min(allx) - 14, max(allx) + 14)
    # Batas y memperhitungkan ujung error bar, bukan hanya titik seed, lalu diberi
    # padding bawah: tanpa itu titik D36 dan cap-nya menempel garis sumbu. 11% dipakai
    # (bukan 8%) karena label $D_{36}$ digantung 13 pt di bawah titiknya.
    ylo = min(min(ally), min(p[2] - p[4] for p in pts))
    yhi = max(max(ally), max(p[2] + p[4] for p in pts))
    span = yhi - ylo
    ax.set_ylim(ylo - 0.11 * span, yhi + 0.06 * span)
    ax.grid(alpha=0.25, lw=0.5)
    panel_label(ax, "(a)")
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
    # Lebar disetel 2,671 in supaya lebar PDF keluaran (setelah bbox="tight") berbanding
    # 0.43/0.50 terhadap panel kiri: dengan begitu kedua panel Gbr. 6 menyusut pada skala
    # cetak yang sama (0,864x) dan fontnya terbaca sama besar, bukan kiri lebih kecil.
    fig, ax = plt.subplots(figsize=(2.671, 2.6))
    ax.errorbar(range(len(hs)), m, yerr=sd, fmt="D-", ms=5.5, mfc="white",
                color=OI["reddish"], ecolor=OI["reddish"], capsize=3, lw=1.3, mew=1.2)
    ax.set_xticks(range(len(hs))); ax.set_xticklabels([str(h) for h in hs])
    ax.tick_params(labelsize=PANEL6_TICK)
    ax.set_xlabel("hidden size $h$", fontsize=PANEL6_LABEL)
    ax.set_ylabel("Exact accuracy (%)", fontsize=PANEL6_LABEL)
    ax.set_xlim(-0.35, len(hs) - 0.65)
    ax.grid(alpha=0.25, lw=0.5)
    panel_label(ax, "(b)")
    fig.savefig(out_path); plt.close(fig)
    print(f"  tulis {out_path}")


def fig_design_plane(out_path, data_root):
    """Bidang rancangan (P, D_eff): titik yang benar-benar dijalankan."""
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    ax.plot([512, 512, 512], [9, 18, 36], "-", color="0.6", lw=0.9, zorder=1)
    ax.plot([256, 512, 768], [18, 18, 18], "-", color="0.6", lw=0.9, zorder=1)
    ax.plot([512] * 3, [9, 18, 36], "o", ms=8, mfc="none", mec=OI["blue"], mew=1.6,
            label=r"depth sweep ($h{=}512$)", zorder=3)
    ax.plot([256, 512, 768], [18] * 3, "s", ms=8, mfc="none", mec=OI["reddish"], mew=1.6,
            label=r"width sweep ($D_{\mathrm{eff}}{=}18$)", zorder=3)
    ax.plot([512], [1], "^", ms=9, mfc="none", mec="0.15", mew=1.6,
            label="non-recursive baseline", zorder=3)
    ax.plot([256, 256], [9, 36], "o", ms=8, mfc="none", mec=OI["blue"], mew=1.6, alpha=0.45,
            zorder=2, label=r"Maze and ARC depth sweep ($h{=}256$)")
    # Titik (256,18) dipakai bersama dengan sapuan lebar. Alih-alih menggesernya ke
    # posisi h yang tidak pernah dijalankan, ia digambar sebagai lingkaran lebih besar
    # yang MELINGKARI kotak sapuan lebar: keduanya terbaca, datanya tetap di tempat.
    ax.plot([256], [18], "o", ms=13, mfc="none", mec=OI["blue"], mew=1.6, alpha=0.45, zorder=2)
    ax.plot([256, 256, 256], [9, 18, 36], color="0.75", lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.set_yscale("log"); ax.set_yticks([1, 9, 18, 36])
    ax.set_yticklabels(["1", "9", "18", "36"])
    ax.set_xticks([256, 512, 768])
    ax.set_xlabel(r"hidden size $h$")
    ax.set_ylabel(r"recursion depth $D_{\mathrm{eff}}$")
    ax.set_xlim(175, 880); ax.set_ylim(0.62, 62)
    ax.grid(alpha=0.25, lw=0.5)
    # Legenda di bawah sumbu, di luar area data: di dalam plot ia selalu bertabrakan,
    # entah dgn titik D=36, penanda baseline, atau titik D=9.
    legend_outside(fig, ax, "below", ncol=1, fontsize=8, handletextpad=0.5,
                   labelspacing=0.3)
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
    fig_baseline_distance(os.path.join(out, "fig_baseline_distance.pdf"), root)
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
    iso_accuracy_energy_arc(os.path.abspath(a.data_root))
    eval_share_table(os.path.abspath(a.data_root))


if __name__ == "__main__":
    main()
