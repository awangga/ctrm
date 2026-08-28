#!/usr/bin/env python3
"""Rekonsiliasi jumlah run dan total energi GPU dari seluruh summary CSV yang di-commit.

Menghasilkan angka agregat yang dipakai manuscript (Section "Single-GPU Calibration")
supaya setiap angka telusur ke artefak, bukan ke ingatan (CLAUDE.md aturan integritas #1).

Pakai (repo kerja):  python3 reconcile_totals.py
Pakai (paket Zenodo): python3 code/reconcile_totals.py --data-root data
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Klasifikasi setiap summary CSV yang di-commit.
#   faithful = resep TRM sebenarnya (EMA, ~1e6 contoh teraugmentasi) -> semua angka hasil di paper
#   pilot    = feasibility/kalibrasi/toy, jauh di bawah konvergensi -> tidak dipakai sbg hasil
SUMMARIES = [
    ("recipe_out/recipe_summary.csv",                    "faithful", "Sudoku depth+width+baseline"),
    ("maze_depth_out/recipe_summary.csv",                "faithful", "Maze-Hard depth grid"),
    ("arc_depth_out/recipe_summary.csv",                 "faithful", "ARC-AGI-1 depth grid"),
    ("maze_real_out/recipe_summary.csv",                 "faithful", "Maze faithful-recipe probe"),
    ("arc_smoke_out/recipe_summary.csv",                 "pilot",    "ARC smoke test"),
    ("frontier_out/frontier_summary.csv",                "pilot",    "frontier pilot"),
    ("isoflop_out/isoflop_summary.csv",                  "pilot",    "iso-FLOP pilot"),
    ("scale_out/scale_summary.csv",                      "pilot",    "scale sweep pilot"),
    ("converge_out/converge_summary.csv",                "pilot",    "convergence probe D36"),
    ("converge_out/converge_summary_h128_d18_aug.csv",   "pilot",    "convergence probe D18 (aug)"),
    ("maze_out/maze_summary.csv",                        "pilot",    "Maze toy sweep"),
    ("budget_out/energy_xval.csv",                       "pilot",    "budget replication seeds 3-4"),
]

# Run yang TIDAK punya catatan energi per-run dan karenanya di luar total energi.
# Jujur dicatat, bukan disembunyikan.
EXCLUDED_NOTE = (
    "budget_out/replication_summary.csv memuat akurasi 3 seed (0,1,2) x 3 depth = 9 run pilot, "
    "tetapi power log-nya ditulis ke nama file tanpa sufiks seed (pw_h128_d{9,18,36}.csv) sehingga "
    "saling menimpa dan energi per-seed tidak dapat diatribusikan. Sembilan run itu dikecualikan "
    "dari total energi; hanya seed 3-4 (energy_xval.csv) yang punya cross-val per-run."
)


def energy_key(row):
    for k in ("smi_net_Wh", "net_energy_Wh"):
        if k in row:
            return k
    return None


def agree_key(row):
    for k in row:
        if "agree" in k.lower():
            return k
    return None


def grid_intensity(root):
    """Intensitas grid (kg CO2e/kWh) yang dipakai CodeCarbon, dibaca dari emissions_*.csv.

    Diverifikasi konstan di seluruh run (Indonesia). Dikembalikan None bila tak ada data.
    """
    import glob
    vals = []
    for f in glob.glob(os.path.join(root, "*", "emissions_*.csv")):
        for r in csv.DictReader(open(f)):
            try:
                e, k = float(r["emissions"]), float(r["energy_consumed"])
            except (KeyError, TypeError, ValueError):
                continue
            if k > 0:
                vals.append(e / k)
    if not vals:
        return None, 0, None
    spread = max(vals) - min(vals)
    return sum(vals) / len(vals), len(vals), spread


def collect(root):
    out = []
    for rel, cat, desc in SUMMARIES:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"WARNING: hilang {rel}", file=sys.stderr)
            continue
        rows = list(csv.DictReader(open(path)))
        if not rows:
            continue
        ek, ak = energy_key(rows[0]), agree_key(rows[0])
        wh = sum(float(r.get(ek) or 0) for r in rows) if ek else 0.0
        ags = [float(r[ak]) for r in rows if ak and r.get(ak) not in (None, "")]
        out.append(dict(rel=rel, cat=cat, desc=desc, n=len(rows), wh=wh,
                        agree_min=min(ags) if ags else None,
                        agree_max=max(ags) if ags else None,
                        agree_n=len(ags)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=HERE,
                    help="folder yang memuat *_out/ (default: folder skrip; "
                         "pada paket Zenodo pakai ../data)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = os.path.abspath(args.data_root)
    # repo kerja memakai ablation/, paket rilis memakai ablation_reports/
    subdir = "ablation_reports" if os.path.isdir(os.path.join(root, "ablation_reports")) else "ablation"
    out = args.out or os.path.join(root, subdir, "run_energy_reconciliation.md")

    recs = collect(root)
    tot = {c: dict(n=0, wh=0.0, ag=[]) for c in ("faithful", "pilot")}
    for r in recs:
        t = tot[r["cat"]]
        t["n"] += r["n"]
        t["wh"] += r["wh"]
        if r["agree_min"] is not None:
            t["ag"] += [r["agree_min"], r["agree_max"]]

    lines = ["# Rekonsiliasi run & energi (auto-generated)", "",
             "Dihasilkan oleh `eksperimen/frontier/reconcile_totals.py` dari summary CSV yang di-commit.",
             "Jangan diedit tangan; jalankan ulang skripnya.", "",
             "| summary CSV | rezim | isi | n run | net Wh | xval min | xval max |",
             "|---|---|---|---:|---:|---:|---:|"]
    for r in sorted(recs, key=lambda x: (x["cat"] != "faithful", x["rel"])):
        amin = f"{r['agree_min']:.2f}" if r["agree_min"] is not None else "-"
        amax = f"{r['agree_max']:.2f}" if r["agree_max"] is not None else "-"
        lines.append(f"| `{r['rel']}` | {r['cat']} | {r['desc']} | {r['n']} | {r['wh']:.1f} | {amin} | {amax} |")

    lines += ["", "## Agregat", "",
              "| rezim | n run | energi | cross-val CodeCarbon vs nvidia-smi |", "|---|---:|---:|---|"]
    for c in ("faithful", "pilot"):
        t = tot[c]
        rng = f"{min(t['ag']):.2f}–{max(t['ag']):.2f}%" if t["ag"] else "-"
        lines.append(f"| {c} | {t['n']} | {t['wh']:.1f} Wh = {t['wh']/1000:.2f} kWh | {rng} |")
    gn = sum(t["n"] for t in tot.values())
    gw = sum(t["wh"] for t in tot.values())
    gag = tot["faithful"]["ag"] + tot["pilot"]["ag"]
    lines.append(f"| **TOTAL** | **{gn}** | **{gw:.1f} Wh = {gw/1000:.2f} kWh** | "
                 f"{min(gag):.2f}–{max(gag):.2f}% |")

    gi, gi_n, gi_spread = grid_intensity(root)
    if gi:
        lines += ["", "## Jejak karbon (CO2e)", "",
                  f"Intensitas grid dari CodeCarbon: **{gi*1000:.2f} g CO2e/kWh** (Indonesia), konstan di "
                  f"{gi_n} run (rentang {gi_spread*1000:.4f} g/kWh). Faktor yang sama diterapkan ke energi "
                  "**net GPU** (idle-corrected) agar konsisten dengan metrik utama paper.", "",
                  "| rezim | energi net | CO2e |", "|---|---:|---:|"]
        for c in ("faithful", "pilot"):
            t = tot[c]
            lines.append(f"| {c} | {t['wh']/1000:.2f} kWh | {t['wh']/1000*gi:.2f} kg |")
        lines.append(f"| **TOTAL** | **{gw/1000:.2f} kWh** | **{gw/1000*gi:.2f} kg** |")
        lines += ["",
                  "Sebagai pembanding, angka CO2e mentah CodeCarbon (CPU+GPU+RAM, tanpa koreksi idle) "
                  "lebih tinggi; nilai di atas sengaja memakai basis net-GPU yang sama dengan seluruh "
                  "metrik energi paper.", ""]

    lines += ["", "## Angka yang dipakai manuscript", "",
              f"- Rezim faithful-recipe (semua hasil yang dilaporkan): **{tot['faithful']['n']} run, "
              f"{tot['faithful']['wh']/1000:.1f} kWh**, cross-val "
              f"**{min(tot['faithful']['ag']):.2f}–{max(tot['faithful']['ag']):.2f}%** pada setiap run.",
              f"- Seluruh studi (faithful + pilot): **{gn} run, {gw/1000:.1f} kWh**.",
              f"- Cross-val pilot turun sampai **{min(tot['pilot']['ag']):.2f}%**, jadi klaim "
              "\"99.1–99.95% pada setiap run\" HANYA sah bila di-scope ke rezim faithful-recipe.",
              "", "## Dikecualikan dari total energi", "", EXCLUDED_NOTE, ""]

    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> ditulis ke {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
