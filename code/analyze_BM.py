#!/usr/bin/env python3
"""Analisis fase BM, dijalankan SEKALI setelah seluruh 18 run selesai (prereg_BM.md butir 3).

Skrip ini ditulis dan di-commit SEBELUM hasil ada, sehingga analisisnya pra-spesifikasi.
Empat bagian, persis kontras yang dipra-registrasi:

  A1  ARC D36 alokasi baru (batch efektif 48) vs D9/D18 lama  -> apakah defisit D36 bertahan
  A2  Sudoku D36 alokasi baru (batch efektif 192) vs D9/D18   -> idem
  B   ARC D9 (TRM) vs baseline non-rekursif pada epoch sama
  C   ARC D9 vs D36 di bawah pembelahan separuh-seleksi / separuh-pelaporan

Uji: Welch dua-sisi + permutasi eksak; ambang Bonferroni per sumbu 0,05/3 = 0,0167.
Usage: python3 analyze_BM.py [data_root]

DIBEKUKAN 17 September 2026. Pengerasan higienis data (buang run cacat, dedup tag, wajibkan
jumlah run dan jumlah checkpoint, cetak lantai permutasi, laporan tidak lagi menimpa repo bila
data_root lain) ditambahkan pada tanggal itu SETELAH audit fase BR membuktikan bahwa lima baris
run gagal yang menumpuk di summary mengubah hasil batch B dari p=0,2707 menjadi p=0,0090.
Tidak ada uji, ambang, kontras, atau definisi metrik yang diubah.

Saat menguji pengerasan itu, skrip dijalankan atas data yang belum lengkap dan keluaran bagian A1
sebagian TERLIHAT. Setelah titik itu skrip tidak diubah lagi. Analisis pra-registrasi yang sah
adalah yang dijalankan sekali setelah seluruh 18 run selesai.
"""
import csv, glob, itertools, json, math, os, re, sys, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = sys.argv[1] if len(sys.argv) > 1 else HERE
sys.path.insert(0, HERE)
from stats_table import welch

ALPHA = 0.05 / 3
WARN = []      # peringatan higienis data, dicetak di kepala laporan


NCKPT = 25            # checkpoint evaluasi per run; run berjalan punya lebih sedikit
MIN_WALL_S = 1000.0   # run faithful terpendek yang sah > 6000 s; run gagal fase BM = 12 s


def rows(rel, expect=None):
    """Baris summary yang SAH saja.

    run_recipe.py meng-APPEND ke recipe_summary.csv, jadi run yang diulang atau gagal-parsial
    menumpuk baris di berkas yang sama. Dibuktikan pada audit fase BR: menempelkan lima baris
    run gagal (12 s, akurasi 0) ke summary batch B mengubah hasilnya dari p=0,2707 (gagal)
    menjadi p=0,0090 (LOLOS). Baris sampah dapat MEMBALIK kesimpulan, jadi disaring di sini.

    Aturan: buang wall_s < MIN_WALL_S, lalu dedup per tag dengan mengambil baris TERAKHIR
    (run ulang menimpa run lama), lalu wajibkan jumlahnya persis `expect` bila diberikan.
    """
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return []
    raw = list(csv.DictReader(open(p)))
    keep, dropped = {}, 0
    for r in raw:
        try:
            if float(r.get("wall_s", 0)) < MIN_WALL_S:
                dropped += 1
                continue
        except ValueError:
            dropped += 1
            continue
        keep[r["tag"]] = r          # tag sama -> baris terakhir menang
    out = list(keep.values())
    if dropped:
        WARN.append(f"{rel}: {dropped} baris dibuang (wall_s < {MIN_WALL_S:.0f} s)")
    if len(raw) - dropped != len(out):
        WARN.append(f"{rel}: {len(raw) - dropped - len(out)} baris duplikat tag, diambil yang terakhir")
    if expect is not None and len(out) != expect:
        WARN.append(f"**{rel}: {len(out)} run sah, DIHARAPKAN {expect}. Analisis batch ini TIDAK SAH.**")
    return out


def cells(rs, col, key="D_eff"):
    out = {}
    for r in rs:
        try:
            out.setdefault(r[key], []).append(float(r[col]))
        except (KeyError, ValueError):
            pass
    return out


def perm_p(a, b):
    obs = st.mean(a) - st.mean(b)
    pool = a + b
    n = len(a)
    hit = tot = 0
    for idx in itertools.combinations(range(len(pool)), n):
        x = [pool[i] for i in idx]
        y = [pool[i] for i in range(len(pool)) if i not in idx]
        tot += 1
        hit += abs(st.mean(x) - st.mean(y)) >= abs(obs) - 1e-12
    return hit / tot, 2 / tot


def line(label, a, b):
    if len(a) < 2 or len(b) < 2:
        return f"| {label} | n terlalu kecil | | | | |"
    try:
        return _line(label, a, b)
    except Exception as e:                      # mis. ZeroDivisionError bila kedua grup varians nol
        return f"| {label} | GAGAL DIHITUNG: {type(e).__name__} | | | | |"


def _line(label, a, b):
    t, df, p, d = welch(a, b)
    pp, floor = perm_p(a, b)
    verd = "LOLOS" if p < ALPHA else "gagal"
    # lantai permutasi dicetak: pada n=3 lawan 3 hanya ada C(6,3)=20 pembelahan, jadi p terkecil
    # yang mungkin adalah 0,10. Tanpa keterangan ini "perm 0.1000" mudah dibaca sebagai bukti null.
    return (f"| {label} | {st.mean(a):.2f}±{st.stdev(a):.2f} | {st.mean(b):.2f}±{st.stdev(b):.2f} | "
            f"{st.mean(a)-st.mean(b):+.2f} | p={p:.4f} (perm {pp:.4f}, lantai {floor:.4f}) | {verd} |")


out = ["# Hasil fase BM (analisis pra-spesifikasi, dijalankan sekali)\n",
       f"Ambang Bonferroni per sumbu: {ALPHA:.4f}.\n"]

# ---------------------------------------------------------------- A1 ARC
arc_old = rows("arc_depth_out/recipe_summary.csv")
arc_new = rows("arc_d36_accum_out/recipe_summary.csv", expect=5)
if arc_old and arc_new:
    o = cells(arc_old, "best_token_pct")
    n36 = [float(r["best_token_pct"]) for r in arc_new]
    out += ["\n## A1. ARC-AGI-1: sel D36 dengan batch efektif 48 (compute sama, epoch sama)\n",
            "| kontras | grup 1 | grup 2 | selisih | uji | Bonferroni |", "|---|---|---|---|---|---|",
            line("D9 vs D36 (alokasi LAMA, batch 24)", o["9"], o["36"]),
            line("D9 vs D36 (alokasi BARU, batch efektif 48)", o["9"], n36),
            line("D18 vs D36 (alokasi BARU)", o["18"], n36),
            line("D36 lama vs D36 baru (efek alokasi saja)", o["36"], n36)]
    e_old = st.mean([float(r["smi_net_Wh"]) for r in arc_old if r["D_eff"] == "36"])
    e_new = st.mean([float(r["smi_net_Wh"]) for r in arc_new])
    out.append(f"\nEnergi net D36: lama {e_old:.0f} Wh, baru {e_new:.0f} Wh "
               f"({100*(e_new-e_old)/e_old:+.0f}%). Langkah optimizer baru kira-kira separuh lama, "
               f"contoh yang dikonsumsi sama.")
else:
    out.append("\n## A1. Ringkasan run belum lengkap (`arc_depth_out/` atau `arc_d36_accum_out/`); batch A1 belum selesai.\n")

# ---------------------------------------------------------------- A2 Sudoku
sud_old = rows("recipe_out/recipe_summary.csv")
sud_new = rows("sudoku_d36_accum_out/recipe_summary.csv", expect=3)
if sud_old and sud_new:
    o = cells([r for r in sud_old if r["hidden"] == "512" and "recipe" in r["tag"]], "best_exact_pct")
    n36 = [float(r["best_exact_pct"]) for r in sud_new]
    out += ["\n## A2. Sudoku-Extreme: sel D36 dengan batch efektif 192\n",
            "| kontras | grup 1 | grup 2 | selisih | uji | Bonferroni |", "|---|---|---|---|---|---|",
            line("D9 vs D36 (alokasi LAMA, batch 96)", o["9"], o["36"]),
            line("D9 vs D36 (alokasi BARU, batch efektif 192)", o["9"], n36),
            line("D18 vs D36 (alokasi BARU)", o["18"], n36),
            line("D36 lama vs D36 baru (efek alokasi saja)", o["36"], n36)]
else:
    out.append("\n## A2. Ringkasan run belum lengkap (`recipe_out/` atau `sudoku_d36_accum_out/`); batch A2 belum selesai.\n")

# ---------------------------------------------------------------- B baseline ARC
base = rows("arc_baseline_out/recipe_summary.csv", expect=5)
if base and arc_old:
    o = cells(arc_old, "best_token_pct")
    b = [float(r["best_token_pct"]) for r in base]
    out += ["\n## B. ARC-AGI-1: TRM dangkal vs baseline non-rekursif (epoch sama)\n",
            "| kontras | grup 1 | grup 2 | selisih | uji | Bonferroni |", "|---|---|---|---|---|---|",
            line("D9 vs baseline non-rekursif", o["9"], b),
            line("D36 (lama) vs baseline non-rekursif", o["36"], b)]
    eb = st.mean([float(r["smi_net_Wh"]) for r in base])
    e9 = st.mean([float(r["smi_net_Wh"]) for r in arc_old if r["D_eff"] == "9"])
    out.append(f"\nEnergi net: D9 {e9:.0f} Wh, baseline {eb:.0f} Wh ({100*(eb-e9)/e9:+.0f}%).")
else:
    out.append("\n## B. Ringkasan run belum lengkap (`arc_baseline_out/`); batch B belum selesai.\n")

# ---------------------------------------------------------------- C held-out split
def best_from_preds(d, half):
    """Akurasi token per run: pilih checkpoint terbaik pada separuh SELEKSI, lapor separuh lain."""
    try:
        import numpy as np
    except ImportError:
        return {}
    res = {}
    for f in sorted(glob.glob(os.path.join(d, "preds_*_step*.npz"))):
        m = re.match(r"preds_(.+)_step(\d+)\.npz$", os.path.basename(f))
        if not m:
            continue
        tag, step = m.group(1), int(m.group(2))
        z = np.load(f)
        acc, seen = z["token_acc"], z["seen"]
        idx = np.where(seen)[0]
        sel, rep = idx[0::2], idx[1::2]          # separuh genap = seleksi, ganjil = pelaporan
        res.setdefault(tag, []).append((step, float(np.nanmean(acc[sel])) * 100,
                                        float(np.nanmean(acc[rep])) * 100))
    chosen = {}
    for tag, v in res.items():
        if len(v) != NCKPT:
            # run yang masih berjalan punya npz kurang; run dengan npz berlebih berarti dua grid
            # step tercampur, dan itu memperbanyak peluang seleksi-max. Keduanya ditolak.
            WARN.append(f"{d}: tag {tag} punya {len(v)} checkpoint, diharapkan {NCKPT}; DIBUANG")
            continue
        v.sort()
        k = max(range(len(v)), key=lambda i: v[i][1])   # pilih pada separuh seleksi
        chosen[tag] = v[k][2] if half == "report" else v[k][1]
    return chosen

c9 = best_from_preds(os.path.join(ROOT, "arc_d9_preds_out", "preds"), "report")
c36 = best_from_preds(os.path.join(ROOT, "arc_d36_accum_out", "preds"), "report")
if c9 and c36:
    a, b = list(c9.values()), list(c36.values())
    out += ["\n## C. ARC-AGI-1: checkpoint dipilih pada separuh subset, dilaporkan pada separuh lain\n",
            "| kontras | grup 1 | grup 2 | selisih | uji | Bonferroni |", "|---|---|---|---|---|---|",
            line("D9 vs D36 (held-out half)", a, b),
            f"\nPer run D9: {', '.join(f'{k}={v:.2f}' for k, v in sorted(c9.items()))}",
            f"\nPer run D36: {', '.join(f'{k}={v:.2f}' for k, v in sorted(c36.items()))}"]
else:
    out.append("\n## C. Belum ada berkas prediksi per-instance; batch C belum selesai.\n")

if WARN:
    out.insert(2, "\n> **PERINGATAN HIGIENIS DATA.** Analisis di bawah hanya sah bila daftar ini kosong.\n"
               + "".join(f">\n> - {w}\n" for w in WARN))
else:
    out.insert(2, "\nHigienis data: tidak ada baris yang dibuang, tidak ada duplikat tag, "
                  "jumlah run tiap batch sesuai pra-registrasi.\n")
txt = "\n".join(out) + "\n"
# keluaran mengikuti data_root, supaya menjalankan skrip atas salinan lain tidak menimpa
# laporan repo (temuan fase BR).
outdir = os.path.join(ROOT, "ablation") if os.path.isdir(os.path.join(ROOT, "ablation")) else ROOT
os.makedirs(outdir, exist_ok=True)
open(os.path.join(outdir, "bm_results.md"), "w").write(txt)
print(txt)
