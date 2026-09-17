#!/usr/bin/env python3
"""Tambah kolom best_token_pct (checkpoint terbaik, all/accuracy) ke recipe_summary.csv.

Fase BI (2026-09-15): naskah beralih ke SATU aturan checkpoint untuk semua task, yakni
checkpoint terbaik (Algoritma 1: acc* = max_s acc(s)). Runner lama hanya menulis
final_token_pct; kolom ini diturunkan post hoc dari progress_<tag>.jsonl yang sama.
Idempotent. Usage: python3 add_best_token.py [data_root]

Folder yang diproses ditulis TETAP di bawah. Audit fase BR menemukan bahwa folder fase BM
tertinggal DIAM-DIAM karena daftar ini tidak diperbarui, dan `continue` pada berkas yang tidak
ada membuatnya tanpa jejak. Sekarang folder yang tidak ditemukan dilaporkan, dan run tanpa
baris eval dilaporkan pula alih-alih diam-diam menjadi titik data "akurasi 0".

Runner baru (`run_recipe.py`) sudah menulis `best_token_pct` sendiri dengan definisi yang
IDENTIK (max `all/accuracy` x100 atas baris eval, dibulatkan 4 desimal; diverifikasi pada 13 run
fase BM, nol selisih), jadi menjalankan skrip ini atas folder baru bersifat idempoten.
"""
import csv, json, os, sys
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
DIRS = ("recipe_out", "maze_depth_out", "arc_depth_out", "maze_real_out",
        # fase BM: WAJIB ditambahkan tiap ada batch baru, daftarnya tetap dan tanpa glob.
        # arc_baseline_FAILED_rope sengaja TIDAK di sini (5 run mati, nol data).
        "arc_d36_accum_out", "sudoku_d36_accum_out", "arc_d9_preds_out", "arc_baseline_out")
missing, noeval = [], []
for d in DIRS:
    sp = os.path.join(ROOT, d, "recipe_summary.csv")
    if not os.path.exists(sp):
        missing.append(d)
        continue
    rows = list(csv.DictReader(open(sp))); fields = list(rows[0].keys())
    if "best_token_pct" not in fields: fields.insert(fields.index("final_token_pct") + 1, "best_token_pct")
    for r in rows:
        best, nev = 0.0, 0
        for ln in open(os.path.join(ROOT, d, f"progress_{r['tag']}.jsonl")):
            if '"phase": "eval"' not in ln: continue
            try: j = json.loads(ln)
            except Exception: continue
            best = max(best, j.get("all/accuracy", 0) * 100)
            nev += 1
        if nev == 0:
            # tanpa baris eval, best=0.0 akan menjadi datapoint "akurasi 0" yang senyap
            noeval.append(f"{d}/{r['tag']}")
        r["best_token_pct"] = round(best, 4)
    with open(sp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"{d}: {len(rows)} run, best_token_pct ditulis")

if missing:
    print("CATATAN, folder tidak ditemukan (belum dijalankan?): " + ", ".join(missing))
if noeval:
    import sys as _s
    print("!! RUN TANPA BARIS EVAL, best_token_pct=0 dan itu BUKAN hasil: " + ", ".join(noeval),
          file=_s.stderr)
    _s.exit(2)
