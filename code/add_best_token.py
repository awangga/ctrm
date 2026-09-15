#!/usr/bin/env python3
"""Tambah kolom best_token_pct (checkpoint terbaik, all/accuracy) ke recipe_summary.csv.

Fase BI (2026-09-15): naskah beralih ke SATU aturan checkpoint untuk semua task, yakni
checkpoint terbaik (Algoritma 1: acc* = max_s acc(s)). Runner lama hanya menulis
final_token_pct; kolom ini diturunkan post hoc dari progress_<tag>.jsonl yang sama.
Idempotent. Usage: python3 add_best_token.py [data_root]
"""
import csv, json, os, sys
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
for d in ("recipe_out", "maze_depth_out", "arc_depth_out", "maze_real_out"):
    sp = os.path.join(ROOT, d, "recipe_summary.csv")
    if not os.path.exists(sp): continue
    rows = list(csv.DictReader(open(sp))); fields = list(rows[0].keys())
    if "best_token_pct" not in fields: fields.insert(fields.index("final_token_pct") + 1, "best_token_pct")
    for r in rows:
        best = 0.0
        for ln in open(os.path.join(ROOT, d, f"progress_{r['tag']}.jsonl")):
            if '"phase": "eval"' not in ln: continue
            try: j = json.loads(ln)
            except Exception: continue
            best = max(best, j.get("all/accuracy", 0) * 100)
        r["best_token_pct"] = round(best, 4)
    with open(sp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"{d}: {len(rows)} run, best_token_pct ditulis")
