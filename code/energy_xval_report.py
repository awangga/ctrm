#!/usr/bin/env python3
"""Report cross-validasi energi: CodeCarbon (gpu_energy) vs nvidia-smi (integral daya).

Dua pembukuan energi atas window training FISIK yang sama (run xval seed 3-4).
Aturan integritas #3: target >95% setuju. Output: energy_xval_report.md + ringkas stdout.
"""
import csv, os, statistics

BASE = "/home/adb/awangga/trm/eksperimen/frontier"
XV = os.path.join(BASE, "budget_out", "energy_xval.csv")
REPORT = os.path.join(BASE, "ablation", "energy_xval_report.md")

rows = list(csv.DictReader(open(XV)))


def f(r, k):
    try: return float(r[k])
    except (KeyError, ValueError, TypeError): return None


agrees = [f(r, "gross_agree_pct") for r in rows if f(r, "gross_agree_pct") is not None]
# selisih relatif absolut per run (CodeCarbon vs nvidia-smi, basis gross)
reldiff = []
for r in rows:
    smi, cc = f(r, "smi_gross_Wh"), f(r, "cc_gpu_Wh")
    if smi and cc:
        reldiff.append(100 * abs(smi - cc) / ((smi + cc) / 2))

mean_agree = statistics.mean(agrees)
min_agree = min(agrees)
max_reldiff = max(reldiff)
mean_reldiff = statistics.mean(reldiff)
n_pass = sum(1 for a in agrees if a >= 95.0)

L = []
L.append("# Cross-validasi energi: CodeCarbon vs nvidia-smi")
L.append("")
L.append(f"Sumber: `budget_out/energy_xval.csv` ({len(rows)} run, hidden=128, D=9/18/36, seed 3-4). "
         "Kedua metode mengukur energi GPU pada window training fisik yang sama "
         "(CodeCarbon `gpu_energy` via pynvml; nvidia-smi `power.draw` 1 Hz diintegral trapezoid). "
         "Pembanding: energi GROSS (total draw), bukan net, karena keduanya mengukur draw total.")
L.append("")
L.append("## Per-run")
L.append("")
L.append("| run | D_eff | seed | wall s | nvidia-smi gross (Wh) | CodeCarbon GPU (Wh) | setuju % |")
L.append("|---|---|---|---|---|---|---|")
for r in rows:
    L.append(f"| {r['tag']} | {r['D_eff']} | {r['seed']} | {f(r,'wall_s'):.0f} | "
             f"{f(r,'smi_gross_Wh'):.3f} | {f(r,'cc_gpu_Wh'):.3f} | {f(r,'gross_agree_pct'):.2f} |")
L.append("")
L.append("## Ringkas")
L.append("")
L.append(f"- Rerata kesepakatan: **{mean_agree:.2f}%** (min {min_agree:.2f}%).")
L.append(f"- Selisih relatif absolut: rata-rata **{mean_reldiff:.2f}%**, maksimum **{max_reldiff:.2f}%**.")
L.append(f"- Run lolos ambang >95%: **{n_pass}/{len(rows)}**.")
L.append("")
verdict = (f"**LOLOS** target integritas #3 (>95% setuju): {n_pass}/{len(rows)} run, rerata {mean_agree:.2f}%. "
           "Pengukuran energi tervalidasi silang; metrik Joule/Wh-untuk-target dapat dipercaya."
           if min_agree >= 95.0 else
           f"Sebagian run di bawah 95% (min {min_agree:.2f}%); periksa kembali.")
L.append("## Vonis")
L.append("")
L.append(verdict)
L.append("")
L.append("Catatan: CodeCarbon `gpu_energy` konsisten sedikit LEBIH RENDAH (~1%) dari integral nvidia-smi; "
         "selisih sistematis kecil ini wajar (beda interval sampling: CodeCarbon 5 s vs nvidia-smi 1 s, "
         "dan penanganan tepi window). Keduanya membaca sensor pynvml yang sama, jadi ini validasi "
         "konsistensi pembukuan, bukan dua sensor fisik independen.")
L.append("")
open(REPORT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("\nwrote", REPORT)
print(f"NOTIF mean_agree={mean_agree:.2f}% min={min_agree:.2f}% pass={n_pass}/{len(rows)}")
