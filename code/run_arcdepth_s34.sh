#!/usr/bin/env bash
# ARC-AGI-1 depth n=3 -> n=5 (seed 3,4). Konfigurasi IDENTIK seed 0-2 (run_arcdepth_seeds.sh):
# D9 b48 24k, D18 b48 12k, D36 b24 12k, HIDDEN=256, GROUPS=3080, NEVAL=25, EMA, e512.
#
# Alasan (pra-registrasi, lihat EXPERIMENT_LOG fase AW): kontras D9-vs-D36 punya
# Cohen d=2.93 dan p=0.063 di n=3. Ia gagal signifikan BUKAN karena efeknya kecil,
# melainkan karena varian dua grup timpang (3.23 vs 0.61) sehingga df Welch anjlok
# ke 2.14. Analisis daya: n=3 sudah cukup; proyeksi n=5 -> p~0.008.
#
# KOMITMEN: jalankan keenam run, laporkan apa pun hasilnya, termasuk bila tetap
# null atau berbalik arah. Berhenti begitu p<0.05 = p-hacking dan dilarang.
set -uo pipefail
export PATH="/home/adb/awangga/trm-env/TRM/.venv/bin:$PATH"
export VIRTUAL_ENV="/home/adb/awangga/trm-env/TRM/.venv"
PY="/home/adb/awangga/trm-env/TRM/.venv/bin/python"
OUT=/home/adb/awangga/trm/eksperimen/frontier/arc_depth_out
cd /home/adb/awangga/trm

# ---- PREFLIGHT: tolak jalan bila instrumentasi energi mati (pelajaran fase AF) ----
if ! nvidia-smi --query-gpu=power.draw --format=csv,noheader >/dev/null 2>&1; then
  echo "PREFLIGHT GAGAL: nvidia-smi tidak bisa membaca power.draw."
  nvidia-smi 2>&1 | head -2
  exit 1
fi
if ! "$PY" -c "import pynvml; pynvml.nvmlInit()" >/dev/null 2>&1; then
  echo "PREFLIGHT GAGAL: pynvml/NVML tak bisa diinisialisasi; CodeCarbon tak akan mengukur GPU."
  exit 1
fi
echo "preflight OK: $(nvidia-smi --query-gpu=power.draw --format=csv,noheader | head -1)"

COMMON="DATA=data/arc1-aug1k-e512 TRM_DIR=/home/adb/awangga/trm-env/TRM OUT_DIR=$OUT HIDDEN=256 GROUPS=3080 NEVAL=25 EMA=True"
echo "=== ARCDEPTH s3,s4 CHAIN START $(date) ==="
for S in 3 4; do
  echo "--- SEED $S : D9 (b48,24000) ---"
  env $COMMON SEED=$S DEPTH=9  BATCH=48 STEPS=24000 "$PY" eksperimen/frontier/run_recipe.py
  echo "--- SEED $S : D18 (b48,12000) ---"
  env $COMMON SEED=$S DEPTH=18 BATCH=48 STEPS=12000 "$PY" eksperimen/frontier/run_recipe.py
  echo "--- SEED $S : D36 (b24,12000) ---"
  env $COMMON SEED=$S DEPTH=36 BATCH=24 STEPS=12000 "$PY" eksperimen/frontier/run_recipe.py
done
echo "=== ARCDEPTH s3,s4 CHAIN DONE $(date) ==="
