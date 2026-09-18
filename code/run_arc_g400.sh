#!/usr/bin/env bash
# Fase BS: kontras kedalaman ARC dijalankan ulang pada subset evaluasi yang SAH.
# Dikunci di prereg_BM.md AMANDEMEN 5: 10 run, D9 lima seed dan D36 lima seed dengan
# akumulasi gradien (micro 24 x accum 2 = batch efektif 48, sama dengan D9).
# Dataset: data/arc1-aug1k-g400 (400 task berbeda), BUKAN arc1-aug1k-e512 (satu task + augmentasi).
set -uo pipefail
R=/home/adb/awangga/trm/eksperimen/frontier
PY=/home/adb/awangga/trm-env/TRM/.venv/bin/python
LOG=$R/arc_g400.log
BASE="TRM_DIR=/home/adb/awangga/trm-env/TRM NEVAL=25 EMA=True"
# run_recipe.py memanggil subprocess "python" polos, jadi venv WAJIB di depan PATH.
# Tanpa baris ini run mati 2 detik dengan ModuleNotFoundError: No module named torch
# (uji-asap fase BS; run_B_retry.sh juga kekurangan ini).
export PATH="/home/adb/awangga/trm-env/TRM/.venv/bin:$PATH"
set -a; . /home/adb/awangga/.env; set +a

tg() { curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
       -d "chat_id=${CHAT_ID}" -d "text=$1" >/dev/null || true; }

apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$apps" -ne 0 ]; then
  echo "TOLAK: $apps proses compute di GPU (aturan #8). Tidak start." | tee -a "$LOG"; exit 1
fi

run_one() {  # $1=label $2=OUT_DIR sisanya env
  local label="$1" out="$2"; shift 2
  echo "--- $label $(date -Is) ---" >> "$LOG"
  env $BASE OUT_DIR="$out" SAVE_PREDS="$out/preds" "$@" "$PY" "$R/run_recipe.py" >> "$LOG" 2>&1
  local s="$out/recipe_summary.csv"
  [ -f "$s" ] && tg "[trm BS] $label: $(tail -1 "$s" | cut -c1-150)" || tg "[trm BS] $label selesai tanpa summary, cek log"
}

echo "=== ARC g400 start $(date -Is) ===" >> "$LOG"
tg "[trm BS] Mulai 10 run ARC di subset SAH (arc1-aug1k-g400, 400 task berbeda). D9 x5 lalu D36 x5."

O=$R/arc_d9_g400_out;  mkdir -p "$O/preds"
for s in 0 1 2 3 4; do
  run_one "D9 g400 s$s" "$O" DATA=data/arc1-aug1k-g400 GROUPS=3080 HIDDEN=256 DEPTH=9 \
          BATCH=48 ACCUM=1 STEPS=24000 SEED=$s
done
tg "[trm BS] D9 selesai (5 run). Lanjut D36 dengan akumulasi."

O=$R/arc_d36_g400_out; mkdir -p "$O/preds"
for s in 0 1 2 3 4; do
  run_one "D36 g400 s$s" "$O" DATA=data/arc1-aug1k-g400 GROUPS=3080 HIDDEN=256 DEPTH=36 \
          BATCH=24 ACCUM=2 STEPS=6000 SEED=$s
done

echo "ARC_G400_DONE $(date -Is)" >> "$LOG"
tg "[trm BS] SELESAI 10 run ARC di subset sah. Siap dianalisis."
