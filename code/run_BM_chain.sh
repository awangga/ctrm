#!/usr/bin/env bash
# Fase BM: rantai 18 run yang dipra-registrasi di prereg_BM.md (+ amandemen 1).
# Empat batch, berurutan pada satu GPU. Notifikasi Telegram tiap run selesai,
# commit + push artefak tiap batch selesai (aturan CLAUDE.md: jangan tunggu akhir).
#
#   A1  ARC-AGI-1  D36  micro 24 x accum 2 (eff 48), EPOCHS=75 (sama dgn D36 lama), seed 0-4
#   A2  Sudoku     D36  micro 96 x accum 2 (eff 192), EPOCHS=2400 (sama dgn D36 lama), seed 0-2
#   B   ARC-AGI-1  baseline non-rekursif, EPOCHS=350 (sama dgn D9), seed 0-4
#   C   ARC-AGI-1  D9 ulang dengan logging prediksi per-instance, seed 0-4
#
# PENTING: variabel diteruskan lewat `env VAR=...`, BUKAN `export`. `GROUPS` adalah
# variabel khusus bash yang tidak ikut ter-export, sehingga `export GROUPS=3080` diam-diam
# membuat runner memakai default 1000 dan epoch jadi salah. Runner ARC lama juga memakai
# bentuk `env` ini.
set -u
R=/home/adb/awangga/trm/eksperimen/frontier
REPO=/home/adb/awangga/trm
export PATH="/home/adb/awangga/trm-env/TRM/.venv/bin:$PATH"
PY=/home/adb/awangga/trm-env/TRM/.venv/bin/python
LOG=$R/bm_chain.log
set -a; . /home/adb/awangga/.env; set +a

tg() {  # notifikasi ringkas; token tak pernah dicetak
  curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
       -d chat_id="${CHAT_ID}" --data-urlencode text="$1" > /dev/null 2>&1 || true
}

last_row() {
  "$PY" - "$1" <<'PYEOF'
import csv, sys
r = list(csv.DictReader(open(sys.argv[1])))[-1]
print(f"{r['tag']} exact={r['best_exact_pct']}% token={r.get('best_token_pct','-')}% "
      f"net={r['smi_net_Wh']}Wh agree={r['gross_agree_pct']}% wall={float(r['wall_s'])/60:.0f}m")
PYEOF
}

commit_push() {
  cd "$REPO" || return 0
  git add -A >/dev/null 2>&1
  git commit -q -m "fase BM: artefak $1" >/dev/null 2>&1 || return 0
  git push -q "https://${GHPAT}@github.com/bukped/trm.git" main >/dev/null 2>&1 || true
}

BASE="TRM_DIR=/home/adb/awangga/trm-env/TRM NEVAL=25 EMA=True"

run_one() {  # $1=label, $2=OUT_DIR, sisanya env assignment
  local label="$1"; local out="$2"; shift 2
  env $BASE OUT_DIR="$out" SAVE_PREDS="$out/preds" "$@" "$PY" "$R/run_recipe.py" >> "$LOG" 2>&1
  local s="$out/recipe_summary.csv"
  if [ -f "$s" ]; then tg "[trm BM] $label: $(last_row "$s")"; else tg "[trm BM] $label selesai tanpa summary, cek log"; fi
}

echo "=== BM chain start $(date -Is)" >> "$LOG"
tg "[trm BM] rantai 18 run DIMULAI: A1 ARC D36 accum (5), A2 Sudoku D36 accum (3), B baseline ARC (5), C ARC D9 preds (5). Perkiraan ~35 jam."

# ---------------------------------------------------------------- A1
O=$R/arc_d36_accum_out; mkdir -p "$O/preds"
for s in 0 1 2 3 4; do
  run_one "A1 ARC D36 accum s$s" "$O" DATA=data/arc1-aug1k-e512 GROUPS=3080 \
          HIDDEN=256 DEPTH=36 BATCH=24 ACCUM=2 STEPS=6000 SEED=$s
done
commit_push "A1 ARC D36 accum"
tg "[trm BM] BATCH A1 SELESAI (5 run ARC D36, batch efektif 48)."

# ---------------------------------------------------------------- A2
O=$R/sudoku_d36_accum_out; mkdir -p "$O/preds"
for s in 0 1 2; do
  run_one "A2 Sudoku D36 accum s$s" "$O" DATA=data/sudoku-aug1k GROUPS=1000 \
          HIDDEN=512 DEPTH=36 BATCH=96 ACCUM=2 STEPS=12500 SEED=$s
done
commit_push "A2 Sudoku D36 accum"
tg "[trm BM] BATCH A2 SELESAI (3 run Sudoku D36, batch efektif 192)."

# ---------------------------------------------------------------- B
O=$R/arc_baseline_out; mkdir -p "$O/preds"
for s in 0 1 2 3 4; do
  run_one "B ARC baseline s$s" "$O" DATA=data/arc1-aug1k-e512 GROUPS=3080 \
          ARCH=transformers_baseline HIDDEN=256 DEPTH=8 BATCH=48 ACCUM=1 STEPS=24000 SEED=$s
done
commit_push "B baseline non-rekursif ARC"
tg "[trm BM] BATCH B SELESAI (5 run baseline non-rekursif ARC)."

# ---------------------------------------------------------------- C
O=$R/arc_d9_preds_out; mkdir -p "$O/preds"
for s in 0 1 2 3 4; do
  run_one "C ARC D9 preds s$s" "$O" DATA=data/arc1-aug1k-e512 GROUPS=3080 \
          HIDDEN=256 DEPTH=9 BATCH=48 ACCUM=1 STEPS=24000 SEED=$s
done
commit_push "C ARC D9 dgn prediksi per-instance"

echo "BM_CHAIN_DONE $(date -Is)" >> "$LOG"
tg "[trm BM] SELURUH 18 RUN SELESAI. Analisis dijalankan sekali sesuai pra-registrasi."
