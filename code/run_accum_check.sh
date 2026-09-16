#!/usr/bin/env bash
# Fase BM: uji kesetaraan gradient accumulation. ARC D9 pendek, dua konfigurasi:
#   (a) batch 48, accum 1  (jalur lama)
#   (b) batch 24, accum 2  (jalur baru, batch efektif 48)
# Kurva train-loss keduanya harus berimpit bila patch benar.
set -u
R=/home/adb/awangga/trm/eksperimen/frontier
export PATH="/home/adb/awangga/trm-env/TRM/.venv/bin:$PATH"
PY=/home/adb/awangga/trm-env/TRM/.venv/bin/python
export TRM_DIR=/home/adb/awangga/trm-env/TRM
# GROUPS is a bash special variable: `export GROUPS=...` never reaches the child process.
# Pass every variable with `env` instead (the same trap is documented in PROTOCOL.md).
COMMON="TRM_DIR=$TRM_DIR OUT_DIR=$R/accum_check_out DATA=data/arc1-aug1k-e512 GROUPS=3080 \
HIDDEN=256 DEPTH=9 STEPS=500 NEVAL=5 EMA=True SEED=0"

env $COMMON BATCH=48 ACCUM=1 "$PY" $R/run_recipe.py >> $R/accum_check_out/chain.log 2>&1
env $COMMON BATCH=24 ACCUM=2 "$PY" $R/run_recipe.py >> $R/accum_check_out/chain.log 2>&1
echo "ACCUM_CHECK_DONE" >> $R/accum_check_out/chain.log
