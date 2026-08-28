#!/usr/bin/env bash
# Rebuild environment TRM di lokasi DURABLE (di luar /tmp) + patch + dataset.
# Penyebab insiden fase L: scratchpad /tmp terhapus saat sesi berganti. Solusi: ENV_DIR durable.
# Idempotent: aman dijalankan ulang (skip langkah yang sudah selesai).
set -uo pipefail
REPO=/home/adb/awangga/trm
ENV_DIR=/home/adb/awangga/trm-env
TRM="$ENV_DIR/TRM"
VENV="$TRM/.venv"
PY="$VENV/bin/python"
log(){ echo "[rebuild $(date +%H:%M:%S)] $*"; }

mkdir -p "$ENV_DIR"

# 1) sumber TRM: pakai salinan ter-vendor di paket (offline, versi terpaku).
#    VENDOR menunjuk zenodo/vendor di repo kerja; di dalam arsip Zenodo, jalankan skrip ini
#    dari akar paket sehingga VENDOR=./vendor. Tidak ada clone jaringan.
VENDOR="${VENDOR:-$REPO/zenodo/vendor}"
if [ ! -f "$TRM/pretrain.py" ]; then
  if [ -d "$VENDOR/TinyRecursiveModels" ]; then
    log "salin TRM dari vendor ($VENDOR/TinyRecursiveModels, commit c011037)..."
    mkdir -p "$TRM"
    cp -a "$VENDOR/TinyRecursiveModels/." "$TRM/"
  else
    log "vendor tak ditemukan di $VENDOR/TinyRecursiveModels; fallback clone upstream"
    git clone --depth 1 https://github.com/SamsungSAILMontreal/TinyRecursiveModels "$TRM" \
      || { log "clone GAGAL"; exit 1; }
  fi
else
  log "TRM sudah ada, skip"
fi

# 1b) data mentah ARC-AGI-1 dari vendor (tanpa clone fchollet/ARC-AGI)
if [ -d "$VENDOR/arc-agi-1-raw" ] && [ ! -f "$ENV_DIR/arc_raw/arc-agi_training_challenges.json" ]; then
  log "salin data mentah ARC-AGI-1 dari vendor..."
  mkdir -p "$ENV_DIR/arc_raw"
  cp "$VENDOR/arc-agi-1-raw"/*.json "$ENV_DIR/arc_raw/"
fi

# 2) venv + deps
if [ ! -x "$PY" ]; then
  log "buat venv (uv)..."
  uv venv --python 3.12 "$VENV" || { log "venv GAGAL"; exit 1; }
fi
log "install torch cu128 + deps (bisa lama, unduh GB)..."
VIRTUAL_ENV="$VENV" uv pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3
VIRTUAL_ENV="$VENV" uv pip install --python "$PY" numpy einops tqdm coolname pydantic argdantic omegaconf hydra-core huggingface_hub "codecarbon>=2.3" wandb 2>&1 | tail -3
log "verifikasi torch+cuda..."
"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" || { log "torch GAGAL"; exit 1; }

# 3) patch: progress logging + AdamATan2 shim
log "patch pretrain.py (progress logging)..."
# patch resmi ada di vendor/patches; skrip python di bawah idempotent & jadi fallback
if [ -f "$VENDOR/patches/0001-emit-per-step-progress-and-eval-metrics.patch" ] \
   && ! grep -q TRM_PROGRESS_LOG "$TRM/pretrain.py" 2>/dev/null; then
  ( cd "$TRM" && git apply "$VENDOR/patches/0001-emit-per-step-progress-and-eval-metrics.patch" 2>/dev/null \
      || patch -p1 --forward < "$VENDOR/patches/0001-emit-per-step-progress-and-eval-metrics.patch" ) \
    && log "patch vendor diterapkan" || log "patch vendor gagal, pakai skrip fallback"
fi
"$PY" "$REPO/zenodo/code/patch_pretrain_print_metrics.py" "$TRM/pretrain.py" || true
# shim: coba install adam-atan2 asli; jika gagal, taruh shim sbg module adam_atan2.py
if ! "$PY" -c "import adam_atan2" 2>/dev/null; then
  log "adam-atan2 tak ada → pasang shim AdamW"
  cp "$VENDOR/patches/adam_atan2.py" "$TRM/adam_atan2.py" 2>/dev/null \
    || cp "$REPO/zenodo/code/adam_atan2_fallback.py" "$TRM/adam_atan2.py"
  "$PY" -c "import sys; sys.path.insert(0,'$TRM'); import adam_atan2; print('shim ok:', adam_atan2.AdamATan2)" || log "shim WARN"
fi

# 4) dataset sudoku-aug1k (subsample 1000 x aug 1000) + test terbatas 512
cd "$TRM"
if [ ! -f "$TRM/data/sudoku-aug1k/train/dataset.json" ]; then
  log "build sudoku-aug1k (train ~1M contoh)..."
  "$PY" dataset/build_sudoku_dataset.py --output-dir data/sudoku-aug1k --subsample-size 1000 --num-aug 1000 2>&1 | tail -2
fi
# truncate test ke 512 (eval terbatas, sesuai protokol biaya-eval)
log "buat test terbatas 512..."
"$PY" - <<'PY'
import numpy as np, json, os, shutil
d="data/sudoku-aug1k/test"; N=512
meta=json.load(open(f"{d}/dataset.json"))
gi=np.load(f"{d}/all__group_indices.npy"); pi=np.load(f"{d}/all__puzzle_indices.npy")
inp=np.load(f"{d}/all__inputs.npy"); lab=np.load(f"{d}/all__labels.npy")
ids=np.load(f"{d}/all__puzzle_identifiers.npy")
if len(gi)-1 > N:
    # ambil N grup pertama (tiap grup 1 contoh pada test, num_aug=0)
    k=N
    gi2=gi[:k+1]; pend=gi2[-1]
    pi2=pi[:pend+1]; inp2=inp[:pend]; lab2=lab[:pend]; ids2=ids[:pend]
    np.save(f"{d}/all__group_indices.npy", gi2); np.save(f"{d}/all__puzzle_indices.npy", pi2)
    np.save(f"{d}/all__inputs.npy", inp2); np.save(f"{d}/all__labels.npy", lab2)
    np.save(f"{d}/all__puzzle_identifiers.npy", ids2)
    meta["total_groups"]=k; meta["total_puzzles"]=k
    json.dump(meta, open(f"{d}/dataset.json","w"))
    print(f"test truncated -> {k} grup")
else:
    print("test sudah <=512, skip")
PY

log "REBUILD DONE. ENV_DIR=$ENV_DIR  PY=$PY"
