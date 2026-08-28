#!/usr/bin/env bash
# Kalibrasi TRM (run only) — deps & dataset sudah ada. Ukur throughput + daya + energi.
set -uo pipefail
TRM=/tmp/claude-1000/-home-adb-awangga-trm/63edeaef-bf18-4055-b033-2ef40a499d44/scratchpad/TRM
SCRATCH=/tmp/claude-1000/-home-adb-awangga-trm/63edeaef-bf18-4055-b033-2ef40a499d44/scratchpad
cd "$TRM"; source .venv/bin/activate
export WANDB_MODE=disabled WANDB_SILENT=true HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA="data/sudoku-cal"

run_point () {  # args: hidden L_cycles tag
  local H=$1 LC=$2 TAG=$3
  local PWRLOG="$SCRATCH/power_${TAG}.csv"
  echo "===== POINT $TAG : hidden_size=$H H_cycles=3 L_cycles=$LC (D_eff=$((3*LC))) ====="
  ( nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used,temperature.gpu \
      --format=csv,noheader,nounits -lms 500 > "$PWRLOG" 2>/dev/null ) &
  local PWR_PID=$!
  local START=$(date +%s.%N)
  timeout 900 python pretrain.py \
    arch=trm data_paths="[$DATA]" evaluators="[]" \
    epochs=1 eval_interval=1 \
    lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
    arch.mlp_t=True arch.pos_encodings=none \
    arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=$LC \
    arch.hidden_size=$H global_batch_size=128 \
    +run_name=${TAG} ema=False 2>&1 | tee "$SCRATCH/run_${TAG}.log" | grep -E "num_params|it/s|Epoch|Error|Traceback|OutOfMemory" | tail -8
  local END=$(date +%s.%N)
  kill $PWR_PID 2>/dev/null
  python - "$PWRLOG" "$START" "$END" "$TAG" <<'PY'
import sys,csv
log,start,end,tag=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]),sys.argv[4]
P=[];U=[];M=[]
for r in csv.reader(open(log)):
    try:
        P.append(float(r[1])); U.append(float(r[2])); M.append(float(r[3]))
    except: pass
wall=end-start
if P:
    avgP=sum(P)/len(P); maxP=max(P); energy_J=avgP*wall; energy_Wh=energy_J/3600
    print(f"[{tag}] wall={wall:.1f}s avgGPUpower={avgP:.1f}W maxP={maxP:.0f}W "
          f"avgUtil={sum(U)/len(U):.0f}% maxMem={max(M):.0f}MiB "
          f"energy={energy_Wh:.4f}Wh ({energy_J:.0f}J) samples={len(P)}")
else:
    print(f"[{tag}] wall={wall:.1f}s NO power samples")
PY
}

run_point 256 6 cal_h256_d18
echo "########## DONE ##########"
