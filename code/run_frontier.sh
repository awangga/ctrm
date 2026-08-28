#!/usr/bin/env bash
# Frontier runner — depth sweep at fixed width, with PER-PROGRESS HISTORY logging.
# Records: train-loss/step + eval acc/loss per checkpoint (progress_<tag>.jsonl) + power time-series
# (pw_<tag>.csv) + summary row (frontier_pilot.csv). Env: EPOCHS (default 1000), NEVAL (default 5).
set -uo pipefail
TRM=/tmp/claude-1000/-home-adb-awangga-trm/63edeaef-bf18-4055-b033-2ef40a499d44/scratchpad/TRM
SCR=/tmp/claude-1000/-home-adb-awangga-trm/63edeaef-bf18-4055-b033-2ef40a499d44/scratchpad
OUT="$SCR/frontier_out"; mkdir -p "$OUT"
cd "$TRM"; source .venv/bin/activate
export WANDB_MODE=disabled WANDB_SILENT=true HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA="data/sudoku-pilot"; EPOCHS=${EPOCHS:-1000}; NEVAL=${NEVAL:-5}
EVAL_INT=$(( EPOCHS / NEVAL ))
CSV="$OUT/frontier_summary.csv"
echo "hidden,H_cycles,L_cycles,D_eff,epochs,eval_interval,wall_s,avg_gpu_W,idle_W,net_energy_Wh,final_eval_json,progress_jsonl,power_csv" > "$CSV"

idle=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -lms 500 2>/dev/null | head -8 | awk '{s+=$1;n++} END{printf "%.1f",(n?s/n:0)}')
echo "idle_W=$idle EPOCHS=$EPOCHS EVAL_INT=$EVAL_INT"

run () {  # hidden H L tag
  local H=$1 HC=$2 LC=$3 TAG=$4
  local PJ="$OUT/progress_$TAG.jsonl" PW="$OUT/pw_$TAG.csv" LOG="$OUT/log_$TAG.txt"
  : > "$PJ"
  echo "===== $TAG hidden=$H D_eff=$((HC*LC)) ====="
  ( nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 1000 > "$PW" 2>/dev/null ) & local PP=$!
  local T0=$(date +%s.%N)
  TRM_PROGRESS_LOG="$PJ" timeout 7200 python pretrain.py arch=trm data_paths="[$DATA]" evaluators="[]" \
    epochs=$EPOCHS eval_interval=$EVAL_INT \
    lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
    arch.mlp_t=True arch.pos_encodings=none arch.L_layers=2 \
    arch.H_cycles=$HC arch.L_cycles=$LC arch.hidden_size=$H global_batch_size=128 \
    +run_name=$TAG ema=False 2>&1 | tee "$LOG" | grep -E "EVAL_METRICS_JSON|Epoch|Error executing|Traceback|OutOfMemory" | tail -4
  local T1=$(date +%s.%N); kill $PP 2>/dev/null
  local EM=$(grep "EVAL_METRICS_JSON" "$LOG" | tail -1); EM=${EM#*\{}; EM="{${EM}"
  python - "$PW" "$T0" "$T1" "$idle" "$H" "$HC" "$LC" "$EPOCHS" "$EVAL_INT" "$EM" "$CSV" "$PJ" <<'PY'
import sys,csv,statistics,json
pw,t0,t1,idle,H,HC,LC,EP,EI,EM,CSVP,PJ=sys.argv[1:13]
P=[]
for ln in open(pw):
    parts=ln.split(",")
    if len(parts)>=2:
        try: P.append(float(parts[1]))
        except: pass
wall=float(t1)-float(t0); avg=statistics.mean(P) if P else float('nan'); idle=float(idle)
net=max(0.0,(avg-idle))*wall/3600 if P else float('nan')
try: json.loads(EM)
except: EM="{}"
import os
csv.writer(open(CSVP,"a",newline="")).writerow([H,HC,LC,int(HC)*int(LC),EP,EI,f"{wall:.1f}",f"{avg:.1f}",f"{idle:.1f}",f"{net:.4f}",EM,os.path.basename(PJ),os.path.basename(pw)])
nlines=sum(1 for _ in open(PJ)) if os.path.exists(PJ) else 0
print(f"[{H}/D{int(HC)*int(LC)}] wall={wall:.0f}s avg={avg:.0f}W net={net:.3f}Wh progress_records={nlines}")
PY
}

run 256 3 3  d09
run 256 3 6  d18
run 256 3 12 d36
echo "########## FRONTIER RUN DONE ##########"; cat "$CSV"
