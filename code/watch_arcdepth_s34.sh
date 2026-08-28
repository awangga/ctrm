#!/usr/bin/env bash
# Pemantau rantai ARC s3,s4: kirim notif Telegram tiap run baru selesai + saat rampung.
set -uo pipefail
set -a; . /home/adb/awangga/.env; set +a
CSV=/home/adb/awangga/trm/eksperimen/frontier/arc_depth_out/recipe_summary.csv
LOG=/home/adb/awangga/trm-env/arcdepth_s34.log
tg(){ curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${CHAT_ID}" --data-urlencode "text=$1" -o /dev/null; }
prev=$(wc -l < "$CSV")
tg "[trm] ARC s3,s4 dimulai. Target 6 run (~11 jam, ~1,7 kWh). Baris summary sekarang: $((prev-1)). Pre-commit n=5: hasil dilaporkan apa pun bunyinya."
while true; do
  sleep 120
  [ -f "$CSV" ] || continue
  cur=$(wc -l < "$CSV")
  if [ "$cur" -gt "$prev" ]; then
    row=$(tail -1 "$CSV")
    tag=$(echo "$row" | cut -d, -f1); acc=$(echo "$row" | cut -d, -f10)
    wh=$(echo "$row" | cut -d, -f11); agr=$(echo "$row" | cut -d, -f13)
    tg "[trm] ARC run selesai: ${tag} | token ${acc}% | ${wh} Wh | xval ${agr}% | total $((cur-1))/15 baris"
    prev=$cur
  fi
  if ! pgrep -f "run_arcdepth_s34.sh" >/dev/null 2>&1; then
    if grep -q "CHAIN DONE" "$LOG" 2>/dev/null; then
      tg "[trm] ARC s3,s4 RAMPUNG. Saya hitung ulang signifikansinya sekarang."
    else
      tg "[trm] ARC s3,s4 BERHENTI tanpa 'CHAIN DONE' -- kemungkinan gagal. Cek $LOG"
    fi
    exit 0
  fi
done
