#!/bin/bash
# Verify worst drawdown for each winning trade using exact entry_price and exit_time
cd "/Users/smalandrakis/Documents/WindSurf/DS:ML trade model playground/CascadeProjects/windsurf-project"
CSV="logs/btc_price_ticks.csv"

echo "=== WINS: Worst dip during trade (entry_price reference, capped at exit_time) ==="
echo ""

# entry_time|exit_time|dir|entry_price
while IFS='|' read -r et xt dir ep; do
  # Convert entry_time format: 2026-03-09T01:35:11 -> 2026-03-09 01:35:11
  es=$(echo "$et" | sed 's/T/ /')
  xs=$(echo "$xt")
  
  if [ "$dir" = "LONG" ]; then
    result=$(awk -F, -v s="$es" -v e="$xs" -v ep="$ep" '
      $1>=s && $1<=e { d=($2/ep-1)*100; if(d<min) min=d; if(d>max) max=d }
      END { printf "dip=%+.3f%% peak=%+.3f%%", min, max }
    ' "$CSV")
  else
    result=$(awk -F, -v s="$es" -v e="$xs" -v ep="$ep" '
      $1>=s && $1<=e { d=(ep/$2-1)*100; if(d<min) min=d; if(d>max) max=d }
      END { printf "dip=%+.3f%% peak=%+.3f%%", min, max }
    ' "$CSV")
  fi
  
  echo "$et $dir @\$$ep  $result"
done << 'EOF'
2026-03-09T01:35:11.208272|2026-03-09 01:37:04|LONG|66447.94
2026-03-09T03:40:14.760641|2026-03-09 03:50:53|LONG|66454.9
2026-03-09T03:55:06.585614|2026-03-09 04:02:37|LONG|66614.11
2026-03-09T04:05:07.249968|2026-03-09 04:09:53|LONG|66862.61
2026-03-09T04:10:16.633487|2026-03-09 04:10:40|LONG|67134.5
2026-03-09T06:40:18.452386|2026-03-09 06:44:08|LONG|67564.6
2026-03-09T08:45:06.370339|2026-03-09 09:01:55|LONG|67419.22
2026-03-09T08:50:16.050795|2026-03-09 10:05:22|LONG|67493.72
2026-03-09T11:05:04.468310|2026-03-09 11:16:22|SHORT|68037.21
2026-03-09T13:05:06.236044|2026-03-09 13:25:16|LONG|67879.56
EOF

echo ""
echo "=== LOSSES: Worst dip during trade ==="
echo ""

while IFS='|' read -r et xt dir ep; do
  es=$(echo "$et" | sed 's/T/ /')
  xs=$(echo "$xt")
  
  if [ "$dir" = "LONG" ]; then
    result=$(awk -F, -v s="$es" -v e="$xs" -v ep="$ep" '
      $1>=s && $1<=e { d=($2/ep-1)*100; if(d<min) min=d; if(d>max) max=d }
      END { printf "dip=%+.3f%% peak=%+.3f%%", min, max }
    ' "$CSV")
  else
    result=$(awk -F, -v s="$es" -v e="$xs" -v ep="$ep" '
      $1>=s && $1<=e { d=(ep/$2-1)*100; if(d<min) min=d; if(d>max) max=d }
      END { printf "dip=%+.3f%% peak=%+.3f%%", min, max }
    ' "$CSV")
  fi
  
  echo "$et $dir @\$$ep  $result"
done << 'EOF'
2026-03-08T23:35:16.669661|2026-03-08 23:37:25|SHORT|66028.83
2026-03-09T00:55:09.432223|2026-03-09 01:05:37|SHORT|66038.0
2026-03-09T01:40:04.756691|2026-03-09 01:54:24|LONG|66554.66
2026-03-09T02:25:13.861633|2026-03-09 02:56:18|LONG|66503.28
2026-03-09T04:15:10.289802|2026-03-09 04:21:22|LONG|67478.94
2026-03-09T05:30:09.846633|2026-03-09 05:55:39|LONG|67553.74
2026-03-09T09:30:43.388425|2026-03-09 10:07:01|LONG|67782.87
2026-03-09T10:15:13.762373|2026-03-09 10:27:15|SHORT|67797.7
2026-03-09T12:50:15.790274|2026-03-09 12:54:45|SHORT|67603.69
2026-03-09T18:15:47.891379|2026-03-09 18:22:05|LONG|68998.39
EOF
