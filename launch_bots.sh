#!/bin/bash
# Launch all 3 BTC bots sequentially after confirming IB Gateway is available.
# Waits patiently (2 min between checks) to avoid hammering IB.

cd "$(dirname "$0")"
LOG="logs/launcher.log"
mkdir -p logs

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG"; }

# Step 1: Wait for IB Gateway to accept connections
log "Waiting for IB Gateway on port 4002..."
while true; do
    if python3 -c "
from ib_insync import IB, util
util.patchAsyncio()
ib = IB()
ib.connect('127.0.0.1', 4002, clientId=999, timeout=10)
ib.disconnect()
print('OK')
" 2>/dev/null | grep -q "OK"; then
        log "IB Gateway is available"
        break
    fi
    log "IB Gateway not ready — waiting 120s..."
    sleep 120
done

# Step 2: Kill any leftover python bot processes
log "Cleaning up any leftover bot processes..."
pkill -9 -f "python3.*btc_ensemble_bot" 2>/dev/null
pkill -9 -f "python3.*btc_tick_bot" 2>/dev/null
pkill -9 -f "python3.*btc_trend_bot" 2>/dev/null
sleep 5

# Step 3: Start ensemble bot
log "Starting btc_ensemble_bot.py (clientId=400)..."
nohup caffeinate -disu python3 btc_ensemble_bot.py >> logs/btc_ensemble_bot.log 2>&1 &
ENSEMBLE_PID=$!
log "Ensemble PID: $ENSEMBLE_PID"
sleep 10

# Step 4: Verify ensemble connected
if python3 -c "
from ib_insync import IB, util
util.patchAsyncio()
ib = IB()
ib.connect('127.0.0.1', 4002, clientId=998, timeout=10)
ib.disconnect()
print('OK')
" 2>/dev/null | grep -q "OK"; then
    log "IB still accepting connections after ensemble start — good"
else
    log "WARNING: IB not accepting connections after ensemble start"
fi

# Step 5: Start tick bot
log "Starting btc_tick_bot.py (clientId=401)..."
nohup caffeinate -disu python3 btc_tick_bot.py >> logs/btc_tick_bot.log 2>&1 &
TICK_PID=$!
log "Tick PID: $TICK_PID"
sleep 10

# Step 6: Start trend bot
log "Starting btc_trend_bot.py (clientId=402)..."
nohup caffeinate -disu python3 btc_trend_bot.py >> logs/btc_trend_bot.log 2>&1 &
TREND_PID=$!
log "Trend PID: $TREND_PID"
sleep 15

# Step 7: Final status
log "=== FINAL STATUS ==="
log "Ensemble PID: $ENSEMBLE_PID"
log "Tick PID: $TICK_PID"
log "Trend PID: $TREND_PID"

log "--- Ensemble last 3 lines ---"
tail -3 logs/btc_ensemble_bot.log >> "$LOG" 2>&1
tail -3 logs/btc_ensemble_bot.log

log "--- Tick last 3 lines ---"
tail -3 logs/btc_tick_bot.log >> "$LOG" 2>&1
tail -3 logs/btc_tick_bot.log

log "--- Trend last 3 lines ---"
tail -3 logs/btc_trend_bot.log >> "$LOG" 2>&1
tail -3 logs/btc_trend_bot.log

log "Active caffeinate processes:"
ps aux | grep "caffeinate.*python3" | grep -v grep | awk '{print $2, $11, $12, $13}' | tee -a "$LOG"

log "Launcher complete."
