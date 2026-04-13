#!/bin/bash
# Bot Health Check and Auto-Restart Script
# Checks if bots are running and restarts them if needed

BOT_DIR="/Users/smalandrakis/Documents/WindSurf/DS:ML trade model playground/CascadeProjects/windsurf-project"
LOG_FILE="${BOT_DIR}/logs/health_check.log"

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] $1" >> "$LOG_FILE"
}

# Create logs directory if it doesn't exist
mkdir -p "${BOT_DIR}/logs"

log "=== Health Check Started ==="

# Check Swing Bot (btc_ib_gateway_bot.py)
SWING_PID=$(ps aux | grep "btc_ib_gateway_bot.py" | grep -v grep | grep -v "_hf" | awk '{print $2}')

if [ -z "$SWING_PID" ]; then
    log "⚠️  Swing Bot not running - restarting..."
    cd "$BOT_DIR"
    nohup python3 btc_ib_gateway_bot.py > /dev/null 2>&1 &
    NEW_PID=$!
    log "✅ Swing Bot restarted with PID: $NEW_PID"
else
    log "✓ Swing Bot running (PID: $SWING_PID)"
fi

# Check HF Bot (btc_ib_gateway_bot_hf.py)
HF_PID=$(ps aux | grep "btc_ib_gateway_bot_hf.py" | grep -v grep | awk '{print $2}')

if [ -z "$HF_PID" ]; then
    log "⚠️  HF Bot not running - restarting..."
    cd "$BOT_DIR"
    nohup python3 btc_ib_gateway_bot_hf.py > /dev/null 2>&1 &
    NEW_PID=$!
    log "✅ HF Bot restarted with PID: $NEW_PID"
else
    log "✓ HF Bot running (PID: $HF_PID)"
fi

# Check if bot log files are being updated (indicates connectivity)
SWING_LOG="${BOT_DIR}/logs/btc_ib_bot.log"
HF_LOG="${BOT_DIR}/logs/btc_ib_bot_hf.log"

if [ -f "$SWING_LOG" ]; then
    LAST_UPDATE=$(stat -f "%m" "$SWING_LOG")
    NOW=$(date +%s)
    AGE=$((NOW - LAST_UPDATE))

    if [ $AGE -gt 300 ]; then
        log "⚠️  Swing Bot log stale (${AGE}s) - may be disconnected"
    else
        log "✓ Swing Bot log active (updated ${AGE}s ago)"
    fi
fi

if [ -f "$HF_LOG" ]; then
    LAST_UPDATE=$(stat -f "%m" "$HF_LOG")
    NOW=$(date +%s)
    AGE=$((NOW - LAST_UPDATE))

    if [ $AGE -gt 300 ]; then
        log "⚠️  HF Bot log stale (${AGE}s) - may be disconnected"
    else
        log "✓ HF Bot log active (updated ${AGE}s ago)"
    fi
fi

log "=== Health Check Complete ==="
