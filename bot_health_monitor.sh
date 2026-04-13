#!/bin/bash

# BTC Bot Health Monitor with Auto-Restart
# Usage: ./bot_health_monitor.sh
# Or add to crontab: */5 * * * * /path/to/bot_health_monitor.sh

BOT_DIR="/Users/smalandrakis/Documents/WindSurf/DS:ML trade model playground/CascadeProjects/windsurf-project"
LOG_FILE="$BOT_DIR/logs/health_monitor.log"
ALERT_FILE="$BOT_DIR/logs/bot_alerts.log"

cd "$BOT_DIR" || exit 1
mkdir -p logs

# Timestamp function
timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# Log function
log_msg() {
    echo "[$(timestamp)] $1" | tee -a "$LOG_FILE"
}

# Alert function (for critical issues)
alert() {
    echo "[$(timestamp)] ⚠️  ALERT: $1" | tee -a "$ALERT_FILE"
}

log_msg "========== Health Check Started =========="

# Check Swing Bot
SWING_PID=$(pgrep -f "btc_ib_gateway_bot.py" | grep -v "_hf")
if [ -z "$SWING_PID" ]; then
    alert "Swing Bot is NOT running - attempting restart"
    nohup python3 btc_ib_gateway_bot.py > /dev/null 2>&1 &
    NEW_PID=$!
    log_msg "✓ Swing Bot restarted with PID $NEW_PID"
else
    log_msg "✓ Swing Bot is running (PID: $SWING_PID)"

    # Check if bot is responsive (last log entry within 10 minutes)
    if [ -f "$BOT_DIR/logs/btc_ib_bot.log" ]; then
        LAST_LOG=$(tail -1 "$BOT_DIR/logs/btc_ib_bot.log")
        log_msg "  Last activity: $LAST_LOG"
    fi
fi

# Check HF Bot
HF_PID=$(pgrep -f "btc_ib_gateway_bot_hf.py")
if [ -z "$HF_PID" ]; then
    alert "HF Bot is NOT running - attempting restart"
    nohup python3 btc_ib_gateway_bot_hf.py > /dev/null 2>&1 &
    NEW_PID=$!
    log_msg "✓ HF Bot restarted with PID $NEW_PID"
else
    log_msg "✓ HF Bot is running (PID: $HF_PID)"

    # Check if bot is responsive
    if [ -f "$BOT_DIR/logs/btc_ib_bot_hf.log" ]; then
        LAST_LOG=$(tail -1 "$BOT_DIR/logs/btc_ib_bot_hf.log")
        log_msg "  Last activity: $LAST_LOG"
    fi
fi

# Check IB Gateway connectivity
IB_CHECK=$(netstat -an | grep ":4002" | grep ESTABLISHED | wc -l)
if [ "$IB_CHECK" -gt 0 ]; then
    log_msg "✓ IB Gateway connection active"
else
    alert "IB Gateway connection NOT detected on port 4002"
fi

# Check recent trades
if [ -f "$BOT_DIR/logs/btc_trades.jsonl" ]; then
    SWING_TRADES=$(wc -l < "$BOT_DIR/logs/btc_trades.jsonl")
    log_msg "  Swing Bot: $SWING_TRADES total trade events logged"
fi

if [ -f "$BOT_DIR/logs/btc_trades_hf.jsonl" ]; then
    HF_TRADES=$(wc -l < "$BOT_DIR/logs/btc_trades_hf.jsonl")
    log_msg "  HF Bot: $HF_TRADES total trade events logged"
fi

# Check for errors in recent logs
ERROR_COUNT=$(tail -100 "$BOT_DIR/logs/btc_ib_bot.log" 2>/dev/null | grep -i "error\|exception\|failed" | wc -l)
if [ "$ERROR_COUNT" -gt 5 ]; then
    alert "High error count ($ERROR_COUNT) in Swing Bot logs"
fi

ERROR_COUNT_HF=$(tail -100 "$BOT_DIR/logs/btc_ib_bot_hf.log" 2>/dev/null | grep -i "error\|exception\|failed" | wc -l)
if [ "$ERROR_COUNT_HF" -gt 5 ]; then
    alert "High error count ($ERROR_COUNT_HF) in HF Bot logs"
fi

# Generate status JSON
python3 generate_status.py 2>/dev/null

log_msg "========== Health Check Complete ==========\n"
