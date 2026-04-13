# Bot Health Monitoring & Auto-Restart Guide

## Overview

The bot health monitoring system automatically:
1. Checks if both bots are running
2. Restarts crashed bots automatically
3. Monitors IB Gateway connectivity
4. Tracks error rates
5. Updates bot_status.json

---

## Setup Automated Monitoring

### Option 1: Cron Job (Every 5 Minutes)

```bash
# Edit crontab
crontab -e

# Add this line:
*/5 * * * * cd "/Users/smalandrakis/Documents/WindSurf/DS:ML trade model playground/CascadeProjects/windsurf-project" && ./bot_health_monitor.sh
```

### Option 2: Manual Check

```bash
cd "/Users/smalandrakis/Documents/WindSurf/DS:ML trade model playground/CascadeProjects/windsurf-project"
./bot_health_monitor.sh
```

---

## What It Monitors

### 1. Bot Process Status
- Checks if both `btc_ib_gateway_bot.py` and `btc_ib_gateway_bot_hf.py` are running
- **Auto-restarts** if either bot has crashed

### 2. IB Gateway Connection
- Checks if port 4002 has active connections
- Alerts if no connection detected

### 3. Recent Activity
- Displays last log entry from each bot
- Counts total trade events

### 4. Error Detection
- Scans last 100 log lines for errors
- Alerts if >5 errors detected

### 5. Status JSON Generation
- Automatically runs `generate_status.py`
- Updates `bot_status.json` with latest metrics

---

## Log Files

| File | Purpose |
|------|---------|
| `logs/health_monitor.log` | All health check results |
| `logs/bot_alerts.log` | Critical alerts only |
| `logs/btc_ib_bot.log` | Swing bot activity |
| `logs/btc_ib_bot_hf.log` | HF bot activity |
| `bot_status.json` | Current bot status (for status page) |

---

## Example Output

```
[2026-04-04 17:20:03] ========== Health Check Started ==========
[2026-04-04 17:20:03] ✓ Swing Bot is running (PID: 12345)
[2026-04-04 17:20:03]   Last activity: [2026-04-04 17:19:52] BTC: $67,215 | LONG @ 2x...
[2026-04-04 17:20:03] ⚠️  ALERT: HF Bot is NOT running - attempting restart
[2026-04-04 17:20:03] ✓ HF Bot restarted with PID 12346
[2026-04-04 17:20:03] ✓ IB Gateway connection active
[2026-04-04 17:20:03]   Swing Bot: 14 total trade events logged
[2026-04-04 17:20:03]   HF Bot: 22 total trade events logged
[2026-04-04 17:20:03] ========== Health Check Complete ==========
```

---

## Manual Bot Commands

### Check Status
```bash
# View running bots
ps aux | grep btc_ib_gateway_bot

# Check logs
tail -f logs/btc_ib_bot.log
tail -f logs/btc_ib_bot_hf.log

# View recent trades
tail -20 logs/btc_trades.jsonl
tail -20 logs/btc_trades_hf.jsonl
```

### Start Bots
```bash
nohup python3 btc_ib_gateway_bot.py > /dev/null 2>&1 &
nohup python3 btc_ib_gateway_bot_hf.py > /dev/null 2>&1 &
```

### Stop Bots
```bash
pkill -f "btc_ib_gateway_bot.py"
pkill -f "btc_ib_gateway_bot_hf.py"
```

### Restart All
```bash
pkill -f "btc_ib_gateway_bot"
./bot_health_monitor.sh  # Will auto-restart both
```

---

## Improvements Added

### ✅ Retry Logic
Both bots now retry Binance API calls with exponential backoff:
- 5 retry attempts
- Delays: 2s, 5s, 10s, 20s, 30s
- Prevents crashes from temporary network issues

### ✅ Auto-Restart
Health monitor automatically restarts crashed bots:
- Checks every 5 minutes (via cron)
- Logs restart events
- Maintains continuous operation

### ✅ Error Monitoring
Tracks error rates in real-time:
- Alerts if >5 errors in last 100 log lines
- Helps identify issues before they compound

---

## Recommended Setup

1. **Enable cron monitoring:**
   ```bash
   crontab -e
   # Add: */5 * * * * cd "/Users/smalandrakis/Documents/WindSurf/DS:ML trade model playground/CascadeProjects/windsurf-project" && ./bot_health_monitor.sh
   ```

2. **Monitor alerts daily:**
   ```bash
   tail -20 logs/bot_alerts.log
   ```

3. **Review health logs weekly:**
   ```bash
   tail -100 logs/health_monitor.log
   ```

---

## GitHub Status Page Integration

The health monitor automatically updates `bot_status.json`, which can be pushed to your GitHub Pages status page:

```bash
# Manual update
python3 generate_status.py
cp bot_status.json ~/trading-status-page/
cd ~/trading-status-page
git add .
git commit -m "Update bot status"
git push
```

Or add to cron for automated updates:
```bash
*/10 * * * * cd "/path/to/windsurf-project" && python3 generate_status.py && cp bot_status.json ~/trading-status-page/ && cd ~/trading-status-page && git add . && git commit -m "Auto-update $(date)" && git push
```

---

## Troubleshooting

### Bot keeps restarting
- Check `logs/bot_alerts.log` for recurring errors
- Verify IB Gateway is running and accessible
- Check Binance API is reachable: `curl -s https://api.binance.com/api/v3/ping`

### No trades being placed
- Check bot logs for NEUTRAL signals
- Verify confidence thresholds (0.65/0.25)
- Ensure market is open (Sunday 6pm ET - Friday 5pm ET)

### High error count
- Review specific errors in bot logs
- May indicate:
  - IB Gateway disconnection
  - Binance API rate limiting
  - Model prediction failures

---

## Files Created

1. ✅ `bot_health_monitor.sh` - Main health check script
2. ✅ `logs/health_monitor.log` - Health check log
3. ✅ `logs/bot_alerts.log` - Critical alerts
4. ✅ Updated both bot files with retry logic
5. ✅ This documentation file

---

## Next Steps

1. Set up cron job for automated monitoring
2. Test during market hours (Sunday evening)
3. Monitor first week of trades
4. Review error logs weekly
