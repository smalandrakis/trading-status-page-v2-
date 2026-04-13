# BTC Trading Bots - Dual Strategy Setup

## Overview

Two automated BTC futures trading bots running on IB Gateway (paper trading):

1. **Swing Bot (3.0/1.5)** - Lower frequency, higher P&L per trade
2. **High Frequency Bot (1.5/0.3)** - Higher frequency for faster validation

Both use V3 ensemble models with confidence-based position sizing (1-5x).

---

## Bot Configurations

### 1. Swing Bot (Optimal Config)
**File:** `btc_ib_gateway_bot.py`
- **TP/SL:** 3.0% / 1.5%
- **Expected:** 618 trades/2yr, 52.6% WR, +$16,577
- **Frequency:** ~0.85 trades/day (5-6 per week)
- **Client ID:** 10
- **Log:** `logs/btc_ib_bot.log`
- **Trades:** `logs/btc_trades.jsonl`

### 2. High Frequency Bot
**File:** `btc_ib_gateway_bot_hf.py`
- **TP/SL:** 1.5% / 0.3% (tighter stops)
- **Expected:** 790 trades/2yr, 33.7% WR, +$11,689
- **Frequency:** ~1.08 trades/day (7-8 per week)
- **Client ID:** 11
- **Log:** `logs/btc_ib_bot_hf.log`
- **Trades:** `logs/btc_trades_hf.jsonl`

---

## Starting the Bots

### Start Both Bots:
```bash
cd "/Users/smalandrakis/Documents/WindSurf/DS:ML trade model playground/CascadeProjects/windsurf-project"

# Start Swing Bot
nohup python3 btc_ib_gateway_bot.py > /dev/null 2>&1 &

# Start High Frequency Bot  
nohup python3 btc_ib_gateway_bot_hf.py > /dev/null 2>&1 &

echo "Both bots started!"
```

### Monitor Activity:
```bash
# Swing Bot
tail -f logs/btc_ib_bot.log

# High Frequency Bot
tail -f logs/btc_ib_bot_hf.log

# Recent trades (both)
tail -20 logs/btc_trades.jsonl
tail -20 logs/btc_trades_hf.jsonl
```

### Stop Bots:
```bash
pkill -f "btc_ib_gateway_bot.py"
pkill -f "btc_ib_gateway_bot_hf.py"
```

---

## Trade Logging

All trades are logged in JSONL format with full entry/exit details:

### Entry Signal Log:
```json
{
  "timestamp": "2026-04-04T12:30:00",
  "event": "ENTRY_SIGNAL",
  "data": {
    "direction": "LONG",
    "confidence": 0.72,
    "entry_price": 67500.0,
    "tp_price": 69525.0,
    "sl_price": 66487.5,
    "position_size": 2,
    "tp_pct": 3.0,
    "sl_pct": 1.5,
    "predictor_details": {...}
  }
}
```

### Exit Log:
```json
{
  "timestamp": "2026-04-04T14:15:00",
  "event": "EXIT",
  "data": {
    "direction": "LONG",
    "exit_reason": "TAKE_PROFIT",
    "entry_price": 67500.0,
    "exit_price": 69525.0,
    "entry_time": "2026-04-04 12:30:00",
    "exit_time": "2026-04-04 14:15:00",
    "position_size": 2,
    "hold_minutes": 105,
    "pnl_pct": 3.0,
    "pnl_dollar": 1350.0
  }
}
```

---

## Status Page Integration

### Generate Status JSON:
```bash
python3 generate_status.py
```

This creates `bot_status.json` with:
- Real-time bot status (running/stopped)
- Performance metrics (trades, W/L, P&L)
- Recent trades (last 10)
- Configuration details

### Setup Automated Updates:

Add to crontab (update every 5 minutes):
```bash
*/5 * * * * cd "/Users/smalandrakis/Documents/WindSurf/DS:ML trade model playground/CascadeProjects/windsurf-project" && python3 generate_status.py && cp bot_status.json ~/path/to/trading-status-page/
```

Or run manually after each trading session.

---

## GitHub Pages Integration

### Repository: https://smalandrakis.github.io/trading-status-page/

1. Clone your status page repo:
```bash
git clone https://github.com/smalandrakis/trading-status-page.git
cd trading-status-page
```

2. Create/update `index.html` to fetch `bot_status.json`:
```html
<!DOCTYPE html>
<html>
<head>
    <title>BTC Trading Bots Status</title>
    <style>
        body { font-family: monospace; max-width: 1200px; margin: 20px auto; }
        .bot { border: 1px solid #ccc; padding: 20px; margin: 20px 0; }
        .running { border-left: 5px solid #4CAF50; }
        .stopped { border-left: 5px solid #f44336; }
        .trades { font-size: 12px; }
        .win { color: #4CAF50; }
        .loss { color: #f44336; }
    </style>
</head>
<body>
    <h1>BTC Trading Bots - Live Status</h1>
    <div id="bots"></div>
    
    <script>
        async function loadStatus() {
            const res = await fetch('bot_status.json');
            const data = await res.json();
            
            const botsDiv = document.getElementById('bots');
            botsDiv.innerHTML = '';
            
            data.bots.forEach(bot => {
                const statusClass = bot.status === 'running' ? 'running' : 'stopped';
                
                const recentTrades = bot.recent_trades.map(t => `
                    <div class="${t.pnl > 0 ? 'win' : 'loss'}">
                        ${t.time} | ${t.direction} | ${t.pnl > 0 ? '+' : ''}$${t.pnl} (${t.pnl_pct}%) | ${t.hold_minutes}min
                    </div>
                `).join('');
                
                botsDiv.innerHTML += `
                    <div class="bot ${statusClass}">
                        <h2>${bot.name} - ${bot.status.toUpperCase()}</h2>
                        <p>Last Update: ${bot.last_update}</p>
                        
                        <h3>Configuration</h3>
                        <p>TP/SL: ${bot.config.tp_pct}% / ${bot.config.sl_pct}%</p>
                        <p>Expected: ${bot.config.expected_trades_per_week} trades/week, ${bot.config.expected_win_rate}% WR</p>
                        
                        <h3>Performance</h3>
                        <p>Trades: ${bot.performance.total_trades} (${bot.performance.wins}W / ${bot.performance.losses}L)</p>
                        <p>Win Rate: ${bot.performance.win_rate}%</p>
                        <p>Total P&L: $${bot.performance.total_pnl}</p>
                        <p>Avg P&L: $${bot.performance.avg_pnl}</p>
                        
                        <h3>Recent Trades</h3>
                        <div class="trades">${recentTrades || 'No trades yet'}</div>
                    </div>
                `;
            });
        }
        
        loadStatus();
        setInterval(loadStatus, 60000); // Refresh every minute
    </script>
</body>
</html>
```

3. Copy bot_status.json and push:
```bash
cp "/Users/smalandrakis/Documents/WindSurf/DS:ML trade model playground/CascadeProjects/windsurf-project/bot_status.json" .
git add .
git commit -m "Update bot status"
git push
```

4. Visit: https://smalandrakis.github.io/trading-status-page/

---

## Validation Plan

### Week 1-2: High Frequency Bot
- **Goal:** 14-16 trades
- **Validate:** Win rate ~33.7%, positive P&L
- **Check:** Trade execution, TP/SL triggers

### Week 3-4: Swing Bot
- **Goal:** 10-12 trades
- **Validate:** Win rate ~52.6%, higher P&L per trade
- **Check:** Hold times, position sizing

### Month 1: Combined Analysis
- **Compare:** Actual vs expected performance
- **Analyze:** Which config matches backtest better
- **Decide:** Scale up winning configuration

---

## Key Metrics to Track

| Metric | Swing Bot Expected | HF Bot Expected |
|--------|-------------------|----------------|
| Trades/week | 5-6 | 7-8 |
| Win Rate | 52.6% | 33.7% |
| Avg P&L/trade | $26.82 | $14.80 |
| Avg hold time | ~2-3 hours | ~1-2 hours |
| Weekly P&L | ~$160 | ~$110 |

---

## Files

- **btc_ib_gateway_bot.py** - Swing bot (3.0/1.5)
- **btc_ib_gateway_bot_hf.py** - High frequency bot (1.5/0.3)
- **generate_status.py** - Status JSON generator
- **bot_status.json** - Current bot status
- **logs/btc_ib_bot.log** - Swing bot activity log
- **logs/btc_ib_bot_hf.log** - HF bot activity log
- **logs/btc_trades.jsonl** - Swing bot trade log
- **logs/btc_trades_hf.jsonl** - HF bot trade log

---

## Troubleshooting

### Bot won't start:
- Check IB Gateway is running on port 4002
- Verify client IDs not in use (10, 11)
- Check logs for errors

### Orders cancelled:
- "TIF set to DAY" = normal for paper trading
- "Market closed" = wait for Sunday 5pm CT

### No trades:
- Confidence threshold is 0.65/0.25 (selective)
- Check log for NEUTRAL signals
- Market may not have strong signals

---

## Contract Rollover

MBT contracts expire monthly. Update front month in both bots:

```python
# Around April 20, change to May contract
lastTradeDateOrContractMonth='20260529'  # MBTK6 (May 2026)
```
