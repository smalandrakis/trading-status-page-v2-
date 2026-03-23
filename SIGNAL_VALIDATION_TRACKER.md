# Signal Validation Tracker
## Started: January 7, 2026

This file tracks actual signals vs expected signals to validate the bot configurations.

---

# Expected Signals Per Day (Baseline)

| Bot | Expected Signals/Day | Expected Trades/Week |
|-----|---------------------|---------------------|
| BTC | ~40 (with cooldown) | ~280 |
| MNQ | ~9 (with cooldown) | ~46 |
| SPY | ~6 (with cooldown) | ~31 |

---

# Daily Validation Log

## Week 1: January 7-13, 2026

### January 7, 2026 (Partial - Started 11:00 PM)
| Bot | Signals | Trades | Notes |
|-----|---------|--------|-------|
| BTC | - | - | Started with new config |
| MNQ | - | - | |
| SPY | - | - | |

### January 8, 2026
| Bot | Signals | Trades | Notes |
|-----|---------|--------|-------|
| BTC | | | |
| MNQ | | | |
| SPY | | | |

### January 9, 2026
| Bot | Signals | Trades | Notes |
|-----|---------|--------|-------|
| BTC | | | |
| MNQ | | | |
| SPY | | | |

### January 10, 2026
| Bot | Signals | Trades | Notes |
|-----|---------|--------|-------|
| BTC | | | |
| MNQ | | | |
| SPY | | | |

---

# Weekly Summary

## Week 1 (Jan 7-13)
| Bot | Expected | Actual | Variance | Notes |
|-----|----------|--------|----------|-------|
| BTC | 280 | | | |
| MNQ | 46 | | | |
| SPY | 31 | | | |

---

# How to Count Signals

Run this command to count signals from the CSV logs:

```bash
# Count BTC signals for a specific date
grep -c "^2026-01-08" signal_logs/btc_signals_2026-01-08.csv

# Count unique signal triggers (where signal column = 1)
awk -F',' 'NR>1 && ($6==1 || $7==1 || $8==1 || $9==1 || $10==1) {count++} END {print count}' signal_logs/btc_signals_2026-01-08.csv

# Count actual trades from trade database
sqlite3 trades.db "SELECT COUNT(*) FROM trades WHERE date(entry_time) = '2026-01-08' AND bot = 'BTC'"
```

---

# Notes

- Signals are logged every 15 seconds
- A "signal" = probability >= threshold
- A "trade" = actual position entered (respects cooldown and position limits)
- Expected signals assume no position limits blocking entries

---

*Last updated: January 7, 2026*
