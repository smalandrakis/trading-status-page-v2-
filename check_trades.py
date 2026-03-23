#!/usr/bin/env python3
"""Quick trade comparison between original and reverse bot."""
import sqlite3
import os
import sys
import subprocess
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
SINCE = sys.argv[1] if len(sys.argv) > 1 else "2026-03-18T08:36:00"

def query_trades(db, since):
    conn = sqlite3.connect(os.path.join(BASE, db))
    c = conn.cursor()
    c.execute(
        "SELECT id, model_id, direction, entry_time, exit_time, entry_price, "
        "exit_price, pnl_pct, pnl_dollar, exit_reason, bars_held "
        "FROM trades WHERE entry_time >= ? ORDER BY entry_time", (since,))
    rows = c.fetchall()
    conn.close()
    return rows

def fmt(t):
    ex = (t[4] or "OPEN")[:19]
    return "#%d %5s %-16s | %s @%.0f | %s @%.0f | %.2f (%.3f%%) | %-14s | %d bars" % (
        t[0], t[2], t[1], t[3][11:19], t[5], ex[11:19] if ex != "OPEN" else "OPEN    ",
        t[6] or 0, t[8] or 0, t[7] or 0, t[9] or "OPEN", t[10] or 0)

print("=" * 100)
print("TRADE COMPARISON: Since %s" % SINCE)
print("=" * 100)

orig = query_trades("tick_trades.db", SINCE)
rev = query_trades("tick_trades_reverse.db", SINCE)

# Open positions
conn = sqlite3.connect(os.path.join(BASE, "tick_trades.db"))
c = conn.cursor()
c.execute("SELECT * FROM open_positions")
openp = c.fetchall()
conn.close()

print("\n--- ORIGINAL BOT (%d closed, %d open) ---" % (len(orig), len(openp)))
for t in orig:
    print("  %s" % fmt(t))
for p in openp:
    print("  [OPEN] %s" % str(p))

print("\n--- REVERSE BOT (%d closed) ---" % len(rev))
for t in rev:
    instant_tag = " ** INSTANT SL **" if t[10] == 0 and t[9] == "STOP_LOSS" else ""
    print("  %s%s" % (fmt(t), instant_tag))

# Side-by-side matching
print("\n--- PAIR MATCHING (within 10s of entry) ---")
matched_rev = set()
issues = []
for o in orig:
    ot = datetime.fromisoformat(o[3])
    match = None
    for r in rev:
        if r[0] in matched_rev:
            continue
        rt = datetime.fromisoformat(r[3])
        if abs((rt - ot).total_seconds()) <= 10:
            match = r
            matched_rev.add(r[0])
            break
    if match:
        r = match
        dir_ok = (o[2] == "LONG" and r[2] == "SHORT") or (o[2] == "SHORT" and r[2] == "LONG")
        instant = r[10] == 0 and r[9] == "STOP_LOSS"
        o_win = o[8] and o[8] > 0
        r_win = r[8] and r[8] > 0
        mirror = (o_win and not r_win) or (not o_win and r_win) or (o[8] == 0 and r[8] == 0)
        status = "OK" if dir_ok and not instant else "ISSUE"
        if not dir_ok:
            issues.append("Dir not reversed: orig #%d" % o[0])
        if instant:
            issues.append("Instant SL: rev #%d" % r[0])
        print("  %s orig #%d %5s -> rev #%d %5s | orig %.2f rev %.2f | mirror=%s%s" % (
            status, o[0], o[2], r[0], r[2], o[8] or 0, r[8] or 0,
            "YES" if mirror else "NO",
            " INSTANT-SL" if instant else ""))
    else:
        issues.append("No match for orig #%d" % o[0])
        print("  MISS orig #%d %5s %s -- no reverse trade found" % (o[0], o[2], o[3][11:19]))

for r in rev:
    if r[0] not in matched_rev:
        issues.append("Orphan rev #%d" % r[0])
        print("  ORPH rev #%d %5s %s -- no original trade matched" % (r[0], r[2], r[3][11:19]))

# Summary
print("\n--- SUMMARY ---")
orig_pnl = sum(t[8] for t in orig if t[8])
rev_pnl = sum(t[8] for t in rev if t[8])
instant_count = len([t for t in rev if t[10] == 0 and t[9] == "STOP_LOSS"])
print("Original: %d trades, PnL=%.2f" % (len(orig), orig_pnl))
print("Reverse:  %d trades, PnL=%.2f" % (len(rev), rev_pnl))
print("Instant SLs: %d" % instant_count)
print("Issues: %d" % len(issues))
for iss in issues:
    print("  - %s" % iss)

# Processes + log recency
result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
lines = [l for l in result.stdout.split("\n") if "btc_tick_bot" in l and "grep" not in l]
print("\n--- STATUS ---")
for l in lines:
    parts = l.split()
    print("  PID %s: %s" % (parts[1], " ".join(parts[10:])))
for logfile in ["logs/btc_tick_bot.log", "logs/btc_tick_bot_reverse.log"]:
    path = os.path.join(BASE, logfile)
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                last = line
            print("  %s: %s" % (os.path.basename(logfile), last.strip()[:120]))
