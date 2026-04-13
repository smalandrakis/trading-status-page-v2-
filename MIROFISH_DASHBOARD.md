# MiroFish Dashboard Integration

## What's New

Your GitHub trading dashboard now displays **MiroFish AI predictions** alongside your bot status!

## Features

✓ **Real-time MiroFish signals** displayed at the top of the dashboard  
✓ **Three signal sources** shown:
  - 🧠 Cerebras Llama 3.3 70B (hourly updates)
  - ⚡ Groq Llama 3.3 70B (daily updates)
  - 🐟 MiroFish Multi-Agent (full simulation - on-demand)

✓ **Signal freshness indicators**:
  - 🟢 FRESH badge: Signal less than 24 hours old
  - 🟠 STALE badge: Signal older than 24 hours

✓ **Detailed signal info**:
  - Direction (LONG/SHORT/NEUTRAL)
  - Confidence percentage
  - Strength (signal power)
  - AI reasoning snippet
  - Signal age in minutes

## How It Works

```
MiroFish Signals (JSON files)
         ↓
update_mirofish_status.py (reads signals)
         ↓
bot_status.json (updated with MiroFish data)
         ↓
index.html (dashboard displays predictions)
```

## Setup Instructions

### 1. Update Dashboard Data

Run the update script to fetch latest MiroFish signals:

```bash
cd /Users/smalandrakis/trading-status-page-v2
python3 update_mirofish_status.py
```

This reads signals from:
- `~/CascadeProjects/mirofish-signal/output/latest_signal_cerebras.json`
- `~/CascadeProjects/mirofish-signal/output/latest_signal_groq.json`
- `~/CascadeProjects/mirofish-signal/output/latest_signal.json` (full MiroFish)

### 2. Automate Updates

**Option A: Cron Job (Recommended)**

```bash
# Edit crontab
crontab -e

# Add line to update every 5 minutes
*/5 * * * * cd /Users/smalandrakis/trading-status-page-v2 && python3 update_mirofish_status.py >> ~/mirofish-dashboard.log 2>&1
```

**Option B: Run Manually**

Update whenever you generate new MiroFish signals:

```bash
# Generate MiroFish prediction
cd ~/mirofish-trading/trading_integration
python mirofish_btc_adapter.py

# Update dashboard
cd /Users/smalandrakis/trading-status-page-v2
python3 update_mirofish_status.py
```

### 3. View Dashboard

Open `index.html` in your browser:

```bash
# Mac
open /Users/smalandrakis/trading-status-page-v2/index.html

# Or navigate to it in your browser
# file:///Users/smalandrakis/trading-status-page-v2/index.html
```

The dashboard refreshes automatically every 60 seconds.

## Publishing to GitHub

If you want to host this on GitHub Pages:

### 1. Push to GitHub

```bash
cd /Users/smalandrakis/trading-status-page-v2

# Check git status
git status

# Add changes
git add index.html update_mirofish_status.py MIROFISH_DASHBOARD.md

# Commit
git commit -m "Add MiroFish predictions to dashboard"

# Push to GitHub
git push origin main
```

### 2. Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** → **Pages**
3. Under "Source", select **main branch**
4. Click **Save**
5. Your dashboard will be live at: `https://[username].github.io/trading-status-page-v2/`

### 3. Auto-Update bot_status.json

Set up a GitHub Action to auto-update:

Create `.github/workflows/update-mirofish.yml`:

```yaml
name: Update MiroFish Status

on:
  schedule:
    - cron: '*/5 * * * *'  # Every 5 minutes
  workflow_dispatch:  # Manual trigger

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Update MiroFish Status
        run: python3 update_mirofish_status.py
        
      - name: Commit changes
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add bot_status.json
          git commit -m "Auto-update MiroFish status" || exit 0
          git push
```

**Note:** For GitHub Actions to work, you'll need to make signal files accessible to the Action (either commit them or fetch from an API).

## Signal Sources Explained

### 🧠 Cerebras Llama 3.3 70B
- **Update frequency:** Hourly
- **Method:** Single LLM consensus
- **Speed:** Fast (~2-5 minutes)
- **Best for:** Quick directional signals

### ⚡ Groq Llama 3.3 70B
- **Update frequency:** Daily
- **Method:** Simple debate between agents
- **Speed:** Moderate (~5-10 minutes)
- **Best for:** Balanced predictions

### 🐟 MiroFish Multi-Agent
- **Update frequency:** On-demand (you trigger)
- **Method:** 25+ AI agents debate and evolve opinions
- **Speed:** Slow (~15-25 minutes)
- **Best for:** Comprehensive analysis before major decisions

## Customization

### Change Refresh Rate

Edit `index.html` line 223:

```javascript
setInterval(loadStatus, 60000); // 60000 = 1 minute
```

### Modify Signal Display

Edit the `signalsHtml` section in `index.html` around line 125-155 to customize:
- Colors
- Layout
- Information shown
- Confidence bar appearance

### Filter Stale Signals

Modify `update_mirofish_status.py` to exclude stale signals:

```python
# Line 32 - change threshold
'is_fresh': age_minutes < 720  # 12 hours instead of 24
```

## Troubleshooting

### No signals showing?

```bash
# Check if signal files exist
ls -la ~/CascadeProjects/mirofish-signal/output/

# Check if bot_status.json has mirofish_predictions
cat bot_status.json | grep -A 5 "mirofish_predictions"

# Regenerate signals
cd ~/mirofish-trading/trading_integration
python mirofish_btc_adapter.py
```

### Signals showing as STALE?

Generate fresh signals:

```bash
# Quick Cerebras update (hourly signal)
cd ~/CascadeProjects/windsurf-project
# Your bot generates these automatically

# Full MiroFish prediction
cd ~/mirofish-trading/trading_integration
python mirofish_btc_adapter.py
```

### Dashboard not updating?

```bash
# Verify update script ran
cd /Users/smalandrakis/trading-status-page-v2
python3 update_mirofish_status.py

# Check bot_status.json modification time
ls -la bot_status.json

# Hard refresh browser (Cmd+Shift+R on Mac)
```

## API Keys Still Needed

To generate full MiroFish predictions, you need:

1. **LLM_API_KEY** - AI model access (https://bailian.console.aliyun.com/)
2. **ZEP_API_KEY** - Agent memory (https://app.getzep.com/)

Once you have these:

```bash
cd ~/mirofish-trading
cp .env.example .env
# Edit .env and add your API keys

# Test MiroFish backend
npm run dev

# Generate first prediction
cd trading_integration
python mirofish_btc_adapter.py
```

## Summary

✅ **Dashboard enhanced** with MiroFish AI predictions  
✅ **Update script** created (`update_mirofish_status.py`)  
✅ **Auto-refresh** every 60 seconds  
✅ **Ready for GitHub Pages** deployment  
✅ **Displays 3 signal sources** (Cerebras, Groq, Full MiroFish)  

Your bots now have visual AI guidance displayed prominently on the dashboard! 🚀
