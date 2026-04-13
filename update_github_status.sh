#!/bin/bash
# Auto-update GitHub Pages with bot status

BOT_DIR="/Users/smalandrakis/Documents/WindSurf/DS:ML trade model playground/CascadeProjects/windsurf-project"
GITHUB_DIR="/Users/smalandrakis/trading-status-page-v2"

# Generate fresh status
cd "$BOT_DIR"
python3 generate_status.py > /dev/null 2>&1

# Copy to GitHub Pages repo
cp "$BOT_DIR/bot_status.json" "$GITHUB_DIR/"

# Commit and push if there are changes
cd "$GITHUB_DIR"
git checkout gh-pages > /dev/null 2>&1

if ! git diff --quiet bot_status.json; then
    git add bot_status.json
    git commit -m "Auto-update bot status - $(date '+%Y-%m-%d %H:%M')" > /dev/null 2>&1
    git push origin gh-pages > /dev/null 2>&1
fi
