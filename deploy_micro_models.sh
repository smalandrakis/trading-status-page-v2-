#!/bin/bash
# Deploy Custom Micro Models to Bot
# Run this after training completes and validation shows improvement

echo "==================================================================="
echo "DEPLOY CUSTOM MICRO MODELS TO BOT"
echo "==================================================================="

BOT_DIR="/Users/smalandrakis/Documents/WindSurf/DS:ML trade model playground/CascadeProjects/windsurf-project"

# 1. Check models exist
echo ""
echo "Step 1: Verify models exist..."
if ! ls "$BOT_DIR/models/"*micro*.pkl > /dev/null 2>&1; then
    echo "ERROR: Micro models not found! Run train_btc_models_micro.py first."
    exit 1
fi

echo "✓ Found micro models:"
ls -lh "$BOT_DIR/models/"*micro*.pkl

# 2. Stop current bot
echo ""
echo "Step 2: Stopping current bot..."
BOT_PID=$(cat "$BOT_DIR/btc_micro_bot.pid" 2>/dev/null)
if [ -n "$BOT_PID" ] && ps -p $BOT_PID > /dev/null 2>&1; then
    echo "Stopping bot (PID $BOT_PID)..."
    kill $BOT_PID
    sleep 2
    echo "✓ Bot stopped"
else
    echo "No running bot found"
fi

# 3. Update bot to use MicroPredictor
echo ""
echo "Step 3: Update bot code to use MicroPredictor..."
echo "Manual step required - edit btc_micro_bot.py:"
echo "  Change: from btc_model_package.predictor import BTCPredictor"
echo "  To: from btc_model_package.micro_predictor import MicroPredictor"
echo "  Change: predictor = BTCPredictor()"
echo "  To: predictor = MicroPredictor()"
echo ""
read -p "Press Enter after making these changes..."

# 4. Restart bot
echo ""
echo "Step 4: Restarting bot with custom models..."
cd "$BOT_DIR"
nohup python3 btc_micro_bot.py > logs/btc_micro_bot.log 2>&1 &
NEW_PID=$!
echo $NEW_PID > btc_micro_bot.pid
echo "✓ Bot restarted with PID $NEW_PID"

# 5. Update monitoring
echo ""
echo "Step 5: Update generate_status.py with new expected metrics..."
echo "✓ Run: python3 generate_status.py"

# 6. Verify
echo ""
echo "Step 6: Monitor first few predictions..."
tail -f logs/btc_micro_bot.log

echo ""
echo "==================================================================="
echo "DEPLOYMENT COMPLETE"
echo "==================================================================="
echo "Monitor for 1 hour to verify custom models are working"
echo "Expected improvements from baseline:"
echo "  - Win Rate: 37.6% → 42%+"
echo "  - Avg P&L: $0.07 → $15+"
