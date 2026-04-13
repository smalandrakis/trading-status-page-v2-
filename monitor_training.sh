#!/bin/bash
# Monitor Full Training Progress

LOG_FILE="/tmp/train_micro_full_nosampling.log"
PID_FILE="/tmp/train_micro.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No training PID file found"
    exit 1
fi

PID=$(cat "$PID_FILE")

echo "================================================================"
echo "MICRO MODELS TRAINING MONITOR"
echo "================================================================"
echo ""

if ps -p $PID > /dev/null 2>&1; then
    ELAPSED=$(ps -p $PID -o etime= | tr -d ' ')
    CPU=$(ps -p $PID -o %cpu= | tr -d ' ')
    echo "Status: RUNNING"
    echo "PID: $PID"
    echo "Elapsed: $ELAPSED"
    echo "CPU: $CPU%"
    echo ""

    # Check log for progress
    if [ -f "$LOG_FILE" ]; then
        echo "Latest progress:"
        tail -10 "$LOG_FILE" | grep -E "(Progress|Loaded|Training|Test Accuracy|COMPLETE)" | tail -5
    fi

    echo ""
    echo "Estimated completion: ~15-20 minutes total"
else
    echo "Status: COMPLETE or FAILED"
    echo ""

    # Check if models exist
    MODELS_DIR="/Users/smalandrakis/Documents/WindSurf/DS:ML trade model playground/CascadeProjects/windsurf-project/models"
    if ls "$MODELS_DIR"/*micro*.pkl > /dev/null 2>&1; then
        echo "✓ SUCCESS: Models created"
        echo ""
        echo "Model files:"
        ls -lht "$MODELS_DIR"/*micro*.pkl | head -7

        echo ""
        echo "Training results:"
        tail -30 "$LOG_FILE" | grep -E "(Test Accuracy|Model Performance)"
    else
        echo "✗ FAILED: No models created"
        echo ""
        echo "Last 20 lines of log:"
        tail -20 "$LOG_FILE"
    fi
fi

echo ""
echo "================================================================"
