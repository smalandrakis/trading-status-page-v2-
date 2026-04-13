#!/bin/bash
# Quick training progress checker

PID=53273

if ps -p $PID > /dev/null 2>&1; then
    TIME=$(ps -p $PID -o etime= | tr -d ' ')
    CPU=$(ps -p $PID -o %cpu= | tr -d ' ')
    echo "Training RUNNING: PID=$PID, Time=$TIME, CPU=$CPU%"

    # Check if models exist
    if ls "/Users/smalandrakis/Documents/WindSurf/DS:ML trade model playground/CascadeProjects/windsurf-project/models/"*micro*.pkl > /dev/null 2>&1; then
        echo "Models found - training likely complete!"
        ls -lh "/Users/smalandrakis/Documents/WindSurf/DS:ML trade model playground/CascadeProjects/windsurf-project/models/"*micro*.pkl
    else
        echo "No models yet - still in feature extraction phase"
    fi
else
    echo "Training COMPLETE or FAILED (process not found)"

    # Check if models exist
    if ls "/Users/smalandrakis/Documents/WindSurf/DS:ML trade model playground/CascadeProjects/windsurf-project/models/"*micro*.pkl > /dev/null 2>&1; then
        echo "SUCCESS: Models created:"
        ls -lh "/Users/smalandrakis/Documents/WindSurf/DS:ML trade model playground/CascadeProjects/windsurf-project/models/"*micro*.pkl
    else
        echo "FAILED: No models created"
    fi
fi
