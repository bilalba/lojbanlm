#!/bin/bash
export PATH="/opt/homebrew/opt/openjdk/bin:/opt/homebrew/bin:$PATH"
cd ~/lojban_experiment

PYTHON=/Applications/Xcode.app/Contents/Developer/usr/bin/python3

echo "=== V3.1 Experiment Started: $(date) ==="

for SIZE in nano micro mini small base; do
    echo ""
    echo "=== Starting $SIZE: $(date) ==="
    $PYTHON -u experiment_v3.py --fixed-steps 10000 --size $SIZE
    echo "=== Finished $SIZE: $(date) ==="
done

echo ""
echo "=== V3.1 Experiment Complete: $(date) ==="
