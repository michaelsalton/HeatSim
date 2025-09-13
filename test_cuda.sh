#!/bin/bash

echo "Testing CUDA Heat Simulation..."
echo "================================"

# Check if CUDA is available
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null
if [ $? -eq 0 ]; then
    echo "CUDA GPU detected"
else
    echo "No CUDA GPU detected"
fi

echo ""
echo "Running HeatSim with CUDA..."
echo "================================"

# Run the application for a few seconds
timeout 5s ./build/bin/HeatSim 2>&1 | head -30

echo ""
echo "Test completed!"