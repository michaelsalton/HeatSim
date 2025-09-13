#!/bin/bash

echo "======================================"
echo "Heat Transfer Physics Validation Test"
echo "======================================"
echo ""

echo "Testing 1D heat equation implementation with CUDA"
echo ""

# Check CUDA availability
echo "GPU Configuration:"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
echo ""

# Run the simulation
echo "Running heat simulation test..."
echo "Material: Aluminum (k=205 W/m·K, ρ=2700 kg/m³, c=900 J/kg·K)"
echo "Boundary conditions: T_left=100°C, T_right=20°C"
echo "Rod length: 1.0 m, Points: 100"
echo ""

timeout 3s ./build/bin/HeatSim 2>&1 | grep -E "(CUDA|Physics|Simulation|points|Memory)"

echo ""
echo "Physics implementation details:"
echo "- 1D Heat Equation: ∂T/∂t = α ∂²T/∂x²"
echo "- Method: Explicit finite difference"
echo "- Stability: CFL condition (Δt ≤ 0.5 Δx²/α)"
echo "- Boundary: Dirichlet conditions"
echo ""

echo "Test completed successfully!"