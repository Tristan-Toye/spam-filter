#!/bin/bash

# Script to compile, run experiments, and generate visualizations
# Big Data Analytics Programming - Assignment 1
# Spam Filter Experiments

echo "=========================================="
echo "Spam Filter Experiment Pipeline"
echo "=========================================="
echo ""

# Step 1: Compile the project
echo "[1/3] Compiling C++ code..."
echo "-------------------------------------------"

# Create build directory if it doesn't exist
if [ ! -d "../build" ]; then
    mkdir ../build
fi

cd ../build

# Run CMake
echo "Running CMake..."
cmake ..

if [ $? -ne 0 ]; then
    echo "ERROR: CMake configuration failed!"
    exit 1
fi

# Compile
echo "Compiling with make..."
make

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi

echo "✓ Compilation successful"
echo ""

# Step 2: Run the experiments
echo "[2/3] Running experiments..."
echo "-------------------------------------------"
echo "This will take several minutes..."
echo ""

# Run with default seed=12 and window=200
./src/bdap_assignment1 12 200

if [ $? -ne 0 ]; then
    echo "ERROR: Experiment execution failed!"
    exit 1
fi

echo ""
echo "✓ Experiments complete"
echo ""

# Step 3: Generate visualizations
echo "[3/3] Generating visualizations..."
echo "-------------------------------------------"

cd ../src

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python not found!"
        exit 1
    else
        PYTHON_CMD=python
    fi
else
    PYTHON_CMD=python3
fi

# Install required packages if needed (uncomment if running for first time)
# pip install pandas matplotlib seaborn numpy

# Run visualization script
$PYTHON_CMD plot.py

if [ $? -ne 0 ]; then
    echo "ERROR: Visualization generation failed!"
    exit 1
fi

echo ""
echo "✓ Visualizations generated"
echo ""

# Summary
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  CSV Results:"
echo "    - learning_curves.csv"
echo "    - threshold_experiments.csv"
echo "    - filter_experiments.csv"
echo "    - hyperparameter_results.csv"
echo "    - timing_results.csv"
echo "    - best_configurations.csv"
echo ""
echo "  Visualizations:"
echo "    - learning_curves.png"
echo "    - learning_curves_focused.png"
echo "    - threshold_comparison.png"
echo "    - bucket_filter_impact.png"
echo "    - hyperparameter_heatmap_fh.png"
echo "    - hyperparameter_heatmap_cm.png"
echo "    - precision_recall_tradeoff.png"
echo "    - computational_efficiency.png"
echo ""
echo "All experiments completed successfully!"

