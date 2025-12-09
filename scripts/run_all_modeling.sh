#!/bin/bash
# Complete modeling pipeline for OSFI B-15 climate risk analytics
# Runs regime shift detection and extreme value analysis using both Python and R

set -e  # Exit on error

echo "=========================================="
echo "OSFI B-15 Climate Risk Modeling Pipeline"
echo "=========================================="
echo ""

# Check if required data exists
if [ ! -f "data_processed/regional_anomalies.csv" ]; then
    echo "ERROR: regional_anomalies.csv not found."
    echo "Please run data_processing_1.py first to generate required data."
    exit 1
fi

echo "[1/2] Running Python regime shift detection and extreme value analysis..."
echo "      (modeling_3.py)"
python scripts/modeling_3.py

if [ $? -eq 0 ]; then
    echo "✓ Python modeling complete"
    echo ""
else
    echo "✗ Python modeling failed"
    exit 1
fi

echo "[2/2] Running R extreme value analysis with extRemes package..."
echo "      (extreme_value_analysis_4.R)"
Rscript scripts/extreme_value_analysis_4.R

if [ $? -eq 0 ]; then
    echo "✓ R extreme value analysis complete"
    echo ""
else
    echo "✗ R extreme value analysis failed"
    exit 1
fi

echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - data_processed/regime_shift_results.csv (Python)"
echo "  - data_processed/extreme_value_results.csv (Python)"
echo "  - data_processed/extreme_value_results_R.csv (R)"
echo "  - figures/regime_shifts/ (diagnostic plots)"
echo "  - figures/extreme_value_diagnostics/ (Q-Q and return level plots)"
echo ""
echo "Review both Python and R results for comprehensive OSFI B-15 analysis."
echo "R analysis provides formal tests for non-stationarity in extremes."
