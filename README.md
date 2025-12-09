# Aviva Climate Risk Analytics Exercise

This repo contains code, data, and reproducible analysis for assessing climate trends in Alberta from public Environment and Climate Change Canada (ECCC) data.

## Dataset

Timeframe: January, 1981 - November, 2025
Spatial resolution: 29 named locations
Temporal resolution: monthly
Variables: 
  - mean temperature (°C)
  - total precipitation (mm)
  - snow on ground last day (cm)
  - mean heating degree days (number of degrees below 18°C)
  - mean cooling degree days (number of degrees above 18°C)

## Analysis

1. **Data Cleaning and Reformatting** (`scripts/data_processing_1.py`): 
   - Restructures data into long, standardized format
   - Joins lat, lon values for future geospatial visualization
   - Calculates proxy Fire Weather Index (FWI) with temperature-gated risk components
   - Produces `data_processed/monthly_climate_data.csv`

2. **Data Visualization** (`scripts/data_visualization_2.py`): 
   - Creates time-series heatmaps for regional anomalies across all variables
   - Produces provincial maps showing 2020-2025 anomalies relative to 1981-2011 baseline
   - Creates line plots comparing locations over time
   - Generates September FWI anomaly comparison by region
   - Outputs to `figures/` directory

3. **Statistical Modeling** (`scripts/modeling_3.py`): 
   - Performs Bayesian change point detection to identify regime shifts
   - Conducts Extreme Value Analysis (EVA) using Generalized Extreme Value (GEV) distributions
   - Calculates return periods (10, 25, 50, 100 years) with bootstrap confidence intervals
   - Includes sample size warnings for n < 20
   - Produces diagnostic plots (Q-Q plots, return level plots)
   - Outputs results to `data_processed/regime_shift_results.csv` and `data_processed/extreme_value_results.csv`

4. **Advanced Extreme Value Analysis** (`scripts/extreme_value_analysis_4.R`):
   - Uses R's extRemes package for non-stationary GEV modeling
   - Tests for temporal trends in location and scale parameters
   - Compares stationary vs non-stationary models using likelihood ratio tests
   - Calculates return levels for baseline (1981-1996) and current (2009-2024) periods
   - Provides extRemes-specific diagnostic plots
   - Outputs results to `data_processed/extreme_value_results_R.csv`

### OSFI B-15 Climate Risk Focus

These analyses support climate scenario analysis under OSFI Guideline B-15 requirements by:
- Quantifying changes in climate extremes over time (EVA)
- Identifying structural breaks in climate patterns (regime shifts)
- Calculating probability of extreme events for risk assessment (return periods)
- Providing uncertainty quantification (confidence intervals)
- Flagging low-confidence results (sample size warnings)

## Data Source
Environment and Climate Change Canada (ECCC). 
https://climate-change.canada.ca/climate-data/?utm_source=perplexity#/monthly-climate-summaries.

## Setup

### Python Environment

1. Activate virtual environment: `source venv/bin/activate`
2. Install Python dependencies: `pip install -r requirements.txt`
3. Run the analysis pipeline:
   ```bash
   python scripts/data_processing_1.py
   python scripts/data_visualization_2.py
   python scripts/modeling_3.py
   ```

### R Environment (for Advanced EVA)

1. Install R packages (one-time setup):
   ```r
   install.packages(c("extRemes", "dplyr", "tidyr", "ggplot2", "lubridate"))
   ```

2. Run the R extreme value analysis:
   ```bash
   Rscript scripts/extreme_value_analysis_4.R
   ```

### Running Complete Modeling Pipeline

To run both Python and R extreme value analyses:
```bash
bash scripts/run_all_modeling.sh
```

### Python vs R Extreme Value Analysis

**Python (`modeling_3.py`)**:
- Uses scipy.stats.genextreme for GEV fitting
- Parametric bootstrap for confidence intervals (200 iterations)
- Custom diagnostic plots (Q-Q, return level)
- Integrated with full Python pipeline

**R (`extreme_value_analysis_4.R`)**:
- Uses extRemes package (industry standard for EVA)
- Non-stationary modeling with time-varying location/scale parameters
- Likelihood ratio tests to compare stationary vs non-stationary models
- extRemes built-in confidence intervals and diagnostics
- Better suited for testing temporal trends in extremes

**Recommendation**: Run both analyses. Use Python for integrated workflow and R for publication-quality non-stationary EVA with formal hypothesis testing.
