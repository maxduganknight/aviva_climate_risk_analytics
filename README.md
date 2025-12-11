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
   - Regional clustering to identify spatial patterns
   - Produces `data_processed/long_df.csv`

2. **Data Visualization** (`scripts/data_visualization_2.py`): 
   - Initial data exploration and visualization
   - Creates line plots showing decadal trends by month for each variable
   - Outputs to `figures/` directory

3. **Regime Shift Analysis** (`scripts/regime_shift_analysis_3.py`): 
   - Performs Bayesian change point detection to identify regime shifts
   - Outputs results to `data_processed/regime_shift_results.csv`

4. **Extreme Value Analysis** (`scripts/extreme_value_analysis_4.R`):
   - Uses R's extRemes package for non-stationary GEV modeling
   - Tests for temporal trends in location and scale parameters
   - Compares stationary vs non-stationary models using likelihood ratio tests
   - Calculates return levels for baseline (1981-1996) and current (2009-2024) periods
   - Provides extRemes-specific diagnostic plots
   - Outputs results to `data_processed/extreme_value_results_R.csv`

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
   python scripts/regime_shift_analysis_3.py
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
