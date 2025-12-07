# Aviva Climate Risk Analytics Exercise

This repo contains code, data, and reproducible analysis for assessing climate trends in Alberta from public Environment and Climate Change Canada (ECCC) data.
https://climate-change.canada.ca/climate-data/?utm_source=perplexity#/monthly-climate-summaries


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

1. Data Cleaning and Reformatting: data_processing_1.py restructures data into long, standardized format and joins lat, lon values for future geospatial visualization. 
2. Data Visualization: data_visualization_2.py creates plots and charts for initial exploration of data to identify trends to look into in further detail. It  produces plots to figures/ showing each variable over time across the whole province and time range, it produces a map of the province for each variable colour-coding 2020-2025 anomalies relative to a 1981-2011 baseline, and finally it will produce line plots comparing each location to to one another over time. 

-- TBD based on steps 1 and 2 -- 
3. Data Analysis: data_analysis_3.py performs statistical analysis on the data to identify significant trends and patterns. It produces tables and charts to summarize the results and provides insights into the climate trends in Alberta.
4. Data Modeling: data_modeling_4.py uses extreme value analysis to predict future climate trends based on historical data. It produces models and visualizations to show the accuracy of the predictions and provides insights into the potential impacts of climate change on Alberta.

## Data Source
Environment and Climate Change Canada (ECCC). 
https://climate-change.canada.ca/climate-data/?utm_source=perplexity#/monthly-climate-summaries.
