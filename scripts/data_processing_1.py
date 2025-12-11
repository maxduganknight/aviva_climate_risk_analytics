"""
This script processes the raw csv data from ECCC, reformats it and saves the processed data
to a new csv file in data_processed/long_df.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent))
from utils import VARIABLES, calculate_drought_accumulation


def load_station_coords(station_inventory_path):
    """
    Load station coordinates from ECCC Station Inventory file.
    Downloaded from https://collaboration.cmc.ec.gc.ca/cmc/climate/Get_More_Data_Plus_de_donnees/

    Parameters
    ----------
    station_inventory_path : str
        Path to the Station Inventory EN.csv file

    Returns
    -------
    dict
        Dictionary mapping station names to (latitude, longitude) tuples
    """
    df_stations = pd.read_csv(station_inventory_path, skiprows=3)
    alberta_stations = df_stations[df_stations["Province"] == "ALBERTA"]
    # Create lookup dictionary
    coord_lookup = {}
    for _, row in alberta_stations.iterrows():
        name = row["Name"]
        lat = row["Latitude (Decimal Degrees)"]
        lon = row["Longitude (Decimal Degrees)"]
        coord_lookup[name] = (lat, lon)

    return coord_lookup


def reformat_eccc_data_to_long(df, coord_lookup=None):
    """
    Reformat ECCC climate data from wide format to long format.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with multi-level columns read from ECCC CSV
    coord_lookup : dict, optional
        Dictionary mapping station names to (latitude, longitude) tuples

    Returns
    -------
    pd.DataFrame
        Long format dataframe with columns:
        - year: int
        - month: int
        - location: str
        - latitude: float
        - longitude: float
        - mean_temperature_c: float
        - total_precipitation_mm: float
        - snow_on_ground_last_day_cm: float
        - mean_heating_days_c: float
        - mean_cooling_days_c: float
    """
    # Extract location names (first column)
    locations = df.iloc[:, 0]

    # Build list of records
    records = []

    for loc_idx, location in enumerate(locations):
        col_idx = 1  # Start after location column
        current_year = None

        # Process each column
        while col_idx < len(df.columns):
            year_val = df.columns[col_idx][1]
            month_val = df.columns[col_idx][2]

            # Check if this is a new year marker (year appears in level 1)
            if not str(year_val).startswith("Unnamed"):
                try:
                    year_int = int(year_val)
                    current_year = year_int
                except (ValueError, TypeError):
                    # Skip yearly averages and other non-year values
                    col_idx += 1
                    continue

            # Check if we have a valid month
            if current_year is not None:
                try:
                    month_int = int(month_val)
                except (ValueError, TypeError):
                    # Not a valid month, move to next column
                    col_idx += 1
                    continue

                # Extract the 5 metrics for this year-month combination
                mean_temp = df.iloc[loc_idx, col_idx]
                total_precip = (
                    df.iloc[loc_idx, col_idx + 1]
                    if col_idx + 1 < len(df.columns)
                    else None
                )
                snow_ground = (
                    df.iloc[loc_idx, col_idx + 2]
                    if col_idx + 2 < len(df.columns)
                    else None
                )
                cooling_days = (
                    df.iloc[loc_idx, col_idx + 3]
                    if col_idx + 3 < len(df.columns)
                    else None
                )
                heating_days = (
                    df.iloc[loc_idx, col_idx + 4]
                    if col_idx + 4 < len(df.columns)
                    else None
                )

                # Look up coordinates if available
                lat, lon = None, None
                if coord_lookup and location in coord_lookup:
                    lat, lon = coord_lookup[location]

                # Add record
                records.append(
                    {
                        "year": current_year,
                        "month": month_int,
                        "location": location,
                        "latitude": lat,
                        "longitude": lon,
                        "mean_temperature_c": mean_temp,
                        "total_precipitation_mm": total_precip,
                        "snow_on_ground_last_day_cm": snow_ground,
                        "mean_cooling_days_c": cooling_days,
                        "mean_heating_days_c": heating_days,
                    }
                )

                # Move to next month (skip 5 metric columns)
                col_idx += 5
            else:
                col_idx += 1

    # Create long format dataframe
    long_df = pd.DataFrame(records)

    # Sort by location, year, month
    long_df = long_df.sort_values(["location", "year", "month"]).reset_index(drop=True)

    return long_df


def calculate_proxy_fwi(df):
    """
    Calculate a proxy Fire Weather Index based on temperature and precipitation.

    This is an adaptation of the Canadian Fire Weather Index System (FWI),
    modified for monthly data without relative humidity or wind speed measurements.

    Key Adaptations for Monthly Data:
    1. Temperature-dependent precipitation thresholds (accounts for evapotranspiration)
    2. Fire season threshold at 5°C (Canadian FWI standard)
    3. Drought accumulation using 3-month rolling precipitation deficit

    Components:
    1. Temperature Risk: Accounts for fire season thresholds and heat effects
    2. Precipitation Deficit: Temperature-adjusted moisture requirements
    3. Drought Accumulation: Rolling 3-month precipitation deficit

    Parameters
    ----------
    df : pd.DataFrame
        Long format dataframe with columns: location, year, month,
        mean_temperature_c, total_precipitation_mm

    Returns
    -------
    pd.DataFrame
        Input dataframe with new 'proxy_fwi' column added (scale 0-100)
    """
    df = df.copy()

    # =========================================================================
    # 1. TEMPERATURE RISK SCORE (0-100)
    # =========================================================================
    # Based on Canadian FWI fire season thresholds:
    # - Fire season starts when temp > 12°C for 3 consecutive days
    # - Fire season ends when temp < 5°C for 3 consecutive days
    # Source: https://climate-adapt.eea.europa.eu/en/metadata/indicators/fire-weather-index

    def calculate_temp_risk(temp):
        if pd.isna(temp):
            return np.nan
        elif temp < 5:
            # Below 5°C: Outside fire season, minimal risk
            return 0
        elif temp < 12:
            # 5-12°C: Transition period, linear ramp-up
            # 5°C = 0 risk, 12°C = 10 risk
            return (temp - 5) * (10 / 7)
        else:
            # Above 12°C: Active fire season
            # Linear scale: 12°C = 10 risk, 30°C = 100 risk
            # Formula: (temp - 12) * 5 + 10
            return min(100, (temp - 12) * 5 + 10)

    df["temp_risk"] = df["mean_temperature_c"].apply(calculate_temp_risk)

    # =========================================================================
    # 2. PRECIPITATION DEFICIT RISK SCORE (0-100)
    # =========================================================================

    # Constants derived from FWI principles and Alberta climate:
    BASE_PRECIP_MM = 25  # Base monthly moisture requirement (mm)
    EVAP_FACTOR = 3.5  # Additional mm needed per °C above 12°C
    FIRE_SEASON_TEMP = 12  # Temperature threshold for active fire season (°C)
    FIRE_SEASON_TEMP_MIN = 5  # Temperature below which fire risk is negligible

    def calculate_precip_risk(row):
        temp = row["mean_temperature_c"]
        precip = row["total_precipitation_mm"]

        if pd.isna(temp) or pd.isna(precip):
            return np.nan

        # Outside fire season: precipitation deficit is irrelevant
        if temp < FIRE_SEASON_TEMP_MIN:
            return 0

        # Calculate temperature-adjusted required precipitation
        # Transition period (5-12°C): use base requirement
        if temp < FIRE_SEASON_TEMP:
            required_precip = BASE_PRECIP_MM
        else:
            # Active fire season (>12°C): add evapotranspiration losses
            evap_loss = EVAP_FACTOR * (temp - FIRE_SEASON_TEMP)
            required_precip = BASE_PRECIP_MM + evap_loss

        # Calculate deficit as percentage of required moisture
        # 100 = complete deficit (0mm when 50mm needed)
        # 0 = surplus (actual >= required)
        deficit_ratio = max(0, (required_precip - precip) / required_precip)
        return 100 * deficit_ratio

    df["precip_risk"] = df.apply(calculate_precip_risk, axis=1)

    # =========================================================================
    # 3. DROUGHT ACCUMULATION SCORE (0-100)
    # =========================================================================
    # Tracks cumulative moisture deficit over 3 months, analogous to FWI's
    # Drought Code (DC) which represents deep soil moisture in compact layers.
    #
    # Reference threshold: 150mm over 3 months is adequate for Alberta
    # - Based on average Alberta summer precip: ~60-70mm/month
    # - 3-month period captures antecedent moisture conditions
    # Source: FWI system uses cumulative moisture tracking across multiple timescales

    DROUGHT_THRESHOLD_MM = 150  # 3-month precipitation threshold
    FIRE_SEASON_TEMP_MIN = 5  # Temperature below which fire risk is negligible

    # Use modular drought calculation function
    df["drought_risk"] = calculate_drought_accumulation(
        df,
        precip_col="total_precipitation_mm",
        temp_col="mean_temperature_c",
        rolling_window=3,
        precip_threshold_mm=DROUGHT_THRESHOLD_MM,
        temp_threshold_c=FIRE_SEASON_TEMP_MIN,
    )

    # =========================================================================
    # 4. COMBINED PROXY FWI (0-100)
    # =========================================================================

    WEIGHT_TEMP = 0.5
    WEIGHT_PRECIP = 0.25
    WEIGHT_DROUGHT = 0.25

    df["proxy_fwi"] = (
        WEIGHT_TEMP * df["temp_risk"]
        + WEIGHT_PRECIP * df["precip_risk"]
        + WEIGHT_DROUGHT * df["drought_risk"]
    )

    # Clean up intermediate columns
    df = df.drop(columns=["temp_risk", "precip_risk", "drought_risk"])

    return df


def create_regional_aggregations(df):
    """
    Create regional cluster aggregations (North, Southwest, Southeast Clusters).

    Uses k-means clustering to divide weather stations into 3 natural geographic
    clusters based on their latitude/longitude coordinates. Computes the mean of
    all available station values per cluster, following the same methodology as Grand Total.

    Parameters
    ----------
    df : pd.DataFrame
        Long format dataframe with location coordinates

    Returns
    -------
    pd.DataFrame
        Original dataframe with 3 new regional aggregate locations added
    """
    from sklearn.cluster import KMeans

    # Get unique stations with coordinates (exclude Grand Total)
    stations = df[
        (df["location"] != "Grand Total")
        & df["latitude"].notna()
        & ~df["location"].str.contains("Alberta", na=False)
    ][["location", "latitude", "longitude"]].drop_duplicates()

    # Prepare coordinates for clustering
    coords = stations[["latitude", "longitude"]].values

    # K-means clustering with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    stations["cluster"] = kmeans.fit_predict(coords)

    # Identify clusters based on geographic position
    # Calculate mean latitude for each cluster to assign names
    cluster_stats = (
        stations.groupby("cluster")
        .agg({"latitude": "mean", "longitude": "mean"})
        .reset_index()
    )

    # Sort by latitude (descending) to identify North vs South
    cluster_stats = cluster_stats.sort_values("latitude", ascending=False)

    # Northernmost cluster = North
    north_cluster = cluster_stats.iloc[0]["cluster"]

    # For southern clusters, distinguish by longitude
    southern_clusters = cluster_stats.iloc[1:]
    southern_clusters = southern_clusters.sort_values("longitude", ascending=True)

    # Western (lower longitude value) = Southwest, Eastern = Southeast
    southwest_cluster = southern_clusters.iloc[0]["cluster"]
    southeast_cluster = southern_clusters.iloc[1]["cluster"]

    # Map cluster numbers to region names
    cluster_to_region = {
        north_cluster: "North Cluster",
        southwest_cluster: "Southwest Cluster",
        southeast_cluster: "Southeast Cluster",
    }

    stations["region"] = stations["cluster"].map(cluster_to_region)

    # Print cluster assignments
    print(f"\nCreating regional clusters (k-means, k=3):")
    for region in cluster_to_region:
        region_stations = stations[stations["region"] == region]

    # Create a mapping from location to region for the full dataframe
    location_to_region = dict(zip(stations["location"], stations["region"]))

    # Assign regions to all rows in df
    df["region"] = df["location"].map(location_to_region)

    # Variables to aggregate
    agg_vars = [
        "mean_temperature_c",
        "total_precipitation_mm",
        "snow_on_ground_last_day_cm",
        "mean_cooling_days_c",
        "mean_heating_days_c",
        "proxy_fwi",
        "drought_accumulation",
    ]

    # Create aggregations for each region
    regional_records = []

    for region_name in cluster_to_region.values():
        region_data = df[df["region"] == region_name].copy()

        # Group by year and month, take mean of all stations
        for (year, month), group in region_data.groupby(["year", "month"]):
            record = {
                "year": year,
                "month": month,
                "location": region_name,
                "latitude": group[
                    "latitude"
                ].mean(),  # Average lat/lon of contributing stations
                "longitude": group["longitude"].mean(),
            }

            # Compute mean for each variable
            for var in agg_vars:
                record[var] = group[var].mean()  # Mean automatically ignores NaN

            regional_records.append(record)

    # Create dataframe from regional records
    regional_df = pd.DataFrame(regional_records)

    # Remove temporary region column
    df = df.drop(columns=["region"])

    # Append regional aggregations to original dataframe
    df = pd.concat([df, regional_df], ignore_index=True)

    # Sort by location, year, month
    df = df.sort_values(["location", "year", "month"]).reset_index(drop=True)

    print(f"\nAdded {len(regional_records)} regional aggregate records (3 clusters)")
    return df


def calculate_regional_anomalies(df, baseline_years=(1981, 1996)):
    """
    Calculate monthly anomalies for regional aggregations relative to baseline period.

    Anomalies are calculated as the difference from the region's baseline mean
    for each month. Only includes regional aggregations (3 clusters + Grand Total).

    Parameters
    ----------
    df : pd.DataFrame
        Long format dataframe with regional aggregations
    baseline_years : tuple of int, optional
        Start and end years (inclusive) for baseline period. Default (1981, 1996)

    Returns
    -------
    pd.DataFrame
        Dataframe with columns: year, month, region, and for each variable:
        {variable}_baseline, {variable}_anomaly
    """
    # Define regional locations to include
    regional_locations = [
        "Grand Total",
        "North Cluster",
        "Southwest Cluster",
        "Southeast Cluster",
    ]

    # Filter for regional aggregations only
    regional_df = df[df["location"].isin(regional_locations)].copy()

    # Variables to calculate anomalies for (using VARIABLES dict)
    variables = [VARIABLES[var]["long_name"] for var in VARIABLES.keys()]

    # Calculate baseline means for each region-month combination
    baseline_df = regional_df[
        (regional_df["year"] >= baseline_years[0])
        & (regional_df["year"] <= baseline_years[1])
    ].copy()

    baseline_means = (
        baseline_df.groupby(["location", "month"])[variables].mean().reset_index()
    )

    # Merge baseline means with full dataset
    regional_df = regional_df.merge(
        baseline_means, on=["location", "month"], suffixes=("", "_baseline")
    )

    # Calculate anomalies
    anomaly_records = []

    for _, row in regional_df.iterrows():
        record = {
            "year": row["year"],
            "month": row["month"],
            "region": row["location"],
        }

        # Add baseline and anomaly for each variable
        for var in variables:
            baseline_val = row[f"{var}_baseline"]
            current_val = row[var]

            # Map long name to short name using VARIABLES dict
            var_short = None
            for var_key, var_config in VARIABLES.items():
                if var_config["long_name"] == var:
                    var_short = var_config["short_name"]
                    break

            # Fallback if variable not found in dict
            if var_short is None:
                var_short = var

            record[f"{var_short}_baseline"] = baseline_val

            # Calculate anomaly (handles NaN gracefully)
            if pd.notna(current_val) and pd.notna(baseline_val):
                record[f"{var_short}_anomaly"] = current_val - baseline_val
            else:
                record[f"{var_short}_anomaly"] = np.nan

        anomaly_records.append(record)

    # Create output dataframe
    anomaly_df = pd.DataFrame(anomaly_records)

    # Sort by region, year, month
    anomaly_df = anomaly_df.sort_values(["region", "year", "month"]).reset_index(
        drop=True
    )

    return anomaly_df


def analyze_data_completeness(df):
    """
    Analyze data completeness by location and year.

    Prints summary statistics showing:
    - How many years each location is missing all variables
    - Which locations have complete data
    - Year ranges for each location

    Parameters
    ----------
    df : pd.DataFrame
        Long format dataframe with climate variables
    """
    print("\n" + "=" * 80)
    print("DATA COMPLETENESS ANALYSIS")
    print("=" * 80 + "\n")

    # Define key variables to check (excluding lat/lon)
    key_vars = [
        "mean_temperature_c",
        "total_precipitation_mm",
        "snow_on_ground_last_day_cm",
        "mean_cooling_days_c",
        "mean_heating_days_c",
        "proxy_fwi",
    ]

    results = []

    for location in sorted(df["location"].unique()):
        loc_data = df[df["location"] == location].copy()

        # Count rows where ALL key variables are missing
        all_missing_mask = loc_data[key_vars].isna().all(axis=1)
        years_all_missing = (
            all_missing_mask.sum() // 12
        )  # Approximate years (12 months/year)

        # Get year range
        years = loc_data["year"].unique()
        year_range = f"{years.min()}-{years.max()}"
        total_years = len(years)

        # Count how many variables have any data
        vars_with_data = (~loc_data[key_vars].isna().all(axis=0)).sum()

        results.append(
            {
                "location": location,
                "year_range": year_range,
                "total_years": total_years,
                "years_all_missing": years_all_missing,
                "pct_missing": f"{(years_all_missing / total_years * 100):.1f}%",
                "vars_with_data": f"{vars_with_data}/{len(key_vars)}",
            }
        )

    # Create summary dataframe
    summary_df = pd.DataFrame(results)

    # Sort by years_all_missing (descending) to see worst offenders first
    summary_df = summary_df.sort_values("years_all_missing", ascending=False)

    print("SUMMARY: Years Missing All Variables by Location")
    print("-" * 80)
    print(summary_df.to_string(index=False))
    print("\n")

    # Print overall statistics
    total_locations = len(summary_df)
    complete_locations = (summary_df["years_all_missing"] == 0).sum()
    partial_locations = (
        (summary_df["years_all_missing"] > 0)
        & (summary_df["years_all_missing"] < summary_df["total_years"])
    ).sum()
    empty_locations = (
        summary_df["years_all_missing"] == summary_df["total_years"]
    ).sum()

    print("OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total locations: {total_locations}")
    print(f"Locations with complete data (0 missing years): {complete_locations}")
    print(f"Locations with partial data: {partial_locations}")
    print(f"Locations with NO data (all years missing): {empty_locations}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    raw_file_path = "data_raw/GoC climate-monthly sample data.csv"
    station_inventory_path = "data_raw/Station Inventory EN.csv"

    # Load data and coordinates
    df = pd.read_csv(raw_file_path, header=[0, 1, 2, 3])
    coord_lookup = load_station_coords(station_inventory_path)

    # Reformat with coordinates
    long_df = reformat_eccc_data_to_long(df, coord_lookup)

    # Calculate proxy FWI
    long_df = calculate_proxy_fwi(long_df)

    # Calculate standalone drought accumulation score (3-month deficit)
    # No temperature-gating for drought analysis (year-round assessment)
    print("\nCalculating drought accumulation score...")
    long_df["drought_accumulation"] = calculate_drought_accumulation(
        long_df,
        precip_col="total_precipitation_mm",
        temp_col="mean_temperature_c",
        rolling_window=3,
        precip_threshold_mm=150,
        temp_threshold_c=None,  # No temp gating - assess drought year-round
    )
    print(
        f"   ✓ Drought accumulation calculated (range: {long_df['drought_accumulation'].min():.1f} - {long_df['drought_accumulation'].max():.1f})"
    )

    # Create regional quadrant aggregations
    long_df = create_regional_aggregations(long_df)

    # Analyze data completeness (including regional aggregations)
    analyze_data_completeness(long_df)

    # Calculate regional anomalies
    anomaly_df = calculate_regional_anomalies(long_df, baseline_years=(1981, 1996))

    # Save outputs to CSV
    long_df.to_csv("data_processed/long_df.csv", index=False)
    anomaly_df.to_csv("data_processed/regional_anomalies.csv", index=False)

    print(
        f"\nSaved {len(anomaly_df)} regional anomaly records to data_processed/regional_anomalies.csv"
    )
