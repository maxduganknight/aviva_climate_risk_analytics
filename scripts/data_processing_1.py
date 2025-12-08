"""
This script processes the raw csv data from ECCC, reformats it and saves the processed data
to a new csv file in data_processed/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent))
from utils import VARIABLES


def load_station_coords(station_inventory_path):
    """
    Load station coordinates from ECCC Station Inventory file.

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

    The input dataframe has a multi-level column structure:
    - Level 0: Column Labels header
    - Level 1: Year (1981, 1982, etc.)
    - Level 2: Month (1-12)
    - Level 3: Metric names (MEAN_TEMPERATURE, TOTAL_PRECIPITATION, etc.)

    Each year has 12 months × 5 metrics = 60 columns, followed by 5 yearly average columns.

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

    This is a simplified proxy of the Canadian Fire Weather Index System,
    adapted for monthly data without humidity or wind speed measurements.

    Components:
    1. Temperature Risk: Higher temps increase fire risk (exponential above 20°C)
    2. Precipitation Deficit: Lower precip increases fire risk
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

    # Temperature Risk Score (0-100)
    # Linear scale from 10°C to 30°C, capped at 100
    df["temp_risk"] = df["mean_temperature_c"].apply(
        lambda x: min(100, max(0, (x - 10) * 5)) if pd.notna(x) else np.nan
    )

    # Precipitation Risk Score (0-100)
    # Inverse relationship: less precip = higher risk
    # Scaled so 50mm = 0 risk, 0mm = 100 risk
    df["precip_risk"] = df["total_precipitation_mm"].apply(
        lambda x: max(0, 100 - (x * 2)) if pd.notna(x) else np.nan
    )

    # Drought Accumulation Score (0-100)
    # Rolling 3-month sum of precipitation deficit
    # Reference: 150mm over 3 months is adequate moisture
    df["drought_risk"] = np.nan
    for location in df["location"].unique():
        mask = df["location"] == location
        precip_series = df.loc[mask, "total_precipitation_mm"]

        # Calculate 3-month rolling sum
        rolling_precip = precip_series.rolling(window=3, min_periods=1).sum()

        # Convert to deficit score (150mm reference = 0 risk, 0mm = 100 risk)
        drought_score = rolling_precip.apply(
            lambda x: max(0, min(100, 100 - (x / 150 * 100))) if pd.notna(x) else np.nan
        )

        df.loc[mask, "drought_risk"] = drought_score.values

    # Combined Proxy FWI (weighted average)
    # Weights: temp (40%), precip (35%), drought (25%)
    df["proxy_fwi"] = (
        0.40 * df["temp_risk"] + 0.35 * df["precip_risk"] + 0.25 * df["drought_risk"]
    )

    # Clean up intermediate columns
    df = df.drop(columns=["temp_risk", "precip_risk", "drought_risk"])

    return df


def create_regional_aggregations(df):
    """
    Create regional quadrant aggregations (NE, NW, SE, SW Alberta).

    Divides locations into 4 quadrants based on median lat/lon and computes
    the mean of all available station values per quadrant, following the same
    methodology as Grand Total.

    Parameters
    ----------
    df : pd.DataFrame
        Long format dataframe with location coordinates

    Returns
    -------
    pd.DataFrame
        Original dataframe with 4 new regional aggregate locations added
    """
    # Calculate geographic center of station coverage for quadrant boundaries
    locations_with_coords = df[
        (df["location"] != "Grand Total") & df["latitude"].notna()
    ].copy()

    # Use geographic center of actual station coverage for equal-sized quadrants
    lat_min = locations_with_coords["latitude"].min()
    lat_max = locations_with_coords["latitude"].max()
    lon_min = locations_with_coords["longitude"].min()
    lon_max = locations_with_coords["longitude"].max()

    lat_center = (lat_min + lat_max) / 2
    lon_center = (lon_min + lon_max) / 2

    print(f"\nCreating regional quadrants:")
    print(
        f"  Station coverage: {lat_min:.2f}°N to {lat_max:.2f}°N, {lon_min:.2f}°W to {lon_max:.2f}°W"
    )
    print(f"  North/South boundary: {lat_center:.2f}°N (geographic center)")
    print(f"  East/West boundary: {lon_center:.2f}°W (geographic center)\n")

    # Assign quadrants to each location
    def assign_quadrant(row):
        if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
            return None
        if row["location"] == "Grand Total":
            return None

        if row["latitude"] >= lat_center:
            if row["longitude"] >= lon_center:
                return "NE Alberta"
            else:
                return "NW Alberta"
        else:
            if row["longitude"] >= lon_center:
                return "SE Alberta"
            else:
                return "SW Alberta"

    df["quadrant"] = df.apply(assign_quadrant, axis=1)

    # Variables to aggregate
    agg_vars = [
        "mean_temperature_c",
        "total_precipitation_mm",
        "snow_on_ground_last_day_cm",
        "mean_cooling_days_c",
        "mean_heating_days_c",
        "proxy_fwi",
    ]

    # Create aggregations for each quadrant
    regional_records = []

    for quadrant_name in ["NE Alberta", "NW Alberta", "SE Alberta", "SW Alberta"]:
        quadrant_data = df[df["quadrant"] == quadrant_name].copy()

        # Group by year and month, take mean of all stations
        for (year, month), group in quadrant_data.groupby(["year", "month"]):
            record = {
                "year": year,
                "month": month,
                "location": quadrant_name,
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

    # Remove temporary quadrant column
    df = df.drop(columns=["quadrant"])

    # Append regional aggregations to original dataframe
    df = pd.concat([df, regional_df], ignore_index=True)

    # Sort by location, year, month
    df = df.sort_values(["location", "year", "month"]).reset_index(drop=True)

    print(f"Added {len(regional_records)} regional aggregate records (4 quadrants)")

    return df


def calculate_regional_anomalies(df, baseline_years=(1981, 1996)):
    """
    Calculate monthly anomalies for regional aggregations relative to baseline period.

    Anomalies are calculated as the difference from the region's baseline mean
    for each month. Only includes regional aggregations (4 quadrants + Grand Total).

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
        "NE Alberta",
        "NW Alberta",
        "SE Alberta",
        "SW Alberta",
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
