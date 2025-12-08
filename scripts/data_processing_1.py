"""
This script processes the raw csv data from ECCC, reformats it and saves the processed data
to a new csv file in data_processed/
"""

import numpy as np
import pandas as pd


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


if __name__ == "__main__":
    raw_file_path = "data_raw/GoC climate-monthly sample data.csv"
    station_inventory_path = "data_raw/Station Inventory EN.csv"

    # Load data and coordinates
    df = pd.read_csv(raw_file_path, header=[0, 1, 2, 3])
    coord_lookup = load_station_coords(station_inventory_path)

    # Reformat with coordinates
    long_df = reformat_eccc_data_to_long(df, coord_lookup)

    # Save long format dataframe to CSV
    long_df.to_csv("data_processed/long_df.csv", index=False)
