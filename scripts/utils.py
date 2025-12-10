"""
Utility functions and constants for data visualization across the project.
Centralizes styling, colors, and common plotting functions.
"""

import matplotlib.pyplot as plt

# ============================================================================
# CLIMATE VARIABLES CONFIGURATION
# ============================================================================


# Central configuration for all climate variables used in the project
# Each variable has metadata for consistent naming and display across scripts
def calculate_drought_accumulation(
    df,
    precip_col="total_precipitation_mm",
    temp_col="mean_temperature_c",
    rolling_window=3,
    precip_threshold_mm=150,
    temp_threshold_c=None,
):
    """
    Calculate drought accumulation score based on precipitation deficit.

    This function calculates a rolling precipitation sum and converts it to a
    drought deficit score (0-100), where:
    - 0 = No drought (adequate precipitation)
    - 100 = Severe drought (minimal precipitation)

    Optional temperature-gating ensures drought is only assessed during periods
    when it's meteorologically relevant (e.g., above freezing/snow season for
    wildfire risk).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with location, precipitation, and temperature data
    precip_col : str
        Column name for precipitation data (mm)
    temp_col : str
        Column name for temperature data (°C)
    rolling_window : int
        Number of months for rolling precipitation sum (default: 3)
    precip_threshold_mm : float
        Precipitation threshold for the rolling window (default: 150mm for 3 months)
    temp_threshold_c : float or None
        Minimum temperature for drought assessment (default: None = no temperature gating)
        If provided, drought score is set to 0 below this temperature threshold.

    Returns
    -------
    pd.Series
        Drought accumulation scores (0-100)

    Notes
    -----
    - If temp_threshold_c is None, drought is calculated year-round
    - If temp_threshold_c is provided, drought score is set to 0 below threshold
    - Score is calculated as: 100 * (1 - rolling_precip / threshold)
    - Clamped to [0, 100] range

    Examples
    --------
    >>> # No temperature gating (year-round drought assessment)
    >>> df['drought_score'] = calculate_drought_accumulation(df, temp_threshold_c=None)
    >>>
    >>> # Temperature-gated for fire season only (>5°C)
    >>> df['fire_drought'] = calculate_drought_accumulation(df, temp_threshold_c=5)
    """
    import numpy as np
    import pandas as pd

    drought_scores = []

    for location in df["location"].unique():
        mask = df["location"] == location
        precip_series = df.loc[mask, precip_col]
        temp_series = df.loc[mask, temp_col]

        # Calculate rolling sum
        rolling_precip = precip_series.rolling(
            window=rolling_window, min_periods=1
        ).sum()

        # Convert to deficit score with optional temperature gating
        location_scores = []
        for precip_sum, temp in zip(rolling_precip, temp_series):
            if pd.isna(precip_sum):
                location_scores.append(np.nan)
            elif temp_threshold_c is not None and (
                pd.isna(temp) or temp < temp_threshold_c
            ):
                # Temperature gating enabled: outside active season
                location_scores.append(0)
            else:
                # Calculate drought deficit
                deficit = max(
                    0, min(100, 100 - (precip_sum / precip_threshold_mm * 100))
                )
                location_scores.append(deficit)

        drought_scores.extend(location_scores)

    return pd.Series(drought_scores, index=df.index)


VARIABLES = {
    "temperature": {
        "long_name": "mean_temperature_c",
        "short_name": "temperature_c",
        "display_name": "Temperature",
        "units": "°C",
        "description": "Mean temperature",
    },
    "precipitation": {
        "long_name": "total_precipitation_mm",
        "short_name": "precipitation_mm",
        "display_name": "Precipitation",
        "units": "mm",
        "description": "Total precipitation",
    },
    "snow_on_ground": {
        "long_name": "snow_on_ground_last_day_cm",
        "short_name": "snow_on_ground_cm",
        "display_name": "Snow on Ground",
        "units": "cm",
        "description": "Snow on ground (last day)",
    },
    "heating_degrees": {
        "long_name": "mean_heating_days_c",
        "short_name": "heating_degrees_c",
        "display_name": "Heating Degree Days",
        "units": "°C",
        "description": "Mean heating degree days",
    },
    "cooling_degrees": {
        "long_name": "mean_cooling_days_c",
        "short_name": "cooling_degrees_c",
        "display_name": "Cooling Degree Days",
        "units": "°C",
        "description": "Mean cooling degree days",
    },
    "proxy_fwi": {
        "long_name": "proxy_fwi",
        "short_name": "proxy_fwi",
        "display_name": "Proxy Fire Weather Index",
        "units": "index (0-100)",
        "description": "Proxy Fire Weather Index",
    },
    "drought_accumulation": {
        "long_name": "drought_accumulation",
        "short_name": "drought",
        "display_name": "Drought Accumulation",
        "units": "index (0-100)",
        "description": "3-month precipitation deficit score (0=no drought, 100=severe drought)",
    },
}

# ============================================================================
# COLOR SCHEMES
# ============================================================================

# Decade color mapping - gradient from blue (1980s) to red (2020s)
# This creates a visual progression showing temporal change
DECADE_COLORS = {
    1980: "#3b4cc0",  # Blue
    1990: "#7f96d4",  # Light blue
    2000: "#b8b8dc",  # Purple-gray (middle)
    2010: "#e08074",  # Coral/salmon
    2020: "#d62728",  # Red
}

# Month abbreviations for x-axis labels
MONTH_LABELS = [
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
]

# ============================================================================
# FONT SETTINGS
# ============================================================================

FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 10

# ============================================================================
# FIGURE SETTINGS
# ============================================================================

FIGURE_SIZE_STANDARD = (10, 6)
FIGURE_SIZE_WIDE = (12, 6)
FIGURE_DPI = 100
GRID_ALPHA = 0.3

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================


def apply_standard_style(ax, title, xlabel, ylabel):
    """
    Apply standard styling to a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to style
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    """
    ax.set_title(title, fontsize=FONT_SIZE_TITLE)
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.grid(True, alpha=GRID_ALPHA)


def get_decade_color(year):
    """
    Get the color for a given year based on its decade.

    Parameters
    ----------
    year : int
        Year value

    Returns
    -------
    str
        Hex color code
    """
    decade = (year // 10) * 10
    return DECADE_COLORS.get(decade, "#7f7f7f")  # Default gray


def setup_month_axis(ax):
    """
    Configure x-axis for monthly data (1-12).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to configure
    """
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS)


def save_and_clear(fig, filepath):
    """
    Save figure to file and clear for next plot.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filepath : str
        Path to save the figure
    """
    plt.tight_layout()
    fig.savefig(filepath, dpi=FIGURE_DPI)
    plt.clf()
