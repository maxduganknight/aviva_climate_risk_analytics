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
