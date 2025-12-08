"""
Utility functions and constants for data visualization across the project.
Centralizes styling, colors, and common plotting functions.
"""

import matplotlib.pyplot as plt

# ============================================================================
# COLOR SCHEMES
# ============================================================================

# Decade color mapping for consistent visualization across all plots
DECADE_COLORS = {
    1980: "#1f77b4",  # Blue
    1990: "#ff7f0e",  # Orange
    2000: "#2ca02c",  # Green
    2010: "#d62728",  # Red
    2020: "#9467bd",  # Purple
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
