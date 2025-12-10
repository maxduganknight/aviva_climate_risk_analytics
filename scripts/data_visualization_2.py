import os

import geopandas as gpd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle
from utils import (
    DECADE_COLORS,
    FIGURE_SIZE_WIDE,
    VARIABLES,
    apply_standard_style,
    save_and_clear,
    setup_month_axis,
)


def add_aviva_branding(fig, ax, caption=""):
    """
    Add Aviva logo and caption to the plot.

    Parameters:
    -----------
    fig : matplotlib figure
        The figure to add branding to
    ax : matplotlib axis
        The axis object
    caption : str, optional
        Caption text to display below x-axis
    """
    # Add caption at bottom left, aligned with plot area
    if caption:
        # Use textwrap to enforce line breaks if caption is too long
        import textwrap

        wrapped_caption = "\n".join(textwrap.wrap(caption, width=160))

        fig.text(
            0.052,
            0.01,
            wrapped_caption,
            fontsize=9,
            color="#666666",
            ha="left",
            va="bottom",
            style="italic",
        )

    # Add Aviva logo at bottom right, aligned with plot area
    logo_path = "design/aviva_logo.jpeg"
    if os.path.exists(logo_path):
        logo = mpimg.imread(logo_path)
        imagebox = OffsetImage(logo, zoom=0.1)  # Smaller logo

        # Position to align with right edge of chart area, below x-axis labels
        ab = AnnotationBbox(
            imagebox,
            (0.966, 0.01),
            xycoords="figure fraction",
            frameon=False,
            box_alignment=(1.0, 0.0),
        )
        ax.add_artist(ab)


def plot_bars(x_values, y_values, title):
    plt.bar(x_values, y_values)
    plt.title(title)
    return plt


def plot_line(x_values, y_values, title):
    plt.plot(x_values, y_values)
    plt.title(title)
    return plt


def plot_scatter(x_values, y_values, title):
    plt.scatter(x_values, y_values)
    plt.title(title)
    return plt


def plot_regional_anomaly_heatmap(
    df,
    anomaly_variable,
    title,
    cmap="RdBu_r",
    figsize=(16, 6),
    vmin=None,
    vmax=None,
    year_step=5,
):
    """
    Create a time-series heatmap showing regional anomalies over time.

    Parameters
    ----------
    df : pd.DataFrame
        Anomaly dataframe with columns: year, month, region, {variable}_anomaly
    anomaly_variable : str
        Name of anomaly variable to plot (e.g., 'temperature_c_anomaly')
    title : str
        Plot title
    cmap : str, optional
        Colormap name. Default 'RdBu_r' (red=positive, blue=negative)
    figsize : tuple, optional
        Figure size (width, height). Default (16, 6)
    vmin : float, optional
        Minimum value for colormap. If None, uses symmetric range around zero
    vmax : float, optional
        Maximum value for colormap. If None, uses symmetric range around zero
    year_step : int, optional
        Step size for year labels on x-axis. Default 5

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Create a pivot table: regions as rows, time as columns
    df_copy = df.copy()
    df_copy["year_month"] = (
        df_copy["year"].astype(str) + "-" + df_copy["month"].astype(str).str.zfill(2)
    )

    # Define region order (spatial organization: NW, NE, SW, SE, Grand Total)
    region_order = [
        "NW Alberta",
        "NE Alberta",
        "SW Alberta",
        "SE Alberta",
        "Grand Total",
    ]

    # Filter to only regions that exist in the data
    region_order = [r for r in region_order if r in df_copy["region"].unique()]

    # Create pivot table
    heatmap_data = df_copy.pivot_table(
        values=anomaly_variable, index="region", columns="year_month", aggfunc="mean"
    )

    # Reorder regions
    heatmap_data = heatmap_data.reindex(region_order)

    # Determine color scale (symmetric around zero for diverging colormap)
    if vmin is None or vmax is None:
        max_abs = np.nanmax(np.abs(heatmap_data.values))
        vmin = -max_abs if vmin is None else vmin
        vmax = max_abs if vmax is None else vmax

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(
        heatmap_data.values,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Set y-axis (regions)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index, fontsize=11)

    # Set x-axis (years) - show every N years
    year_months = heatmap_data.columns
    years = [ym.split("-")[0] for ym in year_months]
    months = [ym.split("-")[1] for ym in year_months]

    # Find indices where year changes and is divisible by year_step
    tick_positions = []
    tick_labels = []
    prev_year = None

    for i, (year, month) in enumerate(zip(years, months)):
        if year != prev_year and int(year) % year_step == 0:
            tick_positions.append(i)
            tick_labels.append(year)
            prev_year = year

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=10)

    # Labels and title
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Region", fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, fraction=0.046)
    cbar.ax.tick_params(labelsize=10)

    # Add zero line to colorbar if it's a diverging scale
    if vmin < 0 < vmax:
        cbar.ax.axhline(y=0.5, color="black", linewidth=0.5, linestyle="--", alpha=0.3)

    plt.tight_layout()

    return fig


def plot_each_decade_by_month(df, y_variable, title, ylabel, caption=None):
    """
    Plot each decade as a separate line showing average monthly values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns: year, month, and the y_variable
    y_variable : str
        Column name to plot on y-axis
    title : str
        Plot title
    ylabel : str
        Y-axis label
    caption : str, optional
        Caption text to display below x-axis

    Returns
    -------
    matplotlib.pyplot
        Plot object ready to save
    """
    # Add decade column
    df_copy = df.copy()
    df_copy["decade"] = (df_copy["year"] // 10) * 10

    # Calculate average by decade and month
    monthly_data = df_copy.pivot_table(
        values=y_variable, index="month", columns="decade", aggfunc="mean"
    )

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    # Plot each decade
    for decade in monthly_data.columns:
        color = DECADE_COLORS.get(decade, "#7f7f7f")
        ax.plot(
            monthly_data.index,
            monthly_data[decade],
            color=color,
            linewidth=2.5,
            label=f"{int(decade)}s",
        )

    # Apply standard styling
    apply_standard_style(ax, title, "Month", ylabel)
    setup_month_axis(ax)

    # Reverse legend order (2020s on top, 1980s on bottom)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title="Decade", loc="best")

    # Make room for caption and logo at bottom
    plt.subplots_adjust(bottom=0.15, left=0.08, right=0.95)

    # Add Aviva branding (caption and logo)
    add_aviva_branding(fig, ax, caption=caption)

    return fig


def plot_alberta_totals():
    """
    Generate standardized time-series plots for all variables.

    Creates two types of plots for each variable in VARIABLES:
    1. Annual time series (bar or line plot)
    2. Monthly by decade (line plot showing each decade)
    """
    totals_df = df[df["location"] == "Grand Total"]

    print("\nGenerating Alberta total visualizations...")

    # Configuration for special handling
    aggregation_methods = {
        "temperature": "mean",
        "precipitation": "sum",
        "snow_on_ground": "sum",
        "heating_degrees": "sum",
        "cooling_degrees": "sum",
        "proxy_fwi": "mean",
        "drought_accumulation": "mean",
    }

    plot_types = {
        "temperature": "bar",
        "precipitation": "bar",
        "snow_on_ground": "bar",
        "heating_degrees": "bar",
        "cooling_degrees": "bar",
        "proxy_fwi": "line",
        "drought_accumulation": "line",
    }

    captions = {
        "proxy_fwi": "Note: Proxy FWI is an estimate of the Canadian Forest Fire Weather Index (FWI) System derived from temperature, precipitation, and drought. It is temperature-gated (values below 5°C set to zero).",
        "drought_accumulation": "Note: Drought accumulation is a 3-month precipitation deficit score (0=no drought, 100=severe drought). Assessed year-round without temperature-gating.",
        "cooling_degrees": "Note: Cooling degree days are calculated as the sum of daily temperatures above 18°C.",
        "heating_degrees": "Note: Heating degree days are calculated as the sum of daily temperatures below 18°C.",
    }

    # Loop through all variables
    for var_key, var_config in VARIABLES.items():
        var_name = var_config["long_name"]
        display_name = var_config["display_name"]
        units = var_config["units"]

        if var_name not in totals_df.columns:
            print(f"  ⚠ Skipping {display_name}: column '{var_name}' not found")
            continue

        # Get aggregation method
        agg_method = aggregation_methods.get(var_key, "mean")
        plot_type = plot_types.get(var_key, "bar")
        caption = captions.get(var_key, "")

        # 1. Annual time series
        annual_data = totals_df.groupby("year")[var_name].agg(agg_method)

        if plot_type == "line":
            fig = plot_line(
                annual_data.index, annual_data.values, f"Alberta Annual {display_name}"
            )
        else:
            fig = plot_bars(
                annual_data.index, annual_data.values, f"Alberta Annual {display_name}"
            )

        plt.xlabel("Year")
        plt.ylabel(f"{display_name} ({units})")
        filename = f"figures/alberta_annual_{var_key}.png"
        fig.savefig(filename)
        plt.clf()
        print(f"  ✓ {filename}")

        # 2. Monthly by decade
        fig = plot_each_decade_by_month(
            totals_df,
            var_name,
            f"{display_name} by Decade",
            f"{display_name} ({units})",
            caption=caption,
        )
        filename = f"figures/alberta_monthly_{var_key}_by_decade.png"
        save_and_clear(fig, filename)
        print(f"  ✓ {filename}")


def plot_regional_anomaly_heatmaps():
    """
    Generate time-series heatmaps for all regional anomaly variables.

    Creates one heatmap per variable showing how anomalies vary across
    regions and time.
    """
    print("\nGenerating regional anomaly heatmaps...")

    # Load regional anomalies data
    anomaly_df = pd.read_csv("data_processed/regional_anomalies.csv")

    # Build variables list from VARIABLES dict
    variables = []
    for var_key, var_config in VARIABLES.items():
        anomaly_column = f"{var_config['short_name']}_anomaly"

        # Only add if column exists in the dataframe
        if anomaly_column in anomaly_df.columns:
            variables.append(
                {
                    "column": anomaly_column,
                    "title": f"Regional {var_config['display_name']} Anomalies ({var_config['units']})\nRelative to 1981-1996 Baseline",
                    "filename": f"regional_{var_key}_anomaly_heatmap.png",
                    "display_name": var_config["display_name"],
                }
            )
        else:
            print(
                f"  ⚠ Skipping {var_config['display_name']}: anomaly column '{anomaly_column}' not found"
            )

    # Generate heatmap for each variable
    for var_config in variables:
        fig = plot_regional_anomaly_heatmap(
            df=anomaly_df,
            anomaly_variable=var_config["column"],
            title=var_config["title"],
            cmap="RdBu_r",  # Red for positive anomalies, Blue for negative
            figsize=(16, 6),
        )
        save_and_clear(fig, f"figures/{var_config['filename']}")
        print(f"  ✓ figures/{var_config['filename']}")


# New map function with provincial boundaries

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_alberta_station_map(df, shapefile_path='data_raw/lpr_000b16a_e/lpr_000b16a_e.shp', figsize=(14, 16)):
    """
    Create a map of Alberta showing regional quadrants and all weather stations with provincial boundaries.
    
    This visualization shows:
    1. Provincial boundaries for Alberta and neighboring provinces
    2. Regional quadrants (NE, NW, SE, SW Alberta) defined by geographic center
    3. All individual weather stations plotted by lat/lon coordinates
    4. Station labels for identification
    
    Parameters
    ----------
    df : pd.DataFrame
        Long format dataframe with columns: location, latitude, longitude
    shapefile_path : str, optional
        Path to provincial boundaries shapefile
    figsize : tuple, optional
        Figure size (width, height). Default (14, 16)
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Load provincial boundaries
    provinces = gpd.read_file(shapefile_path)
    
    # Convert to WGS84 (lat/lon) for easier plotting with station coordinates
    provinces = provinces.to_crs(epsg=4326)
    
    # Extract unique stations with coordinates (exclude regional aggregations)
    stations = df[
        df["latitude"].notna() 
        & (df["location"] != "Grand Total")
        & ~df["location"].str.contains("Alberta", na=False)
    ][["location", "latitude", "longitude"]].drop_duplicates()
    
    # Calculate geographic boundaries and center of stations
    lat_min = stations["latitude"].min()
    lat_max = stations["latitude"].max()
    lon_min = stations["longitude"].min()
    lon_max = stations["longitude"].max()
    
    lat_center = (lat_min + lat_max) / 2
    lon_center = (lon_min + lon_max) / 2
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get Alberta boundaries
    alberta = provinces[provinces["PRENAME"] == "Alberta"]
    alberta_bounds = alberta.total_bounds  # [minx, miny, maxx, maxy]
    
    # Plot provincial boundaries
    # Alberta in a distinct color
    alberta.boundary.plot(ax=ax, edgecolor="black", linewidth=2.5, zorder=2)
    alberta.plot(ax=ax, color="#F5F5F5", alpha=0.5, zorder=1)
    
    # Neighboring provinces in light gray
    neighbors = provinces[provinces["PRENAME"].isin([
        "British Columbia", "Saskatchewan", "Northwest Territories", "Montana"
    ])]
    neighbors.boundary.plot(ax=ax, edgecolor="gray", linewidth=1.5, zorder=2)
    neighbors.plot(ax=ax, color="#E8E8E8", alpha=0.3, zorder=1)
    
    # Define quadrant colors (light, transparent)
    quadrant_colors = {
        "NW": "#FFE6E6",  # Light red
        "NE": "#E6F3FF",  # Light blue
        "SW": "#FFF4E6",  # Light orange
        "SE": "#E6FFE6",  # Light green
    }
    
    # Draw quadrant rectangles
    # NW quadrant (top-left)
    nw_rect = Rectangle(
        (lon_min, lat_center),
        (lon_center - lon_min),
        (lat_max - lat_center),
        facecolor=quadrant_colors["NW"],
        edgecolor="#CC0000",
        linewidth=2,
        linestyle="--",
        alpha=0.5,
        zorder=3
    )
    ax.add_patch(nw_rect)
    
    # NE quadrant (top-right)
    ne_rect = Rectangle(
        (lon_center, lat_center),
        (lon_max - lon_center),
        (lat_max - lat_center),
        facecolor=quadrant_colors["NE"],
        edgecolor="#0000CC",
        linewidth=2,
        linestyle="--",
        alpha=0.5,
        zorder=3
    )
    ax.add_patch(ne_rect)
    
    # SW quadrant (bottom-left)
    sw_rect = Rectangle(
        (lon_min, lat_min),
        (lon_center - lon_min),
        (lat_center - lat_min),
        facecolor=quadrant_colors["SW"],
        edgecolor="#CC8800",
        linewidth=2,
        linestyle="--",
        alpha=0.5,
        zorder=3
    )
    ax.add_patch(sw_rect)
    
    # SE quadrant (bottom-right)
    se_rect = Rectangle(
        (lon_center, lat_min),
        (lon_max - lon_center),
        (lat_center - lat_min),
        facecolor=quadrant_colors["SE"],
        edgecolor="#00CC00",
        linewidth=2,
        linestyle="--",
        alpha=0.5,
        zorder=3
    )
    ax.add_patch(se_rect)
    
    # Add quadrant labels (centered in each quadrant)
    label_size = 14
    label_weight = "bold"
    
    # NW label
    ax.text(
        (lon_min + lon_center) / 2,
        (lat_center + lat_max) / 2,
        "NW Alberta",
        ha="center",
        va="center",
        fontsize=label_size,
        weight=label_weight,
        color="#990000",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#CC0000", linewidth=1.5),
        zorder=4
    )
    
    # NE label
    ax.text(
        (lon_center + lon_max) / 2,
        (lat_center + lat_max) / 2,
        "NE Alberta",
        ha="center",
        va="center",
        fontsize=label_size,
        weight=label_weight,
        color="#000099",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#0000CC", linewidth=1.5),
        zorder=4
    )
    
    # SW label
    ax.text(
        (lon_min + lon_center) / 2,
        (lat_min + lat_center) / 2,
        "SW Alberta",
        ha="center",
        va="center",
        fontsize=label_size,
        weight=label_weight,
        color="#996600",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#CC8800", linewidth=1.5),
        zorder=4
    )
    
    # SE label
    ax.text(
        (lon_center + lon_max) / 2,
        (lat_min + lat_center) / 2,
        "SE Alberta",
        ha="center",
        va="center",
        fontsize=label_size,
        weight=label_weight,
        color="#009900",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#00CC00", linewidth=1.5),
        zorder=4
    )
    
    # Plot stations as points
    ax.scatter(
        stations["longitude"],
        stations["latitude"],
        s=80,
        c="#CC0000",
        marker="o",
        edgecolors="#660000",
        linewidths=1.5,
        alpha=0.9,
        zorder=5,
        label=f"Weather Stations (n={len(stations)})"
    )
    
    # Add station labels (with slight offset to avoid overlap with points)
    for _, station in stations.iterrows():
        ax.annotate(
            station["location"],
            xy=(station["longitude"], station["latitude"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.5,
            alpha=0.85,
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none")
        )
    
    # Add province labels for neighboring provinces
    for _, prov in neighbors.iterrows():
        centroid = prov.geometry.centroid
        # Only label if centroid is within our view
        ax.text(
            centroid.x, centroid.y,
            prov["PRENAME"],
            ha="center",
            va="center",
            fontsize=12,
            style="italic",
            alpha=0.6,
            color="#333333",
            zorder=2
        )
    
    # Set axis limits to show full Alberta plus some padding
    lon_padding = 1.5
    lat_padding = 1.5
    
    ax.set_xlim(alberta_bounds[0] - lon_padding, alberta_bounds[2] + lon_padding)
    ax.set_ylim(alberta_bounds[1] - lat_padding, alberta_bounds[3] + lat_padding)
    
    # Labels and title
    ax.set_xlabel("Longitude (°W)", fontsize=13)
    ax.set_ylabel("Latitude (°N)", fontsize=13)
    ax.set_title(
        f"Alberta Weather Stations and Regional Analysis Quadrants\n"
        f"Quadrant Boundary: {lat_center:.2f}°N, {lon_center:.2f}°W",
        fontsize=16,
        pad=20,
        weight="bold"
    )
    
    # Add grid
    ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.5, color="gray", zorder=0)
    
    # Add legend
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95)
    
    # Add comprehensive info text box
    info_text = (
        f"Weather Station Coverage:\n"
        f"  Latitude: {lat_min:.2f}°N to {lat_max:.2f}°N\n"
        f"  Longitude: {lon_min:.2f}°W to {lon_max:.2f}°W\n"
        f"  Total Stations: {len(stations)}\n\n"
        f"Regional Quadrants:\n"
        f"  Divided at geographic center\n"
        f"  of station network\n\n"
        f"Note: Quadrants cover only the\n"
        f"area with weather station data\n"
        f"(central Alberta)"
    )
    ax.text(
        0.98, 0.02,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92, edgecolor="gray", linewidth=1),
        zorder=7
    )
    
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    df = pd.read_csv("data_processed/long_df.csv")
    plot_alberta_totals()

    # Generate regional anomaly heatmaps
    plot_regional_anomaly_heatmaps()

    # Generate Alberta station map
    print("\nGenerating Alberta station map...")
    fig = plot_alberta_station_map(df, figsize=(12, 14))
    save_and_clear(fig, "figures/alberta_station_map.png")
    print("  ✓ figures/alberta_station_map.png")
