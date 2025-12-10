import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
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
    totals_df = df[df["location"] == "Grand Total"]

    # Plotting annual temperature
    annual_temps = totals_df.groupby("year")["mean_temperature_c"].mean()
    total_temps = plot_bars(
        annual_temps.index, annual_temps, "Alberta Annual Temperature °C"
    )
    total_temps.savefig("figures/alberta_annual_temperature.png")
    plt.clf()

    # Plotting annual precipitation
    annual_precip = totals_df.groupby("year")["total_precipitation_mm"].sum()
    total_precip = plot_bars(
        annual_precip.index, annual_precip, "Alberta Annual Precipitation mm"
    )
    total_precip.savefig("figures/alberta_annual_precipitation.png")
    plt.clf()

    # Snow on Ground
    annual_snow = totals_df.groupby("year")["snow_on_ground_last_day_cm"].sum()
    total_snow = plot_bars(
        annual_snow.index, annual_snow, "Alberta Annual Snow on Ground Sum cm"
    )
    total_snow.savefig("figures/alberta_annual_snow_on_ground.png")
    plt.clf()

    # Proxy FWI - Annual Average
    annual_fwi = totals_df.groupby("year")["proxy_fwi"].mean()
    total_fwi = plot_line(
        annual_fwi.index, annual_fwi, "Alberta Annual Average Proxy FWI"
    )
    plt.xlabel("Year")
    plt.ylabel("Proxy FWI (0-100)")
    total_fwi.savefig("figures/alberta_annual_proxy_fwi.png")
    plt.clf()

    # Proxy FWI - Summer Peak (May-Sep average)
    summer_months = totals_df[totals_df["month"].isin([5, 6, 7, 8, 9])]
    summer_fwi = summer_months.groupby("year")["proxy_fwi"].mean()
    summer_fwi_plot = plot_line(
        summer_fwi.index, summer_fwi, "Alberta Summer Peak Proxy FWI (May-Sep)"
    )
    plt.xlabel("Year")
    plt.ylabel("Proxy FWI (0-100)")
    summer_fwi_plot.savefig("figures/alberta_summer_proxy_fwi.png")
    plt.clf()

    # Heating degree days
    annual_heating = totals_df.groupby("year")["mean_heating_days_c"].sum()
    total_heating = plot_bars(
        annual_heating.index, annual_heating, "Alberta Annual Heating Degree Days"
    )
    total_heating.savefig("figures/alberta_annual_heating_degree_days.png")
    plt.clf()

    # Cooling degree days
    annual_cooling = totals_df.groupby("year")["mean_cooling_days_c"].sum()
    total_cooling = plot_bars(
        annual_cooling.index, annual_cooling, "Alberta Annual Cooling Degree Days"
    )
    total_cooling.savefig("figures/alberta_annual_cooling_degree_days.png")
    plt.clf()

    # Monthly temperature by decade
    fig = plot_each_decade_by_month(
        totals_df,
        "mean_temperature_c",
        "Alberta Monthly Temperature by Decade",
        "Temperature (°C)",
    )
    save_and_clear(fig, "figures/alberta_monthly_temperature_by_decade.png")

    # Monthly precipitation by decade
    fig = plot_each_decade_by_month(
        totals_df,
        "total_precipitation_mm",
        "Precipitation Decreasing Especially in Hottest Months",
        "Alberta Total Precipitation (mm)",
    )
    save_and_clear(fig, "figures/alberta_monthly_precipitation_by_decade.png")

    # Monthly proxy FWI by decade
    fig = plot_each_decade_by_month(
        totals_df,
        "proxy_fwi",
        "Fire Weather Increasing and Extending into Fall",
        "Alberta Proxy Fire Weather Index",
        caption="Note: Proxy FWI is an estimate of the Canadian Forest Fire Weather Index (FWI) System derived from temperature, precipitation, and drought. It is temperature-gated (values below 5°C set to zero).",
    )
    save_and_clear(fig, "figures/alberta_monthly_proxy_fwi_by_decade.png")


def plot_regional_anomaly_heatmaps():
    """
    Generate time-series heatmaps for all regional anomaly variables.

    Creates one heatmap per variable showing how anomalies vary across
    regions and time.
    """
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
                }
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
        print(f"Saved {var_config['filename']}")


if __name__ == "__main__":
    df = pd.read_csv("data_processed/long_df.csv")
    plot_alberta_totals()

    # Generate regional anomaly heatmaps
    plot_regional_anomaly_heatmaps()
