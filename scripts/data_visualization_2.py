import matplotlib.pyplot as plt
import pandas as pd
from utils import (
    DECADE_COLORS,
    FIGURE_SIZE_WIDE,
    apply_standard_style,
    save_and_clear,
    setup_month_axis,
)


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


def plot_each_decade_by_month(df, y_variable, title, ylabel):
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
    ax.legend(title="Decade", loc="best")

    plt.tight_layout()
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
        "Alberta Monthly Precipitation by Decade",
        "Precipitation (mm)",
    )
    save_and_clear(fig, "figures/alberta_monthly_precipitation_by_decade.png")

    # Monthly proxy FWI by decade
    fig = plot_each_decade_by_month(
        totals_df,
        "proxy_fwi",
        "Alberta Monthly Fire Weather Index by Decade",
        "Proxy FWI (0-100)",
    )
    save_and_clear(fig, "figures/alberta_monthly_proxy_fwi_by_decade.png")


if __name__ == "__main__":
    df = pd.read_csv("data_processed/long_df.csv")
    plot_alberta_totals()
