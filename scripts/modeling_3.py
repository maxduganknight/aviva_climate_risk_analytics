"""
Climate Risk Modeling: Regime Shift Detection and Extreme Value Analysis

This script performs advanced statistical analysis for OSFI B-15 climate risk assessment:
1. Bayesian change point detection to identify regime shifts
2. Generalized Extreme Value (GEV) analysis for return period calculations

Aligns with OSFI B-15 Climate Risk Management guidance requirements for
scenario analysis and physical risk quantification.
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import genextreme

warnings.filterwarnings("ignore")

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent))
from utils import VARIABLES, apply_standard_style, save_and_clear

# ============================================================================
# BAYESIAN CHANGE POINT DETECTION
# ============================================================================


def simple_bayesian_changepoint(data, years):
    """
    Simple Bayesian change point detection using likelihood ratio.

    For each potential change point, calculates the likelihood that the data
    comes from two different distributions (before/after) vs. a single distribution.

    Parameters
    ----------
    data : array-like
        Time series data (annual values)
    years : array-like
        Corresponding years for each data point

    Returns
    -------
    dict
        {
            'changepoint_year': int or None,
            'changepoint_prob': float,
            'mean_before': float,
            'mean_after': float,
            'significance': str ('high', 'moderate', 'low', 'none')
        }

    Notes
    -----
    This is a simplified Bayesian approach that assumes normal distributions.
    For production use, consider PyMC for full Bayesian inference.
    """
    data = np.array(data)
    years = np.array(years)
    n = len(data)

    if n < 10:  # Need minimum data points
        return {
            "changepoint_year": None,
            "changepoint_prob": 0.0,
            "mean_before": np.nan,
            "mean_after": np.nan,
            "significance": "insufficient_data",
        }

    # Calculate log-likelihood for each potential change point
    # Avoid edges (need at least 5 points on each side)
    max_log_likelihood_ratio = -np.inf
    best_changepoint_idx = None

    for i in range(5, n - 5):
        before = data[:i]
        after = data[i:]

        # Log-likelihood of two-segment model
        if (
            len(before) > 1
            and len(after) > 1
            and np.std(before) > 0
            and np.std(after) > 0
        ):
            ll_before = np.sum(
                stats.norm.logpdf(before, np.mean(before), np.std(before))
            )
            ll_after = np.sum(stats.norm.logpdf(after, np.mean(after), np.std(after)))
            ll_two_segment = ll_before + ll_after

            # Log-likelihood of single model
            if np.std(data) > 0:
                ll_single = np.sum(stats.norm.logpdf(data, np.mean(data), np.std(data)))

                # Log-likelihood ratio
                ll_ratio = ll_two_segment - ll_single

                if ll_ratio > max_log_likelihood_ratio:
                    max_log_likelihood_ratio = ll_ratio
                    best_changepoint_idx = i

    if best_changepoint_idx is None:
        return {
            "changepoint_year": None,
            "changepoint_prob": 0.0,
            "mean_before": np.mean(data),
            "mean_after": np.mean(data),
            "significance": "none",
        }

    # Calculate approximate posterior probability using BIC penalty
    # BIC = -2 * log_likelihood + k * log(n), where k = number of parameters
    # Two-segment model has 4 params (2 means, 2 stds), single has 2 params
    bic_penalty = (4 - 2) * np.log(n) / 2
    posterior_odds = np.exp(max_log_likelihood_ratio - bic_penalty)
    posterior_prob = posterior_odds / (1 + posterior_odds)

    # Classify significance
    if posterior_prob > 0.95:
        significance = "high"
    elif posterior_prob > 0.80:
        significance = "moderate"
    elif posterior_prob > 0.60:
        significance = "low"
    else:
        significance = "none"

    before_data = data[:best_changepoint_idx]
    after_data = data[best_changepoint_idx:]

    return {
        "changepoint_year": years[best_changepoint_idx],
        "changepoint_prob": posterior_prob,
        "mean_before": np.mean(before_data),
        "mean_after": np.mean(after_data),
        "std_before": np.std(before_data),
        "std_after": np.std(after_data),
        "significance": significance,
    }


# ============================================================================
# EXTREME VALUE ANALYSIS
# ============================================================================


def bootstrap_gev_confidence_intervals(
    data, return_periods=[10, 25, 50, 100], n_bootstrap=200, confidence_level=0.95
):
    """
    Calculate bootstrap confidence intervals for GEV return levels.

    Uses parametric bootstrap: resample from fitted GEV distribution.

    Parameters
    ----------
    data : array-like
        Extreme values
    return_periods : list of int
        Return periods to calculate CIs for
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (0.95 for 95% CI)

    Returns
    -------
    dict
        {period: {'lower': value, 'upper': value}} for each return period
    """
    data = np.array(data)
    data = data[~np.isnan(data)]
    n = len(data)

    if n < 5:
        return {rp: {"lower": np.nan, "upper": np.nan} for rp in return_periods}

    try:
        # Fit GEV to original data
        params_original = genextreme.fit(data)

        # Bootstrap: resample from fitted distribution
        bootstrap_return_levels = {rp: [] for rp in return_periods}

        for _ in range(n_bootstrap):
            # Generate bootstrap sample from fitted GEV
            bootstrap_sample = genextreme.rvs(*params_original, size=n)

            # Fit GEV to bootstrap sample
            try:
                params_boot = genextreme.fit(bootstrap_sample)

                # Calculate return levels for this bootstrap sample
                for T in return_periods:
                    p = 1 - 1 / T
                    return_level = genextreme.ppf(p, *params_boot)
                    bootstrap_return_levels[T].append(return_level)
            except:
                # If fit fails, skip this bootstrap iteration
                continue

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        confidence_intervals = {}

        for T in return_periods:
            if len(bootstrap_return_levels[T]) > 0:
                lower = np.percentile(bootstrap_return_levels[T], 100 * alpha / 2)
                upper = np.percentile(bootstrap_return_levels[T], 100 * (1 - alpha / 2))
                confidence_intervals[T] = {"lower": lower, "upper": upper}
            else:
                confidence_intervals[T] = {"lower": np.nan, "upper": np.nan}

        return confidence_intervals

    except Exception as e:
        return {rp: {"lower": np.nan, "upper": np.nan} for rp in return_periods}


def fit_gev_and_calculate_return_periods(
    data, return_periods=[10, 25, 50, 100], calculate_ci=False
):
    """
    Fit Generalized Extreme Value distribution and calculate return levels.

    Includes bootstrap confidence intervals and improved fit quality assessment
    that accounts for small sample sizes.

    Parameters
    ----------
    data : array-like
        Extreme values (e.g., annual maxima)
    return_periods : list of int
        Return periods in years to calculate
    calculate_ci : bool
        Whether to calculate bootstrap confidence intervals

    Returns
    -------
    dict
        {
            'n_samples': int,
            'gev_params': tuple (shape, loc, scale),
            'return_levels': dict {period: level},
            'confidence_intervals': dict {period: {'lower': x, 'upper': y}},
            'ks_statistic': float,
            'ks_pvalue': float,
            'fit_quality': str,
            'sample_size_warning': bool
        }
    """
    data = np.array(data)
    data = data[~np.isnan(data)]
    n = len(data)

    if n < 5:
        return {
            "n_samples": n,
            "gev_params": (np.nan, np.nan, np.nan),
            "return_levels": {rp: np.nan for rp in return_periods},
            "confidence_intervals": {
                rp: {"lower": np.nan, "upper": np.nan} for rp in return_periods
            },
            "ks_statistic": np.nan,
            "ks_pvalue": np.nan,
            "fit_quality": "insufficient_data",
            "sample_size_warning": True,
        }

    # Fit GEV distribution (shape, loc, scale)
    try:
        params = genextreme.fit(data)
        shape, loc, scale = params

        # Calculate return levels
        # For return period T, probability is 1/T
        # Return level = GEV quantile at probability (1 - 1/T)
        return_levels = {}
        for T in return_periods:
            p = 1 - 1 / T
            return_level = genextreme.ppf(p, shape, loc, scale)
            return_levels[T] = return_level

        # Bootstrap confidence intervals
        if calculate_ci:
            confidence_intervals = bootstrap_gev_confidence_intervals(
                data, return_periods
            )
        else:
            confidence_intervals = {
                rp: {"lower": np.nan, "upper": np.nan} for rp in return_periods
            }

        # Goodness-of-fit test (Kolmogorov-Smirnov)
        ks_stat, ks_pval = stats.kstest(data, lambda x: genextreme.cdf(x, *params))

        # Classify fit quality based on p-value AND sample size
        # With small samples (n < 20), KS test has low power, so be more conservative
        if n < 20:
            sample_size_warning = True
            # More conservative classification for small samples
            if ks_pval > 0.10:
                fit_quality = "acceptable_small_n"
            elif ks_pval > 0.05:
                fit_quality = "marginal_small_n"
            else:
                fit_quality = "questionable_small_n"
        else:
            sample_size_warning = False
            # Standard classification for adequate samples
            if ks_pval > 0.10:
                fit_quality = "good"
            elif ks_pval > 0.05:
                fit_quality = "acceptable"
            elif ks_pval > 0.01:
                fit_quality = "marginal"
            else:
                fit_quality = "poor"

        return {
            "n_samples": n,
            "gev_params": params,
            "return_levels": return_levels,
            "confidence_intervals": confidence_intervals,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pval,
            "fit_quality": fit_quality,
            "sample_size_warning": sample_size_warning,
        }

    except Exception as e:
        print(f"    Warning: GEV fitting failed - {str(e)}")
        return {
            "n_samples": n,
            "gev_params": (np.nan, np.nan, np.nan),
            "return_levels": {rp: np.nan for rp in return_periods},
            "confidence_intervals": {
                rp: {"lower": np.nan, "upper": np.nan} for rp in return_periods
            },
            "ks_statistic": np.nan,
            "ks_pvalue": np.nan,
            "fit_quality": "failed",
            "sample_size_warning": n < 20,
        }


def calculate_return_period_shift(
    baseline_return_level, current_data, current_gev_params
):
    """
    Calculate how the return period has changed for a baseline extreme event.

    For example: "The baseline 100-year event now occurs every 30 years"

    Parameters
    ----------
    baseline_return_level : float
        The extreme value from baseline period (e.g., 100-year temp)
    current_data : array-like
        Current period extreme values
    current_gev_params : tuple
        GEV parameters for current period

    Returns
    -------
    float
        New return period in years (e.g., 30.5)
    """
    if np.isnan(baseline_return_level) or any(np.isnan(current_gev_params)):
        return np.nan

    # Calculate probability of exceeding baseline level in current distribution
    shape, loc, scale = current_gev_params
    prob_exceedance = 1 - genextreme.cdf(baseline_return_level, shape, loc, scale)

    if prob_exceedance <= 0:
        return np.inf  # Event is now so common it's off the scale

    # Return period = 1 / probability of exceedance
    new_return_period = 1 / prob_exceedance

    return new_return_period


# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================


def analyze_regime_shifts(df, regions, variables, months=None, label_suffix=""):
    """
    Detect regime shifts for all region-variable combinations.

    Parameters
    ----------
    df : pd.DataFrame
        Data with columns: location, year, month, and variable columns
    regions : list
        List of region names to analyze
    variables : dict
        Dictionary of variable configurations
    months : list of int, optional
        List of months to filter (e.g., [6, 7, 8] for summer months).
        If None, uses all months.
    label_suffix : str, optional
        Suffix to add to variable display name (e.g., " (Summer)")

    Returns
    -------
    pd.DataFrame
        Results with columns: region, variable, changepoint_year, probability, etc.
    """
    month_desc = f" (Months: {months})" if months else ""
    print("\n" + "=" * 80)
    print(f"REGIME SHIFT DETECTION (Bayesian Change Point Analysis){month_desc}")
    print("=" * 80 + "\n")

    results = []

    for region in regions:
        region_data = df[df["location"] == region].copy()

        # Filter by months if specified
        if months is not None:
            region_data = region_data[region_data["month"].isin(months)].copy()

        for var_key, var_config in variables.items():
            var_name = var_config["long_name"]

            if var_name not in region_data.columns:
                continue

            # Aggregate to annual means (for the selected months only)
            annual_data = region_data.groupby("year")[var_name].mean()
            years = annual_data.index.values
            values = annual_data.values

            # Remove NaNs
            mask = ~np.isnan(values)
            years_clean = years[mask]
            values_clean = values[mask]

            if len(values_clean) < 10:
                continue

            # Detect change point
            result = simple_bayesian_changepoint(values_clean, years_clean)

            # Store results
            display_name = var_config["display_name"] + label_suffix
            results.append(
                {
                    "region": region,
                    "variable": display_name,
                    "variable_key": var_key,
                    "changepoint_year": result["changepoint_year"],
                    "probability": result["changepoint_prob"],
                    "significance": result["significance"],
                    "mean_before": result["mean_before"],
                    "mean_after": result["mean_after"],
                    "change": result["mean_after"] - result["mean_before"]
                    if result["changepoint_year"]
                    else 0,
                    "pct_change": (
                        (result["mean_after"] - result["mean_before"])
                        / abs(result["mean_before"])
                        * 100
                    )
                    if result["changepoint_year"] and result["mean_before"] != 0
                    else 0,
                }
            )

            # Print significant findings
            if result["significance"] in ["high", "moderate"]:
                print(
                    f"{region:15s} | {display_name:25s} | "
                    f"Changepoint: {result['changepoint_year']} (prob={result['changepoint_prob']:.3f}) | "
                    f"Change: {result['mean_before']:.2f} → {result['mean_after']:.2f}"
                )

    return pd.DataFrame(results)


def analyze_extreme_values(df, regions, variables, extrema_config):
    """
    Perform GEV analysis on extreme values.

    Parameters
    ----------
    extrema_config : dict
        {variable_key: 'max' or 'min' or 'both'}

    Returns
    -------
    pd.DataFrame
        Results with return period calculations
    """
    print("\n" + "=" * 80)
    print("EXTREME VALUE ANALYSIS (GEV Distribution Fitting)")
    print("=" * 80 + "\n")

    baseline_years = (1981, 1996)
    current_years = (2010, 2025)
    return_periods = [10, 25, 50, 100]

    results = []

    for region in regions:
        region_data = df[df["location"] == region].copy()

        for var_key, var_config in variables.items():
            var_name = var_config["long_name"]

            if var_name not in region_data.columns:
                continue

            # Determine which extrema to analyze
            extrema_types = []
            if var_key in extrema_config:
                if extrema_config[var_key] == "both":
                    extrema_types = ["max", "min"]
                else:
                    extrema_types = [extrema_config[var_key]]

            for extrema_type in extrema_types:
                # Extract annual extrema
                if extrema_type == "max":
                    annual_extrema = region_data.groupby("year")[var_name].max()
                    extrema_label = "Maximum"
                else:
                    annual_extrema = region_data.groupby("year")[var_name].min()
                    extrema_label = "Minimum"

                # Split into baseline and current periods
                baseline_extrema = annual_extrema[
                    (annual_extrema.index >= baseline_years[0])
                    & (annual_extrema.index <= baseline_years[1])
                ].values

                current_extrema = annual_extrema[
                    (annual_extrema.index >= current_years[0])
                    & (annual_extrema.index <= current_years[1])
                ].values

                if len(baseline_extrema) < 5 or len(current_extrema) < 5:
                    continue

                # Fit GEV for both periods
                baseline_fit = fit_gev_and_calculate_return_periods(
                    baseline_extrema, return_periods
                )
                current_fit = fit_gev_and_calculate_return_periods(
                    current_extrema, return_periods
                )

                # Calculate return period shift for 100-year event
                baseline_100yr = baseline_fit["return_levels"].get(100, np.nan)
                new_return_period = calculate_return_period_shift(
                    baseline_100yr, current_extrema, current_fit["gev_params"]
                )

                # Store results
                result_row = {
                    "region": region,
                    "variable": var_config["display_name"],
                    "variable_key": var_key,
                    "extrema_type": extrema_label,
                    "baseline_n_samples": baseline_fit["n_samples"],
                    "current_n_samples": current_fit["n_samples"],
                    "baseline_sample_size_warning": baseline_fit["sample_size_warning"],
                    "current_sample_size_warning": current_fit["sample_size_warning"],
                    "baseline_fit_quality": baseline_fit["fit_quality"],
                    "current_fit_quality": current_fit["fit_quality"],
                    "baseline_ks_pvalue": baseline_fit["ks_pvalue"],
                    "current_ks_pvalue": current_fit["ks_pvalue"],
                }

                # Add return levels and confidence intervals for both periods
                for rp in return_periods:
                    # Baseline period
                    result_row[f"baseline_{rp}yr"] = baseline_fit["return_levels"].get(
                        rp, np.nan
                    )
                    result_row[f"baseline_{rp}yr_ci_lower"] = (
                        baseline_fit["confidence_intervals"]
                        .get(rp, {})
                        .get("lower", np.nan)
                    )
                    result_row[f"baseline_{rp}yr_ci_upper"] = (
                        baseline_fit["confidence_intervals"]
                        .get(rp, {})
                        .get("upper", np.nan)
                    )

                    # Current period
                    result_row[f"current_{rp}yr"] = current_fit["return_levels"].get(
                        rp, np.nan
                    )
                    result_row[f"current_{rp}yr_ci_lower"] = (
                        current_fit["confidence_intervals"]
                        .get(rp, {})
                        .get("lower", np.nan)
                    )
                    result_row[f"current_{rp}yr_ci_upper"] = (
                        current_fit["confidence_intervals"]
                        .get(rp, {})
                        .get("upper", np.nan)
                    )

                    # Change
                    result_row[f"{rp}yr_change"] = current_fit["return_levels"].get(
                        rp, np.nan
                    ) - baseline_fit["return_levels"].get(rp, np.nan)

                result_row["baseline_100yr_new_period"] = new_return_period

                results.append(result_row)

                # Print significant findings (where 100-year event has shifted)
                if not np.isnan(new_return_period) and new_return_period < 90:
                    print(
                        f"{region:15s} | {var_config['display_name']:25s} ({extrema_label}) | "
                        f"Baseline 100-yr event ({baseline_100yr:.2f} {var_config['units']}) "
                        f"now occurs every {new_return_period:.1f} years"
                    )

    return pd.DataFrame(results)


# ============================================================================
# VISUALIZATION
# ============================================================================


def create_gev_diagnostic_plots(data, gev_params, title, units):
    """
    Create Q-Q plot and return level plot for GEV fit diagnostics.

    Parameters
    ----------
    data : array-like
        Extreme values
    gev_params : tuple
        GEV parameters (shape, loc, scale)
    title : str
        Plot title
    units : str
        Units for axis labels

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = np.array(data)
    data = data[~np.isnan(data)]
    n = len(data)

    if n < 5 or any(np.isnan(gev_params)):
        return None

    shape, loc, scale = gev_params

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Q-Q Plot ----
    # Theoretical quantiles from GEV
    sorted_data = np.sort(data)
    plotting_positions = (np.arange(1, n + 1) - 0.5) / n
    theoretical_quantiles = genextreme.ppf(plotting_positions, shape, loc, scale)

    ax1.scatter(theoretical_quantiles, sorted_data, alpha=0.7, s=50)

    # Add 1:1 line
    min_val = min(theoretical_quantiles.min(), sorted_data.min())
    max_val = max(theoretical_quantiles.max(), sorted_data.max())
    ax1.plot(
        [min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect fit"
    )

    ax1.set_xlabel(f"Theoretical Quantiles ({units})", fontsize=11)
    ax1.set_ylabel(f"Observed Quantiles ({units})", fontsize=11)
    ax1.set_title("Q-Q Plot: GEV Fit Quality", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---- Return Level Plot ----
    # Calculate return levels for a range of return periods
    return_periods = np.logspace(0, 2.5, 50)  # 1 to ~300 years
    return_levels = genextreme.ppf(1 - 1 / return_periods, shape, loc, scale)

    # Plot fitted GEV return levels
    ax2.semilogx(return_periods, return_levels, "b-", linewidth=2, label="GEV fit")

    # Plot empirical return levels (using plotting positions)
    empirical_return_periods = n / (n - np.arange(n) + 0.5)
    ax2.semilogx(
        empirical_return_periods,
        sorted_data,
        "ro",
        markersize=6,
        alpha=0.7,
        label="Observed",
    )

    ax2.set_xlabel("Return Period (years)", fontsize=11)
    ax2.set_ylabel(f"Return Level ({units})", fontsize=11)
    ax2.set_title("Return Level Plot", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    return fig


def create_changepoint_heatmap(regime_df):
    """Create heatmap showing change point years across regions and variables."""

    # Pivot to create matrix
    heatmap_data = regime_df.pivot_table(
        values="changepoint_year", index="variable", columns="region", aggfunc="first"
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    im = ax.imshow(
        heatmap_data.values, aspect="auto", cmap="YlOrRd", vmin=1985, vmax=2020
    )

    # Set ticks
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)

    # Add values as text
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            val = heatmap_data.iloc[i, j]
            if not np.isnan(val):
                text = ax.text(
                    j,
                    i,
                    f"{int(val)}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9,
                    weight="bold",
                )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Changepoint Year", rotation=270, labelpad=20)

    ax.set_title(
        "Climate Regime Shift Detection: Changepoint Years by Region and Variable",
        fontsize=14,
        pad=20,
    )

    plt.tight_layout()
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\nClimate Risk Modeling: Regime Shifts & Extreme Value Analysis")
    print("=" * 80)

    # Load data
    df = pd.read_csv("data_processed/long_df.csv")

    # Define regions and variables
    regions = ["Grand Total", "NE Alberta", "NW Alberta", "SE Alberta", "SW Alberta"]

    # Define which extrema to analyze for each variable
    extrema_config = {
        "temperature": "both",  # Max (heat waves) and Min (cold snaps)
        "precipitation": "both",  # Max (extreme rainfall) and Min (drought)
        "snow_on_ground": "both",  # Max (extreme snow) and Min (no snow)
        "heating_degrees": "max",  # Max only (extreme cold periods)
        "cooling_degrees": "max",  # Max only (extreme heat periods)
        "proxy_fwi": "max",  # Max only (extreme fire risk)
    }

    # Run regime shift detection - all months
    regime_results = analyze_regime_shifts(df, regions, VARIABLES)

    # Run regime shift detection - summer months only (June, July, August)
    print("\n")  # Add spacing
    summer_regime_results = analyze_regime_shifts(
        df, regions, VARIABLES, months=[6, 7, 8], label_suffix=" (Summer)"
    )

    # Combine regime shift results
    regime_results_combined = pd.concat(
        [regime_results, summer_regime_results], ignore_index=True
    )

    # Run extreme value analysis
    gev_results = analyze_extreme_values(df, regions, VARIABLES, extrema_config)

    # Save results to CSV
    regime_results_combined.to_csv(
        "data_processed/regime_shift_results.csv", index=False
    )
    gev_results.to_csv("data_processed/extreme_value_results.csv", index=False)

    print("\n" + "=" * 80)
    print("RESULTS SAVED")
    print("=" * 80)
    print("  - data_processed/regime_shift_results.csv")
    print("  - data_processed/extreme_value_results.csv")

    # Create visualizations
    print("\nGenerating visualizations...")

    # Changepoint heatmap
    fig = create_changepoint_heatmap(regime_results)
    save_and_clear(fig, "figures/regime_shift_changepoint_heatmap.png")
    print("  - figures/regime_shift_changepoint_heatmap.png")

    # GEV diagnostic plots for key variables (Grand Total, current period)
    print("\nGenerating GEV diagnostic plots for key variables...")

    diagnostic_vars = [
        ("temperature", "max", "°C"),
        ("proxy_fwi", "max", "index (0-100)"),
        ("precipitation", "max", "mm"),
    ]

    for var_key, extrema_type, units in diagnostic_vars:
        var_config = VARIABLES[var_key]
        var_name = var_config["long_name"]
        region = "Grand Total"

        # Get current period data (2010-2025)
        region_data = df[df["location"] == region]
        if extrema_type == "max":
            annual_extrema = region_data.groupby("year")[var_name].max()
        else:
            annual_extrema = region_data.groupby("year")[var_name].min()

        current_extrema = annual_extrema[
            (annual_extrema.index >= 2010) & (annual_extrema.index <= 2025)
        ].values

        if len(current_extrema) >= 5:
            # Fit GEV
            try:
                gev_params = genextreme.fit(current_extrema)

                # Create diagnostic plot
                title = f"{region} - {var_config['display_name']} ({extrema_type.capitalize()}) GEV Diagnostics (2010-2025)"
                fig = create_gev_diagnostic_plots(
                    current_extrema, gev_params, title, units
                )

                if fig is not None:
                    filename = f"figures/gev_diagnostic_{region.replace(' ', '_').lower()}_{var_key}_{extrema_type}.png"
                    save_and_clear(fig, filename)
                    print(f"  - {filename}")
            except Exception as e:
                print(
                    f"    Warning: Could not create diagnostic plot for {var_key}: {e}"
                )

    print("\nAnalysis complete!")
