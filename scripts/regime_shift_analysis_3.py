"""
Climate Risk Modeling: Regime Shift Detection

This script performs Bayesian change point detection to identify regime shifts
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


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\nClimate Risk Modeling: Regime Shifts & Extreme Value Analysis")
    print("=" * 80)

    # Load data
    df = pd.read_csv("data_processed/long_df.csv")

    # Define regions and variables
    regions = ["Grand Total", "North Cluster", "Southwest Cluster", "Southeast Cluster"]

    # Define which extrema to analyze for each variable
    extrema_config = {
        "temperature": "both",  # Max (heat waves) and Min (cold snaps)
        "precipitation": "both",  # Max (extreme rainfall) and Min (drought)
        "snow_on_ground": "both",  # Max (extreme snow) and Min (no snow)
        "heating_degrees": "max",  # Max only (extreme cold periods)
        "cooling_degrees": "max",  # Max only (extreme heat periods)
        "proxy_fwi": "max",  # Max only (extreme fire risk)
        "drought_accumulation": "max",  # Max only (extreme drought conditions)
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

    # Save results to CSV
    regime_results_combined.to_csv(
        "data_processed/regime_shift_results.csv", index=False
    )

    print("\n" + "=" * 80)
    print("RESULTS SAVED")
    print("=" * 80)
    print("  - data_processed/regime_shift_results.csv")
    # Create visualizations
    print("\nGenerating visualizations...")

    print("\nAnalysis complete!")
