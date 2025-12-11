#!/usr/bin/env Rscript

################################################################################
# Extreme Value Analysis
#
# This script performs extreme value analysis using the extRemes package,
# including non-stationary GEV models to detect temporal trends in extremes.
#
# Based on methodology from:
# - extRemes package documentation
# - IPCC AR6 extreme event attribution methods
# - Previous rainfall EVA analysis
################################################################################

# Load required libraries
suppressPackageStartupMessages({
  library(extRemes)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(lubridate)
})

cat("\n================================================================================\n")
cat("EXTREME VALUE ANALYSIS - extRemes Package (R)\n")
cat("================================================================================\n\n")

################################################################################
# CONFIGURATION
################################################################################

# Set working directory to script location
if (interactive()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
} else {
  # When run from command line, assume we're in project root
  args <- commandArgs(trailingOnly = FALSE)
  script_path <- sub("--file=", "", args[grep("--file=", args)])
  if (length(script_path) > 0) {
    setwd(dirname(script_path))
  }
}

# Navigate to project root if we're in scripts/
if (basename(getwd()) == "scripts") {
  setwd("..")
}

cat(paste0("Working directory: ", getwd(), "\n\n"))

# File paths
input_file <- "data_processed/long_df.csv"
output_file <- "data_processed/extreme_value_results_R.csv"
plots_dir <- "figures/R_eva"

# Create plots directory if it doesn't exist
if (!dir.exists(plots_dir)) {
  dir.create(plots_dir, recursive = TRUE)
  cat(paste0("Created directory: ", plots_dir, "\n"))
}

# Analysis parameters
regions <- c("Grand Total", "North Cluster", "Southwest Cluster", "Southeast Cluster")
baseline_years <- c(1981, 1996)
current_years <- c(2009, 2024)
return_periods <- c(10, 25, 50, 100)

################################################################################
# LOAD AND PREPARE DATA
################################################################################

cat("Loading data...\n")
df <- read.csv(input_file)

# Convert year/month to proper types
df$year <- as.integer(df$year)
df$month <- as.integer(df$month)

cat(paste0("Loaded ", nrow(df), " rows of climate data\n\n"))

################################################################################
# HELPER FUNCTIONS
################################################################################

#' Fit stationary and non-stationary GEV models and compare
#'
#' @param data Vector of extreme values
#' @param years Vector of corresponding years
#' @return List with model results and diagnostics
fit_gev_models <- function(data, years) {

  # Remove NAs
  valid_idx <- !is.na(data)
  data_clean <- data[valid_idx]
  years_clean <- years[valid_idx]

  if (length(data_clean) < 10) {
    return(list(
      stationary_fit = NULL,
      nonstationary_fit = NULL,
      lr_test_pvalue = NA,
      best_model = "insufficient_data",
      n_samples = length(data_clean)
    ))
  }

  # Fit stationary GEV model
  stationary_fit <- tryCatch({
    fevd(data_clean, type = "GEV", method = "MLE")
  }, error = function(e) {
    NULL
  })

  # Fit non-stationary GEV model
  # Try location and scale varying first, fall back to location-only if that fails
  nonstationary_fit <- tryCatch({
    fevd(data_clean,
         location.fun = ~ years_clean,
         scale.fun = ~ years_clean,
         type = "GEV",
         method = "MLE")
  }, error = function(e) {
    NULL
  })

  # If full non-stationary model failed, try simpler model (location only)
  if (is.null(nonstationary_fit)) {
    nonstationary_fit <- tryCatch({
      fevd(data_clean,
           location.fun = ~ years_clean,
           type = "GEV",
           method = "MLE")
    }, error = function(e) {
      NULL
    })
  }

  # Likelihood ratio test
  lr_pvalue <- NA
  best_model <- "stationary"

  if (!is.null(stationary_fit) && !is.null(nonstationary_fit)) {
    lr_result <- tryCatch({
      lr.test(stationary_fit, nonstationary_fit)
    }, error = function(e) {
      NULL
    })

    if (!is.null(lr_result)) {
      lr_pvalue <- lr_result$p.value
      # If p < 0.05, non-stationary model is significantly better
      if (lr_pvalue < 0.05) {
        best_model <- "nonstationary"
      }
    }
  }

  return(list(
    stationary_fit = stationary_fit,
    nonstationary_fit = nonstationary_fit,
    lr_test_pvalue = lr_pvalue,
    best_model = best_model,
    n_samples = length(data_clean),
    years = years_clean
  ))
}

#' Calculate return levels for baseline and current periods
#'
#' @param model_fit GEV model object
#' @param return_periods Vector of return periods
#' @param baseline_year Year for baseline calculation (for non-stationary)
#' @param current_year Year for current calculation (for non-stationary)
#' @param is_nonstationary Boolean indicating if model is non-stationary
#' @return Data frame with return levels and confidence intervals
calculate_return_levels <- function(model_fit, return_periods,
                                   baseline_year = NULL,
                                   current_year = NULL,
                                   is_nonstationary = FALSE) {

  if (is.null(model_fit)) {
    return(data.frame(
      return_period = return_periods,
      baseline_rl = NA,
      baseline_rl_lower = NA,
      baseline_rl_upper = NA,
      current_rl = NA,
      current_rl_lower = NA,
      current_rl_upper = NA
    ))
  }

  results <- data.frame(
    return_period = integer(),
    baseline_rl = numeric(),
    baseline_rl_lower = numeric(),
    baseline_rl_upper = numeric(),
    current_rl = numeric(),
    current_rl_lower = numeric(),
    current_rl_upper = numeric()
  )

  for (rp in return_periods) {

    if (is_nonstationary && !is.null(baseline_year) && !is.null(current_year)) {
      # Non-stationary model: calculate for both periods

      # Create covariate matrices
      qcov_baseline <- make.qcov(model_fit,
                                  vals = list(mu1 = baseline_year, sigma1 = baseline_year))
      qcov_current <- make.qcov(model_fit,
                                vals = list(mu1 = current_year, sigma1 = current_year))

      # Calculate return levels with confidence intervals
      ci_baseline <- tryCatch({
        ci(model_fit, alpha = 0.05, type = 'return.level',
           return.period = rp, qcov = qcov_baseline, method = 'normal')
      }, error = function(e) {
        NULL
      })

      ci_current <- tryCatch({
        ci(model_fit, alpha = 0.05, type = 'return.level',
           return.period = rp, qcov = qcov_current, method = 'normal')
      }, error = function(e) {
        NULL
      })

      # Check if CIs are valid (ci() returns a named numeric vector, not a matrix)
      baseline_valid <- !is.null(ci_baseline) && length(ci_baseline) >= 3
      current_valid <- !is.null(ci_current) && length(ci_current) >= 3

      # If CI calculation failed, try to get return level without CIs
      if (!baseline_valid) {
        rl_baseline <- tryCatch({
          rl_result <- return.level(model_fit, return.period = rp, qcov = qcov_baseline)
          if (is.numeric(rl_result)) rl_result[1] else NA
        }, error = function(e) {
          NA
        })
      } else {
        rl_baseline <- ci_baseline[2]
      }

      if (!current_valid) {
        rl_current <- tryCatch({
          rl_result <- return.level(model_fit, return.period = rp, qcov = qcov_current)
          if (is.numeric(rl_result)) rl_result[1] else NA
        }, error = function(e) {
          NA
        })
      } else {
        rl_current <- ci_current[2]
      }

      row_data <- data.frame(
        return_period = rp,
        baseline_rl = rl_baseline,
        baseline_rl_lower = if(baseline_valid) ci_baseline[1] else NA,
        baseline_rl_upper = if(baseline_valid) ci_baseline[3] else NA,
        current_rl = rl_current,
        current_rl_lower = if(current_valid) ci_current[1] else NA,
        current_rl_upper = if(current_valid) ci_current[3] else NA
      )
      results <- rbind(results, row_data)

    } else {
      # Stationary model: same for both periods

      ci_stationary <- tryCatch({
        ci(model_fit, alpha = 0.05, type = 'return.level',
           return.period = rp, method = 'normal')
      }, error = function(e) {
        NULL
      })

      # Check if ci_stationary is valid (ci() returns a named numeric vector, not a matrix)
      if (is.null(ci_stationary) || length(ci_stationary) < 3) {
        # Use return level without CI if CI calculation failed
        rl_value <- tryCatch({
          rl_result <- return.level(model_fit, return.period = rp)
          # return.level returns a named vector, extract the value
          if (is.numeric(rl_result)) rl_result[1] else NA
        }, error = function(e) {
          NA
        })

        row_data <- data.frame(
          return_period = rp,
          baseline_rl = rl_value,
          baseline_rl_lower = NA,
          baseline_rl_upper = NA,
          current_rl = rl_value,
          current_rl_lower = NA,
          current_rl_upper = NA
        )
      } else {
        row_data <- data.frame(
          return_period = rp,
          baseline_rl = ci_stationary[2],
          baseline_rl_lower = ci_stationary[1],
          baseline_rl_upper = ci_stationary[3],
          current_rl = ci_stationary[2],
          current_rl_lower = ci_stationary[1],
          current_rl_upper = ci_stationary[3]
        )
      }
      results <- rbind(results, row_data)
    }
  }

  results
}

#' Create diagnostic plots for GEV fit
#'
#' @param model_fit GEV model object
#' @param title Plot title
#' @param filename Output filename
create_diagnostic_plots <- function(model_fit, title, filename) {

  if (is.null(model_fit)) {
    return(NULL)
  }

  png(filename, width = 14, height = 10, units = "in", res = 150)

  tryCatch({
    # extRemes package provides built-in diagnostic plots
    # Suppress the model call text to prevent overlap issues
    plot(model_fit, show.call = FALSE)
  }, error = function(e) {
    cat(paste0("Warning: Could not create plot for ", title, "\n"))
  })

  dev.off()
}

################################################################################
# MAIN ANALYSIS LOOP
################################################################################

cat("================================================================================\n")
cat("EXTREME VALUE ANALYSIS BY REGION AND VARIABLE\n")
cat("================================================================================\n\n")

# Variables to analyze (matching Python script)
variables_config <- list(
  list(var_name = "mean_temperature_c", display_name = "Temperature",
       units = "°C", extrema_types = c("max", "min")),
  list(var_name = "total_precipitation_mm", display_name = "Precipitation",
       units = "mm", extrema_types = c("max", "min")),
  list(var_name = "snow_on_ground_last_day_cm", display_name = "Snow on Ground",
       units = "cm", extrema_types = c("max", "min")),
  list(var_name = "mean_heating_days_c", display_name = "Heating Degree Days",
       units = "°C", extrema_types = c("max")),
  list(var_name = "mean_cooling_days_c", display_name = "Cooling Degree Days",
       units = "°C", extrema_types = c("max")),
  list(var_name = "proxy_fwi", display_name = "Proxy Fire Weather Index",
       units = "index", extrema_types = c("max")),
  list(var_name = "drought_accumulation", display_name = "Drought Accumulation",
       units = "index", extrema_types = c("max"))
)

# Initialize results dataframe
all_results <- data.frame()

# Loop through regions and variables
for (region in regions) {

  region_data <- df %>% filter(location == region)

  for (var_config in variables_config) {

    var_name <- var_config$var_name
    display_name <- var_config$display_name
    units <- var_config$units
    extrema_types <- var_config$extrema_types

    for (extrema_type in extrema_types) {

      # Extract annual extrema
      if (extrema_type == "max") {
        annual_extrema <- region_data %>%
          group_by(year) %>%
          summarise(extreme_value = max(.data[[var_name]], na.rm = TRUE),
                   .groups = "drop")
        extrema_label <- "Maximum"
      } else {
        annual_extrema <- region_data %>%
          group_by(year) %>%
          summarise(extreme_value = min(.data[[var_name]], na.rm = TRUE),
                   .groups = "drop")
        extrema_label <- "Minimum"
      }

      # Remove infinite values
      annual_extrema <- annual_extrema %>%
        filter(is.finite(extreme_value))

      if (nrow(annual_extrema) < 10) {
        cat(paste0("Skipping ", region, " - ", display_name, " (", extrema_label,
                  "): insufficient data\n"))
        next
      }

      cat(paste0("\nAnalyzing: ", region, " - ", display_name, " (", extrema_label, ")\n"))

      # Fit GEV models
      model_results <- fit_gev_models(annual_extrema$extreme_value,
                                      annual_extrema$year)

      # Choose best model
      if (model_results$best_model == "nonstationary") {
        cat("  → Non-stationary model selected (p < 0.05)\n")
        best_fit <- model_results$nonstationary_fit
        is_nonstat <- TRUE
      } else {
        cat("  → Stationary model selected\n")
        best_fit <- model_results$stationary_fit
        is_nonstat <- FALSE
      }

      # Calculate return levels
      baseline_mid_year <- mean(baseline_years)
      current_mid_year <- mean(current_years)

      return_levels <- calculate_return_levels(
        best_fit,
        return_periods,
        baseline_mid_year,
        current_mid_year,
        is_nonstat
      )

      # Add metadata (use as.character to ensure consistent types)
      return_levels$region <- as.character(region)
      return_levels$variable <- as.character(display_name)
      return_levels$extrema_type <- as.character(extrema_label)
      return_levels$model_type <- as.character(model_results$best_model)
      return_levels$lr_test_pvalue <- as.numeric(model_results$lr_test_pvalue)
      return_levels$n_samples <- as.integer(model_results$n_samples)
      return_levels$sample_size_warning <- as.logical(model_results$n_samples < 20)

      # Calculate change metrics
      return_levels$absolute_change <- as.numeric(return_levels$current_rl - return_levels$baseline_rl)
      return_levels$percent_change <- as.numeric((return_levels$absolute_change /
                                        abs(return_levels$baseline_rl)) * 100)

      # Initialize baseline_100yr_new_period column (always present for consistent rbind)
      return_levels$baseline_100yr_new_period <- NA_real_

      # Calculate return period shift (how often baseline 100-yr event now occurs)
      if (nrow(return_levels) > 0 && is_nonstat) {
        baseline_100yr <- return_levels$baseline_rl[return_levels$return_period == 100]
        if (!is.na(baseline_100yr) && !is.null(best_fit)) {
          # For non-stationary model at current year
          qcov_current <- make.qcov(best_fit,
                                    vals = list(mu1 = current_mid_year,
                                               sigma1 = current_mid_year))

          # Calculate return period of baseline 100-yr event in current climate
          # This is complex for non-stationary models, so we'll approximate
          # by finding exceedance probability
          tryCatch({
            params_current <- return.level(best_fit, return.period = c(10, 25, 50, 100),
                                          qcov = qcov_current)
            # Interpolate to find new return period
            # (simplified approximation)
            # For maxima: if baseline > current, event is rarer (longer return period)
            # For minima: if baseline > current, values are lower (more extreme), event is more common (shorter return period)
            baseline_100 <- return_levels$baseline_rl[return_levels$return_period == 100]
            current_100 <- return_levels$current_rl[return_levels$return_period == 100]

            if (extrema_type == "max") {
              # For maxima: ratio stays as is
              return_levels$baseline_100yr_new_period[return_levels$return_period == 100] <-
                100 * (baseline_100 / current_100)
            } else {
              # For minima: invert the ratio
              return_levels$baseline_100yr_new_period[return_levels$return_period == 100] <-
                100 * (current_100 / baseline_100)
            }
          }, error = function(e) {
            # Column already initialized to NA
          })
        }
      }

      # Append to results
      all_results <- rbind(all_results, return_levels)

      # Create diagnostic plots for key variables
      if (region == "Grand Total" && extrema_type == "max" &&
          var_name %in% c("mean_temperature_c", "proxy_fwi", "total_precipitation_mm")) {

        plot_title <- paste0(region, " - ", display_name, " (", extrema_label, ")")
        plot_filename <- file.path(plots_dir,
                                   paste0("gev_diagnostic_R_", gsub(" ", "_", tolower(region)),
                                         "_", gsub("_.*", "", var_name), "_", extrema_type, ".png"))

        create_diagnostic_plots(best_fit, plot_title, plot_filename)
        cat(paste0("  → Saved diagnostic plot: ", plot_filename, "\n"))
      }
    }
  }
}

################################################################################
# SAVE RESULTS
################################################################################

cat("\n================================================================================\n")
cat("SAVING RESULTS\n")
cat("================================================================================\n\n")

write.csv(all_results, output_file, row.names = FALSE)
cat(paste0("Saved results to: ", output_file, "\n"))
cat(paste0("Total analyses: ", nrow(all_results), "\n"))

################################################################################
# SUMMARY STATISTICS
################################################################################

cat("\n================================================================================\n")
cat("SUMMARY OF FINDINGS\n")
cat("================================================================================\n\n")

# Count non-stationary vs stationary models
model_summary <- all_results %>%
  group_by(model_type) %>%
  summarise(count = n(), .groups = "drop")

cat("Model Selection Summary:\n")
print(model_summary)

# Significant changes (100-year return period)
significant_changes <- all_results %>%
  filter(return_period == 100, abs(percent_change) > 20)

if (nrow(significant_changes) > 0) {
  cat("\n\nSignificant Changes in 100-year Return Levels (>20% change):\n")
  cat("----------------------------------------------------------------\n")
  for (i in 1:nrow(significant_changes)) {
    row <- significant_changes[i, ]
    cat(sprintf("%s | %s (%s) | Change: %+.1f%% (%.2f → %.2f)\n",
                row$region, row$variable, row$extrema_type,
                row$percent_change, row$baseline_rl, row$current_rl))
  }
}

cat("\n\nAnalysis complete!\n")
