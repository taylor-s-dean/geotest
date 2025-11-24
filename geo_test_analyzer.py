#!/usr/bin/env python3
"""
Geo Test Analysis Tool

Analyzes geo test experiments using multiple statistical methods:
- Difference-in-Differences (DiD)
- Synthetic Control
- T-test with Baseline Adjustment
- Bayesian Approach
"""

import argparse
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from dateutil import parser as date_parser
from bokeh.plotting import figure, output_file, save
from bokeh.models import (
    HoverTool,
    ColumnDataSource,
    Span,
    BoxAnnotation,
    Label,
    Div,
    LinearAxis,
    Range1d,
)
from bokeh.layouts import column, row
from bokeh.palettes import Category10

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

try:
    import pymc as pm
    import arviz as az

    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("Warning: PyMC not available. Bayesian method will be skipped.")


def parse_date(date_str: str) -> datetime:
    """Parse date string in M/D/YY format."""
    try:
        # Try M/D/YY format first
        parts = date_str.split("/")
        if len(parts) == 3:
            month, day, year = parts
            year = int(year)
            if year < 100:
                year += 2000 if year < 50 else 1900
            return datetime(year, int(month), int(day))
        # Fallback to dateutil parser
        return date_parser.parse(date_str)
    except Exception as e:
        raise ValueError(f"Could not parse date: {date_str}. Error: {e}")


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess CSV data."""
    df = pd.read_csv(csv_path)

    # Handle different column name formats
    date_col = df.columns[0]
    control_col = None
    test_col = None

    for col in df.columns:
        if col.lower() in ["control", "control group"]:
            control_col = col
        elif col.lower() in ["test", "test group"]:
            test_col = col

    if control_col is None or test_col is None:
        raise ValueError("CSV must contain 'Control' and 'Test' columns")

    # Parse dates
    df["date"] = df[date_col].apply(parse_date)
    df["control"] = pd.to_numeric(df[control_col], errors="coerce")
    df["test"] = pd.to_numeric(df[test_col], errors="coerce")

    # Remove rows with missing data
    df = df.dropna(subset=["date", "control", "test"])
    df = df.sort_values("date").reset_index(drop=True)

    return df[["date", "control", "test"]]


def split_periods(
    df: pd.DataFrame,
    pre_start: datetime,
    pre_end: datetime,
    exp_start: datetime,
    exp_end: datetime,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into pre-experiment and experiment periods."""
    pre_period = df[(df["date"] >= pre_start) & (df["date"] <= pre_end)].copy()
    exp_period = df[(df["date"] >= exp_start) & (df["date"] <= exp_end)].copy()

    if len(pre_period) == 0:
        raise ValueError("No data found in pre-experiment period")
    if len(exp_period) == 0:
        raise ValueError("No data found in experiment period")

    return pre_period, exp_period


def calculate_power(effect_size: float, n: int, alpha: float = 0.05) -> float:
    """Calculate statistical power for a two-sample test.

    Args:
        effect_size: Cohen's d effect size
        n: Sample size per group (assumes equal group sizes)
        alpha: Significance level (default: 0.05)

    Returns:
        Statistical power (probability of detecting effect if it exists)
    """
    try:
        from scipy.stats import norm

        z_alpha = norm.ppf(1 - alpha / 2)
        # For two-sample test: effect_size * sqrt(n/2)
        # This accounts for variance in both groups
        z_beta = abs(effect_size) * np.sqrt(n / 2) - z_alpha
        power = norm.cdf(z_beta)
        return max(0, min(1, power))
    except:
        return np.nan


# ============================================================================
# STATISTICAL METHODS
# ============================================================================


def difference_in_differences(
    pre_period: pd.DataFrame, exp_period: pd.DataFrame, alpha: float = 0.05
) -> Dict:
    """Difference-in-Differences analysis."""
    # Calculate pre-period difference
    pre_diff = pre_period["test"].mean() - pre_period["control"].mean()

    # Calculate experiment period difference
    exp_diff = exp_period["test"].mean() - exp_period["control"].mean()

    # DiD estimator
    did_estimate = exp_diff - pre_diff

    # Calculate standard errors using robust method
    pre_control_mean = pre_period["control"].mean()
    pre_test_mean = pre_period["test"].mean()
    exp_control_mean = exp_period["control"].mean()
    exp_test_mean = exp_period["test"].mean()

    # Variance components
    pre_var = pre_period["test"].var() / len(pre_period) + pre_period[
        "control"
    ].var() / len(pre_period)
    exp_var = exp_period["test"].var() / len(exp_period) + exp_period[
        "control"
    ].var() / len(exp_period)

    se = np.sqrt(pre_var + exp_var)

    # T-statistic and p-value
    t_stat = did_estimate / se if se > 0 else 0
    df = len(pre_period) + len(exp_period) - 2
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    # Confidence interval
    t_critical = stats.t.ppf(1 - alpha / 2, df)
    ci_lower = did_estimate - t_critical * se
    ci_upper = did_estimate + t_critical * se

    # Effect size (standardized)
    pooled_std = np.sqrt((pre_period["control"].var() + pre_period["test"].var()) / 2)
    effect_size = did_estimate / pooled_std if pooled_std > 0 else 0

    # Power calculation
    n = min(len(pre_period), len(exp_period))
    power = calculate_power(effect_size, n, alpha)

    return {
        "method": "Difference-in-Differences",
        "estimate": did_estimate,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "effect_size": effect_size,
        "power": power,
        "significant": p_value < alpha,
        "pre_diff": pre_diff,
        "exp_diff": exp_diff,
    }


def synthetic_control(
    pre_period: pd.DataFrame, exp_period: pd.DataFrame, alpha: float = 0.05
) -> Dict:
    """Synthetic Control method."""
    # Use control as donor pool (simplified - in practice would use multiple controls)
    # Fit weights to minimize pre-period prediction error
    control_pre = pre_period["control"].values
    test_pre = pre_period["test"].values

    # Simple approach: use control directly as synthetic control
    # (In full implementation, would optimize weights across multiple donors)
    synthetic_pre = control_pre.copy()
    synthetic_exp = exp_period["control"].values

    # Calculate treatment effect
    actual_exp = exp_period["test"].values
    treatment_effect = actual_exp.mean() - synthetic_exp.mean()

    # Permutation test: randomly assign observations to test/control
    n_permutations = 1000
    permuted_effects = []

    # Combine experiment period data for permutation
    combined_exp = np.concatenate([actual_exp, synthetic_exp])
    n_test = len(actual_exp)

    for _ in range(n_permutations):
        # Randomly permute the combined data
        perm_indices = np.random.permutation(len(combined_exp))
        perm_combined = combined_exp[perm_indices]
        # Split into "test" and "control"
        perm_test = perm_combined[:n_test]
        perm_control = perm_combined[n_test:]
        # Calculate permuted treatment effect
        perm_effect = perm_test.mean() - perm_control.mean()
        permuted_effects.append(perm_effect)

    # P-value from permutation test (two-tailed)
    p_value = np.mean(np.abs(permuted_effects) >= abs(treatment_effect))

    # Confidence interval from permutation distribution
    ci_lower = np.percentile(permuted_effects, alpha / 2 * 100)
    ci_upper = np.percentile(permuted_effects, (1 - alpha / 2) * 100)

    # Effect size
    pooled_std = np.sqrt((pre_period["control"].var() + pre_period["test"].var()) / 2)
    effect_size = treatment_effect / pooled_std if pooled_std > 0 else 0

    # Power
    n = len(exp_period)
    power = calculate_power(effect_size, n, alpha)

    return {
        "method": "Synthetic Control",
        "estimate": treatment_effect,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "effect_size": effect_size,
        "power": power,
        "significant": p_value < alpha,
        "synthetic_series": synthetic_exp,
    }


def ttest_baseline_adjusted(
    pre_period: pd.DataFrame, exp_period: pd.DataFrame, alpha: float = 0.05
) -> Dict:
    """T-test with baseline adjustment."""
    # Normalize by pre-period characteristics
    pre_control_mean = pre_period["control"].mean()
    pre_test_mean = pre_period["test"].mean()
    pre_control_std = pre_period["control"].std()
    pre_test_std = pre_period["test"].std()

    # Baseline adjustment: subtract pre-period means
    exp_control_adj = exp_period["control"].values - pre_control_mean
    exp_test_adj = exp_period["test"].values - pre_test_mean

    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(exp_test_adj, exp_control_adj)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((pre_control_std**2 + pre_test_std**2) / 2)
    mean_diff = exp_test_adj.mean() - exp_control_adj.mean()
    effect_size = mean_diff / pooled_std if pooled_std > 0 else 0

    # Confidence interval
    n1, n2 = len(exp_test_adj), len(exp_control_adj)
    se_diff = np.sqrt(exp_test_adj.var() / n1 + exp_control_adj.var() / n2)
    df = n1 + n2 - 2
    t_critical = stats.t.ppf(1 - alpha / 2, df)
    ci_lower = mean_diff - t_critical * se_diff
    ci_upper = mean_diff + t_critical * se_diff

    # Power
    n = min(n1, n2)
    power = calculate_power(effect_size, n, alpha)

    return {
        "method": "T-test (Baseline Adjusted)",
        "estimate": mean_diff,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "effect_size": effect_size,
        "power": power,
        "significant": p_value < alpha,
    }


def bayesian_analysis(
    pre_period: pd.DataFrame, exp_period: pd.DataFrame, alpha: float = 0.05
) -> Dict:
    """Bayesian analysis using PyMC."""
    if not BAYESIAN_AVAILABLE:
        return {
            "method": "Bayesian",
            "estimate": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "effect_size": np.nan,
            "power": np.nan,
            "significant": False,
            "error": "PyMC not available",
        }

    try:
        # Prepare data
        pre_control = pre_period["control"].values
        pre_test = pre_period["test"].values
        exp_control = exp_period["control"].values
        exp_test = exp_period["test"].values

        with pm.Model() as model:
            # Priors from pre-period
            pre_control_mean = pre_control.mean()
            pre_test_mean = pre_test.mean()
            pre_control_std = pre_control.std()
            pre_test_std = pre_test.std()

            # Prior for control group (experiment period)
            mu_control = pm.Normal(
                "mu_control", mu=pre_control_mean, sigma=pre_control_std
            )
            sigma_control = pm.HalfNormal("sigma_control", sigma=pre_control_std)

            # Prior for test group (experiment period)
            mu_test = pm.Normal("mu_test", mu=pre_test_mean, sigma=pre_test_std)
            sigma_test = pm.HalfNormal("sigma_test", sigma=pre_test_std)

            # Likelihood
            control_obs = pm.Normal(
                "control_obs", mu=mu_control, sigma=sigma_control, observed=exp_control
            )
            test_obs = pm.Normal(
                "test_obs", mu=mu_test, sigma=sigma_test, observed=exp_test
            )

            # Treatment effect
            treatment_effect = pm.Deterministic(
                "treatment_effect", mu_test - mu_control
            )

            # Sample
            trace = pm.sample(
                2000, tune=1000, return_inferencedata=True, progressbar=False
            )

        # Extract results
        posterior = az.extract(trace, var_names=["treatment_effect"])
        estimate = float(posterior.mean())
        ci_lower = float(posterior.quantile(alpha / 2))
        ci_upper = float(posterior.quantile(1 - alpha / 2))

        # P-value (proportion of posterior < 0 or > 0 depending on sign)
        if estimate > 0:
            p_value = float((posterior < 0).mean())
        else:
            p_value = float((posterior > 0).mean())
        p_value = 2 * p_value  # Two-tailed

        # Effect size
        pooled_std = np.sqrt((pre_control_std**2 + pre_test_std**2) / 2)
        effect_size = estimate / pooled_std if pooled_std > 0 else 0

        # Power (approximate)
        n = len(exp_period)
        power = calculate_power(effect_size, n, alpha)

        return {
            "method": "Bayesian",
            "estimate": estimate,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "effect_size": effect_size,
            "power": power,
            "significant": p_value < alpha,
            "trace": trace,
        }
    except Exception as e:
        return {
            "method": "Bayesian",
            "estimate": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "effect_size": np.nan,
            "power": np.nan,
            "significant": False,
            "error": str(e),
        }


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================


def calculate_time_varying_stats(
    method_name: str,
    df: pd.DataFrame,
    pre_period: pd.DataFrame,
    exp_period: pd.DataFrame,
    exp_start: datetime,
    exp_end: datetime,
    alpha: float,
) -> Dict:
    """Calculate time-varying statistics for a given method."""
    date_series = pd.to_datetime(df["date"])
    exp_mask = (date_series >= exp_start) & (date_series <= exp_end)
    exp_indices = np.where(exp_mask.values)[0]

    n = len(df)
    p_values = np.full(n, np.nan)
    ci_lower = np.full(n, np.nan)
    ci_upper = np.full(n, np.nan)
    estimates = np.full(n, np.nan)
    effect_sizes = np.full(n, np.nan)
    power_values = np.full(n, np.nan)
    is_significant = np.full(n, False)

    pre_control_mean = pre_period["control"].mean()
    pre_test_mean = pre_period["test"].mean()
    pre_diff = pre_test_mean - pre_control_mean

    for i, idx in enumerate(exp_indices):
        exp_subset = exp_period.iloc[: i + 1] if i < len(exp_period) else exp_period

        if len(exp_subset) < 2:
            continue

        try:
            if method_name == "Difference-in-Differences":
                exp_control_mean = exp_subset["control"].mean()
                exp_test_mean = exp_subset["test"].mean()
                exp_diff = exp_test_mean - exp_control_mean
                did_estimate = exp_diff - pre_diff

                # Standard error
                pre_var = pre_period["test"].var() / len(pre_period) + pre_period[
                    "control"
                ].var() / len(pre_period)
                exp_var = exp_subset["test"].var() / len(exp_subset) + exp_subset[
                    "control"
                ].var() / len(exp_subset)
                se = np.sqrt(pre_var + exp_var)

                if se > 0:
                    t_stat = did_estimate / se
                    df_val = len(pre_period) + len(exp_subset) - 2
                    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df_val))
                    t_critical = stats.t.ppf(1 - alpha / 2, df_val)
                    ci_lower[idx] = did_estimate - t_critical * se
                    ci_upper[idx] = did_estimate + t_critical * se
                    estimates[idx] = did_estimate
                    p_values[idx] = p_val
                    is_significant[idx] = p_val < alpha

                    pooled_std = np.sqrt(
                        (pre_period["control"].var() + pre_period["test"].var()) / 2
                    )
                    effect_size = did_estimate / pooled_std if pooled_std > 0 else 0
                    effect_sizes[idx] = effect_size
                    power_values[idx] = calculate_power(
                        effect_size, len(exp_subset), alpha
                    )

            elif method_name == "Synthetic Control":
                # Simplified synthetic control
                control_exp = exp_subset["control"].values
                test_exp = exp_subset["test"].values
                treatment_effect = test_exp.mean() - control_exp.mean()

                # Simple t-test for p-value
                _, p_val = stats.ttest_ind(test_exp, control_exp)
                se = stats.sem(test_exp - control_exp) if len(test_exp) > 1 else np.nan

                if not np.isnan(se) and se > 0:
                    df_val = len(exp_subset) - 1
                    t_critical = stats.t.ppf(1 - alpha / 2, df_val)
                    ci_lower[idx] = treatment_effect - t_critical * se
                    ci_upper[idx] = treatment_effect + t_critical * se
                    estimates[idx] = treatment_effect
                    p_values[idx] = p_val
                    is_significant[idx] = p_val < alpha

                    pooled_std = np.sqrt(
                        (exp_subset["control"].var() + exp_subset["test"].var()) / 2
                    )
                    effect_size = treatment_effect / pooled_std if pooled_std > 0 else 0
                    effect_sizes[idx] = effect_size
                    power_values[idx] = calculate_power(
                        effect_size, len(exp_subset), alpha
                    )

            elif method_name == "T-test (Baseline Adjusted)":
                pre_control_mean = pre_period["control"].mean()
                pre_test_mean = pre_period["test"].mean()

                exp_control_adj = exp_subset["control"].values - pre_control_mean
                exp_test_adj = exp_subset["test"].values - pre_test_mean

                _, p_val = stats.ttest_ind(exp_test_adj, exp_control_adj)
                mean_diff = exp_test_adj.mean() - exp_control_adj.mean()

                n1, n2 = len(exp_test_adj), len(exp_control_adj)
                se_diff = np.sqrt(exp_test_adj.var() / n1 + exp_control_adj.var() / n2)
                df_val = n1 + n2 - 2
                t_critical = stats.t.ppf(1 - alpha / 2, df_val)

                ci_lower[idx] = mean_diff - t_critical * se_diff
                ci_upper[idx] = mean_diff + t_critical * se_diff
                estimates[idx] = mean_diff
                p_values[idx] = p_val
                is_significant[idx] = p_val < alpha

                pooled_std = np.sqrt(
                    (pre_period["control"].std() ** 2 + pre_period["test"].std() ** 2)
                    / 2
                )
                effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
                effect_sizes[idx] = effect_size
                power_values[idx] = calculate_power(effect_size, min(n1, n2), alpha)

            elif method_name == "Bayesian":
                # Simplified Bayesian - use t-test approximation for time-varying
                _, p_val = stats.ttest_ind(exp_subset["test"], exp_subset["control"])
                mean_diff = exp_subset["test"].mean() - exp_subset["control"].mean()

                # Approximate CI
                se = stats.sem(exp_subset["test"] - exp_subset["control"])
                if not np.isnan(se) and se > 0:
                    df_val = len(exp_subset) - 1
                    t_critical = stats.t.ppf(1 - alpha / 2, df_val)
                    ci_lower[idx] = mean_diff - t_critical * se
                    ci_upper[idx] = mean_diff + t_critical * se
                    estimates[idx] = mean_diff
                    p_values[idx] = p_val
                    is_significant[idx] = p_val < alpha

                    pooled_std = np.sqrt(
                        (exp_subset["control"].var() + exp_subset["test"].var()) / 2
                    )
                    effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
                    effect_sizes[idx] = effect_size
                    power_values[idx] = calculate_power(
                        effect_size, len(exp_subset), alpha
                    )

        except Exception:
            continue

    return {
        "p_values": p_values,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "estimates": estimates,
        "effect_sizes": effect_sizes,
        "power": power_values,
        "is_significant": is_significant,
    }


def create_summary_box(
    result: Dict,
    pre_period: pd.DataFrame = None,
    exp_period: pd.DataFrame = None,
    dark_mode: bool = False,
) -> Div:
    """Create a summary box for a method's results."""

    method = result["method"]
    estimate = result.get("estimate", np.nan)
    p_val = result.get("p_value", np.nan)
    sig = result.get("significant", False)
    ci_lower = result.get("ci_lower", np.nan)
    ci_upper = result.get("ci_upper", np.nan)
    effect_size = result.get("effect_size", np.nan)
    power = result.get("power", np.nan)

    # Calculate lift CI if we have the necessary data
    lift_ci_lower = np.nan
    lift_ci_upper = np.nan
    if pre_period is not None and exp_period is not None:
        pre_control_mean = pre_period["control"].mean()
        pre_test_mean = pre_period["test"].mean()
        pre_ratio = pre_test_mean / pre_control_mean if pre_control_mean > 0 else 1.0
        exp_control_mean = exp_period["control"].mean()
        expected_test_mean = exp_control_mean * pre_ratio

        if expected_test_mean > 0 and not np.isnan(ci_lower) and not np.isnan(ci_upper):
            # Different methods calculate CI differently
            if method in ["Difference-in-Differences", "T-test (Baseline Adjusted)"]:
                # CI is already relative to counterfactual, so just divide by expected_test
                lift_ci_lower = (ci_lower / expected_test_mean) * 100
                lift_ci_upper = (ci_upper / expected_test_mean) * 100
            elif method in ["Synthetic Control", "Bayesian"]:
                # CI is for raw difference (test - control)
                # Need to convert to (test - expected_test) = (test - control) - (expected_test - control)
                adjustment = exp_control_mean * (pre_ratio - 1)
                lift_ci_lower = ((ci_lower - adjustment) / expected_test_mean) * 100
                lift_ci_upper = ((ci_upper - adjustment) / expected_test_mean) * 100
            else:
                # Default: assume CI is relative to counterfactual
                lift_ci_lower = (ci_lower / expected_test_mean) * 100
                lift_ci_upper = (ci_upper / expected_test_mean) * 100

    # Format values
    estimate_str = f"{estimate:.4f}" if not np.isnan(estimate) else "N/A"
    p_val_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
    sig_str = "✓ Yes" if sig else "✗ No"
    sig_color = "#4CAF50" if sig else "#F44336"
    ci_lower_str = f"{ci_lower:.4f}" if not np.isnan(ci_lower) else "N/A"
    ci_upper_str = f"{ci_upper:.4f}" if not np.isnan(ci_upper) else "N/A"
    lift_ci_lower_str = (
        f"{lift_ci_lower:+.2f}%" if not np.isnan(lift_ci_lower) else "N/A"
    )
    lift_ci_upper_str = (
        f"{lift_ci_upper:+.2f}%" if not np.isnan(lift_ci_upper) else "N/A"
    )
    effect_size_str = f"{effect_size:.4f}" if not np.isnan(effect_size) else "N/A"
    power_str = f"{power:.4f}" if not np.isnan(power) else "N/A"

    # Set colors based on mode
    if dark_mode:
        bg_color = "#1e1e1e"
        text_color = "#e0e0e0"
        border_color = "#404040"
        label_color = "#b0b0b0"
    else:
        bg_color = "#ffffff"
        text_color = "#000000"
        border_color = "#cccccc"
        label_color = "#666666"

    # Create HTML content
    html_content = f"""
    <div style="background-color: {bg_color}; color: {text_color}; padding: 15px; 
                border: 1px solid {border_color}; border-radius: 5px; 
                font-family: monospace; font-size: 12px; width: 300px; height: 300px;">
        <div style="margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid {border_color};">
            <strong style="font-size: 13px;">Results Summary</strong>
        </div>
        
        <div style="margin-bottom: 8px;">
            <span style="color: {label_color};">Method:</span><br>
            <strong>{method}</strong>
        </div>
        
        <div style="margin-bottom: 8px;">
            <span style="color: {label_color};">Estimate:</span> <strong>{estimate_str}</strong>
        </div>
        
        <div style="margin-bottom: 8px;">
            <span style="color: {label_color};">P-value:</span> <strong>{p_val_str}</strong>
        </div>
        
        <div style="margin-bottom: 8px;">
            <span style="color: {label_color};">Significant:</span> 
            <strong style="color: {sig_color};">{sig_str}</strong>
        </div>
        
        <div style="margin-bottom: 8px;">
            <span style="color: {label_color};">95% CI:</span><br>
            <strong>[{ci_lower_str}, {ci_upper_str}]</strong>
        </div>
        
        <div style="margin-bottom: 8px;">
            <span style="color: {label_color};">Lift 95% CI:</span><br>
            <strong>[{lift_ci_lower_str}, {lift_ci_upper_str}]</strong>
        </div>
        
        <div style="margin-bottom: 8px;">
            <span style="color: {label_color};">Effect Size:</span> <strong>{effect_size_str}</strong>
        </div>
        
        <div style="margin-bottom: 8px;">
            <span style="color: {label_color};">Power:</span> <strong>{power_str}</strong>
        </div>
    </div>
    """

    summary_box = Div(text=html_content, width=320, height=320)
    return summary_box


def create_method_chart(
    method_name: str,
    result: Dict,
    df: pd.DataFrame,
    pre_period: pd.DataFrame,
    exp_period: pd.DataFrame,
    pre_start: datetime,
    pre_end: datetime,
    exp_start: datetime,
    exp_end: datetime,
    alpha: float,
    dark_mode: bool = False,
) -> Tuple[figure, Div]:
    """Create a chart for a specific statistical method."""

    # Calculate time-varying statistics
    time_stats = calculate_time_varying_stats(
        method_name, df, pre_period, exp_period, exp_start, exp_end, alpha
    )

    # Prepare data
    dates = pd.to_datetime(df["date"]).values
    control_values = df["control"].values
    test_values = df["test"].values
    date_strs = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").values

    # Calculate % change relative to expected value based on pre-period relationship
    pre_control_mean = pre_period["control"].mean()
    pre_test_mean = pre_period["test"].mean()
    pre_ratio = pre_test_mean / pre_control_mean if pre_control_mean > 0 else 1.0

    # Expected test value = current control * pre-period ratio (counterfactual)
    expected_test = control_values * pre_ratio
    # % change = (actual - expected) / expected * 100
    daily_pct_dev = ((test_values - expected_test) / expected_test) * 100

    # Convert confidence intervals to % lift
    # Different methods calculate CI differently:
    # - DiD and T-test: CI is for treatment effect relative to counterfactual (already adjusted)
    # - Synthetic Control and Bayesian: CI is for raw difference (test - control)
    ci_lower_pct = np.full(len(dates), np.nan)
    ci_upper_pct = np.full(len(dates), np.nan)

    # Only convert where we have valid CI and non-zero expected values
    valid_ci_mask = ~(
        np.isnan(time_stats["ci_lower"]) | np.isnan(time_stats["ci_upper"])
    )
    valid_expected_mask = expected_test > 0

    combined_mask = valid_ci_mask & valid_expected_mask
    if np.any(combined_mask):
        if method_name in ["Difference-in-Differences", "T-test (Baseline Adjusted)"]:
            # CI is already relative to counterfactual, so just divide by expected_test
            ci_lower_pct[combined_mask] = (
                time_stats["ci_lower"][combined_mask] / expected_test[combined_mask]
            ) * 100
            ci_upper_pct[combined_mask] = (
                time_stats["ci_upper"][combined_mask] / expected_test[combined_mask]
            ) * 100
        elif method_name in ["Synthetic Control", "Bayesian"]:
            # CI is for raw difference (test - control)
            # Need to convert to (test - expected_test) = (test - control) - (expected_test - control)
            # where expected_test - control = control * (pre_ratio - 1)
            adjustment = control_values[combined_mask] * (pre_ratio - 1)
            ci_lower_pct[combined_mask] = (
                (time_stats["ci_lower"][combined_mask] - adjustment)
                / expected_test[combined_mask]
            ) * 100
            ci_upper_pct[combined_mask] = (
                (time_stats["ci_upper"][combined_mask] - adjustment)
                / expected_test[combined_mask]
            ) * 100
        else:
            # Default: assume CI is relative to counterfactual
            ci_lower_pct[combined_mask] = (
                time_stats["ci_lower"][combined_mask] / expected_test[combined_mask]
            ) * 100
            ci_upper_pct[combined_mask] = (
                time_stats["ci_upper"][combined_mask] / expected_test[combined_mask]
            ) * 100

    # Create source with method-specific metrics
    source = ColumnDataSource(
        data={
            "date": dates,
            "control": control_values,
            "test": test_values,
            "counterfactual": expected_test,
            "pct_dev": daily_pct_dev,
            "p_value": time_stats["p_values"],
            "ci_lower": time_stats["ci_lower"],
            "ci_upper": time_stats["ci_upper"],
            "ci_lower_pct": ci_lower_pct,
            "ci_upper_pct": ci_upper_pct,
            "estimate": time_stats["estimates"],
            "effect_size": time_stats["effect_sizes"],
            "power": time_stats["power"],
            "is_significant": time_stats["is_significant"],
            "date_str": date_strs,
            "sig_color": [
                "green" if sig else "red" for sig in time_stats["is_significant"]
            ],
        }
    )

    # Set colors based on mode
    if dark_mode:
        bg_color = "#1e1e1e"
        grid_color = "#404040"
        text_color = "#e0e0e0"
        border_color = "#404040"
        tooltip_bg = "#2d2d2d"
        tooltip_border = "#505050"
        control_color = "#5DA5DA"  # Lighter blue for dark mode
        test_color = "#FAA43A"  # Lighter orange for dark mode
        counterfactual_color = "#60BD68"  # Lighter green for dark mode
    else:
        bg_color = "#ffffff"
        grid_color = "#e0e0e0"
        text_color = "#000000"
        border_color = "#cccccc"
        tooltip_bg = "white"
        tooltip_border = "#ddd"
        control_color = Category10[10][0]
        test_color = Category10[10][1]
        counterfactual_color = Category10[10][2]  # Green

    # Create figure
    p = figure(
        width=1000,
        height=500,
        x_axis_type="datetime",
        title=f"{method_name} Analysis",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        background_fill_color=bg_color,
        border_fill_color=bg_color,
    )

    # Add period shading
    pre_box = BoxAnnotation(
        left=pre_start, right=pre_end, fill_alpha=0.1, fill_color="blue"
    )
    exp_box = BoxAnnotation(
        left=exp_start, right=exp_end, fill_alpha=0.1, fill_color="orange"
    )
    p.add_layout(pre_box)
    p.add_layout(exp_box)

    # Get y-axis range
    y_max = max(control_values.max(), test_values.max())
    y_min = min(control_values.min(), test_values.min())
    y_range = y_max - y_min

    # Add period labels
    pre_label = Label(
        x=pre_start,
        y=y_max - y_range * 0.05,
        text="Pre-Experiment",
        text_font_size="10pt",
        text_color="blue",
    )
    exp_label = Label(
        x=exp_start,
        y=y_max - y_range * 0.05,
        text="Experiment",
        text_font_size="10pt",
        text_color="orange",
    )
    p.add_layout(pre_label)
    p.add_layout(exp_label)

    # Add vertical line at experiment start
    exp_start_line = Span(
        location=exp_start,
        dimension="height",
        line_color="red",
        line_dash="dashed",
        line_width=2,
    )
    p.add_layout(exp_start_line)

    # Set up secondary y-axis for % change
    # Include CI bounds in range calculation
    pct_min = daily_pct_dev.min()
    pct_max = daily_pct_dev.max()

    # Include CI bounds if available
    valid_ci_pct_mask = ~(np.isnan(ci_lower_pct) | np.isnan(ci_upper_pct))
    if np.any(valid_ci_pct_mask):
        pct_min = min(pct_min, np.nanmin(ci_lower_pct[valid_ci_pct_mask]))
        pct_max = max(pct_max, np.nanmax(ci_upper_pct[valid_ci_pct_mask]))

    pct_range = pct_max - pct_min
    pct_padding = pct_range * 0.1 if pct_range > 0 else 1

    p.extra_y_ranges = {
        "pct_change": Range1d(start=pct_min - pct_padding, end=pct_max + pct_padding)
    }
    pct_axis = LinearAxis(y_range_name="pct_change", axis_label="Lift (%)")
    p.add_layout(pct_axis, "right")

    # Plot control and test lines on main y-axis
    control_line = p.line(
        "date",
        "control",
        source=source,
        legend_label="Control",
        line_width=2,
        color=control_color,
    )
    test_line = p.line(
        "date",
        "test",
        source=source,
        legend_label="Test",
        line_width=2,
        color=test_color,
    )

    # Plot counterfactual (expected test value) on main y-axis
    p.line(
        "date",
        "counterfactual",
        source=source,
        legend_label="Counterfactual (Expected Test)",
        line_width=2,
        color=counterfactual_color,
        line_dash="dotted",
        alpha=0.8,
    )

    # Plot lift on secondary y-axis
    pct_color = "#9467bd" if not dark_mode else "#c5b0d5"  # Purple color
    lift_line = p.line(
        "date",
        "pct_dev",
        source=source,
        legend_label="Lift (%)",
        line_width=1.5,
        color=pct_color,
        alpha=0.7,
        y_range_name="pct_change",
        line_dash="dashed",
    )

    # Add confidence interval band and significance shading (for experiment period only)
    exp_mask = (pd.to_datetime(df["date"]) >= exp_start) & (
        pd.to_datetime(df["date"]) <= exp_end
    )
    exp_dates = dates[exp_mask]
    exp_ci_lower = time_stats["ci_lower"][exp_mask]
    exp_ci_upper = time_stats["ci_upper"][exp_mask]
    exp_estimates = time_stats["estimates"][exp_mask]
    exp_is_sig = time_stats["is_significant"][exp_mask]

    # Add confidence interval band for lift (%)
    exp_ci_lower_pct = ci_lower_pct[exp_mask]
    exp_ci_upper_pct = ci_upper_pct[exp_mask]

    # Add CI band for lift percentage (on secondary y-axis)
    valid_pct_mask = ~(np.isnan(exp_ci_lower_pct) | np.isnan(exp_ci_upper_pct))
    if np.any(valid_pct_mask):
        lift_ci_source = ColumnDataSource(
            data={
                "date": exp_dates[valid_pct_mask],
                "ci_lower_pct": exp_ci_lower_pct[valid_pct_mask],
                "ci_upper_pct": exp_ci_upper_pct[valid_pct_mask],
            }
        )
        p.varea(
            "date",
            y1="ci_lower_pct",
            y2="ci_upper_pct",
            source=lift_ci_source,
            alpha=0.2,
            color=pct_color,
            y_range_name="pct_change",
            legend_label="95% CI (Lift %)",
        )

    # Add significance period shading
    sig_periods = []
    in_sig_period = False
    sig_start = None
    for i in range(len(exp_dates)):
        date = exp_dates[i]
        is_sig = exp_is_sig[i]
        if is_sig and not in_sig_period:
            sig_start = date
            in_sig_period = True
        elif not is_sig and in_sig_period:
            if sig_start is not None and i > 0:
                sig_periods.append((sig_start, exp_dates[i - 1]))
            in_sig_period = False
            sig_start = None
    if in_sig_period and sig_start is not None:
        sig_periods.append((sig_start, exp_dates[-1]))

    for sig_start_date, sig_end_date in sig_periods:
        sig_box = BoxAnnotation(
            left=sig_start_date,
            right=sig_end_date,
            fill_alpha=0.15,
            fill_color="green",
            line_color="green",
            line_alpha=0.3,
        )
        p.add_layout(sig_box)

    # Add CI band (showing uncertainty around treatment effect)
    # Different methods calculate CI differently, so we need method-specific logic
    valid_mask = ~(np.isnan(exp_ci_lower) | np.isnan(exp_ci_upper))
    if np.any(valid_mask):
        # Get counterfactual and control values for experiment period
        exp_counterfactual = expected_test[exp_mask][valid_mask]
        exp_control = control_values[exp_mask][valid_mask]

        # CI represents uncertainty in the treatment effect
        # Different methods interpret CI differently:
        # - DiD and T-test: CI is for treatment effect relative to baseline → add to counterfactual
        # - Synthetic Control and Bayesian: CI is for raw difference (test - control) → add to control
        if method_name in ["Difference-in-Differences", "T-test (Baseline Adjusted)"]:
            # CI is for treatment effect relative to baseline difference
            # Shows uncertainty around counterfactual-adjusted treatment effect
            ci_y1 = exp_counterfactual + exp_ci_lower[valid_mask]
            ci_y2 = exp_counterfactual + exp_ci_upper[valid_mask]
        elif method_name in ["Synthetic Control", "Bayesian"]:
            # CI is for raw treatment effect (test - control)
            # Shows uncertainty around absolute difference
            # Visualize as: control + treatment_effect = test value range
            ci_y1 = exp_control + exp_ci_lower[valid_mask]
            ci_y2 = exp_control + exp_ci_upper[valid_mask]
        else:
            # Default: add to counterfactual
            ci_y1 = exp_counterfactual + exp_ci_lower[valid_mask]
            ci_y2 = exp_counterfactual + exp_ci_upper[valid_mask]

        ci_source = ColumnDataSource(
            data={
                "date": exp_dates[valid_mask],
                "ci_lower": exp_ci_lower[valid_mask],
                "ci_upper": exp_ci_upper[valid_mask],
                "estimate": exp_estimates[valid_mask],
                "counterfactual": exp_counterfactual,
                "y1": ci_y1,
                "y2": ci_y2,
            }
        )
        p.varea(
            "date",
            y1="y1",
            y2="y2",
            source=ci_source,
            alpha=0.2,
            color="gray",
            legend_label="95% CI (Effect)",
        )

    # Add scatter points with color coding for significance
    p.circle("date", "control", source=source, size=4, color=control_color, alpha=0.6)
    p.circle("date", "test", source=source, size=4, color=test_color, alpha=0.6)
    p.circle(
        "date",
        "counterfactual",
        source=source,
        size=3,
        color=counterfactual_color,
        alpha=0.5,
    )

    # Add significance indicators on test line
    sig_dates = dates[time_stats["is_significant"]]
    sig_test_values = test_values[time_stats["is_significant"]]
    if len(sig_dates) > 0:
        p.diamond(
            sig_dates,
            sig_test_values,
            size=12,
            color="green",
            alpha=0.8,
            line_color="darkgreen",
            line_width=2,
            legend_label="Significant (p<0.05)",
        )

    # Add non-significant indicators
    non_sig_mask = ~time_stats["is_significant"] & ~np.isnan(time_stats["p_values"])
    non_sig_mask = non_sig_mask & exp_mask
    if np.any(non_sig_mask):
        non_sig_dates = dates[non_sig_mask]
        non_sig_test_values = test_values[non_sig_mask]
        p.diamond(
            non_sig_dates,
            non_sig_test_values,
            size=8,
            color="red",
            alpha=0.5,
            legend_label="Not Significant",
        )

    # Single tooltip with key statistical values - attach only to test line
    tooltips = f"""
        <div style="background-color: {tooltip_bg}; color: {text_color}; padding: 10px; border: 1px solid {tooltip_border}; border-radius: 5px;">
            <div style="margin-bottom: 5px;"><strong>@date_str</strong></div>
            <hr style="margin: 5px 0; border-color: {tooltip_border};">
            <div><strong>Test:</strong> @test{{0,0}}</div>
            <div><strong>Counterfactual:</strong> @counterfactual{{0,0}}</div>
            <div><strong>Control:</strong> @control{{0,0}}</div>
            <div><strong>Lift:</strong> @pct_dev{{+0.1f}}%</div>
            <div><strong>Lift 95% CI:</strong> [@ci_lower_pct{{+0.1f}}%, @ci_upper_pct{{+0.1f}}%]</div>
            <hr style="margin: 5px 0; border-color: {tooltip_border};">
            <div><strong>P-value:</strong> @p_value{{0.4f}}</div>
            <div><strong>Power:</strong> @power{{0.3f}}</div>
            <div><strong>Significant:</strong> @is_significant</div>
        </div>
    """

    hover = HoverTool(tooltips=tooltips, renderers=[test_line, lift_line])
    p.add_tools(hover)

    # Style
    p.xaxis.axis_label = "Date"
    p.yaxis.axis_label = "Metric Value"
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    # Apply dark mode styling
    if dark_mode:
        # Style the secondary y-axis
        pct_axis.axis_line_color = text_color
        pct_axis.major_tick_line_color = text_color
        pct_axis.minor_tick_line_color = text_color
        pct_axis.major_label_text_color = text_color
        pct_axis.axis_label_text_color = text_color

        # Style primary axes and other elements
        p.xgrid.grid_line_color = grid_color
        p.ygrid.grid_line_color = grid_color
        p.xaxis.axis_line_color = text_color
        p.yaxis.axis_line_color = text_color
        p.xaxis.major_tick_line_color = text_color
        p.yaxis.major_tick_line_color = text_color
        p.xaxis.minor_tick_line_color = text_color
        p.yaxis.minor_tick_line_color = text_color
        p.xaxis.major_label_text_color = text_color
        p.yaxis.major_label_text_color = text_color
        p.xaxis.axis_label_text_color = text_color
        p.yaxis.axis_label_text_color = text_color
        p.title.text_color = text_color
        p.legend.background_fill_color = bg_color
        p.legend.border_line_color = border_color
        p.legend.label_text_color = text_color
        p.outline_line_color = border_color

    # Create summary box
    summary_box = create_summary_box(result, pre_period, exp_period, dark_mode)

    return p, summary_box


# ============================================================================
# VISUALIZATION
# ============================================================================


def create_visualization(
    df: pd.DataFrame,
    pre_period: pd.DataFrame,
    exp_period: pd.DataFrame,
    results: List[Dict],
    pre_start: datetime,
    pre_end: datetime,
    exp_start: datetime,
    exp_end: datetime,
    alpha: float,
    output_path: str,
    dark_mode: bool = False,
):
    """Create interactive Bokeh visualizations - one chart per method."""

    # Set body background color based on mode
    bg_color = "#121212" if dark_mode else "#ffffff"
    text_color = "#e0e0e0" if dark_mode else "#000000"

    charts = []

    # Create a chart for each method
    for result in results:
        method_name = result["method"]
        if "error" in result and result["error"]:
            continue

        chart, table = create_method_chart(
            method_name,
            result,
            df,
            pre_period,
            exp_period,
            pre_start,
            pre_end,
            exp_start,
            exp_end,
            alpha,
            dark_mode,
        )

        # Add a title div for the method
        title_style = f"color: {text_color}; background-color: {bg_color};"
        title_div = Div(
            text=f"<h2 style='{title_style}'>{method_name}</h2>", width=1000, height=40
        )

        # Combine chart and table in a row
        method_layout = row(chart, table)

        # Combine title and chart/table
        full_layout = column(title_div, method_layout)
        charts.append(full_layout)

    # Add spacing between charts and combine all charts vertically
    final_layout = column(*charts)

    # Save with custom template for dark mode
    output_file(output_path)

    if dark_mode:
        # Inject custom CSS for dark mode
        from bokeh.resources import CDN
        from bokeh.embed import file_html

        html = file_html(final_layout, CDN, "Geo Test Analysis")
        # Add dark mode styling
        dark_css = """
        <style>
            body { background-color: #121212 !important; color: #e0e0e0 !important; }
            .bk-root { background-color: #121212 !important; }
        </style>
        """
        html = html.replace("</head>", dark_css + "</head>")

        with open(output_path, "w") as f:
            f.write(html)
    else:
        save(final_layout)

    return final_layout


# ============================================================================
# CLI INTERFACE
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Geo Test Analysis Tool - Statistical analysis of geo test experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to CSV file with date, Control, and Test columns",
    )
    parser.add_argument(
        "--pre-start", required=True, help="Pre-experiment start date (M/D/YY format)"
    )
    parser.add_argument(
        "--pre-end", required=True, help="Pre-experiment end date (M/D/YY format)"
    )
    parser.add_argument(
        "--exp-start", required=True, help="Experiment start date (M/D/YY format)"
    )
    parser.add_argument(
        "--exp-end",
        required=True,
        help="Experiment end date / post period end (M/D/YY format)",
    )
    parser.add_argument(
        "--method",
        nargs="+",
        default=["all"],
        choices=["did", "synthetic", "ttest", "bayesian", "all"],
        help="Statistical method(s) to use",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Significance level (default: 0.05)"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="geo_test_results.html",
        help="Output HTML file path (default: geo_test_results.html)",
    )
    parser.add_argument(
        "--show", action="store_true", help="Open chart in browser after generation"
    )
    parser.add_argument(
        "--dark-mode",
        action="store_true",
        help="Use dark mode theme for visualizations",
    )

    args = parser.parse_args()

    # Parse dates
    try:
        pre_start = parse_date(args.pre_start)
        pre_end = parse_date(args.pre_end)
        exp_start = parse_date(args.exp_start)
        exp_end = parse_date(args.exp_end)
    except ValueError as e:
        print(f"Error parsing dates: {e}", file=sys.stderr)
        sys.exit(1)

    # Load data
    try:
        df = load_data(args.input)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    # Split periods
    try:
        pre_period, exp_period = split_periods(
            df, pre_start, pre_end, exp_start, exp_end
        )
    except ValueError as e:
        print(f"Error splitting periods: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine which methods to run
    methods_to_run = []
    if "all" in args.method:
        methods_to_run = ["did", "synthetic", "ttest", "bayesian"]
    else:
        methods_to_run = args.method

    # Run analyses
    results = []

    if "did" in methods_to_run:
        print("Running Difference-in-Differences analysis...")
        results.append(difference_in_differences(pre_period, exp_period, args.alpha))

    if "synthetic" in methods_to_run:
        print("Running Synthetic Control analysis...")
        results.append(synthetic_control(pre_period, exp_period, args.alpha))

    if "ttest" in methods_to_run:
        print("Running T-test (Baseline Adjusted) analysis...")
        results.append(ttest_baseline_adjusted(pre_period, exp_period, args.alpha))

    if "bayesian" in methods_to_run:
        if BAYESIAN_AVAILABLE:
            print("Running Bayesian analysis...")
            results.append(bayesian_analysis(pre_period, exp_period, args.alpha))
        else:
            print("Warning: Bayesian analysis skipped (PyMC not available)")

    # Print results
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)

    for result in results:
        print(f"\n{result['method']}:")
        print(f"  Estimate: {result.get('estimate', 'N/A'):.4f}")
        print(f"  P-value: {result.get('p_value', 'N/A'):.4f}")
        print(f"  Significant: {'Yes' if result.get('significant', False) else 'No'}")
        if result.get("ci_lower") is not None and not np.isnan(result["ci_lower"]):
            print(f"  95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
        print(f"  Effect Size: {result.get('effect_size', 'N/A'):.4f}")
        print(f"  Power: {result.get('power', 'N/A'):.4f}")
        if "error" in result:
            print(f"  Error: {result['error']}")

    # Create visualization
    print(f"\nGenerating visualization...")
    try:
        create_visualization(
            df,
            pre_period,
            exp_period,
            results,
            pre_start,
            pre_end,
            exp_start,
            exp_end,
            args.alpha,
            args.output,
            args.dark_mode,
        )
        print(f"Chart saved to: {args.output}")

        if args.show:
            import webbrowser
            import os

            webbrowser.open(f"file://{os.path.abspath(args.output)}")
    except Exception as e:
        print(f"Error creating visualization: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
