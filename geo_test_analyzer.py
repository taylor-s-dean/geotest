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
from bokeh.models import HoverTool, ColumnDataSource, Span, BoxAnnotation, Label
from bokeh.palettes import Category10

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

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
        parts = date_str.split('/')
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
        if col.lower() in ['control', 'control group']:
            control_col = col
        elif col.lower() in ['test', 'test group']:
            test_col = col
    
    if control_col is None or test_col is None:
        raise ValueError("CSV must contain 'Control' and 'Test' columns")
    
    # Parse dates
    df['date'] = df[date_col].apply(parse_date)
    df['control'] = pd.to_numeric(df[control_col], errors='coerce')
    df['test'] = pd.to_numeric(df[test_col], errors='coerce')
    
    # Remove rows with missing data
    df = df.dropna(subset=['date', 'control', 'test'])
    df = df.sort_values('date').reset_index(drop=True)
    
    return df[['date', 'control', 'test']]


def split_periods(df: pd.DataFrame, pre_start: datetime, pre_end: datetime,
                  exp_start: datetime, exp_end: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into pre-experiment and experiment periods."""
    pre_period = df[(df['date'] >= pre_start) & (df['date'] <= pre_end)].copy()
    exp_period = df[(df['date'] >= exp_start) & (df['date'] <= exp_end)].copy()
    
    if len(pre_period) == 0:
        raise ValueError("No data found in pre-experiment period")
    if len(exp_period) == 0:
        raise ValueError("No data found in experiment period")
    
    return pre_period, exp_period


def calculate_power(effect_size: float, n: int, alpha: float = 0.05) -> float:
    """Calculate statistical power for a given effect size and sample size."""
    try:
        # Using Cohen's conventions for effect size
        # For t-test power calculation
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = np.sqrt(n) * abs(effect_size) - z_alpha
        power = norm.cdf(z_beta)
        return max(0, min(1, power))
    except:
        return np.nan


# ============================================================================
# STATISTICAL METHODS
# ============================================================================

def difference_in_differences(pre_period: pd.DataFrame, exp_period: pd.DataFrame,
                             alpha: float = 0.05) -> Dict:
    """Difference-in-Differences analysis."""
    # Calculate pre-period difference
    pre_diff = (pre_period['test'].mean() - pre_period['control'].mean())
    
    # Calculate experiment period difference
    exp_diff = (exp_period['test'].mean() - exp_period['control'].mean())
    
    # DiD estimator
    did_estimate = exp_diff - pre_diff
    
    # Calculate standard errors using robust method
    pre_control_mean = pre_period['control'].mean()
    pre_test_mean = pre_period['test'].mean()
    exp_control_mean = exp_period['control'].mean()
    exp_test_mean = exp_period['test'].mean()
    
    # Variance components
    pre_var = pre_period['test'].var() / len(pre_period) + pre_period['control'].var() / len(pre_period)
    exp_var = exp_period['test'].var() / len(exp_period) + exp_period['control'].var() / len(exp_period)
    
    se = np.sqrt(pre_var + exp_var)
    
    # T-statistic and p-value
    t_stat = did_estimate / se if se > 0 else 0
    df = len(pre_period) + len(exp_period) - 2
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    # Confidence interval
    t_critical = stats.t.ppf(1 - alpha/2, df)
    ci_lower = did_estimate - t_critical * se
    ci_upper = did_estimate + t_critical * se
    
    # Effect size (standardized)
    pooled_std = np.sqrt((pre_period['control'].var() + pre_period['test'].var()) / 2)
    effect_size = did_estimate / pooled_std if pooled_std > 0 else 0
    
    # Power calculation
    n = min(len(pre_period), len(exp_period))
    power = calculate_power(effect_size, n, alpha)
    
    return {
        'method': 'Difference-in-Differences',
        'estimate': did_estimate,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'effect_size': effect_size,
        'power': power,
        'significant': p_value < alpha,
        'pre_diff': pre_diff,
        'exp_diff': exp_diff
    }


def synthetic_control(pre_period: pd.DataFrame, exp_period: pd.DataFrame,
                      alpha: float = 0.05) -> Dict:
    """Synthetic Control method."""
    # Use control as donor pool (simplified - in practice would use multiple controls)
    # Fit weights to minimize pre-period prediction error
    control_pre = pre_period['control'].values
    test_pre = pre_period['test'].values
    
    # Simple approach: use control directly as synthetic control
    # (In full implementation, would optimize weights across multiple donors)
    synthetic_pre = control_pre.copy()
    synthetic_exp = exp_period['control'].values
    
    # Calculate treatment effect
    actual_exp = exp_period['test'].values
    treatment_effect = actual_exp.mean() - synthetic_exp.mean()
    
    # Permutation test: shuffle pre-period differences
    pre_diffs = test_pre - control_pre
    n_permutations = 1000
    permuted_effects = []
    
    for _ in range(n_permutations):
        np.random.shuffle(pre_diffs)
        # Simulate experiment period effect
        perm_effect = np.random.choice(pre_diffs, size=len(exp_period), replace=True).mean()
        permuted_effects.append(perm_effect)
    
    # P-value from permutation test
    p_value = np.mean(np.abs(permuted_effects) >= abs(treatment_effect))
    
    # Confidence interval from permutation distribution
    ci_lower = np.percentile(permuted_effects, alpha/2 * 100)
    ci_upper = np.percentile(permuted_effects, (1 - alpha/2) * 100)
    
    # Effect size
    pooled_std = np.sqrt((pre_period['control'].var() + pre_period['test'].var()) / 2)
    effect_size = treatment_effect / pooled_std if pooled_std > 0 else 0
    
    # Power
    n = len(exp_period)
    power = calculate_power(effect_size, n, alpha)
    
    return {
        'method': 'Synthetic Control',
        'estimate': treatment_effect,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'effect_size': effect_size,
        'power': power,
        'significant': p_value < alpha,
        'synthetic_series': synthetic_exp
    }


def ttest_baseline_adjusted(pre_period: pd.DataFrame, exp_period: pd.DataFrame,
                           alpha: float = 0.05) -> Dict:
    """T-test with baseline adjustment."""
    # Normalize by pre-period characteristics
    pre_control_mean = pre_period['control'].mean()
    pre_test_mean = pre_period['test'].mean()
    pre_control_std = pre_period['control'].std()
    pre_test_std = pre_period['test'].std()
    
    # Baseline adjustment: subtract pre-period means
    exp_control_adj = exp_period['control'].values - pre_control_mean
    exp_test_adj = exp_period['test'].values - pre_test_mean
    
    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(exp_test_adj, exp_control_adj)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((pre_control_std**2 + pre_test_std**2) / 2)
    mean_diff = exp_test_adj.mean() - exp_control_adj.mean()
    effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
    
    # Confidence interval
    n1, n2 = len(exp_test_adj), len(exp_control_adj)
    se_diff = np.sqrt(exp_test_adj.var()/n1 + exp_control_adj.var()/n2)
    df = n1 + n2 - 2
    t_critical = stats.t.ppf(1 - alpha/2, df)
    ci_lower = mean_diff - t_critical * se_diff
    ci_upper = mean_diff + t_critical * se_diff
    
    # Power
    n = min(n1, n2)
    power = calculate_power(effect_size, n, alpha)
    
    return {
        'method': 'T-test (Baseline Adjusted)',
        'estimate': mean_diff,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'effect_size': effect_size,
        'power': power,
        'significant': p_value < alpha
    }


def bayesian_analysis(pre_period: pd.DataFrame, exp_period: pd.DataFrame,
                     alpha: float = 0.05) -> Dict:
    """Bayesian analysis using PyMC."""
    if not BAYESIAN_AVAILABLE:
        return {
            'method': 'Bayesian',
            'estimate': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'effect_size': np.nan,
            'power': np.nan,
            'significant': False,
            'error': 'PyMC not available'
        }
    
    try:
        # Prepare data
        pre_control = pre_period['control'].values
        pre_test = pre_period['test'].values
        exp_control = exp_period['control'].values
        exp_test = exp_period['test'].values
        
        with pm.Model() as model:
            # Priors from pre-period
            pre_control_mean = pre_control.mean()
            pre_test_mean = pre_test.mean()
            pre_control_std = pre_control.std()
            pre_test_std = pre_test.std()
            
            # Prior for control group (experiment period)
            mu_control = pm.Normal('mu_control', mu=pre_control_mean, sigma=pre_control_std)
            sigma_control = pm.HalfNormal('sigma_control', sigma=pre_control_std)
            
            # Prior for test group (experiment period)
            mu_test = pm.Normal('mu_test', mu=pre_test_mean, sigma=pre_test_std)
            sigma_test = pm.HalfNormal('sigma_test', sigma=pre_test_std)
            
            # Likelihood
            control_obs = pm.Normal('control_obs', mu=mu_control, sigma=sigma_control, observed=exp_control)
            test_obs = pm.Normal('test_obs', mu=mu_test, sigma=sigma_test, observed=exp_test)
            
            # Treatment effect
            treatment_effect = pm.Deterministic('treatment_effect', mu_test - mu_control)
            
            # Sample
            trace = pm.sample(2000, tune=1000, return_inferencedata=True, progressbar=False)
        
        # Extract results
        posterior = az.extract(trace, var_names=['treatment_effect'])
        estimate = float(posterior.mean())
        ci_lower = float(posterior.quantile(alpha/2))
        ci_upper = float(posterior.quantile(1 - alpha/2))
        
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
            'method': 'Bayesian',
            'estimate': estimate,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'effect_size': effect_size,
            'power': power,
            'significant': p_value < alpha,
            'trace': trace
        }
    except Exception as e:
        return {
            'method': 'Bayesian',
            'estimate': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'effect_size': np.nan,
            'power': np.nan,
            'significant': False,
            'error': str(e)
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualization(df: pd.DataFrame, pre_period: pd.DataFrame, exp_period: pd.DataFrame,
                        results: List[Dict], pre_start: datetime, pre_end: datetime,
                        exp_start: datetime, exp_end: datetime,
                        alpha: float, output_path: str):
    """Create interactive Bokeh visualization."""
    
    # Prepare data for plotting
    # Ensure dates are datetime objects for proper comparison
    dates = pd.to_datetime(df['date']).values
    control_values = df['control'].values
    test_values = df['test'].values
    
    # Calculate daily differences and pct deviation
    daily_diff = test_values - control_values
    daily_pct_dev = ((test_values - control_values) / control_values) * 100
    
    # Determine significance for each day (using rolling window or cumulative)
    # For simplicity, we'll use cumulative analysis from exp_start
    # Use pandas Series for proper datetime comparison
    date_series = pd.to_datetime(df['date'])
    exp_mask = (date_series >= exp_start) & (date_series <= exp_end)
    exp_indices = np.where(exp_mask.values)[0]
    
    # Calculate rolling p-values (simplified - using cumulative t-test)
    daily_p_values = np.full(len(dates), np.nan)
    daily_power = np.full(len(dates), np.nan)
    significant_methods = [''] * len(dates)
    
    for i, idx in enumerate(exp_indices):
        # Cumulative data up to this point
        exp_subset = exp_period.iloc[:i+1] if i < len(exp_period) else exp_period
        
        if len(exp_subset) > 1:
            # Quick t-test for this window
            try:
                _, p_val = stats.ttest_ind(exp_subset['test'], exp_subset['control'])
                daily_p_values[idx] = p_val
                
                # Effect size for power calculation
                pooled_std = np.sqrt((exp_subset['control'].var() + exp_subset['test'].var()) / 2)
                mean_diff = exp_subset['test'].mean() - exp_subset['control'].mean()
                effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
                daily_power[idx] = calculate_power(effect_size, len(exp_subset), alpha)
            except:
                pass
        
        # Which methods are significant overall
        sig_methods = [r['method'] for r in results if r.get('significant', False)]
        significant_methods[idx] = ', '.join(sig_methods) if sig_methods else 'None'
    
    # Create ColumnDataSource
    # Convert dates to strings for tooltip display
    date_strs = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d').values
    
    source = ColumnDataSource(data={
        'date': dates,
        'control': control_values,
        'test': test_values,
        'diff': daily_diff,
        'pct_dev': daily_pct_dev,
        'p_value': daily_p_values,
        'power': daily_power,
        'significant_methods': significant_methods,
        'date_str': date_strs
    })
    
    # Create figure
    p = figure(
        width=1200, height=700,
        x_axis_type='datetime',
        title='Geo Test Analysis: Control vs Test Groups',
        tools='pan,wheel_zoom,box_zoom,reset,save'
    )
    
    # Add period shading
    pre_box = BoxAnnotation(left=pre_start, right=pre_end, fill_alpha=0.1, fill_color='blue')
    exp_box = BoxAnnotation(left=exp_start, right=exp_end, fill_alpha=0.1, fill_color='orange')
    p.add_layout(pre_box)
    p.add_layout(exp_box)
    
    # Get y-axis range for label positioning
    y_max = max(control_values.max(), test_values.max())
    y_min = min(control_values.min(), test_values.min())
    y_range = y_max - y_min
    
    # Add period labels
    pre_label = Label(x=pre_start, y=y_max - y_range*0.05, text='Pre-Experiment', 
                     text_font_size='10pt', text_color='blue')
    exp_label = Label(x=exp_start, y=y_max - y_range*0.05, text='Experiment', 
                     text_font_size='10pt', text_color='orange')
    p.add_layout(pre_label)
    p.add_layout(exp_label)
    
    # Add vertical line at experiment start
    exp_start_line = Span(location=exp_start, dimension='height', line_color='red',
                         line_dash='dashed', line_width=2)
    p.add_layout(exp_start_line)
    
    # Plot control and test lines
    p.line('date', 'control', source=source, legend_label='Control', 
           line_width=2, color=Category10[10][0])
    p.line('date', 'test', source=source, legend_label='Test', 
           line_width=2, color=Category10[10][1])
    
    # Add scatter points for better interactivity
    p.circle('date', 'control', source=source, size=4, color=Category10[10][0], alpha=0.6)
    p.circle('date', 'test', source=source, size=4, color=Category10[10][1], alpha=0.6)
    
    # Add hover tool with proper formatting
    hover = HoverTool(
        tooltips=[
            ('Date', '@date_str'),
            ('Control', '@control{0.2f}'),
            ('Test', '@test{0.2f}'),
            ('% Deviation', '@pct_dev{0.2f}%'),
            ('P-value', '@p_value{0.4f}'),
            ('Power', '@power{0.2f}'),
            ('Significant Methods', '@significant_methods')
        ],
        formatters={
            '@p_value': 'numeral',
            '@power': 'numeral',
            '@pct_dev': 'numeral'
        },
        mode='vline'
    )
    p.add_tools(hover)
    
    # Add confidence intervals if available
    for result in results:
        if result.get('ci_lower') is not None and not np.isnan(result['ci_lower']):
            method_name = result['method']
            estimate = result['estimate']
            
            # Add horizontal lines for CI bounds (simplified visualization)
            # In full implementation, would show time-varying CIs
    
    # Style
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Metric Value'
    p.legend.location = 'top_left'
    p.legend.click_policy = 'hide'
    
    # Add results summary as text
    results_text = "Results Summary:\n"
    for result in results:
        method = result['method']
        sig = "✓" if result.get('significant', False) else "✗"
        p_val = result.get('p_value', np.nan)
        if not np.isnan(p_val):
            results_text += f"{sig} {method}: p={p_val:.4f}\n"
    
    results_label = Label(x=exp_start, y=y_min + y_range*0.1,
                         text=results_text, text_font_size='9pt', 
                         background_fill_color='white', background_fill_alpha=0.8,
                         text_align='left')
    p.add_layout(results_label)
    
    # Save
    output_file(output_path)
    save(p)
    
    return p


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Geo Test Analysis Tool - Statistical analysis of geo test experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Path to CSV file with date, Control, and Test columns')
    parser.add_argument('--pre-start', required=True,
                       help='Pre-experiment start date (M/D/YY format)')
    parser.add_argument('--pre-end', required=True,
                       help='Pre-experiment end date (M/D/YY format)')
    parser.add_argument('--exp-start', required=True,
                       help='Experiment start date (M/D/YY format)')
    parser.add_argument('--exp-end', required=True,
                       help='Experiment end date / post period end (M/D/YY format)')
    parser.add_argument('--method', nargs='+', default=['all'],
                       choices=['did', 'synthetic', 'ttest', 'bayesian', 'all'],
                       help='Statistical method(s) to use')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level (default: 0.05)')
    parser.add_argument('-o', '--output', default='geo_test_results.html',
                       help='Output HTML file path (default: geo_test_results.html)')
    parser.add_argument('--show', action='store_true',
                       help='Open chart in browser after generation')
    
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
        pre_period, exp_period = split_periods(df, pre_start, pre_end, exp_start, exp_end)
    except ValueError as e:
        print(f"Error splitting periods: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Determine which methods to run
    methods_to_run = []
    if 'all' in args.method:
        methods_to_run = ['did', 'synthetic', 'ttest', 'bayesian']
    else:
        methods_to_run = args.method
    
    # Run analyses
    results = []
    
    if 'did' in methods_to_run:
        print("Running Difference-in-Differences analysis...")
        results.append(difference_in_differences(pre_period, exp_period, args.alpha))
    
    if 'synthetic' in methods_to_run:
        print("Running Synthetic Control analysis...")
        results.append(synthetic_control(pre_period, exp_period, args.alpha))
    
    if 'ttest' in methods_to_run:
        print("Running T-test (Baseline Adjusted) analysis...")
        results.append(ttest_baseline_adjusted(pre_period, exp_period, args.alpha))
    
    if 'bayesian' in methods_to_run:
        if BAYESIAN_AVAILABLE:
            print("Running Bayesian analysis...")
            results.append(bayesian_analysis(pre_period, exp_period, args.alpha))
        else:
            print("Warning: Bayesian analysis skipped (PyMC not available)")
    
    # Print results
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    
    for result in results:
        print(f"\n{result['method']}:")
        print(f"  Estimate: {result.get('estimate', 'N/A'):.4f}")
        print(f"  P-value: {result.get('p_value', 'N/A'):.4f}")
        print(f"  Significant: {'Yes' if result.get('significant', False) else 'No'}")
        if result.get('ci_lower') is not None and not np.isnan(result['ci_lower']):
            print(f"  95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
        print(f"  Effect Size: {result.get('effect_size', 'N/A'):.4f}")
        print(f"  Power: {result.get('power', 'N/A'):.4f}")
        if 'error' in result:
            print(f"  Error: {result['error']}")
    
    # Create visualization
    print(f"\nGenerating visualization...")
    try:
        create_visualization(df, pre_period, exp_period, results,
                           pre_start, pre_end, exp_start, exp_end,
                           args.alpha, args.output)
        print(f"Chart saved to: {args.output}")
        
        if args.show:
            import webbrowser
            import os
            webbrowser.open(f'file://{os.path.abspath(args.output)}')
    except Exception as e:
        print(f"Error creating visualization: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()

