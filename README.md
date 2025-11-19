# Geo Test Analysis Tool

A comprehensive Python CLI tool for analyzing geo test experiments using multiple statistical methods and interactive visualizations.

## Overview

Geo testing is a product testing method where a test condition is applied to one geography while another geography serves as a control. This tool analyzes the statistical significance of differences between test and control groups during the experiment period, using the pre-experiment period to establish baseline and variance estimates.

## Installation

1. Ensure Python 3.13 is installed (using pyenv):
   ```bash
   pyenv install 3.13
   pyenv local 3.13
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python geo_test_analyzer.py -i "your_data.csv" \
  --pre-start "7/1/25" --pre-end "8/3/25" \
  --exp-start "8/5/25" --exp-end "9/2/25" \
  --method all \
  -o output.html --show
```

### Dark Mode

```bash
python geo_test_analyzer.py -i "your_data.csv" \
  --pre-start "7/1/25" --pre-end "8/3/25" \
  --exp-start "8/5/25" --exp-end "9/2/25" \
  --method all \
  -o output.html --show --dark-mode
```

### Command Line Options

- `-i, --input`: Path to CSV file (required)
  - CSV must have columns: date (first column), Control, Test
  - Date format: M/D/YY (e.g., "7/1/25")

- `--pre-start`: Pre-experiment period start date (M/D/YY format)
- `--pre-end`: Pre-experiment period end date (M/D/YY format)
- `--exp-start`: Experiment start date (M/D/YY format)
- `--exp-end`: Experiment end date / post period end (M/D/YY format)

- `--method`: Statistical method(s) to use
  - Options: `did`, `synthetic`, `ttest`, `bayesian`, `all`
  - Default: `all`
  - Can specify multiple: `--method did ttest`

- `--alpha`: Significance level (default: 0.05)

- `-o, --output`: Output HTML file path for interactive chart (default: `geo_test_results.html`)

- `--show`: Automatically open chart in browser after generation

- `--dark-mode`: Use dark mode theme for visualizations

## Statistical Methods

The tool provides four statistical methods for analyzing geo test experiments. Each method has different assumptions and interpretations. It's recommended to run all methods (`--method all`) to compare results and assess robustness.

### 1. Difference-in-Differences (DiD)

**How it works:**
The DiD method compares the change in the test group to the change in the control group:

**Estimator**: `DiD = (Test_exp - Control_exp) - (Test_pre - Control_pre)`

This removes the baseline difference between groups and focuses on the differential change during the experiment period.

**Key Features:**
- Uses pre-period to establish baseline difference between groups
- Assumes parallel trends (both groups would follow similar trends absent treatment)
- Provides robust standard errors accounting for variance in both periods
- Confidence intervals represent uncertainty in the treatment effect relative to the counterfactual

**How to Read Results:**
- **Estimate**: The treatment effect (difference in differences). Positive values indicate test group improved relative to control.
- **P-value**: Probability of observing this effect if there's no true treatment effect. < 0.05 indicates statistical significance.
- **95% CI**: Range containing the true treatment effect with 95% confidence (relative to counterfactual).
- **Effect Size**: Standardized measure (Cohen's d) of the treatment effect magnitude.

**When to Use:**
- Control and test groups follow similar trends pre-experiment (parallel trends assumption)
- You want to account for baseline differences between groups
- You need a method that's widely understood and accepted

**Limitations:**
- Requires parallel trends assumption (groups would trend similarly without treatment)
- May be biased if there are time-varying confounders affecting groups differently

---

### 2. Synthetic Control

**How it works:**
This implementation uses a simplified synthetic control approach where the control group serves directly as the synthetic counterfactual. The method calculates the treatment effect as the difference between test and control during the experiment period, using permutation tests for statistical inference.

**Key Features:**
- Uses control group directly as synthetic counterfactual (simplified implementation)
- Uses permutation tests (1000 permutations) for p-value calculation
- Confidence intervals derived from permutation distribution
- Confidence intervals represent uncertainty in the raw difference (test - control)

**How to Read Results:**
- **Estimate**: Raw difference between test and control means during experiment period.
- **P-value**: From permutation test - proportion of permuted effects as extreme as observed.
- **95% CI**: Range from permutation distribution (represents raw difference range).
- **Effect Size**: Standardized treatment effect.

**When to Use:**
- You want a non-parametric approach (no distributional assumptions)
- You prefer permutation-based inference
- Quick comparison when control serves as reasonable counterfactual

**Limitations:**
- Simplified implementation (doesn't optimize weights across multiple donors)
- Assumes control is a good counterfactual for test group
- May be less powerful than parametric methods with small samples

---

### 3. T-test with Baseline Adjustment

**How it works:**
A standard two-sample t-test applied to baseline-adjusted values. The method subtracts pre-period means from experiment period values for both groups, then compares the adjusted differences.

**Key Features:**
- Normalizes by subtracting pre-period means from experiment values
- Uses standard t-test on adjusted values
- Calculates effect size using Cohen's d
- Confidence intervals represent uncertainty in the baseline-adjusted treatment effect

**How to Read Results:**
- **Estimate**: Difference between baseline-adjusted test and control means.
- **P-value**: From two-sample t-test on adjusted values. < 0.05 indicates significance.
- **95% CI**: Range for the baseline-adjusted treatment effect.
- **Effect Size**: Cohen's d - standardized difference (0.2=small, 0.5=medium, 0.8=large).

**When to Use:**
- Quick analysis when assumptions are met (normality, equal variances)
- You want a simple, interpretable method
- Sample sizes are adequate for t-test assumptions

**Limitations:**
- Assumes normality and equal variances (though t-test is robust to mild violations)
- Doesn't account for time trends as explicitly as DiD
- May be less appropriate with very small samples

---

### 4. Bayesian Approach

**How it works:**
Bayesian inference using PyMC with prior distributions informed by pre-period data. The method estimates posterior distributions for control and test group means, then calculates the treatment effect as their difference.

**Key Features:**
- Prior distributions estimated from pre-period means and standard deviations
- Uses MCMC sampling (2000 samples, 1000 tuning) for posterior estimation
- Provides credible intervals (Bayesian equivalent of confidence intervals)
- P-value calculated as proportion of posterior distribution on opposite side of zero
- Confidence intervals represent uncertainty in the raw difference (test - control)

**How to Read Results:**
- **Estimate**: Mean of posterior distribution for treatment effect.
- **P-value**: Proportion of posterior distribution indicating opposite effect (two-tailed).
- **95% CI**: Credible interval - range containing 95% of posterior probability (raw difference).
- **Effect Size**: Standardized treatment effect.

**When to Use:**
- You want to incorporate prior knowledge from pre-period
- You prefer Bayesian interpretation of uncertainty
- You want posterior distributions for more detailed analysis
- You need to quantify evidence strength beyond p-values

**Limitations:**
- Requires PyMC installation (optional dependency)
- Computationally more intensive than other methods
- Requires understanding of Bayesian concepts for full interpretation
- May fail with very small samples or convergence issues

**Note:** If PyMC is not installed, this method will be skipped with a warning.

## Output

The tool generates both console output and interactive HTML visualizations. Each statistical method gets its own chart panel with detailed results.

### Interactive Chart

The tool generates separate interactive Bokeh charts for each selected method. Each chart includes:

**Visual Elements:**
- **Time series lines**:
  - **Control** (blue): Control group metric values over time
  - **Test** (orange): Test group metric values over time
  - **Counterfactual** (green, dotted): Expected test values based on pre-period relationship (what test would be if no treatment effect)
  - **Lift %** (purple, dashed): Percentage lift relative to counterfactual on secondary y-axis

- **Period shading**:
  - Blue background: Pre-experiment period
  - Orange background: Experiment period
  - Green shading: Periods where results are statistically significant (p < 0.05)

- **Confidence intervals**:
  - Gray shaded area: 95% confidence/credible interval for treatment effect
  - Purple shaded area: 95% confidence interval for lift percentage

- **Significance indicators**:
  - Green diamonds: Statistically significant points (p < 0.05)
  - Red diamonds: Non-significant points during experiment period
  - Vertical red dashed line: Experiment start

- **Summary box**: Side panel showing key statistics for the method

**Tooltips** (on hover over test or lift lines):
- Date
- Test value (actual)
- Counterfactual value (expected based on pre-period)
- Control value
- **Lift**: `((Test - Counterfactual) / Counterfactual) * 100%` - percentage change relative to expected value
- Lift 95% CI: Confidence interval for lift percentage
- P-value: Statistical significance for the cumulative analysis up to that point
- Power: Statistical power for detecting the observed effect
- Significant: Whether the result is statistically significant (True/False)

**Note on Lift Calculation:**
The lift percentage is calculated relative to the counterfactual (expected test value), not the control. The counterfactual is derived from the pre-period relationship: `Expected_Test = Control * (Pre_Test_Mean / Pre_Control_Mean)`. This accounts for baseline differences between groups.

**Interpreting Confidence Intervals:**
- **DiD and T-test**: CI represents uncertainty in treatment effect relative to counterfactual (baseline-adjusted)
- **Synthetic Control and Bayesian**: CI represents uncertainty in raw difference (test - control)

### Console Output

Summary statistics printed to console for each method:
- **Estimate**: Treatment effect estimate
- **P-value**: Statistical significance (p < 0.05 indicates significance)
- **Significant**: Yes/No based on p-value threshold
- **95% CI**: Confidence/credible interval bounds
- **Effect Size**: Standardized effect size (Cohen's d)
- **Power**: Statistical power (probability of detecting effect if it exists)
- **Error messages**: If any method fails (e.g., Bayesian method without PyMC)

## Example Data Format

CSV files should have the following structure:

```csv
Row Labels,Control,Test
4/15/25,434.00,402.00
4/16/25,439.00,400.00
...
```

**Column Requirements:**
- **First column**: Date values (M/D/YY format, e.g., "4/15/25")
- **Control column**: Control group metric values (column name must contain "control" or "control group", case-insensitive)
- **Test column**: Test group metric values (column name must contain "test" or "test group", case-insensitive)

**Data Requirements:**
- Dates must be parseable (M/D/YY format preferred)
- Control and Test values must be numeric
- Missing dates or values are automatically excluded
- Data is sorted by date after loading

## Interpreting Results

### Comparing Methods

When multiple methods are run, you may see different results. This is normal and expected:

- **Consistent results across methods**: Strong evidence for treatment effect
- **Divergent results**: May indicate:
  - Violation of method-specific assumptions (e.g., parallel trends for DiD)
  - Small sample sizes affecting some methods more than others
  - Different interpretations of uncertainty (confidence vs. credible intervals)

**Recommendation**: If methods disagree, investigate assumptions and consider:
- Visual inspection of pre-period trends (parallel trends check)
- Sample size adequacy
- Which method's assumptions best fit your data

### Key Metrics to Focus On

1. **P-value < 0.05**: Statistically significant result (unlikely due to chance)
2. **Effect Size**: Magnitude of the effect (beyond just significance)
   - Small (0.2): May be statistically significant but practically small
   - Medium (0.5): Moderate practical impact
   - Large (0.8+): Strong practical impact
3. **Power**: If power is low (< 0.8), you may have missed detecting real effects
4. **Confidence Intervals**: 
   - Narrow CI: More precise estimate
   - CI includes zero: Effect may not be statistically significant
   - CI entirely positive/negative: Consistent direction of effect

### Making Decisions

- **All methods significant**: Strong evidence to proceed with treatment
- **Some methods significant**: Consider which methods' assumptions fit your data best
- **No methods significant**: Either no effect exists, or sample size is too small to detect it
- **Check power**: Low power suggests you may need more data or a longer experiment period

## Technical Notes

- **Missing dates**: Handled gracefully - rows with missing dates or values are excluded
- **Small samples**: Statistical inference accounts for small sample sizes using appropriate distributions (t-distribution)
- **Method differences**: Multiple methods may produce different results - all are presented for transparency and robustness checking
- **Power calculations**: Use effect size estimates derived from pre-period variance and sample sizes
- **Date parsing**: Handles various formats but expects M/D/YY as primary format (e.g., "7/1/25")
- **Time-varying analysis**: Charts show cumulative analysis - p-values and CIs update as more experiment data accumulates
- **Counterfactual calculation**: Based on pre-period ratio, assumes this relationship would continue without treatment
- **Bayesian method**: Requires PyMC and ArviZ. If not installed, method is skipped with a warning
- **Permutation tests**: Synthetic Control uses 1000 permutations by default
- **MCMC sampling**: Bayesian method uses 2000 samples with 1000 tuning iterations

## Dependencies

**Required:**
- pandas: Data manipulation and CSV loading
- numpy: Numerical computations
- scipy: Statistical functions (t-tests, distributions)
- bokeh: Interactive visualizations
- python-dateutil: Date parsing

**Optional (for Bayesian method):**
- pymc: Bayesian inference and MCMC sampling
- arviz: Bayesian analysis and diagnostics

All dependencies are listed in `requirements.txt`. The Bayesian method will be skipped if PyMC/ArviZ are not installed.

