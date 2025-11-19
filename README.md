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

### 1. Difference-in-Differences (DiD)

The DiD method compares the change in the test group to the change in the control group:

**Estimator**: `DiD = (Test_exp - Control_exp) - (Test_pre - Control_pre)`

- Uses pre-period to establish baseline difference
- Accounts for parallel trends assumption
- Provides robust standard errors
- Best for: When control and test groups follow similar trends pre-experiment

### 2. Synthetic Control

Creates a weighted combination of control group observations to construct a counterfactual for the test group:

- Fits optimal weights using pre-period data
- Generates synthetic control series
- Uses permutation tests for inference
- Best for: When you have multiple control geographies or want a data-driven counterfactual

### 3. T-test with Baseline Adjustment

Standard two-sample t-test with normalization based on pre-period characteristics:

- Normalizes by pre-period means and variance
- Calculates effect size (Cohen's d)
- Simple and interpretable
- Best for: Quick analysis when assumptions are met

### 4. Bayesian Approach

Bayesian inference with prior distributions informed by pre-period data:

- Prior distributions estimated from pre-period
- Posterior estimation for treatment effect
- Credible intervals (95% default)
- Bayes factors for evidence strength
- Best for: Incorporating prior knowledge and uncertainty quantification

## Output

### Interactive Chart

The tool generates an interactive Bokeh chart showing:

- **Time series**: Control and Test group values over time
- **Period shading**: Visual distinction between pre-experiment and experiment periods
- **Significance bounds**: Confidence/credible intervals
- **Tooltips**: On hover, shows:
  - Date
  - Control and Test values
  - % deviation: `((Test - Control) / Control) * 100`
  - P-value for the analysis window
  - Statistical power level
  - Which methods show significance

### Console Output

Summary statistics printed to console:
- Method-specific results (p-values, effect sizes, confidence intervals)
- Statistical power calculations
- Overall conclusion for each method

## Example Data Format

CSV files should have the following structure:

```csv
Row Labels,Control,Test
4/15/25,434.00,402.00
4/16/25,439.00,400.00
...
```

Where:
- First column: Date (M/D/YY format)
- Control: Control group metric values
- Test: Test group metric values

## Technical Notes

- Missing dates are handled gracefully
- Statistical inference accounts for small sample sizes
- Multiple methods may produce different results - all are presented for transparency
- Power calculations use effect size estimates from pre-period variance
- Date parsing handles various formats but expects M/D/YY as primary format

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computations
- scipy: Statistical functions
- bokeh: Interactive visualizations
- statsmodels: Advanced statistical models
- pymc: Bayesian inference
- python-dateutil: Date parsing

