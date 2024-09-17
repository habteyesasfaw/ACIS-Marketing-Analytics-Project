# ab_testing_analysis.py
import pandas as pd
import numpy as np
from scipy import stats

# Load dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Data Cleaning
def clean_data(df):
    # Drop duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df = df.dropna(subset=['TotalClaims', 'TotalPremium', 'Gender', 'PostalCode', 'Province'])

    # Ensure correct data types
    df['PostalCode'] = df['PostalCode'].astype(str)
    df['Gender'] = df['Gender'].astype('category')

    # Handle outliers
    df['TotalClaims'] = np.where(df['TotalClaims'] > df['TotalClaims'].quantile(0.99), 
                                 df['TotalClaims'].quantile(0.99), df['TotalClaims'])
    df['TotalPremium'] = np.where(df['TotalPremium'] > df['TotalPremium'].quantile(0.99), 
                                  df['TotalPremium'].quantile(0.99), df['TotalPremium'])
    return df

# Select KPI (Key Performance Indicator)
def select_kpi(df, kpi_col):
    if kpi_col not in df.columns:
        raise ValueError(f"KPI '{kpi_col}' is not in the dataset.")
    return df[kpi_col]

# Segment data into control and test groups
def segment_data(df, feature, control_value, test_value):
    group_a = df[df[feature] == control_value]  # Control Group (A)
    group_b = df[df[feature] == test_value]     # Test Group (B)
    return group_a, group_b

# T-test for numerical KPIs
def perform_t_test(group_a, group_b, kpi_col):
    t_stat, p_value = stats.ttest_ind(group_a[kpi_col], group_b[kpi_col], equal_var=False)
    return t_stat, p_value

# Z-test for large samples
def perform_z_test(group_a, group_b, kpi_col):
    mean_a, mean_b = group_a[kpi_col].mean(), group_b[kpi_col].mean()
    std_a, std_b = group_a[kpi_col].std(), group_b[kpi_col].std()
    n_a, n_b = len(group_a), len(group_b)
    z_stat = (mean_a - mean_b) / ((std_a**2/n_a + std_b**2/n_b)**0.5)
    p_value = stats.norm.sf(abs(z_stat)) * 2  # Two-tailed test
    return z_stat, p_value

# Report result based on p-value
def report_results(p_value, alpha=0.05):
    if p_value < alpha:
        return f"Reject the null hypothesis (p-value = {p_value:.4f}). Significant difference detected."
    else:
        return f"Fail to reject the null hypothesis (p-value = {p_value:.4f}). No significant difference."
