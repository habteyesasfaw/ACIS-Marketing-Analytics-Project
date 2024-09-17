import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower

# Load dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Data Cleaning Function
def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna(subset=['TotalClaims', 'TotalPremium', 'Gender', 'PostalCode', 'Province'])
    df['PostalCode'] = df['PostalCode'].astype(str)
    df['Gender'] = df['Gender'].astype('category')
    df['TotalClaims'] = np.where(df['TotalClaims'] > df['TotalClaims'].quantile(0.99), 
                                 df['TotalClaims'].quantile(0.99), df['TotalClaims'])
    df['TotalPremium'] = np.where(df['TotalPremium'] > df['TotalPremium'].quantile(0.99), 
                                  df['TotalPremium'].quantile(0.99), df['TotalPremium'])
    return df

# Updated segment_data function


def segment_data(df, column_name, value_a, value_b):
    group_a = df[df[column_name] == value_a]
    group_b = df[df[column_name] == value_b]
    
    print(f"Group A ({value_a}): {len(group_a)} records")
    print(f"Group B ({value_b}): {len(group_b)} records")
    
    return group_a, group_b






# Function to calculate Cramér's V
def cramers_v(confusion_matrix):
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1))))

# Manual chi-square power calculation
def chi2_power_manual(effect_size, df, alpha=0.05, n=None):
    critical_value = stats.chi2.ppf(1 - alpha, df)
    noncentrality_param = effect_size ** 2 * n
    power = 1 - stats.ncx2.cdf(critical_value, df, noncentrality_param)
    return power

# Function to perform power analysis for chi-squared test
def power_analysis_chi2(effect_size, alpha=0.05, power=0.8):
    return chi2_power_manual(effect_size, df=1, alpha=alpha, n=100)

# Function to perform power analysis for t-test
def power_analysis_ttest(effect_size, alpha=0.05, power=0.8):
    analysis = TTestIndPower()
    return analysis.solve_power(effect_size, power=power, alpha=alpha)

# Function to perform t-test
def perform_t_test(group_a, group_b, kpi_col):
    t_stat, p_value = stats.ttest_ind(group_a[kpi_col], group_b[kpi_col], equal_var=False)
    return t_stat, p_value

# Function to perform chi-squared test
# def perform_chi_squared_test(group_a, group_b, kpi_col):
#     contingency_table = pd.crosstab(group_a[kpi_col], group_b[kpi_col])
#     chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
#     return chi2_stat, p_value

# Updated perform_chi_squared_test function
def perform_chi_squared_test(group_a, group_b, kpi_col):
    if group_a.empty or group_b.empty:
        raise ValueError("One or both groups are empty. Check segmentation and KPI column values.")
    
    contingency_table = pd.crosstab(group_a[kpi_col], group_b[kpi_col])
    if contingency_table.empty:
        raise ValueError("Contingency table is empty. Check the values in the KPI column.")
    
    chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
    return chi2_stat, p_value

# Function to perform z-test
# Function to perform z-test
def perform_z_test(group_a, group_b, kpi_col):
    mean_a, mean_b = group_a[kpi_col].mean(), group_b[kpi_col].mean()
    std_a, std_b = group_a[kpi_col].std(), group_b[kpi_col].std()
    n_a, n_b = len(group_a), len(group_b)
    
    # Check if either group is empty
    if n_a == 0 or n_b == 0:
        raise ValueError("One of the groups is empty. Cannot perform z-test.")
    
    # Calculate z-statistic
    z_stat = (mean_a - mean_b) / np.sqrt((std_a**2/n_a) + (std_b**2/n_b))
    p_value = stats.norm.sf(abs(z_stat)) * 2  # Two-tailed test
    return z_stat, p_value


# Calculate effect size for t-test (Cohen's d)
def cohen_d(group_a, group_b, kpi_col):
    mean_a, mean_b = group_a[kpi_col].mean(), group_b[kpi_col].mean()
    std_a, std_b = group_a[kpi_col].std(), group_b[kpi_col].std()
    pooled_std = np.sqrt(((len(group_a) - 1) * std_a**2 + (len(group_b) - 1) * std_b**2) / (len(group_a) + len(group_b) - 2))
    return (mean_a - mean_b) / pooled_std

# Function to interpret p-values
def interpret_p_value(p_value, alpha=0.05):
    if p_value < alpha:
        return f"Reject the null hypothesis (p-value = {p_value:.4f}). Significant difference detected."
    else:
        return f"Fail to reject the null hypothesis (p-value = {p_value:.4f}). No significant difference."

# Hypothesis 1: Risk Differences Across Provinces
def hypothesis_1(df, control_value, test_value, kpi_col='TotalClaims'):
    group_a, group_b = segment_data(df, 'Province', control_value, test_value)
    t_stat, p_value = perform_t_test(group_a, group_b, kpi_col)
    return interpret_p_value(p_value)

# Hypothesis 2: Risk Differences Between Zipcodes
# Updated hypothesis_2 function
def hypothesis_2(df, control_value, test_value, kpi_col='TotalClaims'):
    group_a, group_b = segment_data(df, 'PostalCode', control_value, test_value)
    
    if group_a.empty or group_b.empty:
        return "Error: One or both groups are empty. Please check the data and segmentation criteria."
    
    chi2_stat, p_value = perform_chi_squared_test(group_a, group_b, kpi_col)
    return interpret_p_value(p_value)

# Hypothesis 3: Margin Differences Between Zip Codes
def hypothesis_3(df, control_value, test_value, kpi_col='TotalPremium'):
    group_a, group_b = segment_data(df, 'PostalCode', control_value, test_value)
    t_stat, p_value = perform_t_test(group_a, group_b, kpi_col)
    return interpret_p_value(p_value)

# Hypothesis 4: Risk Differences Between Women and Men
def hypothesis_4(df, control_value, test_value, kpi_col):
    """
    Hypothesis 4: Test for risk differences between gender groups.
    :param df: DataFrame containing the data.
    :param control_value: The value representing the control group (e.g., 'F').
    :param test_value: The value representing the test group (e.g., 'M').
    :param kpi_col: Column name for the KPI (e.g., 'TotalClaims').
    :return: Result of the hypothesis test.
    """
    # Verify that the control_value and test_value exist in the Gender column
    if control_value not in df['Gender'].values:
        raise ValueError(f"No data for {control_value} in Gender column.")
    if test_value not in df['Gender'].values:
        raise ValueError(f"No data for {test_value} in Gender column.")
    
    # Segment data into control and test groups
    group_a = df[df['Gender'] == control_value]
    group_b = df[df['Gender'] == test_value]
    
   



# Analyze and report findings, linking them to business strategy and customer experience
def analyze_and_report(hypothesis_func, df, control_value, test_value, kpi_col, hypothesis_description):
    result = hypothesis_func(df, control_value, test_value, kpi_col)
    print(f"--- {hypothesis_description} ---")
    print(result)
    
    # Business Strategy Impact
    if "Reject the null hypothesis" in result:
        print(f"Business Impact: A significant difference was found, indicating {control_value} and {test_value} groups perform differently in terms of {kpi_col}.")
        print("Actionable Insights: Consider adjusting marketing strategies or pricing models to target the underperforming group.")
    else:
        print(f"Business Impact: No significant difference was found between {control_value} and {test_value} groups for {kpi_col}.")
        print("Actionable Insights: No immediate changes are necessary. Continue monitoring.")
    
    print("Customer Experience: These results could influence how different customer segments perceive the product. For instance, addressing significant differences could help in enhancing customer satisfaction.")
    print("------------------------------------------------------\n")

# Hypotheses Testing with Report
def run_hypotheses_tests(df):
    # Hypothesis 1: Risk Differences Across Provinces
    analyze_and_report(hypothesis_1, df, 'Province1', 'Province2', 'TotalClaims', "Hypothesis 1: Risk Differences Across Provinces")
    
    # Hypothesis 2: Risk Differences Between Zipcodes
    analyze_and_report(hypothesis_2, df, '12345', '67890', 'TotalClaims', "Hypothesis 2: Risk Differences Between Zipcodes")
    
    # Hypothesis 3: Margin Differences Between Zip Codes
    analyze_and_report(hypothesis_3, df, '12345', '67890', 'TotalPremium', "Hypothesis 3: Margin Differences Between Zip Codes")
    
    # Hypothesis 4: Risk Differences Between Women and Men
    analyze_and_report(hypothesis_4, df, 'F', 'M', 'TotalClaims', "Hypothesis 4: Risk Differences Between Women and Men")
