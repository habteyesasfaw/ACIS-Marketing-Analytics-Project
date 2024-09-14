import pandas as pd

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def summarize_data(df):
    """Summarize the data."""
    return df.describe(), df.dtypes

def check_missing_values(df):
    """Check for missing values in the dataframe."""
    return df.isnull().sum()
def plot_histograms(df, numerical_columns):
    """
    Plot histograms for numerical columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing data.
    numerical_columns (list): List of numerical column names to plot histograms for.
    """
    import matplotlib.pyplot as plt

    if df is None or not numerical_columns:
        print("DataFrame is not loaded or no numerical columns provided.")
        return

    df[numerical_columns].hist(bins=30, figsize=(12, 10))
    plt.suptitle('Distribution of Numerical Features')
    plt.show()

def plot_categorical_counts(df, categorical_columns):
    """
    Plot bar charts for categorical columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing data.
    categorical_columns (list): List of categorical column names to plot bar charts for.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if df is None or not categorical_columns:
        print("DataFrame is not loaded or no categorical columns provided.")
        return

    sns.set(style="whitegrid")
    for col in categorical_columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index)
        plt.title(f'Distribution of {col}')
        plt.show()

def detect_outliers(df, numerical_columns):
    """
    Detect outliers in numerical columns using box plots.

    Parameters:
    df (pd.DataFrame): The DataFrame containing data.
    numerical_columns (list): List of numerical column names to check for outliers.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if df is None or not numerical_columns:
        print("DataFrame is not loaded or no numerical columns provided.")
        return

    sns.set(style="whitegrid")
    for col in numerical_columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[col])
        plt.title(f'Box Plot for {col}')
        plt.show()