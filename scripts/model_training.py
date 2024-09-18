# Import necessary libraries
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import shap
import matplotlib.pyplot as plt
def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input DataFrame by handling missing data, removing duplicates,
    and addressing outliers.

    Args:
        data (pd.DataFrame): The raw data as a pandas DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # 1. Handle missing data
    # Drop columns with more than 50% missing values
    threshold = len(data) * 0.5
    data = data.dropna(thresh=threshold, axis=1)
    
    # Fill missing numerical values with the column median
    num_cols = data.select_dtypes(include=[np.number]).columns
    data[num_cols] = data[num_cols].fillna(data[num_cols].median())

    # Fill missing categorical values with the mode (most frequent)
    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        data[col] = data[col].fillna(data[col].mode()[0])

    # 2. Remove duplicates
    data = data.drop_duplicates()

    # 3. Handle outliers (using Z-score method for numerical columns)
    z_thresh = 3  # threshold for Z-score
    for col in num_cols:
        col_zscore = (data[col] - data[col].mean()) / data[col].std()
        data = data[(col_zscore < z_thresh) & (col_zscore > -z_thresh)]
    
    # 4. Data type conversion (optional but recommended for efficiency)
    for col in num_cols:
        if data[col].dtype == 'float64':
            data[col] = data[col].astype('float32')
        if data[col].dtype == 'int64':
            data[col] = data[col].astype('int32')

    return data
# Data Preparation Function
def data_preparation(data):
    # Handle missing data: Impute or remove
    data.fillna(data.mean(), inplace=True)

def feature_engineering(data):
    #  feature engineering
    data['ClaimsPerPremium'] = data['TotalClaims'] / data['TotalPremium']
    return data


def label_encode(data, columns):
    for column in columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
    return data

def one_hot_encode(data, columns):
    data = pd.get_dummies(data, columns=columns, prefix=columns)
    return data

def encode_categorical_data(data, categorical_columns, encode_type='one-hot'):
    if encode_type == 'label':
        data = label_encode(data, categorical_columns)
    elif encode_type == 'one-hot':
        data = one_hot_encode(data, categorical_columns)
    return data

    return data

# Split Data into Train and Test sets
def split_data(data, target_column, test_size=0.3):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Linear Regression Model
def train_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2

# Random Forest Model
def random_forest_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# XGBoost Model
def xgboost_model(X_train, y_train):
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

# Feature Importance using SHAP
def feature_importance_shap(model, X_train):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train)
