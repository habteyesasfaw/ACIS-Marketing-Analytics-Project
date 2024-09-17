# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import shap
import matplotlib.pyplot as plt

# Data Cleaning Function
def clean_data(data):
    # Handle missing data: Impute missing numerical data with mean
    for col in data.select_dtypes(include=[np.number]).columns:
        data[col].fillna(data[col].mean(), inplace=True)

    # Drop rows with missing values in categorical columns
    data.dropna(subset=data.select_dtypes(include=[object]).columns, inplace=True)

    # Remove duplicate rows
    data.drop_duplicates(inplace=True)

    # Handle outliers (example: capping at 95th percentile)
    for col in data.select_dtypes(include=[np.number]).columns:
        upper_limit = data[col].quantile(0.95)
        data[col] = np.where(data[col] > upper_limit, upper_limit, data[col])

    return data

# Data Preparation Function
def data_preparation(data):
    # Call clean_data function
    data = clean_data(data)

    # Feature Engineering: Example feature creation
    data['ClaimsPerPremium'] = data['TotalClaims'] / data['TotalPremium']

    # Encoding categorical data
    label_encoder = LabelEncoder()
    data['CategoricalFeature'] = label_encoder.fit_transform(data['CategoricalFeature'])

    return data

# Split Data into Train and Test sets
def split_data(data, target_column, test_size=0.3):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Linear Regression Model
def linear_regression_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

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
