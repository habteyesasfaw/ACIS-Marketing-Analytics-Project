import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../scripts')))

from data_processing import (
    load_data, summarize_data, check_missing_values, 
    plot_histograms, plot_categorical_counts, detect_outliers
)

def file_exists(file_path):
    return os.path.isfile(file_path)

def test_load_data():
    print("Testing load_data function...")
    file_path = '../data/MachineLearningRating_v3.txt'
    
    if not file_exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return
    
    df = load_data(file_path)
    assert isinstance(df, pd.DataFrame), "Failed: load_data did not return a DataFrame"
    assert not df.empty, "Failed: DataFrame is empty"
    print("Passed: load_data")

def test_summarize_data():
    print("Testing summarize_data function...")
    file_path = '../data/MachineLearningRating_v3.txt'
    
    if not file_exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return
    
    df = load_data(file_path)
    desc_stats, data_types = summarize_data(df)
    assert isinstance(desc_stats, pd.DataFrame), "Failed: summarize_data did not return a DataFrame"
    assert isinstance(data_types, pd.Series), "Failed: summarize_data did not return a Series"
    print("Passed: summarize_data")

def test_check_missing_values():
    print("Testing check_missing_values function...")
    file_path = '../data/MachineLearningRating_v3.txt'
    
    if not file_exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return
    
    df = load_data(file_path)
    missing_values = check_missing_values(df)
    assert isinstance(missing_values, pd.Series), "Failed: check_missing_values did not return a Series"
    assert missing_values.sum() == 0, "Failed: There are missing values"
    print("Passed: check_missing_values")

def run_tests():
    test_load_data()
    test_summarize_data()
    test_check_missing_values()

if __name__ == "__main__":
    run_tests()
