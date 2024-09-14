import unittest
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../scripts')))

from data_processing import (
    load_data, summarize_data, check_missing_values
)

class TestDataProcessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Class-level setup to define data file path.
        """
        cls.file_path = '../data/MachineLearningRating_v3.txt'

    def setUp(self):
        """
        Method-level setup to run before each test.
        """
        if not os.path.isfile(self.file_path):
            self.skipTest(f"File {self.file_path} not found")

    def test_load_data(self):
        """
        Test load_data function to check if it returns a DataFrame
        and that the DataFrame is not empty.
        """
        print("Testing load_data function...")
        df = load_data(self.file_path)
        self.assertIsInstance(df, pd.DataFrame, "load_data did not return a DataFrame")
        self.assertFalse(df.empty, "DataFrame is empty")
        print("Passed: load_data")

    def test_summarize_data(self):
        """
        Test summarize_data function to check if it returns
        descriptive statistics as a DataFrame and data types as a Series.
        """
        print("Testing summarize_data function...")
        df = load_data(self.file_path)
        desc_stats, data_types = summarize_data(df)
        self.assertIsInstance(desc_stats, pd.DataFrame, "summarize_data did not return a DataFrame for statistics")
        self.assertIsInstance(data_types, pd.Series, "summarize_data did not return a Series for data types")
        print("Passed: summarize_data")

    def test_check_missing_values(self):
        """
        Test check_missing_values function to check if it correctly
        identifies missing values and returns a Series.
        """
        print("Testing check_missing_values function...")
        df = load_data(self.file_path)
        missing_values = check_missing_values(df)
        self.assertIsInstance(missing_values, pd.Series, "check_missing_values did not return a Series")
        self.assertEqual(missing_values.sum(), 0, "There are missing values in the data")
        print("Passed: check_missing_values")

if __name__ == '__main__':
    unittest.main()
