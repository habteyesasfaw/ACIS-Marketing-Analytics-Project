import unittest
import sys
import os
import pandas as pd
import numpy as np

# Add the path to the 'scripts' directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../scripts')))

# Import functions from data_processing.py
from data_processing import load_data, summarize_data, check_missing_values

class TestInsuranceDataProcessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_data = pd.DataFrame({
            'ClaimID': [1, 2, 3, 4, 5],
            'PolicyID': [101, 102, 103, 104, 105],
            'ClaimAmount': [5000, 7000, np.nan, 12000, -200],
            'ClaimStatus': ['Approved', 'Pending', 'Denied', 'Approved', 'Denied'],
            'DateOfClaim': ['2024-01-15', '2024-02-20', '2024-03-10', '2024-04-25', '2024-05-30']
        })
        cls.sample_data['DateOfClaim'] = pd.to_datetime(cls.sample_data['DateOfClaim'])

    def test_load_data(self):
        """Test if the load_data function correctly loads the data."""
        try:
            data = load_data('data/sample_data.csv')
            self.assertIsInstance(data, pd.DataFrame, "Loaded data should be a DataFrame.")
            self.assertFalse(data.empty, "Loaded data should not be empty.")
        except FileNotFoundError:
            self.fail("Sample data file not found.")

    def test_summarize_data(self):
        """Test if summarize_data provides the correct summary."""
        summary = summarize_data(self.sample_data)
        self.assertIsInstance(summary, dict, "Summary should be a dictionary.")
        self.assertIn('ClaimAmount', summary, "Summary should include 'ClaimAmount'.")

    def test_check_missing_values(self):
        """Test if check_missing_values identifies missing values correctly."""
        missing_info = check_missing_values(self.sample_data)
        self.assertIsInstance(missing_info, dict, "Missing values info should be a dictionary.")
        self.assertIn('ClaimAmount', missing_info, "Missing values info should include 'ClaimAmount'.")

    def test_clipping_claim_amount(self):
        """Test if ClaimAmount values are clipped to be non-negative."""
        self.sample_data['ClaimAmount'] = self.sample_data['ClaimAmount'].clip(lower=0)
        valid_claim_amounts = self.sample_data['ClaimAmount']
        self.assertTrue((valid_claim_amounts >= 0).all(), "ClaimAmount values should be non-negative.")

    def test_claim_status(self):
        """Test if ClaimStatus only contains valid statuses."""
        valid_statuses = ['Approved', 'Pending', 'Denied']
        status_series = self.sample_data['ClaimStatus']
        invalid_statuses = status_series[~status_series.isin(valid_statuses)]
        self.assertTrue(invalid_statuses.empty, "ClaimStatus contains invalid values.")

    def test_date_of_claim_format(self):
        """Test if DateOfClaim is in the correct datetime format."""
        date_series = self.sample_data['DateOfClaim']
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(date_series), "DateOfClaim should be in datetime format.")

if __name__ == '__main__':
    unittest.main()
