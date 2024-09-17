import numpy as np
import pandas as pd
import sys
import os
import unittest
from scipy.stats import ttest_ind, chi2_contingency


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from ab_testing_analysis import (load_data, clean_data, segment_data, perform_t_test, perform_chi_squared_test, perform_z_test, 
    cohen_d, interpret_p_value, hypothesis_1, hypothesis_2, hypothesis_3, hypothesis_4, analyze_and_report)


class TestABTestingAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_data = pd.DataFrame({
            'TotalClaims': [10, 15, 20, 25, 30, 10, 15, 20],
            'PostalCode': ['12345', '12345', '67890', '67890', '12345', '12345', '67890', '67890'],
            'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female'],
            'Province': ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B']
        })

    def test_perform_t_test(self):
        group_a = self.mock_data[self.mock_data['Province'] == 'A']
        group_b = self.mock_data[self.mock_data['Province'] == 'B']
        t_stat, p_value = perform_t_test(group_a, group_b, 'TotalClaims')
        expected_t_stat = 0.1  # Example expected value, adjust as necessary
        self.assertTrue(np.isclose(t_stat, expected_t_stat, atol=0.01), "T-statistic does not match the expected value")

    def test_perform_chi_squared_test(self):
        group_a = self.mock_data[self.mock_data['PostalCode'] == '12345']
        group_b = self.mock_data[self.mock_data['PostalCode'] == '67890']
        chi2_stat, p_value = perform_chi_squared_test(group_a, group_b, 'PostalCode')
        self.assertTrue(np.isfinite(chi2_stat) and np.isfinite(p_value), "Chi-squared test result is not finite")

  

    def test_hypothesis_2(self):
        result = hypothesis_2(self.mock_data, '12345', '67890', kpi_col='TotalClaims')
        self.assertIsNotNone(result, "Hypothesis result should not be None")

    def test_hypothesis_4(self):
        result = hypothesis_4(self.mock_data, 'Female', 'Male', kpi_col='TotalClaims')
        self.assertIsNotNone(result, "Hypothesis result should not be None")

if __name__ == '__main__':
    unittest.main()