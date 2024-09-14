import unittest
import pandas as pd
import numpy as np

# Sample DataFrame for testing
data = {
    'UnderwrittenCoverID': [145249, 145249, 145249, 145255, 145255],
    'PolicyID': [12827, 12827, 12827, 12827, 12827],
    'TransactionMonth': ['2015-03-01', '2015-05-01', '2015-07-01', '2015-05-01', '2015-07-01'],
    'IsVATRegistered': [True, True, True, True, True],
    'Citizenship': ['', '', '', '', ''],
    'LegalType': ['Close Corporation'] * 5,
    'Title': ['Mr'] * 5,
    'Language': ['English'] * 5,
    'Bank': ['First National Bank'] * 5,
    'AccountType': ['Current account'] * 5,
    'ExcessSelected': ['Mobility - Windscreen', 'Mobility - Windscreen', 'Mobility - Windscreen', 'Mobility - Metered Taxis - R2000', 'Mobility - Metered Taxis - R2000'],
    'CoverCategory': ['Windscreen'] * 3 + ['Own damage'] * 2,
    'CoverType': ['Windscreen'] * 3 + ['Own Damage'] * 2,
    'CoverGroup': ['Comprehensive - Taxi'] * 5,
    'Section': ['Motor Comprehensive'] * 5,
    'Product': ['Mobility Metered Taxis: Monthly'] * 5,
    'StatutoryClass': ['Commercial'] * 5,
    'StatutoryRiskType': ['IFRS Constant'] * 5,
    'TotalPremium': [21.929825, 21.929825, 0.0, 512.848070, 0.0],
    'TotalClaims': [0.0] * 5
}

df = pd.DataFrame(data)

class TestInsuranceData(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.sample_data = df.copy()  # Use the sample DataFrame for testing

    def test_check_missing_values(self):
        def check_missing_values(df):
            missing_info = df.isnull().sum()
            return missing_info.to_dict()

        missing_values = check_missing_values(self.sample_data)
        expected_missing_values = {
            'UnderwrittenCoverID': 0,
            'PolicyID': 0,
            'TransactionMonth': 0,
            'IsVATRegistered': 0,
            'Citizenship': 0,
            'LegalType': 0,
            'Title': 0,
            'Language': 0,
            'Bank': 0,
            'AccountType': 0,
            'ExcessSelected': 0,
            'CoverCategory': 0,
            'CoverType': 0,
            'CoverGroup': 0,
            'Section': 0,
            'Product': 0,
            'StatutoryClass': 0,
            'StatutoryRiskType': 0,
            'TotalPremium': 0,
            'TotalClaims': 0
        }
        self.assertEqual(missing_values, expected_missing_values)
    
    def test_claim_amount_clipping(self):
        self.sample_data['TotalPremium'] = self.sample_data['TotalPremium'].clip(lower=0)
        valid_claim_amounts = self.sample_data['TotalPremium']
        self.assertTrue((valid_claim_amounts >= 0).all(), "ClaimAmount values should be non-negative.")
    


    def test_summarize_data(self):
        def summarize_data(df):
            summary = df.describe(include='all')
            return summary.to_dict()

        summary = summarize_data(self.sample_data)
        self.assertIsInstance(summary, dict, "Summary should be a dictionary.")

if __name__ == '__main__':
    unittest.main()
