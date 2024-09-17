import unittest
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../scripts')))
from model_training import (data_preparation, split_data, linear_regression_model, random_forest_model, xgboost_model, evaluate_model, feature_importance_shap
)
class TestModeling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Sample test data for the tests
        data = {
            'TotalPremium': [1000, 1500, 2000, 1200, 1800],
            'TotalClaims': [500, 800, 700, 600, 900],
            'CategoricalFeature': ['A', 'B', 'A', 'B', 'C'],
            'Retention': [1, 0, 1, 0, 1]
        }
        cls.data = pd.DataFrame(data)
        cls.prepared_data = data_preparation(cls.data)

    def test_data_preparation(self):
        # Test if the feature 'ClaimsPerPremium' is created correctly
        self.assertIn('ClaimsPerPremium', self.prepared_data.columns)
        self.assertNotIn('CategoricalFeature', self.prepared_data.select_dtypes(include=['object']).columns)

    def test_split_data(self):
        # Test if the data splits into the correct shapes
        X_train, X_test, y_train, y_test = split_data(self.prepared_data, target_column='Retention')
        self.assertEqual(X_train.shape[0], 3)  # 70% of 5 records should be 3
        self.assertEqual(X_test.shape[0], 2)   # 30% of 5 records should be 2

    def test_linear_regression_model(self):
        X_train, X_test, y_train, y_test = split_data(self.prepared_data, target_column='Retention')
        lr_model = linear_regression_model(X_train, y_train)
        self.assertIsNotNone(lr_model)

    def test_random_forest_model(self):
        X_train, X_test, y_train, y_test = split_data(self.prepared_data, target_column='Retention')
        rf_model = random_forest_model(X_train, y_train)
        self.assertIsNotNone(rf_model)

    def test_xgboost_model(self):
        X_train, X_test, y_train, y_test = split_data(self.prepared_data, target_column='Retention')
        xgb_model = xgboost_model(X_train, y_train)
        self.assertIsNotNone(xgb_model)

    def test_model_evaluation(self):
        X_train, X_test, y_train, y_test = split_data(self.prepared_data, target_column='Retention')
        lr_model = linear_regression_model(X_train, y_train)
        mse = evaluate_model(lr_model, X_test, y_test)
        self.assertIsInstance(mse, float)

if __name__ == '__main__':
    unittest.main()
