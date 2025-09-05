#!/usr/bin/env python3
"""
Debug categorical features to understand feature mismatch
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from testcase import AccuracyTester

def debug_categorical_features():
    print("=== DEBUGGING CATEGORICAL FEATURES ===")
    
    # Initialize tester and load data
    tester = AccuracyTester()
    tester.load_data_and_models()
    
    # Split data same as training
    SPLIT_DATE = datetime(2024, 1, 1)
    train_df = tester.df[tester.df['Date'] < SPLIT_DATE].copy()
    test_df = tester.df[tester.df['Date'] >= SPLIT_DATE].copy()
    
    print(f"Train data: {len(train_df)} records")
    print(f"Test data: {len(test_df)} records")
    
    categorical_features = ['Country', 'Channel', 'Product_Category']
    
    for cat_feat in categorical_features:
        train_values = set(train_df[cat_feat].unique())
        test_values = set(test_df[cat_feat].unique())
        
        print(f"\n{cat_feat}:")
        print(f"  Train unique values ({len(train_values)}): {sorted(train_values)}")
        print(f"  Test unique values ({len(test_values)}): {sorted(test_values)}")
        print(f"  Test-only values: {sorted(test_values - train_values)}")
        print(f"  Train-only values: {sorted(train_values - test_values)}")
    
    # Load preprocessor and check its feature names
    try:
        preprocessor = joblib.load('models/preprocessor.pkl')
        
        # Use the prepared features same as testcase.py
        df_prepared = tester.prepare_features(tester.df)
        test_split_date = pd.to_datetime('2024-01-01')
        test_df_prepared = df_prepared[df_prepared['Date'] >= test_split_date].copy()
        
        # Get expected feature names
        expected_sales_cogs_features = [
            'Net_Sales_Volume_Litres', 'Marketing_Spend_EUR', 'Promotional_Event',
            'Consumer_Confidence_Index', 'Inflation_Rate_EUR', 'Avg_Temp_C',
            'Holiday_Indicator', 'Competitor_Activity_Index',
            'day_of_week', 'month', 'year', 'day_of_year', 'week_of_year',
            'Net_Sales_Revenue_EUR_lag1', 'COGS_EUR_lag1',
            'Country', 'Channel', 'Product_Category'
        ]
        
        # Prepare test features exactly as in testcase.py
        test_features_for_preprocessor = test_df_prepared[expected_sales_cogs_features].copy()
        
        print(f"\nTest features shape before preprocessing: {test_features_for_preprocessor.shape}")
        print(f"Test features columns: {list(test_features_for_preprocessor.columns)}")
        
        # Apply preprocessor
        test_transformed = preprocessor.transform(test_features_for_preprocessor.head(10))
        print(f"Test features shape after preprocessing: {test_transformed.shape}")
        
        # Get feature names from preprocessor
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = preprocessor.get_feature_names_out()
            print(f"Preprocessor feature names ({len(feature_names)}): {list(feature_names)}")
        
        # Check what the model expects
        xgb_sales_model = tester.loaded_models['xgboost']['sales']
        if hasattr(xgb_sales_model, 'named_steps'):
            regressor = xgb_sales_model.named_steps['regressor']
            if hasattr(regressor, 'n_features_in_'):
                print(f"XGBoost regressor expects: {regressor.n_features_in_} features")
            
    except Exception as e:
        print(f"Error checking preprocessor: {e}")

if __name__ == "__main__":
    debug_categorical_features()
