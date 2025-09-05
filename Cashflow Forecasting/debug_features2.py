#!/usr/bin/env python3
"""
Debug script to check exact features being passed to models
"""

import pandas as pd
import numpy as np
from testcase import AccuracyTester

def debug_features():
    print("=== DEBUGGING FEATURE EXTRACTION ===")
    
    # Initialize tester
    tester = AccuracyTester()
    tester.load_data_and_models()  # Load data first
    
    # Prepare data same as testcase
    df_prepared = tester.prepare_features(tester.df)
    df_v_prepared = tester.volume_features(tester.df)
    test_split_date = pd.to_datetime('2024-01-01')
    
    # Split data
    test_df = df_prepared[df_prepared['Date'] >= test_split_date].copy()
    test_v_df = df_v_prepared[df_v_prepared['Date'] >= test_split_date].copy()
    
    print(f"Test data shapes: Sales/COGS={test_df.shape}, Volume={test_v_df.shape}")
    print(f"Test data columns: Sales/COGS={list(test_df.columns)}")
    print(f"Test data columns: Volume={list(test_v_df.columns)}")
    
    # Check exact features being used
    expected_sales_cogs_features = [
        'Net_Sales_Volume_Litres', 'Marketing_Spend_EUR', 'Promotional_Event',
        'Consumer_Confidence_Index', 'Inflation_Rate_EUR', 'Avg_Temp_C',
        'Holiday_Indicator', 'Competitor_Activity_Index',
        'day_of_week', 'month', 'year', 'day_of_year', 'week_of_year',
        'Net_Sales_Revenue_EUR_lag1', 'COGS_EUR_lag1',
        'Country', 'Channel', 'Product_Category'
    ]
    
    expected_volume_features = [
        'Marketing_Spend_EUR', 'Promotional_Event',
        'Consumer_Confidence_Index', 'Inflation_Rate_EUR', 'Avg_Temp_C',
        'Holiday_Indicator', 'Competitor_Activity_Index',
        'day_of_week', 'month', 'year', 'day_of_year', 'week_of_year',
        'Net_Sales_Volume_Litres_lag1',
        'Country', 'Channel', 'Product_Category'
    ]
    
    print(f"\nExpected Sales/COGS features ({len(expected_sales_cogs_features)}): {expected_sales_cogs_features}")
    print(f"Expected Volume features ({len(expected_volume_features)}): {expected_volume_features}")
    
    # Check if all expected features exist
    missing_sales_cogs = [f for f in expected_sales_cogs_features if f not in test_df.columns]
    missing_volume = [f for f in expected_volume_features if f not in test_v_df.columns]
    
    print(f"\nMissing Sales/COGS features: {missing_sales_cogs}")
    print(f"Missing Volume features: {missing_volume}")
    
    # Extract features
    try:
        X_test = test_df[expected_sales_cogs_features]
        vX_test = test_v_df[expected_volume_features]
        
        print(f"\nActual feature shapes extracted:")
        print(f"  Sales/COGS X_test: {X_test.shape}")
        print(f"  Volume vX_test: {vX_test.shape}")
        
        # Load and test with a Random Forest model (if available)
        import joblib
        try:
            preprocessor = joblib.load('models/randomforest/preprocessor.pkl')
            sales_model = joblib.load('models/randomforest/sales_forecasting_model.pkl')
            
            print(f"\nProcessing with RandomForest preprocessor...")
            X_processed = preprocessor.transform(X_test)
            print(f"  After preprocessing: {X_processed.shape}")
            print(f"  Model expects: {sales_model.n_features_in_} features")
            
        except Exception as e:
            print(f"  Error loading RandomForest models: {e}")
            
    except Exception as e:
        print(f"Error extracting features: {e}")

if __name__ == "__main__":
    debug_features()
