#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime

# Load data
df = pd.read_csv('alcobev_europe_sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

print(f"Original data shape: {df.shape}")
print(f"Original columns: {df.columns.tolist()}")

# Replicate feature engineering from train_models.py
df['day_of_week'] = df['Date'].dt.dayofweek
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df['day_of_year'] = df['Date'].dt.dayofyear
df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)

# Create lagged features
df = df.sort_values(by=['Country', 'Channel', 'Product_Category', 'Date'])
df['Net_Sales_Revenue_EUR_lag1'] = df.groupby(['Country', 'Channel', 'Product_Category'])['Net_Sales_Revenue_EUR'].shift(1)
df['COGS_EUR_lag1'] = df.groupby(['Country', 'Channel', 'Product_Category'])['COGS_EUR'].shift(1)
df['Net_Sales_Volume_Litres_lag1'] = df.groupby(['Country', 'Channel', 'Product_Category'])['Net_Sales_Volume_Litres'].shift(1)

# Drop rows with any NaNs created by shifting
df.dropna(inplace=True)

print(f"\nAfter feature engineering: {df.shape}")
print(f"Columns after engineering: {df.columns.tolist()}")

# Test split
SPLIT_DATE = datetime(2024, 1, 1)
test_df = df[df['Date'] >= SPLIT_DATE].copy()

print(f"\nTest data shape: {test_df.shape}")

# Expected features from train_models.py
numerical_features = [
    'Net_Sales_Volume_Litres', 'Marketing_Spend_EUR', 'Promotional_Event',
    'Consumer_Confidence_Index', 'Inflation_Rate_EUR', 'Avg_Temp_C',
    'Holiday_Indicator', 'Competitor_Activity_Index',
    'day_of_week', 'month', 'year', 'day_of_year', 'week_of_year',
    'Net_Sales_Revenue_EUR_lag1', 'COGS_EUR_lag1'
]

categorical_features = ['Country', 'Channel', 'Product_Category']
all_features_for_preprocessor = numerical_features + categorical_features

print(f"\nExpected features for sales/cogs models: {len(all_features_for_preprocessor)}")
print(f"Expected features list: {all_features_for_preprocessor}")

# Check if all expected features exist
missing_features = [f for f in all_features_for_preprocessor if f not in test_df.columns]
if missing_features:
    print(f"Missing features: {missing_features}")
else:
    print("All expected features are present")

# Test feature extraction like in testcase.py
feature_cols = [col for col in test_df.columns if col not in ['Net_Sales_Revenue_EUR', 'COGS_EUR', 'Date']]
X_test = test_df[feature_cols]

print(f"\nActual X_test shape: {X_test.shape}")
print(f"Actual feature columns: {X_test.columns.tolist()}")

print(f"\nDifference: Expected {len(all_features_for_preprocessor)} features, got {X_test.shape[1]} features")

# Check volume features
volume_numerical_features = [
    'Marketing_Spend_EUR', 'Promotional_Event',
    'Consumer_Confidence_Index', 'Inflation_Rate_EUR', 'Avg_Temp_C',
    'Holiday_Indicator', 'Competitor_Activity_Index',
    'day_of_week', 'month', 'year', 'day_of_year', 'week_of_year',
    'Net_Sales_Volume_Litres_lag1'
]

volume_features = volume_numerical_features + categorical_features

print(f"\nExpected volume features: {len(volume_features)}")
print(f"Expected volume features list: {volume_features}")

volume_feature_cols = [col for col in test_df.columns if col not in ['Net_Sales_Volume_Litres', 'Date']]
vX_test = test_df[volume_feature_cols]

print(f"Actual volume X_test shape: {vX_test.shape}")
print(f"Actual volume feature columns: {vX_test.columns.tolist()}")
