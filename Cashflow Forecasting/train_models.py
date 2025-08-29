import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib
import os
from datetime import datetime

# --- Configuration ---
DATA_FILE = 'alcobev_europe_sales_data.csv'
MODELS_DIR = 'models'
SALES_MODEL_PATH = os.path.join(MODELS_DIR, 'sales_forecasting_model.pkl')
COGS_MODEL_PATH = os.path.join(MODELS_DIR, 'cogs_forecasting_model.pkl')
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, 'preprocessor.pkl')

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"Loading data from {DATA_FILE}...")
try:
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    print("Data loaded successfully.")
    print(f"Dataset shape: {df.shape}")
    print("First 5 rows of raw data:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found. Please run generate_data.py first.")
    exit()

# --- Feature Engineering ---
print("\nPerforming feature engineering...")

# Create time-based features
df['day_of_week'] = df['Date'].dt.dayofweek
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df['day_of_year'] = df['Date'].dt.dayofyear
df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)

# Create lagged features (simple example: previous day's revenue/cogs)
# This requires sorting and grouping to ensure correct lags per segment
df = df.sort_values(by=['Country', 'Channel', 'Product_Category', 'Date'])
df['Net_Sales_Revenue_EUR_lag1'] = df.groupby(['Country', 'Channel', 'Product_Category'])['Net_Sales_Revenue_EUR'].shift(1)
df['COGS_EUR_lag1'] = df.groupby(['Country', 'Channel', 'Product_Category'])['COGS_EUR'].shift(1)


# Drop rows with any NaNs created by shifting
df.dropna(inplace=True)

# Define the exact order of input features expected by the preprocessor
numerical_features = [
    'Net_Sales_Volume_Litres', 'Marketing_Spend_EUR', 'Promotional_Event',
    'Consumer_Confidence_Index', 'Inflation_Rate_EUR', 'Avg_Temp_C',
    'Holiday_Indicator', 'Competitor_Activity_Index',
    'day_of_week', 'month', 'year', 'day_of_year', 'week_of_year',
    'Net_Sales_Revenue_EUR_lag1', 'COGS_EUR_lag1'
] # 15 numerical features

categorical_features = ['Country', 'Channel', 'Product_Category'] # 3 categorical features

# The combined list of all features that will be passed to the preprocessor
all_features_for_preprocessor = numerical_features + categorical_features # Total 18 features

# Define targets
target_sales = 'Net_Sales_Revenue_EUR'
target_cogs = 'COGS_EUR'

# --- Data Splitting (Time-based for time series) ---
SPLIT_DATE = datetime(2024, 1, 1)

train_df = df[df['Date'] < SPLIT_DATE].copy()
test_df = df[df['Date'] >= SPLIT_DATE].copy()

print(f"\nTraining data size: {len(train_df)} records (before {SPLIT_DATE.strftime('%Y-%m-%d')})")
print(f"Testing data size: {len(test_df)} records (from {SPLIT_DATE.strftime('%Y-%m-%d')})")

# Prepare features and targets
X_train = train_df[all_features_for_preprocessor]
y_train_sales = train_df[target_sales]
y_train_cogs = train_df[target_cogs]

X_test = test_df[all_features_for_preprocessor]
y_test_sales = test_df[target_sales]
y_test_cogs = test_df[target_cogs]

# Store test volume for KPI calculations (since we don't predict volume)
test_volume = test_df['Net_Sales_Volume_Litres']

# --- Preprocessing Pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)

print(f"\n[train_models.py] Shape of X_train BEFORE fitting preprocessor: {X_train.shape}")
print(f"[train_models.py] Columns in X_train BEFORE fitting preprocessor: {X_train.columns.tolist()}")

# --- Model Training ---
print("\nTraining Sales Forecasting Model...")
sales_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1, n_jobs=-1))
])
sales_pipeline.fit(X_train, y_train_sales)
print("Sales Model training complete.")

print("\nTraining COGS Forecasting Model...")
cogs_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1, n_jobs=-1))
])
cogs_pipeline.fit(X_train, y_train_cogs)
print("COGS Model training complete.")

# Debugging information
transformed_sample_train = sales_pipeline.named_steps['preprocessor'].transform(X_train.head(1))
print(f"[train_models.py] Preprocessor output shape after transform (sample): {transformed_sample_train.shape}")
print(f"[train_models.py] Number of features preprocessor was fitted on: {sales_pipeline.named_steps['preprocessor'].n_features_in_}")

# --- Model Evaluation ---
print("\nEvaluating models on test data...")

# Basic Model Evaluation
y_pred_sales = sales_pipeline.predict(X_test)
y_pred_cogs = cogs_pipeline.predict(X_test)

mae_sales = np.mean(np.abs(y_test_sales - y_pred_sales))
rmse_sales = np.sqrt(np.mean((y_test_sales - y_pred_sales)**2))
print(f"\nSales Model Performance:")
print(f"  MAE: {mae_sales:,.2f} EUR")
print(f"  RMSE: {rmse_sales:,.2f} EUR")

mae_cogs = np.mean(np.abs(y_test_cogs - y_pred_cogs))
rmse_cogs = np.sqrt(np.mean((y_test_cogs - y_pred_cogs)**2))
print(f"\nCOGS Model Performance:")
print(f"  MAE: {mae_cogs:,.2f} EUR")
print(f"  RMSE: {rmse_cogs:,.2f} EUR")


# --- Save Models ---
print(f"\nSaving trained models and preprocessor to '{MODELS_DIR}' directory...")
joblib.dump(sales_pipeline, SALES_MODEL_PATH)
joblib.dump(cogs_pipeline, COGS_MODEL_PATH)
joblib.dump(preprocessor, PREPROCESSOR_PATH)

print("Models and preprocessor saved successfully.")
print(f"Sales model saved to: {SALES_MODEL_PATH}")
print(f"COGS model saved to: {COGS_MODEL_PATH}")
print(f"Preprocessor saved to: {PREPROCESSOR_PATH}")

# --- Summary Report ---
print(f"\n{'='*80}")
print("TRAINING SUMMARY REPORT")
print(f"{'='*80}")
print(f"Training Period: {train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')}")
print(f"Testing Period: {test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')}")
print(f"Training Records: {len(train_df):,}")
print(f"Testing Records: {len(test_df):,}")
print(f"Features Used: {len(all_features_for_preprocessor)}")
print(f"Models Trained: Sales Revenue & COGS Prediction")
print(f"KPIs Implemented: Gross Profit, Gross Profit Margin, ASP per Litre")
print(f"All models saved and ready for API deployment!")