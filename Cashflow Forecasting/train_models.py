import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import os
from datetime import datetime

# --- Configuration ---
DATA_FILE = 'alcobev_europe_sales_data.csv'
MODELS_DIR = 'models'

# XGBoost model paths (main models - unchanged)
SALES_MODEL_PATH = os.path.join(MODELS_DIR, 'sales_forecasting_model.pkl')
COGS_MODEL_PATH = os.path.join(MODELS_DIR, 'cogs_forecasting_model.pkl')
VOLUME_MODEL_PATH = os.path.join(MODELS_DIR, 'volume_forecasting_model.pkl')
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, 'preprocessor.pkl')

# Algorithm-specific directories
XGBOOST_DIR = os.path.join(MODELS_DIR, 'xgboost')
LIGHTGBM_DIR = os.path.join(MODELS_DIR, 'lightgbm')
RANDOMFOREST_DIR = os.path.join(MODELS_DIR, 'randomforest')

# Ensure all directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(XGBOOST_DIR, exist_ok=True)
os.makedirs(LIGHTGBM_DIR, exist_ok=True)
os.makedirs(RANDOMFOREST_DIR, exist_ok=True)

# --- Helper Functions ---
def create_algorithm_models():
    """Create model instances for each algorithm"""
    return {
        'xgboost': XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1, n_jobs=-1),
        'lightgbm': LGBMRegressor(random_state=42, n_estimators=100, learning_rate=0.1, n_jobs=-1, verbose=-1),
        'randomforest': RandomForestRegressor(random_state=42, n_estimators=50, n_jobs=-1)  # Reduced from 100 to 50
    }

def train_and_evaluate_algorithm(algorithm_name, model, preprocessor, X_train, y_train, X_test, y_test, target_name):
    """Train and evaluate a single algorithm"""
    print(f"\nTraining {target_name} Model with {algorithm_name.upper()}...")
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    
    if target_name.lower() == 'volume':
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        print(f"  {algorithm_name.upper()} {target_name} Model - MAE: {mae:,.2f} Litres, RMSE: {rmse:,.2f} Litres, MAPE: {mape:.2f}%")
    else:
        print(f"  {algorithm_name.upper()} {target_name} Model - MAE: {mae:,.2f} EUR, RMSE: {rmse:,.2f} EUR")
    
    return pipeline, mae, rmse

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
df['Net_Sales_Volume_Litres_lag1'] = df.groupby(['Country', 'Channel', 'Product_Category'])['Net_Sales_Volume_Litres'].shift(1)

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

# NEW: Volume model features (EXCLUDES volume itself)
volume_numerical_features = [
    'Marketing_Spend_EUR', 'Promotional_Event',
    'Consumer_Confidence_Index', 'Inflation_Rate_EUR', 'Avg_Temp_C',
    'Holiday_Indicator', 'Competitor_Activity_Index',
    'day_of_week', 'month', 'year', 'day_of_year', 'week_of_year',
    'Net_Sales_Volume_Litres_lag1'  # Use yesterday's volume to predict today's
] # 13 numerical features

categorical_features = ['Country', 'Channel', 'Product_Category'] # 3 categorical features

# The combined list of all features
all_features_for_preprocessor = numerical_features + categorical_features # Total 18 features
volume_features = volume_numerical_features + categorical_features # Total 16 features

# Define targets
target_sales = 'Net_Sales_Revenue_EUR'
target_cogs = 'COGS_EUR'
target_volume = 'Net_Sales_Volume_Litres'  # NEW

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

# NEW: Volume model data
X_train_volume = train_df[volume_features]
y_train_volume = train_df[target_volume]
X_test_volume = test_df[volume_features]
y_test_volume = test_df[target_volume]

# Store test volume for KPI calculations
test_volume = test_df['Net_Sales_Volume_Litres']

# --- Preprocessing Pipeline ---
# EXISTING PREPROCESSOR - NO CHANGES
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)

# NEW: Volume preprocessor (separate for now, but simple)
volume_preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', volume_numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)

print(f"\n[train_models.py] Shape of X_train BEFORE fitting preprocessor: {X_train.shape}")
print(f"[train_models.py] Columns in X_train BEFORE fitting preprocessor: {X_train.columns.tolist()}")

# --- Model Training ---
print("\n" + "="*80)
print("TRAINING ALL ALGORITHMS (XGBoost, LightGBM, Random Forest)")
print("="*80)

# Get algorithm models
algorithms = create_algorithm_models()

# Store trained models and their performance
trained_models = {}
model_performance = {}

# Train Sales Models for all algorithms
print(f"\n{'='*60}")
print("TRAINING SALES FORECASTING MODELS")
print(f"{'='*60}")

for algo_name, model in algorithms.items():
    pipeline, mae, rmse = train_and_evaluate_algorithm(
        algo_name, model, preprocessor, X_train, y_train_sales, X_test, y_test_sales, "Sales"
    )
    trained_models[f'{algo_name}_sales'] = pipeline
    model_performance[f'{algo_name}_sales'] = {'mae': mae, 'rmse': rmse}

# Train COGS Models for all algorithms
print(f"\n{'='*60}")
print("TRAINING COGS FORECASTING MODELS")
print(f"{'='*60}")

for algo_name, model in algorithms.items():
    pipeline, mae, rmse = train_and_evaluate_algorithm(
        algo_name, model, preprocessor, X_train, y_train_cogs, X_test, y_test_cogs, "COGS"
    )
    trained_models[f'{algo_name}_cogs'] = pipeline
    model_performance[f'{algo_name}_cogs'] = {'mae': mae, 'rmse': rmse}

# Train Volume Models for all algorithms
print(f"\n{'='*60}")
print("TRAINING VOLUME FORECASTING MODELS")
print(f"{'='*60}")

for algo_name, model in algorithms.items():
    pipeline, mae, rmse = train_and_evaluate_algorithm(
        algo_name, model, volume_preprocessor, X_train_volume, y_train_volume, X_test_volume, y_test_volume, "Volume"
    )
    trained_models[f'{algo_name}_volume'] = pipeline
    model_performance[f'{algo_name}_volume'] = {'mae': mae, 'rmse': rmse}

# Keep XGBoost models as the main models for backward compatibility
sales_pipeline = trained_models['xgboost_sales']
cogs_pipeline = trained_models['xgboost_cogs']
volume_pipeline = trained_models['xgboost_volume']

# Debugging information
transformed_sample_train = sales_pipeline.named_steps['preprocessor'].transform(X_train.head(1))
print(f"\n[train_models.py] Preprocessor output shape after transform (sample): {transformed_sample_train.shape}")
print(f"[train_models.py] Number of features preprocessor was fitted on: {sales_pipeline.named_steps['preprocessor'].n_features_in_}")

# --- Model Evaluation Summary ---
print("\n" + "="*80)
print("MODEL PERFORMANCE COMPARISON")
print("="*80)

# Display performance comparison
print(f"\n{'Algorithm':<15} {'Target':<10} {'MAE':<15} {'RMSE':<15}")
print("-" * 60)

for model_key, performance in model_performance.items():
    algo, target = model_key.split('_', 1)
    mae = performance['mae']
    rmse = performance['rmse']
    
    if target == 'volume':
        print(f"{algo.upper():<15} {target.upper():<10} {mae:>10,.0f} L   {rmse:>10,.0f} L")
    else:
        print(f"{algo.upper():<15} {target.upper():<10} €{mae:>10,.0f}   €{rmse:>10,.0f}")

# Extract XGBoost metrics for backward compatibility reporting
mae_sales = model_performance['xgboost_sales']['mae']
rmse_sales = model_performance['xgboost_sales']['rmse']
mae_cogs = model_performance['xgboost_cogs']['mae']
rmse_cogs = model_performance['xgboost_cogs']['rmse']
mae_volume = model_performance['xgboost_volume']['mae']
rmse_volume = model_performance['xgboost_volume']['rmse']

# Calculate MAPE for volume (using the stored values from training)
# Note: MAPE calculation is already done in the training loop

# --- Save Models ---
print(f"\n" + "="*80)
print("SAVING ALL MODELS")
print("="*80)

# Save main XGBoost models (for backward compatibility)
print(f"Saving main XGBoost models to '{MODELS_DIR}' directory...")
joblib.dump(sales_pipeline, SALES_MODEL_PATH)
joblib.dump(cogs_pipeline, COGS_MODEL_PATH)
joblib.dump(volume_pipeline, VOLUME_MODEL_PATH)
joblib.dump(preprocessor, PREPROCESSOR_PATH)

print("Main XGBoost models saved successfully:")
print(f"  Sales model: {SALES_MODEL_PATH}")
print(f"  COGS model: {COGS_MODEL_PATH}")
print(f"  Volume model: {VOLUME_MODEL_PATH}")
print(f"  Preprocessor: {PREPROCESSOR_PATH}")

# Save all algorithm-specific models
print(f"\nSaving algorithm-specific models...")

algorithm_dirs = {
    'xgboost': XGBOOST_DIR,
    'lightgbm': LIGHTGBM_DIR,
    'randomforest': RANDOMFOREST_DIR
}

for algo_name, algo_dir in algorithm_dirs.items():
    print(f"\n  Saving {algo_name.upper()} models to '{algo_dir}'...")
    
    # Save each model type for this algorithm
    sales_model_path = os.path.join(algo_dir, f'{algo_name}_sales_model.pkl')
    cogs_model_path = os.path.join(algo_dir, f'{algo_name}_cogs_model.pkl')
    volume_model_path = os.path.join(algo_dir, f'{algo_name}_volume_model.pkl')
    
    joblib.dump(trained_models[f'{algo_name}_sales'], sales_model_path)
    joblib.dump(trained_models[f'{algo_name}_cogs'], cogs_model_path)
    joblib.dump(trained_models[f'{algo_name}_volume'], volume_model_path)
    
    print(f"    Sales: {sales_model_path}")
    print(f"    COGS: {cogs_model_path}")
    print(f"    Volume: {volume_model_path}")

print(f"\nAll models saved successfully!")

# --- Summary Report ---
print(f"\n{'='*100}")
print(f"MULTI-ALGORITHM TRAINING SUMMARY")
print(f"{'='*100}")
print(f"Training Period: {train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')}")
print(f"Testing Period: {test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')}")
print(f"Training Records: {len(train_df):,}")
print(f"Testing Records: {len(test_df):,}")
print(f"")
print(f"ALGORITHMS TRAINED: XGBoost, LightGBM, Random Forest")
print(f"MODELS PER ALGORITHM: Volume, Sales Revenue & COGS Prediction")
print(f"TOTAL MODELS TRAINED: {len(trained_models)} models")
print(f"")
print(f"MAIN MODELS (XGBoost - for API/Frontend):")
print(f"  Volume Model MAE: {mae_volume:,.0f} Litres")
print(f"  Sales Model MAE: €{mae_sales:,.0f}")
print(f"  COGS Model MAE: €{mae_cogs:,.0f}")
print(f"")
print(f"All algorithm models saved in respective directories:")
print(f"  - models/xgboost/")
print(f"  - models/lightgbm/")
print(f"  - models/randomforest/")
print(f"")
print(f"Main XGBoost models ready for API deployment!")
print(f"Additional algorithms available for research/comparison!")
print(f"{'='*100}")