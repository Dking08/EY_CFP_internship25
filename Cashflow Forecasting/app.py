# app.py - FIXED VERSION
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import date, timedelta
import joblib
import os
import random

# Initialize FastAPI application
app = FastAPI(
    title="AlcoBev Cash Flow Forecasting API",
    description="API for predicting sales revenue and COGS for European markets.",
    version="0.1.0"
)

# --- Model Loading Paths ---
MODELS_DIR = 'models'
SALES_MODEL_PATH = os.path.join(MODELS_DIR, 'sales_forecasting_model.pkl')
COGS_MODEL_PATH = os.path.join(MODELS_DIR, 'cogs_forecasting_model.pkl')

# These will hold the complete pipelines (preprocessor + model)
sales_pipeline = None
cogs_pipeline = None

# Define the exact order of input features expected by the models
# This must match the all_features_for_preprocessor list in train_models.py
INPUT_FEATURES_ORDER = [
    'Net_Sales_Volume_Litres', 'Marketing_Spend_EUR', 'Promotional_Event',
    'Consumer_Confidence_Index', 'Inflation_Rate_EUR', 'Avg_Temp_C',
    'Holiday_Indicator', 'Competitor_Activity_Index',
    'day_of_week', 'month', 'year', 'day_of_year', 'week_of_year',
    'Net_Sales_Revenue_EUR_lag1', 'COGS_EUR_lag1',
    'Country', 'Channel', 'Product_Category'
]

# Attempt to load complete model pipelines on application startup
@app.on_event("startup")
async def load_assets():
    global sales_pipeline, cogs_pipeline
    try:
        if os.path.exists(SALES_MODEL_PATH):
            sales_pipeline = joblib.load(SALES_MODEL_PATH)
            print(f"Successfully loaded sales pipeline from {SALES_MODEL_PATH}")
            print(f"Sales pipeline expects {sales_pipeline.named_steps['preprocessor'].n_features_in_} features")
        else:
            print(f"Sales model not found at {SALES_MODEL_PATH}. Using dummy predictions.")

        if os.path.exists(COGS_MODEL_PATH):
            cogs_pipeline = joblib.load(COGS_MODEL_PATH)
            print(f"Successfully loaded COGS pipeline from {COGS_MODEL_PATH}")
            print(f"COGS pipeline expects {cogs_pipeline.named_steps['preprocessor'].n_features_in_} features")
        else:
            print(f"COGS model not found at {COGS_MODEL_PATH}. Using dummy predictions.")

    except Exception as e:
        print(f"Error loading assets: {e}. Using dummy predictions.")
        sales_pipeline = None
        cogs_pipeline = None

# --- Pydantic Models for API Request and Response ---
class ForecastRequest(BaseModel):
    """
    Defines the structure for the forecast request payload.
    """
    country: str = "Germany"
    channel: str = "Off-Trade"
    product_category: str = "Beer"
    start_date: date = date.today() + timedelta(days=1)
    end_date: date = date.today() + timedelta(days=30)

class ForecastResponse(BaseModel):
    """
    Defines the structure for each daily forecast entry in the API response.
    """
    forecast_date: date
    predicted_sales_revenue_eur: float
    predicted_cogs_eur: float
    predicted_operating_cash_flow_eur: float

# --- Helper Functions for Data Generation and Feature Engineering ---
def generate_mock_future_data(request: ForecastRequest) -> pd.DataFrame:
    """
    Generates a DataFrame with future dates and mock values for external regressors
    based on the forecast request.
    """
    dates = pd.date_range(start=request.start_date, end=request.end_date, freq='D')
    data = []
    
    for d in dates:
        # Dummy external data for POC
        mock_cci = 100 + (d.day % 10) * 0.5 + random.uniform(-2, 2)
        mock_inflation = 2.0 + (d.month % 3) * 0.1 + random.uniform(-0.5, 0.5)
        mock_temp = 10 + (d.day % 15) + random.uniform(-3, 3)
        mock_holiday = 1 if d.month == 12 and d.day == 25 else 0
        mock_comp_act = 0.8 + (d.day % 7) * 0.01 + random.uniform(-0.1, 0.1)

        mock_marketing_spend = 5000 + (d.day % 5) * 100 + random.uniform(-500, 500)
        if d.day % 7 == 0 or d.day % 15 == 0:
             mock_marketing_spend *= 1.5

        # Lagged features - use fixed placeholders for POC
        mock_sales_volume_litres = 10000
        mock_net_sales_revenue_lag1 = 300000
        mock_cogs_lag1 = 120000

        data.append({
            'Date': d,
            'Country': request.country,
            'Channel': request.channel,
            'Product_Category': request.product_category,
            'Net_Sales_Volume_Litres': mock_sales_volume_litres,
            'Marketing_Spend_EUR': mock_marketing_spend,
            'Promotional_Event': 1 if d.day % 7 == 0 or d.day % 15 == 0 else 0,
            'Consumer_Confidence_Index': mock_cci,
            'Inflation_Rate_EUR': mock_inflation,
            'Avg_Temp_C': mock_temp,
            'Holiday_Indicator': mock_holiday,
            'Competitor_Activity_Index': mock_comp_act,
            'Net_Sales_Revenue_EUR_lag1': mock_net_sales_revenue_lag1,
            'COGS_EUR_lag1': mock_cogs_lag1
        })
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIXED: Prepares features for prediction by the complete pipeline.
    The pipeline will handle preprocessing internally.
    """
    # Create time-based features (same as in training)
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)

    # Select only the features that the model expects, in the correct order
    feature_df = df[INPUT_FEATURES_ORDER].copy()
    
    print(f"\n[DEBUG] Feature DataFrame shape: {feature_df.shape}")
    print(f"[DEBUG] Feature DataFrame columns: {feature_df.columns.tolist()}")
    print(f"[DEBUG] Expected {len(INPUT_FEATURES_ORDER)} features, got {len(feature_df.columns)}")
    
    # Verify we have exactly 18 features
    if len(feature_df.columns) != 18:
        raise ValueError(f"Feature mismatch: expected 18 features, got {len(feature_df.columns)}")
    
    return feature_df

# --- API Endpoints ---
@app.get("/")
async def read_root():
    """
    Root endpoint for basic API health check.
    """
    return {"message": "Welcome to the AlcoBev Cash Flow Forecasting API! Visit /docs for API documentation."}

@app.post("/forecast", response_model=list[ForecastResponse])
async def get_cash_flow_forecast(request: ForecastRequest):
    """
    Generates a cash flow forecast for a specified period and segment.
    """
    try:
        # Generate raw future data based on request parameters
        future_df_raw = generate_mock_future_data(request)
        print(f"[DEBUG] Generated {len(future_df_raw)} rows of future data")

        predicted_sales = []
        predicted_cogs = []

        if sales_pipeline and cogs_pipeline:
            # Prepare features for the pipeline (no separate preprocessing needed)
            feature_df = prepare_features_for_prediction(future_df_raw.copy())
            
            # Use the complete pipelines for prediction (they handle preprocessing internally)
            try:
                predicted_sales = sales_pipeline.predict(feature_df).tolist()
                predicted_cogs = cogs_pipeline.predict(feature_df).tolist()
                print("[DEBUG] Successfully used trained pipelines for prediction")
            except Exception as e:
                print(f"Error during pipeline prediction: {e}")
                raise HTTPException(status_code=500, detail=f"Pipeline prediction failed: {e}")
        else:
            # Dummy prediction logic if pipelines are not loaded
            print("[DEBUG] Using dummy predictions as pipelines are not loaded")
            for i, row in future_df_raw.iterrows():
                base_sales = 250000
                if request.product_category == "Beer":
                    base_sales *= 1.2
                elif request.product_category == "Wine":
                    base_sales *= 1.0
                elif request.product_category == "Spirits":
                    base_sales *= 1.5
                elif request.product_category == "RTD":
                    base_sales *= 0.8

                # Add seasonality and randomness
                sales_val = base_sales + (row['day_of_week'] * 1000) + (row['month'] * 5000) + (row['Promotional_Event'] * 50000) + random.uniform(-10000, 10000)
                cogs_val = sales_val * 0.4 + random.uniform(-5000, 5000)

                predicted_sales.append(sales_val)
                predicted_cogs.append(cogs_val)

        # Build response
        response_list = []
        for i, row in future_df_raw.iterrows():
            sales = predicted_sales[i]
            cogs = predicted_cogs[i]
            operating_cash_flow = sales - cogs - row['Marketing_Spend_EUR']

            response_list.append(ForecastResponse(
                forecast_date=row['Date'].date(),
                predicted_sales_revenue_eur=round(max(0, sales), 2),
                predicted_cogs_eur=round(max(0, cogs), 2),
                predicted_operating_cash_flow_eur=round(operating_cash_flow, 2)
            ))

        return response_list
        
    except Exception as e:
        print(f"[ERROR] Forecast generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

# Instructions to run this API:
# 1. Save this code as `app.py`.
# 2. Ensure you have a `models` directory with the trained model files.
# 3. Install libraries: `pip install fastapi uvicorn pandas scikit-learn joblib pydantic xgboost`
# 4. Run: `uvicorn app:app --reload`
# 5. Visit: `http://127.0.0.1:8000/docs`