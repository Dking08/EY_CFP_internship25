from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import date, timedelta
import joblib
import os
import random
from typing import List, Dict, Any

# Initialize FastAPI application
app = FastAPI(
    title="AlcoBev Cash Flow Forecasting API with KPIs",
    description="API for predicting sales revenue, COGS, and calculating business KPIs for European markets.",
    version="1.0.0"
)

# --- Model Loading Paths ---
MODELS_DIR = 'models'
SALES_MODEL_PATH = os.path.join(MODELS_DIR, 'sales_forecasting_model.pkl')
COGS_MODEL_PATH = os.path.join(MODELS_DIR, 'cogs_forecasting_model.pkl')
VOLUME_MODEL_PATH = os.path.join(MODELS_DIR, 'volume_forecasting_model.pkl')

# Model pipelines
sales_pipeline = None
cogs_pipeline = None
volume_pipeline = None

# Define the exact order of input features expected by the models
INPUT_FEATURES_ORDER = [
    'Net_Sales_Volume_Litres', 'Marketing_Spend_EUR', 'Promotional_Event',
    'Consumer_Confidence_Index', 'Inflation_Rate_EUR', 'Avg_Temp_C',
    'Holiday_Indicator', 'Competitor_Activity_Index',
    'day_of_week', 'month', 'year', 'day_of_year', 'week_of_year',
    'Net_Sales_Revenue_EUR_lag1', 'COGS_EUR_lag1',
    'Country', 'Channel', 'Product_Category'
]

VOLUME_FEATURES = [
    'Marketing_Spend_EUR', 'Promotional_Event',
    'Consumer_Confidence_Index', 'Inflation_Rate_EUR', 'Avg_Temp_C',
    'Holiday_Indicator', 'Competitor_Activity_Index',
    'day_of_week', 'month', 'year', 'day_of_year', 'week_of_year',
    'Net_Sales_Volume_Litres_lag1',
    'Country', 'Channel', 'Product_Category'
]

# --- Enhanced KPI Calculation Functions (Same KPIs, Better Infrastructure) ---
def calculate_business_kpis(
    sales_revenue: float,
    cogs: float,
    volume_litres: float,
    marketing: float
) -> Dict[str, float]:
    """
    Calculate business KPIs based on predicted/actual values with improved validation.

    Args:
        sales_revenue: Net sales revenue in EUR
        cogs: Cost of goods sold in EUR
        volume_litres: Sales volume in litres
        marketing: Marketing spend in EUR

    Returns:
        Dictionary containing calculated KPIs with validation
    """

    # Input validation and sanitization
    sales_revenue = max(0, float(sales_revenue))
    cogs = max(0, float(cogs))
    volume_litres = max(0, float(volume_litres))
    marketing = max(0, float(marketing))

    # Core KPI Calculations
    # 1. Gross Profit (EUR)
    gross_profit = sales_revenue - cogs

    # 2. Gross Profit Margin (%)
    gross_profit_margin = (gross_profit / sales_revenue * 100) if sales_revenue > 0 else 0

    # 3. Average Selling Price per Litre (EUR)
    asp_per_litre = (sales_revenue / volume_litres) if volume_litres > 0 else 0

    # 4. Operating Cash Flow (EUR) - after marketing spend
    operating_cash_flow = sales_revenue - cogs - marketing

    # 5. COGS per Litre (EUR)
    cogs_per_litre = (cogs / volume_litres) if volume_litres > 0 else 0

    # 6. Marketing Spend Ratio (%)
    marketing_spend_ratio = (marketing / sales_revenue * 100) if sales_revenue > 0 else 0

    # Additional validation - ensure realistic ranges
    gross_profit_margin = max(-100, min(100, gross_profit_margin))  # Cap at realistic range

    return {
        'gross_profit_eur': round(gross_profit, 2),
        'gross_profit_margin_pct': round(gross_profit_margin, 2),
        'asp_per_litre_eur': round(asp_per_litre, 2),
        'predicted_operating_cash_flow_eur': round(operating_cash_flow, 2),
        'cogs_per_litre_eur': round(cogs_per_litre, 2),
        'marketing_spend_ratio': round(marketing_spend_ratio, 2)
    }

def calculate_additional_metrics(
    sales_revenue: float,
    cogs: float,
    volume_litres: float
) -> Dict[str, float]:
    """
    Calculate additional business metrics for enhanced analysis.
    """
    # Input validation
    sales_revenue = max(0, float(sales_revenue))
    cogs = max(0, float(cogs))
    volume_litres = max(0, float(volume_litres))

    return {
        'cogs_ratio_pct': round((cogs / sales_revenue * 100) if sales_revenue > 0 else 0, 2)
    }

# Load models on startup
@app.on_event("startup")
async def load_assets():
    global sales_pipeline, cogs_pipeline, volume_pipeline
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

        if os.path.exists(VOLUME_MODEL_PATH):
            volume_pipeline = joblib.load(VOLUME_MODEL_PATH)
            print(f"Successfully loaded volume pipeline from {VOLUME_MODEL_PATH}")
            print(f"Volume pipeline expects {volume_pipeline.named_steps['preprocessor'].n_features_in_} features")
        else:
            print(f"Volume model not found at {VOLUME_MODEL_PATH}. Using dummy predictions.")

    except Exception as e:
        print(f"Error loading assets: {e}. Using dummy predictions.")
        sales_pipeline = None
        cogs_pipeline = None
        volume_pipeline = None

# --- Pydantic Models (Fixed Naming Consistency) ---
class ForecastRequest(BaseModel):
    """Request payload for forecasting."""
    country: str = "Germany"
    channel: str = "Off-Trade"
    product_category: str = "Beer"
    start_date: date = date.today() + timedelta(days=1)
    end_date: date = date.today() + timedelta(days=30)

class DailyForecastResponse(BaseModel):
    """Daily forecast response with KPIs - fixed naming consistency."""
    forecast_date: date
    predicted_sales_revenue_eur: float
    predicted_cogs_eur: float
    predicted_sales_volume_litres: float
    marketing_spend_eur: float

    # Core KPIs
    gross_profit_eur: float
    gross_profit_margin_pct: float
    asp_per_litre_eur: float
    predicted_operating_cash_flow_eur: float
    cogs_per_litre: float
    marketing_spend_ratio: float

class ForecastSummary(BaseModel):
    """Summary statistics for the forecast period."""
    period_start: date
    period_end: date
    total_days: int

    # Totals
    total_sales_revenue_eur: float
    total_cogs_eur: float
    total_volume_litres: float
    total_marketing_spend_eur: float
    total_gross_profit_eur: float
    total_operating_cash_flow_eur: float

    # Averages
    avg_daily_sales_revenue_eur: float
    avg_daily_gross_profit_eur: float
    avg_gross_profit_margin_pct: float
    avg_asp_per_litre_eur: float
    avg_cogs_per_litre: float
    avg_marketing_spend_ratio: float

    # Performance metrics
    best_day_sales: Dict[str, Any]
    worst_day_sales: Dict[str, Any]
    highest_margin_day: Dict[str, Any]

class ComprehensiveForecastResponse(BaseModel):
    """Complete forecast response with daily data and summary."""
    request_params: ForecastRequest
    daily_forecasts: List[DailyForecastResponse]
    summary: ForecastSummary

# --- Helper Functions (Improved Error Handling) ---
def generate_mock_future_data(request: ForecastRequest) -> pd.DataFrame:
    """Generate mock future data for forecasting with enhanced validation."""
    try:
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

            # Volume estimation based on product category and seasonality
            base_volume = 10000
            if request.product_category == "Beer":
                base_volume *= 1.3
            elif request.product_category == "Wine":
                base_volume *= 1.0
            elif request.product_category == "Spirits":
                base_volume *= 0.7
            elif request.product_category == "RTD":
                base_volume *= 1.1

            # Add seasonality for volume
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * d.day_of_year / 365)
            mock_sales_volume_litres = base_volume * seasonal_factor + random.uniform(-1000, 1000)

            # Lagged features - use fixed placeholders for POC
            mock_net_sales_revenue_lag1 = 300000
            mock_cogs_lag1 = 120000
            mock_sales_volume_litres_lag1 = mock_sales_volume_litres * 0.95 + random.uniform(-500, 500)

            data.append({
                'Date': d,
                'Country': request.country,
                'Channel': request.channel,
                'Product_Category': request.product_category,
                'Net_Sales_Volume_Litres': max(0, mock_sales_volume_litres),
                'Marketing_Spend_EUR': max(0, mock_marketing_spend),
                'Promotional_Event': 1 if d.day % 7 == 0 or d.day % 15 == 0 else 0,
                'Consumer_Confidence_Index': mock_cci,
                'Inflation_Rate_EUR': mock_inflation,
                'Avg_Temp_C': mock_temp,
                'Holiday_Indicator': mock_holiday,
                'Competitor_Activity_Index': mock_comp_act,
                'Net_Sales_Revenue_EUR_lag1': mock_net_sales_revenue_lag1,
                'COGS_EUR_lag1': mock_cogs_lag1,
                'Net_Sales_Volume_Litres_lag1': mock_sales_volume_litres_lag1
            })

        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    except Exception as e:
        raise ValueError(f"Failed to generate mock data: {str(e)}")

def prepare_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for prediction by the pipeline with validation."""
    try:
        # Create time-based features
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['year'] = df['Date'].dt.year
        df['day_of_year'] = df['Date'].dt.dayofyear
        df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)

        # Select features in correct order
        feature_df = df[INPUT_FEATURES_ORDER].copy()
        if len(feature_df.columns) != 18:
            raise ValueError(f"Feature mismatch: expected 18 features, got {len(feature_df.columns)}")

        return feature_df

    except Exception as e:
        raise ValueError(f"Feature preparation failed: {str(e)}")

def prepare_volume_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for volume prediction by the volume pipeline with validation."""
    try:
        # Create time-based features
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['year'] = df['Date'].dt.year
        df['day_of_year'] = df['Date'].dt.dayofyear
        df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)

        # Select features in correct order
        feature_df = df[VOLUME_FEATURES].copy()
        if len(feature_df.columns) != 16:
            raise ValueError(f"Volume feature mismatch: expected 16 features, got {len(feature_df.columns)}")

        return feature_df

    except Exception as e:
        raise ValueError(f"Volume feature preparation failed: {str(e)}")

def create_forecast_summary(daily_forecasts: List[DailyForecastResponse],
                          request: ForecastRequest) -> ForecastSummary:
    """Create comprehensive summary statistics with fixed field references."""
    if not daily_forecasts:
        raise ValueError("No daily forecasts to summarize")

    try:
        # Calculate totals
        total_sales = sum(f.predicted_sales_revenue_eur for f in daily_forecasts)
        total_cogs = sum(f.predicted_cogs_eur for f in daily_forecasts)
        total_volume = sum(f.predicted_sales_volume_litres for f in daily_forecasts)
        total_marketing = sum(f.marketing_spend_eur for f in daily_forecasts)
        total_gross_profit = sum(f.gross_profit_eur for f in daily_forecasts)
        total_operating_cash_flow = sum(f.predicted_operating_cash_flow_eur for f in daily_forecasts)

        # Calculate averages
        num_days = len(daily_forecasts)
        avg_daily_sales = total_sales / num_days
        avg_daily_gross_profit = total_gross_profit / num_days
        avg_gross_profit_margin = (total_gross_profit / total_sales * 100) if total_sales > 0 else 0
        avg_asp = (total_sales / total_volume) if total_volume > 0 else 0
        avg_cogs_per_litre = (total_cogs / total_volume) if total_volume > 0 else 0
        avg_marketing_spend_ratio = (total_marketing / total_sales * 100) if total_sales > 0 else 0

        # Find best/worst performing days
        best_day = max(daily_forecasts, key=lambda x: x.predicted_sales_revenue_eur)
        worst_day = min(daily_forecasts, key=lambda x: x.predicted_sales_revenue_eur)
        highest_margin_day = max(daily_forecasts, key=lambda x: x.gross_profit_margin_pct)

        return ForecastSummary(
            period_start=request.start_date,
            period_end=request.end_date,
            total_days=num_days,

            total_sales_revenue_eur=round(total_sales, 2),
            total_cogs_eur=round(total_cogs, 2),
            total_volume_litres=round(total_volume, 2),
            total_marketing_spend_eur=round(total_marketing, 2),
            total_gross_profit_eur=round(total_gross_profit, 2),
            total_operating_cash_flow_eur=round(total_operating_cash_flow, 2),

            avg_daily_sales_revenue_eur=round(avg_daily_sales, 2),
            avg_daily_gross_profit_eur=round(avg_daily_gross_profit, 2),
            avg_gross_profit_margin_pct=round(avg_gross_profit_margin, 2),
            avg_asp_per_litre_eur=round(avg_asp, 2),
            avg_cogs_per_litre=round(avg_cogs_per_litre, 2),
            avg_marketing_spend_ratio=round(avg_marketing_spend_ratio, 2),

            best_day_sales={
                "date": best_day.forecast_date,
                "sales_revenue_eur": best_day.predicted_sales_revenue_eur,
                "gross_profit_eur": best_day.gross_profit_eur,
                "gross_profit_margin_pct": best_day.gross_profit_margin_pct,
                "predicted_operating_cash_flow_eur": best_day.predicted_operating_cash_flow_eur,
                "asp_per_litre_eur": best_day.asp_per_litre_eur,
                "cogs_per_litre": best_day.cogs_per_litre,
                "marketing_spend_ratio": best_day.marketing_spend_ratio
            },
            worst_day_sales={
                "date": worst_day.forecast_date,
                "sales_revenue_eur": worst_day.predicted_sales_revenue_eur,
                "gross_profit_eur": worst_day.gross_profit_eur,
                "gross_profit_margin_pct": worst_day.gross_profit_margin_pct,
                "predicted_operating_cash_flow_eur": worst_day.predicted_operating_cash_flow_eur,
                "asp_per_litre_eur": worst_day.asp_per_litre_eur,
                "cogs_per_litre": worst_day.cogs_per_litre,
                "marketing_spend_ratio": worst_day.marketing_spend_ratio
            },
            highest_margin_day={
                "date": highest_margin_day.forecast_date,
                "gross_profit_margin_pct": highest_margin_day.gross_profit_margin_pct,
                "sales_revenue_eur": highest_margin_day.predicted_sales_revenue_eur,
                "predicted_operating_cash_flow_eur": highest_margin_day.predicted_operating_cash_flow_eur,
                "asp_per_litre_eur": highest_margin_day.asp_per_litre_eur,
                "cogs_per_litre": highest_margin_day.cogs_per_litre,
                "marketing_spend_ratio": highest_margin_day.marketing_spend_ratio
            }
        )

    except Exception as e:
        raise ValueError(f"Failed to create forecast summary: {str(e)}")

# --- API Endpoints (Enhanced Infrastructure, Same KPIs) ---
@app.get("/")
async def read_root():
    """Root endpoint for API health check."""
    return {
        "message": "Welcome to the AlcoBev Cash Flow Forecasting API with KPIs!",
        "version": "1.0.0",
        "features": [
            "Sales Revenue Prediction",
            "COGS Prediction",
            "Volume Prediction",
            "Gross Profit Calculation",
            "Gross Profit Margin Analysis",
            "Average Selling Price per Litre",
            "Operating Cash Flow Forecasting",
            "Comprehensive Business Analytics"
        ],
        "infrastructure": "Enhanced with improved validation and consistent naming",
        "docs": "Visit /docs for interactive API documentation"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with enhanced model status."""
    return {
        "status": "healthy",
        "models_loaded": {
            "sales_pipeline": sales_pipeline is not None,
            "cogs_pipeline": cogs_pipeline is not None,
            "volume_pipeline": volume_pipeline is not None
        },
        "kpi_engine": "Enhanced KPI calculations with validation",
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.post("/forecast", response_model=List[DailyForecastResponse])
async def get_daily_forecasts(request: ForecastRequest):
    """
    Generate daily cash flow forecasts with KPIs for a specified period.
    Enhanced with improved error handling and consistent KPI calculations.
    """
    try:
        # Generate future data with validation
        future_df_raw = generate_mock_future_data(request)
        print(f"[DEBUG] Generated {len(future_df_raw)} rows of future data")

        predicted_sales = []
        predicted_cogs = []
        predicted_volume = []

        if sales_pipeline and cogs_pipeline and volume_pipeline:
            # Use trained pipelines with enhanced error handling
            try:
                feature_df = prepare_features_for_prediction(future_df_raw.copy())
                feature_df_volume = prepare_volume_features_for_prediction(future_df_raw.copy())

                predicted_sales = sales_pipeline.predict(feature_df).tolist()
                predicted_cogs = cogs_pipeline.predict(feature_df).tolist()
                predicted_volume = volume_pipeline.predict(feature_df_volume).tolist()
                print("[DEBUG] Successfully used trained pipelines for prediction")

            except Exception as e:
                print(f"Error during pipeline prediction: {e}")
                raise HTTPException(status_code=500, detail=f"Pipeline prediction failed: {e}")
        else:
            # Enhanced dummy prediction logic
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
                sales_val = (base_sales + 
                           (row['day_of_week'] * 1000) +
                           (row['month'] * 5000) +
                           (row['Promotional_Event'] * 50000) +
                           random.uniform(-10000, 10000))
                cogs_val = sales_val * 0.4 + random.uniform(-5000, 5000)

                predicted_sales.append(max(0, sales_val))
                predicted_cogs.append(max(0, cogs_val))
                predicted_volume.append(max(0, row['Net_Sales_Volume_Litres']))

        # Build daily forecast responses with enhanced KPI calculations
        daily_forecasts = []
        for i, row in future_df_raw.iterrows():
            sales = predicted_sales[i]
            cogs = predicted_cogs[i]
            volume = predicted_volume[i]
            marketing = row['Marketing_Spend_EUR']

            # Use enhanced KPI calculation with validation
            kpis = calculate_business_kpis(sales, cogs, volume, marketing)

            daily_forecasts.append(DailyForecastResponse(
                forecast_date=row['Date'].date(),
                predicted_sales_revenue_eur=round(max(0, sales), 2),
                predicted_cogs_eur=round(max(0, cogs), 2),
                predicted_sales_volume_litres=round(max(0, volume), 2),
                marketing_spend_eur=round(max(0, marketing), 2),
                # KPIs:
                gross_profit_eur=kpis['gross_profit_eur'],
                gross_profit_margin_pct=kpis['gross_profit_margin_pct'],
                asp_per_litre_eur=kpis['asp_per_litre_eur'],
                predicted_operating_cash_flow_eur = kpis['predicted_operating_cash_flow_eur'],
                cogs_per_litre=kpis['cogs_per_litre_eur'],
                marketing_spend_ratio=kpis['marketing_spend_ratio']
            ))

        return daily_forecasts

    except Exception as e:
        print(f"[ERROR] Daily forecast generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

@app.post("/forecast/comprehensive", response_model=ComprehensiveForecastResponse)
async def get_comprehensive_forecast(request: ForecastRequest):
    """
    Generate comprehensive forecast with daily data and summary analytics.
    Enhanced infrastructure with improved error handling and validation.
    """
    try:
        # Get daily forecasts with enhanced infrastructure
        daily_forecasts = await get_daily_forecasts(request)

        # Create summary with fixed field references
        summary = create_forecast_summary(daily_forecasts, request)

        return ComprehensiveForecastResponse(
            request_params=request,
            daily_forecasts=daily_forecasts,
            summary=summary
        )

    except Exception as e:
        print(f"[ERROR] Comprehensive forecast generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comprehensive forecast failed: {str(e)}")

@app.get("/kpi/calculate")
async def calculate_kpi_standalone(
    sales_revenue: float,
    cogs: float,
    volume_litres: float,
    marketing: float = 0.0
):
    """
    Calculate KPIs for given input values with enhanced validation.
    Same KPIs as before but with improved infrastructure.
    """
    try:
        # Enhanced input validation
        if sales_revenue < 0 or cogs < 0 or volume_litres < 0 or marketing < 0:
            raise HTTPException(status_code=400, detail="All input values must be non-negative")

        # Use enhanced KPI calculation
        kpis = calculate_business_kpis(sales_revenue, cogs, volume_litres, marketing)
        additional_metrics = calculate_additional_metrics(sales_revenue, cogs, volume_litres)

        return {
            "input": {
                "sales_revenue_eur": sales_revenue,
                "cogs_eur": cogs,
                "volume_litres": volume_litres,
                "marketing_spend_eur": marketing
            },
            "calculated_kpis": kpis,
            "additional_metrics": additional_metrics,
            "validation_status": "All inputs validated and calculations completed successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KPI calculation failed: {str(e)}")

@app.get("/analytics/summary")
async def get_analytics_summary():
    """
    Get summary of available analytics and KPIs with enhanced documentation.
    """
    return {
        "available_kpis": {
            "gross_profit_eur": {
                "description": "Revenue minus Cost of Goods Sold",
                "formula": "Net Sales Revenue - COGS",
                "unit": "EUR",
                "validation": "Always calculated, no negative revenue assumptions"
            },
            "gross_profit_margin_pct": {
                "description": "Gross profit as percentage of revenue",
                "formula": "(Gross Profit / Net Sales Revenue) x 100",
                "unit": "%",
                "validation": "Capped between -100% and 100% for realistic business scenarios"
            },
            "asp_per_litre_eur": {
                "description": "Average selling price per litre",
                "formula": "Net Sales Revenue / Net Sales Volume",
                "unit": "EUR per litre",
                "validation": "Only calculated when volume > 0"
            },
            "predicted_operating_cash_flow_eur": {
                "description": "Cash flow from operations after marketing spend",
                "formula": "Net Sales Revenue - COGS - Marketing Spend",
                "unit": "EUR",
                "validation": "Can be negative indicating cash outflow periods"
            },
            "cost_per_litre_eur": {
                "description": "Cost of goods sold per litre",
                "formula": "COGS / Net Sales Volume",
                "unit": "EUR per litre"
            },
            "marketing_spend_ratio": {
                "description": "Marketing spend as percentage of revenue",
                "formula": "(Marketing Spend / Net Sales Revenue) x 100",
                "unit": "%",
                "validation": "Only calculated when revenue > 0"
            }
        },
        "additional_metrics": {
            "cogs_ratio_pct": {
                "description": "Cost of goods sold as percentage of revenue",
                "formula": "(COGS / Net Sales Revenue) x 100",
                "unit": "%"
            }
        },
        "forecast_capabilities": [
            "Daily sales revenue prediction",
            "Daily COGS prediction",
            "Daily volume prediction",
            "Automated KPI calculation with validation",
            "Operating cash flow analysis",
            "Period summary statistics",
            "Best/worst day identification"
        ],
        "supported_segments": {
            "countries": ["Germany", "France", "Italy", "Spain", "UK"],
            "channels": ["Off-Trade", "On-Trade"],
            "product_categories": ["Beer", "Wine", "Spirits", "RTD"]
        }
    }

# Instructions to run this improved infrastructure API:
# 1. Save this code as `app_improved_infrastructure.py`
# 2. Ensure you have a `models` directory with trained model files
# 3. Install required packages: `pip install fastapi uvicorn pandas scikit-learn joblib pydantic numpy xgboost`
# 4. Run: `uvicorn app_improved_infrastructure:app --reload`
# 5. Visit: `http://127.0.0.1:8000/docs` for interactive documentation