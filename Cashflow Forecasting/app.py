from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import date, timedelta
import joblib
import os
import random
from typing import List, Dict, Any, Tuple

# Initialize FastAPI application
app = FastAPI(
    title="AlcoBev Cash Flow Forecasting API with KPIs",
    description="API for predicting sales revenue, COGS, and calculating business KPIs for European markets.",
    version="1.5.0"
)

# --- Constants (Extracted to reduce repetition) ---
MODELS_DIR = 'models'
MODEL_PATHS = {
    'sales': os.path.join(MODELS_DIR, 'sales_forecasting_model.pkl'),
    'cogs': os.path.join(MODELS_DIR, 'cogs_forecasting_model.pkl'),
    'volume': os.path.join(MODELS_DIR, 'volume_forecasting_model.pkl')
}

# Product category multipliers
PRODUCT_CATEGORY_MULTIPLIERS = {
    "Beer": 1.2,
    "Wine": 1.0,
    "Spirits": 1.5,
    "RTD": 0.8
}

VOLUME_BASE_MULTIPLIERS = {
    "Beer": 1.3,
    "Wine": 1.0,
    "Spirits": 0.7,
    "RTD": 1.1
}

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

# --- Utility Functions ---
def validate_and_sanitize_inputs(*values: float) -> List[float]:
    """Validate and sanitize numeric inputs - common pattern extracted."""
    return [max(0, float(value)) for value in values]

def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Safe division with fallback - extracted common pattern."""
    return (numerator / denominator) if denominator > 0 else fallback

def cap_percentage(value: float, min_val: float = -100, max_val: float = 100) -> float:
    """Cap percentage values within realistic range - extracted common pattern."""
    return max(min_val, min(max_val, value))

def load_single_model(model_path: str, model_name: str) -> Tuple[object, bool]:
    """Load a single model with error handling - extracted pattern."""
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"Successfully loaded {model_name} pipeline from {model_path}")
            print(f"{model_name} pipeline expects {model.named_steps['preprocessor'].n_features_in_} features")
            return model, True
        else:
            print(f"{model_name} model not found at {model_path}. Using dummy predictions.")
            return None, False
    except Exception as e:
        print(f"Error loading {model_name} model: {e}")
        return None, False

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features - extracted common pattern."""
    df = df.copy()
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    return df

def get_promotional_event_indicator(day: int) -> int:
    """Get promotional event indicator - extracted common pattern."""
    return 1 if day % 7 == 0 or day % 15 == 0 else 0

def calculate_seasonal_factor(day_of_year: int) -> float:
    """Calculate seasonal factor - extracted common pattern."""
    return 1 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)

# --- Enhanced KPI Calculation Functions (Reduced repetition) ---
def calculate_business_kpis(
    sales_revenue: float,
    cogs: float,
    volume_litres: float,
    marketing: float
) -> Dict[str, float]:
    """
    Calculate business KPIs with improved validation and reduced repetition.
    """
    # Validate and sanitize inputs using utility function
    sales_revenue, cogs, volume_litres, marketing = validate_and_sanitize_inputs(
        sales_revenue, cogs, volume_litres, marketing
    )

    # Core calculations
    gross_profit = sales_revenue - cogs

    # Use safe division utility to reduce repetition
    gross_profit_margin = cap_percentage(safe_divide(gross_profit * 100, sales_revenue))
    asp_per_litre = safe_divide(sales_revenue, volume_litres)
    cogs_per_litre = safe_divide(cogs, volume_litres)
    marketing_spend_ratio = safe_divide(marketing * 100, sales_revenue)
    revenue_per_euro_marketing = safe_divide(sales_revenue, marketing)
    litres_per_euro_marketing = safe_divide(volume_litres, marketing)

    # Operating cash flow
    operating_cash_flow = sales_revenue - cogs - marketing

    # Return with consistent rounding
    return {
        'gross_profit_eur': round(gross_profit, 2),
        'gross_profit_margin_pct': round(gross_profit_margin, 2),
        'asp_per_litre_eur': round(asp_per_litre, 2),
        'predicted_operating_cash_flow_eur': round(operating_cash_flow, 2),
        'cogs_per_litre_eur': round(cogs_per_litre, 2),
        'marketing_spend_ratio': round(marketing_spend_ratio, 2),
        'revenue_per_euro_marketing': round(revenue_per_euro_marketing, 2),
        'litres_per_euro_marketing': round(litres_per_euro_marketing, 2)
    }

def calculate_additional_metrics(
    sales_revenue: float,
    cogs: float,
    volume_litres: float
) -> Dict[str, float]:
    """Calculate additional metrics with utility functions."""
    sales_revenue, cogs, volume_litres = validate_and_sanitize_inputs(sales_revenue, cogs, volume_litres)

    return {
        'cogs_ratio_pct': round(safe_divide(cogs * 100, sales_revenue), 2)
    }

def aggregate_kpi_totals(daily_forecasts: List) -> Dict[str, float]:
    """Aggregate KPI totals from daily forecasts - extracted common pattern."""
    return {
        'total_sales': sum(f.predicted_sales_revenue_eur for f in daily_forecasts),
        'total_cogs': sum(f.predicted_cogs_eur for f in daily_forecasts),
        'total_volume': sum(f.predicted_sales_volume_litres for f in daily_forecasts),
        'total_marketing': sum(f.marketing_spend_eur for f in daily_forecasts),
        'total_gross_profit': sum(f.gross_profit_eur for f in daily_forecasts),
        'total_operating_cash_flow': sum(f.predicted_operating_cash_flow_eur for f in daily_forecasts)
    }

def calculate_summary_averages(totals: Dict[str, float], num_days: int) -> Dict[str, float]:
    """Calculate summary averages - extracted common pattern."""
    return {
        'avg_daily_sales': totals['total_sales'] / num_days,
        'avg_daily_gross_profit': totals['total_gross_profit'] / num_days,
        'avg_gross_profit_margin': safe_divide(totals['total_gross_profit'] * 100, totals['total_sales']),
        'avg_asp': safe_divide(totals['total_sales'], totals['total_volume']),
        'avg_cogs_per_litre': safe_divide(totals['total_cogs'], totals['total_volume']),
        'avg_marketing_spend_ratio': safe_divide(totals['total_marketing'] * 100, totals['total_sales']),
        'avg_revenue_per_euro_marketing': safe_divide(totals['total_sales'], totals['total_marketing']),
        'avg_litres_per_euro_marketing': safe_divide(totals['total_volume'], totals['total_marketing'])
    }

def find_performance_days(daily_forecasts: List) -> Dict[str, Any]:
    """Find best/worst performing days - extracted common pattern."""
    best_day = max(daily_forecasts, key=lambda x: x.predicted_sales_revenue_eur)
    worst_day = min(daily_forecasts, key=lambda x: x.predicted_sales_revenue_eur)
    highest_margin_day = max(daily_forecasts, key=lambda x: x.gross_profit_margin_pct)

    def create_day_summary(day_forecast):
        """Create day summary - extracted common pattern."""
        return {
            "date": day_forecast.forecast_date,
            "sales_revenue_eur": day_forecast.predicted_sales_revenue_eur,
            "gross_profit_eur": day_forecast.gross_profit_eur,
            "gross_profit_margin_pct": day_forecast.gross_profit_margin_pct,
            "predicted_operating_cash_flow_eur": day_forecast.predicted_operating_cash_flow_eur,
            "asp_per_litre_eur": day_forecast.asp_per_litre_eur,
            "cogs_per_litre": day_forecast.cogs_per_litre,
            "marketing_spend_ratio": day_forecast.marketing_spend_ratio,
            "revenue_per_euro_marketing": day_forecast.revenue_per_euro_marketing,
            "litres_per_euro_marketing": day_forecast.litres_per_euro_marketing
        }

    return {
        'best_day_sales': create_day_summary(best_day),
        'worst_day_sales': create_day_summary(worst_day),
        'highest_margin_day': create_day_summary(highest_margin_day)
    }

# Load models on startup (refactored to reduce repetition)
@app.on_event("startup")
async def load_assets():
    global sales_pipeline, cogs_pipeline, volume_pipeline

    try:
        # Load all models using utility function
        sales_pipeline, _ = load_single_model(MODEL_PATHS['sales'], 'sales')
        cogs_pipeline, _ = load_single_model(MODEL_PATHS['cogs'], 'COGS')
        volume_pipeline, _ = load_single_model(MODEL_PATHS['volume'], 'volume')

    except Exception as e:
        print(f"Error during model loading: {e}. Using dummy predictions.")
        sales_pipeline = None
        cogs_pipeline = None
        volume_pipeline = None

# --- Pydantic Models (Same as before) ---
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
    revenue_per_euro_marketing: float
    litres_per_euro_marketing: float

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
    avg_revenue_per_euro_marketing: float
    avg_litres_per_euro_marketing: float

    # Performance metrics
    best_day_sales: Dict[str, Any]
    worst_day_sales: Dict[str, Any]
    highest_margin_day: Dict[str, Any]

class ComprehensiveForecastResponse(BaseModel):
    """Complete forecast response with daily data and summary."""
    request_params: ForecastRequest
    daily_forecasts: List[DailyForecastResponse]
    summary: ForecastSummary

# --- Helper Functions ---
def generate_mock_external_data(d: date) -> Dict[str, float]:
    """Generate mock external data for a single date - extracted common pattern."""
    return {
        'cci': 100 + (d.day % 10) * 0.5 + random.uniform(-2, 2),
        'inflation': 2.0 + (d.month % 3) * 0.1 + random.uniform(-0.5, 0.5),
        'temp': 10 + (d.day % 15) + random.uniform(-3, 3),
        'holiday': 1 if d.month == 12 and d.day == 25 else 0,
        'comp_act': 0.8 + (d.day % 7) * 0.01 + random.uniform(-0.1, 0.1)
    }

def calculate_marketing_spend(d: date) -> float:
    """Calculate marketing spend for a date - extracted common pattern."""
    base_spend = 5000 + (d.day % 5) * 100 + random.uniform(-500, 500)
    if get_promotional_event_indicator(d.day):
        base_spend *= 1.5
    return max(0, base_spend)

def calculate_base_volume(product_category: str) -> float:
    """Calculate base volume for product category - extracted common pattern."""
    return 10000 * VOLUME_BASE_MULTIPLIERS.get(product_category, 1.0)

def generate_mock_future_data(request: ForecastRequest) -> pd.DataFrame:
    """Generate mock future data with reduced repetition."""
    try:
        dates = pd.date_range(start=request.start_date, end=request.end_date, freq='D')
        data = []

        # Pre-calculate base values to avoid repetition
        base_volume = calculate_base_volume(request.product_category)

        for d in dates:
            # Use utility functions to reduce repetition
            external_data = generate_mock_external_data(d)
            marketing_spend = calculate_marketing_spend(d)

            # Calculate volume with seasonality
            seasonal_factor = calculate_seasonal_factor(d.day_of_year)
            mock_sales_volume_litres = base_volume * seasonal_factor + random.uniform(-1000, 1000)

            # Lagged features (same as before)
            mock_net_sales_revenue_lag1 = 300000
            mock_cogs_lag1 = 120000
            mock_sales_volume_litres_lag1 = mock_sales_volume_litres * 0.95 + random.uniform(-500, 500)

            data.append({
                'Date': d,
                'Country': request.country,
                'Channel': request.channel,
                'Product_Category': request.product_category,
                'Net_Sales_Volume_Litres': max(0, mock_sales_volume_litres),
                'Marketing_Spend_EUR': marketing_spend,
                'Promotional_Event': get_promotional_event_indicator(d.day),
                'Consumer_Confidence_Index': external_data['cci'],
                'Inflation_Rate_EUR': external_data['inflation'],
                'Avg_Temp_C': external_data['temp'],
                'Holiday_Indicator': external_data['holiday'],
                'Competitor_Activity_Index': external_data['comp_act'],
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
    """Prepare features for prediction with utility functions."""
    try:
        df = create_time_features(df)
        feature_df = df[INPUT_FEATURES_ORDER].copy()

        if len(feature_df.columns) != 18:
            raise ValueError(f"Feature mismatch: expected 18 features, got {len(feature_df.columns)}")

        return feature_df

    except Exception as e:
        raise ValueError(f"Feature preparation failed: {str(e)}")

def prepare_volume_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare volume features with utility functions."""
    try:
        df = create_time_features(df)
        feature_df = df[VOLUME_FEATURES].copy()

        if len(feature_df.columns) != 16:
            raise ValueError(f"Volume feature mismatch: expected 16 features, got {len(feature_df.columns)}")

        return feature_df

    except Exception as e:
        raise ValueError(f"Volume feature preparation failed: {str(e)}")

def create_forecast_summary(daily_forecasts: List[DailyForecastResponse],
                          request: ForecastRequest) -> ForecastSummary:
    """Create comprehensive summary with utility functions."""
    if not daily_forecasts:
        raise ValueError("No daily forecasts to summarize")

    try:
        num_days = len(daily_forecasts)

        # Use utility functions to reduce repetition
        totals = aggregate_kpi_totals(daily_forecasts)
        averages = calculate_summary_averages(totals, num_days)
        performance_days = find_performance_days(daily_forecasts)

        return ForecastSummary(
            period_start=request.start_date,
            period_end=request.end_date,
            total_days=num_days,

            # Totals (with consistent rounding)
            total_sales_revenue_eur=round(totals['total_sales'], 2),
            total_cogs_eur=round(totals['total_cogs'], 2),
            total_volume_litres=round(totals['total_volume'], 2),
            total_marketing_spend_eur=round(totals['total_marketing'], 2),
            total_gross_profit_eur=round(totals['total_gross_profit'], 2),
            total_operating_cash_flow_eur=round(totals['total_operating_cash_flow'], 2),

            # Averages (with consistent rounding)
            avg_daily_sales_revenue_eur=round(averages['avg_daily_sales'], 2),
            avg_daily_gross_profit_eur=round(averages['avg_daily_gross_profit'], 2),
            avg_gross_profit_margin_pct=round(averages['avg_gross_profit_margin'], 2),
            avg_asp_per_litre_eur=round(averages['avg_asp'], 2),
            avg_cogs_per_litre=round(averages['avg_cogs_per_litre'], 2),
            avg_marketing_spend_ratio=round(averages['avg_marketing_spend_ratio'], 2),
            avg_revenue_per_euro_marketing=round(averages['avg_revenue_per_euro_marketing'], 2),
            avg_litres_per_euro_marketing=round(averages['avg_litres_per_euro_marketing'], 2),

            # Performance metrics
            best_day_sales=performance_days['best_day_sales'],
            worst_day_sales=performance_days['worst_day_sales'],
            highest_margin_day=performance_days['highest_margin_day']
        )

    except Exception as e:
        raise ValueError(f"Failed to create forecast summary: {str(e)}")

def generate_dummy_predictions(future_df_raw: pd.DataFrame, request: ForecastRequest) -> Tuple[List[float], List[float], List[float]]:
    """Generate dummy predictions with reduced repetition."""
    predicted_sales = []
    predicted_cogs = []
    predicted_volume = []

    # Get multiplier once
    base_sales_multiplier = PRODUCT_CATEGORY_MULTIPLIERS.get(request.product_category, 1.0)
    base_sales = 250000 * base_sales_multiplier

    for i, row in future_df_raw.iterrows():
        # Calculate sales with consistent pattern
        sales_val = (base_sales + 
                   (row['day_of_week'] * 1000) +
                   (row['month'] * 5000) +
                   (row['Promotional_Event'] * 50000) +
                   random.uniform(-10000, 10000))

        cogs_val = sales_val * 0.4 + random.uniform(-5000, 5000)

        predicted_sales.append(max(0, sales_val))
        predicted_cogs.append(max(0, cogs_val))
        predicted_volume.append(max(0, row['Net_Sales_Volume_Litres']))

    return predicted_sales, predicted_cogs, predicted_volume

# --- API Endpoints (Using refactored functions) ---
@app.get("/")
async def read_root():
    """Root endpoint for API health check."""
    return {
        "message": "Welcome to the AlcoBev Cash Flow Forecasting API with KPIs!",
        "version": "1.5.0",
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
        "infrastructure": "Refactored with reduced repetition and improved maintainability",
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
    Generate daily cash flow forecasts with KPIs using refactored functions.
    """
    try:
        # Generate future data using refactored function
        future_df_raw = generate_mock_future_data(request)
        print(f"[DEBUG] Generated {len(future_df_raw)} rows of future data")

        if sales_pipeline and cogs_pipeline and volume_pipeline:
            # Use trained pipelines
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
            # Use refactored dummy prediction function
            print("[DEBUG] Using dummy predictions as pipelines are not loaded")
            predicted_sales, predicted_cogs, predicted_volume = generate_dummy_predictions(future_df_raw, request)

        # Build daily forecast responses using refactored KPI calculation
        daily_forecasts = []
        for i, row in future_df_raw.iterrows():
            sales = predicted_sales[i]
            cogs = predicted_cogs[i]
            volume = predicted_volume[i]
            marketing = row['Marketing_Spend_EUR']

            # Use refactored KPI calculation
            kpis = calculate_business_kpis(sales, cogs, volume, marketing)

            daily_forecasts.append(DailyForecastResponse(
                forecast_date=row['Date'].date(),
                predicted_sales_revenue_eur=round(max(0, sales), 2),
                predicted_cogs_eur=round(max(0, cogs), 2),
                predicted_sales_volume_litres=round(max(0, volume), 2),
                marketing_spend_eur=round(max(0, marketing), 2),

                # KPIs from refactored function
                gross_profit_eur=kpis['gross_profit_eur'],
                gross_profit_margin_pct=kpis['gross_profit_margin_pct'],
                asp_per_litre_eur=kpis['asp_per_litre_eur'],
                predicted_operating_cash_flow_eur=kpis['predicted_operating_cash_flow_eur'],
                cogs_per_litre=kpis['cogs_per_litre_eur'],
                marketing_spend_ratio=kpis['marketing_spend_ratio'],
                revenue_per_euro_marketing=kpis['revenue_per_euro_marketing'],
                litres_per_euro_marketing=kpis['litres_per_euro_marketing']
            ))

        return daily_forecasts

    except Exception as e:
        print(f"[ERROR] Daily forecast generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

@app.post("/forecast/comprehensive", response_model=ComprehensiveForecastResponse)
async def get_comprehensive_forecast(request: ForecastRequest):
    """
    Generate comprehensive forecast using refactored functions.
    """
    try:
        # Get daily forecasts using refactored endpoint
        daily_forecasts = await get_daily_forecasts(request)

        # Create summary using refactored function
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
    Calculate KPIs using refactored functions.
    """
    try:
        # Enhanced input validation using utility function
        if any(value < 0 for value in [sales_revenue, cogs, volume_litres, marketing]):
            raise HTTPException(status_code=400, detail="All input values must be non-negative")

        # Use refactored KPI calculation functions
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
    Get summary of available analytics and KPIs.
    """
    # KPI descriptions organized to reduce repetition
    base_kpis = {
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
        }
    }

    extended_kpis = {
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
        },
        "revenue_per_euro_marketing": {
            "description": "Revenue generated per euro spent on marketing",
            "formula": "Net Sales Revenue / Marketing Spend",
            "unit": "EUR per EUR",
            "validation": "Only calculated when marketing spend > 0"
        },
        "litres_per_euro_marketing": {
            "description": "Litres sold per euro spent on marketing",
            "formula": "Net Sales Volume / Marketing Spend",
            "unit": "Litres per EUR",
            "validation": "Only calculated when marketing spend > 0"
        }
    }

    return {
        "available_kpis": {**base_kpis, **extended_kpis},
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

# Instructions to run this refactored API:
# 1. Save this code as `app_refactored.py`
# 2. Ensure you have a `models` directory with trained model files
# 3. Install required packages: `pip install fastapi uvicorn pandas scikit-learn joblib pydantic numpy xgboost`
# 4. Run: `uvicorn app_refactored:app --reload`
# 5. Visit: `http://127.0.0.1:8000/docs` for interactive documentation