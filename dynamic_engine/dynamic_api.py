"""
Dynamic Forecasting API - Universal ML API for Any Dataset
Transforms fixed AlcoBev API into a generic forecasting service
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import joblib
import os
import io
from typing import List, Dict, Any, Optional
import uuid
from dynamic_forecasting_engine import DynamicForecastingEngine

# Initialize FastAPI application
app = FastAPI(
    title="Dynamic Forecasting Engine API",
    description="Universal ML API that can learn and forecast from any time-series dataset",
    version="2.0.0"
)

# --- Global Storage ---
# In production, use Redis/Database instead of in-memory storage
active_engines = {}  # Store trained engines by session_id
MODELS_DIR = 'dynamic_models'
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Pydantic Models ---
class DatasetInfo(BaseModel):
    """Information about uploaded dataset."""
    session_id: str
    total_rows: int
    total_columns: int
    numerical_features: List[str]
    categorical_features: List[str]
    potential_targets: List[str]
    date_column: Optional[str]
    sample_data: List[Dict[str, Any]]

class TrainingRequest(BaseModel):
    """Request to train models on uploaded dataset."""
    session_id: str
    target_columns: List[str]
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)

class TrainingResponse(BaseModel):
    """Response after training models."""
    session_id: str
    targets_trained: List[str]
    model_performance: Dict[str, Dict[str, float]]
    training_summary: str

class PredictionRequest(BaseModel):
    """Request for predictions on new data."""
    session_id: str
    prediction_data: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    """Response with predictions."""
    session_id: str
    predictions: List[Dict[str, Any]]
    prediction_summary: Dict[str, Any]

class ForecastRequest(BaseModel):
    """Request for future forecasting."""
    session_id: str
    forecast_days: int = Field(default=30, ge=1, le=365)
    base_values: Optional[Dict[str, Any]] = None

# --- Helper Functions ---
def generate_session_id() -> str:
    """Generate unique session ID."""
    return str(uuid.uuid4())[:8]

def safe_convert_to_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Safely convert columns to numeric, handling errors gracefully."""
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    return df_copy

def create_future_data_template(engine: DynamicForecastingEngine, 
                              forecast_days: int, 
                              base_values: Dict[str, float] = None) -> pd.DataFrame:
    """Create template for future forecasting."""
    
    if not base_values:
        base_values = {}
    
    # Generate future dates
    start_date = datetime.now().date() + timedelta(days=1)
    future_dates = pd.date_range(start=start_date, periods=forecast_days, freq='D')
    
    future_data = []
    
    for date in future_dates:
        row = {}
        
        # Add date
        if engine.date_column:
            row[engine.date_column] = date
        
        # Add categorical features with default values
        for cat_feature in engine.categorical_features:
            if cat_feature in base_values:
                row[cat_feature] = base_values[cat_feature]
            else:
                row[cat_feature] = 'default'  # Will be handled by model
        
        # Add numerical features with base values or defaults
        for num_feature in engine.numerical_features:
            if 'lag' in num_feature.lower():
                # For lag features, use a reasonable default based on the feature name
                if any(target in num_feature.lower() for target in engine.target_columns):
                    # Use average of target if it's a lag of target
                    row[num_feature] = 1000.0  # Default reasonable value
                else:
                    row[num_feature] = 100.0  # Default for other lag features
            elif num_feature in base_values:
                row[num_feature] = base_values[num_feature]
            elif any(time_feature in num_feature for time_feature in ['year', 'month', 'day', 'week']):
                # Time features will be calculated automatically
                continue
            elif 'temp' in num_feature.lower():
                row[num_feature] = 20.0  # Default temperature
            elif 'spend' in num_feature.lower() or 'cost' in num_feature.lower():
                row[num_feature] = 1000.0  # Default spend
            elif 'rate' in num_feature.lower() or 'index' in num_feature.lower():
                row[num_feature] = 1.0  # Default rate/index
            else:
                row[num_feature] = 0.0  # Default to 0
        
        future_data.append(row)
    
    df = pd.DataFrame(future_data)
    
    # Create time features if date column exists
    if engine.date_column and engine.date_column in df.columns:
        df, _ = engine.create_time_features(df, engine.date_column)
    
    return df

# --- API Endpoints ---
@app.get("/")
async def read_root():
    """Root endpoint with API information."""
    return {
        "message": "🚀 Welcome to the Dynamic Forecasting Engine API!",
        "version": "2.0.0",
        "features": [
            "Upload any time-series dataset",
            "Automatic feature detection", 
            "Multi-target forecasting",
            "Advanced ML algorithms",
            "Future period forecasting",
            "Model performance analytics"
        ],
        "workflow": {
            "1": "POST /upload - Upload your dataset",
            "2": "POST /train - Train forecasting models", 
            "3": "POST /predict - Make predictions",
            "4": "POST /forecast - Generate future forecasts"
        },
        "docs": "Visit /docs for interactive API documentation"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_sessions": len(active_engines),
        "models_directory": MODELS_DIR,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/upload", response_model=DatasetInfo)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV dataset and get automatic feature analysis.
    
    The system will automatically detect:
    - Numerical and categorical features
    - Date/time columns
    - Potential target variables
    - Data quality issues
    """
    try:
        # Generate session ID
        session_id = generate_session_id()
        
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        print(f"📁 Uploaded dataset: {file.filename}")
        print(f"   Session ID: {session_id}")
        print(f"   Shape: {df.shape}")
        
        # Initialize dynamic engine
        engine = DynamicForecastingEngine()
        
        # Auto-detect features
        feature_info = engine.auto_detect_features(df)
        
        # Store engine for this session
        active_engines[session_id] = {
            'engine': engine,
            'raw_data': df,
            'feature_info': feature_info,
            'upload_time': datetime.now()
        }
        
        # Create response
        response = DatasetInfo(
            session_id=session_id,
            total_rows=len(df),
            total_columns=len(df.columns),
            numerical_features=feature_info['numerical'],
            categorical_features=feature_info['categorical'],
            potential_targets=feature_info['potential_targets'],
            date_column=feature_info['date'],
            sample_data=df.head(3).to_dict('records')
        )
        
        print(f"✅ Dataset analysis complete for session {session_id}")
        return response
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process dataset: {str(e)}")

@app.post("/train", response_model=TrainingResponse)
async def train_models(request: TrainingRequest):
    """
    Train forecasting models on the uploaded dataset.
    
    Automatically selects best algorithms and provides performance metrics.
    """
    try:
        if request.session_id not in active_engines:
            raise HTTPException(status_code=404, detail="Session not found. Please upload dataset first.")
        
        session_data = active_engines[request.session_id]
        engine = session_data['engine']
        df = session_data['raw_data']
        
        print(f"🎯 Training models for session {request.session_id}")
        print(f"   Targets: {request.target_columns}")
        
        # Set target columns
        engine.target_columns = request.target_columns
        
        # Prepare data
        prepared_data = engine.prepare_data(df, request.target_columns)
        
        # Train models
        training_results = engine.train_models(prepared_data['data'], request.test_size)
        
        # Save models
        session_model_dir = os.path.join(MODELS_DIR, request.session_id)
        engine.save_models(session_model_dir)
        
        # Update session data
        active_engines[request.session_id]['prepared_data'] = prepared_data['data']
        active_engines[request.session_id]['training_results'] = training_results
        
        # Prepare performance summary (convert to JSON-serializable format)
        performance_summary = {}
        for target, models in training_results.items():
            best_model = max(models.keys(), key=lambda x: models[x]['metrics']['r2'])
            performance_summary[target] = {
                'r2': float(models[best_model]['metrics']['r2']),
                'mae': float(models[best_model]['metrics']['mae']),
                'rmse': float(models[best_model]['metrics']['rmse']),
                'mape': float(models[best_model]['metrics']['mape'])
            }
        
        # Generate training summary
        summary_lines = [
            f"✅ Training completed successfully!",
            f"📊 Targets trained: {len(request.target_columns)}",
            f"🔧 Features used: {engine.data_schema['total_features']}",
            f"📈 Best models selected automatically",
        ]
        
        response = TrainingResponse(
            session_id=request.session_id,
            targets_trained=request.target_columns,
            model_performance=performance_summary,
            training_summary="\\n".join(summary_lines)
        )
        
        print(f"✅ Training complete for session {request.session_id}")
        return response
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def make_predictions(request: PredictionRequest):
    """
    Make predictions on new data using trained models.
    """
    try:
        if request.session_id not in active_engines:
            raise HTTPException(status_code=404, detail="Session not found.")
        
        session_data = active_engines[request.session_id]
        engine = session_data['engine']
        
        if not engine.pipelines:
            raise HTTPException(status_code=400, detail="No trained models found. Train models first.")
        
        # Convert prediction data to DataFrame
        new_data = pd.DataFrame(request.prediction_data)
        
        # Make predictions
        predictions_df = engine.predict(new_data)
        
        # Calculate summary statistics
        prediction_cols = [col for col in predictions_df.columns if col.startswith('predicted_')]
        summary = {}
        
        for col in prediction_cols:
            summary[col] = {
                'mean': float(predictions_df[col].mean()),
                'min': float(predictions_df[col].min()),
                'max': float(predictions_df[col].max()),
                'std': float(predictions_df[col].std())
            }
        
        response = PredictionResponse(
            session_id=request.session_id,
            predictions=predictions_df.to_dict('records'),
            prediction_summary=summary
        )
        
        print(f"✅ Predictions generated for session {request.session_id}")
        return response
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/forecast", response_model=PredictionResponse)
async def generate_forecast(request: ForecastRequest):
    """
    Generate future forecasts for specified number of days.
    
    Creates synthetic future data based on patterns learned from training data.
    """
    try:
        if request.session_id not in active_engines:
            raise HTTPException(status_code=404, detail="Session not found.")
        
        session_data = active_engines[request.session_id]
        engine = session_data['engine']
        
        if not engine.pipelines:
            raise HTTPException(status_code=400, detail="No trained models found. Train models first.")
        
        print(f"🔮 Generating {request.forecast_days}-day forecast for session {request.session_id}")
        
        # Create future data template
        try:
            future_data = create_future_data_template(
                engine, request.forecast_days, request.base_values
            )
            print(f"Created future data template with shape: {future_data.shape}")
            print(f"Columns: {future_data.columns.tolist()}")
        except Exception as e:
            print(f"Error creating future data template: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create future data template: {str(e)}")
        
        # Make predictions
        try:
            forecast_df = engine.predict(future_data)
            print(f"Generated predictions with shape: {forecast_df.shape}")
        except Exception as e:
            print(f"Error making predictions: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate predictions: {str(e)}")
        
        # Calculate forecast summary
        prediction_cols = [col for col in forecast_df.columns if col.startswith('predicted_')]
        summary = {
            'forecast_period': f"{request.forecast_days} days",
            'start_date': str(future_data[engine.date_column].min().date()) if engine.date_column and engine.date_column in future_data.columns else 'N/A',
            'end_date': str(future_data[engine.date_column].max().date()) if engine.date_column and engine.date_column in future_data.columns else 'N/A'
        }
        
        for col in prediction_cols:
            target_name = col.replace('predicted_', '')
            summary[f'total_{target_name}'] = float(forecast_df[col].sum())
            summary[f'avg_daily_{target_name}'] = float(forecast_df[col].mean())
            summary[f'peak_{target_name}'] = float(forecast_df[col].max())
        
        response = PredictionResponse(
            session_id=request.session_id,
            predictions=forecast_df.to_dict('records'),
            prediction_summary=summary
        )
        
        print(f"✅ Forecast generated for session {request.session_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Forecast failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

@app.get("/sessions")
async def list_active_sessions():
    """List all active sessions with their status."""
    sessions = []
    
    for session_id, session_data in active_engines.items():
        engine = session_data['engine']
        sessions.append({
            'session_id': session_id,
            'upload_time': session_data['upload_time'].isoformat(),
            'data_shape': session_data['raw_data'].shape,
            'targets_trained': len(engine.pipelines),
            'has_models': len(engine.pipelines) > 0,
            'features': {
                'numerical': len(engine.numerical_features),
                'categorical': len(engine.categorical_features)
            }
        })
    
    return {
        'total_sessions': len(sessions),
        'sessions': sessions
    }

@app.get("/session/{session_id}/info")
async def get_session_info(session_id: str):
    """Get detailed information about a specific session."""
    if session_id not in active_engines:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    session_data = active_engines[session_id]
    engine = session_data['engine']
    
    # Convert model performance to JSON-serializable format
    model_performance = {}
    if hasattr(engine, 'model_performance') and engine.model_performance:
        for target, models in engine.model_performance.items():
            model_performance[target] = {}
            for model_name, model_info in models.items():
                if isinstance(model_info, dict) and 'metrics' in model_info:
                    model_performance[target][model_name] = {
                        'r2': float(model_info['metrics']['r2']),
                        'mae': float(model_info['metrics']['mae']),
                        'rmse': float(model_info['metrics']['rmse']),
                        'mape': float(model_info['metrics']['mape'])
                    }
    
    return {
        'session_id': session_id,
        'upload_time': session_data['upload_time'].isoformat(),
        'data_info': {
            'rows': len(session_data['raw_data']),
            'columns': len(session_data['raw_data'].columns),
            'date_column': engine.date_column
        },
        'features': {
            'numerical': engine.numerical_features,
            'categorical': engine.categorical_features,
            'targets': engine.target_columns or []
        },
        'models': {
            'trained_targets': list(engine.pipelines.keys()),
            'performance': model_performance
        },
        'report': engine.generate_report() if engine.pipelines else "No models trained yet."
    }

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its associated data."""
    if session_id not in active_engines:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    # Remove from memory
    del active_engines[session_id]
    
    # Remove saved models
    session_model_dir = os.path.join(MODELS_DIR, session_id)
    if os.path.exists(session_model_dir):
        import shutil
        shutil.rmtree(session_model_dir)
    
    return {
        'message': f'Session {session_id} deleted successfully',
        'session_id': session_id
    }

@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get overall analytics about the API usage."""
    total_sessions = len(active_engines)
    total_models = sum(len(session['engine'].pipelines) for session in active_engines.values())
    
    feature_types = {}
    for session in active_engines.values():
        engine = session['engine']
        for feature in engine.numerical_features:
            feature_types[feature] = feature_types.get(feature, 0) + 1
    
    return {
        'api_info': {
            'version': '2.0.0',
            'type': 'Dynamic Forecasting Engine',
            'capabilities': [
                'Auto feature detection',
                'Multi-algorithm training',
                'Time series forecasting',
                'Universal data support'
            ]
        },
        'usage_stats': {
            'active_sessions': total_sessions,
            'total_models_trained': total_models,
            'common_features': list(feature_types.keys())[:10]
        },
        'supported_algorithms': [
            'LightGBM',
            'XGBoost', 
            'Random Forest',
            'Auto ML Pipeline'
        ]
    }

# Instructions to run this Dynamic API:
# 1. Save this code as `dynamic_api.py`
# 2. Install: `pip install fastapi uvicorn pandas scikit-learn lightgbm xgboost python-multipart`
# 3. Run: `uvicorn dynamic_api:app --reload`
# 4. Visit: `http://127.0.0.1:8000/docs`
# 5. Upload ANY CSV dataset and start forecasting!

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
