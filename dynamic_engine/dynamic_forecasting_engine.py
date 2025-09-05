"""
Dynamic Forecasting Engine - Generic ML Pipeline for Any Dataset
Transforms the AlcoBev-specific system into a universal forecasting tool
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DynamicForecastingEngine:
    """
    A generic forecasting engine that can adapt to any time-series dataset
    with automatic feature detection, preprocessing, and model selection.
    """
    
    def __init__(self, target_columns: List[str] = None):
        """
        Initialize the dynamic forecasting engine.
        
        Args:
            target_columns: List of column names to predict. If None, will auto-detect.
        """
        self.target_columns = target_columns
        self.numerical_features = []
        self.categorical_features = []
        self.date_column = None
        self.pipelines = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.data_schema = {}
        
    def auto_detect_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Automatically detect feature types and date columns in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with detected feature types
        """
        feature_info = {
            'numerical': [],
            'categorical': [],
            'date': None,
            'potential_targets': []
        }
        
        for col in df.columns:
            # Detect date columns
            if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
                feature_info['date'] = col
                continue
                
            # Try to convert to datetime
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].head(100))
                    feature_info['date'] = col
                    continue
                except:
                    pass
            
            # Detect numerical features
            if df[col].dtype in ['int64', 'float64']:
                feature_info['numerical'].append(col)
                
                # Potential targets are usually revenue, sales, profit, volume etc.
                if any(keyword in col.lower() for keyword in 
                      ['revenue', 'sales', 'profit', 'volume', 'amount', 'value', 'cost']):
                    feature_info['potential_targets'].append(col)
                    
            # Detect categorical features
            elif df[col].dtype == 'object' or df[col].nunique() < 20:
                feature_info['categorical'].append(col)
        
        return feature_info
    
    def create_time_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Create time-based features from date column.
        
        Args:
            df: DataFrame with date column
            date_col: Name of the date column
            
        Returns:
            DataFrame with additional time features
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Basic time features
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['day_of_year'] = df[date_col].dt.dayofyear
        df['week_of_year'] = df[date_col].dt.isocalendar().week.astype(int)
        df['quarter'] = df[date_col].dt.quarter
        
        # Cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Add these to numerical features
        time_features = ['year', 'month', 'day', 'day_of_week', 'day_of_year', 
                        'week_of_year', 'quarter', 'month_sin', 'month_cos', 
                        'day_sin', 'day_cos']
        
        return df, time_features
    
    def create_lag_features(self, df: pd.DataFrame, target_cols: List[str], 
                          group_cols: List[str] = None, lags: List[int] = [1, 7, 30]) -> pd.DataFrame:
        """
        Create lagged features for time series forecasting.
        
        Args:
            df: Input DataFrame
            target_cols: Target columns to create lags for
            group_cols: Columns to group by when creating lags
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        if group_cols:
            df = df.sort_values(group_cols + [self.date_column])
        else:
            df = df.sort_values(self.date_column)
        
        lag_features = []
        for target in target_cols:
            for lag in lags:
                lag_col = f"{target}_lag{lag}"
                if group_cols:
                    df[lag_col] = df.groupby(group_cols)[target].shift(lag)
                else:
                    df[lag_col] = df[target].shift(lag)
                lag_features.append(lag_col)
        
        return df, lag_features
    
    def prepare_data(self, df: pd.DataFrame, target_columns: List[str] = None) -> Dict[str, Any]:
        """
        Automatically prepare data for modeling.
        
        Args:
            df: Input DataFrame
            target_columns: Target columns to predict
            
        Returns:
            Dictionary with prepared data and metadata
        """
        # Auto-detect features
        feature_info = self.auto_detect_features(df)
        
        self.date_column = feature_info['date']
        self.numerical_features = feature_info['numerical'].copy()
        self.categorical_features = feature_info['categorical'].copy()
        
        # Set target columns
        if target_columns:
            self.target_columns = target_columns
        elif self.target_columns is None:
            self.target_columns = feature_info['potential_targets'][:3]  # Take top 3
        
        # Remove targets from features
        self.numerical_features = [col for col in self.numerical_features 
                                 if col not in self.target_columns]
        
        print(f"📊 Auto-detected features:")
        print(f"   Date column: {self.date_column}")
        print(f"   Numerical features: {len(self.numerical_features)}")
        print(f"   Categorical features: {len(self.categorical_features)}")
        print(f"   Target columns: {self.target_columns}")
        
        # Create time features if date column exists
        if self.date_column:
            df, time_features = self.create_time_features(df, self.date_column)
            self.numerical_features.extend(time_features)
            
            # Create lag features for targets
            df, lag_features = self.create_lag_features(
                df, self.target_columns, self.categorical_features
            )
            self.numerical_features.extend(lag_features)
        
        # Remove rows with NaN (from lag features)
        df = df.dropna()
        
        # Store data schema
        self.data_schema = {
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'target_columns': self.target_columns,
            'date_column': self.date_column,
            'total_features': len(self.numerical_features) + len(self.categorical_features)
        }
        
        return {
            'data': df,
            'schema': self.data_schema,
            'feature_info': feature_info
        }
    
    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """
        Create a preprocessing pipeline based on detected features.
        
        Returns:
            Fitted preprocessing pipeline
        """
        transformers = []
        
        if self.numerical_features:
            transformers.append(('num', StandardScaler(), self.numerical_features))
        
        if self.categorical_features:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), 
                               self.categorical_features))
        
        return ColumnTransformer(transformers=transformers, remainder='drop')
    
    def train_models(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train multiple models for each target variable.
        
        Args:
            df: Prepared DataFrame
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training results
        """
        # Prepare features and targets
        all_features = self.numerical_features + self.categorical_features
        X = df[all_features]
        
        # Create preprocessor
        preprocessor = self.create_preprocessing_pipeline()
        
        # Models to try
        models = {
            'lightgbm': LGBMRegressor(random_state=42, n_estimators=100, verbose=-1),
            'xgboost': XGBRegressor(random_state=42, n_estimators=100, verbosity=0),
            'random_forest': RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=-1)
        }
        
        results = {}
        
        for target in self.target_columns:
            print(f"\n🎯 Training models for target: {target}")
            y = df[target]
            
            # Split data
            if self.date_column:
                # Time-based split for time series
                split_idx = int(len(df) * (1 - test_size))
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            else:
                # Random split for non-time series
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            
            target_results = {}
            
            for model_name, model in models.items():
                print(f"   Training {model_name}...")
                
                # Create pipeline
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', model)
                ])
                
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Evaluate
                y_pred = pipeline.predict(X_test)
                
                metrics = {
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'r2': r2_score(y_test, y_pred),
                    'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                }
                
                target_results[model_name] = {
                    'pipeline': pipeline,
                    'metrics': metrics,
                    'predictions': y_pred,
                    'actual': y_test
                }
                
                print(f"      MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}")
            
            # Select best model based on R² score
            best_model = max(target_results.keys(), 
                           key=lambda x: target_results[x]['metrics']['r2'])
            
            print(f"   🏆 Best model for {target}: {best_model}")
            
            self.pipelines[target] = target_results[best_model]['pipeline']
            self.model_performance[target] = target_results
            
            results[target] = target_results
        
        return results
    
    def save_models(self, model_dir: str = 'dynamic_models') -> None:
        """
        Save trained models and metadata.
        
        Args:
            model_dir: Directory to save models
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save individual model pipelines
        for target, pipeline in self.pipelines.items():
            model_path = os.path.join(model_dir, f'{target}_model.pkl')
            joblib.dump(pipeline, model_path)
            print(f"✅ Saved {target} model to {model_path}")
        
        # Save data schema and metadata
        metadata = {
            'data_schema': self.data_schema,
            'model_performance': {
                target: {model: info['metrics'] for model, info in models.items()}
                for target, models in self.model_performance.items()
            }
        }
        
        metadata_path = os.path.join(model_dir, 'metadata.pkl')
        joblib.dump(metadata, metadata_path)
        print(f"✅ Saved metadata to {metadata_path}")
    
    def load_models(self, model_dir: str = 'dynamic_models') -> None:
        """
        Load trained models and metadata.
        
        Args:
            model_dir: Directory containing saved models
        """
        # Load metadata
        metadata_path = os.path.join(model_dir, 'metadata.pkl')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.data_schema = metadata['data_schema']
            self.model_performance = metadata['model_performance']
            self.target_columns = self.data_schema['target_columns']
            self.numerical_features = self.data_schema['numerical_features']
            self.categorical_features = self.data_schema['categorical_features']
            print(f"✅ Loaded metadata from {metadata_path}")
        
        # Load model pipelines
        for target in self.target_columns:
            model_path = os.path.join(model_dir, f'{target}_model.pkl')
            if os.path.exists(model_path):
                self.pipelines[target] = joblib.load(model_path)
                print(f"✅ Loaded {target} model from {model_path}")
    
    def predict(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Args:
            new_data: New data to predict on
            
        Returns:
            DataFrame with predictions
        """
        if not self.pipelines:
            raise ValueError("No models loaded. Train models first or load from disk.")
        
        # Make a copy to avoid modifying original data
        data_copy = new_data.copy()
        
        # Prepare features (assuming same preprocessing as training)
        if self.date_column and self.date_column in data_copy.columns:
            data_copy, _ = self.create_time_features(data_copy, self.date_column)
        
        # Ensure all required features are present
        all_features = self.numerical_features + self.categorical_features
        
        # Add missing features with default values
        for feature in all_features:
            if feature not in data_copy.columns:
                if feature in self.categorical_features:
                    data_copy[feature] = 'default'
                else:
                    # For numerical features, use reasonable defaults
                    if 'lag' in feature.lower():
                        data_copy[feature] = 1000.0  # Default for lag features
                    elif 'temp' in feature.lower():
                        data_copy[feature] = 20.0
                    elif 'rate' in feature.lower() or 'index' in feature.lower():
                        data_copy[feature] = 1.0
                    else:
                        data_copy[feature] = 0.0
        
        # Select only the required features in the correct order
        X_new = data_copy[all_features]
        
        predictions = {}
        
        for target, pipeline in self.pipelines.items():
            try:
                predictions[f'predicted_{target}'] = pipeline.predict(X_new)
            except Exception as e:
                print(f"Warning: Failed to predict {target}: {e}")
                # Provide dummy predictions if model fails
                predictions[f'predicted_{target}'] = [1000.0] * len(X_new)
        
        # Combine with original data
        result_df = new_data.copy()
        for pred_col, pred_values in predictions.items():
            result_df[pred_col] = pred_values
        
        return result_df
    
    def get_feature_importance(self, target: str) -> pd.DataFrame:
        """
        Get feature importance for a specific target.
        
        Args:
            target: Target variable name
            
        Returns:
            DataFrame with feature importance
        """
        if target not in self.pipelines:
            raise ValueError(f"No model found for target: {target}")
        
        pipeline = self.pipelines[target]
        
        # Get feature names after preprocessing
        feature_names = (pipeline.named_steps['preprocessor']
                        .get_feature_names_out())
        
        # Get importance from the regressor
        regressor = pipeline.named_steps['regressor']
        
        if hasattr(regressor, 'feature_importances_'):
            importance_values = regressor.feature_importances_
        elif hasattr(regressor, 'coef_'):
            importance_values = np.abs(regressor.coef_)
        else:
            return pd.DataFrame({'feature': feature_names, 'importance': [0] * len(feature_names)})
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive report of the forecasting engine.
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("📊 DYNAMIC FORECASTING ENGINE REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Data schema
        report.append("📋 DATA SCHEMA:")
        report.append(f"   • Total features: {self.data_schema.get('total_features', 0)}")
        report.append(f"   • Numerical features: {len(self.numerical_features)}")
        report.append(f"   • Categorical features: {len(self.categorical_features)}")
        report.append(f"   • Target variables: {len(self.target_columns)}")
        report.append(f"   • Date column: {self.date_column}")
        report.append("")
        
        # Model performance
        report.append("🎯 MODEL PERFORMANCE:")
        for target, models in self.model_performance.items():
            report.append(f"   Target: {target}")
            for model_name, info in models.items():
                metrics = info['metrics']
                report.append(f"      {model_name}: R² = {metrics['r2']:.3f}, "
                             f"MAE = {metrics['mae']:.2f}, MAPE = {metrics['mape']:.1f}%")
        report.append("")
        
        # Target columns
        report.append("🎯 TARGET VARIABLES:")
        for target in self.target_columns:
            report.append(f"   • {target}")
        report.append("")
        
        report.append("✅ Engine ready for predictions!")
        
        return "\n".join(report)


# Example usage demonstration
def demo_dynamic_engine():
    """
    Demonstrate the dynamic forecasting engine with sample data.
    """
    print("🚀 DYNAMIC FORECASTING ENGINE DEMO")
    print("=" * 50)
    
    # Create sample data (could be ANY business dataset)
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    
    sample_data = []
    for date in dates:
        sample_data.append({
            'date': date,
            'product_type': np.random.choice(['A', 'B', 'C']),
            'region': np.random.choice(['North', 'South', 'East', 'West']),
            'sales_amount': 1000 + 500 * np.sin(date.dayofyear / 365 * 2 * np.pi) + np.random.normal(0, 100),
            'profit': 300 + 150 * np.sin(date.dayofyear / 365 * 2 * np.pi) + np.random.normal(0, 30),
            'marketing_spend': 50 + 25 * np.random.rand(),
            'temperature': 20 + 15 * np.sin(date.dayofyear / 365 * 2 * np.pi) + np.random.normal(0, 5)
        })
    
    df = pd.DataFrame(sample_data)
    
    # Initialize and use the dynamic engine
    engine = DynamicForecastingEngine()
    
    # Prepare data automatically
    prepared_data = engine.prepare_data(df)
    
    # Train models
    results = engine.train_models(prepared_data['data'])
    
    # Save models
    engine.save_models('demo_models')
    
    # Generate report
    print(engine.generate_report())
    
    return engine

if __name__ == "__main__":
    demo_dynamic_engine()
