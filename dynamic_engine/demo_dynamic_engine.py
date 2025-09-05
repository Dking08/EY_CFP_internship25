"""
Demo: Dynamic Forecasting Engine
Demonstrates how the universal system works with any dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dynamic_forecasting_engine import DynamicForecastingEngine

def create_sample_datasets():
    """Create various sample datasets to demonstrate universality."""
    
    # Dataset 1: E-commerce Sales
    print("📊 Creating E-commerce Sales Dataset...")
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    ecommerce_data = []
    
    for date in dates:
        for product in ['Electronics', 'Clothing', 'Books']:
            for region in ['North', 'South', 'East', 'West']:
                seasonal = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
                trend = 1 + 0.02 * (date - dates[0]).days / 365
                noise = np.random.normal(1, 0.1)
                
                base_sales = {'Electronics': 5000, 'Clothing': 3000, 'Books': 1000}[product]
                sales = base_sales * seasonal * trend * noise
                
                ecommerce_data.append({
                    'date': date,
                    'product_category': product,
                    'region': region,
                    'daily_sales': max(0, sales),
                    'marketing_spend': max(0, sales * 0.1 + np.random.normal(0, 50)),
                    'customer_visits': max(0, sales * 2 + np.random.normal(0, 100)),
                    'conversion_rate': max(0.01, min(0.15, 0.05 + np.random.normal(0, 0.02))),
                    'avg_order_value': max(10, sales / (sales * 0.02) + np.random.normal(0, 5))
                })
    
    ecommerce_df = pd.DataFrame(ecommerce_data)
    ecommerce_df.to_csv('demo_ecommerce_data.csv', index=False)
    print(f"   ✅ Saved demo_ecommerce_data.csv ({len(ecommerce_df)} rows)")
    
    # Dataset 2: Restaurant Revenue
    print("📊 Creating Restaurant Revenue Dataset...")
    restaurant_data = []
    
    for date in dates:
        for restaurant_type in ['Fast Food', 'Casual Dining', 'Fine Dining']:
            for location in ['Downtown', 'Suburb', 'Mall']:
                # More complex seasonality for restaurants
                weekly_pattern = 1 + 0.4 * np.sin(2 * np.pi * date.weekday() / 7)
                holiday_boost = 1.3 if date.month == 12 and date.day in [24, 25, 31] else 1.0
                weather_factor = 1 + 0.1 * np.sin(2 * np.pi * date.dayofyear / 365)
                
                base_revenue = {'Fast Food': 2000, 'Casual Dining': 4000, 'Fine Dining': 8000}[restaurant_type]
                revenue = base_revenue * weekly_pattern * holiday_boost * weather_factor * np.random.normal(1, 0.15)
                
                restaurant_data.append({
                    'date': date,
                    'restaurant_type': restaurant_type,
                    'location': location,
                    'daily_revenue': max(0, revenue),
                    'customer_count': max(0, revenue / 45 + np.random.normal(0, 10)),
                    'avg_check_size': max(20, revenue / (revenue / 50) + np.random.normal(0, 5)),
                    'staff_cost': max(0, revenue * 0.3 + np.random.normal(0, 50)),
                    'food_cost': max(0, revenue * 0.35 + np.random.normal(0, 30)),
                    'weather_score': max(1, min(10, 5 + np.random.normal(0, 2)))
                })
    
    restaurant_df = pd.DataFrame(restaurant_data)
    restaurant_df.to_csv('demo_restaurant_data.csv', index=False)
    print(f"   ✅ Saved demo_restaurant_data.csv ({len(restaurant_df)} rows)")
    
    # Dataset 3: Stock/Crypto Prices
    print("📊 Creating Stock Price Dataset...")
    stock_data = []
    
    for date in dates:
        for symbol in ['TECH_A', 'HEALTH_B', 'FINANCE_C']:
            # Simulate stock price movement
            if not stock_data:
                price = 100  # Starting price
            else:
                last_price = [d['closing_price'] for d in stock_data if d['symbol'] == symbol][-1] if any(d['symbol'] == symbol for d in stock_data) else 100
                daily_return = np.random.normal(0.001, 0.02)  # 0.1% average daily return, 2% volatility
                price = last_price * (1 + daily_return)
            
            volume = max(1000, 50000 + np.random.normal(0, 10000))
            
            stock_data.append({
                'date': date,
                'symbol': symbol,
                'opening_price': max(1, price * (1 + np.random.normal(0, 0.005))),
                'closing_price': max(1, price),
                'high_price': max(price, price * (1 + abs(np.random.normal(0, 0.01)))),
                'low_price': max(1, min(price, price * (1 - abs(np.random.normal(0, 0.01))))),
                'volume': volume,
                'market_cap': price * 1000000,
                'pe_ratio': max(5, min(50, 15 + np.random.normal(0, 5))),
                'volatility': max(0.1, min(1.0, 0.2 + np.random.normal(0, 0.05)))
            })
    
    stock_df = pd.DataFrame(stock_data)
    stock_df.to_csv('demo_stock_data.csv', index=False)
    print(f"   ✅ Saved demo_stock_data.csv ({len(stock_df)} rows)")
    
    return ['demo_ecommerce_data.csv', 'demo_restaurant_data.csv', 'demo_stock_data.csv']

def test_dynamic_engine(csv_file):
    """Test the dynamic engine on a specific dataset."""
    
    print(f"\n🚀 TESTING DYNAMIC ENGINE ON: {csv_file}")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"📊 Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Initialize engine
    engine = DynamicForecastingEngine()
    
    # Prepare data (automatic feature detection)
    print("\n🔍 Auto-detecting features...")
    prepared_data = engine.prepare_data(df)
    
    print(f"   ✅ Detected {len(engine.numerical_features)} numerical features")
    print(f"   ✅ Detected {len(engine.categorical_features)} categorical features")
    print(f"   ✅ Detected {len(engine.target_columns)} target variables")
    print(f"   ✅ Date column: {engine.date_column}")
    
    # Train models
    print("\n🎯 Training models...")
    results = engine.train_models(prepared_data['data'], test_size=0.2)
    
    # Show results
    print("\n📈 TRAINING RESULTS:")
    for target, models in results.items():
        print(f"\n   Target: {target}")
        best_model = max(models.keys(), key=lambda x: models[x]['metrics']['r2'])
        best_metrics = models[best_model]['metrics']
        print(f"      🏆 Best Model: {best_model}")
        print(f"      📊 R² Score: {best_metrics['r2']:.3f}")
        print(f"      📊 MAE: {best_metrics['mae']:.2f}")
        print(f"      📊 MAPE: {best_metrics['mape']:.1f}%")
    
    # Generate sample predictions
    print("\n🔮 Generating sample forecasts...")
    
    # Create future data (30 days)
    sample_future = prepared_data['data'].tail(30).copy()
    predictions = engine.predict(sample_future)
    
    print(f"   ✅ Generated predictions for {len(predictions)} records")
    
    # Show prediction summary
    pred_columns = [col for col in predictions.columns if col.startswith('predicted_')]
    for pred_col in pred_columns:
        avg_pred = predictions[pred_col].mean()
        print(f"      {pred_col}: Average = {avg_pred:.2f}")
    
    # Generate report
    print("\n📋 FINAL REPORT:")
    print(engine.generate_report())
    
    return engine

def main():
    """Main demonstration function."""
    
    print("🚀 DYNAMIC FORECASTING ENGINE DEMONSTRATION")
    print("=" * 70)
    print()
    print("This demo shows how the universal forecasting engine can work")
    print("with ANY time-series dataset automatically!")
    print()
    
    # Create sample datasets
    print("📁 STEP 1: Creating Sample Datasets")
    print("-" * 40)
    datasets = create_sample_datasets()
    
    # Test each dataset
    print("\n🧪 STEP 2: Testing Dynamic Engine")
    print("-" * 40)
    
    for dataset in datasets:
        try:
            engine = test_dynamic_engine(dataset)
            print(f"✅ Successfully processed {dataset}")
        except Exception as e:
            print(f"❌ Failed to process {dataset}: {e}")
        
        print("\n" + "="*60)
    
    print("\n🎉 DEMONSTRATION COMPLETE!")
    print("\nKey Benefits of Dynamic Engine:")
    print("✅ Works with ANY CSV dataset")
    print("✅ Automatic feature detection")  
    print("✅ Multiple algorithm selection")
    print("✅ No domain-specific code needed")
    print("✅ Universal API and UI")
    
    print("\nTo use with your own data:")
    print("1. Run: python dynamic_api.py")
    print("2. Run: streamlit run frontend/dynamic_dashboard.py")
    print("3. Upload any CSV file and start forecasting!")

if __name__ == "__main__":
    main()
