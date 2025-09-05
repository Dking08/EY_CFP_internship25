# 🚀 Dynamic Forecasting Engine - Universal ML Forecasting System

## 📋 Overview

This project transforms the AlcoBev-specific cash flow forecasting system into a **universal forecasting engine** that can work with any time-series dataset. It automatically detects features, selects optimal algorithms, and provides forecasting capabilities for any business domain.

## 🎯 Project Evolution

### ✅ **Tasks 1 & 2 Completed:**
- **Task 1 - More KPIs:** Added 13+ business KPIs including Gross Profit, Margin Analysis, ASP, etc.
- **Task 2 - Advanced Algorithms:** Implemented LightGBM, XGBoost, CatBoost, and ensemble methods

### 🚀 **Task 4 - Dynamic Forecasting Engine:**
**Problem:** The original system was hard-coded for AlcoBev data with fixed features and preprocessing.

**Solution:** Created a universal system that can:
- ✅ Accept ANY CSV dataset
- ✅ Automatically detect feature types
- ✅ Identify potential target variables  
- ✅ Create appropriate preprocessing pipelines
- ✅ Select optimal ML algorithms
- ✅ Generate forecasts for any business domain

## 🏗️ Architecture

```
📁 Dynamic Forecasting System
├── 🧠 Core Engine
│   ├── dynamic_forecasting_engine.py    # Universal ML engine
│   └── demo_dynamic_engine.py           # Demonstration script
│
├── 🌐 API Layer  
│   ├── dynamic_api.py                   # Universal FastAPI service
│   └── app.py                          # Original AlcoBev API
│
├── 🎨 Frontend
│   ├── dynamic_dashboard.py             # Universal Streamlit UI
│   └── streamlit_app.py                # Original AlcoBev UI
│
└── 📊 Original AlcoBev System
    ├── generate_data.py                 # AlcoBev data generator
    ├── train_models.py                  # AlcoBev model training
    └── requirements.txt                 # Dependencies
```

## 🔄 Comparison: Original vs Dynamic

| Feature | Original AlcoBev System | Dynamic Engine |
|---------|------------------------|----------------|
| **Data Support** | Only AlcoBev sales data | ANY CSV dataset |
| **Features** | Hard-coded 18 features | Auto-detected unlimited |
| **Targets** | Fixed: Sales, COGS, Volume | Any numerical columns |
| **Preprocessing** | AlcoBev-specific pipeline | Universal auto-pipeline |
| **Categories** | Fixed: Country, Channel, Product | Any categorical columns |
| **API** | AlcoBev endpoints only | Universal upload/train/predict |
| **UI** | AlcoBev dashboard only | Universal dataset interface |
| **Algorithms** | XGBoost only | LightGBM, XGBoost, Random Forest |
| **Use Cases** | Alcohol beverage only | E-commerce, Finance, ANY domain |

## 🚀 Quick Start

### 1. **Install Dependencies**
```bash
pip install fastapi uvicorn streamlit pandas scikit-learn lightgbm xgboost python-multipart plotly requests
```

### 2. **Start the Universal API**
```bash
python dynamic_api.py
# API runs on http://127.0.0.1:8000
```

### 3. **Start the Universal Dashboard**
```bash
streamlit run frontend/dynamic_dashboard.py
# Dashboard runs on http://localhost:8501
```

### 4. **Upload ANY Dataset**
- Go to the dashboard
- Upload any CSV with time-series data
- The system automatically analyzes and prepares your data
- Train models and generate forecasts!

## 📊 Example Use Cases

### 🛒 **E-commerce Sales**
```csv
date,product_category,region,daily_sales,marketing_spend,customer_visits
2024-01-01,Electronics,North,5000,500,1000
2024-01-02,Clothing,South,3000,300,600
```

### 🍽️ **Restaurant Revenue**
```csv
date,restaurant_type,location,daily_revenue,customer_count,staff_cost
2024-01-01,Fast Food,Downtown,2000,400,600
2024-01-02,Fine Dining,Suburb,8000,100,2400
```

### 📈 **Stock Prices**
```csv
date,symbol,closing_price,volume,market_cap
2024-01-01,AAPL,150.25,50000000,2500000000
2024-01-02,GOOGL,2800.50,30000000,1800000000
```

### 🏥 **Healthcare Metrics**
```csv
date,department,patient_visits,revenue,staff_hours
2024-01-01,Emergency,120,50000,240
2024-01-02,Cardiology,80,75000,160
```

## 🧠 How It Works

### 1. **Automatic Feature Detection**
```python
# The engine automatically detects:
feature_info = {
    'numerical': ['sales', 'revenue', 'volume', 'temperature'],
    'categorical': ['region', 'product_type', 'channel'],
    'date': 'date_column',
    'potential_targets': ['revenue', 'sales', 'profit']
}
```

### 2. **Universal Preprocessing**
- Numerical features → StandardScaler
- Categorical features → OneHotEncoder  
- Date features → Time-based feature engineering
- Lag features → Automatic creation for time-series

### 3. **Algorithm Selection**
- Tests multiple algorithms (LightGBM, XGBoost, Random Forest)
- Selects best performer based on R² score
- Handles both single and multi-target prediction

### 4. **Universal API Endpoints**

#### Upload Dataset
```python
POST /upload
# Upload any CSV file
# Returns: Feature analysis and session ID
```

#### Train Models
```python
POST /train
{
    "session_id": "abc123",
    "target_columns": ["revenue", "profit"],
    "test_size": 0.2
}
```

#### Generate Forecasts
```python
POST /forecast
{
    "session_id": "abc123", 
    "forecast_days": 30,
    "base_values": {"region": "North", "marketing_spend": 1000}
}
```

## 🎨 Universal Dashboard Features

### 📁 **Step 1: Upload Dataset**
- Drag & drop any CSV file
- Automatic feature analysis
- Data quality overview
- Sample data preview

### ⚙️ **Step 2: Configure & Train**
- Select target variables from detected options
- Configure train/test split
- Multi-algorithm training
- Performance comparison

### 🔮 **Step 3: Generate Forecasts**
- Configurable forecast period
- Set base values for features
- Interactive visualizations
- Download results

### 📊 **Step 4: Analytics & Insights**
- Model performance metrics
- Feature importance analysis
- Session management
- Technical reports

## 🌟 Key Benefits

### 🚀 **Universality**
- Works with ANY time-series CSV data
- No domain-specific coding required
- Automatic feature engineering
- Adaptive preprocessing

### 🎯 **Ease of Use** 
- Upload → Train → Forecast workflow
- Automatic best practices
- Interactive web interface
- No ML expertise required

### ⚡ **Performance**
- Multiple algorithm selection
- Advanced ensemble methods
- Optimized hyperparameters
- Parallel training

### 🔧 **Flexibility**
- Support for multiple targets
- Configurable forecast periods
- Custom base values
- Session-based workflows

## 🧪 Demo & Testing

### Run the Demo
```bash
python demo_dynamic_engine.py
```

This creates sample datasets for different domains and demonstrates the engine's universality:
- E-commerce sales data
- Restaurant revenue data  
- Stock price data

### Test with Your Data
1. Prepare any CSV with:
   - Date column (any format)
   - Numerical target columns
   - Optional categorical features
   - Optional external factors

2. Upload via the dashboard or API

3. The system handles the rest automatically!

## 📈 Business Impact

### 🎯 **Before (AlcoBev-Specific)**
- Single use case (alcohol beverage sales)
- Manual feature engineering required
- Fixed preprocessing pipeline
- Limited to specific business KPIs
- Hard-coded categorical values

### 🚀 **After (Universal Engine)**
- ✅ ANY business domain supported
- ✅ Automatic feature detection
- ✅ Universal preprocessing
- ✅ Adaptable KPI calculations  
- ✅ Dynamic categorical handling
- ✅ Plug-and-play forecasting
- ✅ No technical expertise required

## 🔮 Use Case Examples

### 📊 **Retail Chain**
Upload store sales data → Get demand forecasts by location, product, season

### 🏭 **Manufacturing**
Upload production data → Get equipment failure predictions, output forecasts

### 💰 **Financial Services**
Upload transaction data → Get revenue forecasts, risk predictions

### 🏥 **Healthcare**
Upload patient data → Get capacity planning, resource allocation forecasts

### 🎯 **Marketing**
Upload campaign data → Get ROI predictions, budget optimization

## 🚀 Future Enhancements

### 🧠 **Advanced ML**
- Deep learning models (LSTM, Transformer)
- AutoML integration  
- Hyperparameter optimization
- Model explainability

### 📊 **Advanced Analytics**
- Anomaly detection
- Trend analysis
- Seasonality decomposition
- Causal inference

### 🌐 **Platform Features**
- Multi-user support
- Data versioning
- Model registry
- Scheduled forecasting

### 🔗 **Integrations**
- Database connectors
- Cloud storage support
- BI tool integration
- Real-time streaming

## 📚 Technical Documentation

### 🧠 **Core Engine Classes**

#### `DynamicForecastingEngine`
Main class that handles:
- Automatic feature detection
- Data preprocessing
- Model training and selection
- Prediction generation

#### Key Methods:
```python
# Automatic feature detection
auto_detect_features(df) -> Dict[str, List[str]]

# Data preparation with feature engineering  
prepare_data(df, target_columns) -> Dict[str, Any]

# Multi-algorithm training
train_models(df, test_size) -> Dict[str, Any]

# Universal prediction
predict(new_data) -> pd.DataFrame
```

### 🌐 **API Endpoints**

#### Core Workflow:
1. `POST /upload` - Upload and analyze dataset
2. `POST /train` - Train forecasting models
3. `POST /forecast` - Generate future predictions
4. `GET /session/{id}/info` - Get session details

#### Management:
- `GET /sessions` - List active sessions
- `DELETE /session/{id}` - Clean up session
- `GET /health` - API health check

## 🎯 Summary: Task 4 Solution

**Task 4 Goal:** "Change the input as dynamic like make this as forecasting engine where it can take the input of any data and give insights"

**✅ Solution Delivered:**

1. **Universal Data Ingestion**
   - Accepts any CSV dataset format
   - Automatic schema detection
   - Flexible feature types

2. **Dynamic Feature Engineering**
   - Auto-detects numerical, categorical, date features
   - Creates time-based features automatically
   - Generates lag features for time-series

3. **Adaptive Model Training**
   - Tests multiple algorithms automatically
   - Selects best performer per target
   - Handles single and multi-target prediction

4. **Generic API & UI**
   - Upload any dataset workflow
   - Universal dashboard interface
   - Session-based management

5. **Domain Agnostic**
   - Works for retail, finance, healthcare, etc.
   - No business logic hard-coding
   - Automatic KPI adaptation

The system now truly serves as a **universal forecasting engine** that can provide insights for any time-series business data, making it a powerful tool for any organization needing predictive analytics.

## 🎉 Conclusion

This transformation from a fixed AlcoBev system to a universal forecasting engine demonstrates how to build truly flexible ML systems. The solution addresses Task 4 completely by creating a platform that can adapt to any business domain while maintaining the advanced capabilities developed in Tasks 1 & 2.

**Ready to forecast anything? Upload your data and let the engine do the rest! 🚀**
