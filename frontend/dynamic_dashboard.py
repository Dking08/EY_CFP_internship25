"""
Dynamic Forecasting Dashboard - Universal Frontend for Any Dataset
Works with the Dynamic Forecasting Engine API to handle any time-series data
"""

import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta
import json

# Set page configuration
st.set_page_config(
    page_title="Dynamic Forecasting Engine", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://127.0.0.1:8000"

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = None
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = None

# Helper functions
def upload_dataset(uploaded_file):
    """Upload dataset to the API."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to upload dataset: {e}")
        return None

def train_models(session_id, target_columns, test_size=0.2):
    """Train models via API."""
    try:
        payload = {
            "session_id": session_id,
            "target_columns": target_columns,
            "test_size": test_size
        }
        response = requests.post(f"{API_BASE_URL}/train", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to train models: {e}")
        return None

def generate_forecast(session_id, forecast_days, base_values=None):
    """Generate forecast via API."""
    try:
        payload = {
            "session_id": session_id,
            "forecast_days": forecast_days,
            "base_values": base_values or {}
        }
        print(f"Sending forecast payload: {payload}")  # Debug line
        response = requests.post(f"{API_BASE_URL}/forecast", json=payload)
        print(f"Response status: {response.status_code}")  # Debug line
        if response.status_code != 200:
            print(f"Response content: {response.text}")  # Debug line
            if response.status_code == 422:
                try:
                    error_detail = response.json()
                    st.error(f"Validation error: {error_detail}")
                except:
                    st.error(f"Validation error (422): {response.text}")
                return None
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to generate forecast: {e}")
        return None

def get_session_info(session_id):
    """Get session information via API."""
    try:
        response = requests.get(f"{API_BASE_URL}/session/{session_id}/info")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to get session info: {e}")
        return None

# Main UI
st.title("🚀 Dynamic Forecasting Engine")
st.markdown("**Universal ML forecasting for any time-series dataset**")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a step:", [
    "1. Upload Dataset", 
    "2. Configure & Train", 
    "3. Generate Forecasts",
    "4. Analytics & Insights"
])

# Page 1: Upload Dataset
if page == "1. Upload Dataset":
    st.header("📁 Step 1: Upload Your Dataset")
    
    st.markdown("""
    Upload any CSV file with time-series data. The system will automatically:
    - Detect feature types (numerical, categorical, dates)
    - Identify potential target variables
    - Analyze data quality
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload any time-series dataset (sales, revenue, stock prices, etc.)"
    )
    
    if uploaded_file is not None:
        st.info("📊 Analyzing your dataset...")
        
        # Show preview
        df_preview = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df_preview.head(10), use_container_width=True)
        
        # Upload to API
        if st.button("🚀 Analyze Dataset", type="primary"):
            with st.spinner("Uploading and analyzing..."):
                dataset_info = upload_dataset(uploaded_file)
                
                if dataset_info:
                    st.session_state.session_id = dataset_info['session_id']
                    st.session_state.dataset_info = dataset_info
                    st.session_state.training_complete = False
                    
                    st.success(f"✅ Dataset analyzed! Session ID: {dataset_info['session_id']}")
                    
                    # Display analysis results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Rows", f"{dataset_info['total_rows']:,}")
                        st.metric("Total Columns", dataset_info['total_columns'])
                    
                    with col2:
                        st.metric("Numerical Features", len(dataset_info['numerical_features']))
                        st.metric("Categorical Features", len(dataset_info['categorical_features']))
                    
                    with col3:
                        st.metric("Potential Targets", len(dataset_info['potential_targets']))
                        st.metric("Date Column", dataset_info['date_column'] or "Not detected")
                    
                    # Feature breakdown
                    st.subheader("🔍 Feature Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Numerical Features:**")
                        for feature in dataset_info['numerical_features'][:10]:  # Show first 10
                            st.write(f"• {feature}")
                        if len(dataset_info['numerical_features']) > 10:
                            st.write(f"... and {len(dataset_info['numerical_features']) - 10} more")
                    
                    with col2:
                        st.write("**Categorical Features:**")
                        for feature in dataset_info['categorical_features']:
                            st.write(f"• {feature}")
                    
                    st.write("**Potential Target Variables:**")
                    for target in dataset_info['potential_targets']:
                        st.write(f"🎯 {target}")
                    
                    st.info("💡 Next: Go to 'Configure & Train' to select targets and train models!")

# Page 2: Configure & Train
elif page == "2. Configure & Train":
    st.header("⚙️ Step 2: Configure & Train Models")
    
    if not st.session_state.session_id:
        st.warning("⚠️ Please upload a dataset first!")
        st.stop()
    
    dataset_info = st.session_state.dataset_info
    
    st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
    
    # Target selection
    st.subheader("🎯 Select Target Variables")
    st.markdown("Choose which variables you want to predict:")
    
    selected_targets = st.multiselect(
        "Target Variables",
        options=dataset_info['potential_targets'],
        default=dataset_info['potential_targets'][:2] if len(dataset_info['potential_targets']) >= 2 else dataset_info['potential_targets'],
        help="Select the variables you want to forecast"
    )
    
    if selected_targets:
        # Training configuration
        st.subheader("🔧 Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider(
                "Test Set Size", 
                min_value=0.1, 
                max_value=0.4, 
                value=0.2, 
                step=0.05,
                help="Proportion of data used for testing"
            )
        
        with col2:
            st.metric("Training Data", f"{(1-test_size)*100:.0f}%")
            st.metric("Testing Data", f"{test_size*100:.0f}%")
        
        # Feature information
        with st.expander("📊 Feature Summary"):
            st.write(f"**Total Features:** {len(dataset_info['numerical_features']) + len(dataset_info['categorical_features'])}")
            st.write(f"**Numerical:** {len(dataset_info['numerical_features'])}")
            st.write(f"**Categorical:** {len(dataset_info['categorical_features'])}")
            st.write(f"**Date Column:** {dataset_info['date_column']}")
        
        # Train models
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models... This may take a few minutes."):
                training_result = train_models(st.session_state.session_id, selected_targets, test_size)
                
                if training_result:
                    st.session_state.training_complete = True
                    st.session_state.model_performance = training_result['model_performance']
                    
                    st.success("✅ Models trained successfully!")
                    
                    # Display results
                    st.subheader("📈 Model Performance")
                    
                    performance_data = []
                    for target, metrics in training_result['model_performance'].items():
                        performance_data.append({
                            'Target': target,
                            'R² Score': f"{metrics['r2']:.3f}",
                            'MAE': f"{metrics['mae']:.2f}",
                            'RMSE': f"{metrics['rmse']:.2f}",
                            'MAPE': f"{metrics['mape']:.1f}%"
                        })
                    
                    performance_df = pd.DataFrame(performance_data)
                    st.dataframe(performance_df, use_container_width=True)
                    
                    # Performance visualization
                    fig = go.Figure()
                    
                    targets = list(training_result['model_performance'].keys())
                    r2_scores = [training_result['model_performance'][target]['r2'] for target in targets]
                    
                    fig.add_trace(go.Bar(
                        x=targets,
                        y=r2_scores,
                        name='R² Score',
                        marker_color='lightblue'
                    ))
                    
                    fig.update_layout(
                        title='Model Performance (R² Score)',
                        xaxis_title='Target Variables',
                        yaxis_title='R² Score',
                        yaxis=dict(range=[0, 1])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("💡 Next: Go to 'Generate Forecasts' to create predictions!")
    
    else:
        st.warning("Please select at least one target variable.")

# Page 3: Generate Forecasts
elif page == "3. Generate Forecasts":
    st.header("🔮 Step 3: Generate Forecasts")
    
    if not st.session_state.training_complete:
        st.warning("⚠️ Please train models first!")
        st.stop()
    
    st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
    
    # Forecast configuration
    st.subheader("📅 Forecast Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_days = st.slider(
            "Forecast Period (Days)", 
            min_value=7, 
            max_value=365, 
            value=30,
            help="Number of days to forecast into the future"
        )
    
    with col2:
        st.metric("Forecast Start", (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"))
        st.metric("Forecast End", (datetime.now() + timedelta(days=forecast_days)).strftime("%Y-%m-%d"))
    
    # Base values for forecasting
    st.subheader("⚙️ Base Values (Optional)")
    st.markdown("Set default values for key features:")
    
    base_values = {}
    
    # Get session info for available features
    session_info = get_session_info(st.session_state.session_id)
    
    if session_info:
        # Categorical features
        if session_info['features']['categorical']:
            st.write("**Categorical Features:**")
            col1, col2 = st.columns(2)
            
            for i, cat_feature in enumerate(session_info['features']['categorical'][:6]):  # Limit to 6
                with col1 if i % 2 == 0 else col2:
                    base_values[cat_feature] = st.text_input(f"{cat_feature}", value="default")
        
        # Some numerical features
        key_numerical = [f for f in session_info['features']['numerical'] 
                        if 'lag' not in f.lower() and 'year' not in f.lower()][:4]
        
        if key_numerical:
            st.write("**Key Numerical Features:**")
            col1, col2 = st.columns(2)
            
            for i, num_feature in enumerate(key_numerical):
                with col1 if i % 2 == 0 else col2:
                    base_values[num_feature] = st.number_input(f"{num_feature}", value=0.0)
    
    # Generate forecast
    if st.button("🚀 Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            try:
                # Clean base_values - remove empty strings and convert to appropriate types
                cleaned_base_values = {}
                for key, value in base_values.items():
                    if value != "" and value != "default":
                        try:
                            # Try to convert to float if it's a number
                            cleaned_base_values[key] = float(value) if isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit() else value
                        except:
                            cleaned_base_values[key] = value
                
                forecast_result = generate_forecast(
                    st.session_state.session_id, 
                    forecast_days, 
                    cleaned_base_values if cleaned_base_values else None
                )
                
                if forecast_result:
                    st.success("✅ Forecast generated successfully!")
                
                # Convert to DataFrame
                forecast_df = pd.DataFrame(forecast_result['predictions'])
                
                # Display summary
                st.subheader("📊 Forecast Summary")
                summary = forecast_result['prediction_summary']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Forecast Period", summary['forecast_period'])
                    st.metric("Start Date", summary.get('start_date', 'N/A'))
                
                with col2:
                    st.metric("End Date", summary.get('end_date', 'N/A'))
                    st.metric("Total Predictions", len(forecast_df))
                
                # Prediction columns
                pred_columns = [col for col in forecast_df.columns if col.startswith('predicted_')]
                
                if pred_columns:
                    with col3:
                        for pred_col in pred_columns[:2]:  # Show first 2
                            target_name = pred_col.replace('predicted_', '')
                            total_key = f'total_{target_name}'
                            if total_key in summary:
                                st.metric(f"Total {target_name}", f"{summary[total_key]:,.0f}")
                
                # Visualization
                st.subheader("📈 Forecast Visualization")
                
                if len(pred_columns) == 1:
                    # Single target
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=forecast_df.index,
                        y=forecast_df[pred_columns[0]],
                        mode='lines+markers',
                        name=pred_columns[0].replace('predicted_', ''),
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f'Forecast: {pred_columns[0].replace("predicted_", "")}',
                        xaxis_title='Days',
                        yaxis_title='Value'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                elif len(pred_columns) > 1:
                    # Multiple targets
                    fig = make_subplots(
                        rows=len(pred_columns), 
                        cols=1,
                        subplot_titles=[col.replace('predicted_', '') for col in pred_columns]
                    )
                    
                    colors = ['blue', 'red', 'green', 'orange', 'purple']
                    
                    for i, pred_col in enumerate(pred_columns):
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_df.index,
                                y=forecast_df[pred_col],
                                mode='lines+markers',
                                name=pred_col.replace('predicted_', ''),
                                line=dict(color=colors[i % len(colors)], width=2)
                            ),
                            row=i+1, col=1
                        )
                    
                    fig.update_layout(height=300*len(pred_columns), showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                st.subheader("📋 Detailed Forecast Data")
                st.dataframe(forecast_df, use_container_width=True)
                
                # Download option
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv,
                    file_name=f"forecast_{st.session_state.session_id}_{forecast_days}days.csv",
                    mime="text/csv"
                )
            
            except Exception as e:
                st.error(f"Error generating forecast: {e}")

# Page 4: Analytics & Insights
elif page == "4. Analytics & Insights":
    st.header("📊 Step 4: Analytics & Insights")
    
    if not st.session_state.session_id:
        st.warning("⚠️ Please upload a dataset first!")
        st.stop()
    
    # Get detailed session info
    session_info = get_session_info(st.session_state.session_id)
    
    if session_info:
        st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
        
        # Overview metrics
        st.subheader("📈 Session Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Data Rows", f"{session_info['data_info']['rows']:,}")
        
        with col2:
            st.metric("Features", f"{len(session_info['features']['numerical']) + len(session_info['features']['categorical'])}")
        
        with col3:
            st.metric("Trained Models", len(session_info['models']['trained_targets']))
        
        with col4:
            st.metric("Target Variables", len(session_info['features']['targets']))
        
        # Feature breakdown
        st.subheader("🔍 Feature Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Target Variables:**")
            for target in session_info['features']['targets']:
                st.write(f"🎯 {target}")
            
            st.write("**Categorical Features:**")
            for cat_feature in session_info['features']['categorical']:
                st.write(f"📊 {cat_feature}")
        
        with col2:
            st.write("**Numerical Features (Top 10):**")
            for num_feature in session_info['features']['numerical'][:10]:
                st.write(f"📈 {num_feature}")
            
            if len(session_info['features']['numerical']) > 10:
                st.write(f"... and {len(session_info['features']['numerical']) - 10} more")
        
        # Model performance
        if session_info['models']['performance']:
            st.subheader("🏆 Model Performance")
            
            performance_data = []
            for target, models_info in session_info['models']['performance'].items():
                # Handle different formats of performance data
                if isinstance(models_info, dict):
                    # If it's nested (models -> metrics), take the first model's metrics
                    if any(isinstance(v, dict) and 'r2' in str(v) for v in models_info.values()):
                        # Find the first valid metrics
                        metrics = None
                        for model_name, model_data in models_info.items():
                            if isinstance(model_data, dict) and 'r2' in model_data:
                                metrics = model_data
                                break
                    else:
                        # Direct metrics format
                        metrics = models_info
                    
                    if metrics and 'r2' in metrics:
                        performance_data.append({
                            'Target': target,
                            'R² Score': f"{float(metrics['r2']):.3f}",
                            'MAE': f"{float(metrics['mae']):.2f}",
                            'RMSE': f"{float(metrics['rmse']):.2f}",
                            'MAPE': f"{float(metrics['mape']):.1f}%"
                        })
            
            if performance_data:
                performance_df = pd.DataFrame(performance_data)
                st.dataframe(performance_df, use_container_width=True)
            else:
                st.info("No performance metrics available yet.")
        
        # Full report
        with st.expander("📄 Detailed Technical Report"):
            st.text(session_info['report'])
        
        # Session management
        st.subheader("⚙️ Session Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh Session Info"):
                st.rerun()
        
        with col2:
            if st.button("🗑️ Delete Session", type="secondary"):
                try:
                    response = requests.delete(f"{API_BASE_URL}/session/{st.session_state.session_id}")
                    if response.status_code == 200:
                        st.success("Session deleted successfully!")
                        st.session_state.session_id = None
                        st.session_state.dataset_info = None
                        st.session_state.training_complete = False
                        st.rerun()
                    else:
                        st.error("Failed to delete session")
                except Exception as e:
                    st.error(f"Error deleting session: {e}")

# Footer
st.markdown("---")
st.markdown("**🚀 Dynamic Forecasting Engine** - Universal ML forecasting for any dataset")

# Instructions in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. **Upload:** Any CSV with time-series data
    2. **Configure:** Select targets and train
    3. **Forecast:** Generate future predictions
    4. **Analyze:** Review insights and performance
    """)
    
    st.markdown("### 🎯 Supported Data")
    st.markdown("""
    - Sales data
    - Financial metrics
    - Inventory levels
    - User engagement
    - Any time-series dataset!
    """)
    
    if st.session_state.session_id:
        st.markdown("### ℹ️ Current Session")
        st.code(st.session_state.session_id)
        
        if st.session_state.training_complete:
            st.success("✅ Models trained")
        else:
            st.info("⏳ Ready to train")
