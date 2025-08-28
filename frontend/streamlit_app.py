# streamlit_app.py
import streamlit as st
import pandas as pd
import requests
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set Streamlit page configuration for a wider layout
st.set_page_config(layout="wide", page_title="AlcoBev Cash Flow Forecaster")

# --- API Configuration ---
# IMPORTANT: Replace with the actual URL of your deployed FastAPI API
API_BASE_URL = "http://127.0.0.1:8000"

# --- Sidebar Inputs for Forecast Parameters ---
st.sidebar.header("Forecast Parameters")

# Selectbox for Country
selected_country = st.sidebar.selectbox(
    "Select Country",
    ['Germany', 'UK', 'France', 'Spain'], # Ensure these match your API's expected countries
    index=0 # Default selected option
)

# Selectbox for Sales Channel
selected_channel = st.sidebar.selectbox(
    "Select Sales Channel",
    ['Off-Trade', 'On-Trade', 'E-commerce'], # Ensure these match your API's expected channels
    index=0
)

# Selectbox for Product Category
selected_product_category = st.sidebar.selectbox(
    "Select Product Category",
    ['Beer', 'Wine', 'Spirits', 'RTD'], # Ensure these match your API's expected categories
    index=0
)

# Slider for Forecast Horizon
forecast_horizon_days = st.sidebar.slider(
    "Forecast Horizon (Days)",
    min_value=7, max_value=180, value=30, step=7 # Forecast from 7 days up to 180 days
)

# Calculate start and end dates for the forecast period
start_date_forecast = date.today() + timedelta(days=1) # Start forecast from tomorrow
end_date_forecast = start_date_forecast + timedelta(days=forecast_horizon_days - 1)

# Button to trigger the forecast
# Using st.session_state to control forecast generation on button click
if st.sidebar.button("Generate Forecast"):
    st.session_state.run_forecast = True
else:
    # Initialize session state if not already set
    if 'run_forecast' not in st.session_state:
        st.session_state.run_forecast = False

# --- Main Content Area ---
st.title("💰 AlcoBev Cash Flow Forecast Dashboard (European Market)")

if st.session_state.run_forecast:
    st.info(f"Generating forecast for **{selected_product_category}** in **{selected_country}**, **{selected_channel}** for the period **{start_date_forecast} to {end_date_forecast}** ({forecast_horizon_days} days).")

    # Prepare the payload for the API request
    payload = {
        "country": selected_country,
        "channel": selected_channel,
        "product_category": selected_product_category,
        "start_date": start_date_forecast.isoformat(), # Convert date object to ISO format string
        "end_date": end_date_forecast.isoformat()
    }

    # Display a spinner while waiting for the API response
    with st.spinner("Fetching forecast..."):
        try:
            # Make the POST request to your FastAPI endpoint
            response = requests.post(f"{API_BASE_URL}/forecast", json=payload)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            forecast_data = response.json()

            # Convert the list of dictionaries to a Pandas DataFrame
            df_forecast = pd.DataFrame(forecast_data)
            # Convert 'forecast_date' column to datetime objects and set as index
            df_forecast['forecast_date'] = pd.to_datetime(df_forecast['forecast_date'])
            df_forecast = df_forecast.set_index('forecast_date').sort_index()

            st.subheader("Daily Forecasted Cash Flow Components")
            st.dataframe(df_forecast.round(2)) # Display the daily forecast data

            # --- Plotting the Forecast ---
            fig = make_subplots(rows=3, cols=1,
                                subplot_titles=("Predicted Net Sales Revenue (€)", "Predicted COGS (€)", "Predicted Operating Cash Flow (€)"))

            # Add trace for Predicted Sales Revenue
            fig.add_trace(go.Scatter(x=df_forecast.index, y=df_forecast['predicted_sales_revenue_eur'],
                                     mode='lines+markers', name='Sales Revenue',
                                     line=dict(color='green', width=2)), row=1, col=1)
            # Add trace for Predicted COGS
            fig.add_trace(go.Scatter(x=df_forecast.index, y=df_forecast['predicted_cogs_eur'],
                                     mode='lines+markers', name='COGS',
                                     line=dict(color='red', width=2)), row=2, col=1)
            # Add trace for Predicted Operating Cash Flow
            fig.add_trace(go.Scatter(x=df_forecast.index, y=df_forecast['predicted_operating_cash_flow_eur'],
                                     mode='lines+markers', name='Operating Cash Flow',
                                     line=dict(color='blue', width=3)), row=3, col=1)

            # Update layout for better visualization
            fig.update_layout(height=800, showlegend=True, title_text="Daily Forecast Visualisation",
                              hovermode="x unified") # Unified hover for better data exploration
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Amount (€)")

            st.plotly_chart(fig, use_container_width=True) # Display the interactive plot

            st.subheader("Aggregated Forecasted KPIs")
            # Aggregate daily forecasts to weekly and monthly summaries
            df_weekly = df_forecast.resample('W').sum()
            df_monthly = df_forecast.resample('M').sum()

            # Display aggregated summaries in two columns
            col1, col2 = st.columns(2)
            with col1:
                st.write("#### Weekly Forecast Summary")
                st.dataframe(df_weekly.round(2))
            with col2:
                st.write("#### Monthly Forecast Summary")
                st.dataframe(df_monthly.round(2))

            st.success("Forecast generated successfully!")

        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to the API at {API_BASE_URL}. Please ensure the FastAPI API is running.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error calling API: {e}. Please check the API response or network connection.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}. Please check the console for details.")
else:
    # Initial display when the app loads or after a forecast run
    st.info("Adjust parameters in the sidebar and click 'Generate Forecast' to get started!")
    # Placeholder image for visual appeal
    st.image("https://placehold.co/1200x400/ADD8E6/000000?text=Cash+Flow+Forecasting+Dashboard",
             caption="Illustrative Dashboard View", use_column_width=True)
    st.write("""
    This dashboard serves as a proof-of-concept for forecasting cash flow components for an AlcoBev company in the European market.
    It demonstrates how a machine learning model (served via an API) can be integrated with an interactive User Interface
    to visualize future trends for Net Sales Revenue, Cost of Goods Sold (COGS), and Operating Cash Flow.
    """)

# Instructions to run this UI:
# 1. Save this code as `streamlit_app.py`.
# 2. Ensure your FastAPI API (`app.py`) is running on `http://127.0.0.1:8000` (or the URL specified in API_BASE_URL).
# 3. Install necessary libraries: `pip install streamlit pandas plotly requests`
# 4. Run the Streamlit app from your terminal: `streamlit run streamlit_app.py`
# 5. Open your web browser to the URL provided by Streamlit (usually `http://localhost:8501`).
