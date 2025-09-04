# streamlit_app.py
import streamlit as st
import pandas as pd
import requests
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Set Streamlit page configuration for a wider layout
st.set_page_config(
    layout="wide", 
    page_title="AlcoBev Enhanced Cash Flow Forecaster",
    page_icon="💰"
)

# --- API Configuration ---
# IMPORTANT: Replace with the actual URL of your deployed FastAPI API
API_BASE_URL = "http://127.0.0.1:8000"

# --- Sidebar Inputs for Forecast Parameters ---
st.sidebar.header("🎯 Forecast Parameters")

# Forecast Type Selection
forecast_type = st.sidebar.radio(
    "Select Forecast Type",
    ["Basic Forecast", "Comprehensive Analysis"])

# Selectbox for Country
selected_country = st.sidebar.selectbox(
    "Select Country",
    ['Germany', 'France', 'Italy', 'Spain', 'UK'],
    index=0
)

# Selectbox for Sales Channel
selected_channel = st.sidebar.selectbox(
    "Select Sales Channel",
    ['Off-Trade', 'On-Trade'],
    index=0
)

# Selectbox for Product Category
selected_product_category = st.sidebar.selectbox(
    "Select Product Category",
    ['Beer', 'Wine', 'Spirits', 'RTD'],
    index=0
)

# Slider for Forecast Horizon
forecast_horizon_days = st.sidebar.slider(
    "Forecast Horizon (Days)",
    min_value=7, max_value=90, value=30, step=7
)

# Calculate start and end dates for the forecast period
start_date_forecast = date.today() + timedelta(days=1)
end_date_forecast = start_date_forecast + timedelta(days=forecast_horizon_days - 1)

# Button to trigger the forecast
if st.sidebar.button("🚀 Generate Forecast", type="primary"):
    st.session_state.run_forecast = True
    st.session_state.forecast_type = forecast_type
else:
    if 'run_forecast' not in st.session_state:
        st.session_state.run_forecast = False

# --- Helper Functions ---
def create_kpi_cards(summary_data):
    """Create KPI summary cards"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Sales Revenue",
            f"€{summary_data['total_sales_revenue_eur']:,.0f}",
            f"€{summary_data['avg_daily_sales_revenue_eur']:,.0f}/day"
        )

    with col2:
        st.metric(
            "Total Gross Profit", 
            f"€{summary_data['total_gross_profit_eur']:,.0f}",
            f"{summary_data['avg_gross_profit_margin_pct']:.1f}% margin"
        )

    with col3:
        st.metric(
            "Operating Cash Flow",
            f"€{summary_data['total_operating_cash_flow_eur']:,.0f}",
            f"€{summary_data['total_operating_cash_flow_eur']/summary_data['total_days']:,.0f}/day"
        )

    with col4:
        st.metric(
            "Average Selling Price",
            f"€{summary_data['avg_asp_per_litre_eur']:.2f}/L",
            f"€{summary_data['avg_cogs_per_litre']:.2f}/L COGS"
        )

def create_promotional_analysis(summary_data):
    """Create promotional impact analysis"""
    if summary_data['promo_days_count'] > 0:
        st.subheader("🎯 Promotional Impact Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Promotional Impact Metrics
            promo_impact = summary_data['promotional_sales_impact_pct']
            volume_impact = summary_data['promotional_volume_impact_pct']

            st.metric(
                "Sales Impact from Promotions",
                f"{promo_impact:+.1f}%",
                f"Volume: {volume_impact:+.1f}%"
            )

            st.metric(
                "Promotional Days",
                f"{summary_data['promo_days_count']} days",
                f"{summary_data['promo_days_count']/summary_data['total_days']*100:.1f}% of period"
            )

        with col2:
            # Promotional vs Non-Promotional Comparison
            promo_data = pd.DataFrame({
                'Period Type': ['Promotional Days', 'Regular Days'],
                'Avg Daily Sales': [summary_data['avg_promo_sales'], summary_data['avg_non_promo_sales']],
                'Avg Daily Volume': [summary_data['avg_promo_volume'], summary_data['avg_non_promo_volume']]
            })

            fig = px.bar(promo_data, x='Period Type', y='Avg Daily Sales',
                        title="Average Daily Sales: Promotional vs Regular Days",
                        color='Period Type', color_discrete_sequence=['#ff6b6b', '#4ecdc4'])
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No promotional events in the selected forecast period.")

def create_holiday_analysis(summary_data):
    """Create holiday sales lift analysis"""
    if summary_data['holiday_days_count'] > 0:
        st.subheader("🏖️ Holiday Sales Lift Analysis")

        col1, col2 = st.columns(2)

        with col1:
            holiday_lift = summary_data['holiday_sales_lift_pct']
            volume_lift = summary_data['holiday_volume_lift_pct']

            st.metric(
                "Holiday Sales Lift",
                f"{holiday_lift:+.1f}%",
                f"Volume: {volume_lift:+.1f}%"
            )

            st.metric(
                "Holiday Days",
                f"{summary_data['holiday_days_count']} days",
                f"{summary_data['holiday_days_count']/summary_data['total_days']*100:.1f}% of period"
            )

        with col2:
            # Holiday vs Non-Holiday Comparison
            holiday_data = pd.DataFrame({
                'Period Type': ['Holiday Days', 'Regular Days'],
                'Avg Daily Sales': [summary_data['avg_holiday_sales'], summary_data['avg_non_holiday_sales']],
                'Avg Daily Volume': [summary_data['avg_holiday_volume'], summary_data['avg_non_holiday_volume']]
            })

            fig = px.bar(holiday_data, x='Period Type', y='Avg Daily Sales',
                        title="Average Daily Sales: Holiday vs Regular Days",
                        color='Period Type', color_discrete_sequence=['#ffa726', '#66bb6a'])
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No holidays in the selected forecast period.")

def create_segment_analysis(summary_data):
    """Create segment performance analysis"""
    st.subheader("📊 Segment Performance Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Sales by Country**")
        country_data = summary_data['sales_by_country']
        for country, sales in country_data.items():
            st.metric(country, f"€{sales:,.0f}")

    with col2:
        st.write("**Sales by Channel**")
        channel_data = summary_data['sales_by_channel']
        for channel, sales in channel_data.items():
            st.metric(channel, f"€{sales:,.0f}")

    with col3:
        st.write("**Sales by Product**")
        product_data = summary_data['sales_by_product_category']
        for product, sales in product_data.items():
            st.metric(product, f"€{sales:,.0f}")

def create_performance_analysis(summary_data):
    """Create best/worst day analysis"""
    st.subheader("🏆 Performance Highlights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**🥇 Best Sales Day**")
        best_day = summary_data['best_day_sales']
        st.info(f"""
        **Date:** {best_day['date']}
        **Sales:** €{best_day['sales_revenue_eur']:,.0f}
        **Margin:** {best_day['gross_profit_margin_pct']:.1f}%
        **Promo:** {'Yes' if best_day.get('promotional_event', False) else 'No'}
        """)

    with col2:
        st.write("**📈 Highest Margin Day**")
        margin_day = summary_data['highest_margin_day']
        st.success(f"""
        **Date:** {margin_day['date']}
        **Margin:** {margin_day['gross_profit_margin_pct']:.1f}%
        **Sales:** €{margin_day['sales_revenue_eur']:,.0f}
        **ASP:** €{margin_day['asp_per_litre_eur']:.2f}/L
        """)

    with col3:
        st.write("**⚠️ Lowest Sales Day**")
        worst_day = summary_data['worst_day_sales']
        st.warning(f"""
        **Date:** {worst_day['date']}
        **Sales:** €{worst_day['sales_revenue_eur']:,.0f}
        **Margin:** {worst_day['gross_profit_margin_pct']:.1f}%
        **Cash Flow:** €{worst_day['predicted_operating_cash_flow_eur']:,.0f}
        """)

def create_daily_forecast_chart(df_forecast):
    """Create enhanced daily forecast visualizations"""
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            "Daily Sales Revenue & Gross Profit (€)",
            "Daily Volume & Marketing Spend",
            "Daily Operating Cash Flow (€)",
            "Daily KPI Trends"
        ),
        specs=[[{"secondary_y": True}],
               [{"secondary_y": True}],
               [{"secondary_y": False}],
               [{"secondary_y": True}]]
    )

    # Row 1: Sales Revenue & Gross Profit
    fig.add_trace(
        go.Scatter(x=df_forecast.index, y=df_forecast['predicted_sales_revenue_eur'],
                  mode='lines+markers', name='Sales Revenue', line=dict(color='#2E86AB', width=3)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_forecast.index, y=df_forecast['gross_profit_eur'],
                  mode='lines+markers', name='Gross Profit', line=dict(color='#A23B72', width=2)),
        row=1, col=1
    )

    # Row 2: Volume & Marketing (with secondary y-axis)
    fig.add_trace(
        go.Scatter(x=df_forecast.index, y=df_forecast['predicted_sales_volume_litres'],
                  mode='lines+markers', name='Volume (L)', line=dict(color='#F18F01', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_forecast.index, y=df_forecast['marketing_spend_eur'],
                  mode='lines+markers', name='Marketing Spend', line=dict(color='#C73E1D', width=2)),
        row=2, col=1, secondary_y=True
    )

    # Row 3: Operating Cash Flow
    fig.add_trace(
        go.Scatter(x=df_forecast.index, y=df_forecast['predicted_operating_cash_flow_eur'],
                  mode='lines+markers', name='Operating Cash Flow', 
                  line=dict(color='#4CAF50', width=3),
                  fill='tonexty', fillcolor='rgba(76, 175, 80, 0.1)'),
        row=3, col=1
    )

    # Row 4: KPI Trends
    fig.add_trace(
        go.Scatter(x=df_forecast.index, y=df_forecast['gross_profit_margin_pct'],
                  mode='lines+markers', name='Gross Margin %', line=dict(color='#9C27B0', width=2)),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_forecast.index, y=df_forecast['asp_per_litre_eur'],
                  mode='lines+markers', name='ASP per Litre (€)', line=dict(color='#FF9800', width=2)),
        row=4, col=1, secondary_y=True
    )

    # Update layout
    fig.update_layout(
        height=1200, 
        showlegend=True, 
        title_text="📈 Enhanced Daily Forecast Analysis",
        hovermode="x unified"
    )

    # Update axes labels
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Amount (€)", row=1, col=1)
    fig.update_yaxes(title_text="Volume (Litres)", row=2, col=1)
    fig.update_yaxes(title_text="Marketing Spend (€)", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Cash Flow (€)", row=3, col=1)
    fig.update_yaxes(title_text="Margin (%)", row=4, col=1)
    fig.update_yaxes(title_text="ASP (€/L)", row=4, col=1, secondary_y=True)

    return fig

# --- Main Content Area ---
st.title("💰 Enhanced AlcoBev Cash Flow Forecast Dashboard")
st.markdown("*Advanced Analytics with Promotional Impact, Holiday Lift & Segment Analysis*")

if st.session_state.run_forecast:
    forecast_type_selected = st.session_state.forecast_type

    # Display forecast info
    if forecast_type_selected == "Comprehensive Analysis":
        st.success(f"🔍 Generating **comprehensive analysis** for **{selected_product_category}** in **{selected_country}**, **{selected_channel}** channel")
    else:
        st.info(f"📊 Generating **basic forecast** for **{selected_product_category}** in **{selected_country}**, **{selected_channel}** channel")

    # Prepare the payload for the API request
    payload = {
        "country": selected_country,
        "channel": selected_channel,
        "product_category": selected_product_category,
        "start_date": start_date_forecast.isoformat(),
        "end_date": end_date_forecast.isoformat()
    }

    # Determine which API endpoint to use
    if forecast_type_selected == "Comprehensive Analysis":
        endpoint = f"{API_BASE_URL}/forecast/comprehensive"
    else:
        endpoint = f"{API_BASE_URL}/forecast"

    # Display a spinner while waiting for the API response
    with st.spinner("🚀 Fetching forecast data..."):
        try:
            # Make the POST request to your FastAPI endpoint
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()

            if forecast_type_selected == "Comprehensive Analysis":
                # Handle comprehensive response
                response_data = response.json()
                forecast_data = response_data['daily_forecasts']
                summary_data = response_data['summary']
                request_params = response_data['request_params']

                # Convert to DataFrame
                df_forecast = pd.DataFrame(forecast_data)
                df_forecast['forecast_date'] = pd.to_datetime(df_forecast['forecast_date'])
                df_forecast = df_forecast.set_index('forecast_date').sort_index()

                # === COMPREHENSIVE ANALYSIS DISPLAY ===

                # 1. KPI Summary Cards
                st.subheader("📊 Executive Summary")
                create_kpi_cards(summary_data)

                st.markdown("---")

                # 2. Enhanced Daily Forecast Visualization
                st.subheader("📈 Daily Forecast Trends")
                daily_chart = create_daily_forecast_chart(df_forecast)
                st.plotly_chart(daily_chart, use_container_width=True)

                st.markdown("---")

                # 3. Advanced Analytics (Two Columns)
                col1, col2 = st.columns(2)

                with col1:
                    create_promotional_analysis(summary_data)

                with col2:
                    create_holiday_analysis(summary_data)

                st.markdown("---")

                # 4. Segment Performance
                create_segment_analysis(summary_data)

                st.markdown("---")

                # 5. Performance Highlights
                create_performance_analysis(summary_data)

                st.markdown("---")

                # 6. Detailed Data Tables
                with st.expander("📋 Detailed Daily Forecast Data", expanded=False):
                    st.dataframe(df_forecast.round(2), use_container_width=True)

                # 7. Weekly/Monthly Aggregations
                st.subheader("📅 Period Aggregations")

                # Create weekly and monthly summaries
                df_weekly = df_forecast.resample('W').agg({
                    'predicted_sales_revenue_eur': 'sum',
                    'predicted_cogs_eur': 'sum',
                    'gross_profit_eur': 'sum',
                    'predicted_operating_cash_flow_eur': 'sum',
                    'marketing_spend_eur': 'sum',
                    'predicted_sales_volume_litres': 'sum',
                    'gross_profit_margin_pct': 'mean',
                    'asp_per_litre_eur': 'mean'
                })

                df_monthly = df_forecast.resample('M').agg({
                    'predicted_sales_revenue_eur': 'sum',
                    'predicted_cogs_eur': 'sum', 
                    'gross_profit_eur': 'sum',
                    'predicted_operating_cash_flow_eur': 'sum',
                    'marketing_spend_eur': 'sum',
                    'predicted_sales_volume_litres': 'sum',
                    'gross_profit_margin_pct': 'mean',
                    'asp_per_litre_eur': 'mean'
                })

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**📅 Weekly Summary**")
                    if len(df_weekly) > 1:
                        st.dataframe(df_weekly.round(2), use_container_width=True)
                    else:
                        st.info("Forecast period too short for weekly aggregation")

                with col2:
                    st.write("**📅 Monthly Summary**")
                    if len(df_monthly) > 0:
                        st.dataframe(df_monthly.round(2), use_container_width=True)
                    else:
                        st.info("Forecast period too short for monthly aggregation")

                # 8. Enhanced Marketing Analysis
                if summary_data.get('total_marketing_spend_eur', 0) > 0:
                    st.subheader("🎯 Marketing Efficiency Analysis")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Marketing ROI",
                            f"{summary_data['avg_revenue_per_euro_marketing']:.1f}x",
                            f"€{summary_data['avg_revenue_per_euro_marketing']:.2f} revenue per €1 spent"
                        )

                    with col2:
                        st.metric(
                            "Volume Efficiency", 
                            f"{summary_data['avg_litres_per_euro_marketing']:.1f}L/€",
                            f"Litres per euro marketing spend"
                        )

                    with col3:
                        st.metric(
                            "Marketing Investment",
                            f"€{summary_data['total_marketing_spend_eur']:,.0f}",
                            f"{summary_data['avg_marketing_spend_ratio']:.1f}% of revenue"
                        )

                st.success(f"✅ **Comprehensive analysis completed!** Generated insights for {summary_data['total_days']} days.")

            else:
                # === BASIC FORECAST DISPLAY ===
                forecast_data = response.json()
                df_forecast = pd.DataFrame(forecast_data)
                df_forecast['forecast_date'] = pd.to_datetime(df_forecast['forecast_date'])
                df_forecast = df_forecast.set_index('forecast_date').sort_index()

                st.subheader("📊 Basic Daily Forecast")

                # Basic KPI Cards
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    total_sales = df_forecast['predicted_sales_revenue_eur'].sum()
                    st.metric("Total Sales Revenue", f"€{total_sales:,.0f}")

                with col2:
                    total_gross_profit = df_forecast['gross_profit_eur'].sum()
                    st.metric("Total Gross Profit", f"€{total_gross_profit:,.0f}")

                with col3:
                    avg_margin = df_forecast['gross_profit_margin_pct'].mean()
                    st.metric("Avg Gross Margin", f"{avg_margin:.1f}%")

                with col4:
                    total_cash_flow = df_forecast['predicted_operating_cash_flow_eur'].sum()
                    st.metric("Operating Cash Flow", f"€{total_cash_flow:,.0f}")

                # Basic Charts
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=("Sales Revenue (€)", "COGS (€)", "Operating Cash Flow (€)")
                )

                fig.add_trace(
                    go.Scatter(x=df_forecast.index, y=df_forecast['predicted_sales_revenue_eur'],
                              mode='lines+markers', name='Sales Revenue', line=dict(color='green', width=2)),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(x=df_forecast.index, y=df_forecast['predicted_cogs_eur'],
                              mode='lines+markers', name='COGS', line=dict(color='red', width=2)),
                    row=2, col=1
                )

                fig.add_trace(
                    go.Scatter(x=df_forecast.index, y=df_forecast['predicted_operating_cash_flow_eur'],
                              mode='lines+markers', name='Operating Cash Flow', line=dict(color='blue', width=2)),
                    row=3, col=1
                )

                fig.update_layout(height=800, showlegend=False, title_text="Daily Forecast Visualization")
                fig.update_xaxes(title_text="Date")
                fig.update_yaxes(title_text="Amount (€)")

                st.plotly_chart(fig, use_container_width=True)

                # Basic Data Table
                st.subheader("📋 Daily Forecast Data")
                st.dataframe(df_forecast.round(2), use_container_width=True)

                # Basic Weekly/Monthly Aggregation
                st.subheader("📅 Aggregated Summaries")
                df_weekly = df_forecast.resample('W').sum()
                df_monthly = df_forecast.resample('M').sum()

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Weekly Summary**")
                    if len(df_weekly) > 1:
                        st.dataframe(df_weekly.round(2), use_container_width=True)
                    else:
                        st.info("Period too short for weekly aggregation")

                with col2:
                    st.write("**Monthly Summary**")
                    if len(df_monthly) > 0:
                        st.dataframe(df_monthly.round(2), use_container_width=True)
                    else:
                        st.info("Period too short for monthly aggregation")

                st.success("✅ **Basic forecast completed!**")

        except requests.exceptions.ConnectionError:
            st.error(f"❌ Could not connect to API at {API_BASE_URL}")
            st.info("Please ensure your FastAPI server is running on the correct port.")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ API Request Error: {e}")

        except Exception as e:
            st.error(f"❌ Unexpected Error: {e}")
            st.info("Check the console for detailed error information.")

else:
    # === WELCOME SCREEN ===
    st.info("👈 **Configure your forecast parameters in the sidebar and click 'Generate Forecast' to begin!**")

    # Welcome content with feature overview
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## 🚀 Welcome to the AlcoBev Forecasting Platform

        This advanced dashboard provides comprehensive cash flow forecasting and business intelligence for your European AlcoBev operations.
        """)

    with col2:
        st.markdown("""
        ### 🔗 **API Status:**
        """)

        # API Health Check
        try:
            health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if health_response.status_code == 200:
                st.success("🟢 API Connected")
            else:
                st.warning("🟡 API Issues")
        except:
            st.error("🔴 API Offline")

    # Placeholder visualization
    st.image(
        "https://placehold.co/1200x400/2E86AB/FFFFFF?text=Enhanced+AlcoBev+Cash+Flow+Forecasting+Dashboard",
        caption="Enhanced Cash Flow Forecasting with Advanced Analytics", 
        use_column_width=True
    )

# === FOOTER ===
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Enhanced AlcoBev Cash Flow Forecaster v2.0</strong></p>
</div>
""", unsafe_allow_html=True)

# Instructions to run this enhanced UI:
# 1. Save this code as `enhanced_streamlit_app.py`.
# 2. Ensure your enhanced FastAPI API is running on `http://127.0.0.1:8000` 
# 3. Install necessary libraries: `pip install streamlit pandas plotly requests`
# 4. Run: `streamlit run enhanced_streamlit_app.py`
# 5. Open your browser to the Streamlit URL (usually `http://localhost:8501`)