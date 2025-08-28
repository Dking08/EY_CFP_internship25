import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Configuration for Data Generation ---
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2024, 12, 31) # 5 years of data
OUTPUT_FILE = 'alcobev_europe_sales_data.csv'

# Define key categorical dimensions
COUNTRIES = ['Germany', 'United Kingdom', 'France', 'Spain']
CHANNELS = ['Off-Trade', 'On-Trade', 'E-commerce']
PRODUCT_CATEGORIES = ['Beer', 'Wine', 'Spirits', 'RTD']

# Base values and multipliers for realism
BASE_REVENUE = {
    'Beer': 350000, 'Wine': 400000, 'Spirits': 500000, 'RTD': 100000
}
BASE_COGS_RATIO = { # COGS as a percentage of revenue
    'Beer': 0.35, 'Wine': 0.40, 'Spirits': 0.45, 'RTD': 0.30
}
BASE_MARKETING_SPEND = {
    'Beer': 5000, 'Wine': 3000, 'Spirits': 6000, 'RTD': 2000
}

# --- Data Generation Logic ---
all_dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
data_records = []

print(f"Generating mock data from {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}...")

for current_date in all_dates:
    # Time-based factors
    day_of_year = current_date.timetuple().tm_yday
    month = current_date.month
    day_of_week = current_date.weekday() # Monday=0, Sunday=6

    # Simple seasonality (e.g., higher sales in summer for beer, holidays)
    seasonal_factor_yearly = 1 + 0.2 * np.sin(2 * np.pi * (day_of_year - 90) / 365) # Peak around summer
    seasonal_factor_weekly = 1 + 0.1 * np.sin(2 * np.pi * (day_of_week - 3) / 7) # Peak mid-week/weekend

    # General trend (slight growth over years)
    year_progress = (current_date - START_DATE).days / (END_DATE - START_DATE).days
    trend_factor = 1 + 0.05 * year_progress # 5% growth over the entire period

    # Mock external factors (vary slightly over time)
    consumer_confidence = 100 + 5 * np.sin(current_date.toordinal() / 50) + np.random.normal(0, 1)
    inflation_rate = 2.0 + 0.5 * np.sin(current_date.toordinal() / 100) + np.random.normal(0, 0.2)
    avg_temp_c = 10 + 15 * np.sin(2 * np.pi * (day_of_year - 90) / 365) + np.random.normal(0, 3)
    
    # Simple holiday indicator (e.g., Christmas, New Year's, Easter proxy)
    holiday_indicator = 0
    if (month == 1 and current_date.day == 1) or \
       (month == 12 and current_date.day == 25) or \
       (month == 4 and current_date.day in [1,2,3]) : # Mock Easter days
        holiday_indicator = 1

    competitor_activity_index = 0.8 + 0.1 * np.random.rand() # Random fluctuation

    for country in COUNTRIES:
        for channel in CHANNELS:
            for product_category in PRODUCT_CATEGORIES:
                # Base values for this segment
                base_rev = BASE_REVENUE[product_category]
                base_cogs_ratio = BASE_COGS_RATIO[product_category]
                base_mktg = BASE_MARKETING_SPEND[product_category]

                # Adjust for country and channel specifics
                country_multiplier = 1.0
                if country == 'United Kingdom': country_multiplier = 0.95
                elif country == 'France': country_multiplier = 1.05
                elif country == 'Spain': country_multiplier = 0.90

                channel_multiplier = 1.0
                if channel == 'On-Trade': channel_multiplier = 1.2
                elif channel == 'E-commerce': channel_multiplier = 0.7

                # Introduce promotional events
                promotional_event = 0
                if (current_date.day % 7 == 0 and np.random.rand() < 0.3) or \
                   (holiday_indicator == 1 and np.random.rand() < 0.7): # Higher chance of promo on holidays
                    promotional_event = 1
                
                promo_uplift = 1 + (0.15 * promotional_event) # 15% uplift during promotions

                # Calculate final values
                net_sales_revenue = base_rev * country_multiplier * channel_multiplier * \
                                    seasonal_factor_yearly * seasonal_factor_weekly * \
                                    trend_factor * promo_uplift * (1 + np.random.normal(0, 0.05)) # Add random noise
                
                net_sales_volume_litres = net_sales_revenue / (np.random.uniform(20, 30) * (1 + (np.random.rand()-0.5)*0.1)) # Derive volume from revenue and avg price
                
                cogs_eur = net_sales_revenue * base_cogs_ratio * (1 + np.random.normal(0, 0.03)) # COGS with some noise
                
                marketing_spend_eur = base_mktg * country_multiplier * channel_multiplier * (1 + np.random.normal(0, 0.1))
                if promotional_event == 1:
                    marketing_spend_eur *= 1.5 # Higher marketing spend during promotions

                data_records.append({
                    'Date': current_date.strftime('%Y-%m-%d'),
                    'Country': country,
                    'Channel': channel,
                    'Product_Category': product_category,
                    'Net_Sales_Volume_Litres': max(0, round(net_sales_volume_litres, 2)),
                    'Net_Sales_Revenue_EUR': max(0, round(net_sales_revenue, 2)),
                    'COGS_EUR': max(0, round(cogs_eur, 2)),
                    'Marketing_Spend_EUR': max(0, round(marketing_spend_eur, 2)),
                    'Promotional_Event': promotional_event,
                    'Consumer_Confidence_Index': round(consumer_confidence, 2),
                    'Inflation_Rate_EUR': round(inflation_rate, 2),
                    'Avg_Temp_C': round(avg_temp_c, 2),
                    'Holiday_Indicator': holiday_indicator,
                    'Competitor_Activity_Index': round(competitor_activity_index, 2)
                })

# Create DataFrame and save to CSV
df = pd.DataFrame(data_records)
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSuccessfully generated {len(df)} rows of mock data to {OUTPUT_FILE}")
print("First 5 rows:")
print(df.head())
print("\nData distribution overview:")
print(df.describe())
