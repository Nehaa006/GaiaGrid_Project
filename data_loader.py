import pandas as pd
import numpy as np

def load_gaia_data():
    print("🛰️ Step 1: Loading Spanish Grid & Weather Data...")
    df_energy = pd.read_csv('data/energy_dataset.csv', parse_dates=['time'], index_col='time')
    df_weather = pd.read_csv('data/weather_features.csv', parse_dates=['dt_iso'], index_col='dt_iso')

    # FIX: Only take the average of numeric columns (ignores city names/descriptions)
    weather_numeric = df_weather.select_dtypes(include=[np.number])
    weather_grouped = weather_numeric.groupby(level=0).mean()

    # Merge Spain Energy and Weather
    df = df_energy.join(weather_grouped, how='inner')
    
    print("🛰️ Step 2: Syncing Consumption Profile...")
    # We use the actual Spanish 'total load' as our consumption baseline
    # Scaling it down to a household level (kW)
    df['Global_active_power'] = df['total load actual'] / 1000 
    
    # Fill any missing values to prevent training errors
    df = df.ffill().bfill()
    
    print(f"✅ Data Fusion Complete! Row count: {len(df)}")
    return df