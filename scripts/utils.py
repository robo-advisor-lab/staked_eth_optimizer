import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from scipy.optimize import minimize, Bounds, LinearConstraint
import plotly.graph_objs as go
import pandas as pd
#import requests
import numpy as np
#import yfinance as yf
import matplotlib
import random
import cvxpy as cp
import matplotlib.pyplot as plt
import datetime as dt
from prophet import Prophet
from sklearn.metrics import r2_score, mean_absolute_error
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.colors as pc

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def forecast_with_prophet(asset_name, df):
    # Prepare the dataframe for Prophet
    df_prophet = df[['ds', asset_name]].rename(columns={asset_name: 'y'})
    
    # Initialize and fit the model
    model = Prophet()
    model.fit(df_prophet)
    
    # Make a future dataframe for predictions
    future = model.make_future_dataframe(periods=365, freq='d')  # Predicting 365 hours into the future
    forecast = model.predict(future)
    
    # Merge the forecast with actual values
    df_merged = df_prophet.merge(forecast[['ds', 'yhat']], on='ds', how='left')
    
    # Plot the forecast
    fig = model.plot(forecast)
    plt.title(f'Forecast for {asset_name}')
    plt.show()
    
    return df_merged

def forecast_with_rebalancing_frequency(asset_name, df, rebalancing_frequency, seed=20):
    # Ensure the 'ds' column is timezone-naive
    df['ds'] = df['ds'].dt.tz_localize(None)
    
    # Prepare the dataframe for Prophet
    random.seed(seed)
    np.random.seed(seed)
    df_prophet = df[['ds', asset_name]].rename(columns={asset_name: 'y'})
    
    # Initialize and fit the model
    model = Prophet()
    model.fit(df_prophet)
    
    # Make a future dataframe for predictions starting from the last date in df_prophet
    future = model.make_future_dataframe(periods=rebalancing_frequency, freq='h')
    forecast = model.predict(future)
    
    return forecast[['ds', 'yhat']]


def normalize_asset_returns(price_timeseries, start_date, end_date, normalize_value=1.0):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Adjust start_date to the latest available date in the price_timeseries
   
    print(f'Adjusted start date: {start_date}')
    print(f'End date: {end_date}')
    
    # Filter the data based on the adjusted start date and end date
    filtered_data = price_timeseries[(price_timeseries['ds'] >= start_date) & (price_timeseries['ds'] <= end_date)].copy()
    print(f'Normalize function filtered data: {filtered_data}')
    
    if filtered_data.empty:
        print("Filtered data is empty after applying start date.")
        return pd.DataFrame()

    # Converting data to float64 to ensure compatibility with numpy functions
    prev_prices = filtered_data.iloc[0][['RETH', 'SFRXETH', 'WSTETH']].astype(np.float64).values
    print(f"Initial previous prices: {prev_prices}")

    normalized_values = {
        'RETH': [normalize_value],
        'SFRXETH': [normalize_value],
        'WSTETH': [normalize_value]
    }
    dates = [start_date]  # Use the original start date for labeling
    
    for i in range(1, len(filtered_data)):
        current_prices = filtered_data.iloc[i][['RETH', 'SFRXETH', 'WSTETH']].astype(np.float64).values
        print(f"Iteration {i}, Current Prices: {current_prices}")

        # Calculate log returns safely
        price_ratio = current_prices / prev_prices
        log_returns = np.log(price_ratio)
        print(f"Price ratio: {price_ratio}")
        print(f"Log returns: {log_returns}")

        # Update the normalized values for each asset using the exponential of log returns
        for idx, asset in enumerate(['RETH', 'SFRXETH', 'WSTETH']):
            normalized_values[asset].append(normalized_values[asset][-1] * np.exp(log_returns[idx]))
            print(f"Updated normalized value for {asset}: {normalized_values[asset][-1]}")

        dates.append(filtered_data.iloc[i]['ds'])
        prev_prices = current_prices
    
    normalized_returns_df = pd.DataFrame({
        'ds': dates,
        'normalized_RETH': normalized_values['RETH'],
        'normalized_SFRXETH': normalized_values['SFRXETH'],
        'normalized_WSTETH': normalized_values['WSTETH']
    })
    normalized_returns_df.set_index('ds', inplace=True)
    
    return normalized_returns_df



def calculate_cumulative_return(portfolio_values_df):
    """
    Calculate the cumulative return of the portfolio.
    
    Parameters:
    portfolio_values_df (pd.DataFrame): DataFrame with 'Portfolio_Value' column
    
    Returns:
    float: Cumulative return of the portfolio
    """
    initial_value = portfolio_values_df['Portfolio_Value'].iloc[0]
    final_value = portfolio_values_df['Portfolio_Value'].iloc[-1]
    cumulative_return = (final_value / initial_value) - 1
    return cumulative_return

def calculate_cagr(history):
    initial_value = history.iloc[0]
    final_value = history.iloc[-1]
    number_of_hours = (history.index[-1] - history.index[0]).total_seconds() / 3600
    number_of_years = number_of_hours / (365.25 * 24)  # Convert hours to years

    if number_of_years == 0:
        return 0

    cagr = (final_value / initial_value) ** (1 / number_of_years) - 1
    cagr_percentage = cagr * 100
    return cagr

