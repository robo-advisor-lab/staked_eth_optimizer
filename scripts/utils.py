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
    future = model.make_future_dataframe(periods=365, freq='H')  # Predicting 365 hours into the future
    forecast = model.predict(future)
    
    # Merge the forecast with actual values
    df_merged = df_prophet.merge(forecast[['ds', 'yhat']], on='ds', how='left')
    
    # Plot the forecast
    fig = model.plot(forecast)
    plt.title(f'Forecast for {asset_name}')
    plt.show()
    
    return df_merged

def forecast_with_rebalancing_frequency(asset_name, df, rebalancing_frequency, seed=20):
    # Prepare the dataframe for Prophet
    random.seed(seed)
    np.random.seed(seed)
    df_prophet = df[['ds', asset_name]].rename(columns={asset_name: 'y'})
    
    # Initialize and fit the model
    model = Prophet()
    model.fit(df_prophet)
    
    # Make a future dataframe for predictions
    future = model.make_future_dataframe(periods=rebalancing_frequency, freq='D')  # Predicting according to rebalancing frequency
    forecast = model.predict(future)
    
    return forecast[['ds', 'yhat']]

def normalize_asset_returns(price_timeseries, start_date, normalize_value=1.0):
    # Ensure the date is in the correct format
    start_date = pd.to_datetime(start_date)
    
    # Filter the timeseries data to start from the given date
    filtered_data = price_timeseries[price_timeseries['ds'] >= start_date].copy()
    
    # Initialize previous prices as the prices on the start date
    prev_prices = filtered_data.iloc[0][['RETH', 'SFRXETH', 'WSTETH']].values
    
    # Initialize normalized values for each asset and lists to store results
    normalized_reth = [normalize_value]
    normalized_sfrxeth = [normalize_value]
    normalized_wsteth = [normalize_value]
    dates = [filtered_data.iloc[0]['ds']]
    
    # Loop through each row in the filtered data starting from the second row
    for i in range(1, len(filtered_data)):
        current_prices = filtered_data.iloc[i][['RETH', 'SFRXETH', 'WSTETH']].values
        returns = (current_prices - prev_prices) / prev_prices
        
        # Apply the returns to the normalized values for each asset
        normalized_reth.append(normalized_reth[-1] * (1 + returns[0]))
        normalized_sfrxeth.append(normalized_sfrxeth[-1] * (1 + returns[1]))
        normalized_wsteth.append(normalized_wsteth[-1] * (1 + returns[2]))
        dates.append(filtered_data.iloc[i]['ds'])
        
        # Update previous prices
        prev_prices = current_prices
    
    # Create a DataFrame for normalized values
    normalized_returns_df = pd.DataFrame({
        'ds': dates,
        'normalized_RETH': normalized_reth,
        'normalized_SFRXETH': normalized_sfrxeth,
        'normalized_WSTETH': normalized_wsteth
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

    cagr = (final_value / initial_value) ** (1 / number_of_years) - 1
    cagr_percentage = cagr * 100
    return cagr
