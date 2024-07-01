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

# Price data from: https://flipsidecrypto.xyz/Brandyn/q/U-IUGHYoxWfY/staked-eth


import pandas as pd
import numpy as np
import pytz

# Load data
price_data_path = 'data/prices.csv' 
prices_data = pd.read_csv(price_data_path)
prices_data = prices_data.dropna()

# Convert 'HOUR' to datetime and localize to UTC
prices_data['HOUR'] = pd.to_datetime(prices_data['HOUR']).dt.tz_localize('UTC')

# Convert 'HOUR' to Panama time
prices_data['HOUR'] = prices_data['HOUR'].dt.tz_convert('America/Panama')

# Aggregate and pivot the data
aggregated_data = prices_data.groupby(['HOUR', 'SYMBOL'], as_index=False).mean()
pivot_prices = aggregated_data.pivot(index='HOUR', columns='SYMBOL', values='PRICE').reset_index()
price_timeseries = pivot_prices.copy()
price_timeseries.set_index('HOUR', inplace=True)

# Filter the latest available data based on your local current time
current_time = pd.Timestamp.now(tz='America/Panama')
print(f"Current time (Panama): {current_time}")

# Assuming you want to use the most recent available price up to the current time
latest_data_time = price_timeseries.index[price_timeseries.index <= current_time].max()
print(f"Latest data time (Panama): {latest_data_time}")

# Use the latest price available
latest_prices = price_timeseries.loc[latest_data_time]
print(f"Latest prices: {latest_prices}")

# The rest of your code follows...
SFRXETH_price = price_timeseries['SFRXETH']
SFRXETH_price_pct = SFRXETH_price.pct_change()
anomalies = SFRXETH_price_pct.where(SFRXETH_price_pct.values > SFRXETH_price_pct.mean() * 0.5).dropna()
percentile_9 = anomalies.quantile(0.99)
anomalies_above_90th = anomalies > percentile_9
anomalies2 = anomalies_above_90th.where(anomalies_above_90th.values == True).dropna()
anomalies_dates = anomalies2.index.unique()
anomalies_dates_list = anomalies_dates.tolist()
weird_dates = price_timeseries.loc[anomalies_dates]
price_timeseries['SFRXETH_MA'] = price_timeseries['SFRXETH'].rolling(window=30, min_periods=1).mean()
price_timeseries = price_timeseries.reset_index()
additional_dates = pd.to_datetime(['2023-04-20 07:00:00', '2023-04-20 03:00:00', '2023-04-20 05:00:00', 
                                   '2023-04-18 12:00:00', '2023-04-18 13:00:00', '2023-04-18 08:00:00',
                                   '2023-04-18 09:00:00', '2023-04-19 00:00:00', '2023-04-19 01:00:00',
                                   '2023-04-19 02:00:00', '2023-04-19 03:00:00', '2023-04-19 04:00:00',
                                   '2023-04-19 05:00:00', '2023-04-20 14:00:00', '2023-04-18 01:00:00',
                                   '2023-04-19 01:00:00', '2023-04-19 02:00:00', '2023-04-19 03:00:00',
                                   '2023-04-19 04:00:00', '2023-04-19 05:00:00', '2023-04-20 07:00:00',
                                   '2023-04-20 14:00:00', '2023-04-17 22:00:00', '2023-04-17 23:00:00',
                                   '2023-04-18 00:00:00', '2023-04-18 13:00:00', '2023-04-18 14:00:00',
                                   '2023-04-18 15:00:00', '2023-04-18 23:00:00', '2023-04-19 00:00:00',
                                   '2023-04-19 01:00:00', '2023-04-19 02:00:00', '2023-04-19 03:00:00',
                                   '2023-04-19 04:00:00', '2023-04-19 05:00:00', '2023-04-19 12:00:00',
                                   '2023-04-19 14:00:00', '2023-04-20 04:00:00', '2023-04-20 05:00:00',
                                   '2023-04-20 06:00:00', '2023-04-20 07:00:00', '2023-04-20 08:00:00',
                                   '2023-04-20 09:00:00', '2023-04-20 10:00:00', '2023-04-20 11:00:00',
                                   '2023-04-20 12:00:00', '2023-04-20 13:00:00', '2023-04-20 14:00:00',
                                   '2023-04-20 15:00:00', '2023-04-20 17:00:00'])
all_dates_to_replace = anomalies_dates.union(additional_dates)
anomalies_dates_list = all_dates_to_replace.tolist()

# Calculate the 30-day moving average for the SFRXETH column
price_timeseries['SFRXETH_MA'] = price_timeseries['SFRXETH'].rolling(window=30, min_periods=1).mean()

# Debug: Print the DataFrame before replacement
print("Before replacement:")
print(price_timeseries[price_timeseries['HOUR'].isin(anomalies_dates_list)])

# Replace the values at the anomalies dates with the 30-day moving average values
price_timeseries.loc[price_timeseries['HOUR'].isin(anomalies_dates_list), 'SFRXETH'] = price_timeseries.loc[price_timeseries['HOUR'].isin(anomalies_dates_list), 'SFRXETH_MA']

# Drop the unnecessary columns if not needed
price_timeseries.drop(columns=['level_0', 'index', 'SFRXETH_MA'], inplace=True, errors='ignore')

# Debug: Print the DataFrame after replacement
print("After replacement:")
print(price_timeseries[price_timeseries['HOUR'].isin(anomalies_dates_list)])

# Continue with the rest of your processing...

# Localize the final price_timeseries to Panama time for consistency
price_timeseries['HOUR'] = price_timeseries['HOUR'].dt.tz_convert('America/Panama')


# In[ ]:





# In[33]:


#price_timeseries.set_index('HOUR').to_csv('../data/sfrx.csv')


# In[34]:


price_timeseries.set_index('HOUR', inplace=True)


# In[35]:


price_timeseries.plot()


# In[36]:


SFRXETH_CLEANED = price_timeseries['SFRXETH'].to_frame('SFRXETH')
SFRXETH_CLEANED_filtered = SFRXETH_CLEANED[SFRXETH_CLEANED.index <= '2023-07-01']


# In[37]:


#SFRXETH_CLEANED_filtered[SFRXETH_CLEANED_filtered.values > 2500].to_csv('../data/sfrx_4000.csv')


# In[38]:


SFRXETH_CLEANED_filtered[SFRXETH_CLEANED_filtered.values > 2500].plot()


# In[39]:


SFRXETH_CLEANED_filtered[SFRXETH_CLEANED_filtered.values > 2500]


# In[40]:


price_timeseries


# ## Forecasting
# 
# For sake of time, will likely not include features other than prices and moving day averages

# In[41]:


price_timeseries.reset_index(inplace=True)  # Reset index to use 'HOUR' as a column
price_timeseries.rename(columns={'HOUR': 'ds'}, inplace=True)  # Rename 'HOUR' to 'ds'

# Display the transformed dataframe
print(price_timeseries.head())


# In[42]:




# Loop through each asset and apply Prophet
#results = {}
#for asset in ['RETH', 'SFRXETH', 'WSTETH']:
    #results[asset] = forecast_with_prophet(asset, price_timeseries)

#for asset, df_merged in results.items():
    # Drop rows with NaN values in 'yhat'
    #df_merged = df_merged.dropna(subset=['yhat'])
    
    # Calculate R2
    #r2 = r2_score(df_merged['y'], df_merged['yhat'])
    
    # Calculate MAE
    #mae = mean_absolute_error(df_merged['y'], df_merged['yhat'])
    
    #print(f'Asset: {asset}')
    #print(f'R2: {r2}')
    #print(f'MAE: {mae}')
    #print()