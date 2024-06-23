import datetime as dt
import requests
import aiohttp
import asyncio
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

from scripts.utils import forecast_with_rebalancing_frequency 

class StakedETHEnv(gym.Env):
    def __init__(self, historical_data, rebalancing_frequency, start_date, end_date, assets, seed, alpha=0.05, flipside_api_key=None):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.rebalancing_frequency = rebalancing_frequency  # In hours
        self.historical_data = historical_data[(historical_data['ds'] >= self.start_date) & (historical_data['ds'] <= self.end_date)]
        self.alpha = alpha
        self.seed(seed)
        self.current_step = 0
        self.prev_prices = None
        self.flipside_api_key = flipside_api_key  # Add Flipside API key
        self.num_assets = len(assets)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2 * self.num_assets,), dtype=np.float32)  # Adjust based on your observation space

        self.portfolio_value = 1.0  # Start with an initial portfolio value of 1.0
        self.portfolio = np.ones(self.num_assets) / self.num_assets  # Start with an equal allocation

        # Initialize logs
        self.states_log = []
        self.rewards_log = []
        self.actions_log = []
        self.portfolio_values_log = []
        self.compositions_log = []

        self.forecast_data = self.generate_forecasts()
        print(f"Initialized StakedETHEnv with {self.num_assets} assets.")

    async def fetch_live_data(self):
        url = "https://api.flipsidecrypto.com/api/v2/queries"
        headers = {'Content-Type': 'application/json', 'x-api-key': self.flipside_api_key}
        query = {
            "sql": "SELECT * FROM your_query_here",
            "resultTTLHours": 1,
            "tags": {"source": "live-data"}
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=query, headers=headers) as response:
                data = await response.json()
                return data['result']

    async def fetch_latest_data(self):
        data = await self.fetch_live_data()
        # Convert the data to a pandas DataFrame
        self.historical_data = pd.DataFrame(data)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_prices = self.historical_data.iloc[self.current_step][['RETH', 'SFRXETH', 'WSTETH']].values
        obs = self._get_obs()
        self.portfolio_value = 1.0  # Reset portfolio value at the beginning of each episode
        self.portfolio = np.ones(self.num_assets) / self.num_assets  # Reset to an equal allocation

        # Reset logs
        self.states_log = []
        self.rewards_log = []
        self.actions_log = []
        self.portfolio_values_log = []
        self.compositions_log = []

        return obs.astype(np.float32), {}

    def step(self, action):
        # Apply hourly returns to the portfolio composition
        if self.current_step > 0:
            current_prices = self.historical_data.iloc[self.current_step][['RETH', 'SFRXETH', 'WSTETH']].values
            returns = (current_prices - self.prev_prices) / self.prev_prices
            returns = returns.astype(np.float32)  # Ensure the returns are float32

            if np.any(np.isnan(returns)):
                print(f"NaN returns detected at step {self.current_step}. Current Prices: {current_prices}, Previous Prices: {self.prev_prices}")
                returns = np.nan_to_num(returns)  # Handle NaN values

            self.portfolio = self.portfolio * (1 + returns)
            self.portfolio /= np.sum(self.portfolio)  # Normalize the portfolio weights
            self.prev_prices = current_prices
            portfolio_value_update = (1 + np.sum(returns * self.portfolio))

            # Check for invalid portfolio value updates
            if np.isnan(portfolio_value_update) or np.isinf(portfolio_value_update):
                print(f"Invalid portfolio value update at step {self.current_step}: {portfolio_value_update}")
                portfolio_value_update = 1  # Default to no change to avoid NaN/inf

            self.portfolio_value *= portfolio_value_update

        # Rebalance the portfolio at specified intervals
        current_time = dt.datetime.now()
        if current_time.minute == 0 and current_time.second == 0:
            action = np.clip(action, 0, 1)
            if np.sum(action) == 0:
                print(f"Invalid action at step {self.current_step}: {action}")
                action = np.ones_like(action) / len(action)  # Default to equal weights to avoid division by zero

            action /= np.sum(action)
            self.portfolio = action

            current_date = self.historical_data.iloc[self.current_step]['ds']
            self.actions_log.append((action, current_date))

        forecasted_prices = self.get_forecasted_prices()
        state = self._get_obs(forecasted_prices)
        reward = self.calculate_reward(state, self.portfolio)
        done = self.check_done()
        truncated = False

        current_date = self.historical_data.iloc[self.current_step]['ds']
        self.states_log.append((state, current_date))
        self.rewards_log.append((reward, current_date))
        self.portfolio_values_log.append((self.portfolio_value, current_date))
        self.compositions_log.append((self.portfolio.copy(), current_date))

        self.current_step += 1

        info = {}
        return state.astype(np.float32), reward, done, truncated, info

    async def live_update(self):
        while True:
            await self.fetch_latest_data()
            await asyncio.sleep(3600)  # Update every hour