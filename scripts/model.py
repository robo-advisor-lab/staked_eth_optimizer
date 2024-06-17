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
    def __init__(self, historical_data, rebalancing_frequency, start_date, end_date, assets, seed, alpha=0.05):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.rebalancing_frequency = rebalancing_frequency  # In hours
        self.historical_data = historical_data[(historical_data['ds'] >= self.start_date) & (historical_data['ds'] <= self.end_date)]
        self.alpha = alpha
        self.seed(seed)
        self.current_step = 0
        self.prev_prices = None
        self.forecast_data = self.generate_forecasts()
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

        

        print(f"Initialized StakedETHEnv with {self.num_assets} assets.")

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.seed_value = int(seed % (2**32 - 1))
        random.seed(self.seed_value)
        np.random.seed(self.seed_value)
        return [self.seed_value]

    def generate_forecasts(self):
        forecast_data = {}
        for asset in ['RETH', 'SFRXETH', 'WSTETH']:
            forecast = forecast_with_rebalancing_frequency(asset, self.historical_data, self.rebalancing_frequency, self.seed_value)
            forecast_data[asset] = forecast
        print("Generated forecasts.")
        return forecast_data

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

        print(f"Reset environment. Initial observation: {obs}")
        return obs.astype(np.float32), {}  # Return the initial state and an empty info dictionary


    def step(self, action):
        # Apply hourly returns to the portfolio composition
        if self.current_step > 0:
            current_prices = self.historical_data.iloc[self.current_step][['RETH', 'SFRXETH', 'WSTETH']].values
            returns = (current_prices - self.prev_prices) / self.prev_prices
            self.portfolio = self.portfolio * (1 + returns)
            self.portfolio /= np.sum(self.portfolio)  # Normalize the portfolio weights

            # Debugging: Print intermediate values to understand where NaN values are introduced
            print(f"Step: {self.current_step}")
            print(f"Current Prices: {current_prices}")
            print(f"Returns: {returns}")
            print(f"Updated Portfolio: {self.portfolio}")

            self.prev_prices = current_prices
            portfolio_value_update = (1 + np.sum(returns * self.portfolio))
            
            # Check for invalid portfolio value updates
            if np.isnan(portfolio_value_update) or np.isinf(portfolio_value_update):
                print(f"Invalid portfolio value update at step {self.current_step}: {portfolio_value_update}")
                portfolio_value_update = 1  # Default to no change to avoid NaN/inf
            
            self.portfolio_value *= portfolio_value_update
            print(f"Updated Portfolio Value: {self.portfolio_value}")

        # Rebalance the portfolio at specified intervals
        if self.current_step % self.rebalancing_frequency == 0 or self.current_step == 0:
            action = np.clip(action, 0, 1)
            # Check for invalid action values
            if np.sum(action) == 0:
                print(f"Invalid action at step {self.current_step}: {action}")
                action = np.ones_like(action) / len(action)  # Default to equal weights to avoid division by zero
            
            action /= np.sum(action)
            print(f"Rebalancing step: {self.current_step}")
            print(f"Action taken: {action}")
            self.portfolio = action

            # Log action only during rebalancing
            current_date = self.historical_data.iloc[self.current_step]['ds']
            self.actions_log.append((action, current_date))

        forecasted_prices = self.get_forecasted_prices()
        state = self._get_obs(forecasted_prices)
        reward = self.calculate_reward(state, self.portfolio)
        done = self.check_done()
        truncated = False  # Assuming no truncation for simplicity

        # Log data with dates
        current_date = self.historical_data.iloc[self.current_step]['ds']
        
        self.states_log.append((state, current_date))
        self.rewards_log.append((reward, current_date))
        self.portfolio_values_log.append((self.portfolio_value, current_date))
        self.compositions_log.append((self.portfolio.copy(), current_date))

        self.current_step += 1  # Move to the next hour
        print(f"Step {self.current_step}: State: {state}, Reward: {reward}, Done: {done}, Truncated: {truncated}")
        print(f"Portfolio Value: {self.portfolio_value}")

        info = {}  # Add an empty info dictionary
        return state.astype(np.float32), reward, done, truncated, info







    def _get_obs(self, forecasted_prices=None):
        if forecasted_prices is None:
            forecasted_prices = self.get_forecasted_prices()
        current_prices = self.historical_data.iloc[self.current_step][['RETH', 'SFRXETH', 'WSTETH']].values
        obs = np.concatenate([current_prices, forecasted_prices]).flatten()
        print(f"Generated observation: {obs}")
        return obs.astype(np.float32)  # Ensure the observation is a float32 ndarray

    def calculate_reward(self, state, portfolio):
        current_prices = state[:self.num_assets]
        forecasted_prices = state[self.num_assets:]
        actual_returns = (current_prices - self.prev_prices) / self.prev_prices if self.prev_prices is not None else np.zeros_like(current_prices)
        forecasted_returns = (forecasted_prices - current_prices) / current_prices
    
        # Calculate VaR and CVaR
        returns = self.get_historical_returns()
        var, cvar = self.calculate_var_cvar(returns, self.alpha)
    
        # Reward function balancing actual, forecasted returns, and penalizing VaR and CVaR
        reward = actual_returns.mean() + 0.5 * forecasted_returns.mean() - 0.1 * var.mean() - 0.1 * np.nan_to_num(cvar).mean()
        self.prev_prices = current_prices
    
        print(f"Calculated reward: {reward}")
        return reward.item()  # Ensure the reward is a scalar

    def get_forecasted_prices(self):
        # Fetch forecasted prices for the current step based on rebalancing frequency
        forecasted_prices = []
        for asset in ['RETH', 'SFRXETH', 'WSTETH']:
            forecasted_price = self.forecast_data[asset].iloc[self.current_step]['yhat']
            forecasted_prices.append(forecasted_price)
        forecasted_prices = np.array(forecasted_prices)
        print(f"Forecasted prices: {forecasted_prices}")
        return forecasted_prices.astype(np.float32)  # Ensure the forecasted prices are float32 ndarray

    def get_historical_returns(self):
        # Get historical returns up to the current step
        historical_prices = self.historical_data.iloc[:self.current_step][['RETH', 'SFRXETH', 'WSTETH']]
        returns = np.log(historical_prices / historical_prices.shift(1)).dropna().values
        print(f"Historical returns: {returns}")
        return returns.astype(np.float32)  # Ensure the returns are float32 ndarray

    def calculate_var_cvar(self, returns, alpha):
        if len(returns) == 0:
            return np.array([0], dtype=np.float32), np.array([0], dtype=np.float32)  # or any other default value
        
        sorted_returns = np.sort(returns, axis=0)
        index = int(alpha * len(sorted_returns))
        var = sorted_returns[index]
        cvar = sorted_returns[:index].mean(axis=0)
        return var.astype(np.float32), cvar.astype(np.float32)

    def check_done(self):
        done = self.current_step >= len(self.historical_data) - 1
        print(f"Check done: {done}")
        return done

    # Methods to get logged data as DataFrames
    def get_states_df(self):
        states, dates = zip(*self.states_log) if self.states_log else ([], [])
        return pd.DataFrame(states, columns=['RETH', 'SFRXETH', 'WSTETH', 'Forecasted_RETH', 'Forecasted_SFRXETH', 'Forecasted_WSTETH']).assign(Date=dates)

    def get_rewards_df(self):
        rewards, dates = zip(*self.rewards_log) if self.rewards_log else ([], [])
        return pd.DataFrame(rewards, columns=['Reward']).assign(Date=dates)

    def get_actions_df(self):
        actions, dates = zip(*self.actions_log) if self.actions_log else ([], [])
        return pd.DataFrame(actions, columns=['RETH_weight', 'SFRXETH_weight', 'WSTETH_weight']).assign(Date=dates)

    def get_portfolio_values_df(self):
        portfolio_values, dates = zip(*self.portfolio_values_log) if self.portfolio_values_log else ([], [])
        return pd.DataFrame(portfolio_values, columns=['Portfolio_Value']).assign(Date=dates)

    def get_compositions_df(self):
        compositions, dates = zip(*self.compositions_log) if self.compositions_log else ([], [])
        return pd.DataFrame(compositions, columns=['RETH_weight', 'SFRXETH_weight', 'WSTETH_weight']).assign(Date=dates)
