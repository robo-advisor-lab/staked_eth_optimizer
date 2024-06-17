import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from scipy.optimize import minimize, Bounds, LinearConstraint
import plotly.graph_objs as go
import pandas as pd
import numpy as np
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

from scripts.utils import set_random_seed, forecast_with_prophet, forecast_with_rebalancing_frequency, normalize_asset_returns
from scripts.model import StakedETHEnv
from scripts.data_processing import price_timeseries

seed = 20

set_random_seed(seed)

all_assets = ['RETH', 'SFRXETH', 'WSTETH']

price_timeseries.reset_index()

start_date = str(price_timeseries['ds'].min())
end_date = str(price_timeseries['ds'].max())

print(f"start date: {start_date}")
print(f"end date: {end_date}")


def run_sim():
    ## 12 hour or 24 hour rebalancing work well so far

    env = StakedETHEnv(historical_data=price_timeseries, rebalancing_frequency=24, start_date=start_date, end_date=end_date, assets=all_assets, seed=seed, alpha=0.05)
    #env.seed(seed)
    # Initialize the PPO model
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=10000)

    # Save the trained model
    model.save("staked_eth_ppo")

    model = PPO.load("staked_eth_ppo")
    env.seed(seed)
    # Initialize lists to store results
    states = []
    rewards = []
    actions = []
    portfolio_values = []
    compositions = []
    dates = []

    # Reset the environment to get the initial state
    state, _ = env.reset()  # Get the initial state
    done = False

    while not done:
        # Use the model to predict the action
        action, _states = model.predict(state)
        next_state, reward, done, truncated, info = env.step(action)
        
        # Normalize the action to ensure it sums to 1
        action = action / np.sum(action)
        
        # Store the results
        states.append(next_state.flatten())  # Ensure the state is flattened
        rewards.append(reward)
        actions.append(action.flatten())  # Ensure the action is flattened
        portfolio_values.append(env.portfolio_value)
        compositions.append(env.portfolio)  # Store the portfolio composition
        
        if env.current_step < len(env.historical_data):
            dates.append(env.historical_data.iloc[env.current_step]['ds'])

        # Update the state
        state = next_state

        # Print debug information
        print(f"Step: {env.current_step}")
        print(f"State: {next_state}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}")
        print(f"Portfolio Value: {env.portfolio_value}")

    # Access the logged data as DataFrames
    states_df = env.get_states_df()
    rewards_df = env.get_rewards_df()
    actions_df = env.get_actions_df()
    portfolio_values_df = env.get_portfolio_values_df()
    compositions_df = env.get_compositions_df()
    return states_df, rewards_df, actions_df, portfolio_values_df, compositions_df

def main():
    states_df, rewards_df, actions_df, portfolio_values_df, compositions_df = run_sim()
    # Analyze the results
    print("States:")
    print(states_df.head())

    print("Rewards:")
    print(rewards_df.describe())

    print("Actions:")
    print(actions_df.describe())

    print("Portfolio Values:")
    print(portfolio_values_df.head())

    print("Hourly Compositions:")
    print(compositions_df.head())

    # Plot the rewards over time

    compositions_df.set_index('Date', inplace=True)

    color_palette = pc.qualitative.Plotly

    traces = []

    # Loop through each column in compositions_df_resampled and create a trace for it
    for i, column in enumerate(compositions_df.columns):
        trace = go.Scatter(
            x=compositions_df.index,
            y=compositions_df[column],
            mode='lines',
            stackgroup='one',
            name=column,
            line=dict(color=color_palette[i % len(color_palette)])
        )
        traces.append(trace)

    layout = go.Layout(
        title='LST Optimizer Composition Over Time',
        barmode='stack',
        xaxis=dict(
            title='Date',
            tickmode='auto',
            nticks=20,
            tickangle=-45
        ),
        yaxis=dict(title='Composition'),
        legend=dict(x=1.05, y=1),
        margin=dict(l=0, r=0, t=0, b=0)  # Adjust the margins to remove extra space
    )

    # Combine the data and layout into a figure
    fig = go.Figure(data=traces, layout=layout)

    # Render the figure
    pyo.iplot(fig)

    normalized_data = normalize_asset_returns(price_timeseries, start_date='2023-01-10 00:00:00', normalize_value=1)

    #portfolio_values_df = portfolio_values_df[~portfolio_values_df.index.duplicated(keep='first')]
    portfolio_values_df.set_index('Date', inplace=True)
    rewards_df.set_index('Date', inplace=True)

    # Create an empty list to hold the traces
    traces = []
    
    # Loop through each column in combined_indicies and create a trace for it
    for i, column in enumerate(rewards_df.columns):
        trace = go.Scatter(
            x=rewards_df.index,
            y=rewards_df[column],
            mode='lines',
            name=column,
            line=dict(color=color_palette[i % len(color_palette)])
        )
        traces.append(trace)
    
    # Create the layout
    layout = go.Layout(
        title='Rewards',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Value'),
        legend=dict(x=0.1, y=0.9)
    )
    
    # Combine the data and layout into a figure
    fig = go.Figure(data=traces, layout=layout)
    
    # Render the figure
    pyo.iplot(fig)

    comparison_end = portfolio_values_df.index.max()

    filtered_normalized = normalized_data[normalized_data.index <= comparison_end]

    comparison = filtered_normalized.merge(portfolio_values_df, left_index=True, right_index=True, how='inner')

    # Create an empty list to hold the traces
    traces = []
    
    # Loop through each column in combined_indicies and create a trace for it
    for i, column in enumerate(comparison.columns):
        trace = go.Scatter(
            x=comparison.index,
            y=comparison[column],
            mode='lines',
            name=column,
            line=dict(color=color_palette[i % len(color_palette)])
        )
        traces.append(trace)
    
    # Create the layout
    layout = go.Layout(
        title='Normalized Comparison',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Value'),
        legend=dict(x=0.1, y=0.9)
    )
    
    # Combine the data and layout into a figure
    fig = go.Figure(data=traces, layout=layout)
    
    # Render the figure
    pyo.iplot(fig)
    print(comparison.tail())

    

if __name__ == "__main__":
    main()
    




