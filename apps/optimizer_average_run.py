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
import seaborn as sns

from scripts.utils import set_random_seed, forecast_with_prophet, forecast_with_rebalancing_frequency, normalize_asset_returns, calculate_cumulative_return, calculate_cagr
from scripts.model import StakedETHEnv
from scripts.data_processing import price_timeseries


all_assets = ['RETH', 'SFRXETH', 'WSTETH']

price_timeseries.reset_index()

start_date = str(price_timeseries['ds'].min())
end_date = str(price_timeseries['ds'].max())

print(f"start date: {start_date}")
print(f"end date: {end_date}")


def run_sim(seed):
    set_random_seed(seed)

    ## 12 hour or 24 hour rebalancing work well so far

    env = StakedETHEnv(historical_data=price_timeseries, rebalancing_frequency=24, start_date=start_date, end_date=end_date, assets=all_assets, seed=seed, alpha=0.05)
    #env.seed(seed)
    # Initialize the PPO model
    # model = PPO("MlpPolicy", env, verbose=1)

    # # Train the model
    # model.learn(total_timesteps=10000)

    # # Save the trained model
    # model.save("staked_eth_ppo")

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
    portfolio_values_df.set_index('Date', inplace=True)
    portfolio_values_df = portfolio_values_df['Portfolio_Value']

    return states_df, rewards_df, actions_df, portfolio_values_df, compositions_df

def main():
    num_runs = 10
    seeds = [5, 10, 15, 20, 40, 100, 200, 300, 500, 800]
    latest_portfolio_values = []
    portfolio_values = []

    for seed in seeds[:num_runs]:
        states_df, rewards_df, actions_df, portfolio_values_df, compositions_df = run_sim(seed)
        latest_value = portfolio_values_df.iloc[-1]
        portfolio_values.append(portfolio_values_df)
        latest_portfolio_values.append(latest_value)
        print('seed used', seed)
    
    avg_portfolio_value = np.mean(portfolio_values)
    avg_port_val = pd.concat(portfolio_values, axis=1).mean(axis=1)
    avg_last_portfolio_value = np.mean(latest_portfolio_values)
    print(f"average portfolio value: {avg_portfolio_value}")
    print(f"average latest value: {avg_last_portfolio_value}")
    print(f'average run {avg_port_val}')

    

    normalized_data = normalize_asset_returns(price_timeseries, start_date=start_date, normalize_value=1)

    comparison_end = portfolio_values[0].index.max()
    print(f"comparison end: {comparison_end}")
    
    comparison_end = pd.to_datetime(comparison_end)
    filtered_normalized = normalized_data[normalized_data.index <= comparison_end]
    print(f"average latest value: {avg_last_portfolio_value}")
    print(f"last lst prices: {filtered_normalized.iloc[-1]}")
    print(f"portfolio :{portfolio_values}")


    
    color_palette = pc.qualitative.Plotly
    custom_colors = ['black', 'grey', 'purple']

    # Generate a larger color palette if necessary
    if num_runs + len(filtered_normalized.columns) > len(color_palette):
        color_palette = pc.qualitative.Alphabet + pc.qualitative.Dark2 + pc.qualitative.Plotly

    traces = []

    # Loop through each run and create a trace for it
    for i in range(num_runs):
        traces.append(go.Scatter(
            x=portfolio_values[i].index,
            y=portfolio_values[i].values.flatten(),
            mode='lines+markers',
            name=f'Run {i + 1}',
            line=dict(color=color_palette[i % len(color_palette)])
        ))

    for i, column in enumerate(filtered_normalized.columns):
        trace = go.Scatter(
            x=filtered_normalized.index,
            y=filtered_normalized[column],
            mode='lines',
            name=column,
            line=dict(color=custom_colors[i % len(custom_colors)], dash='dash')
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

    filtered_normalized = filtered_normalized.sort_index()  # Ensure the index is sorted

    # Use .first and .last for DatetimeIndex
    from_date = filtered_normalized.index[0]
    to_date = filtered_normalized.index[-1]

    print(f"From Date: {from_date}")
    print(f"To Date: {to_date}")

    current_risk_free = 0.052

    optimizer_cumulative_return = calculate_cumulative_return(avg_port_val.to_frame('Portfolio_Value'))
    cumulative_reth = calculate_cumulative_return(filtered_normalized['normalized_RETH'].to_frame('Portfolio_Value'))
    cumualtive_wsteth = calculate_cumulative_return(filtered_normalized['normalized_WSTETH'].to_frame('Portfolio_Value'))
    cumulative_sfrxeth = calculate_cumulative_return(filtered_normalized['normalized_SFRXETH'].to_frame('Portfolio_Value'))
    excess_return_reth = optimizer_cumulative_return - cumulative_reth
    excess_return_wsteth = optimizer_cumulative_return - cumualtive_wsteth
    excess_return_sfrxeth = optimizer_cumulative_return - cumulative_sfrxeth

    print(f"Excess Return over rETH:{excess_return_reth*100:.2f}%")
    print(f"Excess Return over wstETH:{excess_return_wsteth*100:.2f}%")
    print(f"Excess Return over sfrxETH:{excess_return_sfrxeth*100:.2f}%")

    optimizer_cagr  = calculate_cagr(avg_port_val.to_frame('Portfolio_Value'))
    reth_cagr = calculate_cagr(filtered_normalized['normalized_RETH'])
    wsteth_cagr = calculate_cagr(filtered_normalized['normalized_WSTETH'])
    sfrxeth_cagr = calculate_cagr(filtered_normalized['normalized_SFRXETH'])
    optimizer_expected_return = optimizer_cagr - current_risk_free
    reth_expected_return = reth_cagr - current_risk_free
    wsteth_expected_return = wsteth_cagr - current_risk_free
    sfrxeth_expected_return = sfrxeth_cagr - current_risk_free

    #latest_port_val = portfolio_values_df.iloc[-1]

    print(f'optimizer cagr: {optimizer_cagr.iloc[-1]*100:.2f}%, optimizer expected return: {optimizer_expected_return.iloc[-1]*100:.2f}%')
    print(f'rETH cagr: {reth_cagr*100:.2f}%, rETH expected return: {reth_expected_return*100:.2f}%')
    print(f'wstETH cagr: {wsteth_cagr*100:.2f}%, rETH expected return: {wsteth_expected_return*100:.2f}%')
    print(f'sfrxETH cagr: {sfrxeth_cagr*100:.2f}%, rETH expected return: {sfrxeth_expected_return*100:.2f}%')

    results = {
    "start date": start_date,
    "end date": end_date,
    "optimizer latest portfolio value": avg_last_portfolio_value,
    "optimizer cumulative return": optimizer_cumulative_return,
    "rEth cumulative return": cumulative_reth,
    "wstEth cumulative return": cumualtive_wsteth,
    "sfrxEth cumulative return": cumulative_sfrxeth,
    "optimizer Excess Return over rETH": excess_return_reth,
    "optimizer Excess Return over wstETH": excess_return_wsteth,
    "optimizer Excess Return over sfrxETH": excess_return_sfrxeth,
    "optimizer CAGR": optimizer_cagr,
    "rEth CAGR": reth_cagr,
    "wstEth CAGR": wsteth_cagr,
    "sfrxEth CAGR": sfrxeth_cagr,
    "optimizer expected return": optimizer_expected_return,
    "rEth expected return": reth_expected_return,
    "wstEth expected return": wsteth_expected_return,
    "sfrxEth expected return": sfrxeth_expected_return
    
    }


    results_df = pd.DataFrame([results])
    results_df.to_csv('data/run_results.csv', index=False)

    
    

    

    

if __name__ == "__main__":
    main()
    




