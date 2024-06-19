
from stable_baselines3 import PPO

import plotly.graph_objs as go
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import datetime as dt


import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.colors as pc
from datetime import timedelta

from scripts.utils import set_random_seed, normalize_asset_returns, calculate_cumulative_return, calculate_cagr
from scripts.testnet_model import StakedETHEnv
from scripts.data_processing import price_timeseries
from scripts.sql_scripts import sql
from scripts.processing_function import data_processing
from flipside import Flipside
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import matplotlib.pyplot as plt
import time

days_start_dev = pd.to_datetime('2024-06-17 01:00:00')
data_points = 1000
data_points_days = data_points / 24
last_date = days_start_dev + timedelta(data_points_days)
last_date = last_date.date() 
today = dt.date.today()
days_left = last_date - today 
lst_prices_query = sql(today)

data_times = {
    "today": today,
    "days start": days_start_dev,
    "data points": data_points,
    "data_points_to_days": data_points_days,
    "last date": last_date,
    "days_left": days_left
}

data_df = pd.DataFrame([data_times])

data_df.to_csv('data/data_times.csv')

api_key = st.secrets["api_key"]

@st.cache_data()
def createQueryRun(sql):
    url = "https://api-v2.flipsidecrypto.xyz/json-rpc"
    payload = json.dumps({
        "jsonrpc": "2.0",
        "method": "createQueryRun",
        "params": [{
            "resultTTLHours": 1,
            "maxAgeMinutes": 0,
            "sql": sql,
            "tags": {"source": "streamlit-demo", "env": "test"},
            "dataSource": "snowflake-default",
            "dataProvider": "flipside"
        }],
        "id": 1
    })
    headers = {'Content-Type': 'application/json', 'x-api-key': api_key}
    response = requests.post(url, headers=headers, data=payload)
    response_data = response.json()
    if 'error' in response_data:
        st.error("Error: " + response_data['error']['message'])
        return None, None
    query_run_id = response_data['result']['queryRun']['id']
    return response_data, query_run_id

@st.cache_data()
def getQueryResults(query_run_id, attempts=10, delay=30):
    """Fetch query results with retries for asynchronous completion."""
    url = "https://api-v2.flipsidecrypto.xyz/json-rpc"
    payload = json.dumps({
        "jsonrpc": "2.0",
        "method": "getQueryRunResults",
        "params": [{"queryRunId": query_run_id, "format": "json", "page": {"number": 1, "size": 10000}}],
        "id": 1
    })
    headers = {'Content-Type': 'application/json', 'x-api-key': api_key}

    for attempt in range(attempts):
        response = requests.post(url, headers=headers, data=payload)
        resp_json = response.json()
        if 'result' in resp_json:
            return resp_json  # Data is ready
        elif 'error' in resp_json and 'message' in resp_json['error'] and 'not yet completed' in resp_json['error']['message']:
            time.sleep(delay)  # Wait for a bit before retrying
        else:
            break  # Break on unexpected error
    return None  # Return None if data isn't ready after all attempts

    
price_response_data, q_id_price = createQueryRun(lst_prices_query)
if q_id_price:
    price_df_json = getQueryResults(q_id_price)
    if price_df_json:
        # Process and display the balance sheet data
        prices_df = pd.DataFrame(price_df_json['result']['rows'])
        #st.write(bs_df)

def fetch_and_process_tbill_data(api_url, data_key, date_column, value_column, date_format='datetime'):
    # Retrieve the API key from Streamlit secrets
    api_key = st.secrets["FRED_API_KEY"]

    # Append the API key to the URL
    api_url_with_key = f"{api_url}&api_key={api_key}"

    response = requests.get(api_url_with_key)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data[data_key])
        
        if date_format == 'datetime':
            df[date_column] = pd.to_datetime(df[date_column])
        
        df.set_index(date_column, inplace=True)
        df[value_column] = df[value_column].astype(float)
        return df
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return pd.DataFrame()  # Return an empty DataFrame in case of failure


price_dataframe = prices_df

price_dataframe.to_csv('data/flipside_price_data.csv')
#print(f"price dataframe {price_dataframe}")

price_timeseries = data_processing(price_dataframe)
price_timeseries.to_csv('data/flipside_price_data_cleaned.csv')




three_month_tbill_historical_api = "https://api.stlouisfed.org/fred/series/observations?series_id=TB3MS&file_type=json"
three_month_tbill = fetch_and_process_tbill_data(three_month_tbill_historical_api, "observations", "date", "value")
#print(f"three month{three_month_tbill}")
three_month_tbill['decimal'] = three_month_tbill['value'] / 100
current_risk_free = three_month_tbill['decimal'].iloc[-1]



#need to make a data cleaning function and do price_timeseries = data_cleaning(prices_df)


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

        

    # Access the logged data as DataFrames
    states_df = env.get_states_df()
    rewards_df = env.get_rewards_df()
    actions_df = env.get_actions_df()
    portfolio_values_df = env.get_portfolio_values_df()
    compositions_df = env.get_compositions_df()
    return states_df, rewards_df, actions_df, portfolio_values_df, compositions_df

def main():
    #Need to train for best seed, or deploy several models (2-3) at each seed and compare for starkhack
    seed = 20
    states_df, rewards_df, actions_df, portfolio_values_df, compositions_df = run_sim(seed)
    # Analyze the results

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
        legend=dict(x=1.05, y=0.5),
        margin=dict(l=0, r=0, t=0, b=0)  # Adjust the margins to remove extra space
    )

    # Combine the data and layout into a figure
    fig = go.Figure(data=traces, layout=layout)

    # Render the figure
    pyo.iplot(fig)

    normalized_data = normalize_asset_returns(price_timeseries, start_date=start_date, normalize_value=1)

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
        title='Normalized Comparison to LSTs',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Value'),
        legend=dict(x=0.1, y=0.9)
    )
    
    # Combine the data and layout into a figure
    fig = go.Figure(data=traces, layout=layout)
    
    # Render the figure
    pyo.iplot(fig)
    print(comparison.tail())

    optimizer_cumulative_return = calculate_cumulative_return(portfolio_values_df)
    cumulative_reth = calculate_cumulative_return(filtered_normalized['normalized_RETH'].to_frame('Portfolio_Value'))
    cumualtive_wsteth = calculate_cumulative_return(filtered_normalized['normalized_WSTETH'].to_frame('Portfolio_Value'))
    cumulative_sfrxeth = calculate_cumulative_return(filtered_normalized['normalized_SFRXETH'].to_frame('Portfolio_Value'))
    excess_return_reth = optimizer_cumulative_return - cumulative_reth
    excess_return_wsteth = optimizer_cumulative_return - cumualtive_wsteth
    excess_return_sfrxeth = optimizer_cumulative_return - cumulative_sfrxeth

    print(f"Excess Return over rETH:{excess_return_reth*100:.2f}%")
    print(f"Excess Return over wstETH:{excess_return_wsteth*100:.2f}%")
    print(f"Excess Return over sfrxETH:{excess_return_sfrxeth*100:.2f}%")

    optimizer_cagr  = calculate_cagr(portfolio_values_df['Portfolio_Value'])
    reth_cagr = calculate_cagr(filtered_normalized['normalized_RETH'])
    wsteth_cagr = calculate_cagr(filtered_normalized['normalized_WSTETH'])
    sfrxeth_cagr = calculate_cagr(filtered_normalized['normalized_SFRXETH'])
    optimizer_expected_return = optimizer_cagr - current_risk_free
    reth_expected_return = reth_cagr - current_risk_free
    wsteth_expected_return = wsteth_cagr - current_risk_free
    sfrxeth_expected_return = sfrxeth_cagr - current_risk_free

    latest_port_val = portfolio_values_df.iloc[-1]

    print(f'optimizer cagr: {optimizer_cagr*100:.2f}%, optimizer expected return: {optimizer_expected_return*100:.2f}%')
    print(f'rETH cagr: {reth_cagr*100:.2f}%, rETH expected return: {reth_expected_return*100:.2f}%')
    print(f'wstETH cagr: {wsteth_cagr*100:.2f}%, rETH expected return: {wsteth_expected_return*100:.2f}%')
    print(f'sfrxETH cagr: {sfrxeth_cagr*100:.2f}%, rETH expected return: {sfrxeth_expected_return*100:.2f}%')

    results = {
    "start date": start_date,
    "end date": end_date,
    "optimizer latest portfolio value": latest_port_val,
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

    data_df = pd.DataFrame([results])

    data_df.to_csv('data/lst_optimizer_testnet_py_results.csv')

    

if __name__ == "__main__":
    main()
    




