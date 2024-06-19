from stable_baselines3 import PPO
from flask import Flask, render_template, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler


import plotly.graph_objs as go
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import datetime as dt
import os



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
import json


app = Flask(__name__, template_folder='templates')
print(f"Current working directory: {os.getcwd()}")


days_start_dev = pd.to_datetime('2024-06-17 01:00:00')
data_points = 1000
data_points_days = data_points / 24
last_date = days_start_dev + timedelta(days=data_points_days)
last_date = last_date.date()
today = dt.date.today()
days_left = last_date - today
lst_prices_query = sql(today)

api_key = st.secrets["api_key"]

@st.cache_data(ttl='1m')
def createQueryRun(sql):
    print('starting createQueryRun...')
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

@st.cache_data(ttl='1m')
def getQueryResults(query_run_id, attempts=10, delay=30):
    print('starting getQueryResults...')
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

  
try:
    price_response_data, q_id_price = createQueryRun(lst_prices_query)
    if q_id_price:
        price_df_json = getQueryResults(q_id_price)
        if price_df_json:
            print('obtaining price json')
            # Process and display the balance sheet data
            prices_df = pd.DataFrame(price_df_json['result']['rows'])
        else:
            print('Failed to get price results')
    else:
        print('Failed to create query run')
except Exception as e:
    print(f"Err or in fetching price data: {e}")

@st.cache_data(ttl='30 days')
def fetch_and_process_tbill_data(api_url, data_key, date_column, value_column, date_format='datetime'):
    print('starting FRED API...')
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
    
three_month_tbill_historical_api = "https://api.stlouisfed.org/fred/series/observations?series_id=TB3MS&file_type=json"
    
try:
    three_month_tbill = fetch_and_process_tbill_data(three_month_tbill_historical_api, "observations", "date", "value")
    three_month_tbill['decimal'] = three_month_tbill['value'] / 100
    current_risk_free = three_month_tbill['decimal'].iloc[-1]
except Exception as e:
    print(f"Error in fetching tbill data: {e}")





print('starting index...')
@app.route('/')
def index():
    print('starting index...')
    return render_template('index.html')

from flask import jsonify

from flask import jsonify
import plotly.io as pio

from plotly.utils import PlotlyJSONEncoder
import json

@app.route('/latest-data')
def latest_data():
    
    global today, days_left, prices_df, three_month_tbill, current_risk_free
    
    # Ensure the latest date
    today = dt.date.today()
    days_left = last_date - today
    print(f'fetching latest data as of {today}...')

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

    price_dataframe = prices_df
    price_dataframe.to_csv('data/flipside_price_data.csv')
    print('starting data processing...')
    price_timeseries = data_processing(price_dataframe)
    price_timeseries.to_csv('data/flipside_price_data_cleaned.csv')

    all_assets = ['RETH', 'SFRXETH', 'WSTETH']

    price_timeseries.reset_index()

    start_date = str((price_timeseries['ds'].min()).strftime('%Y-%m-%d %H:%M:%S'))
    end_date = dt.datetime.now().strftime('%Y-%m-%d %H:00:00')
    end_time_fix = dt.datetime.now().strftime('%Y-%m-%d %H-00-00')

    def run_sim(seed):
        app.logger.info('starting optimizer...')
        set_random_seed(seed)
        env = StakedETHEnv(historical_data=price_timeseries, rebalancing_frequency=24, start_date=start_date, end_date=end_date, assets=all_assets, seed=seed, alpha=0.05)
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000)
        model.save("staked_eth_ppo")
        model = PPO.load("staked_eth_ppo")
        env.seed(seed)
        states = []
        rewards = []
        actions = []
        portfolio_values = []
        compositions = []
        dates = []

        state, _ = env.reset()
        done = False

        while not done:
            action, _states = model.predict(state)
            next_state, reward, done, truncated, info = env.step(action)
            action = action / np.sum(action)
            states.append(next_state.flatten())
            rewards.append(reward)
            actions.append(action.flatten())
            portfolio_values.append(env.portfolio_value)
            compositions.append(env.portfolio)
            
            if env.current_step < len(env.historical_data):
                dates.append(env.historical_data.iloc[env.current_step]['ds'])
            state = next_state

        states_df = env.get_states_df()
        rewards_df = env.get_rewards_df()
        actions_df = env.get_actions_df()
        portfolio_values_df = env.get_portfolio_values_df()
        compositions_df = env.get_compositions_df()
        return states_df, rewards_df, actions_df, portfolio_values_df, compositions_df

    seed = 20
    app.logger.info('initializing model...')
    states_df, rewards_df, actions_df, portfolio_values_df, compositions_df = run_sim(seed)
    compositions_df.set_index('Date', inplace=True)
    print(f'composition df index: {compositions_df.index}')
    color_palette = pc.qualitative.Plotly
    traces = []

    fig1 = go.Figure()

    for i, column in enumerate(compositions_df.columns):
        fig1.add_trace(go.Scatter(
            x=compositions_df.index,
            y=compositions_df[column],
            mode='lines',
            stackgroup='one',
            name=column,
            line=dict(color=color_palette[i % len(color_palette)])
        ))

    fig1.update_layout(
        title='LST Optimizer Composition Over Time',
        barmode='stack',
        xaxis=dict(
            title='Date',
            tickmode='auto',
            nticks=20,
            tickformat='%Y-%m-%d',
            tickangle=-45
        ),
        yaxis=dict(
            title='Composition',
            tickmode='auto',
            nticks=10
        ),
        legend=dict(x=1.05, y=0.5),
        margin=dict(l=40, r=40, t=40, b=80)
    )
    graph_json_1 = json.dumps(fig1, cls=PlotlyJSONEncoder)

    normalized_data = normalize_asset_returns(price_timeseries, start_date=start_date, normalize_value=1)
    portfolio_values_df.set_index('Date', inplace=True)
    rewards_df.set_index('Date', inplace=True)
    traces = []

    for i, column in enumerate(rewards_df.columns):
        trace = go.Scatter(
            x=rewards_df.index,
            y=rewards_df[column],
            mode='lines',
            name=column,
            line=dict(color=color_palette[i % len(color_palette)])
        )
        traces.append(trace)
    
    layout = go.Layout(
        title='Rewards',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Value'),
        legend=dict(x=0.1, y=0.9)
    )
    
    fig2 = go.Figure(data=traces, layout=layout)
    graph_json_2 = json.dumps(fig2, cls=PlotlyJSONEncoder)

    comparison_end = portfolio_values_df.index.max()
    filtered_normalized = normalized_data[normalized_data.index <= comparison_end]
    comparison = filtered_normalized.merge(portfolio_values_df, left_index=True, right_index=True, how='inner')
    traces = []

    for i, column in enumerate(comparison.columns):
        trace = go.Scatter(
            x=comparison.index,
            y=comparison[column],
            mode='lines',
            name=column,
            line=dict(color=color_palette[i % len(color_palette)])
        )
        traces.append(trace)
    
    layout = go.Layout(
        title='Normalized Comparison to LSTs',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Value'),
        legend=dict(x=0.1, y=0.9)
    )
    
    fig3 = go.Figure(data=traces, layout=layout)
    graph_json_3 = json.dumps(fig3, cls=PlotlyJSONEncoder)

    optimizer_cumulative_return = calculate_cumulative_return(portfolio_values_df)
    cumulative_reth = calculate_cumulative_return(filtered_normalized['normalized_RETH'].to_frame('Portfolio_Value'))
    cumulative_wsteth = calculate_cumulative_return(filtered_normalized['normalized_WSTETH'].to_frame('Portfolio_Value'))
    cumulative_sfrxeth = calculate_cumulative_return(filtered_normalized['normalized_SFRXETH'].to_frame('Portfolio_Value'))
    excess_return_reth = optimizer_cumulative_return - cumulative_reth
    excess_return_wsteth = optimizer_cumulative_return - cumulative_wsteth
    excess_return_sfrxeth = optimizer_cumulative_return - cumulative_sfrxeth

    app.logger.info(f"Excess Return over rETH:{excess_return_reth*100:.2f}%")
    app.logger.info(f"Excess Return over wstETH:{excess_return_wsteth*100:.2f}%")
    app.logger.info(f"Excess Return over sfrxETH:{excess_return_sfrxeth*100:.2f}%")

    optimizer_cagr  = calculate_cagr(portfolio_values_df['Portfolio_Value'])
    reth_cagr = calculate_cagr(filtered_normalized['normalized_RETH'])
    wsteth_cagr = calculate_cagr(filtered_normalized['normalized_WSTETH'])
    sfrxeth_cagr = calculate_cagr(filtered_normalized['normalized_SFRXETH'])
    optimizer_expected_return = optimizer_cagr - current_risk_free
    reth_expected_return = reth_cagr - current_risk_free
    wsteth_expected_return = wsteth_cagr - current_risk_free
    sfrxeth_expected_return = sfrxeth_cagr - current_risk_free

    latest_port_val = portfolio_values_df['Portfolio_Value'].iloc[-1]
    latest_reth_val = filtered_normalized['normalized_RETH'].iloc[-1]
    latest_wsteth_val = filtered_normalized['normalized_WSTETH'].iloc[-1]
    latest_sfrxeth_val = filtered_normalized['normalized_SFRXETH'].iloc[-1]

    app.logger.info(f'optimizer cagr: {optimizer_cagr*100:.2f}%, optimizer expected return: {optimizer_expected_return*100:.2f}%')
    app.logger.info(f'rETH cagr: {reth_cagr*100:.2f}%, rETH expected return: {reth_expected_return*100:.2f}%')
    app.logger.info(f'wstETH cagr: {wsteth_cagr*100:.2f}%, wstETH expected return: {wsteth_expected_return*100:.2f}%')
    app.logger.info(f'sfrxETH cagr: {sfrxeth_cagr*100:.2f}%, sfrxETH expected return: {sfrxeth_expected_return*100:.2f}%')

    results = {
        "start date": start_date,
        "current date": end_date,
        "today": today,
        "optimizer latest portfolio value": f"{latest_port_val:.5f}",
        "reth latest value": f"{latest_reth_val:.5f}",
        "wsteth latest value": f"{latest_wsteth_val:.5f}",
        "sfrxeth latest value": f"{latest_sfrxeth_val:.5f}",
        "optimizer cumulative return": f"{optimizer_cumulative_return*100:.2f}%",
        "rETH cumulative return": f"{cumulative_reth*100:.2f}%",
        "wstETH cumulative return": f"{cumulative_wsteth*100:.2f}%",
        "sfrxETH cumulative return": f"{cumulative_sfrxeth*100:.2f}%",
        "optimizer Excess Return over rETH": f"{excess_return_reth*100:.2f}%",
        "optimizer Excess Return over wstETH": f"{excess_return_wsteth*100:.2f}%",
        "optimizer Excess Return over sfrxETH": f"{excess_return_sfrxeth*100:.2f}%",
        "optimizer CAGR": f"{optimizer_cagr*100:.2f}%",
        "rETH CAGR": f"{reth_cagr*100:.2f}%",
        "wstETH CAGR": f"{wsteth_cagr*100:.2f}%",
        "sfrxETH CAGR": f"{sfrxeth_cagr*100:.2f}%",
        "optimizer expected return": f"{optimizer_expected_return*100:.2f}%",
        "rETH expected return": f"{reth_expected_return*100:.2f}%",
        "wstETH expected return": f"{wsteth_expected_return*100:.2f}%",
        "sfrxETH expected return": f"{sfrxeth_expected_return*100:.2f}%"
    }

    app.logger.info(f"results: {results}")
    print(f"results: {results}")
    data_df = pd.DataFrame([results])

    data_df.to_csv(f'data/app_results/flask_app_results{end_time_fix}.csv')

    return jsonify({"results": results, "graph_1": graph_json_1, "graph_2": graph_json_2, "graph_3": graph_json_3})

def fetch_data():
    global prices_df, three_month_tbill, current_risk_free
    try:
        price_response_data, q_id_price = createQueryRun(lst_prices_query)
        if q_id_price:
            price_df_json = getQueryResults(q_id_price)
            if price_df_json:
                print('obtaining price json')
                prices_df = pd.DataFrame(price_df_json['result']['rows'])
            else:
                print('Failed to get price results')
        else:
            print('Failed to create query run')
    except Exception as e:
        print(f"Error in fetching price data: {e}")

    try:
        three_month_tbill = fetch_and_process_tbill_data(three_month_tbill_historical_api, "observations", "date", "value")
        three_month_tbill['decimal'] = three_month_tbill['value'] / 100
        current_risk_free = three_month_tbill['decimal'].iloc[-1]
    except Exception as e:
        print(f"Error in fetching tbill data: {e}")

    with app.app_context():
        latest_data()  # Call index() to update the displayed results

#fetch_data()  # Initial fetch

#scheduler = BackgroundScheduler()
#scheduler.add_job(fetch_data, 'interval', hours=1)
#scheduler.start()

if __name__ == "__main__":
    fetch_data()  # Initial fetch

    # Set up the scheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(fetch_data, 'cron', minute=0)  # This will run the job at the start of every hour
    scheduler.start()

    print('Starting Flask app...')
    app.run(debug=True, use_debugger=True, use_reloader=False)
    print('Flask app ending.')
