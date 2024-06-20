from stable_baselines3 import PPO
from flask import Flask, render_template, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import plotly.io as pio
from plotly.utils import PlotlyJSONEncoder
import json
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import datetime as dt
import os
import asyncio
from starknet_py.net.full_node_client import FullNodeClient
from starknet_py.net.signer.stark_curve_signer import StarkCurveSigner, KeyPair
from starknet_py.contract import Contract
from starknet_py.net.models import StarknetChainId
from starknet_py.net.account.account import Account
from flask import Flask, render_template_string
from pyngrok import ngrok, conf, installer
import ssl
import urllib.request
import certifi


import streamlit as st

from dotenv import load_dotenv


import plotly.colors as pc
from datetime import timedelta
from scripts.utils import set_random_seed, normalize_asset_returns, calculate_cumulative_return, calculate_cagr
from scripts.testnet_model import StakedETHEnv
from scripts.sql_scripts import sql
from scripts.processing_function import data_processing
from flipside import Flipside
import requests
import warnings
import matplotlib.pyplot as plt
import time

# Create a default SSL context that bypasses certificate verification
context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE

# Set the path to the ngrok executable installed by Chocolatey
ngrok_path = "C:\\ProgramData\\chocolatey\\bin\\ngrok.exe"

# Update the pyngrok configuration with the ngrok path
pyngrok_config = conf.PyngrokConfig(ngrok_path=ngrok_path)

# Check if ngrok is installed at the specified path, if not, install it using the custom SSL context
if not os.path.exists(pyngrok_config.ngrok_path):
    installer.install_ngrok(pyngrok_config.ngrok_path, context=context)

# Configure ngrok with custom SSL context
conf.set_default(pyngrok_config)
conf.get_default().ssl_context = context

# Set your ngrok auth token
ngrok.set_auth_token("2dJnEh2BuhCkFPQMSgCnalDSang_2oJv3XAMyVJse8yyhhiNJ")

# Start ngrok
public_url = ngrok.connect(5000, pyngrok_config=pyngrok_config).public_url
print("ngrok public URL:", public_url)

app = Flask(__name__)
deployment_version = dt.datetime.now().strftime('%Y-%m-%d %H-00-00')
#deployment_version = dt.datetime.now().strftime('%Y%m%d%H%M%S')


print(f"Current working directory: {os.getcwd()}")

# Set up the account and client
load_dotenv()

PRIVATE_KEY = os.getenv('PRIVATE_KEY')
#print(f'private key{PRIVATE_KEY}')
ACCOUNT_ADDRESS = os.getenv('ACCOUNT_ADDRESS')
#CONTRACT_ADDRESS = os.getenv('CONTRACT_ADDRESS')
GATEWAY_URL = "https://starknet-sepolia.public.blastapi.io"

if not PRIVATE_KEY or not ACCOUNT_ADDRESS:
    raise EnvironmentError("One or more environment variables (PRIVATE_KEY, ACCOUNT_ADDRESS) are not set.")

try:
    key_pair = KeyPair.from_private_key(int(PRIVATE_KEY, 16))
except ValueError as e:
    raise ValueError("Invalid PRIVATE_KEY format. It should be a valid hexadecimal string.") from e

client = FullNodeClient(node_url=GATEWAY_URL)
signer = StarkCurveSigner(account_address=ACCOUNT_ADDRESS, key_pair=key_pair, chain_id=StarknetChainId.SEPOLIA)
account = Account(client=client, address=ACCOUNT_ADDRESS, signer=signer, chain=StarknetChainId.SEPOLIA)


print(f"Connected to Starknet testnet with account: {ACCOUNT_ADDRESS}, chain: {StarknetChainId.SEPOLIA}")

async def get_balance():
    # Query the account balance
    balance_wei = await account.get_balance()
    eth_balance = balance_wei / 10**18
    #steth_address = address for deployed test net token
    #steth_balance = await account.get_balance(steth_address)
    #balance = {"ETH":eth_balance, "stETH": steth_blance, ...}
    #balance to pd to apply price feeds to them from price_df?
    print(f"ETH Balance for account {ACCOUNT_ADDRESS}: {eth_balance}")
    return eth_balance

asyncio.run(get_balance())


days_start_dev = pd.to_datetime('2024-06-17 01:00:00')
data_points = 1000
data_points_days = data_points / 24
last_date = days_start_dev + timedelta(days=data_points_days)
last_date = last_date.date()
today = dt.date.today()
days_left = last_date - today
lst_prices_query = sql(today)
api_key = os.getenv("FLIPSIDE_API_KEY")

@st.cache_data(ttl='15m')
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
        raise Exception("Error: " + response_data['error']['message'])
    query_run_id = response_data['result']['queryRun']['id']
    return response_data, query_run_id

@st.cache_data(ttl='15m')
def getQueryResults(query_run_id, attempts=10, delay=30):
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

@st.cache_data(ttl=86400)
def fetch_and_process_tbill_data(api_url, data_key, date_column, value_column, date_format='datetime'):
    api_key = os.getenv("FRED_API_KEY")
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

@app.route('/')
def index():
    return render_template('index.html', version=deployment_version)

cached_data = None

print('at latest data')
print('cached data:', cached_data)
@app.route('/latest-data')
def latest_data():
    global cached_data, today, days_left, prices_df, three_month_tbill, current_risk_free
    print('at latest data')
    print('cached data:', cached_data)
    today = dt.date.today()
    days_left = last_date - today
    balance = asyncio.run(get_balance())
    #data_version = dt.datetime.now().strftime('%Y%m%d%H%M%S')
    data_version = dt.datetime.now().strftime('%Y-%m-%d %H-00-00')


    if cached_data is not None:
        print("Cached Data:", cached_data)
        if 'data_version' in cached_data:
            print("Cached Data Version:", cached_data["data_version"])
        else:
            print("Cached Data does not have 'data_version' key")
    else:
        print("No cached data found")

    # Check if cached data is valid
    if cached_data and 'data_version' in cached_data and cached_data["data_version"] == data_version:
        return jsonify(cached_data)

    try:
        price_response_data, q_id_price = createQueryRun(lst_prices_query)
        if q_id_price:
            price_df_json = getQueryResults(q_id_price)
            if price_df_json:
                prices_df = pd.DataFrame(price_df_json['result']['rows'])
                print(f"Price data fetched: {prices_df.head()}")
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
        print(f"3-month T-bill data fetched: {three_month_tbill.head()}")
    except Exception as e:
        print(f"Error in fetching tbill data: {e}")

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
    price_timeseries = data_processing(price_dataframe)

    all_assets = ['RETH', 'SFRXETH', 'WSTETH']
    price_timeseries.reset_index()

    start_date = str((price_timeseries['ds'].min()).strftime('%Y-%m-%d %H:%M:%S'))
    end_date = dt.datetime.now().strftime('%Y-%m-%d %H:00:00')
    end_time_fix = dt.datetime.now().strftime('%Y-%m-%d %H-00-00')

    def run_sim(seed):
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

        new_compositions = compositions_df.iloc[-1].to_dict()
        print(f'new compositions: {new_compositions}')
        #asyncio.run(rebalance_portfolio(new_compositions))
        
        return states_df, rewards_df, actions_df, portfolio_values_df, compositions_df

    seed = 20
    states_df, rewards_df, actions_df, portfolio_values_df, compositions_df = run_sim(seed)
    compositions_df.set_index('Date', inplace=True)

    color_palette = pc.qualitative.Plotly
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
    wsteth_price = price_timeseries['WSTETH'].to_frame('WSTETH Price').iloc[-1]
    reth_price = price_timeseries['RETH'].to_frame('RETH Price').iloc[-1]
    sfrxeth_price = price_timeseries['SFRXETH'].to_frame('SFRXETH Price').iloc[-1]

    print(f"Prices: wstETH: {wsteth_price.values[0]}, rETH: {reth_price.values[0]}, sfrxETH: {sfrxeth_price.values[0]}")

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
    cumualtive_wsteth = calculate_cumulative_return(filtered_normalized['normalized_WSTETH'].to_frame('Portfolio_Value'))
    cumulative_sfrxeth = calculate_cumulative_return(filtered_normalized['normalized_SFRXETH'].to_frame('Portfolio_Value'))
    excess_return_reth = optimizer_cumulative_return - cumulative_reth
    excess_return_wsteth = optimizer_cumulative_return - cumualtive_wsteth
    excess_return_sfrxeth = optimizer_cumulative_return - cumulative_sfrxeth

    optimizer_cagr = calculate_cagr(portfolio_values_df['Portfolio_Value'])
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

    network = "Starknet Sepolia"

    # Ensure price values are extracted correctly
    wsteth_price_value = wsteth_price.values[0]
    reth_price_value = reth_price.values[0]
    sfrxeth_price_value = sfrxeth_price.values[0]

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
        "wstETH cumulative return": f"{cumualtive_wsteth*100:.2f}%",
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
        "sfrxETH expected return": f"{sfrxeth_expected_return*100:.2f}%",
        "current risk free": f"{current_risk_free*100:.2f}%",
        "address": ACCOUNT_ADDRESS,
        "eth balance": balance,
        "network": network,
        "wsteth price": wsteth_price_value,
        "reth price": reth_price_value, 
        "sfrxeth price": sfrxeth_price_value,
        "data_version": data_version
    }

    print(f"wsteth price {wsteth_price_value}")
    print(f"reth price {reth_price_value}")
    print(f"sfrxeth price {sfrxeth_price_value}")
    print(f"Data Version: {data_version}")

    results_df = pd.DataFrame([results])
    results_df.to_csv(f'data/sepolia_app_results/sepolia_app_results{end_time_fix}.csv')

    # Cache the results data
    cached_data = {"results": results, "graph_1": graph_json_1, "graph_2": graph_json_2, "graph_3": graph_json_3}

    return jsonify(cached_data)


if __name__ == "__main__":
    with app.app_context():
        print('getting latest data...')
        latest_data()

    print('Starting Flask app...')
    app.run(debug=True, use_debugger=True, use_reloader=False)
    print('Flask app ending.')

