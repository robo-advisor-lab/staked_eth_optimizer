import requests
import json

# URL for the Flask application
url = 'http://127.0.0.1:5001'

# Data to be sent to the /rebalance endpoint
rebalance_data = {
    'prices': {
        'wsteth_price': 10,
        'reth_price': 20,
        'sfrxeth_price': 30,
        'eth_price': 40
    },
    'new_compositions': {
        'wsteth': 0.3,
        'reth': 0.3,
        'sfrxeth': 0.2,
    },
    'initial_holdings': {
        'wsteth': 0.4,
        'reth': 0.3,
        'sfrxeth': 0.2,
        'eth': 0.1
    },
    'recipient_address': '0x05ACcd31c28E14E9f9Da38e7407Ae1958344179CfBf48b6b1795c9965d464e45'  # Add the recipient address here
}

# Send POST request to /rebalance
response = requests.post(f'{url}/rebalance', json=rebalance_data)
print("Rebalance Response:")
try:
    print(response.json())
except requests.exceptions.JSONDecodeError as e:
    print("Failed to decode JSON response:", e)
    print("Response text:", response.text)

# To test the current balances endpoint
response = requests.get(f'{url}/get-balances')
print("Current Balances Response:")
try:
    print(response.json())
except requests.exceptions.JSONDecodeError as e:
    print("Failed to decode JSON response:", e)
    print("Response text:", response.text)
