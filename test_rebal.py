import requests
import json

# URL for the Flask application
url = 'http://127.0.0.1:5001'

# Replace with an actual recipient address
valid_recipient_address = '0x05ACcd31c28E14E9f9Da38e7407Ae1958344179CfBf48b6b1795c9965d464e45'

# Data to be sent to the /swap-eth endpoint
swap_data = {
    'recipient_address': valid_recipient_address,
    'amount': '10000000000000000'  # Amount in wei (0.01 ETH in this example)
}

print("Sending swap request with data:", swap_data)

# Send POST request to /swap-eth
response = requests.post(f'{url}/swap-eth', json=swap_data)
print("Swap ETH Response:")
try:
    print(response.json())
except requests.exceptions.JSONDecodeError as e:
    print("Failed to decode JSON response:", e)
    print("Response text:", response.text)
