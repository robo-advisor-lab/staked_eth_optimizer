import os
import json
import asyncio
from flask import Flask, request, jsonify
from starknet_py.net.full_node_client import FullNodeClient
from starknet_py.net.signer.stark_curve_signer import StarkCurveSigner, KeyPair
from starknet_py.contract import Contract
from starknet_py.net.models import StarknetChainId
from starknet_py.net.account.account import Account
from starknet_py.hash.selector import get_selector_from_name
from starknet_py.net.client_models import Call
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PRIVATE_KEY = os.getenv('FUND_ACCOUNT_PRIVATE_KEY')
ACCOUNT_ADDRESS = os.getenv('FUND_ACCOUNT_ADDRESS')
GATEWAY_URL = "https://starknet-sepolia.public.blastapi.io"
WSTETH_CONTRACT_ADDRESS = os.getenv('WSTETH_CONTRACT_ADDRESS')
RETH_CONTRACT_ADDRESS = os.getenv('RETH_CONTRACT_ADDRESS')
SFRXETH_CONTRACT_ADDRESS = os.getenv('SFRXETH_CONTRACT_ADDRESS')
ETH_CONTRACT_ADDRESS = '0x049d36570d4e46f48e99674bd3fcc84644ddd6b96f7c741b1562b82f9e004dc7'

print('private key', PRIVATE_KEY)

# Initialize StarkNet client and account
key_pair = KeyPair.from_private_key(int(PRIVATE_KEY, 16))
client = FullNodeClient(node_url=GATEWAY_URL)
signer = StarkCurveSigner(account_address=ACCOUNT_ADDRESS, key_pair=key_pair, chain_id=StarknetChainId.SEPOLIA)
account = Account(client=client, address=ACCOUNT_ADDRESS, signer=signer, chain=StarknetChainId.SEPOLIA)

print(f"Connected to Starknet testnet with account: {ACCOUNT_ADDRESS}, chain: {StarknetChainId.SEPOLIA}")

async def get_balance():
    # Query the account balance
    balance_wei = await account.get_balance()
    eth_balance = balance_wei / 10**18
    print(f"ETH Balance for account {ACCOUNT_ADDRESS}: {eth_balance}")
    return eth_balance

asyncio.run(get_balance())

app = Flask(__name__)

# Initial setup
initial_value = 100  # $100
initial_prices = {
    "wsteth": 10,  # Example prices
    "reth": 20,
    "sfrxeth": 30,
    "eth": 40
}

initial_composition = {
    "wsteth": 0.25,  # 25%
    "reth": 0.25,    # 25%
    "sfrxeth": 0.25, # 25%
    "eth": 0.25      # 25%
}

current_balances = {token: initial_value * initial_composition[token] / initial_prices[token] for token in initial_composition}

@app.route('/rebalance', methods=['POST'])
def rebalance():
    data = request.json
    print("Received Rebalance Data:", data)
    prices = data['prices']
    new_compositions = data['new_compositions']
    print("Prices:", prices)
    print("New Compositions:", new_compositions)

    asyncio.run(rebalance_fund_account(prices, current_balances, new_compositions))
    return jsonify({"status": "success"})

async def rebalance_fund_account(prices, current_balances, new_compositions):
    # Calculate the total value of the fund account in USD
    total_value = sum(current_balances[token] * prices[f"{token}_price"] for token in current_balances)
    print("Total Fund Value (USD):", total_value)
    
    # Calculate the target balances based on new compositions
    target_balances = {token: total_value * new_compositions[token] / prices[f"{token}_price"] for token in new_compositions}
    print("Target Balances:", target_balances)
    
    # Calculate the differences
    differences = {token: target_balances[token] - current_balances[token] for token in current_balances}
    print("Differences:", differences)
    
    # Transfer tokens to achieve the new composition
    for token, difference in differences.items():
        if difference > 0:
            print(f"Buying {difference} of {token}")
            # Buy tokens: Transfer tokens from the account to the fund account
            await transfer_tokens_to_fund(token, difference)
        elif difference < 0:
            print(f"Selling {difference} of {token}")
            # Sell tokens: Transfer tokens from the fund account to the account
            await transfer_tokens_from_fund(token, -difference)
    
    # Update current balances
    for token in current_balances:
        current_balances[token] = target_balances[token]

    print("Rebalance completed")
    return "Rebalance completed"

async def transfer_tokens_to_fund(token, amount):
    contract_address = get_contract_address(token)
    selector = get_selector_from_name("transfer")
    call = Call(
        to_addr=int(contract_address, 16),
        selector=selector,
        calldata=[int(ACCOUNT_ADDRESS, 16), int(amount)]
    )
    try:
        response = await account.execute_v1(calls=call, max_fee=int(1e16))
        await account.client.wait_for_tx(response.transaction_hash)
        print(f"Transferred {amount} of {token} to fund account")
    except Exception as e:
        print(f"Error transferring tokens to fund: {e}")

async def transfer_tokens_from_fund(token, amount):
    contract_address = get_contract_address(token)
    selector = get_selector_from_name("transfer")
    call = Call(
        to_addr=int(contract_address, 16),
        selector=selector,
        calldata=[int(ACCOUNT_ADDRESS, 16), int(amount)]
    )
    try:
        response = await account.execute_v1(calls=call, max_fee=int(1e16))
        await account.client.wait_for_tx(response.transaction_hash)
        print(f"Transferred {amount} of {token} from fund account")
    except Exception as e:
        print(f"Error transferring tokens from fund: {e}")

def get_contract_address(token):
    if token == 'wsteth':
        return WSTETH_CONTRACT_ADDRESS
    elif token == 'reth':
        return RETH_CONTRACT_ADDRESS
    elif token == 'sfrxeth':
        return SFRXETH_CONTRACT_ADDRESS
    elif token == 'eth':
        return ETH_CONTRACT_ADDRESS
    else:
        raise ValueError("Unknown token")

if __name__ == "__main__":
    app.run(port=5001)
