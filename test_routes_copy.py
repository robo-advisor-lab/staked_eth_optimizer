import os
import json
import asyncio
import traceback
from flask import Flask, request, jsonify
from starknet_py.net.full_node_client import FullNodeClient
from starknet_py.net.signer.stark_curve_signer import StarkCurveSigner, KeyPair
from starknet_py.contract import Contract
from starknet_py.net.models import StarknetChainId
from starknet_py.net.account.account import Account
from starknet_py.hash.selector import get_selector_from_name
from starknet_py.net.client_models import Call
from dotenv import load_dotenv
import pandas as pd
from starknet_py.net.client_errors import ClientError
import marshmallow

# Load environment variables
load_dotenv()

PRIVATE_KEY = os.getenv('FUND_ACCOUNT_PRIVATE_KEY')
ACCOUNT_ADDRESS = os.getenv('FUND_ACCOUNT_ADDRESS')

GATEWAY_URL = "https://starknet-sepolia.infura.io/v3/22b286f565734e3e80221a4212adc370"
WSTETH_CONTRACT_ADDRESS = os.getenv('WSTETH_CONTRACT_ADDRESS')
RETH_CONTRACT_ADDRESS = os.getenv('RETH_CONTRACT_ADDRESS')
SFRXETH_CONTRACT_ADDRESS = os.getenv('SFRXETH_CONTRACT_ADDRESS')
ETH_CONTRACT_ADDRESS = '0x049d36570d4e46f48e99674bd3fcc84644ddd6b96f7c741b1562b82f9e004dc7'

# Initialize StarkNet client and account
key_pair = KeyPair.from_private_key(int(PRIVATE_KEY, 16))
client = FullNodeClient(node_url=GATEWAY_URL)
signer = StarkCurveSigner(account_address=ACCOUNT_ADDRESS, key_pair=key_pair, chain_id=StarknetChainId.SEPOLIA)
account = Account(client=client, address=ACCOUNT_ADDRESS, signer=signer, chain=StarknetChainId.SEPOLIA)

print(f"Connected to Starknet testnet with account: {ACCOUNT_ADDRESS}, chain: {StarknetChainId.SEPOLIA}")

print(f'key pair: {key_pair}')
print(f'client: {client}')
print(f'signer: {signer}')
print(f'account: {account}')

app = Flask(__name__)

@app.route('/rebalance', methods=['POST'])
def rebalance():
    print(f'starting rebalance function...')
    data = request.json
    recipient_address = data['recipient_address']
    token = data['token']
    amount = data['amount_to_send']
    # initial_holdings = data['initial_holdings']
    print(f'rebalance prices {recipient_address}')
    print(f'rebalance new comp {token}')
    print(f'rebalance initial holdings {amount}')

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(transfer_tokens_from_fund(token, amount, recipient_address))
    loop.close()
    
    return jsonify({"status": "success"})

async def rebalance_fund_account(prices, initial_holdings, new_compositions, recipient_address):
    print(f'starting rebalance fund account function...')
    print(f'rebalance fund func prices: {prices}')
    print(f'rebalance fund func initial holdings: {initial_holdings}')
    print(f'rebalance fund func new comp: {new_compositions}')
    
    total_value = sum(initial_holdings[token] * prices[f"{token}_price"] for token in initial_holdings)
    print(f'initial total val {total_value}')
    
    target_balances = {token: total_value * new_compositions.get(token, 0) / prices[f"{token}_price"] for token in initial_holdings}
    print(f'target bal {target_balances}')

    target_bal_results = {"target balances": target_balances}
    target_bal_df = pd.DataFrame([target_bal_results])
    target_bal_df.to_csv('data/rebal_target_bal.csv')
    
    differences = {token: target_balances[token] - initial_holdings[token] for token in initial_holdings}
    print(f'differences to adjust: {differences}')
    
    for token, difference in differences.items():
        if difference != 0:  # Adjust if there is any difference
            await transfer_tokens_from_fund(token, abs(difference), recipient_address)
        
    # Log new balances after rebalancing
    new_balances = await get_balance(recipient_address)
    print(f'new balances after rebalancing: {new_balances}')
    
    total_final_value = sum(new_balances[token] * prices[f"{token}_price"] for token in new_balances)
    print(f'Final total value after rebalance: {total_final_value}')
    
    return target_balances

async def transfer_tokens_from_fund(token, amount, recipient_address):
    print(f'starting transfer from fund account function...')
    print(f'transfer from fund token: {token}')
    print(f'transfer from fund amt: {amount}')
    contract_address = get_contract_address(token)
    selector = get_selector_from_name("transfer")
    amount_int = int(amount * 10**18)  # Convert amount to the correct decimal format
    amount_low = amount_int & ((1 << 128) - 1)
    amount_high = amount_int >> 128
    call = Call(
        to_addr=int(contract_address, 16),
        selector=selector,
        calldata=[int(recipient_address, 16), amount_low, amount_high]
    )
    print(f'contract address: {contract_address}, selector: {selector}, call: {call}')
    try:
        response = await account.execute_v1(calls=call, max_fee=int(1e16))
        print(f'Transaction response: {response}')
        tx_hash = response.transaction_hash
        print(f'Transaction hash: {tx_hash}')
        await wait_for_transaction(tx_hash)
        print(f"Transferred {amount} of {token} from fund account to {recipient_address}")
    except Exception as e:
        print(f"Error transferring tokens from fund: {e}")
        traceback.print_exc()



async def send_back_balances(target_balances, recipient_address):
    print(f'starting send back balance function...')
    print(f'send back balances: {target_balances}')
    for token, amount in target_balances.items():
        print(f'token: {token}, amount:{amount}')
        await transfer_tokens_from_fund(token, amount, recipient_address)

async def wait_for_transaction(tx_hash, retries=10, delay=10):
    print(f'Waiting for transaction {tx_hash} to be confirmed...')
    for attempt in range(retries):
        try:
            # Fetch raw transaction receipt
            raw_receipt = await account.client.get_transaction_receipt(tx_hash)
            # Manually handle and print the raw receipt for debugging
            print(f'Raw transaction receipt: {raw_receipt}')
            
            # Check if the receipt contains the necessary fields indicating success
            if 'execution_status' in raw_receipt and raw_receipt['execution_status'] == 'SUCCEEDED':
                print(f'Transaction {tx_hash} confirmed successfully.')
                return raw_receipt
            else:
                print(f'Attempt {attempt + 1}/{retries}: Transaction not yet confirmed. Retrying in {delay} seconds...')
                await asyncio.sleep(delay)
        except marshmallow.exceptions.ValidationError as e:
            print(f'Validation error: {e.messages}')
            print(f'Raw response: {e.valid_data}')
            traceback.print_exc()
            # Proceed despite the validation error
            return {'status': 'Validation error, proceeding as if successful.'}
        except ClientError as e:
            if 'Transaction hash not found' in e.message:
                print(f'Attempt {attempt + 1}/{retries}: Transaction hash not found. Retrying in {delay} seconds...')
                await asyncio.sleep(delay)
            else:
                print(f'Client error: {e.message}')
                traceback.print_exc()
                await asyncio.sleep(delay)

    raise Exception(f"Transaction {tx_hash} could not be confirmed after {retries} attempts.")


def get_contract_address(token):
    print(f'starting get contract address function...')
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

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the rebalancer!"})

if __name__ == "__main__":
    app.run(port=5001)
