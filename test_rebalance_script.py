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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PRIVATE_KEY = os.getenv('FUND_ACCOUNT_PRIVATE_KEY')
ACCOUNT_ADDRESS = os.getenv('FUND_ACCOUNT_ADDRESS')
GATEWAY_URL = "https://starknet-sepolia.infura.io/v3/22b286f565734e3e80221a4212adc370"
ETH_CONTRACT_ADDRESS = '0x049d36570d4e46f48e99674bd3fcc84644ddd6b96f7c741b1562b82f9e004dc7'

# Initialize StarkNet client and account
key_pair = KeyPair.from_private_key(int(PRIVATE_KEY, 16))
client = FullNodeClient(node_url=GATEWAY_URL)
signer = StarkCurveSigner(account_address=ACCOUNT_ADDRESS, key_pair=key_pair, chain_id=StarknetChainId.SEPOLIA)
account = Account(client=client, address=ACCOUNT_ADDRESS, signer=signer, chain=StarknetChainId.SEPOLIA)

print(f'key pair: {key_pair}')
print(f'client: {client}')
print(f'signer: {signer}')
print(f'account: {account}')

app = Flask(__name__)

@app.route('/swap-eth', methods=['POST'])
def swap_eth():
    data = request.json
    recipient_address = data['recipient_address']
    amount = data['amount']
    try:
        asyncio.run(swap_eth_tokens(recipient_address, amount))
        return jsonify({"status": "success"})
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        print(f"Unhandled error: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

ETH_CONTRACT_ABI = [
    {
        "name": "transfer",
        "type": "function",
        "inputs": [
            {"name": "recipient", "type": "core::starknet::contract_address::ContractAddress"},
            {"name": "amount", "type": "core::integer::u256"}
        ],
        "outputs": [
            {"name": "success", "type": "core::bool"}
        ],
        "state_mutability": "external"
    },
    {
        "name": "balanceOf",
        "type": "function",
        "inputs": [
            {"name": "account", "type": "core::starknet::contract_address::ContractAddress"}
        ],
        "outputs": [
            {"name": "balance", "type": "core::integer::u256"}
        ],
        "state_mutability": "view"
    }
]

async def swap_eth_tokens(recipient_address, amount):
    contract = await Contract.from_address(address=ETH_CONTRACT_ADDRESS, provider=account)
    contract.abi = ETH_CONTRACT_ABI  # Set the ABI after fetching the contract
    
    try:
        recipient_address_int = int(recipient_address, 16)
        amount_int = int(amount)
        # Split amount into low and high for u256
        amount_low = amount_int & ((1 << 128) - 1)
        amount_high = amount_int >> 128
    except ValueError as e:
        raise ValueError("Invalid address or amount format")

    try:
        # Check initial balance
        balance_initial = await contract.functions["balanceOf"].call(account.address)
        print(f'Initial balance: {balance_initial}')

        # Ensure sufficient balance
        if balance_initial[0] < amount_int:
            raise ValueError("Insufficient balance for the transfer")

        # Prepare the call
        call = contract.functions["transfer"].prepare_invoke_v1(
            recipient=recipient_address_int,
            amount={"low": amount_low, "high": amount_high}
        )
        
        # Log calldata
        print(f"Calldata for transfer: recipient={recipient_address_int}, amount_low={amount_low}, amount_high={amount_high}")

        # Execute the transaction using the account
        response = await account.execute_v1(calls=[call], max_fee=int(1e16))
        print(f"Transaction response: {response}")

        # Print response data for debugging
        print(f"Transaction response data: {response}")

        await client.wait_for_tx(response.transaction_hash)
        print(f"Transferred {amount} of ETH to {recipient_address}")

        # Check balance after transfer
        balance_after_transfer = await contract.functions["balanceOf"].call(account.address)
        print(f'Balance after transfer: {balance_after_transfer}')
    except Exception as e:
        print(f"Error swapping ETH: {e}")
        traceback.print_exc()
        raise

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the rebalancer!"})

if __name__ == "__main__":
    app.run(port=5001)
