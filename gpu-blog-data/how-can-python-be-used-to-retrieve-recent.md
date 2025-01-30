---
title: "How can Python be used to retrieve recent Uniswap V3 transactions?"
date: "2025-01-30"
id: "how-can-python-be-used-to-retrieve-recent"
---
Uniswap V3, operating on the Ethereum blockchain, records all trades as transactions immutably. Accessing these records programmatically for analysis or monitoring requires interaction with an Ethereum node and subsequent decoding of the transaction data. I've personally developed a series of data pipelines that leverage this process to analyze liquidity provider behavior on Uniswap V3, and I’ll outline the core mechanisms I employ.

The fundamental hurdle is that transaction data on the blockchain is not stored in a query-friendly relational database format. It's effectively a sequence of encoded blocks, each containing multiple transactions. Each transaction on Uniswap V3—which may be a swap, a liquidity addition, or a removal—is represented by a transaction hash, a set of addresses, and crucially, encoded function call data. This encoding follows the Ethereum ABI (Application Binary Interface) specification. To extract meaningful information from these transactions, we need to connect to an Ethereum node, retrieve these raw transactions, and decode the data field using the known ABI of the Uniswap V3 smart contracts.

Firstly, access to an Ethereum node is essential. I generally utilize a dedicated Infura or Alchemy endpoint to avoid running my own node. This connection allows querying for block data, specific transactions, and retrieving current chain state. Libraries like `web3.py` facilitate these interactions. The next challenge arises when dealing with decoding the raw transaction data. This data is typically hexadecimal and opaque without understanding the associated ABI. Uniswap V3 has several critical contracts, most notably the `UniswapV3Pool`, `NonfungiblePositionManager`, and `SwapRouter`. Each contract contains functions with specific input parameters, and understanding their ABI is critical to extracting relevant data.

Let's focus on decoding swap transactions, specifically. The `UniswapV3Pool` contract’s `swap` function is responsible for execution trades. When a swap occurs, the transaction's `data` field contains the calldata for this function encoded using ABI rules. Extracting the relevant swap details requires us to: 1) identify swap transactions; 2) obtain the ABI of the `UniswapV3Pool` contract, 3) decode the transaction data based on this ABI; 4) identify swap parameters such as token addresses, amounts in, and amounts out.

The first step is filtering for the relevant transactions. This is typically done using `web3.eth.get_transaction_receipt` to identify transactions that interacted with a known Uniswap V3 Pool address. Once the relevant transactions are identified, the crucial decoding can start. We need to instantiate the `UniswapV3Pool` contract object using the ABI and the known pool address via `web3.eth.contract`.

Here's a code snippet illustrating how to obtain and decode `swap` transaction data. Note that error handling is omitted for conciseness. The `pool_abi` variable would be replaced with a JSON representation of the Uniswap V3 Pool ABI, which can be found on Etherscan or in the Uniswap GitHub repository. The `web3` variable is assumed to be a connected `web3.Web3` object.
```python
from web3 import Web3
import json

# Assume web3 is already connected
# web3 = Web3(Web3.HTTPProvider("YOUR_INFURA_OR_ALCHEMY_ENDPOINT"))

pool_abi = json.loads("""
[{"inputs": [ /* ABI for the swap function omitted for brevity*/  ] , "name": "swap",  "outputs": [], "stateMutability": "nonpayable", "type": "function"}]
""")

pool_address = "0x8ad599c3A0ff1De0Cb73225443A2006Cb7A18307" # Example USDC/WETH pool
pool_contract = web3.eth.contract(address=pool_address, abi=pool_abi)


def decode_swap_data(tx_hash):
    transaction = web3.eth.get_transaction(tx_hash)
    try:
        decoded_input = pool_contract.decode_function_input(transaction.input)
        if decoded_input[0].fn_name == "swap":
            return decoded_input[1] # Return parsed swap arguments

    except Exception as e:
         print(f"Error decoding transaction: {tx_hash}: {e}")
         return None
    return None

#Example usage:
tx_hash = "0x1234abcd..." #Replace with actual transaction hash
decoded_data = decode_swap_data(tx_hash)
if decoded_data:
  print(f"Swap data : {decoded_data}")
```

This example focuses on retrieving and decoding swap transactions. To analyze other transaction types such as liquidity adds or removals, one would need to target the relevant function calls on the `NonfungiblePositionManager` contract. Similar to the swap example, it requires the ABI and the contract address, with the crucial difference being the function name to filter on (e.g., `mint`, `burn`, etc.). The decoding process follows the same structure, leveraging the `web3.eth.contract` method and the appropriate ABI.

Here's an example for decoding `mint` calls on the NonfungiblePositionManager contract:
```python
import json
from web3 import Web3


# Assuming web3 is a connected web3 object
# web3 = Web3(Web3.HTTPProvider("YOUR_INFURA_OR_ALCHEMY_ENDPOINT"))

position_manager_abi = json.loads("""
[
{"inputs": [/* ABI for mint function omitted */], "name": "mint", "outputs": [], "stateMutability": "nonpayable", "type": "function"}]
""")
position_manager_address = "0xC36442b4a4522E872De73BdaC4b3d1869542d1eb" # NonfungiblePositionManager address

position_manager_contract = web3.eth.contract(address=position_manager_address, abi=position_manager_abi)



def decode_mint_data(tx_hash):
    transaction = web3.eth.get_transaction(tx_hash)
    try:
       decoded_input = position_manager_contract.decode_function_input(transaction.input)
       if decoded_input[0].fn_name == "mint":
            return decoded_input[1] # Return parsed mint arguments

    except Exception as e:
         print(f"Error decoding transaction: {tx_hash}: {e}")
         return None
    return None


# Example Usage:
mint_tx_hash = "0x4567efgh..." # Replace with a relevant tx hash
decoded_mint_data = decode_mint_data(mint_tx_hash)
if decoded_mint_data:
    print(f"Mint Data: {decoded_mint_data}")

```
This demonstrates extracting parameters from liquidity add operations via the `mint` function. The process mirrors the `swap` decoding, except it uses the NonfungiblePositionManager ABI and contract address. Similar logic can be applied for other calls such as `burn`, `collect`, `increaseLiquidity`, etc.

Finally, to implement real-time monitoring or historical analysis effectively, this decoding logic must be combined with block subscription mechanisms or historical block processing capabilities. Instead of retrieving transactions one-by-one, one should aim to retrieve block data and then filter transactions relevant to Uniswap V3.  I’ve used `web3.eth.get_block` and  `web3.eth.subscribe('newHeads')` alongside transaction data processing to do this effectively.

To provide another layer of filtering beyond contract interaction, one can examine the event logs emitted by these Uniswap contracts. When a trade, mint, or burn occurs, event logs are included in the transaction receipt. These logs follow a specific structure as defined in the contract ABIs, and they provide a pre-parsed representation of the relevant data. Instead of directly decoding function calls, these event logs can often simplify the data extraction process. The `web3` library can decode event logs as well.  This allows for filtering based on specific events (e.g., only `Swap` events) and their parameters. The following provides an illustrative example of retrieving event logs, though specifics of filtering on event topic data are omitted.
```python
import json
from web3 import Web3

# Assuming web3 is connected
# web3 = Web3(Web3.HTTPProvider("YOUR_INFURA_OR_ALCHEMY_ENDPOINT"))

pool_abi = json.loads("""
[
{"anonymous": false, "inputs": [ /* Swap Event ABI */ ], "name": "Swap", "type": "event"}
]
""")


pool_address = "0x8ad599c3A0ff1De0Cb73225443A2006Cb7A18307" # USDC/WETH pool
pool_contract = web3.eth.contract(address=pool_address, abi=pool_abi)

def get_pool_events(tx_hash):
    receipt = web3.eth.get_transaction_receipt(tx_hash)
    swap_events = []
    for log in receipt.logs:
       try:
         decoded_log = pool_contract.events.Swap().process_log(log)
         swap_events.append(decoded_log.args)

       except Exception as e:
           pass # not a swap event, continue
    return swap_events

# Example Usage:
tx_hash = "0x7890abcd...." #Example Transaction Hash
events = get_pool_events(tx_hash)

if events:
  for event in events:
    print(f"Swap event data: {event}")
```
This final example demonstrates how to process transaction receipts and extract specific event data by leveraging the corresponding event ABI. This simplifies data parsing when event logs contain all the necessary information.

For further exploration, the Ethereum documentation offers detailed information on transaction structure and ABI specifications. The Uniswap GitHub repository contains the official contract ABIs for all V3 contracts. Also, Etherscan allows one to explore transaction data and contract ABIs. A firm understanding of these resources is vital for developing robust blockchain data retrieval pipelines.
