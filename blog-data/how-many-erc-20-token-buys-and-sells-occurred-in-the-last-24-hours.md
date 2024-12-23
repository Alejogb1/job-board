---
title: "How many ERC-20 token buys and sells occurred in the last 24 hours?"
date: "2024-12-23"
id: "how-many-erc-20-token-buys-and-sells-occurred-in-the-last-24-hours"
---

Alright, let's unpack this. Tracking the precise number of erc-20 token buy and sell transactions within a specific timeframe, like the last 24 hours, isn't as straightforward as querying a single database. It necessitates a nuanced approach, combining on-chain data analysis with careful filtering. I've tackled similar challenges numerous times, usually when needing to audit decentralized exchange (dex) activity or validate data for internal reporting tools. It’s not like a simple SQL query where you’d get the total number of rows. Let's break it down methodically.

The crux of the matter lies in the nature of the Ethereum blockchain. Transactions are recorded immutably, but not in a structured way that allows for direct “buy” or “sell” query. Instead, what we have are *transfers* of erc-20 tokens between addresses. These transfers are the fundamental unit of activity. To determine whether a transfer constitutes a "buy" or "sell," we have to examine the context within which the transfer occurred, typically by monitoring events emitted by decentralized exchanges or other smart contracts.

There's no magic bullet here. I remember a particularly tricky project where I needed to reconcile discrepancies between a custom dex analytics dashboard and the actual on-chain activity for a client. It took a few late nights poring over bytecode and event logs, but ultimately, the issue stemmed from a subtle misunderstanding about how certain dex smart contracts emitted events. The lesson was, and remains: careful attention to the fine details of the smart contract in question is paramount.

So, how do we practically approach this? We typically need to follow these steps:

1.  **Data Acquisition:** We begin by fetching on-chain data. This involves connecting to an Ethereum node (or using a service like infura, Alchemy, or QuickNode) and requesting relevant blocks and transaction receipts. These receipts contain the logs we're interested in, specifically event logs emitted by contracts.

2.  **Event Filtering:** We filter these logs to pinpoint the specific event signatures related to token transfers (the `Transfer` event is standard for erc-20 but some contracts may use custom events). Crucially, we *also* need to identify events emitted by relevant exchange contracts which indicate buys or sells. These are highly contract-specific. A typical dex will emit events such as `Swap`, `Deposit`, or `Withdrawal`, but their naming convention and structure is something you need to retrieve from the particular contract’s ABI. Therefore, having the ABI’s (Application Binary Interface) is very important.

3.  **Contextual Analysis:** Once we have the transfer events, we need to analyse their context, i.e., determine whether the transfer was part of a swap (which would then become a buy/sell). This often involves looking at the addresses of the *sender* and *receiver* of the tokens, comparing those to the contract address of a dex contract we are targeting, and analyzing the event logs. For example, if tokens moved to a user from a dex's smart contract, and the event fired was related to a swap, it may signal a buy. Conversely, a transfer to the dex contract coupled with a “swap” event emitted from that dex contract may signal a sell.

4.  **Time Windowing:** Finally, we restrict our analysis to the timeframe we're interested in, in this case, the last 24 hours. This means we need to convert the block timestamps into human-readable times and do the filtering based on those values.

Here are three code snippets, using Python and web3.py, that illustrate these steps. Remember that for clarity these snippets are simplified and not production-ready, and that the exact implementation will vary based on the dex being analyzed.

**Snippet 1: Basic Event Fetching and Filtering:**

```python
from web3 import Web3

# Replace with your Ethereum node endpoint
infura_url = "YOUR_INFURA_ENDPOINT_HERE"
w3 = Web3(Web3.HTTPProvider(infura_url))

# Replace with ERC20 contract address and ABI
contract_address = '0xdAC17F958D2ee523a2206206994597C13D831ec7'
abi = [...] # Load your erc-20 abi here
erc20_contract = w3.eth.contract(address=contract_address, abi=abi)

# Get the most recent block
latest_block = w3.eth.block_number

# Calculate the block number 24 hours ago
blocks_per_day = 60 * 60 * 24 / 12
start_block = latest_block - int(blocks_per_day)

# Fetch Transfer events within the block range
events = erc20_contract.events.Transfer.get_logs(fromBlock=start_block, toBlock=latest_block)

for event in events:
    print(f"From: {event.args.from}, To: {event.args.to}, Value: {event.args.value}")
```

This first snippet focuses on how to connect to a web3 endpoint, how to load a contract, and how to retrieve transfer events in a certain block range. Please note that `abi = [...]` must be filled with the appropriate ABI (Application Binary Interface).

**Snippet 2: Dex Interaction Filtering (Uniswap V2 example):**

```python
# Replace with Uniswap V2 Router contract address and ABI
uniswap_address = '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D'
uniswap_abi = [...] # Load uniswap V2 router abi here
uniswap_contract = w3.eth.contract(address=uniswap_address, abi=uniswap_abi)

# Assuming we have 'events' from Snippet 1
for transfer_event in events:
  tx_hash = transfer_event.transactionHash
  tx_receipt = w3.eth.get_transaction_receipt(tx_hash)

  # Check if the transaction involved the uniswap router contract
  for log in tx_receipt.logs:
        if log.address == uniswap_address:
            # Look for events like "Swap" to identify buys/sells. This is an example!
            try:
                event_sig = w3.keccak(text=log.topics[0].hex()).hex()
                if event_sig == w3.keccak(text='Swap(address,uint256,uint256,uint256,uint256,address)').hex():
                    decoded_event = uniswap_contract.events.Swap().process_log(log)
                    print(f"Uniswap Swap Event: Sender: {decoded_event.args['sender']}, amount0In: {decoded_event.args['amount0In']}, amount1In: {decoded_event.args['amount1In']}, amount0Out: {decoded_event.args['amount0Out']}, amount1Out: {decoded_event.args['amount1Out']}")
            except:
                pass # handle errors parsing events
```

This second snippet extends on the previous one by retrieving transaction receipts and then filtering transaction logs for events emitted by a dex router contract (in this example, uniswap v2 router). It shows how to find a “swap” event and decode its content. Again, it is important to note that contract addresses and ABI must be filled in appropriately.

**Snippet 3: Aggregating Buy and Sell Counts:**

```python
buy_count = 0
sell_count = 0

# Assuming we have processed events as shown above and have now context
for transfer_event in events:
  tx_hash = transfer_event.transactionHash
  tx_receipt = w3.eth.get_transaction_receipt(tx_hash)
  # (insert here processing steps similar to snippet 2)
  #...
  # now lets say, after analyzing event logs we know that if the address of the token is the first address of a swap event it signals a sale, and a buy otherwise.

  if "buy" in context: # where context is a variable that stores the result of our earlier analysis
      buy_count +=1
  elif "sell" in context: # where context is a variable that stores the result of our earlier analysis
      sell_count +=1


print(f"Buy transactions in the last 24 hours: {buy_count}")
print(f"Sell transactions in the last 24 hours: {sell_count}")
```

This final snippet gives an example of how one can count the buys and sells of a specific token given that the context has been analysed previously based on the emitted events by a dex.

Important considerations to keep in mind:

*   **Dex Specifics:** The above examples are vastly simplified. Different dexes have different smart contracts, emit different events, and have different underlying mechanisms. Always refer to the specific dex’s smart contract code and documentation.
*   **Performance:** Looping through every single transaction receipt and log, especially for frequently traded tokens, can be incredibly slow. For production environments, this must be optimized significantly. Data warehousing or indexing solutions are often used for this task.
*   **Accuracy:** Due to the complexities of decentralized finance and smart contracts, 100% accuracy may be challenging to achieve. However, with sufficient care, we can obtain very reliable results.
*   **Transaction Types:** Transactions can encompass more than simple buy/sell. Liquidity additions, removals, or other contract interactions may produce transfer events but do not always indicate a buy or sell.

For resources, I recommend the official Ethereum documentation, specifically the sections on contracts and events. Also, the book “Mastering Ethereum” by Andreas Antonopoulos provides an in-depth explanation of blockchain technology. Exploring resources by Chainlink such as their documentation on data feeds and decentralized oracles will help understand how to handle off chain data in these situations. Finally, looking into the official documentation of the most important dexes like Uniswap or Pancakeswap (Uniswap Whitepaper, Pancakeswap API documentation) can provide valuable insights into their specific implementation.

In conclusion, determining the number of buy/sell transactions is an exercise in contextual interpretation of on-chain data. The steps and examples outlined above provide a solid framework for starting, but they are merely a starting point to this challenging task. Good luck and, as always, test thoroughly.
