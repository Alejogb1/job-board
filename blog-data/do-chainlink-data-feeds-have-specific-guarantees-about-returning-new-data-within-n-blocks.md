---
title: "Do chainlink data-feeds have specific guarantees about returning new data within N-Blocks?"
date: "2024-12-14"
id: "do-chainlink-data-feeds-have-specific-guarantees-about-returning-new-data-within-n-blocks"
---

alright, so you're asking about the latency guarantees of chainlink data feeds, specifically if there's a hard promise about new data showing up within a certain number of blocks. it’s a good question, and it's something i’ve spent a fair amount of time grappling with, particularly in the early days of trying to build reliable defi stuff. i can tell you first hand it's a critical aspect when dealing with time sensitive applications.

let's break it down, and i'll give you some context from my own battles with this problem.

first, there's no *absolute* guarantee that a chainlink data feed will update within a specific number of blocks. it's not like they have some kind of atomic clock that ticks in perfect sync with the blockchain's block production. the whole thing is probabilistic and relies on several moving parts. you need to understand that there's a distinction between a *target* update frequency, and the *actual* update frequency you observe.

the target update is dictated by a couple of factors: the heartbeat or time based trigger, which is like a scheduled task for updates, and the deviation threshold. this deviation trigger is what kicks off an update when the current value drifts too far from the last reported value. these parameters vary by feed and are configurable by the feed owners and often are detailed on the chainlink website for specific contracts if they are public data feeds.

the heartbeat is a minimum frequency, and an update can happen faster if the deviation threshold is crossed. it is not the time to update it is the minimum time period between any two updates regardless of the deviation parameter.

now, here's where the 'real world' gets messy, and where i've seen things get frustrating from personal projects that had serious implications. the chainlink oracle network is not a singular entity. it's a decentralized network of independent nodes. these nodes collect data from various sources off chain, consolidate the data on chain, and then submit it to the feed’s contract.

the time it takes for a change to show up depends on the speed of all of these separate steps and the time it takes to be included in a block in the end. not all the nodes will respond at the same rate; some might be faster than others and some could even be temporarily down for maintenance or network issues. it is a game of many moving parts and it is the very nature of a distributed system, the consistency of this system is probabilistic and not definite.

additionally, each chain has its own block time variability. for example, ethereum has a target of roughly 12 seconds but we can see variability of blocks being produced more or less on average. other chains might have different target times and variability. these block times make it hard to talk about guarantees in terms of block numbers, because an x number of blocks in one chain can be less time than x number of blocks in another.

so, instead of thinking about block guarantees, it's more helpful to think about *expected* update times based on the feed's parameters and the network's normal operating conditions. that's where monitoring becomes critical, and that's what i had to learn the hard way when my first defi dApp failed after a rapid movement in price.

to simulate and check these aspects i ended up doing something like this to check the last update of a specific data feed. i wrote a small script that fetches data at regular intervals and calculate the time difference between each update. i would also use that data to analyse the price deviation and compare it with the heartbeat time of the contract. i can see that on the chain link website too, but checking for myself made me more confident. here’s a simplified example with python:

```python
import time
import requests

def get_chainlink_data(contract_address):
    api_url = f"https://api.etherscan.io/api?module=account&action=tokentx&contractaddress={contract_address}&apikey=your_etherscan_api_key&sort=asc&page=1&offset=100"
    response = requests.get(api_url)
    data = response.json()
    if data["status"] == "1":
      transactions = data["result"]
      latest_tx = transactions[-1]
      timestamp = int(latest_tx['timeStamp'])
      return timestamp

    return None


def monitor_data_feed(contract_address, interval=60):
    last_update_time = None

    while True:
      current_update_time = get_chainlink_data(contract_address)

      if current_update_time:
        if last_update_time:
          time_diff = current_update_time - last_update_time
          print(f"Time since last update: {time_diff} seconds")
        last_update_time = current_update_time
      else:
        print("couldn't retrieve last update information")

      time.sleep(interval)


if __name__ == "__main__":
    # use an actual data feed contract address
    contract_address = "0x01BE23585060835E5635461190E1589679D345D9" # eth/usd contract example
    monitor_data_feed(contract_address)

```

*note*: remember to replace `your_etherscan_api_key` with a real one and this script can be adapted to other blockchains api providers like block explorers. it is just a simple script to exemplify the process to monitor.

this script is simple, but it shows the principle. it's a good practice to build your own to understand exactly how data updates occur, and then you can write more complex code to react to any discrepancies.
the key takeaway here is that you need to watch the data, you need to get familiarized with the contract interface and all the properties it expose. doing some code to monitor is beneficial in that way.

i had another situation where i was using chainlink data feeds for a time-sensitive lending protocol, and it was really important that the price updates were as close to real-time as possible. here the key is that i used multiple oracle sources to triangulate the price data and try to minimize data discrepancy across multiple providers.

it also helps to understand the contract. you can use web3 libraries to get all the relevant data from the contract directly. here’s an example with web3.py:

```python
from web3 import Web3

def get_contract_data(contract_address, rpc_url):
    web3 = Web3(Web3.HTTPProvider(rpc_url))
    if not web3.is_connected():
        print("not connected")
        return
    contract_abi =  [
    {
        "inputs": [],
        "name": "latestAnswer",
        "outputs": [
            {
                "internalType": "int256",
                "name": "",
                "type": "int256"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "latestTimestamp",
        "outputs": [
            {
                "internalType": "uint256",
                "name": "",
                "type": "uint256"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    }

]  # this is a simplified abi, consult documentation for a full abi
    contract = web3.eth.contract(address=contract_address, abi=contract_abi)

    latest_answer = contract.functions.latestAnswer().call()
    latest_timestamp = contract.functions.latestTimestamp().call()
    return latest_answer, latest_timestamp


if __name__ == "__main__":
    # use a real data feed contract address and rpc url for the chain you are interested in
    contract_address = "0x5f4eC3DF9cbd43714FE2740f5E3616155c5b8419" # eth/usd contract example
    rpc_url = "https://mainnet.infura.io/v3/<your_infura_key>" #replace with your provider or local node

    answer, time_stamp = get_contract_data(contract_address, rpc_url)
    if answer is not None:
        print(f"the latest answer is: {answer}")
        print(f"the latest timestamp is: {time_stamp}")

```

*note*: replace `<your_infura_key>` with your actual infura key or any other rpc provider. make sure to consult chainlink website for the contracts abi.

this code lets you get the latest price data reported and the timestamp which is very helpful if you are building your own custom monitoring solutions.

i also learned the importance of fallback mechanisms. when you are building time sensitive dApps you need to have a backup system in case there is a problem with the current oracle. for example, a simple fallback mechanism can be to use the last known valid value if there is no new update within a certain time window.

finally, let’s say you need to process the updates in a batch, well, you can do something like this. this function aggregates several updates and then processes them, this is a basic example using python for the principle, in reality you would use something like nodejs in production setting. this function will wait for several new prices and add them to a list.

```python
import time
import requests
from datetime import datetime

def get_chainlink_data(contract_address):
    api_url = f"https://api.etherscan.io/api?module=account&action=tokentx&contractaddress={contract_address}&apikey=your_etherscan_api_key&sort=asc&page=1&offset=100"
    response = requests.get(api_url)
    data = response.json()
    if data["status"] == "1":
      transactions = data["result"]
      latest_tx = transactions[-1]
      timestamp = int(latest_tx['timeStamp'])
      price_value = latest_tx['value']
      return timestamp, price_value

    return None, None


def aggregate_updates(contract_address, num_updates=5):
    updates = []
    last_update_time = None

    while len(updates) < num_updates:
        current_update_time, price_value = get_chainlink_data(contract_address)

        if current_update_time and last_update_time != current_update_time:
            updates.append({
                "time": datetime.fromtimestamp(current_update_time),
                "price": price_value
            })
            last_update_time = current_update_time
            print(f"new price added {price_value} at {datetime.fromtimestamp(current_update_time)}")
        elif current_update_time is not None and last_update_time == current_update_time:
          print("price not changed yet")
        else:
          print("couldn't retrieve information")

        time.sleep(10) #wait to avoid too many requests

    return updates


def process_updates(updates):
    for update in updates:
        print(f"Processing update at {update['time']}, price: {update['price']}")

if __name__ == "__main__":
    contract_address = "0x01BE23585060835E5635461190E1589679D345D9" # eth/usd contract example
    updates = aggregate_updates(contract_address)
    if updates:
        process_updates(updates)

```
*note*: remember to replace `your_etherscan_api_key` with your real etherscan api key.

it is important to point out that you should be very careful when doing any time sensitive logic based on blockchain data. the blockchain has its own rhythm and relying on "real" time data can be dangerous.

for further study, i can recommend looking at “the blockchain and the new architecture of trust” by kevin werbach for the general background and principles of blockchain technology and for more in depth specifics on how chainlink works you can consult its documentation or if you really are in need of a deep dive into consensus algorithms and how it works i would recommend “mastering bitcoin” by andreas antonopoulos as a solid start. that is the place i usually start when i want to go deeper in any of these topics.

in short, chainlink feeds offer good reliability for most applications but they are not as deterministic as some people wish. if your application has time sensitive logic, you need to monitor the feeds closely, understand the contract properties and have fallback mechanisms. there are no hard guarantees, but there are patterns you can use to achieve satisfactory results and build reliable applications. also be sure to check for more up to date information by consulting their documentation.

i hope that my past mistakes are useful for you, remember to be patient and take small steps when using these systems. you got this.
