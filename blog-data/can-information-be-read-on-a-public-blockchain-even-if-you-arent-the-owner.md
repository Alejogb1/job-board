---
title: "Can information be read on a public blockchain even if you aren't the owner?"
date: "2024-12-14"
id: "can-information-be-read-on-a-public-blockchain-even-if-you-arent-the-owner"
---

so, yeah, you're asking if you can see stuff on a blockchain even if it's not your stuff. the short answer is, absolutely, and it's kinda the whole point of a public blockchain. i've spent a good chunk of my career neck-deep in these things, and trust me, i've seen it all, from basic transactions to some pretty convoluted smart contract logic.

let’s break this down without getting too theoretical. think of a public blockchain like a shared, append-only database. everyone gets a copy, and any new data gets added in a way that's transparent for all to see. now, the key thing here is "public." it means that every transaction, every piece of data, is generally visible to anyone who wants to look. there's no magic password or "owner-only" access gate for reading.

i remember once, back in my early days, i was working on a project that involved a decentralized marketplace. we were trying to build some analytics tools, and initially, we struggled a bit with this. i was like, "how do i even get at all this data?". after spending a week on a wild goose chase thinking we would need some advanced secret keys i sat back and realized it was literally public information, just like the name implies. it was an embarrassing but important learning curve.

this public nature is fundamental to how blockchains achieve transparency and trust. nobody can sneak in changes without everyone noticing. that's pretty huge in situations where you need to avoid tampering, or double spending. for example, a simple cryptocurrency transaction. you see who sent what to whom, the exact time, and how much of the digital asset was transferred, all recorded publicly.

now, there are some nuances to how the data is structured. it's not like pulling out records from your local database. you're dealing with blocks, transactions, hashes, and all that good stuff. it's typically a structure that’s designed to be easy to verify and hard to change or break.

let’s get some code involved. for simple cases you can use different libraries depending on the specific chain. let's imagine we are using a popular library to look at transactions in the ethereum blockchain.

```python
from web3 import Web3

# replace with your actual endpoint, a testnet is fine
infura_url = "YOUR_INFURA_ENDPOINT"
w3 = Web3(Web3.HTTPProvider(infura_url))

# example transaction hash, feel free to get one from etherscan or other block explorer
transaction_hash = '0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef'  

transaction = w3.eth.get_transaction(transaction_hash)

if transaction:
    print(f"transaction details: {transaction}")
else:
  print("couldn't find transaction")

```

this snippet uses `web3.py`, a library i've used extensively and is very good for interacting with the ethereum network. it connects to an rpc endpoint, and requests a transaction. if the transaction is valid, it prints the complete structure of it. this shows you exactly what is included in a transaction, and this data is completely available without any need to have an “owner” status.

note, you'd need to get a node endpoint to use it, like from infura or alchemy, as those are the most popular node providers. running your own node is also an option, but it is a beast, and probably not the best starting point for this kind of work.

ok, another example. let's say you want to check the balance of an address.

```python
from web3 import Web3

# Replace with your actual endpoint, a testnet is fine
infura_url = "YOUR_INFURA_ENDPOINT"
w3 = Web3(Web3.HTTPProvider(infura_url))


# example wallet address
address = '0xabcdef1234567890abcdef1234567890abcdef12'  

balance = w3.eth.get_balance(address)

if balance:
    print(f"balance of {address}: {balance} wei")
else:
    print("couldn't get balance")

```

this code snippet uses the same library to pull the balance for an address, it's simple, but effective. again, anyone can do this. no permissions or special access needed.

this is the core idea: that the data is public and accessible. you just need to know how to query it using the correct tools for your specific blockchain.

now, it is essential to note that while the data is *readable*, it is not *editable*. unless you are the owner, or more specifically the private key holder for an address, you can’t change anything. i’ve seen cases where newbies try to modify balances and of course they fail, they just don't understand that the data is immutable.

now, lets get a little bit more involved, imagine we want to look at some smart contract events.

```python
from web3 import Web3
import json

# Replace with your actual endpoint, a testnet is fine
infura_url = "YOUR_INFURA_ENDPOINT"
w3 = Web3(Web3.HTTPProvider(infura_url))

# example smart contract address and abi
contract_address = '0xabcdef1234567890abcdef1234567890abcdef12'
contract_abi = [
    # example event
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": True,
                "internalType": "address",
                "name": "from",
                "type": "address"
            },
            {
                "indexed": True,
                "internalType": "address",
                "name": "to",
                "type": "address"
            },
            {
                "indexed": False,
                "internalType": "uint256",
                "name": "value",
                "type": "uint256"
            }
        ],
        "name": "Transfer",
        "type": "event"
    }
]


contract = w3.eth.contract(address=contract_address, abi=contract_abi)

# specify the block range you want to look at
start_block = 10000000
end_block = 10000100

event_filter = contract.events.Transfer.create_filter(fromBlock=start_block, toBlock=end_block)

events = event_filter.get_all_entries()

if events:
    for event in events:
        print(f"Event: {event}")
else:
  print("no events found")

```

this snippet is a bit more involved. it uses a contract's abi, which can be usually found on block explorers, to decode events. it searches for events inside a specific block range and prints them. this shows how you can interact with smart contracts data through events, which are usually public and can be used to track actions made by the contract.

now, i know what you might be thinking, "what about privacy?" yes, public blockchains don’t provide *transactional privacy* out of the box. every transaction is recorded in the blockchain publicly, you can't just disappear and create transactions without it being public knowledge. this is the main trade-off that exists in most public blockchains: transparency in exchange for lack of transactional privacy.

however, that is not the end of the story. there are several projects and techniques that try to tackle this, like zk-snarks and zk-starks, and a bunch of other sophisticated technologies being researched and developed, i highly recommend you research these if you are interested in the topic. reading research papers on cryptographic technologies is the best way to get to the bottom of it, i also recommend the book "mastering bitcoin" and "mastering ethereum", those two will give you a solid base.

so, can you read information on a public blockchain, even if you aren’t the owner? absolutely. it’s designed that way. it’s the fundamental concept that allows for transparency and immutability. and honestly, i've spent years navigating this space and it continues to fascinate me. i wouldn't have it any other way, even though i did spent one weekend tracking down the most complex smart contract, only to realize i was checking the wrong address. i was chasing my own tail on the blockchain, what a waste of gas! haha

hope that clears things up for you.
